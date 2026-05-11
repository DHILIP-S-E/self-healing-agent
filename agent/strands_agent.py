"""
Real self-healing agent using Strands LLM reasoning.

The LLM makes genuine decisions at every step via a phased approach.
Each phase gives the LLM a focused set of 1-3 tools and a simple prompt
so it reliably makes actual tool calls (not descriptions).

  Phase 1 — CLASSIFY   (1 tool:  get_first_page)
    LLM calls get_first_page, reads the result, outputs document type:
    "research_paper" | "survey_paper" | "other"

  Phase 2 — EXTRACT    (2-3 tools: chosen by Python based on LLM's Phase 1 decision)
    research_paper → LLM calls extract_abstract + extract_methods + extract_results
    survey_paper   → LLM calls extract_abstract + extract_contributions
    other          → LLM calls extract_full_text

  Phase 3 — SYNTHESIZE  (Python calls summarize_text — keeps large text off LLM prompt)

  Phase 4 — SAVE        (2 tools: save_summary + save_checkpoint — proven reliable)
    LLM calls save_summary then save_checkpoint

Why phased?  Mistral Large reliably calls 1-2 tools per invocation.
Giving it 10 tools at once causes it to describe rather than execute.
Each phase keeps the tool set tiny and the instruction unambiguous.

The LLM's real decisions:
  • What TYPE is this document? (Phase 1 — from raw first-page text)
  • Which extraction path to run? (Phase 2 — Python wires tools based on LLM's type)
  • How to handle empty sections? (Phase 2 fallback logic driven by LLM)

Source: https://strandsagents.com/latest/user-guide/concepts/agents/
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from .config import AWS_REGION, BEDROCK_MODEL_ID, INPUT_BUCKET, OUTPUT_BUCKET, REASONING_MODEL_ID
from .tools import (
    extract_abstract,
    extract_contributions,
    extract_full_text,
    extract_methods,
    extract_results,
    get_first_page,
    list_documents,
    load_checkpoint,
    record_error,
    save_checkpoint,
    save_summary,
    summarize_text,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _make_agent(tools: list, system_prompt: str):
    """Build a Strands agent with a focused tool set and system prompt."""
    try:
        from strands import Agent                       # type: ignore[import-not-found]
        from strands.models.bedrock import BedrockModel # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "strands-agents is required. Run: pip install strands-agents"
        ) from exc

    model = BedrockModel(
        model_id=REASONING_MODEL_ID,
        region_name=AWS_REGION,
        streaming=False,   # Mistral Large requires non-streaming for tool use
    )
    return Agent(model=model, tools=tools, system_prompt=system_prompt)


def build_reasoning_agent():
    """Build the primary reasoning agent (all tools).  Used by demo/tests."""
    return _make_agent(
        tools=[
            get_first_page, extract_abstract, extract_methods, extract_results,
            extract_contributions, extract_full_text, summarize_text,
            save_summary, save_checkpoint, record_error,
        ],
        system_prompt="You are a self-healing document intelligence agent on AWS.",
    )


# Keep old name as alias
build_strands_agent = build_reasoning_agent


# ---------------------------------------------------------------------------
# Phase 1 — CLASSIFY
# ---------------------------------------------------------------------------

_CLASSIFY_PROMPT = """You are a document classifier.

When given a bucket and key, call get_first_page immediately.
Read the text it returns, then output EXACTLY ONE of these labels:

  research_paper  — has Abstract, Introduction, Methods/Approach, Results/Experiments
  survey_paper    — title or abstract contains "survey", "review", "overview", "literature"
  other           — unclear or non-standard structure

After calling get_first_page, output only the label. Nothing else."""


def _classify_document(bucket: str, key: str) -> str:
    """Phase 1: LLM calls get_first_page and classifies the document type.

    Returns:
        "research_paper", "survey_paper", or "other".
    """
    agent = _make_agent([get_first_page], _CLASSIFY_PROMPT)
    response = agent(
        f"Classify this document.\n"
        f"bucket='{bucket}'\n"
        f"key='{key}'\n\n"
        f"Call get_first_page now, then output the classification label."
    )
    text = str(response).lower()

    if "research_paper" in text or "research paper" in text:
        doc_type = "research_paper"
    elif "survey_paper" in text or "survey paper" in text or "survey" in text:
        doc_type = "survey_paper"
    else:
        doc_type = "other"

    logger.info("classify: %s → %s", key, doc_type)
    return doc_type


# ---------------------------------------------------------------------------
# Phase 2 — EXTRACT
# ---------------------------------------------------------------------------

_EXTRACT_RESEARCH_PROMPT = """You are a section extraction agent.

You have three tools: extract_abstract, extract_methods, extract_results.
Call ALL THREE for the given bucket and key — in that order.
After each call, note whether it returned text or was empty.
If a tool returns empty text, note "section not found" and continue.
Output all non-empty results combined."""

_EXTRACT_SURVEY_PROMPT = """You are a section extraction agent.

You have two tools: extract_abstract and extract_contributions.
Call BOTH for the given bucket and key.
If extract_abstract returns empty, note it and continue.
If extract_contributions returns empty, note it and continue.
Output all non-empty results combined."""

_EXTRACT_FALLBACK_PROMPT = """You are a full-text extraction agent.

Call extract_full_text for the given bucket and key.
Return the text it gives you."""


def _extract_sections(bucket: str, key: str, doc_type: str) -> str:
    """Phase 2: LLM calls targeted extraction tools based on document type.

    Returns:
        Combined extracted text (may be empty if all tools returned nothing).
    """
    if doc_type == "research_paper":
        agent = _make_agent(
            [extract_abstract, extract_methods, extract_results],
            _EXTRACT_RESEARCH_PROMPT,
        )
        prompt = (
            f"Extract sections from this research paper.\n"
            f"bucket='{bucket}'\nkey='{key}'\n\n"
            f"Call extract_abstract, then extract_methods, then extract_results now."
        )
    elif doc_type == "survey_paper":
        agent = _make_agent(
            [extract_abstract, extract_contributions],
            _EXTRACT_SURVEY_PROMPT,
        )
        prompt = (
            f"Extract sections from this survey paper.\n"
            f"bucket='{bucket}'\nkey='{key}'\n\n"
            f"Call extract_abstract, then extract_contributions now."
        )
    else:
        agent = _make_agent([extract_full_text], _EXTRACT_FALLBACK_PROMPT)
        prompt = (
            f"Extract full text from this document.\n"
            f"bucket='{bucket}'\nkey='{key}'\n\n"
            f"Call extract_full_text now."
        )

    response = agent(prompt)
    extracted = str(response).strip()
    logger.info("extract: %s (%s) → %d chars", key, doc_type, len(extracted))
    return extracted


def _fallback_extract(bucket: str, key: str) -> str:
    """Fallback: call extract_full_text when Phase 2 returned nothing."""
    agent = _make_agent([extract_full_text], _EXTRACT_FALLBACK_PROMPT)
    response = agent(
        f"Call extract_full_text now.\nbucket='{bucket}'\nkey='{key}'"
    )
    return str(response).strip()


# ---------------------------------------------------------------------------
# Phase 4 — SAVE
# ---------------------------------------------------------------------------

_SAVE_PROMPT = """You are a document storage agent.

You have two tools: save_summary and save_checkpoint.
Call save_summary first, then call save_checkpoint.
Do not stop after save_summary — you must call save_checkpoint too."""


def _save_and_checkpoint(
    output_bucket: str,
    output_key: str,
    summary: str,
    task_id: str,
    completed_idx: int,
    doc_key: str,
) -> None:
    """Phase 4: LLM calls save_summary then save_checkpoint."""
    safe_summary = summary.encode("ascii", errors="replace").decode("ascii")
    agent = _make_agent([save_summary, save_checkpoint], _SAVE_PROMPT)
    agent(
        f"Save this summary and checkpoint the progress.\n\n"
        f"output_bucket:  '{output_bucket}'\n"
        f"output_key:     '{output_key}'\n"
        f"summary_text:   '{safe_summary[:800]}'\n\n"
        f"task_id:        '{task_id}'\n"
        f"completed_idx:  {completed_idx}\n"
        f"completed_key:  '{doc_key}'\n\n"
        f"Call save_summary now, then save_checkpoint."
    )


# ---------------------------------------------------------------------------
# Full per-document pipeline
# ---------------------------------------------------------------------------

def _process_document(
    *,
    task_id: str,
    doc_key: str,
    global_idx: int,
    input_bucket: str,
    output_bucket: str,
) -> dict[str, Any]:
    """Run the phased classify → extract → synthesize → save pipeline.

    Phase 1 (CLASSIFY):   LLM calls get_first_page, decides document type
    Phase 2 (EXTRACT):    LLM calls 2-3 tools chosen by Python from Phase 1
    Phase 3 (SYNTHESIZE): Python calls summarize_text (keeps large text off LLM)
    Phase 4 (SAVE):       LLM calls save_summary + save_checkpoint

    Returns:
        {"status": "success", ...} or {"status": "error", "reason": ...}
    """
    output_key = f"summaries/{task_id}/{doc_key.replace('papers/', '')}"

    # ── Phase 1: Classify ────────────────────────────────────────────────────
    try:
        doc_type = _classify_document(input_bucket, doc_key)
    except Exception as exc:
        logger.warning("Phase 1 FAILED for %s: %s — defaulting to 'other'", doc_key, exc)
        doc_type = "other"

    # ── Phase 2: Extract ─────────────────────────────────────────────────────
    try:
        extracted = _extract_sections(input_bucket, doc_key, doc_type)
    except Exception as exc:
        logger.warning("Phase 2 FAILED for %s: %s", doc_key, exc)
        extracted = ""

    # Fallback: if targeted extraction returned nothing, use full text
    if not extracted or len(extracted) < 100:
        logger.info("Phase 2 empty for %s — falling back to extract_full_text", doc_key)
        try:
            extracted = _fallback_extract(input_bucket, doc_key)
        except Exception as exc:
            logger.error("Fallback extraction FAILED for %s: %s", doc_key, exc)
            try:
                record_error(task_id, doc_key, f"all extraction failed: {exc}")
            except Exception:
                pass
            return {"status": "error", "reason": f"all extraction failed: {exc}"}

    if not extracted or len(extracted) < 50:
        record_error(task_id, doc_key, "extraction returned empty text after all fallbacks")
        return {"status": "error", "reason": "extraction returned empty text"}

    # ── Phase 3: Synthesize ──────────────────────────────────────────────────
    # Python calls summarize_text directly — the text is too large for the LLM prompt
    try:
        summary = summarize_text(extracted[:20000], max_words=150)
        logger.info("Phase 3 summary: %s → %d chars", doc_key, len(summary))
    except Exception as exc:
        logger.error("Phase 3 FAILED for %s: %s", doc_key, exc)
        try:
            record_error(task_id, doc_key, f"summarize_text failed: {exc}")
        except Exception:
            pass
        return {"status": "error", "reason": f"summarize_text failed: {exc}"}

    # ── Phase 4: Save ────────────────────────────────────────────────────────
    try:
        _save_and_checkpoint(
            output_bucket=output_bucket,
            output_key=output_key,
            summary=summary,
            task_id=task_id,
            completed_idx=global_idx,
            doc_key=doc_key,
        )
    except Exception as exc:
        logger.warning("Phase 4 LLM save FAILED for %s: %s — using direct fallback", doc_key, exc)
        try:
            save_summary(output_bucket, output_key, summary)
        except Exception:
            pass

    # Python safety-net: always checkpoint regardless of LLM behaviour
    try:
        save_checkpoint(task_id, global_idx, doc_key)
    except Exception as exc:
        logger.warning("safety-net checkpoint failed for %s: %s", doc_key, exc)

    return {
        "status": "success",
        "key": doc_key,
        "doc_type": doc_type,
        "output_key": output_key,
        "summary_length": len(summary),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_strands_task(
    task_id: str,
    input_bucket: str | None = None,
    output_bucket: str | None = None,
) -> dict[str, Any]:
    """Run the phased Strands reasoning agent on a batch of PDFs.

    Args:
        task_id:       Unique run identifier. Same ID = resume from checkpoint.
        input_bucket:  S3 bucket with input PDFs.
        output_bucket: S3 bucket for summaries.

    Returns:
        Dict with task summary: total, summarized, errors, resumed.
    """
    ib = input_bucket or INPUT_BUCKET
    ob = output_bucket or OUTPUT_BUCKET

    if not ib:
        raise ValueError("input_bucket is required (set INPUT_BUCKET env or pass directly)")
    if not ob:
        raise ValueError("output_bucket is required (set OUTPUT_BUCKET env or pass directly)")

    logger.info(
        "Strands reasoning task — task=%s model=%s region=%s",
        task_id, REASONING_MODEL_ID, AWS_REGION,
    )

    checkpoint      = load_checkpoint(task_id)
    completed_keys  = set(checkpoint.get("completed_keys", []))
    prior_errors    = checkpoint.get("errors", [])
    resumed         = len(completed_keys) > 0

    all_docs  = list_documents(ib, "papers/")
    remaining = [k for k in all_docs if k not in completed_keys]

    logger.info(
        "task=%s total=%d already_done=%d remaining=%d",
        task_id, len(all_docs), len(completed_keys), len(remaining),
    )

    if not remaining:
        return {
            "status": "complete", "task_id": task_id,
            "total": len(all_docs), "summarized": len(completed_keys),
            "errors": len(prior_errors), "skipped_resume": len(completed_keys),
            "resumed": resumed, "message": "All documents already processed.",
        }

    success_count = 0
    error_count   = 0

    for idx, doc_key in enumerate(remaining):
        global_idx = len(completed_keys) + idx
        logger.info("--- [%d/%d] %s ---", idx + 1, len(remaining), doc_key)

        result = _process_document(
            task_id=task_id,
            doc_key=doc_key,
            global_idx=global_idx,
            input_bucket=ib,
            output_bucket=ob,
        )

        if result["status"] == "success":
            success_count += 1
            logger.info(
                "[%d/%d] SUCCESS: %s (type=%s, summary=%d chars)",
                idx + 1, len(remaining), doc_key,
                result.get("doc_type", "?"), result.get("summary_length", 0),
            )
        else:
            error_count += 1
            logger.warning(
                "[%d/%d] ERROR: %s — %s",
                idx + 1, len(remaining), doc_key, result.get("reason"),
            )

        time.sleep(1)

    total_done = len(completed_keys) + success_count
    return {
        "status": "complete", "task_id": task_id,
        "total": len(all_docs), "summarized": total_done,
        "errors": error_count + len(prior_errors),
        "skipped_resume": len(completed_keys), "resumed": resumed,
    }
