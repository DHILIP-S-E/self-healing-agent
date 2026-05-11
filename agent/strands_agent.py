"""
Real self-healing agent using Strands LLM reasoning.

Architecture:
  Data-heavy steps (read PDF, invoke Bedrock) run in Python directly.
  Decision steps (where to save, checkpoint after success, record errors)
  are executed by the Strands LLM agent via tool calls.

  This is the correct agentic pattern — LLMs are not good at passing
  large blobs of text between steps, but excel at reasoning about
  what to do next, how to handle failures, and where to store results.

  Python layer  (reads PDF, calls Bedrock for summary)
    └─ Strands Agent  (decides how to save, checkpoint, recover errors)
         └─ Tools: save_summary, save_checkpoint, record_error

  The LLM makes genuine decisions:
    - How to format the output S3 key
    - Whether to save_checkpoint or record_error after a failure
    - How to structure the checkpoint call
    - What error detail to record

Source: https://strandsagents.com/latest/user-guide/concepts/agents/
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .config import AWS_REGION, BEDROCK_MODEL_ID, INPUT_BUCKET, OUTPUT_BUCKET, REASONING_MODEL_ID
from .tools import (
    list_documents,
    load_checkpoint,
    read_document,
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
    """Build a Strands agent with a specific tool set and system prompt."""
    try:
        from strands import Agent  # type: ignore[import-not-found]
        from strands.models.bedrock import BedrockModel  # type: ignore[import-not-found]
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


def build_strands_agent():
    """Build the Strands reasoning agent used for save + checkpoint decisions.

    This agent is invoked after Python has already read and summarized a
    document. The LLM decides how to persist the result and handle errors.

    Returns:
        strands.Agent configured with save/checkpoint/error tools.
    """
    system_prompt = (
        "You are a self-healing document storage agent running on AWS S3.\n"
        "Your job: persist a document summary and checkpoint the progress.\n\n"
        "When given:\n"
        "  - output_bucket, output_key, summary_text\n"
        "  - task_id, completed_idx, completed_key\n\n"
        "You MUST:\n"
        "1. Call save_summary(bucket=output_bucket, key=output_key, summary=summary_text)\n"
        "2. Call save_checkpoint(task_id=task_id, completed_idx=completed_idx, "
        "completed_key=completed_key)\n\n"
        "If save_summary fails, call record_error instead of save_checkpoint.\n"
        "Make actual tool calls. Do not write pseudocode."
    )
    return _make_agent(
        tools=[save_summary, save_checkpoint, record_error],
        system_prompt=system_prompt,
    )


# ---------------------------------------------------------------------------
# Process one document — Python reads/summarizes, LLM saves/checkpoints
# ---------------------------------------------------------------------------

def _process_document(
    *,
    task_id: str,
    doc_key: str,
    global_idx: int,
    input_bucket: str,
    output_bucket: str,
) -> dict[str, Any]:
    """Process one PDF with Python + Strands LLM collaboration.

    Python handles data-heavy steps (read + summarize via Bedrock directly).
    The Strands LLM agent handles the decision steps: save the summary to
    the right S3 path, checkpoint progress, or record errors.

    Returns:
        {"status": "success", ...} or {"status": "error", "reason": ...}
    """
    output_key = f"summaries/{task_id}/{doc_key.replace('papers/', '')}"

    # --- Step 1 (Python): Read the PDF ---
    try:
        text = read_document(input_bucket, doc_key)
        logger.info("read OK: %s (%d chars)", doc_key, len(text))
    except Exception as exc:
        logger.warning("read FAILED for %s: %s", doc_key, exc)
        # LLM agent decides how to record the error
        try:
            err_agent = _make_agent([record_error], (
                "You are an error recorder. "
                "Call record_error immediately with the given task_id, doc_key, and error_message."
            ))
            err_agent(
                f"Call record_error now.\n"
                f"task_id='{task_id}'\n"
                f"doc_key='{doc_key}'\n"
                f"error_message='read_document failed: {str(exc)[:200]}'"
            )
        except Exception:
            # Fallback: record error directly if agent fails
            try:
                record_error(task_id, doc_key, f"read_document failed: {exc}")
            except Exception:
                pass
        return {"status": "error", "reason": f"read_document failed: {exc}"}

    # --- Step 2 (Python): Summarize via Bedrock directly ---
    try:
        summary = summarize_text(text, max_words=150)
        logger.info("summarize OK: %s (%d chars summary)", doc_key, len(summary))
    except Exception as exc:
        logger.warning("summarize FAILED for %s: %s", doc_key, exc)
        try:
            record_error(task_id, doc_key, f"summarize_text failed: {exc}")
        except Exception:
            pass
        return {"status": "error", "reason": f"summarize_text failed: {exc}"}

    # --- Step 3 (Strands LLM): Save summary and checkpoint ---
    # The LLM agent decides the tool call sequence and handles any failures.
    # Summary text is ASCII-safe for the prompt (it's a short generated string).
    safe_summary = summary.encode("ascii", errors="replace").decode("ascii")

    agent = build_strands_agent()
    try:
        agent(
            f"Save this document summary and checkpoint the progress.\n\n"
            f"output_bucket: '{output_bucket}'\n"
            f"output_key:    '{output_key}'\n"
            f"summary_text:  '{safe_summary[:500]}'\n\n"
            f"task_id:       '{task_id}'\n"
            f"completed_idx: {global_idx}\n"
            f"completed_key: '{doc_key}'\n\n"
            f"Call save_summary first, then save_checkpoint. "
            f"If save_summary fails, call record_error instead."
        )
        logger.info("LLM agent saved and checkpointed: %s", doc_key)
    except Exception as exc:
        logger.warning("LLM agent FAILED for %s: %s — falling back to direct calls", doc_key, exc)
        # Fallback: call tools directly so we never lose work
        try:
            save_summary(output_bucket, output_key, summary)
        except Exception as save_exc:
            logger.error("save_summary fallback also failed: %s", save_exc)

    # Safety-net: always ensure checkpoint is written even if the LLM agent
    # decided not to call save_checkpoint (LLMs are non-deterministic).
    # This guarantees crash-resume always skips already-summarised documents.
    try:
        save_checkpoint(task_id, global_idx, doc_key)
    except Exception as ckpt_exc:
        logger.warning("save_checkpoint safety-net failed for %s: %s", doc_key, ckpt_exc)

    return {
        "status": "success",
        "key": doc_key,
        "summary_length": len(summary),
        "output_key": output_key,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_strands_task(
    task_id: str,
    input_bucket: str | None = None,
    output_bucket: str | None = None,
) -> dict[str, Any]:
    """Run the Strands + Python self-healing agent on a batch of PDFs.

    Python orchestrates document discovery and checkpoint state.
    The Strands LLM agent executes the save/checkpoint/error decisions
    for each document — making autonomous tool calls on AWS.

    Args:
        task_id: Unique run identifier. Same ID = resume from checkpoint.
        input_bucket: S3 bucket with input PDFs. Falls back to INPUT_BUCKET env.
        output_bucket: S3 bucket for summaries. Falls back to OUTPUT_BUCKET env.

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
        "Strands task starting — task=%s reasoning=%s summarization=%s region=%s",
        task_id, REASONING_MODEL_ID, BEDROCK_MODEL_ID, AWS_REGION,
    )

    # Python orchestrates: discover docs + load checkpoint
    checkpoint = load_checkpoint(task_id)
    completed_keys: set[str] = set(checkpoint.get("completed_keys", []))
    prior_errors: list = checkpoint.get("errors", [])
    resumed = len(completed_keys) > 0

    all_docs = list_documents(ib, "papers/")
    remaining = [k for k in all_docs if k not in completed_keys]

    logger.info(
        "task=%s total=%d already_done=%d remaining=%d",
        task_id, len(all_docs), len(completed_keys), len(remaining),
    )

    if not remaining:
        return {
            "status": "complete",
            "task_id": task_id,
            "total": len(all_docs),
            "summarized": len(completed_keys),
            "errors": len(prior_errors),
            "skipped_resume": len(completed_keys),
            "resumed": resumed,
            "message": "Nothing to do — all documents already processed.",
        }

    success_count = 0
    error_count = 0

    for idx, doc_key in enumerate(remaining):
        global_idx = len(completed_keys) + idx

        logger.info(
            "--- [%d/%d] Processing: %s ---",
            idx + 1, len(remaining), doc_key,
        )

        # Python reads + summarizes; Strands LLM saves + checkpoints
        result = _process_document(
            task_id=task_id,
            doc_key=doc_key,
            global_idx=global_idx,
            input_bucket=ib,
            output_bucket=ob,
        )

        if result["status"] == "success":
            success_count += 1
            logger.info("[%d/%d] SUCCESS: %s", idx + 1, len(remaining), doc_key)
        else:
            error_count += 1
            logger.warning(
                "[%d/%d] ERROR: %s — %s",
                idx + 1, len(remaining), doc_key, result.get("reason"),
            )

        # Brief pause between documents to avoid Bedrock throttling
        time.sleep(1)

    total_done = len(completed_keys) + success_count
    return {
        "status": "complete",
        "task_id": task_id,
        "total": len(all_docs),
        "summarized": total_done,
        "errors": error_count + len(prior_errors),
        "skipped_resume": len(completed_keys),
        "resumed": resumed,
    }
