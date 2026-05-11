"""
Entrypoint for the self-healing document processing agent.

Hosted on AgentCore Runtime. Exposes a single async entrypoint, `invoke`,
that processes a list of S3 PDFs and saves a checkpoint after every doc so
the run can resume from the last completed index after any kind of crash.

Public API:
    ToolBundle           Container of the 4 tool callables (overridable in tests).
    run_task(payload)    The actual loop, separable from the AgentCore wrapper.
    invoke(payload)      AgentCore Runtime entrypoint (calls run_task).
    app                  BedrockAgentCoreApp instance, with @app.entrypoint decorator.

The @app.entrypoint glue is created only if `bedrock_agentcore` is importable —
this lets unit tests import `run_task` and `ToolBundle` without the AgentCore
SDK installed.

Source: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-getting-started.html
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable

# Optional AgentCore wrapper. If not installed, run_task still works and tests pass.
try:
    from bedrock_agentcore import BedrockAgentCoreApp  # type: ignore[import-not-found]
    _AGENTCORE_AVAILABLE = True
except ImportError:
    _AGENTCORE_AVAILABLE = False
    BedrockAgentCoreApp = None  # type: ignore[assignment,misc]

from . import tools as default_tools
from .config import (
    AWS_REGION,
    BEDROCK_MODEL_ID,
    INPUT_BUCKET,
    LOG_LEVEL,
    OUTPUT_BUCKET,
)
from .state_manager import StateManager, build_default_manager

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# ToolBundle — overridable tool callables
# =============================================================================

class ToolBundle:
    """Holds the 4 tool callables. Tests can pass custom functions here."""

    def __init__(
        self,
        list_documents: Callable[..., list[str]] = default_tools.list_documents,
        read_document:  Callable[..., str]       = default_tools.read_document,
        summarize_text: Callable[..., str]       = default_tools.summarize_text,
        save_summary:   Callable[..., dict]      = default_tools.save_summary,
    ):
        self.list_documents = list_documents
        self.read_document = read_document
        self.summarize_text = summarize_text
        self.save_summary = save_summary


# =============================================================================
# run_task — the heart of the agent loop
# =============================================================================

async def run_task(
    payload: dict[str, Any],
    *,
    tools: ToolBundle | None = None,
    state_manager: StateManager | None = None,
) -> dict[str, Any]:
    """Process a list of S3 PDFs, checkpointing after every document.

    Payload shape:
        {
            "task_id":       "task-001",          # required
            "input_bucket":  "...",               # falls back to INPUT_BUCKET env
            "input_prefix":  "papers/",           # default ""
            "output_bucket": "...",               # falls back to OUTPUT_BUCKET env
            "resume":        true                 # default true
        }

    Behavior:
        - On startup, loads any prior checkpoint for `task_id` (if resume).
        - Iterates documents starting at `completed_idx + 1`.
        - After each successful doc, calls state_manager.save_progress.
        - On any exception, the checkpoint is already persisted; we re-raise.
        - On clean completion, calls state_manager.mark_complete.

    Returns:
        Status dict with `total_documents`, `processed_in_this_run`,
        `skipped_resume`, `started_at`, `finished_at`.
    """
    tools = tools or ToolBundle()
    sm = state_manager or build_default_manager()

    task_id = payload.get("task_id") or "task-default"
    input_bucket = payload.get("input_bucket") or INPUT_BUCKET
    input_prefix = payload.get("input_prefix", "")
    output_bucket = payload.get("output_bucket") or OUTPUT_BUCKET
    resume = bool(payload.get("resume", True))

    if not input_bucket:
        raise ValueError("input_bucket is required (set INPUT_BUCKET env or pass in payload)")
    if not output_bucket:
        raise ValueError("output_bucket is required (set OUTPUT_BUCKET env or pass in payload)")

    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # ---- 1. Resume from prior checkpoint? --------------------------------
    completed_idx = -1
    results: list[dict] = []
    if resume:
        prev = sm.load_progress(task_id)
        if prev is not None:
            completed_idx = int(prev["completed_idx"])
            results = list(prev.get("results", []))
            logger.info(
                "Resuming task=%s: completed_idx=%d, results=%d",
                task_id, completed_idx, len(results),
            )
        else:
            logger.info("No prior state for task=%s; starting fresh", task_id)
    else:
        # Explicit fresh start — wipe any stale checkpoint.
        sm.clear_state(task_id)

    # ---- 2. Discover work ------------------------------------------------
    keys = tools.list_documents(bucket=input_bucket, prefix=input_prefix)
    total = len(keys)
    if total == 0:
        return {
            "status": "no_input",
            "task_id": task_id,
            "input_bucket": input_bucket,
            "input_prefix": input_prefix,
            "processed_in_this_run": 0,
        }

    skipped_resume = completed_idx + 1
    logger.info(
        "Task=%s: %d total docs, resuming at index %d (%d already done)",
        task_id, total, skipped_resume, skipped_resume,
    )

    # ---- 3. Loop. Checkpoint after every doc. Skip bad docs, never crash. ---
    current_idx = skipped_resume
    skipped_docs: list[dict] = []
    try:
        for current_idx in range(skipped_resume, total):
            key = keys[current_idx]
            logger.info("[%d/%d] %s", current_idx + 1, total, key)

            try:
                text = tools.read_document(bucket=input_bucket, key=key)
                summary = tools.summarize_text(text=text)
                save_result = tools.save_summary(
                    bucket=output_bucket,
                    key=f"summaries/{task_id}/{key}",
                    summary=summary,
                )
                results.append({
                    "doc_idx": current_idx,
                    "doc_key": key,
                    "summary_key": save_result.get("key"),
                    "summary_preview": (summary or "")[:200],
                    "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                })
            except Exception as doc_err:
                # Skip this doc — log and continue. Self-healing behaviour.
                logger.warning(
                    "Skipping [%d/%d] %s: %s", current_idx + 1, total, key, doc_err
                )
                skipped_docs.append({"doc_idx": current_idx, "doc_key": key, "error": str(doc_err)})

            # CHECKPOINT after every doc (success or skip) — survives any crash.
            sm.save_progress(task_id, completed_idx=current_idx, results=results)
            # Brief pause to stay within Bedrock rate limits.
            time.sleep(1)

    except KeyboardInterrupt:
        logger.warning(
            "KeyboardInterrupt at index %d — checkpoint persisted, re-raising",
            current_idx,
        )
        raise

    # ---- 4. Done ---------------------------------------------------------
    sm.mark_complete(task_id)
    finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    return {
        "status": "complete",
        "task_id": task_id,
        "total_documents": total,
        "processed_in_this_run": len(results) - skipped_resume,
        "skipped_resume": skipped_resume,
        "skipped_errors": len(skipped_docs),
        "skipped_error_keys": [d["doc_key"] for d in skipped_docs],
        "started_at": started_at,
        "finished_at": finished_at,
    }


# =============================================================================
# AgentCore entrypoint
# =============================================================================

if _AGENTCORE_AVAILABLE:
    app = BedrockAgentCoreApp()  # type: ignore[misc]

    @app.entrypoint  # type: ignore[union-attr]
    async def invoke(payload: dict[str, Any]) -> dict[str, Any]:
        """AgentCore Runtime entrypoint."""
        return await run_task(payload)
else:
    app = None  # type: ignore[assignment]

    async def invoke(payload: dict[str, Any]) -> dict[str, Any]:  # type: ignore[no-redef]
        """Fallback invoke when bedrock_agentcore is not installed."""
        return await run_task(payload)


if __name__ == "__main__":
    if app is None:
        sys.exit(
            "bedrock-agentcore is not installed. "
            "Run `pip install bedrock-agentcore` to enable the AgentCore Runtime entrypoint."
        )
    # Source: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-getting-started.html
    app.run()
