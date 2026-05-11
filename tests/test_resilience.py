"""
Crash + resume integration test — the proof that the agent is self-healing.

Strategy: replace the 4 tools with deterministic in-memory fakes so the test
runs offline (no S3, no Bedrock). Inject a crash at the Nth summarize call,
verify the checkpoint survived, restart, verify the run completes with no
duplicate work.

This test alone validates the core promise of the project.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.main import ToolBundle, run_task
from agent.state_manager import FileBackend, StateManager


# =============================================================================
# Fakes
# =============================================================================

def make_fake_tools(doc_count: int = 100, crash_at: int | None = None):
    """Return a (ToolBundle, save_log) pair.

    If `crash_at` is set, summarize_text raises RuntimeError on its `crash_at`-th
    invocation (1-indexed). save_log captures every save_summary call so the
    test can assert no duplicates were written.
    """
    save_log: list[dict] = []
    state = {"summarize_calls": 0}

    def list_documents(bucket: str, prefix: str = "") -> list[str]:
        return [f"doc-{i:03d}.pdf" for i in range(doc_count)]

    def read_document(bucket: str, key: str) -> str:
        return f"Body of {key}"

    def summarize_text(text: str, max_words: int = 100) -> str:
        state["summarize_calls"] += 1
        if crash_at is not None and state["summarize_calls"] == crash_at:
            raise RuntimeError(f"simulated crash at summarize call {crash_at}")
        return f"Summary of: {text[:50]}"

    def save_summary(bucket: str, key: str, summary: str) -> dict:
        save_log.append({"bucket": bucket, "key": key, "summary": summary})
        return {
            "bucket": bucket,
            "key": key,
            "size_bytes": len(summary),
            "content_type": "text/markdown",
        }

    bundle = ToolBundle(
        list_documents=list_documents,
        read_document=read_document,
        summarize_text=summarize_text,
        save_summary=save_summary,
    )
    return bundle, save_log


@pytest.fixture
def state_manager(tmp_path: Path) -> StateManager:
    return StateManager(backends=[FileBackend(state_dir=tmp_path / "state")])


# =============================================================================
# The headline test
# =============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_crash_and_resume_processes_all_100_docs_exactly_once(state_manager):
    """The marquee proof: crash mid-run, resume, end with all 100 done, no dups."""
    payload = {
        "task_id": "task-resilience",
        "input_bucket": "fake-in",
        "output_bucket": "fake-out",
        "resume": True,
    }

    # ---- Run 1: crash at summarize call #30 -------------------------------
    tools_run1, saved_run1 = make_fake_tools(doc_count=100, crash_at=30)
    with pytest.raises(RuntimeError, match="simulated crash"):
        await run_task(payload, tools=tools_run1, state_manager=state_manager)

    # Doc i=29 (the 30th iter) crashed during summarize, BEFORE save_summary
    # and save_progress were called. So:
    #   - save_log has indices 0..28 (29 entries)
    #   - state.completed_idx = 28
    assert len(saved_run1) == 29, f"expected 29 saves before crash, got {len(saved_run1)}"
    prev = state_manager.load_progress("task-resilience")
    assert prev is not None
    assert prev["completed_idx"] == 28
    assert len(prev["results"]) == 29

    # ---- Run 2: resume to completion --------------------------------------
    tools_run2, saved_run2 = make_fake_tools(doc_count=100, crash_at=None)
    result = await run_task(payload, tools=tools_run2, state_manager=state_manager)

    # Headline result
    assert result["status"] == "complete"
    assert result["total_documents"] == 100
    assert result["skipped_resume"] == 29
    assert result["processed_in_this_run"] == 71  # docs 29..99

    # No duplicates across both runs
    all_keys = [s["key"] for s in saved_run1] + [s["key"] for s in saved_run2]
    assert len(all_keys) == 100
    assert len(set(all_keys)) == 100, "duplicate save detected"

    # Task complete -> load_progress returns None
    assert state_manager.load_progress("task-resilience") is None


# =============================================================================
# Auxiliary cases
# =============================================================================

@pytest.mark.asyncio
async def test_resume_false_starts_from_zero(state_manager):
    """resume=False wipes prior state and processes everything from scratch."""
    payload = {
        "task_id": "task-no-resume",
        "input_bucket": "fake-in",
        "output_bucket": "fake-out",
        "resume": False,
    }
    # Pre-populate state with completed_idx=50 — should be ignored.
    state_manager.save_progress(
        "task-no-resume", completed_idx=50, results=[{"x": i} for i in range(51)]
    )

    tools, saved = make_fake_tools(doc_count=100, crash_at=None)
    result = await run_task(payload, tools=tools, state_manager=state_manager)

    assert result["skipped_resume"] == 0
    assert result["processed_in_this_run"] == 100
    assert len(saved) == 100


@pytest.mark.asyncio
async def test_already_completed_task_starts_fresh(state_manager):
    """A previously-completed task (state cleared) starts a new run from 0."""
    payload = {
        "task_id": "task-was-done",
        "input_bucket": "fake-in",
        "output_bucket": "fake-out",
        "resume": True,
    }
    state_manager.save_progress("task-was-done", completed_idx=99, results=[])
    state_manager.mark_complete("task-was-done")

    assert state_manager.load_progress("task-was-done") is None

    tools, saved = make_fake_tools(doc_count=100, crash_at=None)
    result = await run_task(payload, tools=tools, state_manager=state_manager)

    assert result["skipped_resume"] == 0
    assert result["processed_in_this_run"] == 100
    assert len(saved) == 100


@pytest.mark.asyncio
async def test_no_input_returns_no_input_status(state_manager):
    payload = {
        "task_id": "task-empty",
        "input_bucket": "fake-in",
        "output_bucket": "fake-out",
        "resume": True,
    }
    tools, _ = make_fake_tools(doc_count=0, crash_at=None)
    result = await run_task(payload, tools=tools, state_manager=state_manager)

    assert result["status"] == "no_input"
    assert result["processed_in_this_run"] == 0


@pytest.mark.asyncio
async def test_multiple_crashes_still_converge(state_manager):
    """Two crashes in a row, then a clean run, all converge to 100 unique saves."""
    payload = {
        "task_id": "task-double-crash",
        "input_bucket": "fake-in",
        "output_bucket": "fake-out",
        "resume": True,
    }

    # Crash 1 at iter 20
    t1, s1 = make_fake_tools(doc_count=100, crash_at=20)
    with pytest.raises(RuntimeError):
        await run_task(payload, tools=t1, state_manager=state_manager)

    # Crash 2 at iter 30 (counting only this run's calls — it sees 71 remaining
    # docs, and we crash at the 11th of them = global iter 29 of remaining run.
    # We crash at summarize call 11 of run 2, which corresponds to global doc index 29.)
    t2, s2 = make_fake_tools(doc_count=100, crash_at=11)
    with pytest.raises(RuntimeError):
        await run_task(payload, tools=t2, state_manager=state_manager)

    # Clean run
    t3, s3 = make_fake_tools(doc_count=100, crash_at=None)
    result = await run_task(payload, tools=t3, state_manager=state_manager)

    assert result["status"] == "complete"
    all_keys = [s["key"] for s in s1] + [s["key"] for s in s2] + [s["key"] for s in s3]
    assert len(all_keys) == 100
    assert len(set(all_keys)) == 100
