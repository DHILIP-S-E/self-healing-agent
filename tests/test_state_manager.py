"""
Unit tests for agent.state_manager.

Covers the 5 mandatory cases from the spec plus a few useful extras:
    1. Save then load -> equal
    2. Load when no state -> None
    3. Save twice -> second wins
    4. Mark complete -> load returns None
    5. Schema version mismatch -> handled gracefully
    +6. clear_state behaves the same as mark_complete
    +7. Atomic write: a stale .tmp file does not poison the read
    +8. S3 backend save+load roundtrip (using moto)
    +9. Fallback chain: primary write fails -> secondary takes over
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import boto3
import pytest

from agent.state_manager import (
    SCHEMA_VERSION,
    STATUS_COMPLETE,
    STATUS_IN_PROGRESS,
    FileBackend,
    S3Backend,
    StateManager,
    build_default_manager,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    """Per-test scratch directory for the file backend."""
    d = tmp_path / "state"
    d.mkdir()
    return d


@pytest.fixture
def sm(state_dir: Path) -> StateManager:
    """A StateManager with a single FileBackend pointed at a tmp dir."""
    return StateManager(backends=[FileBackend(state_dir=state_dir)])


# =============================================================================
# 1. Save then load -> equal
# =============================================================================

def test_save_then_load_returns_same_data(sm: StateManager) -> None:
    results = [{"doc_key": "a.pdf", "summary": "x"}, {"doc_key": "b.pdf", "summary": "y"}]
    sm.save_progress("task-001", completed_idx=2, results=results)

    loaded = sm.load_progress("task-001")
    assert loaded is not None
    assert loaded["task_id"] == "task-001"
    assert loaded["completed_idx"] == 2
    assert loaded["results"] == results
    assert loaded["status"] == STATUS_IN_PROGRESS
    assert loaded["schema_version"] == SCHEMA_VERSION


# =============================================================================
# 2. Load when no state -> None
# =============================================================================

def test_load_when_no_state_returns_none(sm: StateManager) -> None:
    assert sm.load_progress("never-existed") is None


# =============================================================================
# 3. Save twice -> second wins
# =============================================================================

def test_save_twice_second_overwrites_first(sm: StateManager) -> None:
    sm.save_progress("task-002", completed_idx=5, results=[{"v": 1}])
    sm.save_progress("task-002", completed_idx=10, results=[{"v": 2}])

    loaded = sm.load_progress("task-002")
    assert loaded is not None
    assert loaded["completed_idx"] == 10
    assert loaded["results"] == [{"v": 2}]


# =============================================================================
# 4. Mark complete -> load returns None
# =============================================================================

def test_mark_complete_makes_load_return_none(sm: StateManager) -> None:
    sm.save_progress("task-003", completed_idx=99, results=[])
    assert sm.load_progress("task-003") is not None  # sanity

    sm.mark_complete("task-003")
    assert sm.load_progress("task-003") is None


# =============================================================================
# 5. Schema version mismatch -> handled gracefully (returns None, no crash)
# =============================================================================

def test_schema_version_mismatch_returns_none(state_dir: Path) -> None:
    # Write a checkpoint by hand with a future schema version.
    bad_payload = {
        "schema_version": 999,
        "task_id": "task-future",
        "completed_idx": 7,
        "results": [],
        "status": STATUS_IN_PROGRESS,
    }
    (state_dir / "task-future.checkpoint.json").write_text(json.dumps(bad_payload))

    sm = StateManager(backends=[FileBackend(state_dir=state_dir)])
    # Must not raise; must return None.
    assert sm.load_progress("task-future") is None


# =============================================================================
# 6. clear_state has same semantics as mark_complete
# =============================================================================

def test_clear_state_removes_checkpoint(sm: StateManager) -> None:
    sm.save_progress("task-clear", completed_idx=1, results=[])
    sm.clear_state("task-clear")
    assert sm.load_progress("task-clear") is None


# =============================================================================
# 7. Atomic write: a leftover .tmp file should not be visible as state
# =============================================================================

def test_stale_tmp_file_is_not_visible_as_checkpoint(state_dir: Path) -> None:
    # Simulate a crash mid-write: the .tmp exists but the final file does not.
    (state_dir / "task-crash.checkpoint.tmp").write_text('{"schema_version":1,"completed_idx":42}')

    sm = StateManager(backends=[FileBackend(state_dir=state_dir)])
    assert sm.load_progress("task-crash") is None


# =============================================================================
# 8. S3 backend roundtrip (mocked with moto)
# =============================================================================

@pytest.fixture
def s3_bucket():
    """Spin up an in-process S3 with moto for the duration of one test."""
    moto = pytest.importorskip("moto")
    from moto import mock_aws  # type: ignore[import-not-found]

    with mock_aws():
        client = boto3.client("s3", region_name="ap-south-1")
        bucket = "test-state-bucket"
        # ap-south-1 is not "us-east-1", so a LocationConstraint is required.
        client.create_bucket(
            Bucket=bucket,
            CreateBucketConfiguration={"LocationConstraint": "ap-south-1"},
        )
        yield bucket, client


def test_s3_backend_save_load_roundtrip(s3_bucket) -> None:
    bucket, _ = s3_bucket
    sm = StateManager(backends=[S3Backend(bucket=bucket, region="ap-south-1")])

    sm.save_progress("task-s3", completed_idx=4, results=[{"doc": "p1"}])
    loaded = sm.load_progress("task-s3")

    assert loaded is not None
    assert loaded["completed_idx"] == 4
    assert loaded["results"] == [{"doc": "p1"}]


def test_s3_backend_returns_none_when_object_missing(s3_bucket) -> None:
    bucket, _ = s3_bucket
    sm = StateManager(backends=[S3Backend(bucket=bucket, region="ap-south-1")])
    assert sm.load_progress("never-saved") is None


# =============================================================================
# 9. Fallback chain: primary write fails -> secondary takes over
# =============================================================================

def test_write_falls_back_when_primary_raises(state_dir: Path) -> None:
    # A backend that always blows up on writes.
    flaky = MagicMock()
    flaky.name = "flaky"
    flaky.write.side_effect = RuntimeError("simulated outage")
    flaky.read.return_value = None
    flaky.delete.return_value = None

    file_backend = FileBackend(state_dir=state_dir)
    sm = StateManager(backends=[flaky, file_backend])

    # Should not raise — fallback (FileBackend) succeeds.
    sm.save_progress("task-fallback", completed_idx=2, results=[{"x": 1}])

    flaky.write.assert_called_once()
    loaded = sm.load_progress("task-fallback")
    assert loaded is not None
    assert loaded["completed_idx"] == 2


def test_write_raises_when_all_backends_fail(state_dir: Path) -> None:
    a = MagicMock(); a.name = "a"; a.write.side_effect = RuntimeError("boom-a")
    b = MagicMock(); b.name = "b"; b.write.side_effect = RuntimeError("boom-b")

    sm = StateManager(backends=[a, b])
    with pytest.raises(RuntimeError, match="Failed to write checkpoint"):
        sm.save_progress("task-doomed", completed_idx=1, results=[])


# =============================================================================
# Construction guards
# =============================================================================

def test_state_manager_requires_at_least_one_backend() -> None:
    with pytest.raises(ValueError, match="at least one backend"):
        StateManager(backends=[])


def test_build_default_manager_uses_file_backend_when_no_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("AGENT_MEMORY_ID", raising=False)
    monkeypatch.delenv("STATE_BUCKET", raising=False)
    monkeypatch.setenv("STATE_DIR", str(tmp_path / "auto-state"))

    sm = build_default_manager()
    assert len(sm.backends) == 1
    assert sm.backends[0].name == "file"
