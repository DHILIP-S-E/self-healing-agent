"""
State manager for the self-healing document processing agent.

Persists checkpoint state across crashes so the agent can resume from the last
completed document. Three backends, auto-picked at construction time:

    1. AgentCore Memory   (primary, when AGENT_MEMORY_ID is set)
    2. S3 JSON            (secondary, when STATE_BUCKET is set)
    3. Local file JSON    (fallback for tests / dev, always available)

If the primary backend raises during write/read, the StateManager falls back
to the next configured backend automatically. A successful write to the
primary OR the fallback counts as a successful checkpoint.

State schema (versioned):

    {
        "schema_version": 1,
        "task_id":        "task-001",
        "completed_idx":  47,            # last completed document index
        "results":        [ {...}, ... ],# accumulated per-doc results
        "status":         "in_progress", # or "complete"
        "started_at":     "2026-05-06T03:14:15Z",
        "updated_at":     "2026-05-06T04:00:00Z"
    }

Public API:
    StateManager(backends: list[_Backend])
        .save_progress(task_id, completed_idx, results) -> None
        .load_progress(task_id) -> dict | None
        .mark_complete(task_id)                          -> None
        .clear_state(task_id)                            -> None

    build_default_manager() -> StateManager
        Reads env vars and assembles the right backend chain.

Sources:
    - https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/memory.html
    - https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# Bump this number any time the on-disk shape of the checkpoint changes.
SCHEMA_VERSION = 1

STATUS_IN_PROGRESS = "in_progress"
STATUS_COMPLETE = "complete"


# =============================================================================
# Backend interface
# =============================================================================

class _Backend(ABC):
    """Stateless storage backend: one JSON blob keyed by task_id."""

    name: str = "abstract"

    @abstractmethod
    def write(self, task_id: str, payload: dict) -> None: ...

    @abstractmethod
    def read(self, task_id: str) -> dict | None: ...

    @abstractmethod
    def delete(self, task_id: str) -> None: ...


# =============================================================================
# File backend (always available — used for local dev + unit tests)
# =============================================================================

class FileBackend(_Backend):
    """Local-file JSON. Atomic writes via tmp-file + rename."""

    name = "file"

    def __init__(self, state_dir: str | Path):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, task_id: str) -> Path:
        return self.state_dir / f"{task_id}.checkpoint.json"

    def write(self, task_id: str, payload: dict) -> None:
        path = self._path(task_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        # os.replace is atomic on POSIX and Windows alike — a crash mid-write
        # never leaves a half-written checkpoint visible at `path`.
        os.replace(tmp, path)
        logger.debug("[file] wrote %s", path)

    def read(self, task_id: str) -> dict | None:
        path = self._path(task_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def delete(self, task_id: str) -> None:
        path = self._path(task_id)
        if path.exists():
            path.unlink()


# =============================================================================
# S3 backend
# =============================================================================

class S3Backend(_Backend):
    """S3 JSON object per task. Key: <prefix>/<task_id>.json"""

    name = "s3"

    def __init__(
        self,
        bucket: str,
        prefix: str = "checkpoints",
        region: str | None = None,
        client: Any = None,  # injectable for tests (moto)
    ):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        # Source: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html
        self.client = client if client is not None else boto3.client("s3", region_name=self.region)

    def _key(self, task_id: str) -> str:
        return f"{self.prefix}/{task_id}.json"

    def write(self, task_id: str, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.client.put_object(
            Bucket=self.bucket,
            Key=self._key(task_id),
            Body=body,
            ContentType="application/json",
            ServerSideEncryption="AES256",
        )
        logger.debug("[s3] wrote s3://%s/%s (%d bytes)", self.bucket, self._key(task_id), len(body))

    def read(self, task_id: str) -> dict | None:
        try:
            resp = self.client.get_object(Bucket=self.bucket, Key=self._key(task_id))
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("NoSuchKey", "404", "NotFound"):
                return None
            raise
        return json.loads(resp["Body"].read())

    def delete(self, task_id: str) -> None:
        try:
            self.client.delete_object(Bucket=self.bucket, Key=self._key(task_id))
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("NoSuchKey", "404", "NotFound"):
                return
            raise


# =============================================================================
# AgentCore Memory backend
# =============================================================================
# UNVERIFIED API — please confirm the exact MemoryClient method names against
# your installed bedrock-agentcore version. Run:
#     python -c "from bedrock_agentcore.memory import MemoryClient; help(MemoryClient)"
# and adjust the create_event / list_events / delete_session calls below if
# the method names differ.
#
# Source (intended): https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/memory-getting-started.html
#
# We use the "events" primitive of AgentCore Memory to store JSON checkpoints:
#     actor_id   = "self-healing-agent"
#     session_id = task_id
#     event body = JSON-encoded checkpoint payload
# The newest event in a session = the current checkpoint.

class MemoryBackend(_Backend):
    """AgentCore Memory backend (primary in production)."""

    name = "memory"
    ACTOR_ID = "self-healing-agent"

    def __init__(self, memory_id: str, region: str | None = None, client: Any = None):
        self.memory_id = memory_id
        self.region = region or os.environ.get("AWS_REGION")
        if client is not None:
            self.client = client
        else:
            try:
                from bedrock_agentcore.memory import MemoryClient  # type: ignore[import-not-found]
            except ImportError as e:
                raise RuntimeError(
                    "bedrock_agentcore.memory.MemoryClient is not importable. "
                    "Install: pip install bedrock-agentcore>=0.1.0"
                ) from e
            self.client = MemoryClient(region_name=self.region)

    def write(self, task_id: str, payload: dict) -> None:
        body = json.dumps(payload)
        # UNVERIFIED — confirm method signature
        self.client.create_event(
            memory_id=self.memory_id,
            actor_id=self.ACTOR_ID,
            session_id=task_id,
            messages=[{"role": "user", "content": body}],
        )
        logger.debug("[memory] wrote event for session=%s", task_id)

    def read(self, task_id: str) -> dict | None:
        # UNVERIFIED — confirm method signature
        events = self.client.list_events(
            memory_id=self.memory_id,
            actor_id=self.ACTOR_ID,
            session_id=task_id,
            max_results=1,
        )
        if not events:
            return None
        # Different SDK versions wrap the response differently; handle both.
        event = events[0] if isinstance(events, list) else events.get("events", [{}])[0]
        messages = event.get("messages") or event.get("content") or []
        if not messages:
            return None
        last = messages[-1] if isinstance(messages, list) else messages
        content = last.get("content") if isinstance(last, dict) else last
        if not content:
            return None
        return json.loads(content)

    def delete(self, task_id: str) -> None:
        # Best-effort: not all SDK versions expose per-session delete.
        delete_fn = getattr(self.client, "delete_session", None)
        if delete_fn is None:
            logger.info("[memory] delete_session not available in this SDK; leaving session %s in place", task_id)
            return
        try:
            delete_fn(memory_id=self.memory_id, actor_id=self.ACTOR_ID, session_id=task_id)
        except Exception as e:  # noqa: BLE001
            logger.warning("[memory] delete_session failed (non-fatal): %s", e)


# =============================================================================
# StateManager
# =============================================================================

class StateManager:
    """
    Orchestrates writes/reads across a chain of backends.

    On write: try each backend in order; succeed if at least one succeeds.
    On read:  try each backend in order; return the first hit.

    The first backend in the list is treated as primary (warnings are louder
    when it fails). Subsequent backends serve as fallbacks.
    """

    def __init__(self, backends: list[_Backend]):
        if not backends:
            raise ValueError("StateManager requires at least one backend")
        self.backends = backends

    # ------------------------------------------------------------------- write

    def save_progress(self, task_id: str, completed_idx: int, results: list[dict]) -> None:
        """Persist progress checkpoint. Called after every completed document."""
        payload = self._build_payload(task_id, completed_idx, results, STATUS_IN_PROGRESS)
        self._write_to_any(task_id, payload, what="checkpoint")

    def mark_complete(self, task_id: str) -> None:
        """
        Mark a task as fully complete.

        After this call, `load_progress(task_id)` returns `None`, so a fresh
        run with the same task_id starts from scratch instead of trying to
        resume a finished job. Implementation: delete the checkpoint from
        every configured backend.
        """
        errors: list[str] = []
        for backend in self.backends:
            try:
                backend.delete(task_id)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{backend.name}: {e}")
        if errors:
            logger.warning("mark_complete: some backends errored (non-fatal): %s", "; ".join(errors))

    def clear_state(self, task_id: str) -> None:
        """Hard delete checkpoint state. Useful for tests + manual reset."""
        # Same semantics as mark_complete but named for the intent of throwing
        # state away rather than declaring success.
        self.mark_complete(task_id)

    # -------------------------------------------------------------------- read

    def load_progress(self, task_id: str) -> dict | None:
        """
        Return the most recent valid checkpoint, or None.

        Returns None when:
            - no checkpoint exists in any backend
            - the checkpoint exists but its schema_version != SCHEMA_VERSION
              (logged as a warning; caller will start from scratch)
        """
        for backend in self.backends:
            try:
                payload = backend.read(task_id)
            except Exception as e:  # noqa: BLE001
                logger.warning("[%s] read failed (trying next backend): %s", backend.name, e)
                continue
            if payload is None:
                continue
            if not self._is_compatible(payload):
                logger.warning(
                    "[%s] checkpoint schema_version=%s does not match expected=%s; ignoring",
                    backend.name,
                    payload.get("schema_version"),
                    SCHEMA_VERSION,
                )
                return None
            return payload
        return None

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _build_payload(
        task_id: str, completed_idx: int, results: list[dict], status: str
    ) -> dict:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return {
            "schema_version": SCHEMA_VERSION,
            "task_id": task_id,
            "completed_idx": completed_idx,
            "results": list(results),
            "status": status,
            "started_at": now,  # callers can preserve this themselves if they want
            "updated_at": now,
        }

    @staticmethod
    def _is_compatible(payload: dict) -> bool:
        return payload.get("schema_version") == SCHEMA_VERSION

    def _write_to_any(self, task_id: str, payload: dict, what: str) -> None:
        last_err: Exception | None = None
        for i, backend in enumerate(self.backends):
            try:
                backend.write(task_id, payload)
                if i > 0:
                    logger.warning(
                        "[%s] %s saved via fallback (primary backend failed)",
                        backend.name, what,
                    )
                return
            except Exception as e:  # noqa: BLE001
                logger.warning("[%s] %s write failed: %s", backend.name, what, e)
                last_err = e
        # Every backend failed — re-raise so the agent loop knows the
        # checkpoint did not land. The caller decides whether to retry.
        raise RuntimeError(
            f"Failed to write {what} to any of {len(self.backends)} backend(s)"
        ) from last_err


# =============================================================================
# Factory
# =============================================================================

def build_default_manager() -> StateManager:
    """
    Read env vars and assemble the right backend chain.

    Order of precedence (each becomes a fallback for the next):
        1. AgentCore Memory   if AGENT_MEMORY_ID is set
        2. S3                 if STATE_BUCKET is set
        3. File               always (uses STATE_DIR or ./state)
    """
    backends: list[_Backend] = []

    memory_id = os.environ.get("AGENT_MEMORY_ID", "").strip()
    if memory_id:
        try:
            backends.append(MemoryBackend(memory_id=memory_id))
        except RuntimeError as e:
            logger.warning("Skipping MemoryBackend: %s", e)

    state_bucket = os.environ.get("STATE_BUCKET", "").strip()
    if state_bucket:
        backends.append(S3Backend(bucket=state_bucket))

    state_dir = os.environ.get("STATE_DIR", "./state")
    backends.append(FileBackend(state_dir=state_dir))

    logger.info("StateManager backends: %s", [b.name for b in backends])
    return StateManager(backends=backends)


# =============================================================================
# Smoke test (run: `python agent/state_manager.py`)
# =============================================================================

if __name__ == "__main__":
    import shutil
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    tmp = tempfile.mkdtemp(prefix="sha-smoke-")
    try:
        sm = StateManager(backends=[FileBackend(state_dir=tmp)])

        print("\n[1] save_progress + load_progress roundtrip")
        sm.save_progress("task-smoke", completed_idx=3, results=[{"doc": "a"}, {"doc": "b"}, {"doc": "c"}])
        loaded = sm.load_progress("task-smoke")
        assert loaded is not None
        assert loaded["completed_idx"] == 3
        assert loaded["status"] == STATUS_IN_PROGRESS
        assert len(loaded["results"]) == 3
        print(f"    OK  -> completed_idx={loaded['completed_idx']}, results={len(loaded['results'])}")

        print("\n[2] mark_complete makes load_progress return None")
        sm.mark_complete("task-smoke")
        assert sm.load_progress("task-smoke") is None
        print("    OK  -> load_progress returned None")

        print("\n[3] no state at all -> None")
        assert sm.load_progress("never-existed") is None
        print("    OK  -> load_progress returned None")

        print("\n[4] schema mismatch -> None (with warning)")
        bad = {"schema_version": 999, "task_id": "task-bad", "completed_idx": 1, "results": []}
        FileBackend(tmp).write("task-bad", bad)
        assert sm.load_progress("task-bad") is None
        print("    OK  -> load_progress returned None for bad schema")

        print("\nAll smoke tests passed.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
