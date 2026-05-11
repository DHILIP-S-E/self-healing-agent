"""
Tools the agent uses to do its job.

Each function is decorated with @tool from Strands so the framework can
register it as an LLM-callable capability. The deterministic loop in
agent/main.py also calls these directly (not through LLM reasoning) — the
@tool decorator preserves the underlying callable so both paths work.

Tools:
    list_documents   List PDF keys in an S3 bucket
    read_document    Download + extract plain text from a PDF
    summarize_text   Call Bedrock Claude Haiku to summarize text
    save_summary     Write a summary as Markdown to S3

Sources:
    https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html
    https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html
    https://pypdf.readthedocs.io/
"""

from __future__ import annotations

import io
import json
import logging
import time
from typing import Any

import boto3
from botocore.exceptions import ClientError
from pypdf import PdfReader
from pypdf.errors import PdfReadError

# UNVERIFIED — exact import path for the @tool decorator may be
# `from strands import tool` or `from strands.tools import tool`
# depending on strands-agents version. We try both at import time.
try:
    from strands import tool
except ImportError:  # pragma: no cover
    try:
        from strands.tools import tool  # type: ignore[no-redef]
    except ImportError:
        # If Strands isn't installed (e.g. unit tests), use a no-op so the
        # underlying functions remain callable.
        def tool(fn):  # type: ignore[no-redef]
            return fn

from .config import (
    AWS_REGION,
    BEDROCK_MAX_TOKENS,
    BEDROCK_MODEL_ID,
    MAX_PDF_BYTES,
    MAX_TEXT_CHARS_FOR_SUMMARY,
    STATE_BUCKET,
    SUMMARY_MAX_WORDS_DEFAULT,
)

logger = logging.getLogger(__name__)

# Module-level clients — replaceable in tests via monkeypatch on the names.
_s3 = boto3.client("s3", region_name=AWS_REGION)
_bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)


# =============================================================================
# list_documents
# =============================================================================

@tool
def list_documents(bucket: str, prefix: str = "") -> list[str]:
    """List every PDF object key in an S3 bucket.

    Use this first to discover what documents are available to process.

    Args:
        bucket: Name of the S3 bucket to list.
        prefix: Optional key prefix filter (e.g. "papers/" lists only that folder).

    Returns:
        Sorted list of S3 object keys ending in .pdf (case-insensitive).
    """
    paginator = _s3.get_paginator("list_objects_v2")
    keys: list[str] = []
    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []) or []:
                key = obj["Key"]
                if key.lower().endswith(".pdf"):
                    keys.append(key)
    except ClientError as e:
        raise RuntimeError(f"list_documents failed for s3://{bucket}/{prefix}: {e}") from e

    keys.sort()
    logger.info("list_documents: %d PDFs at s3://%s/%s", len(keys), bucket, prefix)
    return keys


# =============================================================================
# read_document
# =============================================================================

@tool
def read_document(bucket: str, key: str) -> str:
    """Download a PDF from S3 and return its extracted plain text.

    Refuses PDFs larger than MAX_PDF_BYTES (10 MB by default) to avoid
    runaway memory use on a poisoned input.

    Args:
        bucket: S3 bucket containing the PDF.
        key: S3 object key (must point to a PDF file).

    Returns:
        All pages concatenated as plain text, separated by blank lines.
    """
    try:
        head = _s3.head_object(Bucket=bucket, Key=key)
    except ClientError as e:
        raise RuntimeError(f"read_document: head_object failed for {key}: {e}") from e

    size = int(head.get("ContentLength", 0))
    if size > MAX_PDF_BYTES:
        raise RuntimeError(
            f"read_document: {key} is {size} bytes, exceeds limit {MAX_PDF_BYTES}"
        )

    try:
        resp = _s3.get_object(Bucket=bucket, Key=key)
        pdf_bytes = resp["Body"].read()
    except ClientError as e:
        raise RuntimeError(f"read_document: get_object failed for {key}: {e}") from e

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = [(p.extract_text() or "") for p in reader.pages]
    except (PdfReadError, Exception) as e:
        raise RuntimeError(f"read_document: failed to parse PDF {key}: {e}") from e

    text = "\n\n".join(pages).strip()
    logger.info("read_document: %s -> %d chars across %d pages", key, len(text), len(pages))
    return text


# =============================================================================
# summarize_text
# =============================================================================

@tool
def summarize_text(text: str, max_words: int = SUMMARY_MAX_WORDS_DEFAULT) -> str:
    """Summarize a body of text using Bedrock (supports Claude and Llama models).

    Automatically detects model family from BEDROCK_MODEL_ID and formats the
    request body accordingly:
      - meta.*   → Llama 3 prompt format
      - anthropic.* → Claude Messages API format

    Args:
        text: The full text to summarize. Empty/whitespace returns "".
        max_words: Approximate target word count for the summary.

    Returns:
        A summary string, roughly `max_words` words. Empty string on empty input.
    """
    if not text or not text.strip():
        return ""

    truncated = text[:MAX_TEXT_CHARS_FOR_SUMMARY]
    user_prompt = (
        f"Summarize the following document in approximately {max_words} words. "
        f"Focus on the main contribution, methods, and key findings. "
        f"Output only the summary text, no preamble.\n\n"
        f"<document>\n{truncated}\n</document>"
    )

    # Build request body based on model family
    # Source: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-meta.html
    # Source: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html
    if BEDROCK_MODEL_ID.startswith("meta."):
        # Llama 3 chat format
        llama_prompt = (
            "<|begin_of_text|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_prompt}"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        body = json.dumps({
            "prompt": llama_prompt,
            "max_gen_len": BEDROCK_MAX_TOKENS,
            "temperature": 0.5,
        })
    else:
        # Claude Messages API format (default)
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": BEDROCK_MAX_TOKENS,
            "messages": [{"role": "user", "content": user_prompt}],
        })

    # Retry with exponential backoff on throttling (standard AWS best practice).
    # Source: https://docs.aws.amazon.com/general/latest/gr/api-retries.html
    max_attempts = 6
    for attempt in range(max_attempts):
        try:
            resp = _bedrock.invoke_model(modelId=BEDROCK_MODEL_ID, body=body)
            payload = json.loads(resp["body"].read())
            break  # success
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code == "ThrottlingException" and attempt < max_attempts - 1:
                wait = 2 ** attempt  # 1s, 2s, 4s, 8s, 16s
                logger.warning(
                    "summarize_text: throttled (attempt %d/%d), retrying in %ds",
                    attempt + 1, max_attempts, wait,
                )
                time.sleep(wait)
            else:
                raise RuntimeError(f"summarize_text: bedrock invoke_model failed: {e}") from e

    # Parse response based on model family
    if BEDROCK_MODEL_ID.startswith("meta."):
        summary = payload.get("generation", "").strip()
    else:
        parts = [
            block.get("text", "")
            for block in payload.get("content", [])
            if block.get("type") == "text"
        ]
        summary = "".join(parts).strip()

    logger.debug("summarize_text: input=%d chars, summary=%d chars", len(truncated), len(summary))
    return summary


# =============================================================================
# save_summary
# =============================================================================

@tool
def save_summary(bucket: str, key: str, summary: str) -> dict[str, Any]:
    """Save a summary string to S3 as a Markdown object.

    A `.md` extension is appended to `key` if not already present.

    Args:
        bucket: Output S3 bucket.
        key: S3 object key (e.g. "summaries/task-001/paper.pdf"). `.md` is auto-appended.
        summary: The summary text to write.

    Returns:
        Dict with `bucket`, `key`, `size_bytes`, `content_type`.
    """
    if not key.lower().endswith(".md"):
        key = f"{key}.md"

    body = (summary or "").encode("utf-8")
    try:
        _s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="text/markdown; charset=utf-8",
            ServerSideEncryption="AES256",
        )
    except ClientError as e:
        raise RuntimeError(f"save_summary: put_object failed for {key}: {e}") from e

    logger.info("save_summary: wrote s3://%s/%s (%d bytes)", bucket, key, len(body))
    return {
        "bucket": bucket,
        "key": key,
        "size_bytes": len(body),
        "content_type": "text/markdown",
    }


# =============================================================================
# Checkpoint tools — callable by the LLM so it can manage its own state
# =============================================================================

@tool
def load_checkpoint(task_id: str) -> dict[str, Any]:
    """Load the agent's saved progress checkpoint from S3.

    Call this at the start of every run so you know which documents are
    already done and where to resume from.

    Args:
        task_id: Unique identifier for this processing run.

    Returns:
        Dict with `completed_idx` (last finished index, -1 if none),
        `completed_keys` (list of already-processed S3 keys),
        `errors` (list of previously failed keys).
        Returns fresh state if no checkpoint exists.
    """
    if not STATE_BUCKET:
        return {"completed_idx": -1, "completed_keys": [], "errors": []}

    key = f"checkpoints/{task_id}/state.json"
    try:
        resp = _s3.get_object(Bucket=STATE_BUCKET, Key=key)
        data = json.loads(resp["Body"].read())
        logger.info("load_checkpoint: task=%s completed_idx=%s", task_id, data.get("completed_idx"))
        return data
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            logger.info("load_checkpoint: no prior checkpoint for task=%s", task_id)
            return {"completed_idx": -1, "completed_keys": [], "errors": []}
        raise RuntimeError(f"load_checkpoint: S3 error: {e}") from e


@tool
def save_checkpoint(task_id: str, completed_idx: int, completed_key: str, errors: list[str] | None = None) -> dict[str, Any]:
    """Save your progress after successfully processing one document.

    Call this immediately after save_summary succeeds — before moving to
    the next document. This is what makes the agent self-healing: if you
    crash after this call, you resume from the next document automatically.

    Args:
        task_id: Unique identifier for this processing run.
        completed_idx: Index of the document just completed (0-based).
        completed_key: S3 key of the document just completed.
        errors: Optional updated list of failed document keys.

    Returns:
        Dict confirming the checkpoint was saved.
    """
    if not STATE_BUCKET:
        logger.warning("save_checkpoint: STATE_BUCKET not set — checkpoint skipped")
        return {"saved": False, "reason": "STATE_BUCKET not configured"}

    # Load existing state to append to completed_keys list
    existing = load_checkpoint(task_id)
    completed_keys = existing.get("completed_keys", [])
    if completed_key not in completed_keys:
        completed_keys.append(completed_key)

    state = {
        "task_id": task_id,
        "completed_idx": completed_idx,
        "completed_keys": completed_keys,
        "errors": errors or existing.get("errors", []),
    }

    s3_key = f"checkpoints/{task_id}/state.json"
    try:
        _s3.put_object(
            Bucket=STATE_BUCKET,
            Key=s3_key,
            Body=json.dumps(state).encode(),
            ContentType="application/json",
            ServerSideEncryption="AES256",
        )
    except ClientError as e:
        raise RuntimeError(f"save_checkpoint: S3 write failed: {e}") from e

    logger.info("save_checkpoint: task=%s idx=%d key=%s", task_id, completed_idx, completed_key)
    return {"saved": True, "completed_idx": completed_idx, "total_completed": len(completed_keys)}


@tool
def record_error(task_id: str, doc_key: str, error_message: str) -> dict[str, Any]:
    """Record that a document failed so you can skip it and continue.

    Call this when a document cannot be processed (too large, corrupted,
    throttled after retries). This lets you make the autonomous decision
    to skip it rather than getting stuck — that is the self-healing behaviour.

    Args:
        task_id: Unique identifier for this processing run.
        doc_key: S3 key of the document that failed.
        error_message: Short description of what went wrong.

    Returns:
        Dict with updated error count.
    """
    existing = load_checkpoint(task_id)
    errors: list[dict] = existing.get("errors", [])
    errors.append({"key": doc_key, "error": error_message})

    s3_key = f"checkpoints/{task_id}/state.json"
    state = {**existing, "errors": errors}
    try:
        _s3.put_object(
            Bucket=STATE_BUCKET,
            Key=s3_key,
            Body=json.dumps(state).encode(),
            ContentType="application/json",
            ServerSideEncryption="AES256",
        )
    except ClientError as e:
        logger.warning("record_error: could not persist error record: %s", e)

    logger.warning("record_error: task=%s skipped %s — %s", task_id, doc_key, error_message)
    return {"recorded": True, "total_errors": len(errors), "doc_key": doc_key}
