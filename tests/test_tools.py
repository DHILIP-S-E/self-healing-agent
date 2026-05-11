"""
Unit tests for agent.tools.

Each tool gets at least one test. S3 is mocked with moto; Bedrock is mocked
with a MagicMock since moto's Bedrock support varies by version.
"""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import boto3
import pytest


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_aws_environment():
    """Patch the agent.tools._s3 client with a moto-mocked client + buckets."""
    moto = pytest.importorskip("moto")
    from moto import mock_aws  # type: ignore[import-not-found]

    with mock_aws():
        client = boto3.client("s3", region_name="ap-south-1")
        for bucket in ("in-bucket", "out-bucket"):
            client.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": "ap-south-1"},
            )
        with patch("agent.tools._s3", client):
            yield client


def _make_blank_pdf_bytes() -> bytes:
    """Build a minimal valid (but blank) PDF using pypdf — no extra deps."""
    from pypdf import PdfWriter
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


# =============================================================================
# list_documents
# =============================================================================

def test_list_documents_returns_only_pdfs(mock_aws_environment):
    s3 = mock_aws_environment
    s3.put_object(Bucket="in-bucket", Key="a.pdf", Body=b"x")
    s3.put_object(Bucket="in-bucket", Key="b.pdf", Body=b"y")
    s3.put_object(Bucket="in-bucket", Key="readme.txt", Body=b"z")

    from agent.tools import list_documents
    keys = list_documents(bucket="in-bucket")

    assert keys == ["a.pdf", "b.pdf"]


def test_list_documents_with_prefix(mock_aws_environment):
    s3 = mock_aws_environment
    s3.put_object(Bucket="in-bucket", Key="papers/p1.pdf", Body=b"x")
    s3.put_object(Bucket="in-bucket", Key="papers/p2.pdf", Body=b"y")
    s3.put_object(Bucket="in-bucket", Key="other/q.pdf",  Body=b"z")

    from agent.tools import list_documents
    keys = list_documents(bucket="in-bucket", prefix="papers/")

    assert keys == ["papers/p1.pdf", "papers/p2.pdf"]


def test_list_documents_empty_bucket(mock_aws_environment):
    from agent.tools import list_documents
    assert list_documents(bucket="in-bucket") == []


# =============================================================================
# read_document
# =============================================================================

def test_read_document_returns_string_for_valid_pdf(mock_aws_environment):
    s3 = mock_aws_environment
    s3.put_object(Bucket="in-bucket", Key="blank.pdf", Body=_make_blank_pdf_bytes())

    from agent.tools import read_document
    text = read_document(bucket="in-bucket", key="blank.pdf")

    # Blank pages produce empty/whitespace text — that's fine, just verify type.
    assert isinstance(text, str)


def test_read_document_raises_on_invalid_pdf(mock_aws_environment):
    s3 = mock_aws_environment
    s3.put_object(Bucket="in-bucket", Key="bad.pdf", Body=b"not a real pdf")

    from agent.tools import read_document
    with pytest.raises(RuntimeError, match="failed to parse PDF"):
        read_document(bucket="in-bucket", key="bad.pdf")


def test_read_document_refuses_oversized_pdf(mock_aws_environment, monkeypatch):
    monkeypatch.setattr("agent.tools.MAX_PDF_BYTES", 100)
    s3 = mock_aws_environment
    s3.put_object(Bucket="in-bucket", Key="big.pdf", Body=b"x" * 500)

    from agent.tools import read_document
    with pytest.raises(RuntimeError, match="exceeds limit"):
        read_document(bucket="in-bucket", key="big.pdf")


# =============================================================================
# summarize_text
# =============================================================================

def test_summarize_text_empty_returns_empty():
    from agent.tools import summarize_text
    assert summarize_text("") == ""
    assert summarize_text("   \n\n  ") == ""


def test_summarize_text_calls_bedrock_with_messages():
    fake_response_body = json.dumps({
        "content": [{"type": "text", "text": "This is the summary."}],
    }).encode()
    fake_resp = {"body": MagicMock(read=lambda: fake_response_body)}
    fake_bedrock = MagicMock()
    fake_bedrock.invoke_model.return_value = fake_resp

    with patch("agent.tools._bedrock", fake_bedrock):
        from agent.tools import summarize_text
        result = summarize_text("Some long document content to summarize.")

    assert result == "This is the summary."
    fake_bedrock.invoke_model.assert_called_once()

    call_kwargs = fake_bedrock.invoke_model.call_args.kwargs
    assert "modelId" in call_kwargs
    body = json.loads(call_kwargs["body"])
    assert body["anthropic_version"] == "bedrock-2023-05-31"
    assert "messages" in body
    assert body["messages"][0]["role"] == "user"


def test_summarize_text_truncates_very_long_input():
    fake_resp = {
        "body": MagicMock(read=lambda: json.dumps({
            "content": [{"type": "text", "text": "ok"}]
        }).encode())
    }
    fake_bedrock = MagicMock()
    fake_bedrock.invoke_model.return_value = fake_resp

    huge = "A" * 200_000
    with patch("agent.tools._bedrock", fake_bedrock):
        from agent.tools import summarize_text
        summarize_text(huge)

    body = json.loads(fake_bedrock.invoke_model.call_args.kwargs["body"])
    sent_prompt = body["messages"][0]["content"]
    # The prompt embeds the truncated text; total should be well below 200k.
    assert len(sent_prompt) < 80_000


# =============================================================================
# save_summary
# =============================================================================

def test_save_summary_writes_markdown(mock_aws_environment):
    from agent.tools import save_summary
    result = save_summary(
        bucket="out-bucket",
        key="summaries/task-1/paper",
        summary="This is the summary text.",
    )

    assert result["bucket"] == "out-bucket"
    assert result["key"] == "summaries/task-1/paper.md"
    assert result["content_type"] == "text/markdown"
    assert result["size_bytes"] == len("This is the summary text.".encode("utf-8"))

    body = mock_aws_environment.get_object(
        Bucket="out-bucket", Key="summaries/task-1/paper.md",
    )["Body"].read()
    assert body == b"This is the summary text."


def test_save_summary_does_not_double_append_md_extension(mock_aws_environment):
    from agent.tools import save_summary
    result = save_summary(bucket="out-bucket", key="report.md", summary="hi")
    assert result["key"] == "report.md"  # not "report.md.md"
