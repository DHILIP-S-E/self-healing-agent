"""
Central configuration for the self-healing agent.

Reads environment variables once at import time. All values have safe
defaults so unit tests can run without any AWS configuration.

Variables:
    AWS_REGION              ap-south-1 (Mumbai)
    INPUT_BUCKET            S3 bucket holding input PDFs
    OUTPUT_BUCKET           S3 bucket where summaries are written
    STATE_BUCKET            S3 bucket for fallback JSON checkpoints
    AGENT_MEMORY_ID         AgentCore Memory resource ID (primary state)
    AGENT_RUNTIME_ARN       AgentCore Runtime ARN (set by deploy)
    BEDROCK_MODEL_ID        Bedrock model used for summarization
    STATE_DIR               Local fallback dir for checkpoints
    LOG_LEVEL               INFO / DEBUG / WARNING

Source: .env.example documents the full surface.
"""

from __future__ import annotations

import os

# Load .env if present. In AgentCore Runtime, env comes from the runtime
# config so .env may not exist — that's fine.
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass


# --- AWS / region ---
AWS_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "ap-south-1"

# --- S3 buckets (filled in by `cdk deploy`) ---
INPUT_BUCKET = os.environ.get("INPUT_BUCKET", "")
OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET", "")
STATE_BUCKET = os.environ.get("STATE_BUCKET", "")

# --- AgentCore IDs (filled in by deploy) ---
AGENT_MEMORY_ID = os.environ.get("AGENT_MEMORY_ID", "")
AGENT_RUNTIME_ARN = os.environ.get("AGENT_RUNTIME_ARN", "")

# --- Bedrock model for summarization ---
# Meta Llama 3 70B Instruct — confirmed ACTIVE in ap-south-1.
# Used by summarize_text tool via invoke_model directly.
BEDROCK_MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID",
    "meta.llama3-70b-instruct-v1:0",
)

# --- Bedrock model for Strands reasoning agent ---
# Amazon Nova Micro — cheapest model with tool use support in ap-south-1.
# Used by the Strands Agent to decide what to do (orchestration only).
# Summarization is still done by BEDROCK_MODEL_ID via invoke_model.
REASONING_MODEL_ID = os.environ.get(
    "REASONING_MODEL_ID",
    "mistral.mistral-large-2402-v1:0",  # tool use via Converse API, ACTIVE in ap-south-1
)

# --- Local checkpoint fallback ---
STATE_DIR = os.environ.get("STATE_DIR", "./state")

# --- Tunables ---
MAX_PDF_BYTES = int(os.environ.get("MAX_PDF_BYTES", str(60 * 1024 * 1024)))   # 60 MB — text is truncated to 20k chars anyway
# Llama 3 70B max context = 8192 tokens. At ~4 chars/token, minus prompt overhead:
# safe limit ≈ 20 000 chars (~5 000 tokens), leaving 3 000 tokens for response.
MAX_TEXT_CHARS_FOR_SUMMARY = int(os.environ.get("MAX_TEXT_CHARS_FOR_SUMMARY", "20000"))

SUMMARY_MAX_WORDS_DEFAULT = int(os.environ.get("SUMMARY_MAX_WORDS", "100"))
BEDROCK_MAX_TOKENS = int(os.environ.get("BEDROCK_MAX_TOKENS", "512"))

# --- Logging ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
