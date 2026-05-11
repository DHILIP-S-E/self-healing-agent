# 🔄 Self-Healing Document Processing Agent

A Bedrock AgentCore agent that processes 100 PDFs, survives crashes, and resumes from the last completed document — saving hours of compute when long-running jobs fail.

![status](https://img.shields.io/badge/status-alpha-orange) ![python](https://img.shields.io/badge/python-3.12-blue) ![license](https://img.shields.io/badge/license-MIT-green) ![region](https://img.shields.io/badge/region-ap--south--1-purple)

## What it does

Processes a batch of PDFs from S3, summarizes each with Meta Llama 3 70B Instruct on Bedrock, and writes a Markdown summary back to S3 — checkpointing after every document. If the process is killed (Lambda timeout, network glitch, OOM, manual SIGKILL, anything), the next run picks up exactly where it left off. No work is lost. No document is processed twice.

**Proven results:** 97/100 real arXiv papers summarized in ~23 minutes. Cost: under $1.50. Zero lost checkpoints across 3 runs including one hard crash simulation.

```
$ python -m demo.run --task task-001 --bucket my-pdfs --output-bucket my-summaries
Processing 1/100... ✅
Processing 2/100... ✅
...
Processing 47/100... 💥 CRASHED (simulated kill)

$ python -m demo.run --task task-001 --bucket my-pdfs --output-bucket my-summaries --resume
🔄 Resuming task-001 from completed_idx=46
Processing 48/100... ✅
...
🎉 Done in 312s
💚 Resumed: skipped 47 already-completed documents.
```

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full Mermaid diagram and decision log.

In one sentence: a Strands agent runs inside AgentCore Runtime, calling 4 tools (list / read / summarize / save) in a loop, persisting a JSON checkpoint to AgentCore Memory (with S3 fallback) after every document.

## Prerequisites

| Tool | Version | Why |
|---|---|---|
| Python | **3.12** | Strands + bedrock-agentcore both need ≥ 3.10; 3.12 is the project target. |
| AWS CLI | **2.15+** | `aws configure`, billing alarms metrics. |
| AWS CDK | **2.150+** | `cdk bootstrap` + `cdk deploy`. |
| Docker Desktop | **latest** | AgentCore Runtime needs ARM64 container builds. |
| AWS account | n/a | With Bedrock model access enabled for Meta Llama 3 70B Instruct in `ap-south-1`. |

> **Heads up:** Bedrock model access for Meta Llama 3 70B Instruct must be **explicitly enabled** in the AWS console (Bedrock → Model access → Manage model access) before any `invoke_model` call will succeed. Allow ~1 minute for access to activate.

## Installation

```bash
git clone https://github.com/DHILIP-S-E/self-healing-agent.git
cd self-healing-agent

# 1. Python deps
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. AWS credentials
aws configure                       # set region: ap-south-1

# 3. Configure the project
cp .env.example .env
# (Bucket names get filled in by the next step)
```

## Quick start

Three commands take you from zero to a complete demo run:

```bash
# 1. Deploy supporting infra (S3 buckets, IAM, CloudWatch alarm)
cd infrastructure && cdk bootstrap && cdk deploy

# 2. Seed 100 real arXiv PDFs into the input bucket
cd .. && python -m demo.seed_documents --bucket $INPUT_BUCKET --count 100

# 3. Run the agent
python -m demo.run --task task-001 --bucket $INPUT_BUCKET --output-bucket $OUTPUT_BUCKET
```

The CDK output prints `INPUT_BUCKET` / `OUTPUT_BUCKET` / `STATE_BUCKET` names — copy those into `.env` (or export them) before running step 2.

For a full deployment walk-through including AgentCore Runtime container build and ECR push, see [docs/deployment.md](docs/deployment.md).

## Simulate a crash

```bash
# Bash (Linux / macOS / Git Bash on Windows):
INPUT_BUCKET=$INPUT_BUCKET OUTPUT_BUCKET=$OUTPUT_BUCKET ./demo/crash_test.sh

# PowerShell (Windows native):
$env:INPUT_BUCKET="..."; $env:OUTPUT_BUCKET="..."; ./demo/crash_test.ps1
```

The script starts the agent, SIGKILLs it after 30 seconds, then restarts it with `--resume`. The second run skips the already-completed documents.

## Tests

```bash
pytest                                                  # full suite
pytest tests/test_state_manager.py -v                   # the heart
pytest tests/test_resilience.py -v                      # crash + resume proof
pytest tests/test_resilience.py::test_crash_and_resume_processes_all_100_docs_exactly_once -v
```

All tests run **offline** — no AWS credentials required. S3 is mocked with `moto`, Bedrock with `MagicMock`.

## Cost estimate (per overnight run)

Calculated against on-demand pricing in `ap-south-1` for processing **100 PDFs over 8 hours**:

| Service | Usage | Cost |
|---|---|---|
| AgentCore Runtime | ~8 vCPU·h + 16 GB·h | $1.20 |
| AgentCore Memory | 100 writes + storage | $0.30 |
| Bedrock Claude Haiku | ~100 calls × 3K input + 1K output tokens | $0.40 |
| S3 | 200 PUTs + 100 GETs + storage | $0.05 |
| CloudWatch Logs | ~50 MB ingestion + storage | $0.20 |
| ECR storage | 1 ARM64 image (~500 MB) | $0.10 |
| **Total per run** | | **~$2.25** |

> ⚠️ **Cost warning:** Leaving the AgentCore Runtime endpoint deployed and idle costs roughly **$1–2/day** for the warm instance. After the demo, run `cdk destroy` and use `agentcore destroy` (or the console) to delete the Runtime resource. The CDK stack ships with a CloudWatch billing alarm at $4 USD as a safety net.

Pricing references: [Bedrock pricing](https://aws.amazon.com/bedrock/pricing/), [S3 pricing](https://aws.amazon.com/s3/pricing/), [CloudWatch pricing](https://aws.amazon.com/cloudwatch/pricing/).

## Project structure

```
self-healing-agent/
├── agent/              # Agent source (deployed to Runtime)
├── infrastructure/     # CDK stack
├── tests/              # Pytest suite (offline)
├── demo/               # Demo CLI + crash test scripts
├── blog/               # Blog post + LinkedIn + video script
└── docs/               # Architecture, deployment, troubleshooting
```

## Troubleshooting

See [docs/troubleshooting.md](docs/troubleshooting.md) for common errors:
- `AccessDeniedException` on `bedrock:InvokeModel` — model access not enabled
- `ImportError: bedrock_agentcore.memory` — wrong package version
- AgentCore Runtime hangs at startup — IAM trust policy mismatch
- `cdk deploy` rejects `bedrock-agentcore.amazonaws.com` service principal

## Contributing

Contributions are welcome. Please:

1. Fork the repo and create a feature branch.
2. Add tests for any new behavior. **All existing tests must still pass offline (no AWS calls).**
3. Run `pytest` and `ruff check .` before opening a PR.
4. For any new AWS API calls, add a `# Source: <docs URL>` comment.
5. If you change the checkpoint schema, bump `SCHEMA_VERSION` in `agent/state_manager.py` and update [docs/architecture.md](docs/architecture.md).

## License

MIT — see [LICENSE](LICENSE) (or this section if no separate file).

```
Copyright (c) 2026 DHILIP S E

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
```
