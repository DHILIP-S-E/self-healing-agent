# Deployment Guide

End-to-end deployment, from a fresh laptop to a running AgentCore Runtime endpoint in `ap-south-1`.

Estimated time: **30–45 minutes** (mostly waiting for CDK and the container build).

## Prerequisites checklist

Run these checks before starting. Each must pass.

```bash
python --version          # 3.12.x
aws --version             # 2.15+
cdk --version             # 2.150+
docker --version          # any recent version
docker info               # must say "Server" — daemon is running
```

If any fail, install before continuing. Docker Desktop on Windows: [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop).

## Step 1: Configure AWS

```bash
aws configure
```

Enter:
- **AWS Access Key ID:** from the IAM user you created
- **AWS Secret Access Key:** matching secret
- **Default region:** `ap-south-1`
- **Default output format:** `json`

Verify:

```bash
aws sts get-caller-identity
# Should print your account ID and IAM user/role ARN.
```

## Step 2: Enable Bedrock model access

This is a console-only step. The CLI cannot do it.

1. Open the [Bedrock console](https://console.aws.amazon.com/bedrock/) in `ap-south-1`.
2. Left nav → **Model access**.
3. Click **Manage model access**.
4. Check the box next to **Anthropic – Claude 3.5 Haiku**.
5. Submit. Approval is usually instant but can take ~10 minutes.

Verify access from the CLI:

```bash
aws bedrock list-foundation-models --region ap-south-1 \
  --query "modelSummaries[?contains(modelId, 'haiku')].modelId"
```

Should list at least one Haiku model.

## Step 3: Install project dependencies

```bash
cd self-healing-agent
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Verify:

```bash
pytest -q                          # all tests pass offline
python agent/state_manager.py     # smoke test should print "All smoke tests passed."
```

## Step 4: Bootstrap CDK (one-time per account/region)

```bash
cd infrastructure
cdk bootstrap aws://$(aws sts get-caller-identity --query Account --output text)/ap-south-1
```

This creates the CDK staging bucket. Skip this step if you have bootstrapped this account+region before.

## Step 5: Deploy the supporting infrastructure

```bash
# Still in infrastructure/
cdk synth                         # sanity check — should not error
cdk deploy --require-approval never
```

Wait ~3–5 minutes. The output prints CfnOutputs:

```
Outputs:
SelfHealingAgentStack.InputBucketName  = selfhealingagentstack-inputbucket-abc123...
SelfHealingAgentStack.OutputBucketName = selfhealingagentstack-outputbucket-abc123...
SelfHealingAgentStack.StateBucketName  = selfhealingagentstack-statebucket-abc123...
SelfHealingAgentStack.AgentRoleArn     = arn:aws:iam::123456789012:role/...
SelfHealingAgentStack.LogGroupName     = /aws/bedrock-agentcore/SelfHealingAgentStack
SelfHealingAgentStack.BillingAlarmName = SelfHealingAgentStack-BillingAlarm-xyz
```

Copy `InputBucketName`, `OutputBucketName`, and `StateBucketName` into your `.env` file.

## Step 6: Deploy the agent to AgentCore Runtime

This step uses the `agentcore` CLI (from `bedrock-agentcore-starter-toolkit`), not CDK. The toolkit handles the ARM64 container build, ECR push, and Runtime registration.

```bash
cd ..   # back to project root
agentcore configure
```

The interactive prompts ask for:
- **Application file:** `agent/main.py`
- **Region:** `ap-south-1`
- **Execution role ARN:** the `AgentRoleArn` from Step 5
- **Build platform:** `linux/arm64`

Once configured:

```bash
agentcore launch
```

Wait ~5–10 minutes. The toolkit:
1. Builds an ARM64 container image (uses your local Docker daemon).
2. Pushes to ECR.
3. Creates the AgentCore Runtime resource.
4. Prints the Runtime ARN.

Copy the Runtime ARN into `.env` as `AGENT_RUNTIME_ARN`.

## Step 7: (Optional) Create an AgentCore Memory resource

For checkpoints to use Memory as the primary backend, create a Memory resource:

```bash
aws bedrock-agentcore create-memory \
  --name self-healing-agent-state \
  --region ap-south-1
```

Copy the returned `memoryId` into `.env` as `AGENT_MEMORY_ID`.

> If you skip this step, the state manager falls back to S3 automatically (using `STATE_BUCKET`). The agent still works; it just uses one fewer storage tier.

## Step 8: Seed demo data

```bash
# Load .env into your shell
source <(grep -v '^#' .env | sed 's/^/export /' | sed 's/=$/="" /')

# Or just export manually:
export INPUT_BUCKET=selfhealingagentstack-inputbucket-abc123
export OUTPUT_BUCKET=selfhealingagentstack-outputbucket-abc123
export STATE_BUCKET=selfhealingagentstack-statebucket-abc123

# Seed 100 arXiv papers
python -m demo.seed_documents --bucket $INPUT_BUCKET --count 100
```

Wait ~2–3 minutes. Verify upload:

```bash
aws s3 ls s3://$INPUT_BUCKET/papers/ | wc -l       # should be ~100
```

## Step 9: Run the agent

### Option A — Run locally (good for iterating)

```bash
python -m demo.run --task task-001 --bucket $INPUT_BUCKET --output-bucket $OUTPUT_BUCKET
```

The agent processes documents in your local Python process, calling Bedrock and S3 directly. Good for development and the crash-test demo.

### Option B — Invoke the deployed AgentCore Runtime

```bash
agentcore invoke '{
  "task_id": "task-001",
  "input_bucket": "'"$INPUT_BUCKET"'",
  "input_prefix": "papers/",
  "output_bucket": "'"$OUTPUT_BUCKET"'",
  "resume": true
}'
```

This runs the agent inside the AgentCore Runtime container in AWS. Logs stream to CloudWatch (use the `LogGroupName` from Step 5).

## Step 10: Run the crash test

```bash
# Bash:
INPUT_BUCKET=$INPUT_BUCKET OUTPUT_BUCKET=$OUTPUT_BUCKET ./demo/crash_test.sh

# PowerShell:
$env:INPUT_BUCKET="..."; $env:OUTPUT_BUCKET="..."; ./demo/crash_test.ps1
```

Expected output:
1. First run starts, prints progress lines
2. After 30 seconds, the script SIGKILLs the process
3. Second run starts with `--resume`
4. The second run reports `Resumed: skipped N already-completed documents`

## Cleanup — stop paying

When you're done, **delete everything** so the warm AgentCore Runtime instance does not keep billing.

```bash
# 1. Stop the AgentCore Runtime
agentcore destroy

# 2. (If you created Memory) delete it
aws bedrock-agentcore delete-memory --memory-id $AGENT_MEMORY_ID --region ap-south-1

# 3. Tear down the CDK stack (buckets, IAM, log group, alarm)
cd infrastructure
cdk destroy --force
```

Verify nothing is left:

```bash
# Should return zero matching resources
aws s3 ls | grep selfhealing
aws bedrock-agentcore list-memories --region ap-south-1
```

## Troubleshooting

If anything in this guide fails, see [troubleshooting.md](troubleshooting.md).
