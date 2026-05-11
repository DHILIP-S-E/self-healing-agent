# Troubleshooting

Common errors, root causes, and fixes.

## Setup errors

### `pip install` fails with `bedrock-agentcore not found`

**Cause:** The package was published in March 2026; old PyPI mirrors may not have it.

**Fix:**

```bash
pip install --index-url https://pypi.org/simple/ bedrock-agentcore bedrock-agentcore-starter-toolkit
```

If still failing, check the package exists at [pypi.org/project/bedrock-agentcore/](https://pypi.org/project/bedrock-agentcore/) and your Python is ≥ 3.10.

### `python --version` says 3.9 or older

**Cause:** Strands and bedrock-agentcore both require ≥ 3.10.

**Fix:** Install Python 3.12 from [python.org/downloads](https://www.python.org/downloads/). On Windows, also re-create your venv with the new interpreter.

### `docker info` shows "Cannot connect to the Docker daemon"

**Cause:** Docker Desktop is not running.

**Fix:** Start Docker Desktop. Wait for the whale icon to stop animating before retrying.

## AWS errors

### `AccessDeniedException` on `bedrock:InvokeModel`

**Cause:** Bedrock model access for Claude Haiku is not enabled in your account.

**Fix:**
1. [Bedrock console](https://console.aws.amazon.com/bedrock/) → Model access → Manage model access.
2. Check Claude 3.5 Haiku.
3. Submit. Wait up to 10 minutes for approval.

Verify with:

```bash
aws bedrock list-foundation-models --region ap-south-1 \
  --query "modelSummaries[?contains(modelId, 'haiku')].[modelId,modelLifecycle.status]"
```

The `modelLifecycle.status` must be `ACTIVE`.

### `cdk deploy` rejects service principal `bedrock-agentcore.amazonaws.com`

**Cause:** The exact AgentCore service principal name may differ from what the stack uses.

**Fix:** Open `infrastructure/stack.py`, find the `iam.ServicePrincipal(...)` line in `AgentExecutionRole`, and change it to whatever the AWS docs say. Try in this order:

```python
iam.ServicePrincipal("bedrock-agentcore.amazonaws.com")        # try first
iam.ServicePrincipal("agentcore.bedrock.amazonaws.com")        # alternate form
iam.ServicePrincipal("bedrock.amazonaws.com")                  # fallback
```

You can also confirm by running:

```bash
aws iam list-roles --query "Roles[?contains(AssumeRolePolicyDocument, 'agentcore')].AssumeRolePolicyDocument" \
  --output text
```

…on an AWS-managed role created by the AgentCore service.

### `cdk bootstrap` fails with "no matching CDK toolkit"

**Cause:** Stale CDK CLI vs. library version mismatch.

**Fix:**

```bash
npm install -g aws-cdk@latest
pip install --upgrade aws-cdk-lib constructs
cdk --version
cdk bootstrap
```

### Billing alarm not firing

**Cause:** AWS/Billing metrics are emitted only in `us-east-1`. The alarm was created in `ap-south-1` but uses a `us-east-1` metric, which is correct — but billing must also be enabled to publish to CloudWatch in your account.

**Fix:**
1. [Billing console](https://console.aws.amazon.com/billing/home) → Billing preferences.
2. Enable **Receive Billing Alerts**. (This is account-wide and one-time.)
3. Wait up to 24 hours for the first metric data point.

## Runtime errors

### `ImportError: cannot import name 'MemoryClient' from 'bedrock_agentcore.memory'`

**Cause:** Your installed `bedrock-agentcore` version exports `MemoryClient` from a different submodule.

**Fix:** Run

```bash
python -c "import bedrock_agentcore; print(dir(bedrock_agentcore))"
python -c "import bedrock_agentcore.memory as m; print(dir(m))"
```

…and update the import in `agent/state_manager.py` to match. The class is most likely named `MemoryClient` or `BedrockAgentCoreMemory`.

### AgentCore Runtime hangs at startup

**Cause:** IAM trust policy mismatch between the role CDK created and what AgentCore Runtime expects.

**Fix:**

```bash
aws iam get-role --role-name <AgentRoleArn-suffix> \
  --query "Role.AssumeRolePolicyDocument"
```

The trust policy must allow `bedrock-agentcore.amazonaws.com` (or whatever the correct principal is — see "cdk deploy rejects service principal" above) to assume the role.

If the role is correct, check CloudWatch Logs at `/aws/bedrock-agentcore/SelfHealingAgentStack` for the actual startup error.

### `read_document: failed to parse PDF`

**Cause:** The PDF is malformed, encrypted, or scanned-image-only (no text layer).

**Fix:** The agent skips no documents — every error stops the run. Options:
1. Manually remove the bad PDF from S3 and re-run.
2. Wrap `read_document` in a try/except that logs and returns `""` for unparseable PDFs (would require a code change in `agent/main.py`).

To find the bad PDF:

```bash
# The error message includes the S3 key. Or check CloudWatch logs.
grep "failed to parse PDF" agent.log
```

### Bedrock throttling: `ThrottlingException`

**Cause:** Hit the per-minute token limit for Claude Haiku in your account.

**Fix:** Either wait a minute and re-run with `--resume`, or [request a quota increase](https://console.aws.amazon.com/servicequotas/home/services/bedrock/quotas) for Bedrock InvokeModel calls per minute.

The agent automatically resumes from the last good checkpoint, so a few-second pause is harmless.

## Test errors

### `pytest tests/test_state_manager.py` fails with `ModuleNotFoundError: No module named 'agent'`

**Cause:** Tests run from the project root but Python doesn't know about the local package.

**Fix:**

```bash
pip install -e .          # editable install — picks up pyproject.toml
pytest -v
```

### `moto` tests fail with `endpoint_url not specified`

**Cause:** Old `moto` version. `moto` 5.x unified the API.

**Fix:**

```bash
pip install --upgrade "moto>=5.0"
```

### Test asks for AWS credentials when it shouldn't

**Cause:** A test is hitting real AWS instead of moto. Likely a misplaced fixture.

**Fix:** Run with no creds in scope to find the leak:

```bash
AWS_ACCESS_KEY_ID=fake AWS_SECRET_ACCESS_KEY=fake pytest -v
```

The failing test will tell you which AWS call escaped the mock.

## Cost surprises

### Bill is climbing past $4

**The CloudWatch alarm fires but you ignored the email.**

**Fix:**

```bash
agentcore destroy                                # stop the warm Runtime
cdk destroy --force                              # tear down the rest
aws bedrock-agentcore list-memories --region ap-south-1   # confirm Memory deleted
```

The warm Runtime instance is the most likely cost — $1–2/day even with zero invocations. After `agentcore destroy`, billing for that resource stops within an hour.

### S3 storage charges accumulating

**The 30-day lifecycle rule will clean up automatically.** If you want to clean up immediately:

```bash
aws s3 rm s3://$INPUT_BUCKET --recursive
aws s3 rm s3://$OUTPUT_BUCKET --recursive
aws s3 rm s3://$STATE_BUCKET --recursive
```

Then `cdk destroy` removes the buckets themselves.

## Still stuck?

1. Check CloudWatch Logs at `/aws/bedrock-agentcore/SelfHealingAgentStack` — the actual error is almost always there.
2. Open an issue on [GitHub](https://github.com/DHILIP-S-E/self-healing-agent/issues) with: the command you ran, the full error output, your region, and the output of `pip freeze | grep -E 'bedrock|strands|cdk'`.
