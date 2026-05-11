"""CDK app entrypoint."""

import os

import aws_cdk as cdk

from infrastructure.stack import SelfHealingAgentStack

app = cdk.App()

SelfHealingAgentStack(
    app,
    "SelfHealingAgentStack",
    env=cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT") or os.environ.get("AWS_ACCOUNT_ID"),
        region=os.environ.get("CDK_DEFAULT_REGION") or os.environ.get("AWS_REGION", "ap-south-1"),
    ),
    description="Self-healing document processing agent — buckets, IAM, logs, billing alarm",
)

app.synth()
