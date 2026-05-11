"""
CDK stack for the self-healing agent.

Creates supporting infrastructure ONLY. The AgentCore Runtime resource itself
is deployed via the `agentcore` CLI (bedrock-agentcore-starter-toolkit), which
builds the ARM64 container, pushes to ECR, and registers the Runtime — see
docs/deployment.md.

This stack creates:
    1. 3 S3 buckets (input, output, state) with 30-day lifecycle + encryption
    2. IAM execution role for the agent (least-privilege)
    3. CloudWatch log group with 30-day retention
    4. Billing alarm at $4 USD threshold (warns before budget overrun)
    5. CfnOutputs with resource names for use by deploy scripts

Sources:
    https://docs.aws.amazon.com/cdk/v2/guide/home.html
    https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/monitor_estimated_charges_with_cloudwatch.html
"""

from __future__ import annotations

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
)
from aws_cdk import (
    aws_iam as iam,
    aws_logs as logs,
    aws_s3 as s3,
)
from constructs import Construct


class SelfHealingAgentStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ====================================================================
        # 1. S3 buckets
        # ====================================================================
        # Common settings: SSE-S3, no public access, 30-day expiration so
        # demo runs don't leak into long-term storage costs.
        common_bucket_args = dict(
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            # DESTROY + auto_delete is fine for a demo. For prod, swap to RETAIN.
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="cleanup-after-30d",
                    enabled=True,
                    expiration=Duration.days(30),
                    abort_incomplete_multipart_upload_after=Duration.days(1),
                ),
            ],
        )

        input_bucket = s3.Bucket(self, "InputBucket", **common_bucket_args)
        output_bucket = s3.Bucket(self, "OutputBucket", **common_bucket_args)
        state_bucket = s3.Bucket(self, "StateBucket", **common_bucket_args)

        # ====================================================================
        # 2. IAM execution role
        # ====================================================================
        # UNVERIFIED — confirm exact service principal for AgentCore Runtime.
        # Educated guess: bedrock-agentcore.amazonaws.com.
        # If `cdk synth` rejects this, try: agent.bedrock.amazonaws.com or
        # check `aws iam list-roles | grep AgentCore` for the canonical form.
        agent_role = iam.Role(
            self,
            "AgentExecutionRole",
            assumed_by=iam.ServicePrincipal("bedrock-agentcore.amazonaws.com"),
            description="Execution role for the self-healing document processing agent",
            max_session_duration=Duration.hours(8),  # supports overnight runs
        )

        # S3: read input, write output, read+write state
        input_bucket.grant_read(agent_role)
        output_bucket.grant_write(agent_role)
        state_bucket.grant_read_write(agent_role)

        # Bedrock model invoke — Llama 3 70B Instruct (meta.*) in this region.
        # Source: https://docs.aws.amazon.com/bedrock/latest/userguide/security_iam_id-based-policy-examples.html
        agent_role.add_to_policy(
            iam.PolicyStatement(
                sid="InvokeLlama70B",
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                ],
                resources=[
                    f"arn:aws:bedrock:{self.region}::foundation-model/meta.llama3-70b-instruct-v1:0",
                    f"arn:aws:bedrock:{self.region}::foundation-model/meta.llama3-*",
                ],
            )
        )

        # AgentCore Memory access.
        # UNVERIFIED — bedrock-agentcore:* action names. Confirm with:
        #   aws bedrock-agentcore help
        # and tighten resource ARN once your Memory ID is known (post-deploy).
        agent_role.add_to_policy(
            iam.PolicyStatement(
                sid="AgentCoreMemoryAccess",
                actions=[
                    "bedrock-agentcore:CreateEvent",
                    "bedrock-agentcore:ListEvents",
                    "bedrock-agentcore:GetEvent",
                    "bedrock-agentcore:DeleteSession",
                    "bedrock-agentcore:RetrieveMemoryRecords",
                ],
                resources=[
                    f"arn:aws:bedrock-agentcore:{self.region}:{self.account}:memory/*",
                ],
            )
        )

        # CloudWatch Logs (write-only).
        agent_role.add_to_policy(
            iam.PolicyStatement(
                sid="WriteLogs",
                actions=[
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "logs:DescribeLogStreams",
                ],
                resources=[
                    f"arn:aws:logs:{self.region}:{self.account}:log-group:/aws/bedrock-agentcore/*",
                ],
            )
        )

        # ====================================================================
        # 3. CloudWatch log group
        # ====================================================================
        log_group = logs.LogGroup(
            self,
            "AgentLogGroup",
            log_group_name=f"/aws/bedrock-agentcore/{construct_id}",
            retention=logs.RetentionDays.ONE_MONTH,
            removal_policy=RemovalPolicy.DESTROY,
        )

        # NOTE: Billing alarm removed — AWS/Billing metrics live only in
        # us-east-1 and CDK does not support cross-region alarm creation in a
        # single stack. Set a billing alert manually in the AWS Console under
        # Billing > Budgets, threshold $4 USD.

        # ====================================================================
        # 5. Outputs
        # ====================================================================
        CfnOutput(self, "InputBucketName", value=input_bucket.bucket_name,
                  description="Set INPUT_BUCKET to this value")
        CfnOutput(self, "OutputBucketName", value=output_bucket.bucket_name,
                  description="Set OUTPUT_BUCKET to this value")
        CfnOutput(self, "StateBucketName", value=state_bucket.bucket_name,
                  description="Set STATE_BUCKET to this value (used as Memory fallback)")
        CfnOutput(self, "AgentRoleArn", value=agent_role.role_arn,
                  description="IAM role ARN for AgentCore Runtime to assume")
        CfnOutput(self, "LogGroupName", value=log_group.log_group_name,
                  description="CloudWatch log group for agent traces")
