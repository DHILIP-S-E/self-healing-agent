"""
Watchdog — automatically detects a crash and restarts the agent.

This is what makes the system TRULY self-healing:
  - Runs the agent as a subprocess
  - If it crashes (any non-zero exit), waits and restarts automatically
  - Always passes --resume so the agent picks up from its checkpoint
  - No human intervention required

Usage:
    python -m demo.watchdog --task task-001 --bucket my-pdfs --output-bucket my-summaries

    # With custom retry settings:
    python -m demo.watchdog --task task-001 --bucket my-pdfs --output-bucket my-summaries \\
        --max-retries 10 --retry-delay 10 --mode strands
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s watchdog: %(message)s",
)
log = logging.getLogger("watchdog")


def run_with_watchdog(
    task_id: str,
    bucket: str,
    output_bucket: str,
    mode: str = "loop",
    max_retries: int = 10,
    retry_delay: int = 5,
    prefix: str = "papers/",
) -> int:
    """Run the agent and restart it automatically on crash.

    Args:
        task_id: Task ID — same across restarts so checkpoint is reused.
        bucket: S3 input bucket.
        output_bucket: S3 output bucket.
        mode: 'loop' (deterministic) or 'strands' (LLM reasoning).
        max_retries: Maximum number of automatic restarts.
        retry_delay: Seconds to wait between restarts.
        prefix: S3 key prefix for input PDFs.

    Returns:
        0 on success, 1 if max retries exceeded.
    """
    attempt = 0

    while attempt <= max_retries:
        attempt += 1
        is_resume = attempt > 1

        log.info(
            "=" * 60
        )
        log.info(
            "Attempt %d/%d — task=%s mode=%s resume=%s",
            attempt, max_retries + 1, task_id, mode, is_resume,
        )
        log.info("=" * 60)

        cmd = [
            sys.executable, "-m", "demo.run",
            "--task", task_id,
            "--bucket", bucket,
            "--prefix", prefix,
            "--output-bucket", output_bucket,
            "--mode", mode,
        ]
        if is_resume:
            cmd.append("--resume")
        else:
            cmd.append("--no-resume")

        start = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - start

        if result.returncode == 0:
            log.info(
                "Agent completed successfully after %d attempt(s) (%.1fs total).",
                attempt, elapsed,
            )
            return 0

        if result.returncode == 130:
            # KeyboardInterrupt — user pressed Ctrl+C intentionally
            log.info("Interrupted by user (Ctrl+C). Stopping watchdog.")
            return 130

        # Crash — decide whether to restart
        log.warning(
            "💥 Agent crashed (exit code %d) after %.1fs on attempt %d/%d.",
            result.returncode, elapsed, attempt, max_retries + 1,
        )

        if attempt > max_retries:
            log.error(
                "Max retries (%d) exceeded. Giving up. "
                "Checkpoint is safe — run manually with --resume to continue.",
                max_retries,
            )
            return 1

        log.info(
            "🔄 Auto-restarting in %ds (attempt %d/%d will resume from checkpoint)...",
            retry_delay, attempt + 1, max_retries + 1,
        )
        time.sleep(retry_delay)

    return 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Watchdog: auto-restart the agent on crash.",
    )
    p.add_argument("--task", default="task-watchdog",
                   help="Task ID (reused across restarts for checkpoint continuity).")
    p.add_argument("--bucket", required=True, help="S3 input bucket.")
    p.add_argument("--prefix", default="papers/", help="S3 key prefix for input PDFs.")
    p.add_argument("--output-bucket", required=True, help="S3 output bucket.")
    p.add_argument("--mode", choices=["loop", "strands"], default="strands",
                   help="'loop' = deterministic, 'strands' = LLM reasoning (default: strands).")
    p.add_argument("--max-retries", type=int, default=10,
                   help="Max automatic restarts before giving up (default: 10).")
    p.add_argument("--retry-delay", type=int, default=5,
                   help="Seconds between restarts (default: 5).")
    args = p.parse_args(argv)

    return run_with_watchdog(
        task_id=args.task,
        bucket=args.bucket,
        output_bucket=args.output_bucket,
        mode=args.mode,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    sys.exit(main())
