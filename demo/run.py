"""
Demo CLI — runs the self-healing agent against an S3 bucket of PDFs.

Two modes:
  --mode loop     Deterministic for-loop (fast, predictable)
  --mode strands  LLM reasoning via Strands (true agent — decides what to do)

Usage:
    # Deterministic loop (default)
    python -m demo.run --task task-001 --bucket my-input --output-bucket my-out

    # Real LLM reasoning agent
    python -m demo.run --task task-001 --bucket my-input --output-bucket my-out --mode strands

    # Resume after a crash (works with both modes — reads S3 checkpoint)
    python -m demo.run --task task-001 --bucket my-input --output-bucket my-out --resume

    # Auto-restart watchdog (truly self-healing — no human needed)
    python -m demo.watchdog --task task-001 --bucket my-input --output-bucket my-out
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time

# Force UTF-8 stdout/stderr on Windows so Greek/math chars in PDF text
# don't cause UnicodeEncodeError when Strands formats them into prompts.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from agent.main import run_task

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("demo.run")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the self-healing document agent.")
    p.add_argument("--task", default="task-001",
                   help="Task ID — controls the checkpoint key. Use the same value to resume.")
    p.add_argument("--bucket", default=os.environ.get("INPUT_BUCKET"),
                   help="S3 bucket holding input PDFs. Defaults to INPUT_BUCKET env.")
    p.add_argument("--prefix", default="papers/",
                   help="S3 key prefix for input PDFs (default: papers/).")
    p.add_argument("--output-bucket", default=os.environ.get("OUTPUT_BUCKET"),
                   help="S3 bucket for summaries. Defaults to OUTPUT_BUCKET env.")

    p.add_argument("--mode", choices=["loop", "strands"], default="loop",
                   help="'loop' = deterministic for-loop, 'strands' = LLM reasoning agent.")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--resume", dest="resume", action="store_true", default=True,
                     help="Resume from any prior checkpoint (default).")
    grp.add_argument("--no-resume", dest="resume", action="store_false",
                     help="Wipe prior checkpoint and start fresh.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.bucket:
        print("ERROR: provide --bucket or set INPUT_BUCKET env var", file=sys.stderr)
        return 2
    if not args.output_bucket:
        print("ERROR: provide --output-bucket or set OUTPUT_BUCKET env var", file=sys.stderr)
        return 2

    payload = {
        "task_id":       args.task,
        "input_bucket":  args.bucket,
        "input_prefix":  args.prefix,
        "output_bucket": args.output_bucket,
        "resume":        args.resume,
    }

    print("=" * 60)
    print(f"  Task ID:       {args.task}")
    print(f"  Mode:          {args.mode}")
    print(f"  Input:         s3://{args.bucket}/{args.prefix}")
    print(f"  Output:        s3://{args.output_bucket}/summaries/{args.task}/")
    print(f"  Resume:        {args.resume}")
    print("=" * 60, flush=True)

    t0 = time.time()
    try:
        if args.mode == "strands":
            from agent.strands_agent import run_strands_task
            result = run_strands_task(
                task_id=args.task,
                input_bucket=args.bucket,
                output_bucket=args.output_bucket,
            )
        else:
            result = asyncio.run(run_task(payload))
    except KeyboardInterrupt:
        elapsed = time.time() - t0
        print()
        print(f"Interrupted after {elapsed:.1f}s - checkpoint saved.")
        print(f"   Run again with --task {args.task} to resume.")
        return 130
    except Exception as e:
        elapsed = time.time() - t0
        print()
        print(f"CRASHED after {elapsed:.1f}s: {type(e).__name__}: {e}")
        print(f"   Checkpoint saved. Re-run with --task {args.task} to resume.")
        return 1

    elapsed = time.time() - t0
    print()
    print(f"Done in {elapsed:.1f}s")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result.get("skipped_resume", 0) > 0:
        print(f"\nResumed: skipped {result['skipped_resume']} already-completed documents.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
