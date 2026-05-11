#!/usr/bin/env bash
# crash_test.sh — start the agent, kill it after KILL_AFTER seconds, then resume.
#
# Required env:
#   INPUT_BUCKET    Bucket holding the seeded PDFs.
#   OUTPUT_BUCKET   Bucket that will receive summaries.
#
# Optional env:
#   TASK_ID         Defaults to task-crashtest-<timestamp>.
#   KILL_AFTER      Seconds before SIGKILL. Default 30.
#
# Works in Git Bash or WSL on Windows; bash 4+ on Linux/macOS.
set -euo pipefail

TASK_ID="${TASK_ID:-task-crashtest-$(date +%s)}"
KILL_AFTER="${KILL_AFTER:-30}"

: "${INPUT_BUCKET:?INPUT_BUCKET must be set}"
: "${OUTPUT_BUCKET:?OUTPUT_BUCKET must be set}"

echo "=========================================="
echo "  Crash test"
echo "  task_id     = $TASK_ID"
echo "  kill_after  = ${KILL_AFTER}s"
echo "  input       = s3://$INPUT_BUCKET/papers/"
echo "  output      = s3://$OUTPUT_BUCKET/summaries/$TASK_ID/"
echo "=========================================="

# Run from the project root regardless of where the script is invoked.
cd "$(dirname "$0")/.."

echo
echo "[1/2] Starting agent (will be SIGKILLed after ${KILL_AFTER}s)..."
python -m demo.run \
  --task "$TASK_ID" \
  --bucket "$INPUT_BUCKET" \
  --output-bucket "$OUTPUT_BUCKET" &
PID=$!

# Wait for the kill window. If the agent finishes early, no kill needed.
SECONDS_WAITED=0
while kill -0 "$PID" 2>/dev/null && [ "$SECONDS_WAITED" -lt "$KILL_AFTER" ]; do
  sleep 1
  SECONDS_WAITED=$((SECONDS_WAITED + 1))
done

if kill -0 "$PID" 2>/dev/null; then
  echo
  echo ">>> Killing PID $PID after ${SECONDS_WAITED}s (simulated crash)"
  kill -9 "$PID" 2>/dev/null || true
  wait "$PID" 2>/dev/null || true
else
  echo
  echo ">>> Agent finished on its own in ${SECONDS_WAITED}s — no kill needed."
  echo "    (Try a smaller --count for the seed, or a larger KILL_AFTER.)"
  exit 0
fi

echo
echo "[2/2] Resuming agent with --resume ..."
python -m demo.run \
  --task "$TASK_ID" \
  --bucket "$INPUT_BUCKET" \
  --output-bucket "$OUTPUT_BUCKET" \
  --resume

echo
echo "Crash test complete."
echo "If everything worked, the second run reported 'Resumed: skipped N already-completed documents'."
