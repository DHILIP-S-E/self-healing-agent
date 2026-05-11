# crash_test.ps1 — Windows PowerShell equivalent of crash_test.sh.
#
# Required env: $env:INPUT_BUCKET, $env:OUTPUT_BUCKET
# Optional env: $env:TASK_ID, $env:KILL_AFTER (seconds, default 30)

$ErrorActionPreference = "Stop"

if (-not $env:INPUT_BUCKET)  { throw "INPUT_BUCKET env var is required" }
if (-not $env:OUTPUT_BUCKET) { throw "OUTPUT_BUCKET env var is required" }

$TaskId = if ($env:TASK_ID) { $env:TASK_ID } else { "task-crashtest-" + [int][double]::Parse((Get-Date -UFormat %s)) }
$KillAfter = if ($env:KILL_AFTER) { [int]$env:KILL_AFTER } else { 30 }

Write-Host "=========================================="
Write-Host "  Crash test"
Write-Host "  task_id     = $TaskId"
Write-Host "  kill_after  = ${KillAfter}s"
Write-Host "  input       = s3://$($env:INPUT_BUCKET)/papers/"
Write-Host "  output      = s3://$($env:OUTPUT_BUCKET)/summaries/$TaskId/"
Write-Host "=========================================="

# Run from the project root.
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $projectRoot

Write-Host ""
Write-Host "[1/2] Starting agent (will be killed after ${KillAfter}s)..."

$proc = Start-Process -FilePath python `
    -ArgumentList "-m","demo.run","--task",$TaskId,"--bucket",$env:INPUT_BUCKET,"--output-bucket",$env:OUTPUT_BUCKET `
    -PassThru -NoNewWindow

# Wait for the kill window or for the agent to finish on its own.
$elapsed = 0
while ((-not $proc.HasExited) -and ($elapsed -lt $KillAfter)) {
    Start-Sleep -Seconds 1
    $elapsed++
}

if ($proc.HasExited) {
    Write-Host ""
    Write-Host ">>> Agent finished on its own in ${elapsed}s — no kill needed."
    Write-Host "    (Try a smaller --count for the seed, or a larger KILL_AFTER.)"
    exit 0
}

Write-Host ""
Write-Host ">>> Killing PID $($proc.Id) after ${elapsed}s (simulated crash)"
Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
$proc.WaitForExit()

Write-Host ""
Write-Host "[2/2] Resuming agent with --resume ..."
& python -m demo.run --task $TaskId --bucket $env:INPUT_BUCKET --output-bucket $env:OUTPUT_BUCKET --resume

Write-Host ""
Write-Host "Crash test complete."
Write-Host "If everything worked, the second run reported 'Resumed: skipped N already-completed documents'."
