#!/bin/bash
# Launch (or no-op if already running) the L8/D1024 Della gputest continue watchdog.
set -euo pipefail
REPO=/home/jdezoort/coupled-particle-edge-networks
CACHE="${HOME}/.cache/jc_pretrain_L8_watchdogs"
mkdir -p "$CACHE"
LOG="$CACHE/nohup_pretrain_L8.out"
SEED=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_pretrain/lambda_L8D1024/checkpoints/last.ckpt
if [[ ! -f "$SEED" ]]; then
  echo "WARNING: mirrored Lambda ckpt not ready yet: $SEED"
  echo "Watchdog will wait until rsync finishes."
fi
if pgrep -f 'pretrain_L8_continue_gputest_watchdog.sh' >/dev/null; then
  echo "L8 watchdog already running"
  pgrep -af 'pretrain_L8_continue_gputest_watchdog.sh' | head -3
  exit 0
fi
nohup bash "$REPO/scans/jets/pretrain_L8_continue_gputest_watchdog.sh" \
  >>"$LOG" 2>&1 &
echo "started pid=$! log=$LOG"
