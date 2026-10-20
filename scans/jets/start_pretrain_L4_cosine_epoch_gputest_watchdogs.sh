#!/bin/bash
# Launch (or no-op if already running) the L4/D256 cosine epoch-2 gputest watchdog.
set -euo pipefail
REPO=/home/jdezoort/coupled-particle-edge-networks
CACHE="${HOME}/.cache/jc_pretrain_L4cos_watchdogs"
mkdir -p "$CACHE"
LOG="$CACHE/nohup_pretrain_L4cos.out"
if pgrep -f 'pretrain_L4_cosine_epoch_gputest_watchdog.sh' >/dev/null; then
  echo "cosine watchdog already running"
  pgrep -af 'pretrain_L4_cosine_epoch_gputest_watchdog.sh' | head -3
  exit 0
fi
nohup bash "$REPO/scans/jets/pretrain_L4_cosine_epoch_gputest_watchdog.sh" \
  >>"$LOG" 2>&1 &
echo "started pid=$! log=$LOG"
