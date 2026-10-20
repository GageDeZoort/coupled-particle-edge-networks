#!/bin/bash
# Launch L6/D512 flat-LR epoch-1 gputest watchdog.
set -euo pipefail
REPO=/home/jdezoort/coupled-particle-edge-networks
CACHE="${HOME}/.cache/jc_pretrain_L6_watchdogs"
mkdir -p "$CACHE"
LOG="$CACHE/nohup_pretrain_L6.out"
rm -f "$CACHE/STOP"
if pgrep -f 'pretrain_L6_epoch_gputest_watchdog.sh' >/dev/null; then
  echo "L6 pretrain watchdog already running"
  pgrep -af 'pretrain_L6_epoch_gputest_watchdog.sh' | head -3
  exit 0
fi
nohup bash "$REPO/scans/jets/pretrain_L6_epoch_gputest_watchdog.sh" \
  >>"$LOG" 2>&1 &
echo "started pid=$! log=$LOG"
