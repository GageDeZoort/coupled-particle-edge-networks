#!/bin/bash
# Launch AdamW t_epoch=2 / η₀=0.05 / B=256×4 gputest watchdog.
set -euo pipefail
REPO=/home/jdezoort/coupled-particle-edge-networks
CACHE="${HOME}/.cache/tt_ft_aw_t2_watchdogs"
mkdir -p "$CACHE"
LOG="$CACHE/nohup_ft_aw_t2.out"
rm -f "$CACHE/STOP"
if pgrep -f 'ft_L4cos_adamw_t2_gputest_watchdog.sh' >/dev/null; then
  echo "AdamW t2 gputest watchdog already running"
  pgrep -af 'ft_L4cos_adamw_t2_gputest_watchdog.sh' | head -3
  exit 0
fi
nohup bash "$REPO/scans/jets/ft_L4cos_adamw_t2_gputest_watchdog.sh" \
  >>"$LOG" 2>&1 &
echo "started pid=$! log=$LOG"
