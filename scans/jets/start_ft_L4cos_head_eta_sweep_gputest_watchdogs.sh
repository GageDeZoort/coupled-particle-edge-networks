#!/bin/bash
# Launch (or no-op if already running) the L4cos→TopTagging head FT η₀ sweep watchdog.
set -euo pipefail
REPO=/home/jdezoort/coupled-particle-edge-networks
CACHE="${HOME}/.cache/tt_ft_L4cos_watchdogs"
mkdir -p "$CACHE"
LOG="$CACHE/nohup_ft_L4cos.out"
rm -f "$CACHE/STOP"
if pgrep -f 'ft_L4cos_head_eta_sweep_gputest_watchdog.sh' >/dev/null; then
  echo "FT L4cos watchdog already running"
  pgrep -af 'ft_L4cos_head_eta_sweep_gputest_watchdog.sh' | head -3
  exit 0
fi
nohup bash "$REPO/scans/jets/ft_L4cos_head_eta_sweep_gputest_watchdog.sh" \
  >>"$LOG" 2>&1 &
echo "started pid=$! log=$LOG"
