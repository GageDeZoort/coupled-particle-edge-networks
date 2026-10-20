#!/bin/bash
# Start the M11-only η₀ × D gpu-test pool watchdog.
#
#   bash scans/jets/start_m11_eta_gputest_watchdogs.sh
#   MAX_CONCURRENT=2 TARGET=2450 bash scans/jets/start_m11_eta_gputest_watchdogs.sh
#
set -euo pipefail

ROOT=/home/jdezoort/coupled-particle-edge-networks
WD="$ROOT/scans/jets/m11_eta_gputest_watchdog.sh"
CACHE="${HOME}/.cache/jc_m11_watchdogs"
mkdir -p "$CACHE"

TARGET="${TARGET:-3950}"
MAX_STEPS="${MAX_STEPS:-4000}"
EPOCHS="${EPOCHS:-2}"
MAX_CONCURRENT="${MAX_CONCURRENT:-3}"
SLEEP_SEC="${SLEEP_SEC:-30}"

rm -f "$CACHE/STOP" "$CACHE/M11_COMPLETE" \
  /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/STOP_m11 \
  2>/dev/null || true

# Kill existing m11 watchdogs by PID (avoid pkill -f matching this shell).
SELF=$$
while read -r pid; do
  [[ -z "$pid" || "$pid" == "$SELF" ]] && continue
  cmd=$(tr '\0' ' ' < /proc/"$pid"/cmdline 2>/dev/null || true)
  [[ "$cmd" != *'/m11_eta_gputest_watchdog.sh'* ]] && continue
  [[ "$cmd" == *extglob* ]] && continue
  kill "$pid" 2>/dev/null || true
done < <(pgrep -f m11_eta_gputest_watchdog.sh || true)
sleep 1
rm -f "$CACHE/watchdog_m11.lock" 2>/dev/null || true

LOG="$CACHE/watchdog_m11.log"
# Watchdog already tee's to $LOG; keep nohup.out for crash traces only.
nohup env TARGET="$TARGET" MAX_STEPS="$MAX_STEPS" EPOCHS="$EPOCHS" \
  MAX_CONCURRENT="$MAX_CONCURRENT" SLEEP_SEC="$SLEEP_SEC" \
  bash "$WD" >>"$CACHE/watchdog_m11.nohup" 2>&1 &
echo "started m11 watchdog pid=$! TARGET=$TARGET MAX_STEPS=$MAX_STEPS EPOCHS=$EPOCHS MAX_CONCURRENT=$MAX_CONCURRENT log=$LOG"
