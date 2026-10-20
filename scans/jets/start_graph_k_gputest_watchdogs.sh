#!/bin/bash
# Start the Plot-1 kNN×shard×seed gpu-test pool watchdog.
#
#   bash scans/jets/start_graph_k_gputest_watchdogs.sh
#   MAX_CONCURRENT=3 TARGET=950 bash scans/jets/start_graph_k_gputest_watchdogs.sh
#
set -euo pipefail

ROOT=/home/jdezoort/coupled-particle-edge-networks
WD="$ROOT/scans/jets/graph_k_gputest_watchdog.sh"
CACHE="${HOME}/.cache/jc_gk_watchdogs"
mkdir -p "$CACHE"

TARGET="${TARGET:-950}"
MAX_STEPS="${MAX_STEPS:-1000}"
MAX_CONCURRENT="${MAX_CONCURRENT:-3}"
SLEEP_SEC="${SLEEP_SEC:-30}"

rm -f "$CACHE/STOP" "$CACHE/GK_COMPLETE" \
  /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/STOP_gk \
  2>/dev/null || true

SELF=$$
while read -r pid; do
  [[ -z "$pid" || "$pid" == "$SELF" ]] && continue
  cmd=$(tr '\0' ' ' < /proc/"$pid"/cmdline 2>/dev/null || true)
  [[ "$cmd" != *'/graph_k_gputest_watchdog.sh'* ]] && continue
  kill "$pid" 2>/dev/null || true
done < <(pgrep -f graph_k_gputest_watchdog.sh || true)
sleep 1
rm -f "$CACHE/watchdog_gk.lock" 2>/dev/null || true

LOG="$CACHE/watchdog_gk.log"
nohup env TARGET="$TARGET" MAX_STEPS="$MAX_STEPS" \
  MAX_CONCURRENT="$MAX_CONCURRENT" SLEEP_SEC="$SLEEP_SEC" \
  bash "$WD" >>"$CACHE/watchdog_gk.nohup" 2>&1 &
echo "started gk watchdog pid=$! TARGET=$TARGET MAX_STEPS=$MAX_STEPS MAX_CONCURRENT=$MAX_CONCURRENT log=$LOG"
