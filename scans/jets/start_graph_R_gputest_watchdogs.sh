#!/bin/bash
# Start the Plot-2 R★×shard×seed gpu-test pool watchdog (k★=12).
#
#   bash scans/jets/start_graph_R_gputest_watchdogs.sh
#   MAX_CONCURRENT=3 KNN_K=12 bash scans/jets/start_graph_R_gputest_watchdogs.sh
#
set -euo pipefail

ROOT=/home/jdezoort/coupled-particle-edge-networks
WD="$ROOT/scans/jets/graph_R_gputest_watchdog.sh"
CACHE="${HOME}/.cache/jc_gR_watchdogs"
mkdir -p "$CACHE"

TARGET="${TARGET:-950}"
MAX_STEPS="${MAX_STEPS:-1000}"
KNN_K="${KNN_K:-12}"
MAX_CONCURRENT="${MAX_CONCURRENT:-3}"
SLEEP_SEC="${SLEEP_SEC:-30}"

rm -f "$CACHE/STOP" "$CACHE/GR_COMPLETE" \
  /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/STOP_gR \
  2>/dev/null || true

SELF=$$
while read -r pid; do
  [[ -z "$pid" || "$pid" == "$SELF" ]] && continue
  cmd=$(tr '\0' ' ' < /proc/"$pid"/cmdline 2>/dev/null || true)
  [[ "$cmd" != *'/graph_R_gputest_watchdog.sh'* ]] && continue
  kill "$pid" 2>/dev/null || true
done < <(pgrep -f graph_R_gputest_watchdog.sh || true)
sleep 1
rm -f "$CACHE/watchdog_gR.lock" 2>/dev/null || true

LOG="$CACHE/watchdog_gR.log"
nohup env TARGET="$TARGET" MAX_STEPS="$MAX_STEPS" KNN_K="$KNN_K" \
  MAX_CONCURRENT="$MAX_CONCURRENT" SLEEP_SEC="$SLEEP_SEC" \
  bash "$WD" >>"$CACHE/watchdog_gR.nohup" 2>&1 &
echo "started gR watchdog pid=$! KNN_K=$KNN_K TARGET=$TARGET MAX_CONCURRENT=$MAX_CONCURRENT log=$LOG"
