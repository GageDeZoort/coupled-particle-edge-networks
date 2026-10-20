#!/bin/bash
# Start panel-(b′) noloop R★ @ k★=7 gputest pool watchdog.
#
#   bash scans/jets/start_graph_R_noloop_gputest_watchdogs.sh
#
set -euo pipefail

ROOT=/home/jdezoort/coupled-particle-edge-networks
WD="$ROOT/scans/jets/graph_R_noloop_gputest_watchdog.sh"
CACHE="${HOME}/.cache/jc_gR_noloop_gt"
mkdir -p "$CACHE"

TARGET="${TARGET:-2495}"
MAX_STEPS="${MAX_STEPS:-2500}"
MAX_CONCURRENT="${MAX_CONCURRENT:-3}"
SLEEP_SEC="${SLEEP_SEC:-30}"
A100_JOB="${A100_JOB:-$(cat "$ROOT/scans/jets/graph_R_noloop_v2cp_jobid.txt" 2>/dev/null || true)}"

rm -f "$CACHE/STOP" "$CACHE/GR_NOLOOP_GT_COMPLETE" \
  /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/STOP_gR_noloop_gt \
  2>/dev/null || true

SELF=$$
while read -r pid; do
  [[ -z "$pid" || "$pid" == "$SELF" ]] && continue
  cmd=$(tr '\0' ' ' < /proc/"$pid"/cmdline 2>/dev/null || true)
  [[ "$cmd" != *'/graph_R_noloop_gputest_watchdog.sh'* ]] && continue
  kill "$pid" 2>/dev/null || true
done < <(pgrep -f graph_R_noloop_gputest_watchdog.sh || true)
sleep 1
rm -f "$CACHE/watchdog.lock" 2>/dev/null || true

LOG="$CACHE/watchdog.log"
nohup env TARGET="$TARGET" MAX_STEPS="$MAX_STEPS" \
  MAX_CONCURRENT="$MAX_CONCURRENT" SLEEP_SEC="$SLEEP_SEC" \
  A100_JOB="${A100_JOB:-}" \
  bash "$WD" >>"$CACHE/watchdog.nohup" 2>&1 &
echo "started gR-noloop-gt watchdog pid=$! A100_JOB=${A100_JOB:-none} TARGET=$TARGET MAX_CONCURRENT=$MAX_CONCURRENT log=$LOG"
