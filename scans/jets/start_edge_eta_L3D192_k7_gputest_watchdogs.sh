#!/usr/bin/env bash
# Start the edge η₀ @ L3/D192 k★=7 R★=0.1 gpu-test pool watchdog.
#
#   bash scans/jets/start_edge_eta_L3D192_k7_gputest_watchdogs.sh
# Stop:
#   touch ~/.cache/jc_edge_L3D192_watchdogs/STOP
#
set -euo pipefail
cd /home/jdezoort/coupled-particle-edge-networks
WD=scans/jets/edge_eta_L3D192_k7_gputest_watchdog.sh
NOHUP_DIR="${NOHUP_DIR:-$HOME/.cache/jc_edge_L3D192_watchdogs}"
mkdir -p "$NOHUP_DIR"
rm -f "$NOHUP_DIR/STOP" /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/STOP_edge_L3D192

SELF=$$
mapfile -t _pids < <(pgrep -f 'scans/jets/edge_eta_L3D192_k7_gputest_watchdog.sh' || true)
for p in "${_pids[@]:-}"; do
  [[ -z "$p" || "$p" == "$SELF" ]] && continue
  kill "$p" 2>/dev/null || true
done
sleep 1

TARGET="${TARGET:-40000}"
MAX_GPTEST="${MAX_GPTEST:-3}"
SLEEP_SEC="${SLEEP_SEC:-60}"

nohup env TARGET="$TARGET" MAX_STEPS="$TARGET" MAX_GPTEST="$MAX_GPTEST" SLEEP_SEC="$SLEEP_SEC" \
  bash "$WD" >>"$NOHUP_DIR/nohup_edge_L3D192.out" 2>&1 &
echo "started edge-L3D192 watchdog pid=$! nohup=$NOHUP_DIR/nohup_edge_L3D192.out TARGET=$TARGET MAX_GPTEST=$MAX_GPTEST"

sleep 3
echo '=== edge-L3D192 watchdog procs ==='
ps -u "$USER" -o pid,etime,cmd | grep -F 'edge_eta_L3D192_k7_gputest_watchdog' | grep -v grep || true
echo '=== nohup tail ==='
tail -20 "$NOHUP_DIR/nohup_edge_L3D192.out" 2>/dev/null || true
