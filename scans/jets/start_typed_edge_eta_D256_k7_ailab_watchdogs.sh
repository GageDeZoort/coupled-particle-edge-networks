#!/usr/bin/env bash
# Start typed-edge §1 η₀ ailab qos=gpu-test watchdog (does not use Della gputest).
#
#   bash scans/jets/start_typed_edge_eta_D256_k7_ailab_watchdogs.sh
# Stop:
#   touch ~/.cache/jc_typed_k7_ailab_watchdogs/STOP
#
set -euo pipefail
cd /home/jdezoort/coupled-particle-edge-networks
WD=scans/jets/typed_edge_eta_D256_k7_ailab_watchdog.sh
NOHUP_DIR="${NOHUP_DIR:-$HOME/.cache/jc_typed_k7_ailab_watchdogs}"
mkdir -p "$NOHUP_DIR"
rm -f "$NOHUP_DIR/STOP" /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/STOP_typed_k7_ailab

SELF=$$
mapfile -t _pids < <(pgrep -f 'scans/jets/typed_edge_eta_D256_k7_ailab_watchdog.sh' || true)
for p in "${_pids[@]:-}"; do
  [[ -z "$p" || "$p" == "$SELF" ]] && continue
  kill "$p" 2>/dev/null || true
done
sleep 1

TARGET="${TARGET:-40000}"
MAX_AILAB="${MAX_AILAB:-3}"
SLEEP_SEC="${SLEEP_SEC:-90}"

nohup env TARGET="$TARGET" MAX_STEPS="$TARGET" MAX_AILAB="$MAX_AILAB" SLEEP_SEC="$SLEEP_SEC" \
  bash "$WD" >>"$NOHUP_DIR/nohup_typed_k7_ailab.out" 2>&1 &
echo "started typed-k7-ailab watchdog pid=$! nohup=$NOHUP_DIR/nohup_typed_k7_ailab.out TARGET=$TARGET MAX_AILAB=$MAX_AILAB"

sleep 3
echo '=== typed-k7-ailab watchdog procs ==='
ps -u "$USER" -o pid,etime,cmd | grep -F 'typed_edge_eta_D256_k7_ailab_watchdog' | grep -v grep || true
echo '=== nohup tail ==='
tail -20 "$NOHUP_DIR/nohup_typed_k7_ailab.out" 2>/dev/null || true
