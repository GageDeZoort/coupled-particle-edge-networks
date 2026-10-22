#!/usr/bin/env bash
# Prioritize filling edge-bias-m11-eta-D256-k7-eb1-live.canvas.tsx on Della gputest.
#
#   bash scans/jets/start_edge_bias_m11_eb1_D256_gputest_watchdogs.sh
# Stop:
#   touch ~/.cache/jc_eb1E_gt_watchdogs/STOP
#
set -euo pipefail
cd /home/jdezoort/coupled-particle-edge-networks
WD=scans/jets/edge_bias_m11_eb1_D256_gputest_watchdog.sh
NOHUP_DIR="${NOHUP_DIR:-$HOME/.cache/jc_eb1E_gt_watchdogs}"
mkdir -p "$NOHUP_DIR"
rm -f "$NOHUP_DIR/STOP" "$NOHUP_DIR/watchdog_eb1E_gt.lock" \
  /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/STOP_eb1E_gt

mapfile -t _pids < <(pgrep -f 'scans/jets/edge_bias_m11_eb1_D256_gputest_watchdog.sh' || true)
for p in "${_pids[@]:-}"; do
  [[ -z "$p" ]] && continue
  kill "$p" 2>/dev/null || true
done
sleep 1

TARGET="${TARGET:-40000}"
MAX_GPUS="${MAX_GPUS:-3}"
SLEEP_SEC="${SLEEP_SEC:-90}"

nohup env TARGET="$TARGET" MAX_STEPS="$TARGET" MAX_GPUS="$MAX_GPUS" SLEEP_SEC="$SLEEP_SEC" \
  bash "$WD" >>"$NOHUP_DIR/nohup_eb1E_gt.out" 2>&1 &
echo "started eb1E-gt watchdog pid=$! nohup=$NOHUP_DIR/nohup_eb1E_gt.out TARGET=$TARGET MAX_GPUS=$MAX_GPUS"
sleep 2
tail -15 "$NOHUP_DIR/nohup_eb1E_gt.out" 2>/dev/null || true
