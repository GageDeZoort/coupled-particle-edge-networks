#!/usr/bin/env bash
# Keep edge-bias-m11-eta-D256-k7-eb1-live.canvas.tsx fresh while arm E rolls.
#
#   nohup bash scans/jets/watch_edge_bias_m11_eta_D256_k7_canvas.sh >/tmp/watch_eb1_canvas.log 2>&1 &
#   touch ~/.cache/jc_eb1_canvas_watch/STOP
#
set -u
PY=/home/jdezoort/.conda/envs/mamba-env/bin/python
REPO=/home/jdezoort/coupled-particle-edge-networks
SCRIPT="$REPO/scans/jets/refresh_edge_bias_m11_eta_D256_k7_canvas.py"
CACHE="${HOME}/.cache/jc_eb1_canvas_watch"
LOG=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/canvas_eb1_refresh_watch.log
INTERVAL="${INTERVAL:-45}"
mkdir -p "$CACHE" "$(dirname "$LOG")"
echo "$(date -Is) eb1 canvas watch start interval=${INTERVAL}s" >>"$LOG"
while true; do
  if [[ -f "$CACHE/STOP" ]]; then
    echo "$(date -Is) STOP $CACHE/STOP; exiting" >>"$LOG"
    exit 0
  fi
  if ! (cd "$REPO" && PYTHONPATH=src "$PY" "$SCRIPT" >>"$LOG" 2>&1); then
    echo "$(date -Is) refresh FAILED" >>"$LOG"
  fi
  sleep "$INTERVAL"
done
