#!/usr/bin/env bash
# Keep edge-eta-L3D192-k7-s40k-live.canvas.tsx fresh while the η₀ sweep rolls.
set -u
PY=/home/jdezoort/.conda/envs/mamba-env/bin/python
SCRIPT=/home/jdezoort/coupled-particle-edge-networks/scans/jets/refresh_edge_eta_L3D192_k7_s40k_canvas.py
LOG=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/canvas_edge_L3D192_refresh_watch.log
mkdir -p "$(dirname "$LOG")"
INTERVAL="${INTERVAL:-90}"
echo "$(date -Is) L3D192 canvas watch start interval=${INTERVAL}s" >>"$LOG"
while true; do
  if ! "$PY" "$SCRIPT" >>"$LOG" 2>&1; then
    echo "$(date -Is) refresh FAILED" >>"$LOG"
  fi
  sleep "$INTERVAL"
done
