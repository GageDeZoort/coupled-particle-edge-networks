#!/usr/bin/env bash
# Keep typed-edge-eta-D256-k7-s40k-live.canvas.tsx fresh while the η₀ sweep rolls.
set -u
PY=/home/jdezoort/.conda/envs/mamba-env/bin/python
SCRIPT=/home/jdezoort/coupled-particle-edge-networks/scans/jets/refresh_typed_edge_eta_D256_k7_canvas.py
LOG=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/canvas_typed_k7_refresh_watch.log
mkdir -p "$(dirname "$LOG")"
INTERVAL="${INTERVAL:-90}"
echo "$(date -Is) typed-k7 canvas watch start interval=${INTERVAL}s" >>"$LOG"
while true; do
  if ! "$PY" "$SCRIPT" >>"$LOG" 2>&1; then
    echo "$(date -Is) refresh FAILED" >>"$LOG"
  fi
  sleep "$INTERVAL"
done
