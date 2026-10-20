#!/usr/bin/env bash
# Keep toptag-ft-L4cos-live.canvas.tsx fresh while frozen/full FT sweeps run.
set -u
PY=/home/jdezoort/.conda/envs/mamba-env/bin/python
SCRIPT=/home/jdezoort/coupled-particle-edge-networks/scans/jets/refresh_toptag_ft_L4cos_canvas.py
LOG=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/toptagging_finetune/logs/canvas_ft_L4cos_refresh_watch.log
mkdir -p "$(dirname "$LOG")"
INTERVAL="${INTERVAL:-90}"
echo "$(date -Is) FT canvas watch start interval=${INTERVAL}s" >>"$LOG"
while true; do
  if ! "$PY" "$SCRIPT" >>"$LOG" 2>&1; then
    echo "$(date -Is) refresh FAILED" >>"$LOG"
  fi
  sleep "$INTERVAL"
done
