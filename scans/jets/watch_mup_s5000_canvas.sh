#!/bin/bash
# Rewrite the live canvas every 2 minutes until the 10-cell grid is done.
set -u
PY=/home/jdezoort/coupled-particle-edge-networks/scans/jets/refresh_mup_s5000_canvas.py
LOG=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/mup_s5000_canvas_refresh.log
mkdir -p "$(dirname "$LOG")"
echo "$(date -Is) watch_mup_s5000_canvas.sh start pid=$$" >>"$LOG"
while true; do
  ts=$(date -Is)
  out=$(python "$PY" 2>&1) || true
  echo "$ts $out" >>"$LOG"
  if echo "$out" | grep -q 'done=10 '; then
    echo "$ts grid complete — stopping" >>"$LOG"
    break
  fi
  sleep 120
done
