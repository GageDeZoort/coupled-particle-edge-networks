#!/bin/bash
# Rewrite the v2cp graph-k live canvas every 2 minutes until 48/48 done.
set -u
PY=/home/jdezoort/coupled-particle-edge-networks/scans/jets/refresh_graph_k_v2cp_canvas.py
LOG=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/graph_k_v2cp_canvas_refresh.log
mkdir -p "$(dirname "$LOG")"
echo "$(date -Is) watch_graph_k_v2cp_canvas.sh start pid=$$" >>"$LOG"
while true; do
  ts=$(date -Is)
  out=$(python "$PY" 2>&1) || true
  echo "$ts $out" >>"$LOG"
  if echo "$out" | grep -q 'done=48 '; then
    echo "$ts grid complete — stopping" >>"$LOG"
    break
  fi
  sleep 120
done
