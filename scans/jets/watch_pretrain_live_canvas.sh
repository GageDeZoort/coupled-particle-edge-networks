#!/usr/bin/env bash
# Keep jetclass-pretrain-live.canvas.tsx fresh while L4/L8 run.
set -u
PY=/home/jdezoort/.conda/envs/mamba-env/bin/python
SCRIPT=/home/jdezoort/coupled-particle-edge-networks/scans/jets/refresh_pretrain_live_canvas.py
LOG=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_pretrain/logs/canvas_refresh_watch.log
mkdir -p "$(dirname "$LOG")"
INTERVAL="${INTERVAL:-120}"
# pick a working SSH agent once
pick_sock() {
  for s in /tmp/ssh-*/agent.*; do
    [ -O "$s" ] || continue
    if SSH_AUTH_SOCK=$s ssh -o BatchMode=yes -o ConnectTimeout=6 -o ControlMaster=no ubuntu@168.110.21.94 'true' 2>/dev/null; then
      echo "$s"; return 0
    fi
  done
  return 1
}
export SSH_AUTH_SOCK="$(pick_sock || true)"
echo "$(date -Is) canvas watch start interval=${INTERVAL}s sock=${SSH_AUTH_SOCK:-none}" >>"$LOG"
while true; do
  # refresh agent sock if dead
  if [ -n "${SSH_AUTH_SOCK:-}" ] && ! [ -S "$SSH_AUTH_SOCK" ]; then
    SSH_AUTH_SOCK="$(pick_sock || true)"; export SSH_AUTH_SOCK
  fi
  if ! "$PY" "$SCRIPT" >>"$LOG" 2>&1; then
    echo "$(date -Is) refresh FAILED" >>"$LOG"
  fi
  sleep "$INTERVAL"
done
