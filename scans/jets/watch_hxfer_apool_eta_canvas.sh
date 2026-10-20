#!/bin/bash
# Refresh dual-arm (attn vs uniform) canvas every 3m
set -u
REPO=/home/jdezoort/coupled-particle-edge-networks
# shellcheck disable=SC1091
[[ -f /tmp/hxfer_lambda_ssh_env.sh ]] && source /tmp/hxfer_lambda_ssh_env.sh
while true; do
  date -Iseconds
  python3 "$REPO/scans/jets/update_hxfer_apool_eta_constlr_live_canvas.py" || true
  sleep 180
done
