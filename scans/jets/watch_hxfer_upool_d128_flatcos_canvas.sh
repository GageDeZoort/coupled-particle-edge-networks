#!/bin/bash
# Refresh D=128 flat→cosine live canvas every 2m
set -u
REPO=/home/jdezoort/coupled-particle-edge-networks
# shellcheck disable=SC1091
[[ -f /tmp/hxfer_lambda_ssh_env.sh ]] && source /tmp/hxfer_lambda_ssh_env.sh
while true; do
  date -Iseconds
  python3 "$REPO/scans/jets/update_hxfer_upool_d128_flatcos_live_canvas.py" || true
  sleep 120
done
