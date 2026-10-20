#!/bin/bash
# Refresh AdamW 100M live canvas (L4/D128 + L6/D256) every 2m
set -u
REPO=/home/jdezoort/coupled-particle-edge-networks
while true; do
  date -Iseconds
  python3 "$REPO/scans/jets/update_hxfer_d128_adamw_t10_n100M_live_canvas.py" || true
  sleep 120
done
