#!/bin/bash
# Refresh panel-(c) M11 + edge η₀×D live canvas (seeds 0+1) every 2m.
set -euo pipefail
REPO=/home/jdezoort/coupled-particle-edge-networks
cd "$REPO"
while true; do
  python3 scans/jets/update_m11_eta_panel_c_live_canvas.py || true
  sleep 120
done
