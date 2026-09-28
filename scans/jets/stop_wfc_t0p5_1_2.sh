#!/bin/bash
# Stop leftover AdamW WFC watchdogs/jobs for retired t_epoch ∈ {0.1, 0.25, 0.5, 1, 2}.
# Leaves t ∈ {5, 10, 25} alone.
set -euo pipefail
CACHE="${HOME}/.cache/jc_L4D256_wfc_watchdogs"
ROOTLOG=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs
mkdir -p "$CACHE" "$ROOTLOG"
for t in 0p1 0p25 0p5 1 2; do
  touch "$CACHE/STOP_t${t}" "$ROOTLOG/STOP_t${t}" 2>/dev/null || true
done
# Kill retired-arm watchdog processes (match T_EPOCH= in the env/cmdline).
for T in 0.1 0.25 0.5 1 2; do
  pkill -f "L4_D256_wfc_gputest_watchdog.sh.*T_EPOCH=${T}" 2>/dev/null || true
done
for name in jc-L4D256-wfc-t0p1 jc-L4D256-wfc-t0p25 jc-L4D256-wfc-t0p5 jc-L4D256-wfc-t1 jc-L4D256-wfc-t2; do
  scancel -u "$USER" -n "$name" 2>/dev/null || true
done
echo "STOP sentinels set; scancel issued for retired t_epoch arms"
squeue -u "$USER" -o '%.12i %.9P %.40j %.2t %.10M %R' | grep -E 'JOBID|wfc' || true
