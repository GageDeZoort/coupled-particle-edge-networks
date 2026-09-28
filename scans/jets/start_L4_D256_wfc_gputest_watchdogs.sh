#!/bin/bash
# Start D=256 AdamW warmup→flat→cosine watchdogs for the active t_epoch arms.
#
# Default: only t ∈ {5, 10, 25} (t=0.5/1/2 are retired via STOP sentinels).
#
#   bash scans/jets/start_L4_D256_wfc_gputest_watchdogs.sh
#   T_EPOCHS="5 10" TARGET=150000 bash scans/jets/start_L4_D256_wfc_gputest_watchdogs.sh
#
set -euo pipefail

ROOT=/home/jdezoort/coupled-particle-edge-networks
WD="$ROOT/scans/jets/L4_D256_wfc_gputest_watchdog.sh"
CACHE="${HOME}/.cache/jc_L4D256_wfc_watchdogs"
ROOTLOG=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs
mkdir -p "$CACHE" "$ROOTLOG"

ETA0="${ETA0:-0.25}"
TARGET="${TARGET:-150000}"
SLEEP_SEC="${SLEEP_SEC:-900}"
# shellcheck disable=SC2206
T_EPOCHS=( ${T_EPOCHS:-5 10 25} )

# Retired arms: keep STOP sentinels so any leftover WD exits.
for t in 0p1 0p25 0p5 1 2; do
  touch "$CACHE/STOP_t${t}" "$ROOTLOG/STOP_t${t}" 2>/dev/null || true
done

# Kill every WFC watchdog first, then start only the requested arms.
pkill -f 'L4_D256_wfc_gputest_watchdog.sh' 2>/dev/null || true
sleep 1

for T in "${T_EPOCHS[@]}"; do
  T_TAG=$(python3 -c "e=float('${T}'); print(f'{e:g}'.replace('.','p'))")
  # Clear STOP for arms we intend to run.
  rm -f "$CACHE/STOP_t${T_TAG}" "$ROOTLOG/STOP_t${T_TAG}" 2>/dev/null || true
  LOG="$CACHE/watchdog_t${T_TAG}.log"
  nohup env ETA0="$ETA0" T_EPOCH="$T" TARGET="$TARGET" SLEEP_SEC="$SLEEP_SEC" \
    bash "$WD" >>"$LOG" 2>&1 &
  echo "started t_epoch=$T pid=$! TARGET=$TARGET log=$LOG"
done
