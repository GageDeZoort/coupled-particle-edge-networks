#!/usr/bin/env bash
# Start AdamW t_epoch sweep watchdog (L4/D256 η₀=0.1, B=256×1 → 40k on a100 2h).
#
#   bash scans/jets/start_adamw_tepoch_D256_1a100_watchdogs.sh
# Stop:
#   touch ~/.cache/jc_aw_tepoch_watchdogs/STOP
#
set -euo pipefail
cd /home/jdezoort/coupled-particle-edge-networks
WD=scans/jets/adamw_tepoch_D256_1a100_watchdog.sh
NOHUP_DIR="${NOHUP_DIR:-$HOME/.cache/jc_aw_tepoch_watchdogs}"
mkdir -p "$NOHUP_DIR"
rm -f "$NOHUP_DIR/STOP" /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/STOP_aw_tepoch

SELF=$$
mapfile -t _pids < <(pgrep -f 'scans/jets/adamw_tepoch_D256_1a100_watchdog.sh' || true)
for p in "${_pids[@]:-}"; do
  [[ -z "$p" || "$p" == "$SELF" ]] && continue
  kill "$p" 2>/dev/null || true
done
sleep 1

TARGET="${TARGET:-40000}"
MAX_A100="${MAX_A100:-4}"
SLEEP_SEC="${SLEEP_SEC:-90}"

: >"$NOHUP_DIR/nohup_aw_tepoch.out"
nohup env TARGET="$TARGET" MAX_STEPS="$TARGET" MAX_A100="$MAX_A100" SLEEP_SEC="$SLEEP_SEC" \
  bash "$WD" >>"$NOHUP_DIR/nohup_aw_tepoch.out" 2>&1 &
echo "started aw-tepoch watchdog pid=$! nohup=$NOHUP_DIR/nohup_aw_tepoch.out TARGET=$TARGET MAX_A100=$MAX_A100"

sleep 3
echo '=== aw-tepoch watchdog procs ==='
ps -u "$USER" -o pid,etime,cmd | grep -F 'adamw_tepoch_D256_1a100_watchdog' | grep -v grep || true
echo '=== nohup tail ==='
tail -20 "$NOHUP_DIR/nohup_aw_tepoch.out" 2>/dev/null || true
