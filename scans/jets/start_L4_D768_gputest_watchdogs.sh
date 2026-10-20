#!/usr/bin/env bash
# Start L4/D=768/H=24 baseline (corr=1) gpu-test watchdogs.
# Default η=0.25, B=128×4, TARGET=45k.
#
#   bash scans/jets/start_L4_D768_gputest_watchdogs.sh
set -euo pipefail
cd /home/jdezoort/coupled-particle-edge-networks
WD=scans/jets/L4_D768_gputest_watchdog.sh
NOHUP_DIR="${NOHUP_DIR:-$HOME/.cache/jc_L4D768_watchdogs}"
mkdir -p "$NOHUP_DIR"
SLEEP_SEC="${SLEEP_SEC:-900}"
TARGET="${TARGET:-45000}"
ETAS="${ETAS:-0.25}"

mapfile -t _pids < <(pgrep -f 'scans/jets/L4_D768_gputest_watchdog.sh' || true)
for p in "${_pids[@]:-}"; do
  kill "$p" 2>/dev/null || true
done
sleep 1

for ETA0 in $ETAS; do
  ETA_TAG=$(python3 -c "e=float('$ETA0'); print(f'{e:g}'.replace('.','p'))")
  nohup env ETA0="$ETA0" TARGET="$TARGET" SLEEP_SEC="$SLEEP_SEC" MAX_RESUBMITS=200 \
    bash "$WD" >>"$NOHUP_DIR/nohup_e${ETA_TAG}.out" 2>&1 &
  echo "started baseline watchdog ETA0=$ETA0 pid=$! nohup=$NOHUP_DIR/nohup_e${ETA_TAG}.out"
done

sleep 3
echo '=== baseline D768 watchdog procs ==='
ps -u "$USER" -o pid,etime,cmd | grep -F 'scans/jets/L4_D768_gputest_watchdog' | grep -v grep || true
echo '=== nohup tails ==='
for ETA0 in $ETAS; do
  ETA_TAG=$(python3 -c "e=float('$ETA0'); print(f'{e:g}'.replace('.','p'))")
  echo "-- e${ETA_TAG} --"
  tail -6 "$NOHUP_DIR/nohup_e${ETA_TAG}.out" 2>/dev/null || true
done
