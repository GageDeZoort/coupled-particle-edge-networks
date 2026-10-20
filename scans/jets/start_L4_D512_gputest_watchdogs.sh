#!/usr/bin/env bash
# Start L4/D=512/H=16 baseline (corr=1) gpu-test watchdogs.
# Default: η ∈ {0.1, 0.25, 0.5}, B=128×4, TARGET=100k.
#
#   bash scans/jets/start_L4_D512_gputest_watchdogs.sh
#   ETAS="0.25" TARGET=45000 bash scans/jets/start_L4_D512_gputest_watchdogs.sh
set -euo pipefail
cd /home/jdezoort/coupled-particle-edge-networks
WD=scans/jets/L4_D512_gputest_watchdog.sh
NOHUP_DIR="${NOHUP_DIR:-$HOME/.cache/jc_L4D512_watchdogs}"
mkdir -p "$NOHUP_DIR"
SLEEP_SEC="${SLEEP_SEC:-900}"
TARGET="${TARGET:-100000}"
ETAS="${ETAS:-0.1 0.25 0.5}"

self=$$
ps -u "$USER" -o pid=,cmd= | while read -r pid cmd; do
  [[ "$pid" == "$self" ]] && continue
  [[ "$cmd" == *L4_D512_gputest_watchdog.sh* ]] || continue
  kill "$pid" 2>/dev/null || true
done
sleep 1

for ETA0 in $ETAS; do
  ETA_TAG=$(python3 -c "e=float('$ETA0'); print(f'{e:g}'.replace('.','p'))")
  setsid env ETA0="$ETA0" TARGET="$TARGET" SLEEP_SEC="$SLEEP_SEC" MAX_RESUBMITS=200 \
    bash "$WD" >>"$NOHUP_DIR/nohup_e${ETA_TAG}_s${TARGET}.out" 2>&1 < /dev/null &
  echo "started baseline watchdog ETA0=$ETA0 TARGET=$TARGET pid=$! nohup=$NOHUP_DIR/nohup_e${ETA_TAG}_s${TARGET}.out"
done

sleep 3
echo '=== baseline D512 watchdog procs ==='
ps -u "$USER" -o pid,etime,cmd | grep -F 'scans/jets/L4_D512_gputest_watchdog' | grep -v grep || true
echo '=== nohup tails ==='
for ETA0 in $ETAS; do
  ETA_TAG=$(python3 -c "e=float('$ETA0'); print(f'{e:g}'.replace('.','p'))")
  echo "-- e${ETA_TAG} --"
  tail -6 "$NOHUP_DIR/nohup_e${ETA_TAG}_s${TARGET}.out" 2>/dev/null || true
done
