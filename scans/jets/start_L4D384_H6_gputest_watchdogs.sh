#!/usr/bin/env bash
# Start L4/D=384/H=6 (d_head=64) gputest (59m) watchdogs.
# Default η ∈ {0.01, 0.05, 0.1, 0.25, 0.5}. Override with ETAS=...
#   bash scans/jets/start_L4D384_H6_gputest_watchdogs.sh
set -euo pipefail
cd /home/jdezoort/coupled-particle-edge-networks
WD=scans/jets/L4_D384_H6_gputest_watchdog.sh
NOHUP_DIR="${NOHUP_DIR:-$HOME/.cache/jc_L4D384_watchdogs}"
mkdir -p "$NOHUP_DIR"
SLEEP_SEC="${SLEEP_SEC:-900}"
ETAS="${ETAS:-0.05 0.1 0.25}"

# Stop prior H6 H6 gpu_short watchdogs only (leave gputest WDs / Slurm jobs alone).
mapfile -t _pids < <(pgrep -f 'scans/jets/L4_D384_H6_gputest_watchdog.sh' || true)
for p in "${_pids[@]:-}"; do
  kill "$p" 2>/dev/null || true
done
sleep 1

for ETA0 in $ETAS; do
  ETA_TAG=$(python3 -c "e=float('$ETA0'); print(f'{e:g}'.replace('.','p'))")
  nohup env ETA0="$ETA0" TARGET=100000 SLEEP_SEC="$SLEEP_SEC" MAX_RESUBMITS=200 \
    bash "$WD" >>"$NOHUP_DIR/nohup_H6_gputest_e${ETA_TAG}.out" 2>&1 &
  echo "started H6 gpu_short watchdog ETA0=$ETA0 pid=$! nohup=$NOHUP_DIR/nohup_H6_gputest_e${ETA_TAG}.out"
done

sleep 3
echo '=== H6 gpu_short watchdog procs ==='
ps -u "$USER" -o pid,etime,cmd | grep -F 'scans/jets/L4_D384_H6_gputest_watchdog' | grep -v grep || true
echo '=== nohup tails ==='
for ETA0 in $ETAS; do
  ETA_TAG=$(python3 -c "e=float('$ETA0'); print(f'{e:g}'.replace('.','p'))")
  echo "-- e${ETA_TAG} --"
  tail -6 "$NOHUP_DIR/nohup_H6_gputest_e${ETA_TAG}.out" 2>/dev/null || true
done
