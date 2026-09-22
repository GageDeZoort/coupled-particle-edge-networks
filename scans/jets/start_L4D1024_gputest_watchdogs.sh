#!/usr/bin/env bash
# Start/refresh the three L4/D=1024 gpu-test watchdogs (η ∈ {0.05, 0.1, 0.25}).
# Run from a login shell (not Cursor agent sandbox — sbatch needs controller access):
#   bash scans/jets/start_L4D1024_gputest_watchdogs.sh
set -euo pipefail
cd /home/jdezoort/coupled-particle-edge-networks
WD=scans/jets/L4_D1024_gputest_watchdog.sh
# Nohup stdout under home (scratch logs dir is often root-squashed / not writable from agents).
NOHUP_DIR="${NOHUP_DIR:-$HOME/.cache/jc_L4D1024_watchdogs}"
mkdir -p "$NOHUP_DIR"

# Stop any prior watchdogs for these arms (leave running Slurm jobs alone).
pkill -f 'L4_D1024_gputest_watchdog.sh' 2>/dev/null || true
sleep 1

for ETA0 in 0.05 0.1 0.25; do
  ETA_TAG=$(python3 -c "e=float('$ETA0'); print(f'{e:g}'.replace('.','p'))")
  nohup env ETA0="$ETA0" TARGET=100000 SLEEP_SEC=3600 MAX_RESUBMITS=200 \
    bash "$WD" >>"$NOHUP_DIR/nohup_e${ETA_TAG}.out" 2>&1 &
  echo "started watchdog ETA0=$ETA0 pid=$! nohup=$NOHUP_DIR/nohup_e${ETA_TAG}.out"
done

sleep 3
echo '=== squeue ==='
squeue -u jdezoort -o '%.12i %.9P %.28j %.2t %.10M %R' | head -20
echo '=== watchdog procs ==='
ps -u "$USER" -o pid,etime,cmd | grep -F L4_D1024_gputest_watchdog | grep -v grep || true
echo '=== nohup tails ==='
for ETA_TAG in 0p05 0p1 0p25; do
  echo "-- e${ETA_TAG} --"
  tail -5 "$NOHUP_DIR/nohup_e${ETA_TAG}.out" 2>/dev/null || true
done
