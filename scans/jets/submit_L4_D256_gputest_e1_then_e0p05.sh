#!/usr/bin/env bash
# One more gputest arm: D=256 baseline η₀=1 → 45k, then watchdog pivots to η₀=0.05.
#
#   bash scans/jets/submit_L4_D256_gputest_e1_then_e0p05.sh
#
# Does not touch D512/D768 watchdogs. Replaces any prior D256 eta-chain watchdog.
set -euo pipefail
cd /home/jdezoort/coupled-particle-edge-networks

SLURM=scans/jets/scaling_law_L4_D256_b128_s45k_gputest.slurm
NOHUP_DIR="${NOHUP_DIR:-$HOME/.cache/jc_L4D256_watchdogs}"
mkdir -p "$NOHUP_DIR"

ETA0=1
ETA_TAG=$(python3 -c "e=float('$ETA0'); print(f'{e:g}'.replace('.','p'))")
JOB_NAME="jc-L4D256-gt-e${ETA_TAG}"
RUN_TAG="L4D256-gputest-b128-g4-e${ETA_TAG}"

echo "whoami=$(whoami)"
echo "Submitting first arm: ETA0=$ETA0 JOB_NAME=$JOB_NAME RUN_TAG=$RUN_TAG"

out=$(sbatch \
  --job-name="$JOB_NAME" \
  --exclude=della-l02g6,della-l03g11 \
  --export=ALL,ETA0="$ETA0",RUN_TAG="$RUN_TAG",MAX_STEPS=45000,BATCH=128 \
  "$SLURM")
echo "$out  ($JOB_NAME eta0=$ETA0)"

# Stop any prior chain watchdog only (never match this process's cmdline).
self=$$
ps -u "$USER" -o pid=,cmd= | while read -r pid cmd; do
  [[ "$pid" == "$self" ]] && continue
  [[ "$cmd" == *L4_D256_gputest_eta_chain_watchdog.sh* ]] || continue
  kill "$pid" 2>/dev/null || true
done
sleep 1

SLEEP_SEC="${SLEEP_SEC:-900}"
TARGET="${TARGET:-45000}"
CHAIN_WD="$PWD/scans/jets/L4_D256_gputest_eta_chain_watchdog.sh"
setsid env ETA_CHAIN="1 0.05" TARGET="$TARGET" SLEEP_SEC="$SLEEP_SEC" MAX_RESUBMITS=200 \
  bash "$CHAIN_WD" \
  >>"$NOHUP_DIR/nohup_eta_chain_e1_e0p05.out" 2>&1 < /dev/null &
echo "started eta-chain watchdog pid=$!  (1 → 0.05)  nohup=$NOHUP_DIR/nohup_eta_chain_e1_e0p05.out"
disown 2>/dev/null || true

sleep 2
echo '=== D256 chain + width watchdogs ==='
ps -u "$USER" -o pid,etime,cmd | grep -E 'L4_D(256|512|768)_gputest' | grep -v grep || true
echo '=== queue (best effort) ==='
timeout 20 squeue -u "$USER" -o '%.18i %.9P %.30j %.8T %.10M %R' 2>&1 || echo "(squeue unavailable)"
