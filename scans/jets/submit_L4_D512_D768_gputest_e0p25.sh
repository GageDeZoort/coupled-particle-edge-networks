#!/usr/bin/env bash
# Submit L4 D=512 H=16 and L4 D=768 H=24 baseline gpu-test @ η₀=0.25 → 45k.
#
#   bash scans/jets/submit_L4_D512_D768_gputest_e0p25.sh
set -euo pipefail
cd /home/jdezoort/coupled-particle-edge-networks

ETA0=0.25
ETA_TAG=0p25
MAX_STEPS=45000
BATCH=128

submit_one() {
  local WIDTH="$1" HEAD="$2"
  local SLURM="scans/jets/scaling_law_L4_D${WIDTH}_b128_s45k_gputest.slurm"
  local name="jc-L4D${WIDTH}-gt-e${ETA_TAG}"
  local RUN_TAG="L4D${WIDTH}-gputest-b${BATCH}-g4-e${ETA_TAG}"
  local out
  out=$(sbatch \
    --job-name="$name" \
    --exclude=della-l02g6,della-l03g11 \
    --export=ALL,ETA0="$ETA0",RUN_TAG="$RUN_TAG",MAX_STEPS="$MAX_STEPS",BATCH="$BATCH" \
    "$SLURM")
  echo "$out  ($name L=4 D=$WIDTH H=$HEAD eta0=$ETA0 → ${MAX_STEPS})"
}

echo "whoami=$(whoami)"
submit_one 512 16
submit_one 768 24

SLEEP_SEC=900 bash scans/jets/start_L4_D512_gputest_watchdogs.sh
SLEEP_SEC=900 bash scans/jets/start_L4_D768_gputest_watchdogs.sh

echo "=== final queue (gputest / jc-L4) ==="
squeue -u "$USER" -o '%.18i %.9P %.40j %.8T %.10M %R' | grep -E 'JOBID|jc-L4D(512|768)|gputest' || squeue -u "$USER" -o '%.18i %.9P %.40j %.8T %.10M %R'
