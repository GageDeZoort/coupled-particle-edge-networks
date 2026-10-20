#!/usr/bin/env bash
# Submit L4/D=256/H=8 baseline η∈{0.1,0.25,0.5} gpu-test arms + watchdogs → 45k.
#
#   bash scans/jets/submit_L4_D256_gputest.sh
set -euo pipefail
cd /home/jdezoort/coupled-particle-edge-networks

echo "whoami=$(whoami)"
SLURM=scans/jets/scaling_law_L4_D256_b128_s45k_gputest.slurm
ETAS=(0.1 0.25 0.5)

for ETA0 in "${ETAS[@]}"; do
  ETA_TAG=$(python3 -c "e=float('$ETA0'); print(f'{e:g}'.replace('.','p'))")
  name="jc-L4D256-gt-e${ETA_TAG}"
  RUN_TAG="L4D256-gputest-b128-g4-e${ETA_TAG}"
  out=$(sbatch \
    --job-name="$name" \
    --exclude=della-l02g6,della-l03g11 \
    --export=ALL,ETA0="$ETA0",RUN_TAG="$RUN_TAG",MAX_STEPS=45000,BATCH=128 \
    "$SLURM")
  echo "$out  ($name eta0=$ETA0)"
done

SLEEP_SEC=900 bash scans/jets/start_L4_D256_gputest_watchdogs.sh

echo "=== final queue ==="
squeue -u "$USER" -o '%.18i %.9P %.40j %.8T %.10M %R'
