#!/bin/bash
# Submit edge panel-(c) seed1 gputest tasks 6–11 once gputest MaxJobsPU (<3 arrays) frees.
set -euo pipefail
LOGS=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs
REPO=/home/jdezoort/coupled-particle-edge-networks
LOG=$LOGS/jc_edge_gt_s1_halfB_watchdog.log
mkdir -p "$LOGS"
echo "watchdog start $(date -Is)" | tee -a "$LOG"
for i in $(seq 1 360); do
  # Unique array job IDs on gputest (MaxJobsPU=3 for this QOS).
  n=$(squeue -u jdezoort -p gputest -h -o "%i" 2>/dev/null \
    | sed 's/_.*//' | sort -u | wc -l)
  echo "$(date -Is) gputest_job_arrays=$n (need <3)" | tee -a "$LOG"
  if (( n < 3 )); then
    out=$(sbatch --parsable --array=6-11 \
      --job-name=jc-edge-gt-s1 \
      --output="$LOGS/jc_edge_s2500_gt_s1_%A_%a.out" \
      --error="$LOGS/jc_edge_s2500_gt_s1_%A_%a.err" \
      --export=ALL,SEED=1 \
      "$REPO/scans/jets/scaling_law_L4_edge_eta_gputest_s2500.slurm" 2>&1) || true
    echo "submit: $out" | tee -a "$LOG"
    if [[ "$out" =~ ^[0-9]+$ ]]; then
      echo "OK edge-gt-s1 tasks6-11 job=$out" | tee -a "$LOG"
      exit 0
    fi
  fi
  sleep 90
done
echo "FAIL timeout" | tee -a "$LOG"
exit 1
