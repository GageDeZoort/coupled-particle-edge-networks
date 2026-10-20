#!/bin/bash
# Panel-(c) shard1 error-bar wave on gputest (D∈{256,384} × η₀×6).
#
# Four array jobs (MaxJobsPU=3 → submit 3 now, queue the 4th via watchdog):
#   1. M11  shard1 seed0  jc-m11-gt-sh1
#   2. M11  shard1 seed1  jc-m11-gt-sh1-s1
#   3. edge shard1 seed0  jc-edge-gt-sh1
#   4. edge shard1 seed1  jc-edge-gt-sh1-s1  (watchdog when a slot frees)
#
#   bash scans/jets/submit_panel_c_shard1_gputest_wave.sh
#
set -euo pipefail

REPO=/home/jdezoort/coupled-particle-edge-networks
LOGS=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs
M11_SLURM="$REPO/scans/jets/scaling_law_L4_m11_eta_gputest_s2500.slurm"
EDGE_SLURM="$REPO/scans/jets/scaling_law_L4_edge_eta_gputest_s2500.slurm"
WATCH_LOG="$LOGS/jc_panel_c_shard1_wave_watchdog.log"
mkdir -p "$LOGS"

gputest_arrays() {
  squeue -u jdezoort -p gputest -h -o "%i" 2>/dev/null \
    | sed 's/_.*//' | sort -u | wc -l
}

submit_one() {
  local name="$1" seed="$2" shard="$3" slurm="$4" log_prefix="$5" array="${6:-0-11}"
  local out
  out=$(sbatch --parsable --array="$array" \
    --job-name="$name" \
    --output="$LOGS/${log_prefix}_%A_%a.out" \
    --error="$LOGS/${log_prefix}_%A_%a.err" \
    --export=ALL,SEED="$seed",SHARD="$shard" \
    "$slurm" 2>&1) || true
  echo "$out"
}

echo "=== panel-c shard1 wave $(date -Is) ==="
echo "gputest arrays now: $(gputest_arrays)"

# Prefer clearing leftover shard0 seed1 half-B watchdog noise; don't cancel running science.
# Submit up to 3 arrays.
SUBMITTED=()

try_submit() {
  local name="$1" seed="$2" shard="$3" slurm="$4" log_prefix="$5"
  local n
  n=$(gputest_arrays)
  if (( n >= 3 )); then
    echo "SKIP $name (gputest arrays=$n ≥ 3)"
    return 1
  fi
  local jid
  jid=$(submit_one "$name" "$seed" "$shard" "$slurm" "$log_prefix")
  if [[ "$jid" =~ ^[0-9]+$ ]]; then
    echo "OK $name job=$jid SEED=$seed SHARD=$shard"
    SUBMITTED+=("$name:$jid")
    return 0
  fi
  echo "FAIL $name: $jid"
  return 1
}

try_submit jc-m11-gt-sh1    0 1 "$M11_SLURM"  jc_m11_s2500_gt_sh1    || true
try_submit jc-m11-gt-sh1-s1 1 1 "$M11_SLURM"  jc_m11_s2500_gt_sh1_s1 || true
try_submit jc-edge-gt-sh1   0 1 "$EDGE_SLURM" jc_edge_s2500_gt_sh1   || true

# Watchdog for edge shard1 seed1 (and any that failed to submit above).
nohup bash -c '
LOGS="'"$LOGS"'"
REPO="'"$REPO"'"
M11_SLURM="'"$M11_SLURM"'"
EDGE_SLURM="'"$EDGE_SLURM"'"
WATCH_LOG="'"$WATCH_LOG"'"
gputest_arrays() {
  squeue -u jdezoort -p gputest -h -o "%i" 2>/dev/null \
    | sed "s/_.*//" | sort -u | wc -l
}
need_job() {
  local name="$1"
  squeue -u jdezoort -n "$name" -h 2>/dev/null | grep -q . && return 1
  # Also skip if we already have a completed array log stamp for all tasks — keep simple: resubmit if not in queue.
  return 0
}
submit_one() {
  local name="$1" seed="$2" shard="$3" slurm="$4" log_prefix="$5"
  sbatch --parsable --array=0-11 \
    --job-name="$name" \
    --output="$LOGS/${log_prefix}_%A_%a.out" \
    --error="$LOGS/${log_prefix}_%A_%a.err" \
    --export=ALL,SEED="$seed",SHARD="$shard" \
    "$slurm" 2>&1 || true
}
echo "watchdog start $(date -Is)" | tee -a "$WATCH_LOG"
# Targets in priority order
declare -a NAMES=(jc-m11-gt-sh1 jc-m11-gt-sh1-s1 jc-edge-gt-sh1 jc-edge-gt-sh1-s1)
declare -a SEEDS=(0 1 0 1)
declare -a SLURMS=("$M11_SLURM" "$M11_SLURM" "$EDGE_SLURM" "$EDGE_SLURM")
declare -a PREFIXES=(jc_m11_s2500_gt_sh1 jc_m11_s2500_gt_sh1_s1 jc_edge_s2500_gt_sh1 jc_edge_s2500_gt_sh1_s1)
declare -a DONE=(0 0 0 0)
for i in $(seq 1 480); do
  n=$(gputest_arrays)
  echo "$(date -Is) gputest_arrays=$n done=${DONE[*]}" | tee -a "$WATCH_LOG"
  for k in 0 1 2 3; do
    (( DONE[k] == 1 )) && continue
    # If job name already queued/running, mark done-submit
    if squeue -u jdezoort -n "${NAMES[k]}" -h 2>/dev/null | grep -q .; then
      DONE[k]=1
      continue
    fi
    # Heuristic: if ≥10 task logs exist with this prefix from a recent job, consider submitted
    nlogs=$(ls "$LOGS"/${PREFIXES[k]}_*_*.out 2>/dev/null | wc -l)
    if (( nlogs >= 10 )); then
      DONE[k]=1
      continue
    fi
    if (( n < 3 )); then
      jid=$(submit_one "${NAMES[k]}" "${SEEDS[k]}" 1 "${SLURMS[k]}" "${PREFIXES[k]}")
      echo "submit ${NAMES[k]} → $jid" | tee -a "$WATCH_LOG"
      if [[ "$jid" =~ ^[0-9]+$ ]]; then
        DONE[k]=1
        n=$((n+1))
      fi
    fi
  done
  if (( DONE[0]+DONE[1]+DONE[2]+DONE[3] == 4 )); then
    echo "ALL four shard1 arrays submitted $(date -Is)" | tee -a "$WATCH_LOG"
    exit 0
  fi
  sleep 90
done
echo "FAIL watchdog timeout $(date -Is)" | tee -a "$WATCH_LOG"
exit 1
' >"$WATCH_LOG" 2>&1 &
echo "watchdog_pid=$! log=$WATCH_LOG"

echo "=== submitted now ==="
printf '%s\n' "${SUBMITTED[@]:-none}"
echo "gputest arrays after: $(gputest_arrays)"
squeue -u jdezoort -o '%.18i %.9P %.28j %.8T %.10M' 2>/dev/null | head -40 || true
