#!/bin/bash
# Resubmit unfinished L=4 D∈{64,128} η₀-sweep cells on MIG until 25000 steps.
# A100 ms/step ×7 (1g.10gb) + 20% buffer; cap 24h so jobs stay gpu-short.
# Logs: /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/watchdog_L4lr.log
set -euo pipefail

ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law
SLURM=/home/jdezoort/coupled-particle-edge-networks/scans/jets/scaling_law_L4_lr_sweep_mig.slurm
LOG="$ROOT/logs/watchdog_L4lr.log"
TARGET=25000
SLEEP_SEC=600
JOB_NAME=jc-scale-L4lr
PYTHON="${PYTHON:-/home/jdezoort/.conda/envs/mamba-env/bin/python}"
mkdir -p "$ROOT/logs"

# 1g.10gb is ~1/7 of an A100; keep B=512 (fits in 10 GB for D=64/128).
declare -A MS_PER_STEP=([64]=3150 [128]=5600)
WIDTHS=(64 128)
ETAS=(0.01 0.05 0.1 0.25 0.5 1 2.5)
ETA_TAGS=(0p01 0p05 0p1 0p25 0p5 1 2p5)

step_of() {
  "$PYTHON" - "$1" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    print(-1)
    raise SystemExit
try:
    import torch
    ck = torch.load(p, map_location="cpu", weights_only=False)
except Exception:
    print(-1)
    raise SystemExit
print(int(ck.get("global_step", -1)))
PY
}

ckpt_for_task() {
  local width="$1" eta_tag="$2"
  find "$ROOT" -type f -name last.ckpt 2>/dev/null \
    | grep -E "capen-llama-att_4_${width}_${eta_tag}_.*scale-lrsweep" \
    | head -n 1 || true
}

walltime_for() {
  local width="$1" remain="$2"
  local ms="${MS_PER_STEP[$width]}"
  local sec=$(( (remain * ms * 12 / 10) / 1000 + 1 ))
  local min=$(( (sec + 59) / 60 ))
  if (( min < 20 )); then min=20; fi
  if (( min % 15 != 0 )); then min=$(( min + 15 - min % 15 )); fi
  if (( min > 1440 )); then min=1440; fi
  printf '%d:%02d:00' $((min / 60)) $((min % 60))
}

{
  echo "$(date -Is) watchdog start pid=$$ target_steps=$TARGET"
  while true; do
    # -r expands pending array ranges (e.g. 13983531_[7-13]) into per-task ids.
    mapfile -t qids < <(squeue -u "$USER" -n "$JOB_NAME" -r -h -o '%i' 2>/dev/null || true)
    queued_tasks=" "
    for jid in "${qids[@]:-}"; do
      [[ -z "$jid" ]] && continue
      queued_tasks+="${jid##*_} "
    done
    need_tasks=()
    need_times=()
    for task in $(seq 0 13); do
      w_idx=$((task / 7))
      e_idx=$((task % 7))
      width=${WIDTHS[$w_idx]}
      eta=${ETAS[$e_idx]}
      eta_tag=${ETA_TAGS[$e_idx]}
      if [[ "$queued_tasks" == *" $task "* ]]; then
        echo "$(date -Is) task=$task D=$width eta0=$eta still queued/running"
        continue
      fi
      ckpt=$(ckpt_for_task "$width" "$eta_tag" || true)
      step=-1
      if [[ -n "${ckpt:-}" ]]; then
        step=$(step_of "$ckpt")
      fi
      echo "$(date -Is) task=$task D=$width eta0=$eta step=$step ckpt=${ckpt:-none}"
      if (( step >= TARGET - 200 )); then
        continue
      fi
      remain=$((TARGET - step))
      if (( step < 0 )); then remain=$TARGET; fi
      wt=$(walltime_for "$width" "$remain")
      echo "$(date -Is) task=$task D=$width eta0=$eta remain=$remain walltime=$wt"
      need_tasks+=("$task")
      need_times+=("$wt")
    done
    if (( ${#need_tasks[@]} == 0 )); then
      if [[ "$queued_tasks" != " " ]]; then
        echo "$(date -Is) waiting on queued job(s):$queued_tasks"
      else
        echo "$(date -Is) all D=64/128 η₀ cells at $TARGET steps; exiting"
        break
      fi
    else
      for i in "${!need_tasks[@]}"; do
        echo "$(date -Is) sbatch --time=${need_times[$i]} --array=${need_tasks[$i]}"
        sbatch --time="${need_times[$i]}" --array="${need_tasks[$i]}" "$SLURM" \
          || echo "$(date -Is) sbatch failed task=${need_tasks[$i]}"
      done
    fi
    sleep "$SLEEP_SEC"
  done
} >>"$LOG" 2>&1
