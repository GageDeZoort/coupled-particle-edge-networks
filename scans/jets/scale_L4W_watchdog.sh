#!/bin/bash
# Resubmit L=4 unique-pass width jobs until each hits 19532 steps.
# Walltime is sized from remaining steps (measured ms/step + 20% buffer).
# Logs: /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/watchdog_L4W.log
set -euo pipefail

ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law
SLURM=/home/jdezoort/coupled-particle-edge-networks/scans/jets/scaling_law_L4_width.slurm
LOG="$ROOT/logs/watchdog_L4W.log"
TARGET=19532
SLEEP_SEC=600
JOB_NAME=jc-scale-L4W
PYTHON="${PYTHON:-/home/jdezoort/.conda/envs/mamba-env/bin/python}"
mkdir -p "$ROOT/logs"

# Measured compute ms/step on 1×A100 B=512 (from the first 5h slices).
declare -A MS_PER_STEP=([64]=450 [128]=800 [256]=1260 [512]=2300)

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

ckpt_for_width() {
  local width="$1"
  # Pin to the 10M unique-pass η₀=0.25 family so the L=4 η₀ sweep
  # (D12.8M / s25k) never shadows these checkpoints.
  find "$ROOT" -type f -name last.ckpt 2>/dev/null \
    | grep -E "capen-llama-att_4_${width}_0p25_.*D10M-s19p532k" \
    | head -n 1 || true
}

walltime_for() {
  local width="$1" remain="$2"
  local ms="${MS_PER_STEP[$width]}"
  local sec=$(( (remain * ms * 12 / 10) / 1000 + 1 ))
  local min=$(( (sec + 59) / 60 ))
  # Round up to 15 min; keep a 20 min floor and a 12h cap.
  if (( min < 20 )); then min=20; fi
  if (( min % 15 != 0 )); then min=$(( min + 15 - min % 15 )); fi
  if (( min > 720 )); then min=720; fi
  printf '%d:%02d:00' $((min / 60)) $((min % 60))
}

{
  echo "$(date -Is) watchdog start pid=$$ target_steps=$TARGET"
  while true; do
    mapfile -t qids < <(squeue -u "$USER" -n "$JOB_NAME" -r -h -o '%i' 2>/dev/null || true)
    queued_tasks=" "
    for jid in "${qids[@]:-}"; do
      [[ -z "$jid" ]] && continue
      queued_tasks+="${jid##*_} "
    done
    need_tasks=()
    need_times=()
    for task_width in 0:64 1:128 2:256 3:512; do
      task=${task_width%%:*}
      width=${task_width##*:}
      if [[ "$queued_tasks" == *" $task "* ]]; then
        echo "$(date -Is) task=$task D=$width still queued/running"
        continue
      fi
      ckpt=$(ckpt_for_width "$width" || true)
      step=-1
      if [[ -n "${ckpt:-}" ]]; then
        step=$(step_of "$ckpt")
      fi
      echo "$(date -Is) task=$task D=$width step=$step ckpt=${ckpt:-none}"
      # last.ckpt is every 200 steps; a completed 19532-step run may sit at 19400.
      if (( step >= TARGET - 200 )); then
        continue
      fi
      remain=$((TARGET - step))
      if (( step < 0 )); then remain=$TARGET; fi
      wt=$(walltime_for "$width" "$remain")
      echo "$(date -Is) task=$task D=$width remain=$remain walltime=$wt"
      need_tasks+=("$task")
      need_times+=("$wt")
    done
    if (( ${#need_tasks[@]} == 0 )); then
      if [[ "$queued_tasks" != " " ]]; then
        echo "$(date -Is) waiting on queued job(s):$queued_tasks"
      else
        echo "$(date -Is) all widths at $TARGET steps; exiting"
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
