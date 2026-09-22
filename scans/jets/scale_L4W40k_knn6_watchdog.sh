#!/bin/bash
# Watch knn6 40k width array + nostar ablation; resubmit on failure / OOM.
# D=512 defaults to B=128 in the slurm; on OOM fallback BATCH=64.
# Logs: $ROOT/logs/watchdog_L4W40k_knn6.log
set -uo pipefail

ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law
SLURM_W=/home/jdezoort/coupled-particle-edge-networks/scans/jets/scaling_law_L4_width_40k_knn6.slurm
SLURM_NS=/home/jdezoort/coupled-particle-edge-networks/scans/jets/scaling_law_L4_ablate_knn6_nostar_1M.slurm
LOG="$ROOT/logs/watchdog_L4W40k_knn6.log"
STATE=/tmp/watchdog_L4W40k_knn6_state.txt
TARGET=40000
SLEEP_SEC=180
JOB_W=jc-scale-L4W40k-knn6
JOB_NS=jc-ablate-knn6ns
PYTHON="${PYTHON:-/home/jdezoort/.conda/envs/mamba-env/bin/python}"
mkdir -p "$ROOT/logs"

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
  find "$ROOT/jetclass_output" -type f -name last.ckpt 2>/dev/null \
    | grep -E "capen-llama-att_4_${width}_0p25_.*knn6" \
    | grep -v nostar \
    | grep 'scale-knn6\|-s40k' \
    | head -n 1 || true
}

log_has_oom() {
  local f="$1"
  [[ -f "$f" ]] || return 1
  grep -qE 'CUDA out of memory|torch\.OutOfMemoryError|OutOfMemoryError' "$f" 2>/dev/null
}

task_in_queue() {
  local task="$1"
  squeue -u "$USER" -n "$JOB_W" -r -h -o '%i' 2>/dev/null \
    | awk -F_ -v t="$task" '$NF==t {found=1} END{exit !found}'
}

ns_in_queue() {
  squeue -u "$USER" -n "$JOB_NS" -h -o '%i' 2>/dev/null | grep -q .
}

latest_task_log() {
  local task="$1" kind="$2"
  ls -t "$ROOT/logs"/jc_scale_L4W40k_knn6_*_"${task}.${kind}" 2>/dev/null | head -n 1 || true
}

latest_ns_log() {
  local kind="$1"
  ls -t "$ROOT/logs"/jc_ablate_knn6ns_*.${kind} 2>/dev/null | head -n 1 || true
}

last_task_failed() {
  local task="$1"
  local line state exitc
  line=$(sacct -n -X -u "$USER" --name="$JOB_W" --format=JobID,State,ExitCode -S now-7days 2>/dev/null \
    | awk -v t="_$task" '$1 ~ t"$" || $1 ~ t"\\." {print; exit}')
  [[ -z "$line" ]] && return 1
  state=$(echo "$line" | awk '{print $2}')
  exitc=$(echo "$line" | awk '{print $3}')
  case "$state" in
    FAILED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|CANCELLED) return 0 ;;
    COMPLETED)
      [[ "$exitc" != "0:0" ]] && return 0
      return 1
      ;;
    *) return 1 ;;
  esac
}

submit_width_task() {
  local task="$1" batch_override="${2:-}"
  local out
  if [[ -n "$batch_override" ]]; then
    out=$(sbatch --export=ALL,BATCH="$batch_override" --array="$task" "$SLURM_W" 2>&1) || true
  else
    out=$(sbatch --array="$task" "$SLURM_W" 2>&1) || true
  fi
  echo "$(date -Is) submit task=$task batch=${batch_override:-default} out=$out"
}

submit_ns() {
  local out
  out=$(sbatch "$SLURM_NS" 2>&1) || true
  echo "$(date -Is) submit nostar out=$out"
}

echo "watchdog_pid=$$ started=$(date -Is)" > "$STATE"

{
  echo "$(date -Is) watchdog start pid=$$ target_steps=$TARGET"
  while true; do
    done_count=0
    any_active=0

    for task_width in 0:64 1:128 2:256 3:512; do
      task=${task_width%%:*}
      width=${task_width##*:}

      if task_in_queue "$task"; then
        any_active=1
        err=$(latest_task_log "$task" err)
        if [[ -n "${err:-}" ]] && log_has_oom "$err"; then
          echo "$(date -Is) OOM while running task=$task D=$width — cancelling + fallback batch"
          mapfile -t hits < <(squeue -u "$USER" -n "$JOB_W" -r -h -o '%i' | awk -F_ -v t="$task" '$NF==t{print}')
          for j in "${hits[@]:-}"; do scancel "$j" 2>/dev/null || true; done
          sleep 2
          if (( width >= 512 )); then submit_width_task "$task" 64; else submit_width_task "$task" 128; fi
          any_active=1
        else
          echo "$(date -Is) task=$task D=$width queued/running"
        fi
        continue
      fi

      ckpt=$(ckpt_for_width "$width" || true)
      step=-1
      [[ -n "${ckpt:-}" ]] && step=$(step_of "$ckpt")
      err=$(latest_task_log "$task" err)
      oom=0
      [[ -n "${err:-}" ]] && log_has_oom "$err" && oom=1

      echo "$(date -Is) task=$task D=$width step=$step oom=$oom ckpt=${ckpt:-none}"

      if (( step >= TARGET - 200 )); then
        done_count=$((done_count + 1))
        echo "$(date -Is) task=$task D=$width DONE"
        continue
      fi

      if (( oom == 1 )); then
        if (( width >= 512 )); then submit_width_task "$task" 64; else submit_width_task "$task" 128; fi
        any_active=1
      elif last_task_failed "$task" || (( step >= 0 && step < TARGET - 200 )); then
        submit_width_task "$task"
        any_active=1
      elif [[ -n "${err:-}" ]] && grep -qE 'Error|Traceback|FAILED' "$err" 2>/dev/null; then
        submit_width_task "$task"
        any_active=1
      else
        echo "$(date -Is) task=$task D=$width waiting for first start (no resubmit)"
      fi
    done

    if ns_in_queue; then
      any_active=1
      echo "$(date -Is) nostar queued/running"
      ns_err=$(latest_ns_log err)
      if [[ -n "${ns_err:-}" ]] && log_has_oom "$ns_err"; then
        echo "$(date -Is) nostar OOM — cancel + resubmit"
        scancel -n "$JOB_NS" 2>/dev/null || true
        sleep 2
        submit_ns
      fi
    else
      ns_err=$(latest_ns_log err)
      ns_out=$(latest_ns_log out)
      if [[ -n "${ns_err:-}" ]] && log_has_oom "$ns_err"; then
        echo "$(date -Is) nostar OOM finished — resubmit"
        submit_ns
        any_active=1
      elif [[ -n "${ns_err:-}" ]] && grep -qE 'Error|Traceback' "$ns_err" 2>/dev/null \
           && ! grep -qE 'step=[0-9]{3,}' "${ns_out:-/dev/null}" 2>/dev/null; then
        echo "$(date -Is) nostar failed early — resubmit"
        submit_ns
        any_active=1
      else
        echo "$(date -Is) nostar not in queue (ok or finished)"
      fi
    fi

    echo "done=$done_count/4 active=$any_active ts=$(date -Is)" > "$STATE"
    if (( done_count == 4 && any_active == 0 )); then
      echo "$(date -Is) all width tasks at target; watchdog exit"
      break
    fi
    sleep "$SLEEP_SEC"
  done
} >>"$LOG" 2>&1
