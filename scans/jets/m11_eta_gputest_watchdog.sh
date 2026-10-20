#!/bin/bash
# Pool watchdog for M11-only η₀ × D grid on gpu-test (B_global=384, 4000 steps).
#
# Keeps up to MAX_CONCURRENT jc-m11 array tasks in the queue until every cell
# reaches TARGET steps (default 3950 ≈ 4000).
#
#   nohup bash scans/jets/m11_eta_gputest_watchdog.sh &
#   bash scans/jets/start_m11_eta_gputest_watchdogs.sh
#
set -uo pipefail

ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law
REPO=/home/jdezoort/coupled-particle-edge-networks
SLURM="$REPO/scans/jets/scaling_law_L4_m11_eta_gputest.slurm"
JOB_NAME="${JOB_NAME:-jc-m11}"
TARGET="${TARGET:-3950}"
MAX_STEPS="${MAX_STEPS:-4000}"
EPOCHS="${EPOCHS:-2}"
MAX_CONCURRENT="${MAX_CONCURRENT:-3}"
SLEEP_SEC="${SLEEP_SEC:-30}"
PYTHON="${PYTHON:-/home/jdezoort/.conda/envs/mamba-env/bin/python}"
CACHE="${HOME}/.cache/jc_m11_watchdogs"
LOG_DIR="$ROOT/logs"
mkdir -p "$CACHE" "$LOG_DIR"
LOG="$CACHE/watchdog_m11.log"
STOP_FILES=("$CACHE/STOP" "$LOG_DIR/STOP_m11")
LOCK="$CACHE/watchdog_m11.lock"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "$(date -Iseconds) another m11 watchdog holds $LOCK; exiting"
  exit 0
fi

ETAS=(0.05 0.1 0.25 0.5 1.0)
WIDTHS=(256 384 512)
D_HEAD=32
BATCH=96
N_GPUS=4

cell_meta() {
  local task=$1
  local width_idx=$((task / 5))
  local eta_idx=$((task % 5))
  echo "${WIDTHS[$width_idx]} ${ETAS[$eta_idx]}"
}

eta_tag() {
  python3 -c "e=float('$1'); print(f'{e:g}'.replace('.','p'))"
}

run_glob_for_task() {
  local task=$1
  read -r width eta <<<"$(cell_meta "$task")"
  local etag
  etag=$(eta_tag "$eta")
  echo "$ROOT/jetclass_output/adam/sweep_lr/baseline-transformer_4_${width}_${etag}_*m11-b${BATCH}-g${N_GPUS}-s${MAX_STEPS}-dh${D_HEAD}*"
}

# Fast parquet-only step scan for all 15 tasks (avoid torch.load on NFS).
all_steps() {
  local globs=()
  local t
  for t in $(seq 0 14); do
    globs+=("$(run_glob_for_task "$t")")
  done
  "$PYTHON" - "${globs[@]}" <<'PY'
import glob, sys
from pathlib import Path

def step_for(pat: str) -> int:
    best = 0
    for c in glob.glob(pat):
        p = Path(c)
        pq = p if p.suffix == ".parquet" else p.parent / f"{p.name}.parquet"
        if p.is_dir():
            pq = p.parent / f"{p.name}.parquet"
        if not pq.is_file():
            continue
        try:
            import pyarrow.parquet as pqmod
            t = pqmod.read_table(pq, columns=["global_step"])
            col = t.column("global_step")
            if len(col):
                # max over column (not just last row)
                best = max(best, max(int(x.as_py()) for x in col if x.as_py() is not None))
        except Exception:
            pass
    return best

for pat in sys.argv[1:]:
    print(step_for(pat))
PY
}

n_in_queue() {
  squeue -u jdezoort -n "$JOB_NAME" -h 2>/dev/null | wc -l
}

task_in_queue() {
  local task=$1
  squeue -u jdezoort -n "$JOB_NAME" -h -o '%i' 2>/dev/null | grep -E "_${task}$|\[${task}\]" >/dev/null
}

# Drop PD jobs stuck on ReqNodeNotAvail so they don't eat the concurrency budget.
# Also drop long-pending jobs with Reason=None that never start (>3 min).
reap_stuck_pending() {
  local line jid state reason age
  while read -r line; do
    jid=$(awk '{print $1}' <<<"$line")
    state=$(awk '{print $2}' <<<"$line")
    reason=$(awk '{print $3}' <<<"$line")
    age=$(awk '{print $4}' <<<"$line")
    [[ -z "$jid" ]] && continue
    if [[ "$reason" == *ReqNodeNotAvail* || "$reason" == *UnavailableNodes* ]]; then
      echo "$(date -Iseconds) scancel stuck PD $jid reason=$reason" | tee -a "$LOG"
      scancel "$jid" 2>/dev/null || true
    fi
  done < <(squeue -u jdezoort -n "$JOB_NAME" -t PD -h -o '%i %T %R %M' 2>/dev/null)

  # Opportunistically scancel any leftover WFC zombies that steal gputest.
  local wfc
  wfc=$(squeue -u jdezoort -h -o '%i %j' 2>/dev/null | awk '/L4D256-wfc/{print $1}')
  if [[ -n "$wfc" ]]; then
    echo "$(date -Iseconds) scancel WFC zombies: $wfc" | tee -a "$LOG"
    scancel $wfc 2>/dev/null || true
  fi
}

echo "$(date -Iseconds) m11 watchdog start TARGET=$TARGET MAX_CONCURRENT=$MAX_CONCURRENT MAX_STEPS=$MAX_STEPS EPOCHS=$EPOCHS SLEEP=$SLEEP_SEC" | tee -a "$LOG"
LAST_PLOT_DONE=""

while true; do
  for stopf in "${STOP_FILES[@]}"; do
    if [[ -f "$stopf" ]]; then
      echo "$(date -Iseconds) STOP $stopf; exiting" | tee -a "$LOG"
      exit 0
    fi
  done

  reap_stuck_pending

  mapfile -t STEPS < <(all_steps)
  incomplete=()
  done_n=0
  for task in $(seq 0 14); do
    step=${STEPS[$task]:-0}
    step=${step:-0}
    read -r width eta <<<"$(cell_meta "$task")"
    if (( step >= TARGET )); then
      done_n=$((done_n + 1))
    else
      incomplete+=("$task")
      echo "$(date -Iseconds) pending task=$task D=$width eta=$eta step=$step" | tee -a "$LOG"
    fi
  done
  echo "$(date -Iseconds) done=$done_n/15 queue=$(n_in_queue) incomplete=${#incomplete[@]}" | tee -a "$LOG"

  if [[ "${done_n}" != "${LAST_PLOT_DONE:-}" ]]; then
    (
      cd "$REPO"
      "$PYTHON" scans/jets/plot_m11_eta_transfer.py --target "$MAX_STEPS"
      "$PYTHON" scans/jets/plot_graph_construction_nature.py --m11-target "$MAX_STEPS"
    ) >>"$LOG" 2>&1 || echo "$(date -Iseconds) plot refresh failed" | tee -a "$LOG"
    LAST_PLOT_DONE=$done_n
  fi

  if (( done_n >= 15 )); then
    echo "$(date -Iseconds) ALL m11 arms >= $TARGET; freezing final figure" | tee -a "$LOG"
    (
      cd "$REPO"
      "$PYTHON" scans/jets/plot_m11_eta_transfer.py --target "$MAX_STEPS"
      "$PYTHON" scans/jets/plot_graph_construction_nature.py --m11-target "$MAX_STEPS"
    ) >>"$LOG" 2>&1 || true
    echo "M11_COMPLETE $(date -Iseconds)" | tee "$CACHE/M11_COMPLETE"
    exit 0
  fi

  queued=$(n_in_queue)
  slots=$((MAX_CONCURRENT - queued))
  if (( slots > 0 )); then
    submitted=0
    for task in "${incomplete[@]}"; do
      (( submitted >= slots )) && break
      if task_in_queue "$task"; then
        continue
      fi
      out=$(sbatch --parsable --job-name="$JOB_NAME" --array="$task" \
        --export=ALL,MAX_STEPS="$MAX_STEPS",EPOCHS="$EPOCHS" \
        "$SLURM" 2>&1) || {
        echo "$(date -Iseconds) sbatch task=$task failed: $out" | tee -a "$LOG"
        continue
      }
      echo "$(date -Iseconds) submitted task=$task job=$out" | tee -a "$LOG"
      submitted=$((submitted + 1))
    done
  fi

  sleep "$SLEEP_SEC"
done
