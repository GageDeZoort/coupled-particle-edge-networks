#!/bin/bash
# Pool watchdog: AdamW + const LR · t_epoch sweep @ L4/D256 η₀=0.1 → 40k.
#
# 2h a100 slots; Lightning resumes last.ckpt.
# n_train = 100M (prescribed D for t_epoch λ); 40k×256 presentations unique.
#
#   bash scans/jets/start_adamw_tepoch_D256_1a100_watchdogs.sh
#   touch ~/.cache/jc_aw_tepoch_watchdogs/STOP
#
set -uo pipefail

ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law
REPO=/home/jdezoort/coupled-particle-edge-networks
SLURM="$REPO/scans/jets/scaling_law_L4_adamw_tepoch_D256_1a100_s40k.slurm"
JOB_NAME="${JOB_NAME:-jc-aw-tepoch}"
TARGET="${TARGET:-40000}"
MAX_STEPS="${MAX_STEPS:-40000}"
MAX_A100="${MAX_A100:-4}"
SLEEP_SEC="${SLEEP_SEC:-90}"
N_TASKS=7
PYTHON="${PYTHON:-/home/jdezoort/.conda/envs/mamba-env/bin/python}"
CACHE="${HOME}/.cache/jc_aw_tepoch_watchdogs"
LOG_DIR="$ROOT/logs"
mkdir -p "$CACHE" "$LOG_DIR"
LOG="$CACHE/watchdog_aw_tepoch.log"
STOP_FILES=("$CACHE/STOP" "$LOG_DIR/STOP_aw_tepoch")
LOCK="$CACHE/watchdog_aw_tepoch.lock"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "$(date -Iseconds) another aw-tepoch watchdog holds $LOCK; exiting"
  exit 0
fi

T_EPOCHS=(0.1 0.5 1 5 10 25 50)
BATCH=256
N_GPUS=1
D_HEAD=32
KNN_K=7
R_TAG=0p1
RUN_TAG_BASE="aw-tepoch-n100M-b${BATCH}-g${N_GPUS}-s${MAX_STEPS}-dh${D_HEAD}-k${KNN_K}-R${R_TAG}"
SWEEP="$ROOT/jetclass_output/adamw/sweep_lr"

cell_meta() {
  echo "${T_EPOCHS[$1]}"
}

t_tag() {
  python3 -c "e=float('$1'); print(f'{e:g}'.replace('.','p'))"
}

run_tag_for() {
  local te=$1
  echo "aw-tepoch-t$(t_tag "$te")-n100M-b${BATCH}-g${N_GPUS}-s${MAX_STEPS}-dh${D_HEAD}-k${KNN_K}-R${R_TAG}"
}

all_steps() {
  "$PYTHON" - "$SWEEP" "$BATCH" "$N_GPUS" "$MAX_STEPS" "$D_HEAD" "$KNN_K" "$R_TAG" "$N_TASKS" <<'PY'
import sys
from pathlib import Path

sweep = sys.argv[1]
batch, n_gpus, max_steps, d_head, knn_k, r_tag = sys.argv[2:8]
n_tasks = int(sys.argv[8])
t_epochs = [0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0]
steps = [0] * n_tasks

def best_step(path: Path) -> int:
    best = 0
    pq = path if path.suffix == ".parquet" else path.parent / f"{path.name}.parquet"
    if pq.is_file():
        try:
            import pyarrow.parquet as pqmod
            t = pqmod.read_table(pq, columns=["global_step"])
            col = t.column("global_step")
            if len(col):
                best = max(int(x.as_py()) for x in col if x.as_py() is not None)
        except Exception:
            pass
    run_dir = path if path.is_dir() else (
        path.parent / path.name.replace(".parquet", "") if path.suffix == ".parquet" else None
    )
    if run_dir is not None and run_dir.is_dir():
        try:
            import torch
            for ckpt in run_dir.glob("last*.ckpt"):
                try:
                    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
                    best = max(best, int(payload.get("global_step", 0) or 0))
                except Exception:
                    pass
        except Exception:
            pass
    return best

for i, t in enumerate(t_epochs):
    tt = f"{t:g}".replace(".", "p")
    tag = f"aw-tepoch-t{tt}-n100M-b{batch}-g{n_gpus}-s{max_steps}-dh{d_head}-k{knn_k}-R{r_tag}"
    best = 0
    seen = set()
    for p in Path(sweep).glob(f"*{tag}*"):
        name = p.name.replace(".parquet", "")
        if name in seen:
            continue
        seen.add(name)
        best = max(best, best_step(p))
    steps[i] = best

for s in steps:
    print(s)
PY
}

n_a100() {
  timeout 30 squeue -u jdezoort -n "$JOB_NAME" -h 2>/dev/null | wc -l
}

task_in_queue() {
  local task=$1
  timeout 30 squeue -u jdezoort -n "$JOB_NAME" -h -o '%i' 2>/dev/null \
    | grep -E "_${task}$|\[${task}\]" >/dev/null
}

reap_stuck_pending() {
  local line jid reason
  while read -r line; do
    jid=$(awk '{print $1}' <<<"$line")
    reason=$(awk '{print $3}' <<<"$line")
    [[ -z "$jid" ]] && continue
    if [[ "$reason" == *ReqNodeNotAvail* || "$reason" == *UnavailableNodes* ]]; then
      echo "$(date -Iseconds) scancel stuck PD $jid reason=$reason" | tee -a "$LOG"
      timeout 30 scancel "$jid" 2>/dev/null || true
    fi
  done < <(timeout 30 squeue -u jdezoort -n "$JOB_NAME" -t PD -h -o '%i %T %R %M' 2>/dev/null)
}

echo "$(date -Iseconds) aw-tepoch watchdog start TARGET=$TARGET MAX_A100=$MAX_A100 RUN_TAG_BASE=$RUN_TAG_BASE SLEEP=$SLEEP_SEC" | tee -a "$LOG"

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
  status=""
  for task in $(seq 0 $((N_TASKS - 1))); do
    step=${STEPS[$task]:-0}
    step=${step:-0}
    te=$(cell_meta "$task")
    if (( step >= TARGET )); then
      done_n=$((done_n + 1))
      status+=" t${te}=done(${step})"
    else
      incomplete+=("$task")
      status+=" t${te}=${step}/${TARGET}"
    fi
  done
  na=$(n_a100)
  echo "$(date -Iseconds) done=$done_n/$N_TASKS aw_q=$na$status" | tee -a "$LOG"

  if (( done_n >= N_TASKS )); then
    (
      cd "$REPO"
      "$PYTHON" scans/jets/refresh_adamw_tepoch_D256_s40k_canvas.py || true
    ) >>"$LOG" 2>&1 || true
    echo "$(date -Iseconds) ALL aw-tepoch cells >= $TARGET; exiting" | tee -a "$LOG"
    echo "AW_TEPOCH_COMPLETE $(date -Iseconds)" | tee "$CACHE/AW_TEPOCH_COMPLETE"
    exit 0
  fi

  # Recompute free slots after each submit (do not gate on a stale "submitted"
  # counter — that under-fills when n_a100 rises mid-loop).
  for task in "${incomplete[@]}"; do
    na=$(n_a100)
    slots=$((MAX_A100 - na))
    (( slots <= 0 )) && break
    if task_in_queue "$task"; then
      continue
    fi
    out=$(timeout 90 sbatch --parsable --job-name="$JOB_NAME" --array="$task" \
      --export=ALL,MAX_STEPS="$MAX_STEPS" \
      "$SLURM" 2>&1) || {
      echo "$(date -Iseconds) sbatch task=$task failed: $out" | tee -a "$LOG"
      continue
    }
    echo "$(date -Iseconds) submitted task=$task t_epoch=$(cell_meta "$task") job=$out" | tee -a "$LOG"
  done

  (
    cd "$REPO"
    "$PYTHON" scans/jets/refresh_adamw_tepoch_D256_s40k_canvas.py || true
  ) >>"$LOG" 2>&1 &

  sleep "$SLEEP_SEC"
done
