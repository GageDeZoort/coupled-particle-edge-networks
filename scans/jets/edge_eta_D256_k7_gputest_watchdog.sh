#!/bin/bash
# Pool watchdog for panel-(c) edge η₀ @ L4/D256, (k★,R★)=(7,0.1), 40k steps.
#
# Keeps ≤ MAX_GPTEST concurrent gputest array parents (MaxJobsPU≈3) and
# resubmits unfinished η cells until each reaches TARGET. Lightning resumes
# last.ckpt under a fixed run_tag. n_train = 40k×256 = 10.24M unique jets.
#
#   bash scans/jets/start_edge_eta_D256_k7_gputest_watchdogs.sh
#   touch ~/.cache/jc_edge_k7_watchdogs/STOP   # stop
#
set -uo pipefail

ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law
REPO=/home/jdezoort/coupled-particle-edge-networks
SLURM="$REPO/scans/jets/scaling_law_L4_edge_eta_D256_k7_gputest_s20k.slurm"
JOB_NAME="${JOB_NAME:-jc-edge-k7}"
TARGET="${TARGET:-40000}"
MAX_STEPS="${MAX_STEPS:-40000}"
MAX_GPTEST="${MAX_GPTEST:-3}"
SLEEP_SEC="${SLEEP_SEC:-60}"
N_TASKS=9
PYTHON="${PYTHON:-/home/jdezoort/.conda/envs/mamba-env/bin/python}"
CACHE="${HOME}/.cache/jc_edge_k7_watchdogs"
LOG_DIR="$ROOT/logs"
mkdir -p "$CACHE" "$LOG_DIR"
LOG="$CACHE/watchdog_edge_k7.log"
STOP_FILES=("$CACHE/STOP" "$LOG_DIR/STOP_edge_k7")
LOCK="$CACHE/watchdog_edge_k7.lock"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "$(date -Iseconds) another edge-k7 watchdog holds $LOCK; exiting"
  exit 0
fi

ETAS=(0.01 0.025 0.05 0.075 0.1 0.25 0.5 0.75 1.0)
BATCH=64
N_GPUS=4
D_HEAD=32
KNN_K=7
R_TAG=0p1
RUN_TAG="edge-b${BATCH}-g${N_GPUS}-s${MAX_STEPS}-dh${D_HEAD}-k${KNN_K}-R${R_TAG}"

cell_meta() {
  local task=$1
  echo "${ETAS[$task]}"
}

eta_tag() {
  python3 -c "e=float('$1'); print(f'{e:g}'.replace('.','p'))"
}

all_steps() {
  "$PYTHON" - "$ROOT/jetclass_output/adam/sweep_lr" "$RUN_TAG" "$N_TASKS" <<'PY'
import glob, re, sys
from pathlib import Path

sweep, run_tag, n_tasks = sys.argv[1], sys.argv[2], int(sys.argv[3])
etas = [0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0]
eta_to_idx = {f"{e:g}".replace(".", "p"): i for i, e in enumerate(etas)}
steps = [0] * n_tasks
pat = re.compile(
    rf"capen-llama-att_4_256_(?P<eta>[0-9p]+)_.*{re.escape(run_tag)}"
)

def best_step(path: Path) -> int:
    best = 0
    pq = path if path.suffix == ".parquet" else path.parent / f"{path.name}.parquet"
    if pq.is_file():
        try:
            import pyarrow.parquet as pqmod
            t = pqmod.read_table(pq, columns=["global_step"])
            col = t.column("global_step")
            if len(col):
                best = max(
                    int(x.as_py()) for x in col if x.as_py() is not None
                )
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

seen = set()
# Run-tag is hyphen-joined (…-seed0-edge-b64-…), not underscore-prefixed.
for p in Path(sweep).glob(f"*{run_tag}*"):
    name = p.name.replace(".parquet", "")
    if name in seen:
        continue
    seen.add(name)
    m = pat.search(name)
    if m is None:
        continue
    et = m.group("eta")
    if et not in eta_to_idx:
        continue
    task = eta_to_idx[et]
    steps[task] = max(steps[task], best_step(p))

for s in steps:
    print(s)
PY
}

n_gputest() {
  timeout 30 squeue -u jdezoort -h -o '%P' 2>/dev/null | grep -c '^gputest$' || true
}

n_in_queue() {
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

echo "$(date -Iseconds) edge-k7 watchdog start TARGET=$TARGET MAX_GPTEST=$MAX_GPTEST RUN_TAG=$RUN_TAG SLEEP=$SLEEP_SEC" | tee -a "$LOG"
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
  status=""
  for task in $(seq 0 $((N_TASKS - 1))); do
    step=${STEPS[$task]:-0}
    step=${step:-0}
    eta=$(cell_meta "$task")
    if (( step >= TARGET )); then
      done_n=$((done_n + 1))
      status+=" e${eta}=done(${step})"
    else
      incomplete+=("$task")
      status+=" e${eta}=${step}/${TARGET}"
    fi
  done
  gt=$(n_gputest)
  echo "$(date -Iseconds) done=$done_n/$N_TASKS edge_q=$(n_in_queue) gputest=$gt$status" | tee -a "$LOG"

  if (( done_n >= N_TASKS )); then
    (
      cd "$REPO"
      "$PYTHON" scans/jets/refresh_edge_eta_D256_k7_s20k_canvas.py || true
      "$PYTHON" scans/jets/plot_graph_construction_nature.py || true
    ) >>"$LOG" 2>&1 || true
    echo "$(date -Iseconds) ALL edge η cells >= $TARGET; exiting" | tee -a "$LOG"
    echo "EDGE_K7_COMPLETE $(date -Iseconds)" | tee "$CACHE/EDGE_K7_COMPLETE"
    exit 0
  fi

  # Submit before canvas/plot so a slow refresh cannot starve the pool.
  slots=$((MAX_GPTEST - gt))
  if (( slots > 0 )); then
    submitted=0
    for task in "${incomplete[@]}"; do
      (( submitted >= slots )) && break
      if task_in_queue "$task"; then
        continue
      fi
      out=$(timeout 90 sbatch --parsable --job-name="$JOB_NAME" --array="$task" \
        --export=ALL,MAX_STEPS="$MAX_STEPS" \
        "$SLURM" 2>&1) || {
        echo "$(date -Iseconds) sbatch task=$task failed: $out" | tee -a "$LOG"
        continue
      }
      echo "$(date -Iseconds) submitted task=$task eta=$(cell_meta "$task") job=$out" | tee -a "$LOG"
      submitted=$((submitted + 1))
      gt=$(n_gputest)
      slots=$((MAX_GPTEST - gt))
      (( slots <= 0 )) && break
    done
  fi

  # Non-blocking canvas refresh; figure only when a cell newly completes.
  (
    cd "$REPO"
    "$PYTHON" scans/jets/refresh_edge_eta_D256_k7_s20k_canvas.py || true
    if [[ "${done_n}" != "${LAST_PLOT_DONE:-}" ]]; then
      "$PYTHON" scans/jets/plot_graph_construction_nature.py || true
    fi
  ) >>"$LOG" 2>&1 &
  LAST_PLOT_DONE=$done_n

  sleep "$SLEEP_SEC"
done
