#!/bin/bash
# Pool watchdog for Plot-1 kNN×shard×seed grid on gpu-test (B_global=384, 1000 steps).
#
# Keeps up to MAX_CONCURRENT jc-gk-gt array tasks in the queue until every cell
# reaches TARGET steps (default 950 ≈ 1000).
#
#   bash scans/jets/start_graph_k_gputest_watchdogs.sh
#
set -uo pipefail

ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law
REPO=/home/jdezoort/coupled-particle-edge-networks
SLURM="$REPO/scans/jets/scaling_law_L4_graph_k_shard_gputest.slurm"
JOB_NAME="${JOB_NAME:-jc-gk-gt}"
TARGET="${TARGET:-950}"
MAX_STEPS="${MAX_STEPS:-1000}"
MAX_CONCURRENT="${MAX_CONCURRENT:-3}"
SLEEP_SEC="${SLEEP_SEC:-30}"
N_TASKS=42
PYTHON="${PYTHON:-/home/jdezoort/.conda/envs/mamba-env/bin/python}"
CACHE="${HOME}/.cache/jc_gk_watchdogs"
LOG_DIR="$ROOT/logs"
mkdir -p "$CACHE" "$LOG_DIR"
LOG="$CACHE/watchdog_gk.log"
STOP_FILES=("$CACHE/STOP" "$LOG_DIR/STOP_gk")
LOCK="$CACHE/watchdog_gk.lock"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "$(date -Iseconds) another gk watchdog holds $LOCK; exiting"
  exit 0
fi

KS=(2 4 6 8 10 12 14)
N_SHARD=3
N_INIT=2
N_PER_K=$((N_SHARD * N_INIT))
BATCH=96
N_GPUS=4
STAR_R=0.125
R_TAG=0p125  # fixed; avoid 42× python3 -c per loop

cell_meta() {
  local task=$1
  local k_idx=$((task / N_PER_K))
  local rem=$((task % N_PER_K))
  local shard=$((rem / N_INIT))
  local seed=$((rem % N_INIT))
  echo "${KS[$k_idx]} $shard $seed"
}

# Honest coverage: shard0 may be legacy v6gt; shard>0 must be v6gt2
# (pre-fix v6gt shard>0 were ram-subset clones — never count them as done).
all_steps() {
  "$PYTHON" - "$ROOT/jetclass_output/adam/sweep_lr" "$BATCH" "$N_GPUS" "$MAX_STEPS" "$R_TAG" "$N_TASKS" <<'PY'
import re, sys
from pathlib import Path

sweep, batch, ngpu, max_steps, r_tag, n_tasks = sys.argv[1:7]
n_tasks = int(n_tasks)
ks = [2, 4, 6, 8, 10, 12, 14]
n_per_k = 6
steps = [0] * n_tasks
prefix = f"graph-k-b{batch}-g{ngpu}-s{max_steps}-R{r_tag}-"
pat = re.compile(
    rf"knn(\d+).*{re.escape(prefix)}shard(\d+)-seed(\d+)-(v6gt2|v6gt)(?:\.parquet)?$"
)
sweep_p = Path(sweep)
pref = {"v6gt2": 2, "v6gt": 1}
best_pref = [0] * n_tasks
for pq in list(sweep_p.glob("*graph-k*v6gt2*.parquet")) + list(
    sweep_p.glob("*graph-k*v6gt*.parquet")
):
    m = pat.search(pq.name)
    if not m:
        continue
    k, shard, seed, tag = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
    # shard>0: only true streaming re-runs (v6gt2)
    if shard > 0 and tag != "v6gt2":
        continue
    if k not in ks:
        continue
    k_idx = ks.index(k)
    task = k_idx * n_per_k + shard * 2 + seed
    if task < 0 or task >= n_tasks:
        continue
    p = pref.get(tag, 0)
    try:
        import pyarrow.parquet as pqmod
        t = pqmod.read_table(pq, columns=["global_step"])
        col = t.column("global_step")
        if len(col):
            best = max(int(x.as_py()) for x in col if x.as_py() is not None)
            if p > best_pref[task] or (p == best_pref[task] and best > steps[task]):
                steps[task] = best
                best_pref[task] = p
    except Exception:
        pass
for s in steps:
    print(s)
PY
}

# Shared gputest MaxJobs with jc-gR-gt / jc-gk16-gt.
n_gputest() {
  squeue -u jdezoort -h -o '%P' 2>/dev/null | grep -c '^gputest$' || true
}

n_in_queue() {
  squeue -u jdezoort -n "$JOB_NAME" -h 2>/dev/null | wc -l
}

task_in_queue() {
  local task=$1
  squeue -u jdezoort -n "$JOB_NAME" -h -o '%i' 2>/dev/null | grep -E "_${task}$|\[${task}\]" >/dev/null
}

reap_stuck_pending() {
  local line jid reason
  while read -r line; do
    jid=$(awk '{print $1}' <<<"$line")
    reason=$(awk '{print $3}' <<<"$line")
    [[ -z "$jid" ]] && continue
    if [[ "$reason" == *ReqNodeNotAvail* || "$reason" == *UnavailableNodes* ]]; then
      echo "$(date -Iseconds) scancel stuck PD $jid reason=$reason" | tee -a "$LOG"
      scancel "$jid" 2>/dev/null || true
    fi
  done < <(squeue -u jdezoort -n "$JOB_NAME" -t PD -h -o '%i %T %R %M' 2>/dev/null)
}

echo "$(date -Iseconds) gk watchdog start TARGET=$TARGET MAX_CONCURRENT=$MAX_CONCURRENT MAX_STEPS=$MAX_STEPS SLEEP=$SLEEP_SEC" | tee -a "$LOG"
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
  for task in $(seq 0 $((N_TASKS - 1))); do
    step=${STEPS[$task]:-0}
    step=${step:-0}
    if (( step >= TARGET )); then
      done_n=$((done_n + 1))
    else
      incomplete+=("$task")
    fi
  done
  gt=$(n_gputest)
  echo "$(date -Iseconds) done=$done_n/$N_TASKS gk_queue=$(n_in_queue) gputest=$gt incomplete=${#incomplete[@]} next=${incomplete[0]:-none}" | tee -a "$LOG"

  if [[ "${done_n}" != "${LAST_PLOT_DONE:-}" ]]; then
    (
      cd "$REPO"
      "$PYTHON" scans/jets/plot_graph_construction_nature.py
    ) >>"$LOG" 2>&1 || echo "$(date -Iseconds) plot refresh failed" | tee -a "$LOG"
    LAST_PLOT_DONE=$done_n
  fi

  if (( done_n >= N_TASKS )); then
    echo "$(date -Iseconds) ALL gk cells >= $TARGET; freezing final figure" | tee -a "$LOG"
    (
      cd "$REPO"
      "$PYTHON" scans/jets/plot_graph_construction_nature.py
    ) >>"$LOG" 2>&1 || true
    echo "GK_COMPLETE $(date -Iseconds)" | tee "$CACHE/GK_COMPLETE"
    exit 0
  fi

  slots=$((MAX_CONCURRENT - gt))
  if (( slots > 0 )); then
    submitted=0
    for task in "${incomplete[@]}"; do
      (( submitted >= slots )) && break
      if task_in_queue "$task"; then
        continue
      fi
      out=$(sbatch --parsable --job-name="$JOB_NAME" --array="$task" \
        --export=ALL,MAX_STEPS="$MAX_STEPS" \
        "$SLURM" 2>&1) || {
        echo "$(date -Iseconds) sbatch task=$task failed: $out" | tee -a "$LOG"
        continue
      }
      echo "$(date -Iseconds) submitted task=$task job=$out" | tee -a "$LOG"
      submitted=$((submitted + 1))
      gt=$(n_gputest)
      slots=$((MAX_CONCURRENT - gt))
      (( slots <= 0 )) && break
    done
  fi

  sleep "$SLEEP_SEC"
done
