#!/bin/bash
# Pool watchdog for Plot-2 R★×shard×seed grid on gpu-test (k★=12, B_global=384).
#
# Keeps total gputest occupancy ≤ MAX_CONCURRENT (shared with jc-gk*),
# submitting jc-gR-gt array tasks until every cell reaches TARGET steps.
#
#   bash scans/jets/start_graph_R_gputest_watchdogs.sh
#
set -uo pipefail

ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law
REPO=/home/jdezoort/coupled-particle-edge-networks
SLURM="$REPO/scans/jets/scaling_law_L4_graph_R_shard_gputest.slurm"
JOB_NAME="${JOB_NAME:-jc-gR-gt}"
TARGET="${TARGET:-950}"
MAX_STEPS="${MAX_STEPS:-1000}"
KNN_K="${KNN_K:-12}"
MAX_CONCURRENT="${MAX_CONCURRENT:-3}"
SLEEP_SEC="${SLEEP_SEC:-30}"
N_TASKS=42
PYTHON="${PYTHON:-/home/jdezoort/.conda/envs/mamba-env/bin/python}"
CACHE="${HOME}/.cache/jc_gR_watchdogs"
LOG_DIR="$ROOT/logs"
mkdir -p "$CACHE" "$LOG_DIR"
LOG="$CACHE/watchdog_gR.log"
STOP_FILES=("$CACHE/STOP" "$LOG_DIR/STOP_gR")
LOCK="$CACHE/watchdog_gR.lock"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "$(date -Iseconds) another gR watchdog holds $LOCK; exiting"
  exit 0
fi

# 0.075 / 0.125 appended so tasks 0–29 keep completed indices.
RADII=(0.05 0.1 0.15 0.2 0.25 0.075 0.125)
N_SHARD=3
N_INIT=2
N_PER_R=$((N_SHARD * N_INIT))
BATCH=96
N_GPUS=4

# Honest coverage: shard0 may be legacy v6gt; shard>0 must be v6gt2
# (pre-fix v6gt shard>0 were ram-subset clones — never count them as done).
all_steps() {
  "$PYTHON" - "$ROOT/jetclass_output/adam/sweep_lr" "$BATCH" "$N_GPUS" "$MAX_STEPS" "$KNN_K" "$N_TASKS" <<'PY'
import re, sys
from pathlib import Path

sweep, batch, ngpu, max_steps, knn_k, n_tasks = sys.argv[1:7]
n_tasks = int(n_tasks)
knn_k = int(knn_k)
radii = [0.05, 0.1, 0.15, 0.2, 0.25, 0.075, 0.125]
n_per_r = 6
steps = [0] * n_tasks

def r_tag(r: float) -> str:
    return f"{r:g}".replace(".", "p")

r_to_idx = {r_tag(r): i for i, r in enumerate(radii)}
pat = re.compile(
    rf"knn{knn_k}.*graph-R-b{batch}-g{ngpu}-s{max_steps}-k{knn_k}-"
    rf"R([0-9p]+)-shard(\d+)-seed(\d+)-(v6gt2|v6gt)(?:\.parquet)?$"
)
sweep_p = Path(sweep)
pref = {"v6gt2": 2, "v6gt": 1}
best_pref = [0] * n_tasks
for pq in list(sweep_p.glob("*graph-R*v6gt2*.parquet")) + list(
    sweep_p.glob("*graph-R*v6gt*.parquet")
):
    m = pat.search(pq.name)
    if not m:
        continue
    r_raw, shard, seed, tag = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
    if shard > 0 and tag != "v6gt2":
        continue
    if r_raw not in r_to_idx:
        continue
    task = r_to_idx[r_raw] * n_per_r + shard * 2 + seed
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

# Shared gputest MaxJobs budget (gk / gk16 / gR all count).
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

echo "$(date -Iseconds) gR watchdog start k=$KNN_K TARGET=$TARGET MAX_CONCURRENT=$MAX_CONCURRENT MAX_STEPS=$MAX_STEPS SLEEP=$SLEEP_SEC" | tee -a "$LOG"
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
  echo "$(date -Iseconds) done=$done_n/$N_TASKS gR_queue=$(n_in_queue) gputest=$gt incomplete=${#incomplete[@]} next=${incomplete[0]:-none}" | tee -a "$LOG"

  if [[ "${done_n}" != "${LAST_PLOT_DONE:-}" ]]; then
    (
      cd "$REPO"
      "$PYTHON" scans/jets/plot_graph_construction_nature.py
    ) >>"$LOG" 2>&1 || echo "$(date -Iseconds) plot refresh failed" | tee -a "$LOG"
    LAST_PLOT_DONE=$done_n
  fi

  if (( done_n >= N_TASKS )); then
    echo "$(date -Iseconds) ALL gR cells >= $TARGET; freezing final figure" | tee -a "$LOG"
    (
      cd "$REPO"
      "$PYTHON" scans/jets/plot_graph_construction_nature.py
    ) >>"$LOG" 2>&1 || true
    echo "GR_COMPLETE $(date -Iseconds)" | tee "$CACHE/GR_COMPLETE"
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
        --export=ALL,MAX_STEPS="$MAX_STEPS",KNN_K="$KNN_K" \
        "$SLURM" 2>&1) || {
        echo "$(date -Iseconds) sbatch task=$task failed: $out" | tee -a "$LOG"
        continue
      }
      echo "$(date -Iseconds) submitted task=$task job=$out" | tee -a "$LOG"
      submitted=$((submitted + 1))
      # Refresh occupancy so we don't overshoot MaxJobs in one loop.
      gt=$(n_gputest)
      slots=$((MAX_CONCURRENT - gt))
      (( slots <= 0 )) && break
    done
  fi

  sleep "$SLEEP_SEC"
done
