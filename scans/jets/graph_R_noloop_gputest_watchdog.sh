#!/bin/bash
# Greedy gputest fill for panel (b′) noloop R★-sweep @ k★=7 (v2cpNL).
#
#   bash scans/jets/start_graph_R_noloop_gputest_watchdogs.sh
#
set -uo pipefail

ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law
REPO=/home/jdezoort/coupled-particle-edge-networks
SLURM="$REPO/scans/jets/scaling_law_L4_graph_R_noloop_k7_gputest_v2cp.slurm"
JOB_NAME="${JOB_NAME:-jc-gR-nl-gt}"
A100_JOB="${A100_JOB:-$(cat "$REPO/scans/jets/graph_R_noloop_v2cp_jobid.txt" 2>/dev/null || echo)}"
TARGET="${TARGET:-2495}"
MAX_STEPS="${MAX_STEPS:-2500}"
MAX_CONCURRENT="${MAX_CONCURRENT:-3}"
SLEEP_SEC="${SLEEP_SEC:-30}"
N_TASKS=30
PYTHON="${PYTHON:-/home/jdezoort/.conda/envs/mamba-env/bin/python}"
CACHE="${HOME}/.cache/jc_gR_noloop_gt"
LOG_DIR="$ROOT/logs"
mkdir -p "$CACHE" "$LOG_DIR"
LOG="$CACHE/watchdog.log"
STOP_FILES=("$CACHE/STOP" "$LOG_DIR/STOP_gR_noloop_gt")
LOCK="$CACHE/watchdog.lock"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "$(date -Iseconds) another gR-noloop-gt watchdog holds $LOCK; exiting"
  exit 0
fi

RADII=(0.06 0.08 0.1 0.12 0.14)
N_SHARD=3
N_INIT=2
N_PER_R=$((N_SHARD * N_INIT))
BATCH=128
N_GPUS=1
KNN_K=7

all_steps() {
  "$PYTHON" - "$ROOT/jetclass_output/adam/sweep_lr" "$BATCH" "$N_GPUS" "$MAX_STEPS" "$KNN_K" "$N_TASKS" <<'PY'
import re, sys
from pathlib import Path

sweep, batch, ngpu, max_steps, knn_k, n_tasks = sys.argv[1:7]
n_tasks = int(n_tasks)
knn_k = int(knn_k)
radii = [0.06, 0.08, 0.1, 0.12, 0.14]
n_per_r = 6
steps = [0] * n_tasks
prefix = f"graph-R-noloop-b{batch}-g{ngpu}-s{max_steps}-k{knn_k}-"
pat = re.compile(
    rf"gstar([0-9p]+).*{re.escape(prefix)}shard(\d+)-seed(\d+)-v2cpNL(?:\.parquet)?$"
)
sweep_p = Path(sweep)
for pq in sweep_p.glob("*graph-R-noloop*v2cpNL.parquet"):
    m = pat.search(pq.name)
    if not m:
        continue
    r_s = m.group(1).replace("p", ".")
    try:
        r = float(r_s)
    except ValueError:
        continue
    if r not in radii:
        continue
    shard, seed = int(m.group(2)), int(m.group(3))
    r_idx = radii.index(r)
    task = r_idx * n_per_r + shard * 2 + seed
    if task < 0 or task >= n_tasks:
        continue
    try:
        import pyarrow.parquet as pqmod
        t = pqmod.read_table(pq, columns=["global_step"])
        col = t.column("global_step")
        if len(col):
            best = max(int(x.as_py()) for x in col if x.as_py() is not None)
            if best > steps[task]:
                steps[task] = best
    except Exception:
        pass
for s in steps:
    print(s)
PY
}

SQ() { timeout 25 squeue "$@" 2>/dev/null || true; }
SC() { timeout 20 scancel "$@" 2>/dev/null || true; }
SB() { timeout 45 sbatch "$@" 2>/dev/null || true; }

n_gputest() {
  SQ -u jdezoort -p gputest -h -o "%i" \
    | sed 's/_.*//' | sort -u | wc -l
}

n_in_queue() {
  SQ -u jdezoort -n "$JOB_NAME" -h | wc -l
}

task_in_gt_queue() {
  local task=$1
  SQ -u jdezoort -n "$JOB_NAME" -h -o '%i' \
    | grep -E "_${task}$|\[${task}\]" >/dev/null
}

task_running_on_a100() {
  local task=$1
  [[ -z "$A100_JOB" ]] && return 1
  SQ -j "$A100_JOB" -t R -h -o '%i' \
    | grep -E "_${task}$" >/dev/null
}

cancel_a100_pending() {
  local task=$1
  [[ -z "$A100_JOB" ]] && return 0
  local line state
  line=$(SQ -j "${A100_JOB}_${task}" -h -o '%i %T' | head -1)
  [[ -z "$line" ]] && return 0
  state=$(awk '{print $2}' <<<"$line")
  if [[ "$state" == "PENDING" || "$state" == "PD" ]]; then
    SC "${A100_JOB}_${task}"
    echo "$(date -Iseconds) scancel pending a100 ${A100_JOB}_${task}" | tee -a "$LOG"
  fi
}

gt_active_tasks() {
  SQ -u jdezoort -n "$JOB_NAME" -h -o '%i' \
    | while read -r jid; do
        if [[ "$jid" == *_\[* ]]; then
          local body="${jid##*_\[}"
          body="${body%\]*}"
          body="${body%%\%*}"
          if [[ "$body" == *-* ]]; then
            local lo=${body%-*} hi=${body#*-} i
            for ((i=lo; i<=hi; i++)); do echo "$i"; done
          else
            echo "$body"
          fi
        elif [[ "$jid" == *_* ]]; then
          echo "${jid##*_}"
        fi
      done | sort -nu
}

sync_cancel_a100_for_gputest() {
  local task i step
  while read -r task; do
    [[ -z "$task" ]] && continue
    cancel_a100_pending "$task"
  done < <(gt_active_tasks)
  for i in $(seq 0 $((N_TASKS - 1))); do
    step=${STEPS[$i]:-0}
    step=${step:-0}
    if (( step >= TARGET )); then
      cancel_a100_pending "$i"
    fi
  done
}

reap_stuck_pending() {
  local line jid reason
  while read -r line; do
    jid=$(awk '{print $1}' <<<"$line")
    reason=$(awk '{print $3}' <<<"$line")
    [[ -z "$jid" ]] && continue
    if [[ "$reason" == *ReqNodeNotAvail* || "$reason" == *UnavailableNodes* ]]; then
      echo "$(date -Iseconds) scancel stuck PD $jid reason=$reason" | tee -a "$LOG"
      SC "$jid"
    fi
  done < <(SQ -u jdezoort -n "$JOB_NAME" -t PD -h -o '%i %T %R %M')
}

echo "$(date -Iseconds) gR-noloop-gt watchdog start A100_JOB=${A100_JOB:-none} TARGET=$TARGET MAX_CONCURRENT=$MAX_CONCURRENT" | tee -a "$LOG"

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

  sync_cancel_a100_for_gputest

  gt=$(n_gputest)
  echo "$(date -Iseconds) done=$done_n/$N_TASKS gt_queue=$(n_in_queue) gputest_parents=$gt incomplete=${#incomplete[@]} next=${incomplete[0]:-none}" | tee -a "$LOG"

  if (( done_n >= N_TASKS )); then
    echo "$(date -Iseconds) ALL gR-noloop cells >= $TARGET" | tee -a "$LOG"
    echo "GR_NOLOOP_GT_COMPLETE $(date -Iseconds)" | tee "$CACHE/GR_NOLOOP_GT_COMPLETE"
    exit 0
  fi

  slots=$((MAX_CONCURRENT - gt))
  if (( slots > 0 )); then
    submitted=0
    for task in "${incomplete[@]}"; do
      (( submitted >= slots )) && break
      if task_in_gt_queue "$task"; then
        cancel_a100_pending "$task"
        continue
      fi
      if task_running_on_a100 "$task"; then
        echo "$(date -Iseconds) skip task=$task (running on a100)" | tee -a "$LOG"
        continue
      fi
      cancel_a100_pending "$task"
      out=$(SB --parsable --job-name="$JOB_NAME" --array="$task" "$SLURM") || true
      if [[ ! "$out" =~ ^[0-9]+$ ]]; then
        echo "$(date -Iseconds) sbatch task=$task failed: $out" | tee -a "$LOG"
        continue
      fi
      echo "$(date -Iseconds) submitted task=$task job=$out" | tee -a "$LOG"
      echo "$out" >> "$CACHE/submitted_jobids.txt"
      cancel_a100_pending "$task"
      submitted=$((submitted + 1))
      gt=$(n_gputest)
      slots=$((MAX_CONCURRENT - gt))
      (( slots <= 0 )) && break
    done
  fi

  sleep "$SLEEP_SEC"
done
