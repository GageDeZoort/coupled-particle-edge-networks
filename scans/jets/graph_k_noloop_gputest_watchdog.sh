#!/bin/bash
# Greedy gputest fill for panel (a′) noloop k-sweep (v2cpNL).
#
# Keeps up to MAX_CONCURRENT unique parent jobs on gputest until every cell
# reaches TARGET steps. Leaves the a100 array (A100_JOB) for backlog, but
# cancels individual pending a100 tasks when gputest takes that cell (no
# double-run on the same run-tag).
#
#   bash scans/jets/start_graph_k_noloop_gputest_watchdogs.sh
#
set -uo pipefail

ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law
REPO=/home/jdezoort/coupled-particle-edge-networks
SLURM="$REPO/scans/jets/scaling_law_L4_graph_k_noloop_gputest_v2cp.slurm"
JOB_NAME="${JOB_NAME:-jc-gk-nl-gt}"
A100_JOB="${A100_JOB:-$(cat "$REPO/scans/jets/graph_k_noloop_v2cp_jobid.txt" 2>/dev/null || echo 15419558)}"
TARGET="${TARGET:-2495}"
MAX_STEPS="${MAX_STEPS:-2500}"
MAX_CONCURRENT="${MAX_CONCURRENT:-3}"
SLEEP_SEC="${SLEEP_SEC:-30}"
N_TASKS=54
PYTHON="${PYTHON:-/home/jdezoort/.conda/envs/mamba-env/bin/python}"
CACHE="${HOME}/.cache/jc_gk_noloop_gt"
LOG_DIR="$ROOT/logs"
mkdir -p "$CACHE" "$LOG_DIR"
LOG="$CACHE/watchdog.log"
STOP_FILES=("$CACHE/STOP" "$LOG_DIR/STOP_gk_noloop_gt")
LOCK="$CACHE/watchdog.lock"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "$(date -Iseconds) another noloop-gt watchdog holds $LOCK; exiting"
  exit 0
fi

KS=(1 2 3 4 5 6 7 8 10)
N_SHARD=3
N_INIT=2
N_PER_K=$((N_SHARD * N_INIT))
BATCH=128
N_GPUS=1
STAR_R=0.1
R_TAG=0p1

all_steps() {
  "$PYTHON" - "$ROOT/jetclass_output/adam/sweep_lr" "$BATCH" "$N_GPUS" "$MAX_STEPS" "$R_TAG" "$N_TASKS" <<'PY'
import re, sys
from pathlib import Path

sweep, batch, ngpu, max_steps, r_tag, n_tasks = sys.argv[1:7]
n_tasks = int(n_tasks)
ks = [1, 2, 3, 4, 5, 6, 7, 8, 10]
n_per_k = 6
steps = [0] * n_tasks
prefix = f"graph-k-noloop-b{batch}-g{ngpu}-s{max_steps}-R{r_tag}-"
pat = re.compile(
    rf"knn(\d+).*{re.escape(prefix)}shard(\d+)-seed(\d+)-v2cpNL(?:\.parquet)?$"
)
sweep_p = Path(sweep)
for pq in sweep_p.glob("*graph-k-noloop*v2cpNL.parquet"):
    m = pat.search(pq.name)
    if not m:
        continue
    k, shard, seed = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if k not in ks:
        continue
    k_idx = ks.index(k)
    task = k_idx * n_per_k + shard * 2 + seed
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

# Slurm controller has been intermittently hanging — always timeout.
SQ() { timeout 25 squeue "$@" 2>/dev/null || true; }
SC() { timeout 20 scancel "$@" 2>/dev/null || true; }
SB() { timeout 45 sbatch "$@" 2>/dev/null || true; }

# Unique parent job IDs on gputest (MaxJobsPU counts these).
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
  SQ -j "$A100_JOB" -t R -h -o '%i' \
    | grep -E "_${task}$" >/dev/null
}

# Cancel a100 array task if still pending (never cancel an a100 RUNNING task).
cancel_a100_pending() {
  local task=$1
  local line state
  line=$(SQ -j "${A100_JOB}_${task}" -h -o '%i %T' | head -1)
  [[ -z "$line" ]] && return 0
  state=$(awk '{print $2}' <<<"$line")
  if [[ "$state" == "PENDING" || "$state" == "PD" ]]; then
    SC "${A100_JOB}_${task}"
    echo "$(date -Iseconds) scancel pending a100 ${A100_JOB}_${task}" | tee -a "$LOG"
  fi
}

# Tasks currently on gputest (pending or running) under our job name.
gt_active_tasks() {
  SQ -u jdezoort -n "$JOB_NAME" -h -o '%i' \
    | while read -r jid; do
        if [[ "$jid" == *_\[* ]]; then
          # Compact forms: 123_[3] or 123_[3-5] — expand singles / ranges lightly.
          local body="${jid##*_\[}"
          body="${body%\]*}"
          body="${body%%\%*}"
          if [[ "$body" == *-* ]]; then
            local lo=${body%-*} hi=${body#*-}
            local i
            for ((i=lo; i<=hi; i++)); do echo "$i"; done
          else
            echo "$body"
          fi
        elif [[ "$jid" == *_* ]]; then
          echo "${jid##*_}"
        fi
      done | sort -nu
}

# Whenever gputest owns a cell (queued/running) OR parquet says done, drop a100 PD.
sync_cancel_a100_for_gputest() {
  local task
  while read -r task; do
    [[ -z "$task" ]] && continue
    cancel_a100_pending "$task"
  done < <(gt_active_tasks)

  # Finished cells: cancel leftover a100 PD so they never start late.
  local i step
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

echo "$(date -Iseconds) noloop-gt watchdog start A100_JOB=$A100_JOB TARGET=$TARGET MAX_CONCURRENT=$MAX_CONCURRENT MAX_STEPS=$MAX_STEPS SLEEP=$SLEEP_SEC" | tee -a "$LOG"

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

  # Always: if gputest has (or finished) a cell, kill the matching a100 PD.
  sync_cancel_a100_for_gputest

  gt=$(n_gputest)
  echo "$(date -Iseconds) done=$done_n/$N_TASKS gt_queue=$(n_in_queue) gputest_parents=$gt incomplete=${#incomplete[@]} next=${incomplete[0]:-none}" | tee -a "$LOG"

  if (( done_n >= N_TASKS )); then
    echo "$(date -Iseconds) ALL noloop cells >= $TARGET" | tee -a "$LOG"
    echo "NOLOOP_GT_COMPLETE $(date -Iseconds)" | tee "$CACHE/NOLOOP_GT_COMPLETE"
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
