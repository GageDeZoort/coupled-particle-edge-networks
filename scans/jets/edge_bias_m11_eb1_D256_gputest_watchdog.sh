#!/usr/bin/env bash
# Pool watchdog: edge-bias M11 arm E @ L4/D256 → fill eb1E-gt canvas (40k).
# η₀ ∈ {0.05, 0.1, 0.25, 0.5}; Della gputest B=64×4.
#
#   bash scans/jets/start_edge_bias_m11_eb1_D256_gputest_watchdogs.sh
#   touch ~/.cache/jc_eb1E_gt_watchdogs/STOP
#
set -uo pipefail

ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law
REPO=/home/jdezoort/coupled-particle-edge-networks
SLURM="$REPO/scans/jets/scaling_law_L4_edge_bias_m11_eta_D256_k7_gputest.slurm"
JOB_NAME="${JOB_NAME:-jc-eb1-gt}"
TARGET="${TARGET:-40000}"
MAX_STEPS="${MAX_STEPS:-40000}"
MAX_GPUS="${MAX_GPUS:-3}"
SLEEP_SEC="${SLEEP_SEC:-90}"
N_TASKS=4
PYTHON="${PYTHON:-/home/jdezoort/.conda/envs/mamba-env/bin/python}"
CACHE="${HOME}/.cache/jc_eb1E_gt_watchdogs"
LOG_DIR="$ROOT/logs"
mkdir -p "$CACHE" "$LOG_DIR"
LOG="$CACHE/watchdog_eb1E_gt.log"
STOP_FILES=("$CACHE/STOP" "$LOG_DIR/STOP_eb1E_gt")
LOCK="$CACHE/watchdog_eb1E_gt.lock"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "$(date -Iseconds) another eb1E-gt watchdog holds $LOCK; exiting"
  exit 0
fi

ETAS=(0.05 0.1 0.25 0.5)
RUN_TAG="eb1E-gt"

cell_meta() { echo "${ETAS[$1]}"; }

all_steps() {
  "$PYTHON" - "$ROOT/jetclass_output/adam/sweep_lr" "$RUN_TAG" "$N_TASKS" "$LOG_DIR" <<'PY'
import re, sys
from pathlib import Path

sweep, run_tag, n_tasks = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
log_dir = Path(sys.argv[4])
etas = [0.05, 0.1, 0.25, 0.5]
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
                best = max(int(x.as_py()) for x in col if x.as_py() is not None)
        except Exception:
            pass
    run_dir = path if path.is_dir() else (
        path.parent / path.name.replace(".parquet", "")
        if path.suffix == ".parquet"
        else None
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
    steps[eta_to_idx[et]] = max(steps[eta_to_idx[et]], best_step(p))

for log in sorted(log_dir.glob("jc_eb1_gt_*.out"), key=lambda q: q.stat().st_mtime):
    text = log.read_text(errors="ignore")[-250000:]
    if run_tag not in text and "edge-bias-m11" not in text:
        continue
    me = re.search(r"eta0=([0-9.]+)", text)
    if me is None:
        continue
    try:
        eta = float(me.group(1))
    except ValueError:
        continue
    et = f"{eta:g}".replace(".", "p")
    if et not in eta_to_idx:
        continue
    st = 0
    for x in re.findall(r"(?:global_step=|\[train\] step=)(\d+)", text):
        st = max(st, int(x))
    steps[eta_to_idx[et]] = max(steps[eta_to_idx[et]], st)

for s in steps:
    print(s)
PY
}

n_qos() {
  timeout 30 squeue -u jdezoort -h -o '%q' 2>/dev/null | grep -c '^gpu-test$' || true
}

n_own() {
  timeout 30 squeue -u jdezoort -n "$JOB_NAME" -h 2>/dev/null | wc -l
}

task_in_queue() {
  local task=$1
  timeout 30 squeue -u jdezoort -n "$JOB_NAME" -h -o '%i' 2>/dev/null \
    | grep -E "_${task}$|\[${task}\]" >/dev/null
}

echo "$(date -Iseconds) eb1E-gt watchdog start TARGET=$TARGET MAX_GPUS=$MAX_GPUS RUN_TAG=$RUN_TAG" | tee -a "$LOG"

while true; do
  for stopf in "${STOP_FILES[@]}"; do
    if [[ -f "$stopf" ]]; then
      echo "$(date -Iseconds) STOP $stopf; exiting" | tee -a "$LOG"
      exit 0
    fi
  done

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
  qos=$(n_qos)
  own=$(n_own)
  echo "$(date -Iseconds) done=$done_n/$N_TASKS own_q=$own qos=$qos$status" | tee -a "$LOG"

  if (( done_n >= N_TASKS )); then
    (
      cd "$REPO"
      PYTHONPATH=src "$PYTHON" scans/jets/refresh_edge_bias_m11_eta_D256_k7_canvas.py || true
    ) >>"$LOG" 2>&1 || true
    echo "$(date -Iseconds) ALL eb1E-gt η cells >= $TARGET; exiting" | tee -a "$LOG"
    echo "EB1E_GT_COMPLETE $(date -Iseconds)" | tee "$CACHE/EB1E_GT_COMPLETE"
    exit 0
  fi

  # Prefer this pool: take up to MAX_GPUS of the shared qos=gpu-test budget.
  slots_qos=$((3 - qos))
  slots_own=$((MAX_GPUS - own))
  slots=$slots_qos
  if (( slots_own < slots )); then slots=$slots_own; fi
  if (( slots < 0 )); then slots=0; fi

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
      qos=$(n_qos)
      own=$(n_own)
      slots_qos=$((3 - qos))
      slots_own=$((MAX_GPUS - own))
      slots=$slots_qos
      if (( slots_own < slots )); then slots=$slots_own; fi
      if (( slots < 0 )); then slots=0; fi
      (( slots <= 0 )) && break
    done
  fi

  (
    cd "$REPO"
    PYTHONPATH=src "$PYTHON" scans/jets/refresh_edge_bias_m11_eta_D256_k7_canvas.py || true
  ) >>"$LOG" 2>&1 &

  sleep "$SLEEP_SEC"
done
