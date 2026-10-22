#!/usr/bin/env bash
# Pool watchdog: edge-bias M11 (arm E) μP transfer η₀ → 40k.
#
# Required env:
#   ARCH_KEY   L4D256 | L6D512 | L3D192
#   PARTITION  gputest | ailab
#
# Optional:
#   TARGET MAX_STEPS MAX_SLOTS SLEEP_SEC
#
#   ARCH_KEY=L4D256 PARTITION=gputest bash scans/jets/ebias_m11_xfer_watchdog.sh
#   touch ~/.cache/jc_ebX_${ARCH_KEY}_${PARTITION}_watchdogs/STOP
#
set -uo pipefail

ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law
REPO=/home/jdezoort/coupled-particle-edge-networks
SLURM="$REPO/scans/jets/scaling_law_ebias_m11_xfer_eta.slurm"
PYTHON="${PYTHON:-/home/jdezoort/.conda/envs/mamba-env/bin/python}"
TARGET="${TARGET:-40000}"
MAX_STEPS="${MAX_STEPS:-40000}"
MAX_SLOTS="${MAX_SLOTS:-1}"
SLEEP_SEC="${SLEEP_SEC:-90}"
N_TASKS=5
ETAS=(0.01 0.05 0.1 0.25 0.5)

ARCH_KEY="${ARCH_KEY:?set ARCH_KEY=L4D256|L6D512|L3D192}"
PARTITION="${PARTITION:?set PARTITION=gputest|ailab}"

case "$ARCH_KEY" in
  L4D256) DEPTH=4; WIDTH=256 ;;
  L6D512) DEPTH=6; WIDTH=512 ;;
  L3D192) DEPTH=3; WIDTH=192 ;;
  *) echo "unknown ARCH_KEY=$ARCH_KEY" >&2; exit 1 ;;
esac

case "$PARTITION" in
  gputest)
    TAG_SUFFIX=gt
    BATCH=64
    N_GPUS=4
    JOB_NAME="jc-ebX-${ARCH_KEY}-gt"
    SBATCH_EXTRA=(--partition=a100 --gpus-per-task=4 --gres=gpu:4 --cpus-per-task=16 --mem=256G)
    ;;
  ailab)
    TAG_SUFFIX=ailab
    BATCH=256
    N_GPUS=1
    JOB_NAME="jc-ebX-${ARCH_KEY}-ailab"
    SBATCH_EXTRA=(--partition=ailab --qos=gpu-test --gpus-per-task=1 --gres=gpu:1 --cpus-per-task=8 --mem=64G)
    ;;
  *) echo "unknown PARTITION=$PARTITION" >&2; exit 1 ;;
esac

RUN_TAG="ebX-L${DEPTH}D${WIDTH}-${TAG_SUFFIX}"
CACHE="${HOME}/.cache/jc_ebX_${ARCH_KEY}_${PARTITION}_watchdogs"
LOG_DIR="$ROOT/logs"
mkdir -p "$CACHE" "$LOG_DIR"
LOG="$CACHE/watchdog.log"
STOP_FILES=("$CACHE/STOP" "$LOG_DIR/STOP_ebX_${ARCH_KEY}_${PARTITION}")
LOCK="$CACHE/watchdog.lock"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "$(date -Iseconds) another ebX ${ARCH_KEY}/${PARTITION} watchdog holds $LOCK; exiting"
  exit 0
fi

cell_meta() {
  echo "${ETAS[$1]}"
}

all_steps() {
  "$PYTHON" - "$ROOT/jetclass_output/adam/sweep_lr" "$RUN_TAG" "$DEPTH" "$WIDTH" "$N_TASKS" "$LOG_DIR" "$JOB_NAME" <<'PY'
import re, sys
from pathlib import Path

sweep, run_tag, depth, width, n_tasks = (
    Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
)
log_dir, job_name = Path(sys.argv[6]), sys.argv[7]
etas = [0.01, 0.05, 0.1, 0.25, 0.5]
eta_to_idx = {f"{e:g}".replace(".", "p"): i for i, e in enumerate(etas)}
steps = [0] * n_tasks
pat = re.compile(
    rf"capen-llama-att_{re.escape(depth)}_{re.escape(width)}_(?P<eta>[0-9p]+)_.*{re.escape(run_tag)}"
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

for log in sorted(log_dir.glob("jc_ebX_*.out"), key=lambda q: q.stat().st_mtime):
    text = log.read_text(errors="ignore")[-250000:]
    if f"run_tag={run_tag}" not in text and run_tag not in text:
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

# Count all of this user's gpu-test QoS jobs (shared cap ≈ 3).
n_qos_gpu_test() {
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

echo "$(date -Iseconds) ebX watchdog start ARCH=$ARCH_KEY PARTITION=$PARTITION TARGET=$TARGET MAX_SLOTS=$MAX_SLOTS RUN_TAG=$RUN_TAG JOB=$JOB_NAME" | tee -a "$LOG"

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
  qos=$(n_qos_gpu_test)
  own=$(n_own)
  echo "$(date -Iseconds) done=$done_n/$N_TASKS own_q=$own qos_gpu_test=$qos$status" | tee -a "$LOG"

  if (( done_n >= N_TASKS )); then
    echo "$(date -Iseconds) ALL η cells >= $TARGET for $ARCH_KEY/$PARTITION; exiting" | tee -a "$LOG"
    echo "EBX_${ARCH_KEY}_${PARTITION}_COMPLETE $(date -Iseconds)" | tee "$CACHE/COMPLETE"
    exit 0
  fi

  # Shared QoS budget: only submit if under the global gpu-test cap.
  # MAX_SLOTS = max concurrent jobs *this* watchdog may own.
  slots_qos=$((3 - qos))
  slots_own=$((MAX_SLOTS - own))
  slots=$slots_qos
  if (( slots_own < slots )); then slots=$slots_own; fi
  if (( slots < 0 )); then slots=0; fi

  if (( slots > 0 )); then
    submitted=0
    # Global flock so the 6 arch×partition watchdogs cannot race the QoS cap.
    SUBMIT_LOCK="${HOME}/.cache/jc_ebX_submit.lock"
    mkdir -p "$(dirname "$SUBMIT_LOCK")"
    exec 8>"$SUBMIT_LOCK"
    if flock -n 8; then
      # Re-read QoS under the lock.
      qos=$(n_qos_gpu_test)
      own=$(n_own)
      slots_qos=$((3 - qos))
      slots_own=$((MAX_SLOTS - own))
      slots=$slots_qos
      if (( slots_own < slots )); then slots=$slots_own; fi
      if (( slots < 0 )); then slots=0; fi
      for task in "${incomplete[@]}"; do
        (( submitted >= slots )) && break
        if task_in_queue "$task"; then
          continue
        fi
        out=$(timeout 90 sbatch --parsable \
          --job-name="$JOB_NAME" \
          --array="$task" \
          "${SBATCH_EXTRA[@]}" \
          --export=ALL,MAX_STEPS="$MAX_STEPS",DEPTH="$DEPTH",WIDTH="$WIDTH",BATCH="$BATCH",N_GPUS="$N_GPUS",TAG_SUFFIX="$TAG_SUFFIX" \
          "$SLURM" 2>&1) || {
          echo "$(date -Iseconds) sbatch task=$task failed: $out" | tee -a "$LOG"
          continue
        }
        echo "$(date -Iseconds) submitted task=$task eta=$(cell_meta "$task") job=$out" | tee -a "$LOG"
        submitted=$((submitted + 1))
        qos=$(n_qos_gpu_test)
        own=$(n_own)
        slots_qos=$((3 - qos))
        slots_own=$((MAX_SLOTS - own))
        slots=$slots_qos
        if (( slots_own < slots )); then slots=$slots_own; fi
        if (( slots < 0 )); then slots=0; fi
        (( slots <= 0 )) && break
      done
      flock -u 8
    fi
    exec 8>&-
  fi

  sleep "$SLEEP_SEC"
done
