#!/bin/bash
# Pool watchdog: TopTagging head-only FT η₀ sweep from frozen L4 cosine ckpt.
# Cycles through ETAS until each reaches EPOCHS (10).
# Default: B=256×1 on a100 (~12h slots; resume last.ckpt if sliced).
#
#   bash scans/jets/start_ft_L4cos_head_eta_sweep_gputest_watchdogs.sh
#   touch ~/.cache/tt_ft_L4cos_watchdogs/STOP
#
set -uo pipefail

FT_ROOT="${FT_ROOT:-/scratch/gpfs/BHANIN/jdezoort/cpen_runs/toptagging_finetune}"
REPO=/home/jdezoort/coupled-particle-edge-networks
SLURM="$REPO/scans/jets/finetune_toptagging_L4cos_head_gputest.slurm"
INIT_FROM="${INIT_FROM:-$FT_ROOT/init_ckpts/L4D256-cosine-s781250-frozen.ckpt}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-tt-ft-L4cos}"
# Drop η₀>0.1: full-FT already worse there; free GPUs for the promising cells.
ETAS="${ETAS:-0.01 0.025 0.05 0.075 0.1}"
EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-256}"
N_GPUS="${N_GPUS:-1}"
WALL="${WALL:-12:00:00}"
PARTITION="${PARTITION:-a100}"
DEPTH=4
WIDTH=256
KNN_K=7
STAR_R=0.1
# One GPU per η₀ cell — submit the full grid in parallel.
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
SLEEP_SEC="${SLEEP_SEC:-60}"
PYTHON="${PYTHON:-/home/jdezoort/.conda/envs/mamba-env/bin/python}"
CACHE="${HOME}/.cache/tt_ft_L4cos_watchdogs"
LOG_DIR="$FT_ROOT/logs"
mkdir -p "$CACHE" "$LOG_DIR"
LOG="$CACHE/watchdog_ft_L4cos.log"
STOP_FILES=("$CACHE/STOP" "$LOG_DIR/STOP_ft_L4cos")
LOCK="$CACHE/watchdog_ft_L4cos.lock"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "$(date -Iseconds) another FT L4cos watchdog holds $LOCK; exiting"
  exit 0
fi

eta_job_name() {
  local eta="$1"
  local tag
  tag=$(python3 -c "e=float('${eta}'); print(f'{e:g}'.replace('.','p'))")
  echo "${JOB_NAME_PREFIX}-e${tag}"
}

submit_one() {
  local eta="$1"
  local jname
  jname=$(eta_job_name "$eta")
  sbatch --parsable -p "$PARTITION" -t "$WALL" \
    --gres=gpu:1 --gpus-per-task=1 --cpus-per-task=8 --mem=128G \
    --job-name="$jname" \
    --export=ALL,INIT_FROM="$INIT_FROM",FT_ROOT="$FT_ROOT",ETA0="$eta",EPOCHS="$EPOCHS",BATCH="$BATCH",N_GPUS="$N_GPUS",DEPTH="$DEPTH",WIDTH="$WIDTH",KNN_K="$KNN_K",STAR_R="$STAR_R" \
    "$SLURM" 2>/dev/null || true
}

# True if this η₀ already has a queued/running job.
eta_in_queue() {
  local eta="$1"
  local jname
  jname=$(eta_job_name "$eta")
  if timeout 30 squeue -u jdezoort -n "$jname" -h 2>/dev/null | grep -q .; then
    return 0
  fi
  # Legacy pre-parallel job name (single cell, was always η₀=0.01).
  if [[ "$eta" == "0.01" ]]; then
    if timeout 30 squeue -u jdezoort -n "$JOB_NAME_PREFIX" -h 2>/dev/null | grep -q .; then
      return 0
    fi
  fi
  return 1
}

# Exit 0 if eta finished (≥ EPOCHS), 1 if still needs work.
eta_done() {
  local eta="$1"
  ETA0="$eta" EPOCHS="$EPOCHS" BATCH="$BATCH" N_GPUS="$N_GPUS" DEPTH="$DEPTH" WIDTH="$WIDTH" \
  KNN_K="$KNN_K" STAR_R="$STAR_R" FT_ROOT="$FT_ROOT" \
  "$PYTHON" - <<'PY'
import csv, os, re
from pathlib import Path

eta = float(os.environ["ETA0"])
epochs_target = int(os.environ["EPOCHS"])
ft_root = Path(os.environ["FT_ROOT"])
batch = int(os.environ["BATCH"])
n_gpus = int(os.environ["N_GPUS"])
depth = int(os.environ["DEPTH"])
width = int(os.environ["WIDTH"])
knn = int(os.environ["KNN_K"])
star = float(os.environ["STAR_R"])
eta_tag = f"{eta:g}".replace(".", "p")
r_tag = f"{star:g}".replace(".", "p")
run_tag = (
    f"ft-L{depth}D{width}-head-apool-cos781k-eta{eta_tag}"
    f"-b{batch}-g{n_gpus}-ep{epochs_target}-k{knn}-R{r_tag}"
)
sweep = ft_root / "toptagging_output" / "adam" / "sweep_lr"
best_epoch = -1
cands = []
if sweep.is_dir():
    cands = [p for p in sweep.iterdir() if p.is_dir() and run_tag in p.name]
if not cands and sweep.is_dir():
    cands = [
        p for p in sweep.iterdir()
        if p.is_dir() and "freezebb" in p.name and f"_{eta_tag}_" in f"_{p.name}_"
        and f"-b{batch}-g{n_gpus}-" in p.name
    ]
for d in cands:
    for met in d.rglob("metrics.csv"):
        try:
            with met.open() as f:
                rows = list(csv.DictReader(f))
        except Exception:
            continue
        for row in rows:
            raw = row.get("epoch") or row.get("Epoch")
            if raw in (None, ""):
                continue
            try:
                best_epoch = max(best_epoch, int(float(raw)))
            except Exception:
                pass
    # Lightning max_epochs=10 → last logged epoch index is 9
    if (d / "FT_EPOCHS_DONE").is_file():
        print(f"eta={eta} DONE marker tag={run_tag}", flush=True)
        raise SystemExit(0)

log_dir = ft_root / "logs"
for log in sorted(log_dir.glob("tt_ft_L4cos_*.out"), key=lambda q: q.stat().st_mtime)[-40:]:
    try:
        text = log.read_text(errors="ignore")
    except Exception:
        continue
    if "_full_" in log.name:
        continue
    if f"η₀={eta}" not in text and f"etas {eta}" not in text:
        continue
    if "B=256×1" not in text and "-b256-g1-" not in text:
        continue
    eps = [int(x) for x in re.findall(r"\bepoch=(\d+)\b", text)]
    if eps:
        best_epoch = max(best_epoch, max(eps))

done = best_epoch >= (epochs_target - 1)
print(
    f"eta={eta} best_epoch={best_epoch} target_last={epochs_target-1} "
    f"done={int(done)} tag={run_tag}",
    flush=True,
)
raise SystemExit(0 if done else 1)
PY
}

n_ft_queue() {
  # Count frozen-head FT jobs only (exclude tt-ft-L4cos-full*).
  timeout 30 squeue -u jdezoort -h -o '%j' 2>/dev/null \
    | grep -E "^${JOB_NAME_PREFIX}(-e|$)" | grep -cv full || true
}

echo "$(date -Iseconds) FT L4cos head sweep start etas=[$ETAS] epochs=$EPOCHS B=${BATCH}×${N_GPUS} wall=$WALL concurrent≤$MAX_CONCURRENT init=$INIT_FROM" | tee -a "$LOG"

if [[ ! -f "$INIT_FROM" ]]; then
  echo "$(date -Iseconds) FATAL missing immutable init ckpt: $INIT_FROM" | tee -a "$LOG"
  exit 1
fi

while true; do
  for stopf in "${STOP_FILES[@]}"; do
    if [[ -f "$stopf" ]]; then
      echo "$(date -Iseconds) STOP $stopf; exiting" | tee -a "$LOG"
      exit 0
    fi
  done

  pending=()
  for eta in $ETAS; do
    if eta_done "$eta"; then
      echo "$(date -Iseconds) eta=$eta DONE" | tee -a "$LOG"
    else
      pending+=("$eta")
    fi
  done

  if (( ${#pending[@]} == 0 )); then
    echo "$(date -Iseconds) ALL etas done — exiting" | tee -a "$LOG"
    exit 0
  fi

  q=$(n_ft_queue)
  echo "$(date -Iseconds) pending_etas=${pending[*]} ft_q=$q max_c=$MAX_CONCURRENT" | tee -a "$LOG"

  for eta in "${pending[@]}"; do
    q=$(n_ft_queue)
    if (( q >= MAX_CONCURRENT )); then
      break
    fi
    if eta_in_queue "$eta"; then
      continue
    fi
    jid=$(submit_one "$eta")
    echo "$(date -Iseconds) submitted job=$jid eta=$eta name=$(eta_job_name "$eta") (${PARTITION}/${WALL} B=${BATCH}×${N_GPUS})" | tee -a "$LOG"
    sleep 2
  done

  sleep "$SLEEP_SEC"
done
