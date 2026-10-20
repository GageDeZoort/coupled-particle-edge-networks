#!/bin/bash
# Pool watchdog: TopTagging FULL FT AdamW+cosine probe (η₀ ∈ 0.01, 0.025, 0.05).
# Cosine → 0.01× peak LR over EPOCHS; AdamW WD from t_epoch=25.
#
#   bash scans/jets/start_ft_L4cos_full_adamw_cos_eta_sweep_gputest_watchdogs.sh
#   touch ~/.cache/tt_ft_L4cos_aw_watchdogs/STOP
#
set -uo pipefail

FT_ROOT="${FT_ROOT:-/scratch/gpfs/BHANIN/jdezoort/cpen_runs/toptagging_finetune}"
REPO=/home/jdezoort/coupled-particle-edge-networks
SLURM="$REPO/scans/jets/finetune_toptagging_L4cos_full_adamw_cos_gputest.slurm"
INIT_FROM="${INIT_FROM:-$FT_ROOT/init_ckpts/L4D256-cosine-s781250-frozen.ckpt}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-tt-ft-L4cos-aw}"
ETAS="${ETAS:-0.01 0.025 0.05}"
EPOCHS="${EPOCHS:-10}"
T_EPOCH="${T_EPOCH:-25}"
LR_DECAY_MIN_FRAC="${LR_DECAY_MIN_FRAC:-0.01}"
BATCH="${BATCH:-256}"
N_GPUS="${N_GPUS:-1}"
WALL="${WALL:-12:00:00}"
PARTITION="${PARTITION:-a100}"
DEPTH=4
WIDTH=256
KNN_K=7
STAR_R=0.1
MAX_CONCURRENT="${MAX_CONCURRENT:-3}"
SLEEP_SEC="${SLEEP_SEC:-90}"
PYTHON="${PYTHON:-/home/jdezoort/.conda/envs/mamba-env/bin/python}"
CACHE="${HOME}/.cache/tt_ft_L4cos_aw_watchdogs"
LOG_DIR="$FT_ROOT/logs"
mkdir -p "$CACHE" "$LOG_DIR"
LOG="$CACHE/watchdog_ft_L4cos_aw.log"
STOP_FILES=("$CACHE/STOP" "$LOG_DIR/STOP_ft_L4cos_aw")
LOCK="$CACHE/watchdog_ft_L4cos_aw.lock"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "$(date -Iseconds) another FT L4cos AdamW watchdog holds $LOCK; exiting"
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
    --constraint=gpu80 \
    --job-name="$jname" \
    --export=ALL,INIT_FROM="$INIT_FROM",FT_ROOT="$FT_ROOT",ETA0="$eta",EPOCHS="$EPOCHS",BATCH="$BATCH",N_GPUS="$N_GPUS",DEPTH="$DEPTH",WIDTH="$WIDTH",KNN_K="$KNN_K",STAR_R="$STAR_R",T_EPOCH="$T_EPOCH",LR_DECAY_MIN_FRAC="$LR_DECAY_MIN_FRAC" \
    "$SLURM" 2>/dev/null || true
}

eta_in_queue() {
  local eta="$1"
  local jname
  jname=$(eta_job_name "$eta")
  if timeout 30 squeue -u jdezoort -n "$jname" -h 2>/dev/null | grep -q .; then
    return 0
  fi
  return 1
}

eta_done() {
  local eta="$1"
  ETA0="$eta" EPOCHS="$EPOCHS" BATCH="$BATCH" N_GPUS="$N_GPUS" DEPTH="$DEPTH" WIDTH="$WIDTH" \
  KNN_K="$KNN_K" STAR_R="$STAR_R" FT_ROOT="$FT_ROOT" T_EPOCH="$T_EPOCH" \
  LR_DECAY_MIN_FRAC="$LR_DECAY_MIN_FRAC" \
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
t_epoch = float(os.environ["T_EPOCH"])
min_frac = float(os.environ["LR_DECAY_MIN_FRAC"])
eta_tag = f"{eta:g}".replace(".", "p")
r_tag = f"{star:g}".replace(".", "p")
t_tag = f"{t_epoch:g}".replace(".", "p")
min_tag = f"{min_frac:g}".replace(".", "p")
run_tag = (
    f"aw-t{t_tag}-c{min_tag}-e{eta_tag}-b{batch}-ep{epochs_target}"
)
sweep = ft_root / "toptagging_output" / "adamw" / "sweep_lr"
best_epoch = -1
cands = []
if sweep.is_dir():
    cands = [p for p in sweep.iterdir() if p.is_dir() and run_tag in p.name]
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
    if (d / "FT_EPOCHS_DONE").is_file():
        print(f"eta={eta} DONE marker tag={run_tag}", flush=True)
        raise SystemExit(0)

log_dir = ft_root / "logs"
for log in sorted(log_dir.glob("tt_ft_L4cos_aw_*.out"), key=lambda q: q.stat().st_mtime)[-40:]:
    try:
        text = log.read_text(errors="ignore")
    except Exception:
        continue
    if f"η₀={eta}" not in text and f"etas {eta}" not in text:
        continue
    if "AdamW+cosine" not in text and "adamw-t" not in text:
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
  timeout 30 squeue -u jdezoort -h -o '%j' 2>/dev/null \
    | grep -cE "^${JOB_NAME_PREFIX}(-e|$)" || true
}

echo "$(date -Iseconds) FT L4cos AdamW+cosine probe start etas=[$ETAS] epochs=$EPOCHS t_epoch=$T_EPOCH min_frac=$LR_DECAY_MIN_FRAC B=${BATCH}×${N_GPUS} wall=$WALL concurrent≤$MAX_CONCURRENT init=$INIT_FROM" | tee -a "$LOG"

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
