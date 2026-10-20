#!/bin/bash
# Keep AdamW t_epoch=2 / η₀=0.05 / B=256×4 rolling on Della gputest (≤59m slices).
#
#   bash scans/jets/start_ft_L4cos_adamw_t2_gputest_watchdogs.sh
#   touch ~/.cache/tt_ft_aw_t2_watchdogs/STOP
#
set -uo pipefail

FT_ROOT="${FT_ROOT:-/scratch/gpfs/BHANIN/jdezoort/cpen_runs/toptagging_finetune}"
REPO=/home/jdezoort/coupled-particle-edge-networks
SLURM="$REPO/scans/jets/finetune_toptagging_L4cos_adamw_t2_gputest.slurm"
INIT_FROM="${INIT_FROM:-$FT_ROOT/init_ckpts/L4D256-cosine-s781250-frozen.ckpt}"
JOB_NAME="${JOB_NAME:-tt-ft-aw-t2-e0p05}"
ETA0="${ETA0:-0.05}"
EPOCHS="${EPOCHS:-10}"
T_EPOCH="${T_EPOCH:-2}"
LR_DECAY_MIN_FRAC="${LR_DECAY_MIN_FRAC:-0.01}"
BATCH="${BATCH:-256}"
N_GPUS="${N_GPUS:-4}"
# Prefer ailab H200 (gputest QOS); Della gputest was stuck on Priority.
WALL="${WALL:-0:59:00}"
PARTITION="${PARTITION:-ailab}"
QOS="${QOS:-gpu-test}"
SLEEP_SEC="${SLEEP_SEC:-90}"
PYTHON="${PYTHON:-/home/jdezoort/.conda/envs/mamba-env/bin/python}"
CACHE="${HOME}/.cache/tt_ft_aw_t2_watchdogs"
LOG_DIR="$FT_ROOT/logs"
mkdir -p "$CACHE" "$LOG_DIR"
LOG="$CACHE/watchdog_ft_aw_t2.log"
STOP_FILES=("$CACHE/STOP" "$LOG_DIR/STOP_ft_aw_t2")
LOCK="$CACHE/watchdog_ft_aw_t2.lock"
# If B=256 OOMs repeatedly, touch this to force BATCH=128 on next submit.
FALLBACK_FLAG="$CACHE/USE_BATCH_128"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "$(date -Iseconds) another aw-t2 watchdog holds $LOCK; exiting"
  exit 0
fi

run_tag_for() {
  local batch="$1"
  local eta_tag min_tag t_tag
  eta_tag=$(python3 -c "e=float('${ETA0}'); print(f'{e:g}'.replace('.','p'))")
  min_tag=$(python3 -c "e=float('${LR_DECAY_MIN_FRAC}'); print(f'{e:g}'.replace('.','p'))")
  t_tag=$(python3 -c "e=float('${T_EPOCH}'); print(f'{e:g}'.replace('.','p'))")
  echo "aw-t${t_tag}-c${min_tag}-e${eta_tag}-b${batch}x${N_GPUS}-ep${EPOCHS}"
}

current_batch() {
  if [[ -f "$FALLBACK_FLAG" ]]; then
    echo 128
  else
    echo "$BATCH"
  fi
}

in_queue() {
  timeout 30 squeue -u jdezoort -n "$JOB_NAME" -h 2>/dev/null | grep -q .
}

eta_done() {
  local batch
  batch=$(current_batch)
  ETA0="$ETA0" EPOCHS="$EPOCHS" BATCH="$batch" N_GPUS="$N_GPUS" \
  T_EPOCH="$T_EPOCH" LR_DECAY_MIN_FRAC="$LR_DECAY_MIN_FRAC" FT_ROOT="$FT_ROOT" \
  "$PYTHON" - <<'PY'
import csv, os, re
from pathlib import Path

eta = float(os.environ["ETA0"])
epochs_target = int(os.environ["EPOCHS"])
ft_root = Path(os.environ["FT_ROOT"])
batch = int(os.environ["BATCH"])
n_gpus = int(os.environ["N_GPUS"])
t_epoch = float(os.environ["T_EPOCH"])
min_frac = float(os.environ["LR_DECAY_MIN_FRAC"])
eta_tag = f"{eta:g}".replace(".", "p")
t_tag = f"{t_epoch:g}".replace(".", "p")
min_tag = f"{min_frac:g}".replace(".", "p")
run_tag = f"aw-t{t_tag}-c{min_tag}-e{eta_tag}-b{batch}x{n_gpus}-ep{epochs_target}"
sweep = ft_root / "toptagging_output" / "adamw" / "sweep_lr"
best_epoch = -1
cands = []
if sweep.is_dir():
    cands = [p for p in sweep.iterdir() if p.is_dir() and run_tag in p.name]
for d in cands:
    if (d / "FT_EPOCHS_DONE").is_file():
        print(f"DONE marker tag={run_tag}", flush=True)
        raise SystemExit(0)
    for met in d.rglob("metrics.csv"):
        try:
            rows = list(csv.DictReader(met.open()))
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

log_dir = ft_root / "logs"
for log in sorted(log_dir.glob("tt_ft_aw_t2_*.out"), key=lambda q: q.stat().st_mtime)[-30:]:
    try:
        text = log.read_text(errors="ignore")
    except Exception:
        continue
    if run_tag not in text and f"t_epoch={t_epoch}" not in text and f"t={t_epoch}" not in text:
        if f"η₀={eta}" not in text:
            continue
    eps = [int(x) for x in re.findall(r"\bepoch=(\d+)\b", text)]
    if eps:
        best_epoch = max(best_epoch, max(eps))

done = best_epoch >= (epochs_target - 1)
print(
    f"best_epoch={best_epoch} target_last={epochs_target-1} done={int(done)} tag={run_tag}",
    flush=True,
)
raise SystemExit(0 if done else 1)
PY
}

maybe_fallback_batch() {
  # If the newest t2 log shows CUDA OOM at B=256, switch to 128.
  local newest
  newest=$(ls -t "$LOG_DIR"/tt_ft_aw_t2_*.out 2>/dev/null | head -1 || true)
  [[ -n "$newest" ]] || return 0
  if grep -qE 'OutOfMemoryError|CUDA out of memory' "$newest" 2>/dev/null; then
    if [[ ! -f "$FALLBACK_FLAG" ]] && grep -qE 'B=256×4|batch-size 256|b256x4' "$newest" 2>/dev/null; then
      touch "$FALLBACK_FLAG"
      echo "$(date -Iseconds) OOM at B=256×4 — falling back to BATCH=128 for subsequent submits" | tee -a "$LOG"
    fi
  fi
}

submit_one() {
  local batch
  batch=$(current_batch)
  local tag
  tag=$(run_tag_for "$batch")
  local qos_args=()
  if [[ -n "${QOS}" && "${QOS}" != "none" ]]; then
    qos_args=(--qos="$QOS")
  fi
  sbatch --parsable -p "$PARTITION" -t "$WALL" "${qos_args[@]}" \
    --account=bhanin \
    --gres=gpu:4 --gpus-per-task=4 --cpus-per-task=16 --mem=256G \
    --job-name="$JOB_NAME" \
    --export=ALL,INIT_FROM="$INIT_FROM",FT_ROOT="$FT_ROOT",ETA0="$ETA0",EPOCHS="$EPOCHS",BATCH="$batch",N_GPUS="$N_GPUS",T_EPOCH="$T_EPOCH",LR_DECAY_MIN_FRAC="$LR_DECAY_MIN_FRAC",RUN_TAG="$tag" \
    "$SLURM" 2>/dev/null || true
}

echo "$(date -Iseconds) AdamW t=${T_EPOCH} η₀=${ETA0} watchdog start partition=$PARTITION qos=${QOS:-none} B=${BATCH}×${N_GPUS} wall=$WALL init=$INIT_FROM" | tee -a "$LOG"

if [[ ! -f "$INIT_FROM" ]]; then
  echo "$(date -Iseconds) FATAL missing init: $INIT_FROM" | tee -a "$LOG"
  exit 1
fi

while true; do
  for stopf in "${STOP_FILES[@]}"; do
    if [[ -f "$stopf" ]]; then
      echo "$(date -Iseconds) STOP $stopf; exiting" | tee -a "$LOG"
      exit 0
    fi
  done

  maybe_fallback_batch

  if eta_done; then
    echo "$(date -Iseconds) DONE — exiting" | tee -a "$LOG"
    exit 0
  fi

  if in_queue; then
    echo "$(date -Iseconds) job $JOB_NAME already queued/running; sleep" | tee -a "$LOG"
  else
    jid=$(submit_one)
    echo "$(date -Iseconds) submitted job=$jid name=$JOB_NAME batch=$(current_batch)×${N_GPUS} tag=$(run_tag_for "$(current_batch)")" | tee -a "$LOG"
  fi

  sleep "$SLEEP_SEC"
done
