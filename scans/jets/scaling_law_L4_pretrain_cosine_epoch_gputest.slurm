#!/bin/bash
# JetClass pretrain — L4/D256 Adam η₀=0.1 · epoch-2 cosine decay to 0.01× peak LR.
#
# Seeds from the finished const-LR epoch (step ≥ 390625) under jetclass_pretrain,
# then trains one more unique epoch (global_step 390625 → 781250) with
#   --scheduler warmup_flat_cosine --warmup-steps 0 --flat-steps 0
#   --cosine-steps 390625 --lr-decay-min-frac 0.01
#
# Prefer the watchdog:
#   bash scans/jets/start_pretrain_L4_cosine_epoch_gputest_watchdogs.sh
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-pretrain-L4cos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=a100
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=0:59:00
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_pretrain/logs/jc_pretrain_L4cos_gt_%j.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_pretrain/logs/jc_pretrain_L4cos_gt_%j.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

REPO=/home/jdezoort/coupled-particle-edge-networks
cd "$REPO/scans/jets"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_pretrain
mkdir -p "$ROOT/logs"

ETA0=0.1
DEPTH=4
WIDTH=256
D_HEAD=32
HEAD=$((WIDTH / D_HEAD))
STAR_R=0.1
KNN_K=7
BATCH=64
N_GPUS=4
N_TRAIN=100000000
# Epoch-1 horizon (const LR) and epoch-2 length (cosine).
EPOCH1_STEPS=$((N_TRAIN / (BATCH * N_GPUS)))   # 390625
MAX_STEPS=$((2 * EPOCH1_STEPS))                 # 781250
COSINE_STEPS=$EPOCH1_STEPS
MIN_FRAC=0.01
SEED=0
N_VAL=25000
# Lightning stop is min(max_epochs, max_steps). Resume ckpt is mid/end of an
# epoch (~390003) with fit_loop epoch already started — EPOCHS=1 or 2 lets
# Trainer finish the ~600-step remainder and exit before max_steps, and
# ModelCheckpoint (every 1000 steps) never writes, so the next slot loops the
# same 600 steps forever. Keep epochs huge; max_steps=781250 is the real stop.
EPOCHS=100
R_TAG=$(python3 -c "e=float('${STAR_R}'); print(f'{e:g}'.replace('.','p'))")
RUN_TAG="pretrain-cos-b${BATCH}-g${N_GPUS}-s${MAX_STEPS}-dh${D_HEAD}-k${KNN_K}-R${R_TAG}-min0p01"
CONST_TAG="pretrain-b64-g4-s390625-dh32-k7-R0p1"

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}  partition=gpu-test (via a100≤59m)"
echo "study=jetclass-pretrain-cosine-epoch  L=$DEPTH  D=$WIDTH  H=$HEAD"
echo "eta0=$ETA0  scheduler=warmup_flat_cosine  cosine_steps=$COSINE_STEPS  min_frac=$MIN_FRAC"
echo "B_gpu=$BATCH  n_gpus=$N_GPUS  max_steps=$MAX_STEPS  run_tag=$RUN_TAG"

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("No CUDA device visible")
if torch.cuda.device_count() < 4:
    sys.exit(f"Need 4 GPUs; visible={torch.cuda.device_count()}")
print(f"n_cuda={torch.cuda.device_count()}")
PY

# Seed cosine run dir from finished const-LR last.ckpt (full Lightning resume).
python - <<PY
import json, shutil
from datetime import datetime
from pathlib import Path
import torch

root = Path("${ROOT}")
sweep = root / "jetclass_output/adam/sweep_lr"
const_tag = "${CONST_TAG}"
run_tag = "${RUN_TAG}"
epoch1 = int("${EPOCH1_STEPS}")
# last.ckpt may sit on a 1k boundary just under max_steps; accept near-complete.
min_step = epoch1 - 2000

consts = [
    p for p in sweep.glob(f"*4_256_0p1*s390p625k*{const_tag}")
    if p.is_dir() and not (p / "SUPERSEDED.txt").exists()
]
assert consts, f"const-LR run dir not found (*s390p625k*{const_tag})"
const = max(consts, key=lambda p: p.stat().st_mtime)
src = const / "last.ckpt"
assert src.is_file(), f"missing {src}"
payload = torch.load(src, map_location="cpu", weights_only=False)
step = int(payload.get("global_step", 0) or 0)
assert step >= min_step, (
    f"const epoch not finished yet: global_step={step} < {min_step} ({src})"
)

# Destination: any existing cosine dir, else create predicted basename.
cands = [
    p for p in sweep.glob(f"*4_256_0p1*{run_tag}*")
    if p.is_dir() and not (p / "SUPERSEDED.txt").exists()
]
if cands:
    dest = max(cands, key=lambda p: p.stat().st_mtime)
else:
    # Mirror const naming with cosine tokens (must match sweep_common tagging).
    name = const.name
    name = name.replace("_ln_op-", "_ln_wfc_op-", 1)
    name = name.replace("-s390p625k-", "-s781p25k-", 1)
    name = name.replace(const_tag, run_tag + "-w0f0c390625min0p01", 1)
    dest = sweep / name
    dest.mkdir(parents=True, exist_ok=True)

dst = dest / "last.ckpt"
if not dst.is_file():
    shutil.copy2(src, dst)
    print(f"[seed] copied {src} → {dst} (step={step})")
else:
    cur = torch.load(dst, map_location="cpu", weights_only=False)
    cur_step = int(cur.get("global_step", 0) or 0)
    # Never clobber a cosine ckpt with an older const seed.
    if cur_step < min_step and step >= min_step:
        shutil.copy2(src, dst)
        print(f"[seed] refreshed stale {dst} ({cur_step}→{step})")
    else:
        print(f"[seed] resume {dst} (step={cur_step})")

manifest = {
    "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
    "source_dir": str(const),
    "dest_dir": str(dest),
    "source_global_step": step,
    "target_max_steps": int("${MAX_STEPS}"),
    "cosine_steps": int("${COSINE_STEPS}"),
    "lr_decay_min_frac": float("${MIN_FRAC}"),
    "scheduler": "warmup_flat_cosine",
    "run_tag": run_tag,
    "note": "Epoch-2 cosine: peak→0.01×peak over one unique JetClass pass",
}
(root / "manifests").mkdir(exist_ok=True)
(root / "manifests" / "L4D256_cosine_epoch2_seed.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
)
print(json.dumps(manifest, indent=2))
PY

python test_cpen_jetclass.py \
  --model capen-llama-att \
  --operators incidence \
  --jetclass-stream \
  --jetclass-raw-root /projects/j/jdezoort/jetclass \
  --jetclass-features kin7 \
  --jetclass-edge-features part-interaction \
  --live-knn-k "$KNN_K" \
  --star-radius "$STAR_R" \
  --live-star-kin7 \
  --mode sweep_lr \
  --optimizer adam \
  --scheduler warmup_flat_cosine \
  --warmup-steps 0 \
  --flat-steps 0 \
  --cosine-steps "$COSINE_STEPS" \
  --lr-decay-min-frac "$MIN_FRAC" \
  --t-epoch 1 \
  --residual-structure transformer-like \
  --etas "$ETA0" \
  --epochs "$EPOCHS" \
  --max-steps "$MAX_STEPS" \
  --n-train "$N_TRAIN" \
  --n-test 0 \
  --data-root /scratch/gpfs/BHANIN/jdezoort/datasets/jetclass \
  --depth "$DEPTH" --width "$WIDTH" --heads "$HEAD" \
  --dropout 0.0 \
  --root "$ROOT" \
  --run-tag "$RUN_TAG" \
  --batch-size "$BATCH" \
  --n-gpus "$N_GPUS" \
  --num-workers 4 \
  --normalization uniform \
  --attention-normalization gamma \
  --num-particles 80 \
  --n-val "$N_VAL" \
  --seed "$SEED" \
  --log-every-n-steps 50 \
  --layer-norm \
  --check-val-every-n-epoch 1 \
  --val-check-interval 1000 \
  --checkpoint-every-n-steps 1000 \
  --checkpoint-monitor val_loss \
  --checkpoint-mode min \
  --no-torch-compile

echo "done pretrain L4 cosine epoch max_steps=$MAX_STEPS min_frac=$MIN_FRAC"
