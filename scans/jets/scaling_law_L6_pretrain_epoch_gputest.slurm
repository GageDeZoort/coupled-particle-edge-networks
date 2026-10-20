#!/bin/bash
# JetClass pretrain — L=6 D=512 Adam η₀=0.1 · flat LR · one unique epoch.
#
# Matches L4/D256 protocol:
#   Seed:  edge η₀=0.1 @ step 40k (k★=7, R★=0.1, B=64×4) from scaling_law
#   Home:  /scratch/.../jetclass_pretrain/
#   Target: max_steps = floor(1e8/256) = 390625
#   Tag:   pretrain-L6D512-b64-g4-s390625-dh32-k7-R0p1
#
# EPOCHS=2 avoids the Lightning max_epochs=1 early-exit that skipped the final
# last.ckpt flush on L4 (see fig_pretrain_L4D256_epoch_cosine.md).
#
#   bash scans/jets/start_pretrain_L6_epoch_gputest_watchdogs.sh
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-pretrain-L6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=a100
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=0:59:00
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_pretrain/logs/jc_pretrain_L6_gt_%j.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_pretrain/logs/jc_pretrain_L6_gt_%j.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

REPO=/home/jdezoort/coupled-particle-edge-networks
cd "$REPO/scans/jets"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_pretrain
SEED_ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law
mkdir -p "$ROOT/logs" "$ROOT/manifests"

ETA0=0.1
DEPTH=6
WIDTH=512
D_HEAD=32
HEAD=$((WIDTH / D_HEAD))
STAR_R=0.1
KNN_K=7
BATCH=64
N_GPUS=4
N_TRAIN=100000000
MAX_STEPS=$((N_TRAIN / (BATCH * N_GPUS)))   # 390625
SEED=0
N_VAL=25000
# epochs=2: stop via max_steps, not max_epochs (L4 bug).
EPOCHS=2
R_TAG=$(python3 -c "e=float('${STAR_R}'); print(f'{e:g}'.replace('.','p'))")
RUN_TAG="pretrain-L6D512-b${BATCH}-g${N_GPUS}-s${MAX_STEPS}-dh${D_HEAD}-k${KNN_K}-R${R_TAG}"

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}  partition=gpu-test (via a100≤59m)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
echo "study=jetclass-pretrain-epoch  L=$DEPTH  D=$WIDTH  H=$HEAD  d_h=$D_HEAD"
echo "model=capen-llama-att  eta0=$ETA0  optimizer=adam  scheduler=none  k=$KNN_K  R=$STAR_R  seed=$SEED"
echo "B_gpu=$BATCH  n_gpus=$N_GPUS  B_global=$((BATCH * N_GPUS))  max_steps=$MAX_STEPS"
echo "n_train=$N_TRAIN  (unique single-pass epoch)  run_tag=$RUN_TAG  root=$ROOT"

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("No CUDA device visible")
print(f"n_cuda={torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU{i} {p.name} {p.total_memory/1024**3:.1f} GiB")
if torch.cuda.device_count() < 4:
    sys.exit(f"Need 4 GPUs; visible count={torch.cuda.device_count()}")
PY

# Seed jetclass_pretrain run dir from scaling_law edge η₀=0.1 @ 40k.
python - <<PY
import json, shutil
from datetime import datetime
from pathlib import Path
import torch

root = Path("${ROOT}")
seed_root = Path("${SEED_ROOT}")
sweep = root / "jetclass_output/adam/sweep_lr"
sweep.mkdir(parents=True, exist_ok=True)
run_tag = "${RUN_TAG}"
width = int("${WIDTH}")
depth = int("${DEPTH}")
head = int("${HEAD}")
max_steps = int("${MAX_STEPS}")

seed_dirs = [
    p for p in (seed_root / "jetclass_output/adam/sweep_lr").glob(
        f"*_{depth}_{width}_0p1_*edge-L6D512-b64-g4-s40000*"
    )
    if p.is_dir()
]
assert seed_dirs, "No L6/D512 η₀=0.1 @40k edge seed under scaling_law"
src_dir = max(seed_dirs, key=lambda p: p.stat().st_mtime)
src = src_dir / "last.ckpt"
assert src.is_file(), f"missing {src}"
payload = torch.load(src, map_location="cpu", weights_only=False)
step = int(payload.get("global_step", 0) or 0)
assert step >= 39000, f"seed step={step} looks incomplete"
print(f"[seed] source {src} global_step={step}")

# Prefer live format_count name (s390p625k).
cands = [
    p for p in sweep.glob(f"*_{depth}_{width}_0p1*{run_tag}*")
    if p.is_dir() and not (p / "SUPERSEDED.txt").exists()
]
live = [p for p in cands if "s390p625k" in p.name]
cands = live or cands
if cands:
    dest = max(cands, key=lambda p: p.stat().st_mtime)
else:
    dest = sweep / (
        f"capen-llama-att_{depth}_{width}_0p1_ln_op-incidence_onorm-degree_res-transformer-like"
        f"_h{head}_ntr100M_nval25k_ntst0_nf7_gstar0p1_stream-kin7-partint-p80-knn7-noloop-sk7"
        f"-D100M-s390p625k-seed0-{run_tag}"
    )
    dest.mkdir(parents=True, exist_ok=True)

dst = dest / "last.ckpt"
dest_step = 0
if dst.is_file():
    try:
        dest_step = int(
            torch.load(dst, map_location="cpu", weights_only=False).get("global_step", 0) or 0
        )
    except Exception:
        dest_step = 0
# Only seed if dest is behind the 40k source (never clobber a longer pretrain).
if (not dst.is_file()) or (dest_step < step and dest_step < 50000):
    shutil.copy2(src, dst)
    # also best.ckpt for monitors that prefer it
    if (src_dir / "best.ckpt").is_file() and not (dest / "best.ckpt").is_file():
        shutil.copy2(src_dir / "best.ckpt", dest / "best.ckpt")
    print(f"[seed] copied step={step} → {dst}")
else:
    print(f"[seed] resume {dst} step={dest_step}")

manifest = {
    "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
    "source_dir": str(src_dir),
    "source_global_step": step,
    "dest_dir": str(dest),
    "target_max_steps": max_steps,
    "eta0": float("${ETA0}"),
    "scheduler": "none",
    "depth": depth,
    "width": width,
    "run_tag": run_tag,
    "note": "L6/D512 flat-LR epoch-1: resume edge η₀=0.1 @40k → 390625 (matched to L4 protocol)",
}
(root / "manifests" / "L6D512_epoch1_seed.json").write_text(
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
  --scheduler none \
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

echo "done pretrain L6 eta0=$ETA0 max_steps=$MAX_STEPS"
