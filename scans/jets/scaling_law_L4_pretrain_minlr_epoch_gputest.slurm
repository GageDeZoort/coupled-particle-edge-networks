#!/bin/bash
# JetClass pretrain — L4/D256 Adam · epoch-3 at constant min LR (0.01× peak).
#
# Seeds from finished cosine epoch (step ≥ 781250). Peak was η₀=0.1 →
# base_lr = 0.1/√256; cosine floor = 0.01× that. Here we set η₀=0.001 so
# base_lr = 0.001/√256 = same floor, with scheduler=none for one unique pass
# (781250 → 1171875).
#
#   bash scans/jets/start_pretrain_L4_minlr_epoch_gputest_watchdogs.sh
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-pretrain-L4min
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=a100
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=0:59:00
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_pretrain/logs/jc_pretrain_L4min_gt_%j.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_pretrain/logs/jc_pretrain_L4min_gt_%j.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

REPO=/home/jdezoort/coupled-particle-edge-networks
cd "$REPO/scans/jets"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_pretrain
mkdir -p "$ROOT/logs"

# 0.01 × original η₀=0.1 → same absolute LR as cosine floor.
ETA0=0.001
DEPTH=4
WIDTH=256
D_HEAD=32
HEAD=$((WIDTH / D_HEAD))
STAR_R=0.1
KNN_K=7
BATCH=64
N_GPUS=4
N_TRAIN=100000000
EPOCH1_STEPS=$((N_TRAIN / (BATCH * N_GPUS)))   # 390625
COSINE_END=$((2 * EPOCH1_STEPS))               # 781250
MAX_STEPS=$((3 * EPOCH1_STEPS))                # 1171875
SEED=0
N_VAL=25000
EPOCHS=100
R_TAG=$(python3 -c "e=float('${STAR_R}'); print(f'{e:g}'.replace('.','p'))")
RUN_TAG="pretrain-minlr-b${BATCH}-g${N_GPUS}-s${MAX_STEPS}-dh${D_HEAD}-k${KNN_K}-R${R_TAG}-eta0p001"
COS_TAG="pretrain-cos-b64-g4-s781250"

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}  partition=gpu-test (via a100≤59m)"
echo "study=jetclass-pretrain-minlr-epoch  L=$DEPTH  D=$WIDTH  H=$HEAD"
echo "eta0=$ETA0  scheduler=none  (0.01× peak of η₀=0.1 run)"
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

# Seed min-LR run dir from finished cosine last.ckpt.
python - <<PY
import json, shutil
from datetime import datetime
from pathlib import Path
import torch

root = Path("${ROOT}")
sweep = root / "jetclass_output/adam/sweep_lr"
cos_tag = "${COS_TAG}"
run_tag = "${RUN_TAG}"
cos_end = int("${COSINE_END}")
min_step = cos_end - 2000

cands = [
    p for p in sweep.glob(f"*4_256_0p1*{cos_tag}*")
    if p.is_dir() and not (p / "SUPERSEDED.txt").exists()
]
assert cands, f"no cosine run dir matching {cos_tag}"
src_dir = max(cands, key=lambda p: p.stat().st_mtime)
src = src_dir / "last.ckpt"
assert src.is_file(), f"missing {src}"
payload = torch.load(src, map_location="cpu", weights_only=False)
step = int(payload.get("global_step", 0) or 0)
assert step >= min_step, f"cosine ckpt step={step} < {min_step}"
print(f"[seed] cosine {src} global_step={step}")

dests = [
    p for p in sweep.glob(f"*4_256*{run_tag}*")
    if p.is_dir() and not (p / "SUPERSEDED.txt").exists()
]
if dests:
    dest = max(dests, key=lambda p: p.stat().st_mtime)
else:
    dest = sweep / (
        f"capen-llama-att_4_256_0p001_ln_wfc_op-incidence_onorm-degree_res-transformer-like"
        f"_h8_ntr100M_nval25k_ntst0_nf7_gstar0p1_stream-kin7-partint-p80-knn7-noloop-sk7"
        f"-D100M-s1p17M-seed0-{run_tag}"
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
if (not dst.is_file()) or step > dest_step:
    shutil.copy2(src, dst)
    print(f"[seed] copied step={step} → {dst}")
else:
    print(f"[seed] keep existing {dst} step={dest_step}")

manifest = {
    "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
    "source_dir": str(src_dir),
    "source_global_step": step,
    "dest_dir": str(dest),
    "target_max_steps": int("${MAX_STEPS}"),
    "eta0": float("${ETA0}"),
    "scheduler": "none",
    "run_tag": run_tag,
    "note": "Epoch-3 constant LR at cosine floor (0.01× peak of η₀=0.1)",
}
(root / "manifests").mkdir(exist_ok=True)
(root / "manifests" / "L4D256_minlr_epoch3_seed.json").write_text(
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

echo "done pretrain L4 minlr epoch max_steps=$MAX_STEPS eta0=$ETA0"
