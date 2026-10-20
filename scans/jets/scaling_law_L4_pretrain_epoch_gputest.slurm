#!/bin/bash
# JetClass pretrain continue — L=4 D=256 Adam η₀=0.1 from unique-40k ckpt → 1 epoch.
#
#   Home:  /scratch/.../jetclass_pretrain/  (not scaling_law sweeps)
#   Seed:  edge η cell @ step 40000 (k★=7, R★=0.1, B=64×4)
#   Target: max_steps = floor(1e8/256) = 390625  (= one unique epoch)
#   Tag:   pretrain-b64-g4-s390625-dh32-k7-R0p1
#
# Della routes time≤59m onto gputest. Lightning resumes last.ckpt under the
# fixed run dir (seeded once from the 40k sweep). Prefer the watchdog:
#   bash scans/jets/start_pretrain_L4_epoch_gputest_watchdogs.sh
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-pretrain-L4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=a100
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=0:59:00
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_pretrain/logs/jc_pretrain_L4_gt_%j.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_pretrain/logs/jc_pretrain_L4_gt_%j.err

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
# One unique epoch over full JetClass train @ B_eff=256.
N_TRAIN=100000000
MAX_STEPS=$((N_TRAIN / (BATCH * N_GPUS)))
SEED=0
N_VAL=25000
EPOCHS=1
R_TAG=$(python3 -c "e=float('${STAR_R}'); print(f'{e:g}'.replace('.','p'))")
RUN_TAG="pretrain-b${BATCH}-g${N_GPUS}-s${MAX_STEPS}-dh${D_HEAD}-k${KNN_K}-R${R_TAG}"

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}  partition=gpu-test (via a100≤59m)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
echo "study=jetclass-pretrain-epoch  L=$DEPTH  D=$WIDTH  H=$HEAD  d_h=$D_HEAD"
echo "model=capen-llama-att  eta0=$ETA0  optimizer=adam  k=$KNN_K  R=$STAR_R  seed=$SEED"
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

# Fail fast if seed ckpt missing. Live run dir uses format_count(max_steps)=s390p625k.
python - <<PY
from pathlib import Path
root = Path("${ROOT}") / "jetclass_output/adam/sweep_lr"
cands = [p for p in root.glob("*_${WIDTH}_0p1_*${RUN_TAG}") if p.is_dir()]
live = [p for p in cands if "s390p625k" in p.name and not (p / "SUPERSEDED.txt").exists()]
dirs = live or [p for p in cands if not (p / "SUPERSEDED.txt").exists()]
assert dirs, f"No pretrain run dir matching *${RUN_TAG} under {root}"
# Prefer newest
d = max(dirs, key=lambda p: p.stat().st_mtime)
ckpt = d / "last.ckpt"
assert ckpt.is_file(), f"Missing seed ckpt {ckpt}"
print(f"[seed] resume from {ckpt}")
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

echo "done pretrain L4 eta0=$ETA0 max_steps=$MAX_STEPS"
