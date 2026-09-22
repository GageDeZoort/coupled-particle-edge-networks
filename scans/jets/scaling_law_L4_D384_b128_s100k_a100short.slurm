#!/bin/bash
# L=4 D=384 H=12 @ B=128 × 4×A100 — 1.5h gpu-short slices toward 100k.
# For η arms that should not compete for qos=gpu-test MaxJobs=3.
# Watchdog resubmits; same max_steps → resume from last.ckpt.
#
#   ETA0=0.01 sbatch --export=ALL,ETA0=0.01 \
#     scans/jets/scaling_law_L4_D384_b128_s100k_a100short.slurm
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-L4D384-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=a100
#SBATCH --qos=gpu-short
# No nomig/gpu80: land on any free 4×A100 pack (incl. mig-capable nodes).
#SBATCH --exclude=della-l02g6,della-l03g11
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=1:30:00
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_L4D384_a100short_%j.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_L4D384_a100short_%j.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

cd /home/jdezoort/coupled-particle-edge-networks/scans/jets
mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs

DEPTH=4
WIDTH=384
HEAD=12
ETA0="${ETA0:-0.1}"
BATCH="${BATCH:-128}"
N_GPUS=4
MAX_STEPS="${MAX_STEPS:-100000}"
N_VAL="${N_VAL:-25000}"
SEED="${SEED:-0}"
STAR_R=0.2
KNN_K=6
RUN_TAG="${RUN_TAG:-L4D384-a100-b${BATCH}-g${N_GPUS}}"
D_HEAD=$((WIDTH / HEAD))

if (( WIDTH % HEAD != 0 )); then
  echo "width $WIDTH not divisible by heads $HEAD" >&2
  exit 1
fi

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "study=L4-D384-a100short-chain  L=$DEPTH  D=$WIDTH  H=$HEAD  d_head=$D_HEAD"
echo "eta0=$ETA0  starR=$STAR_R  knn=$KNN_K  edge=part-interaction"
echo "max_steps=$MAX_STEPS  B_gpu=$BATCH  n_gpus=$N_GPUS  B_global=$((BATCH * N_GPUS))"
echo "run_tag=$RUN_TAG"

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1

python - <<'PY'
import os
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("No CUDA device visible")
props = torch.cuda.get_device_properties(0)
gib = props.total_memory / 1024**3
print(
    f"GPU0 {props.name} {gib:.1f} GiB  "
    f"count={torch.cuda.device_count()}  "
    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}"
)
if torch.cuda.device_count() < 4:
    sys.exit(f"Need 4 GPUs; visible count={torch.cuda.device_count()}")
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
  --mode sweep_lr \
  --optimizer adam \
  --t-epoch 1 \
  --residual-structure transformer-like \
  --etas "$ETA0" \
  --epochs 1 \
  --max-steps "$MAX_STEPS" \
  --data-root /scratch/gpfs/BHANIN/jdezoort/datasets/jetclass \
  --depth "$DEPTH" --width "$WIDTH" --heads "$HEAD" \
  --dropout 0.0 \
  --root /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law \
  --run-tag "$RUN_TAG" \
  --batch-size "$BATCH" \
  --n-gpus "$N_GPUS" \
  --num-workers 4 \
  --normalization uniform \
  --attention-normalization gamma \
  --num-particles 80 \
  --n-val "$N_VAL" \
  --seed "$SEED" \
  --log-every-n-steps 100 \
  --layer-norm \
  --check-val-every-n-epoch 1 \
  --val-check-interval 1000 \
  --checkpoint-every-n-steps 500 \
  --checkpoint-monitor val_loss \
  --checkpoint-mode min \
  --no-torch-compile
