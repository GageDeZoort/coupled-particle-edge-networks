#!/bin/bash
# L=4 D=1024 H=32 CAPEN-Llama-att, longer fair-width run.
#
#   4×A100-80GB  (gpu80 — B=128 OOMs on 40GB for this width)
#   batch=128 per GPU → global 512
#   max_steps=100000
#   η₀=0.1  (best on the 5k transfer grid for this arch)
#   stack: kin7 + part-interaction + knn=6 + star-R=0.2
#
#   sbatch scans/jets/scaling_law_L4_D1024_b128_s100k.slurm
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-L4D1024-s100k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=a100
#SBATCH --constraint=gpu80
#SBATCH --qos=gpu-long
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=6-00:00:00
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_L4D1024_s100k_%j.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_L4D1024_s100k_%j.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

cd /home/jdezoort/coupled-particle-edge-networks/scans/jets
mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs

DEPTH=4
WIDTH=1024
HEAD=32
ETA0="${ETA0:-0.1}"
BATCH="${BATCH:-128}"
N_GPUS=4
MAX_STEPS="${MAX_STEPS:-100000}"
N_VAL="${N_VAL:-25000}"
SEED="${SEED:-0}"
STAR_R=0.2
KNN_K=6
D_HEAD=$((WIDTH / HEAD))

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "study=L4-D1024-long  L=$DEPTH  D=$WIDTH  H=$HEAD  d_head=$D_HEAD"
echo "eta0=$ETA0  starR=$STAR_R  knn=$KNN_K  edge=part-interaction"
echo "max_steps=$MAX_STEPS  B_gpu=$BATCH  n_gpus=$N_GPUS  B_global=$((BATCH * N_GPUS))"
echo "n_train=full (stream)  n_val=$N_VAL"

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
if gib < 70:
    sys.exit(
        f"Need A100 80GB (#SBATCH --constraint=gpu80); this node has {gib:.1f} GiB. "
        "D=1024 at B=128 does not fit on 40GB."
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
  --run-tag L4D1024-s100k-b${BATCH}-g${N_GPUS} \
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
  --val-check-interval 5000 \
  --checkpoint-every-n-steps 2000 \
  --checkpoint-monitor val_loss \
  --checkpoint-mode min \
  --no-torch-compile
