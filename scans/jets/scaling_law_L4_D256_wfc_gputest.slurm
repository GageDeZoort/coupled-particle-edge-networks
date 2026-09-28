#!/bin/bash
# GPU-test: L=4 D=256 H=8, AdamW + warmup→flat→cosine schedule.
#
#   warmup 10k → flat 20k → cosine over ≥1 unique pass → 0.01 × peak LR
#   peak LR = η₀/√D  (floor is NOT 0.01×η₀)
#
#   T_EPOCH=0.5 ETA0=0.25 sbatch --export=ALL,T_EPOCH=0.5,ETA0=0.25 \
#     scans/jets/scaling_law_L4_D256_wfc_gputest.slurm
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-L4D256-wfc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
# Della routes time<=59m onto gputest; do NOT set partition=gputest.
#SBATCH --partition=a100
#SBATCH --exclude=della-l02g6,della-l03g11
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=0:59:00
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_L4D256_wfc_%j.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_L4D256_wfc_%j.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

cd /home/jdezoort/coupled-particle-edge-networks/scans/jets
mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs

DEPTH=4
WIDTH=256
HEAD=8
ETA0="${ETA0:-0.25}"
T_EPOCH="${T_EPOCH:-1}"
BATCH="${BATCH:-128}"
N_GPUS=4
N_TRAIN=100000000
# One unique pass at global batch B×G: ceil(n_train / (B×G)).
COSINE_STEPS="${COSINE_STEPS:-$(python3 -c "import math; print(math.ceil(${N_TRAIN}/(${BATCH}*${N_GPUS})))")}"
WARMUP_STEPS="${WARMUP_STEPS:-10000}"
FLAT_STEPS="${FLAT_STEPS:-20000}"
MAX_STEPS="${MAX_STEPS:-$((WARMUP_STEPS + FLAT_STEPS + COSINE_STEPS))}"
N_VAL="${N_VAL:-25000}"
SEED="${SEED:-0}"
STAR_R=0.2
KNN_K=6
LR_MIN_FRAC="${LR_MIN_FRAC:-0.01}"
T_TAG=$(python3 -c "e=float('${T_EPOCH}'); print(f'{e:g}'.replace('.','p'))")
RUN_TAG="${RUN_TAG:-L4D256-wfc-b${BATCH}-g${N_GPUS}-t${T_TAG}}"
D_HEAD=$((WIDTH / HEAD))

if (( WIDTH % HEAD != 0 )); then
  echo "width $WIDTH not divisible by heads $HEAD" >&2
  exit 1
fi

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "study=L4-D256-wfc  L=$DEPTH  D=$WIDTH  H=$HEAD  d_head=$D_HEAD"
echo "eta0=$ETA0  t_epoch=$T_EPOCH  optimizer=adamw  scheduler=warmup_flat_cosine"
echo "warmup=$WARMUP_STEPS  flat=$FLAT_STEPS  cosine=$COSINE_STEPS  lr_min_frac=$LR_MIN_FRAC"
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
  --optimizer adamw \
  --t-epoch "$T_EPOCH" \
  --scheduler warmup_flat_cosine \
  --warmup-steps "$WARMUP_STEPS" \
  --flat-steps "$FLAT_STEPS" \
  --cosine-steps "$COSINE_STEPS" \
  --lr-decay-min-frac "$LR_MIN_FRAC" \
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
