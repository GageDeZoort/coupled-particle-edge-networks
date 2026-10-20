#!/bin/bash
# Transfer-validation (debug): uniform-pool μP η₀×D at short horizon.
#
# Purpose: confirm Adam η=η₀/√D transfers across width under CompleteP +
# uniform output pool — NOT a paper figure; see
#   paper/figures/fig_stability_transfer_validation.md
#
# Recipe:
#   L=4, d_h=32, k=10, R★=0.1, uniform output pool
#   B_gpu=96 × 4 → B_eff=384, Adam, 1000 steps
#   D ∈ {256, 512} × η₀ ∈ {0.05, 0.1, 0.25, 0.5, 1.0}
#
# Array 0–9: width_idx = task//5, eta_idx = task%5
#
#   sbatch scans/jets/scaling_law_L4_upool_xfer_validate_gputest.slurm
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-upool-xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
# Della routes time<=59m onto gputest; do NOT set partition=gputest.
#SBATCH --partition=a100
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=0:59:00
#SBATCH --array=0-9
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_upool_xfer_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_upool_xfer_%A_%a.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

cd /home/jdezoort/coupled-particle-edge-networks/scans/jets
mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs

TASK=${SLURM_ARRAY_TASK_ID:-0}
WIDTHS=(256 512)
ETAS=(0.05 0.1 0.25 0.5 1.0)
W_IDX=$((TASK / 5))
E_IDX=$((TASK % 5))
if (( W_IDX < 0 || W_IDX >= ${#WIDTHS[@]} || E_IDX < 0 || E_IDX >= ${#ETAS[@]} )); then
  echo "array index $TASK out of range (need 0-9)" >&2
  exit 1
fi

WIDTH=${WIDTHS[$W_IDX]}
ETA0=${ETAS[$E_IDX]}
D_HEAD=32
HEAD=$((WIDTH / D_HEAD))
DEPTH=4
BATCH=96
N_GPUS=4
MAX_STEPS="${MAX_STEPS:-1000}"
EPOCHS="${EPOCHS:-2}"
N_TRAIN=1000000
N_VAL=25000
SEED=0
STAR_R=0.1
KNN_K=10
ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law

if (( HEAD * D_HEAD != WIDTH )); then
  echo "D=$WIDTH not divisible by d_h=$D_HEAD" >&2
  exit 1
fi

ETA_TAG=$(python3 -c "e=float('${ETA0}'); print(f'{e:g}'.replace('.','p'))")
R_TAG=$(python3 -c "e=float('${STAR_R}'); print(f'{e:g}'.replace('.','p'))")
RUN_TAG="upool-xfer-b${BATCH}-g${N_GPUS}-s${MAX_STEPS}-dh${D_HEAD}-k${KNN_K}-R${R_TAG}"

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  TASK=$TASK  partition=gpu-test (via a100≤59m)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
echo "study=upool-xfer-validate-gputest  L=$DEPTH  D=$WIDTH  H=$HEAD  d_h=$D_HEAD"
echo "model=capen-llama-att  --uniform-output-pool  eta0=$ETA0  optimizer=adam"
echo "k=$KNN_K  R=$STAR_R  B_gpu=$BATCH  n_gpus=$N_GPUS  B_global=$((BATCH * N_GPUS))"
echo "max_steps=$MAX_STEPS  epochs=$EPOCHS  n_train=$N_TRAIN  run_tag=$RUN_TAG"

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

python test_cpen_jetclass.py \
  --model capen-llama-att \
  --uniform-output-pool \
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
  --val-check-interval 200 \
  --checkpoint-every-n-steps 500 \
  --checkpoint-monitor val_loss \
  --checkpoint-mode min \
  --no-torch-compile
