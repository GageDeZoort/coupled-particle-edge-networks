#!/bin/bash
# M11-only D=768 η₀ sweep on 1×A100-80GB (B=384, matched global batch).
#
# Extends panel (c) above D=512. Fixed d_h=32 ⇒ H=24. Adam, 4000 steps.
# Same recipe as m11-b96-g4-s4000-dh32 except 1 GPU @ B=384 (tag -a80).
#
# Array (5 tasks): η₀ ∈ {0.05, 0.1, 0.25, 0.5, 1.0}
#
#   sbatch scans/jets/scaling_law_L4_m11_D768_eta_a10080.slurm
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-m11-D768-a80
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --constraint=gpu80
#SBATCH --qos=gpu-short
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --array=0-4
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_m11_D768_a80_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_m11_D768_a80_%A_%a.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

cd /home/jdezoort/coupled-particle-edge-networks/scans/jets
mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs

TASK=${SLURM_ARRAY_TASK_ID:-0}
ETAS=(0.05 0.1 0.25 0.5 1.0)
if (( TASK < 0 || TASK >= ${#ETAS[@]} )); then
  echo "array index $TASK out of range (need 0-4)" >&2
  exit 1
fi

ETA0=${ETAS[$TASK]}
WIDTH=768
D_HEAD=32
HEAD=$((WIDTH / D_HEAD))
DEPTH=4
MODEL=baseline-transformer
N_TRAIN=1000000
N_VAL=25000
SEED=0
BATCH=384
N_GPUS=1
MAX_STEPS="${MAX_STEPS:-4000}"
EPOCHS="${EPOCHS:-2}"
ETA_TAG=$(python3 -c "e=float('${ETA0}'); print(f'{e:g}'.replace('.','p'))")
RUN_TAG="m11-b${BATCH}-g${N_GPUS}-s${MAX_STEPS}-dh${D_HEAD}-a80"

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  TASK=$TASK"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
echo "study=m11-eta-a10080  L=$DEPTH  D=$WIDTH  H=$HEAD  d_head=$D_HEAD"
echo "model=$MODEL  (M11 all-to-all, no edges)  eta0=$ETA0  optimizer=adam"
echo "B_gpu=$BATCH  n_gpus=$N_GPUS  B_global=$((BATCH * N_GPUS))  max_steps=$MAX_STEPS  epochs=$EPOCHS"
echo "run_tag=$RUN_TAG"

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("No CUDA device visible")
p = torch.cuda.get_device_properties(0)
print(f"GPU0 {p.name} {p.total_memory/1024**3:.1f} GiB")
if p.total_memory < 60 * 1024**3:
    sys.exit(f"Need ~80GB A100; got {p.total_memory/1024**3:.1f} GiB")
PY

python test_cpen_jetclass.py \
  --model "$MODEL" \
  --operators incidence \
  --jetclass-stream \
  --jetclass-raw-root /projects/j/jdezoort/jetclass \
  --jetclass-features kin7 \
  --mode sweep_lr \
  --optimizer adam \
  --t-epoch 1 \
  --residual-structure transformer-like \
  --etas "$ETA0" \
  --epochs "$EPOCHS" \
  --max-steps "$MAX_STEPS" \
  --n-train "$N_TRAIN" \
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
  --log-every-n-steps 50 \
  --layer-norm \
  --check-val-every-n-epoch 1 \
  --val-check-interval 500 \
  --checkpoint-every-n-steps 500 \
  --checkpoint-monitor val_loss \
  --checkpoint-mode min \
  --no-torch-compile
