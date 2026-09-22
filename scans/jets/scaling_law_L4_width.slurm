#!/bin/bash
# Unique-pass width family for the JetClass scaling-law plot.
#
#   L = 4 frozen  (same depth Vigl et al. used)
#   d_head = 32   H = D / 32
#   D, H = (64, 2), (128, 4), (256, 8), (512, 16)
#   1×A100, B=512  (not 4-GPU: same unique-pass FLOPs at n, much shorter queue)
#   n = 10M unique jets → S = ceil(10e6 / 512) = 19532
#   5h gpu-short is the fallback in this file; the watchdog overrides --time
#   from remaining steps so D=256/512 do not sit behind a full 5h request.
#
# Checkpoints: /scratch/.../jetclass_scaling_law/  plus --run-tag scale
#
#   sbatch scans/jets/scaling_law_L4_width.slurm
#   sbatch --array=2 scans/jets/scaling_law_L4_width.slurm  # D=256 only
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-scale-L4W
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --qos=gpu-short
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_scale_L4W_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_scale_L4W_%A_%a.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

cd /home/jdezoort/coupled-particle-edge-networks/scans/jets

mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs

# d_head = 32 → H = D/32
WIDTHS=(64 128 256 512)
HEADS=(2 4 8 16)
TASK=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK < 0 || TASK >= ${#WIDTHS[@]} )); then
  echo "array index $TASK out of range (need 0-$((${#WIDTHS[@]} - 1)))" >&2
  exit 1
fi
WIDTH=${WIDTHS[$TASK]}
HEAD=${HEADS[$TASK]}
ETA0="${ETA0:-0.25}"
N_TRAIN="${N_TRAIN:-10000000}"
N_VAL="${N_VAL:-25000}"
SEED="${SEED:-0}"
BATCH=512
N_GPUS=1
GLOBAL_BATCH=$((BATCH * N_GPUS))
MAX_STEPS=$(( (N_TRAIN + GLOBAL_BATCH - 1) / GLOBAL_BATCH ))
DEPTH=4
D_HEAD=$((WIDTH / HEAD))
if (( WIDTH % HEAD != 0 )); then
  echo "width $WIDTH not divisible by heads $HEAD" >&2
  exit 1
fi

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  TASK=$TASK"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "study=scaling-law  L=$DEPTH  D=$WIDTH  H=$HEAD  d_head=$D_HEAD"
echo "eta0=$ETA0  n_train=$N_TRAIN  max_steps=$MAX_STEPS  B_gpu=$BATCH  n_gpus=$N_GPUS  B=$GLOBAL_BATCH"
echo "val_check_interval=1000  root=jetclass_scaling_law  run_tag=scale"

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1

python test_cpen_jetclass.py \
  --model capen-llama-att \
  --operators incidence \
  --jetclass-stream \
  --jetclass-raw-root /projects/j/jdezoort/jetclass \
  --jetclass-features kin7 \
  --jetclass-edge-features part-interaction \
  --star-radius 0.2 \
  --mode sweep_lr \
  --optimizer adam \
  --t-epoch 1 \
  --residual-structure transformer-like \
  --etas "$ETA0" \
  --epochs 1 \
  --max-steps "$MAX_STEPS" \
  --n-train "$N_TRAIN" \
  --data-root /scratch/gpfs/BHANIN/jdezoort/datasets/jetclass \
  --depth "$DEPTH" --width "$WIDTH" --heads "$HEAD" \
  --dropout 0.0 \
  --root /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law \
  --run-tag scale \
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
  --checkpoint-every-n-steps 200 \
  --checkpoint-monitor val_loss \
  --checkpoint-mode min \
  --no-torch-compile
