#!/bin/bash
# Unique-pass η₀ sweep, L=4, d_head = D/H = 32 frozen.
#
#   D ∈ {64, 128, 256}  H = (2, 4, 8)
#   η₀ ∈ {0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5}
#   1×A100, B=512  (D=256 does not fit 1g.10gb at B=512)
#   S = 25000 → n = 12.8M unique jets (still one pass of the 100M stream)
#   val every 1000 steps so the η* vs S curve can be read off
#
# Distinct from the D=256/512 η₀=0.25 10M family (s19p532k):
#   --run-tag scale-lrsweep  D12p8M-s25k
#
#   sbatch --time=4:00:00 --array=0-6   scans/jets/scaling_law_L4_lr_sweep.slurm  # D=64
#   sbatch --time=7:00:00 --array=7-13  scans/jets/scaling_law_L4_lr_sweep.slurm  # D=128
#   sbatch --time=16:00:00 --array=14-20 scans/jets/scaling_law_L4_lr_sweep.slurm # D=256
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-scale-L4lr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --qos=gpu-short
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --array=0-6
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_scale_L4lr_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_scale_L4lr_%A_%a.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

cd /home/jdezoort/coupled-particle-edge-networks/scans/jets

mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs

WIDTHS=(64 128 256)
HEADS=(2 4 8)
ETAS=(0.01 0.05 0.1 0.25 0.5 1 2.5)

TASK=${SLURM_ARRAY_TASK_ID:-0}
N_ETA=${#ETAS[@]}
N_WIDTH=${#WIDTHS[@]}
N_TASKS=$((N_WIDTH * N_ETA))
if (( TASK < 0 || TASK >= N_TASKS )); then
  echo "array index $TASK out of range (need 0-$((N_TASKS - 1)))" >&2
  exit 1
fi

W_IDX=$((TASK / N_ETA))
E_IDX=$((TASK % N_ETA))
WIDTH=${WIDTHS[$W_IDX]}
HEAD=${HEADS[$W_IDX]}
ETA0=${ETAS[$E_IDX]}

N_TRAIN="${N_TRAIN:-12800000}"
N_VAL="${N_VAL:-25000}"
SEED="${SEED:-0}"
BATCH=512
N_GPUS=1
GLOBAL_BATCH=$((BATCH * N_GPUS))
MAX_STEPS="${MAX_STEPS:-25000}"
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
echo "study=lrsweep  L=$DEPTH  D=$WIDTH  H=$HEAD  d_head=$D_HEAD"
echo "eta0=$ETA0  n_train=$N_TRAIN  max_steps=$MAX_STEPS  B_gpu=$BATCH  n_gpus=$N_GPUS  B=$GLOBAL_BATCH"
echo "val_check_interval=1000  root=jetclass_scaling_law  run_tag=scale-lrsweep"

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
  --run-tag scale-lrsweep \
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
