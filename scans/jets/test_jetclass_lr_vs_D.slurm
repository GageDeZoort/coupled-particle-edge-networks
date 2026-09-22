#!/bin/bash
# JetClass η₀ × D study (ParT kin7 transfer recipe).
#
# Fixed model / graph / features; only learning-rate knob and train-set size move.
# Nested stream budgets: smaller D is a prefix of larger D (class-balanced).
#
# Default grid (25 tasks):
#   D    = 1e5, 3e5, 1e6, 3e6, 1e7
#   eta0 = 0.05, 0.1, 0.25, 0.5, 1.0
#   epochs = 1  (override with EPOCHS=3 etc.)
#
# Usage:
#   sbatch scans/jets/test_jetclass_lr_vs_D.slurm
#   sbatch --export=ALL,EPOCHS=2,WIDTH=256 scans/jets/test_jetclass_lr_vs_D.slurm
#   sbatch --array=0-4  # only the five eta0's at the first D, for a smoke pass
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-lr-D
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=a100
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=3:59:00
#SBATCH --array=0-24
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_runs/jc_lr_D_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_runs/jc_lr_D_%A_%a.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1

cd /home/jdezoort/coupled-particle-edge-networks/scans/jets

# ---- grid (row-major: D slow, eta0 fast) ------------------------------------
N_TRAINS=(100000 300000 1000000 3000000 10000000)
ETAS=(0.05 0.1 0.25 0.5 1.0)

TASK=${SLURM_ARRAY_TASK_ID:-0}
N_D=${#N_TRAINS[@]}
N_ETA=${#ETAS[@]}
D_IDX=$((TASK / N_ETA))
ETA_IDX=$((TASK % N_ETA))

if (( D_IDX >= N_D )); then
  echo "array index $TASK out of range (need 0-$((N_D * N_ETA - 1)))" >&2
  exit 1
fi

N_TRAIN=${N_TRAINS[$D_IDX]}
ETA0=${ETAS[$ETA_IDX]}

# ---- fixed recipe (ParT JetClass→TopTagging kin overlap) --------------------
RAW_ROOT="${RAW_ROOT:-/projects/j/jdezoort/jetclass}"
RUN_ROOT="${RUN_ROOT:-/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_runs}"
EPOCHS="${EPOCHS:-1}"
WIDTH="${WIDTH:-256}"
DEPTH="${DEPTH:-4}"
HEADS="${HEADS:-8}"
NUM_PARTICLES="${NUM_PARTICLES:-128}"
BATCH_SIZE="${BATCH_SIZE:-256}"
N_VAL="${N_VAL:-100000}"
SEED="${SEED:-0}"

echo "Host: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  TASK=$TASK  n_train=$N_TRAIN  eta0=$ETA0"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "epochs=$EPOCHS width=$WIDTH depth=$DEPTH heads=$HEADS P=$NUM_PARTICLES"

python test_cpen_jetclass.py \
  --model capen-llama \
  --operators incidence \
  --jetclass-stream \
  --jetclass-raw-root "$RAW_ROOT" \
  --jetclass-features kin7 \
  --jetclass-edge-features part-interaction \
  --star-radius 0.15 \
  --mode sweep_lr \
  --optimizer adam \
  --t-epoch 1 \
  --residual-structure transformer-like \
  --etas "$ETA0" \
  --epochs "$EPOCHS" \
  --data-root "$RAW_ROOT" \
  --depth "$DEPTH" --width "$WIDTH" --heads "$HEADS" \
  --dropout 0.1 \
  --root "$RUN_ROOT" \
  --batch-size "$BATCH_SIZE" \
  --n-gpus 4 \
  --num-workers 4 \
  --normalization uniform \
  --attention-normalization gamma \
  --gamma-11 54 \
  --gamma-12 8.5 \
  --gamma-21 8.5 \
  --gamma-22 5.9 \
  --num-particles "$NUM_PARTICLES" \
  --n-train "$N_TRAIN" \
  --n-val "$N_VAL" \
  --seed "$SEED" \
  --log-every-n-steps 100 \
  --layer-norm \
  --check-val-every-n-epoch 1 \
  --no-torch-compile
