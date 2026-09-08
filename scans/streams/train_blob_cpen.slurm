#!/bin/bash
#SBATCH --account=bhanin
#SBATCH --job-name=cpen-blob
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=1:59:00

# CPEN on blob graphs. One blob per step.
#   L = node_CE + w * (edge_CE + consistency)
# Train is 50/50 mock-rich vs empty; val/test stay at natural occupancy.
#
# Turn a knob with --export (do not put commas inside a value):
#   EPOCHS DEPTH WIDTH ETAS DROPOUT     model / schedule
#   RICH_MIN_MOCK EMPTY_PER_RICH        train split
#   EDGE_AUX EDGE_LOSS_WEIGHT           loss  (hard+product | hard-ce | product-consistency)
#   DROP_FEATURES                       node ablations; default parallax
#   DATA_ROOT RUN_ROOT                  paths

set -u
module purge
module load anaconda3/2023.9
conda activate mamba-env
cd /home/jdezoort/coupled-particle-edge-networks/scans/streams

DATA_ROOT="${DATA_ROOT:-/scratch/gpfs/BHANIN/jgdezoort/streams/galaxies}"
RUN_ROOT="${RUN_ROOT:-/scratch/gpfs/BHANIN/jdezoort/cpen_runs/blobs}"
mkdir -p "${RUN_ROOT}"

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1

echo "Host=$(hostname) Job=${SLURM_JOB_ID} GPU=${CUDA_VISIBLE_DEVICES:-unset}"
echo "data=${DATA_ROOT}"
echo "runs=${RUN_ROOT}"
echo "model depth=${DEPTH:-4} width=${WIDTH:-256} drop=${DROPOUT:-0.2} eta=${ETAS:-0.05} epochs=${EPOCHS:-40}"
echo "split rich>=${RICH_MIN_MOCK:-100} empty:rich=${EMPTY_PER_RICH:-1.0}"
echo "loss ${EDGE_AUX:-hard+product} w=${EDGE_LOSS_WEIGHT:-1.0}  drop=${DROP_FEATURES:-parallax}"

python test_cpen_blob.py \
  --model cpen --operators incidence --operator-backend sparse \
  --mode sweep_lr --optimizer adamw --t-epoch 5 --scheduler cosine \
  --residual-structure transformer-like --layer-norm \
  --normalization uniform --operator-normalization degree \
  --batch-size 1 --n-gpus 1 --num-workers 4 \
  --check-val-every-n-epoch 1 --log-every-n-steps 25 \
  --no-torch-compile \
  --data-root "${DATA_ROOT}" --root "${RUN_ROOT}" \
  --depth "${DEPTH:-4}" --width "${WIDTH:-256}" --dropout "${DROPOUT:-0.2}" \
  --etas "${ETAS:-0.2}" --epochs "${EPOCHS:-20}" \
  --blob-rich-min-mock "${RICH_MIN_MOCK:-100}" \
  --blob-empty-per-rich "${EMPTY_PER_RICH:-1.0}" \
  --blob-drop-features "${DROP_FEATURES:-parallax}" \
  --edge-aux "${EDGE_AUX:-hard+product}" \
  --edge-loss-weight "${EDGE_LOSS_WEIGHT:-1.0}"

echo "Done blob training"
