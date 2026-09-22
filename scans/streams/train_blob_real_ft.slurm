#!/bin/bash
#SBATCH --account=bhanin
#SBATCH --job-name=cpen-real-ft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=1:19:00

# Fine-tune a mock-pretrained blob CPEN on real galaxy 0000.
# LEGACY cell-split protocol (supervise train-stream cells only).
# Preferred: train_blob_real_ft_mask.slurm (label-holdout / Stage B).
#
# Required:
#   INIT_FROM=/path/to/mock/best.ckpt
# Optional overrides via --export:
#   DATA_ROOT RUN_ROOT GALAXY RICH_MIN_MOCK ETAS EPOCHS DEPTH WIDTH
#   TRAIN_STREAMS HOLDOUT_STREAMS CELL_VAL_FRAC

set -euo pipefail
module purge
module load anaconda3/2023.9
conda activate mamba-env
cd /home/jdezoort/coupled-particle-edge-networks/scans/streams

DATA_ROOT="${DATA_ROOT:-/scratch/gpfs/BHANIN/jgdezoort/streams/galaxies_real}"
RUN_ROOT="${RUN_ROOT:-/scratch/gpfs/BHANIN/jdezoort/cpen_runs/blobs_real_ft}"
GALAXY="${GALAXY:-0000}"
INIT_FROM="${INIT_FROM:?Set INIT_FROM to a real mock blob best.ckpt path (not a placeholder)}"
if [[ ! -f "${INIT_FROM}" ]]; then
  echo "INIT_FROM does not exist: ${INIT_FROM}" >&2
  exit 1
fi
mkdir -p "${RUN_ROOT}"

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1

echo "Host=$(hostname) Job=${SLURM_JOB_ID} GPU=${CUDA_VISIBLE_DEVICES:-unset}"
echo "data=${DATA_ROOT} galaxy=${GALAXY}"
echo "init_from=${INIT_FROM}"
echo "runs=${RUN_ROOT}"
echo "model depth=${DEPTH:-4} width=${WIDTH:-256} eta=${ETAS:-0.05} epochs=${EPOCHS:-10}"
echo "rich>=${RICH_MIN_MOCK:-10} empty:rich=${EMPTY_PER_RICH:-1.0}"

EXTRA=()
if [[ -n "${TRAIN_STREAMS:-}" ]]; then
  EXTRA+=(--blob-train-streams "${TRAIN_STREAMS}")
fi
if [[ -n "${HOLDOUT_STREAMS:-}" ]]; then
  EXTRA+=(--blob-holdout-streams "${HOLDOUT_STREAMS}")
fi

python test_cpen_blob.py \
  --model cpen --operators incidence --operator-backend sparse \
  --mode sweep_lr --optimizer adamw --t-epoch 10 --scheduler cosine \
  --residual-structure transformer-like --layer-norm \
  --normalization uniform --operator-normalization degree \
  --batch-size 1 --n-gpus 1 --num-workers 4 \
  --check-val-every-n-epoch 1 --log-every-n-steps 25 \
  --no-torch-compile \
  --data-root "${DATA_ROOT}" --root "${RUN_ROOT}" \
  --depth "${DEPTH:-4}" --width "${WIDTH:-256}" --dropout "${DROPOUT:-0.1}" \
  --etas "${ETAS:-0.005}" --epochs "${EPOCHS:-25}" \
  --blob-cell-split \
  --blob-galaxies "${GALAXY}" \
  --blob-rich-min-mock "${RICH_MIN_MOCK:-10}" \
  --blob-empty-per-rich "${EMPTY_PER_RICH:-1.0}" \
  --blob-cell-val-frac "${CELL_VAL_FRAC:-0.15}" \
  --blob-drop-features "${DROP_FEATURES:-parallax}" \
  --edge-aux "${EDGE_AUX:-hard+product}" \
  --edge-loss-weight "${EDGE_LOSS_WEIGHT:-1.0}" \
  --init-from "${INIT_FROM}" \
  "${EXTRA[@]}"

echo "Done real-0000 cell-split fine-tune"
