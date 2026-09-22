#!/bin/bash
#SBATCH --account=bhanin
#SBATCH --job-name=cpen-real-l1so
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=1:19:00
#SBATCH --array=0-4

# Leave-one-stream-out Stage B: each array task masks one holdout stream.
# Default fold streams (override with L1SO_STREAMS comma-list):
#   0:AAU 1:Elqui 2:Chenab 3:Indus 4:Phoenix
#
# Required:
#   INIT_FROM=/path/to/mock/best.ckpt

set -euo pipefail
module purge
module load anaconda3/2023.9
conda activate mamba-env
cd /home/jdezoort/coupled-particle-edge-networks/scans/streams

DATA_ROOT="${DATA_ROOT:-/scratch/gpfs/BHANIN/jgdezoort/streams/galaxies_real}"
RUN_ROOT="${RUN_ROOT:-/scratch/gpfs/BHANIN/jdezoort/cpen_runs/blobs_real_l1so}"
GALAXY="${GALAXY:-0000}"
INIT_FROM="${INIT_FROM:?Set INIT_FROM to a real mock blob best.ckpt path}"
if [[ ! -f "${INIT_FROM}" ]]; then
  echo "INIT_FROM does not exist: ${INIT_FROM}" >&2
  exit 1
fi
mkdir -p "${RUN_ROOT}"

IFS=',' read -r -a FOLDS <<< "${L1SO_STREAMS:-AAU,Elqui,Chenab,Indus,Phoenix}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if (( IDX < 0 || IDX >= ${#FOLDS[@]} )); then
  echo "SLURM_ARRAY_TASK_ID=${IDX} out of range for ${#FOLDS[@]} folds" >&2
  exit 1
fi
HOLDOUT_STREAMS="${FOLDS[$IDX]}"
# sanitize spaces
HOLDOUT_STREAMS="$(echo "${HOLDOUT_STREAMS}" | xargs)"

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1

echo "Host=$(hostname) Job=${SLURM_JOB_ID}_${IDX} fold=${HOLDOUT_STREAMS}"
echo "init_from=${INIT_FROM} runs=${RUN_ROOT}"

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
  --etas "${ETAS:-0.01}" --epochs "${EPOCHS:-25}" \
  --blob-mask-holdout \
  --blob-galaxies "${GALAXY}" \
  --blob-holdout-streams "${HOLDOUT_STREAMS}" \
  --blob-rich-min-mock "${RICH_MIN_MOCK:-10}" \
  --blob-empty-per-rich "${EMPTY_PER_RICH:-1.0}" \
  --blob-cell-val-frac "${CELL_VAL_FRAC:-0.15}" \
  --blob-drop-features "${DROP_FEATURES:-parallax}" \
  --edge-aux "${EDGE_AUX:-hard+product}" \
  --edge-loss-weight "${EDGE_LOSS_WEIGHT:-1.0}" \
  --checkpoint-monitor val_sup_auroc \
  --init-from "${INIT_FROM}"

echo "Done L1SO fold holdout=${HOLDOUT_STREAMS}"
