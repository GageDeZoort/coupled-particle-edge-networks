#!/bin/bash
#SBATCH --account=bhanin
#SBATCH --job-name=cpen-qm9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu80
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=3:59:00
#SBATCH --array=0

# QM9 molecular property regression (CPEN), graph readout with out_dim=1.
#
# TARGET:
#   mu    — dipole moment, Debye (default)
#   gap   — HOMO-LUMO gap, eV
#   u0    — internal energy at 0K, eV
#
# The loss runs on the standardized target; val_mae / test_mae are reported in
# the target's physical unit and the checkpoint monitors val_mae (min).
#
# Requires the incidence cache: sbatch scans/qm9/build_qm9_graphs.slurm
#
#   sbatch scans/qm9/test_cpen_qm9.slurm mu 256 0.25 0
#   bash scans/qm9/submit_qm9.sh

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

REPO_ROOT="/home/jdezoort/coupled-particle-edge-networks"
cd "${REPO_ROOT}/scans/common"

TARGET="${TARGET:-${1:-mu}}"
WIDTH="${WIDTH:-${2:-256}}"
ETAS="${ETAS:-${3:-0.25}}"
SEED="${SEED:-${4:-0}}"
DEPTH="${DEPTH:-6}"
EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-128}"
QM9_LOSS="${QM9_LOSS:-l1}"
DATA_ROOT="${DATA_ROOT:-/scratch/gpfs/BHANIN/jdezoort/datasets}"
RUN_ROOT="${RUN_ROOT:-/scratch/gpfs/BHANIN/jdezoort/cpen_runs}"

HEADS="${HEADS:-}"
if [[ -z "${HEADS}" ]]; then
  case "${WIDTH}" in
    128) HEADS=4 ;;
    256) HEADS=8 ;;
    512) HEADS=16 ;;
    *) HEADS=8 ;;
  esac
fi

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "TARGET=${TARGET}  WIDTH=${WIDTH}  ETAS=${ETAS}  SEED=${SEED}"
echo "DEPTH=${DEPTH}  EPOCHS=${EPOCHS}  BATCH_SIZE=${BATCH_SIZE}  HEADS=${HEADS}  LOSS=${QM9_LOSS}"

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1

python test_cpen.py \
  --dataset qm9 \
  --qm9-target "${TARGET}" \
  --qm9-loss "${QM9_LOSS}" \
  --model cpen \
  --operators incidence \
  --operator-backend sparse \
  --operator-normalization degree \
  --readout-mode graph \
  --heads "${HEADS}" \
  --seed "${SEED}" \
  --mode sweep_lr \
  --optimizer adam \
  --residual-structure transformer-like \
  --etas "${ETAS}" \
  --dropout 0 \
  --epochs "${EPOCHS}" \
  --data-root "${DATA_ROOT}" \
  --depth "${DEPTH}" --width "${WIDTH}" \
  --root "${RUN_ROOT}" \
  --batch-size "${BATCH_SIZE}" \
  --n-gpus 1 \
  --num-workers 4 \
  --normalization uniform \
  --log-every-n-steps 50 \
  --layer-norm \
  --check-val-every-n-epoch 1 \
  --no-torch-compile
