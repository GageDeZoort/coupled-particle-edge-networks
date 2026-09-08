#!/bin/bash
#SBATCH --account=bhanin
#SBATCH --job-name=cpen-mnist-ro
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu80
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=1:29:00
#SBATCH --array=0

# MNISTSuperpixels graph-classification readout ablation (CPEN).
# Notebook / later: notebooks/mnist/ (not yet).
#
# READOUT:
#   node       — pool z_X only
#   node+edge  — (z_X + z_E)/sqrt(2) after pooling
#   attn       — class-token MHSA over live nodes+edges (--heads from D)
#
# Unlike Pascal there is no edge CE. Metric is graph accuracy.
# Stock CPEN graph mode already pools; the node vs node+edge mix still needs
# an explicit x-only vs (z_X+z_E)/√2 graph head (same idea as Pascal's
# x_only_node_readout, but after pooling).
#
# Graph classification: --dataset mnist-sp, checkpoint val_acc.
#
#   sbatch scans/mnist/test_cpen_readout.slurm node 256 0.25 0
#   bash scans/mnist/submit_readout.sh

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

REPO_ROOT="/home/jdezoort/coupled-particle-edge-networks"
cd "${REPO_ROOT}/scans/common"

READOUT_MODE="${READOUT_MODE:-${1:-node}}"
WIDTH="${WIDTH:-${2:-512}}"
ETAS="${ETAS:-${3:-0.01}}"
SEED="${SEED:-${4:-0}}"
DEPTH="${DEPTH:-6}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-256}"
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
echo "READOUT_MODE=${READOUT_MODE}  WIDTH=${WIDTH}  ETAS=${ETAS}  SEED=${SEED}"
echo "DEPTH=${DEPTH}  EPOCHS=${EPOCHS}  BATCH_SIZE=${BATCH_SIZE}  HEADS=${HEADS}"

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1

python test_cpen.py \
  --dataset mnist-sp \
  --model cpen \
  --operators incidence \
  --operator-backend sparse \
  --operator-normalization degree \
  --readout-mode "${READOUT_MODE}" \
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
