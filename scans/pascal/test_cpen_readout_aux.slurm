#!/bin/bash
#SBATCH --account=bhanin
#SBATCH --job-name=cpen-voc-aux
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu40
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=0:59:00
#SBATCH --array=0

# Short CPEN PascalVOC-SP readout-asymmetry run (node-only vs node+boundary).
# Notebook: notebooks/pascal/readout_edge_cotrain.ipynb
#
# 1 GPU, 5 epochs, full val every epoch, checkpoint val_f1.
# --partition=a100 + --time=0:59:00 is Della gputest
# (do not set --partition=gpu or --partition=gputest; both are rejected).
#
# Launch from repo root (positionals survive Della dropping --export=ALL):
#   sbatch scans/pascal/test_cpen_readout_aux.slurm node 128 0.25 0
#   sbatch scans/pascal/test_cpen_readout_aux.slurm node+edge 128 0.25 1
# Grid (seeds 0,1,2 by default → error bars):
#   bash scans/pascal/submit_readout_aux.sh
#   SEEDS=0 bash scans/pascal/submit_readout_aux.sh   # smoke, 30 jobs
#
# Uses scans/common/test_cpen.py (not the broken cd scans/pascal; python test_cpen.py).
#
# Stock CPEN --operators incidence: T11 is D_X^{-1} S^T D_E^{-1} S, not binary A.
# On Pascal 2-edges that is A+D up to D_E^{-1}=1/2. No WIRE. Dropout 0.
# Node readout is W_X on X only (not the stock (z_X + T21 z_E)/sqrt(2) mix).

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

REPO_ROOT="/home/jdezoort/coupled-particle-edge-networks"
cd "${REPO_ROOT}/scans/common"

READOUT_MODE="${READOUT_MODE:-${1:-node}}"
WIDTH="${WIDTH:-${2:-512}}"
ETAS="${ETAS:-${3:-1}}"
SEED="${SEED:-${4:-0}}"
DEPTH="${DEPTH:-4}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EDGE_LOSS_WEIGHT="${EDGE_LOSS_WEIGHT:-1.0}"
DATA_ROOT="${DATA_ROOT:-/scratch/gpfs/BHANIN/jdezoort/datasets}"
RUN_ROOT="${RUN_ROOT:-/scratch/gpfs/BHANIN/jdezoort/cpen_runs}"

if [[ "${READOUT_MODE}" == "node+edge" ]]; then
  EDGE_AUX="hard-ce"
else
  EDGE_AUX="hard-ce"
fi

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "READOUT_MODE=${READOUT_MODE}  WIDTH=${WIDTH}  ETAS=${ETAS}  SEED=${SEED}"
echo "DEPTH=${DEPTH}  EPOCHS=${EPOCHS}  BATCH_SIZE=${BATCH_SIZE}"
echo "EDGE_LOSS_WEIGHT=${EDGE_LOSS_WEIGHT}"

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1

python test_cpen.py \
  --dataset pascalvoc-sp \
  --model cpen \
  --operators incidence \
  --operator-backend sparse \
  --operator-normalization degree \
  --readout-mode "${READOUT_MODE}" \
  --seed "${SEED}" \
  --edge-aux "${EDGE_AUX}" \
  --edge-loss-weight "${EDGE_LOSS_WEIGHT}" \
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
