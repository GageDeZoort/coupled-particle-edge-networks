#!/bin/bash
#SBATCH --account=bhanin
#SBATCH --job-name=llama-voc-aux
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4 
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu40
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --time=0:59:00
#SBATCH --array=0

# CAPEN-Llama PascalVOC-SP readout-asymmetry run (node-only vs node+boundary).
# Twin of scans/pascal/test_cpen_readout_aux.slurm with attention instead of
# linear incidence operators. Edge–edge *attention* (M_22) is OFF: Pascal graphs
# pad to M ~ thousands, so dense (B, M, M) attention OOMs. f_22 = S S^T h_e
# (--incidence-m22): line-graph message passing via two (M, N) matmuls. No WIRE.
# Dropout 0. Node logits are W_X on X^{(L)} only (not the stock mix).
# Full M_22 attention: INCIDENCE_M22=0. Identity f_22: IDENTITY_M22=1.
#
# 1 GPU, checkpoint val_f1. --partition=a100 + --time=0:59:00 is Della gputest
# (do not set --partition=gpu or --partition=gputest; both are rejected).
#
# Launch from repo root (positionals survive Della dropping --export=ALL):
#   sbatch scans/pascal/test_capen_llama_readout_aux.slurm node 256 0.25 0
#   sbatch scans/pascal/test_capen_llama_readout_aux.slurm node+edge 256 0.25 0
# Grid:
#   bash scans/pascal/submit_capen_llama_readout_aux.sh
#
# Uses scans/common/test_cpen.py. HEADS must divide WIDTH (default 8).

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

REPO_ROOT="/home/jdezoort/coupled-particle-edge-networks"
cd "${REPO_ROOT}/scans/common"

READOUT_MODE="${READOUT_MODE:-${1:-node}}"
WIDTH="${WIDTH:-${2:-512}}"
ETAS="${ETAS:-${3:-0.25}}"
SEED="${SEED:-${4:-0}}"
DEPTH="${DEPTH:-6}"
HEADS="${HEADS:-16}"
EPOCHS="${EPOCHS:-25}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EDGE_LOSS_WEIGHT="${EDGE_LOSS_WEIGHT:-1.0}"
IDENTITY_M22="${IDENTITY_M22:-0}"
INCIDENCE_M22="${INCIDENCE_M22:-1}"
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
echo "DEPTH=${DEPTH}  HEADS=${HEADS}  EPOCHS=${EPOCHS}  BATCH_SIZE=${BATCH_SIZE}"
echo "EDGE_LOSS_WEIGHT=${EDGE_LOSS_WEIGHT}  IDENTITY_M22=${IDENTITY_M22}  INCIDENCE_M22=${INCIDENCE_M22}"

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1

IDENTITY_M22_FLAG=()
INCIDENCE_M22_FLAG=()
if [[ "${IDENTITY_M22}" == "1" || "${IDENTITY_M22}" == "true" || "${IDENTITY_M22}" == "yes" ]]; then
  IDENTITY_M22_FLAG=(--identity-m22)
fi
if [[ "${INCIDENCE_M22}" == "1" || "${INCIDENCE_M22}" == "true" || "${INCIDENCE_M22}" == "yes" ]]; then
  INCIDENCE_M22_FLAG=(--incidence-m22)
fi

python test_cpen.py \
  --dataset pascalvoc-sp \
  --model capen-llama \
  --operators incidence \
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
  --depth "${DEPTH}" --width "${WIDTH}" --heads "${HEADS}" \
  --root "${RUN_ROOT}" \
  --batch-size "${BATCH_SIZE}" \
  --n-gpus 4 \
  --num-workers 4 \
  --normalization uniform \
  --log-every-n-steps 50 \
  --layer-norm \
  --check-val-every-n-epoch 1 \
  --no-torch-compile \
  "${IDENTITY_M22_FLAG[@]}" \
  "${INCIDENCE_M22_FLAG[@]}"
