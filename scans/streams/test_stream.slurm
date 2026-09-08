#!/bin/bash
#SBATCH --account=bhanin
#SBATCH --job-name=cpen-stream
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=0:59:00
#SBATCH --array=0

# Semisupervised CPEN on stellar-stream graphs (one optimizer step per graph).
#
# Defaults (stream-star discovery):
#   --edge-aux product-consistency   soft aux: BCE(p_e, stopgrad(p_u p_v))
#   --checkpoint-monitor val_auroc   best.ckpt = best node val ROC-AUC
#
# Overrides:
#   sbatch --export=ALL,INCLUDE_REAL_STREAMS=1 ...
#   sbatch --export=ALL,EDGE_LOSS_WEIGHT=10 ...
#   sbatch --export=ALL,EDGE_AUX=hard-ce ...          # old hard edge CE
#   sbatch --export=ALL,CHECKPOINT_MONITOR=val_edge_auroc ...
#   sbatch --export=ALL,STREAM_NAME=MOCK_1863,MOCK_3095 ...
#
# Data: pmcoh-pruned PyG graphs (labels audited Aug 2026; edge_y = y_s ∧ y_d).
# Val after every train step (--val-check-interval 1). Obvious stream stars
# re-split ~1/3 into train/val/test (--stream-mask-seed).
#
# CAPEN-Llama is intentionally blocked here: these graphs have N ~ 5e4–8e5
# (dense attention supports), while stream training needs sparse incidence +
# readout_mode=node+edge (CPEN only today).

module purge
module load anaconda3/2023.9
conda activate mamba-env

cd /home/jdezoort/coupled-particle-edge-networks/scans/streams

EDGE_AUX="${EDGE_AUX:-product-consistency}"
EDGE_LOSS_WEIGHT="${EDGE_LOSS_WEIGHT:-1.0}"
CHECKPOINT_MONITOR="${CHECKPOINT_MONITOR:-val_auroc}"
CHECKPOINT_MODE="${CHECKPOINT_MODE:-max}"
STREAM_NAME="${STREAM_NAME:-all}"
DATA_ROOT="${DATA_ROOT:-/projects/BHANIN/adri_gage_gnn_streams/gnn_streams/mocks/streams_per_region_iter3_pm_stepping2_k4_edges_pmcoh/pyg_data_pmcoh_dir0p97_dlog0p25}"

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "STREAM_NAME=${STREAM_NAME}"
echo "INCLUDE_REAL_STREAMS=${INCLUDE_REAL_STREAMS:-0}"
echo "EDGE_AUX=${EDGE_AUX}"
echo "EDGE_LOSS_WEIGHT=${EDGE_LOSS_WEIGHT}"
echo "CHECKPOINT_MONITOR=${CHECKPOINT_MONITOR} (mode=${CHECKPOINT_MODE})"

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1

REAL_FLAG=()
if [[ "${INCLUDE_REAL_STREAMS:-0}" == "1" || "${INCLUDE_REAL_STREAMS:-0}" == "true" ]]; then
  REAL_FLAG=(--include-real-streams)
fi

python test_cpen_stream.py \
  --dataset stream \
  --model cpen \
  --operators incidence \
  --operator-backend sparse \
  --mode sweep_lr \
  --optimizer adamw \
  --t-epoch 5 \
  --residual-structure transformer-like \
  --etas 0.05 \
  --epochs 10 \
  --stream-name "${STREAM_NAME}" \
  "${REAL_FLAG[@]}" \
  --data-root "${DATA_ROOT}" \
  --depth 4 --width 256 \
  --dropout 0.4 \
  --root /scratch/gpfs/BHANIN/jdezoort/cpen_runs \
  --batch-size 1 \
  --n-gpus 1 \
  --num-workers 0 \
  --normalization uniform \
  --operator-normalization degree \
  --log-every-n-steps 1 \
  --layer-norm \
  --check-val-every-n-epoch 1 \
  --val-check-interval 1 \
  --stream-mask-seed 0 \
  --edge-aux "${EDGE_AUX}" \
  --edge-loss-weight "${EDGE_LOSS_WEIGHT}" \
  --checkpoint-monitor "${CHECKPOINT_MONITOR}" \
  --checkpoint-mode "${CHECKPOINT_MODE}" \
  --no-torch-compile
