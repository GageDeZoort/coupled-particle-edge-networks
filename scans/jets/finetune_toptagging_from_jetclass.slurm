#!/bin/bash
# JetClass → TopTagging fine-tune (live star-R + kNN, matched to pretrain).
#
#   INIT_FROM=/path/to/jetclass/last.ckpt \
#   ETA0=0.5 WIDTH=384 HEAD=12 NUM_PARTICLES=80 \
#   sbatch scans/jets/finetune_toptagging_from_jetclass.slurm
#
# Omitting ETA0 defaults to the pretrain η₀ stored in the ckpt.
#
#SBATCH --account=bhanin
#SBATCH --job-name=tt-ft-jc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=a100
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=1:30:00
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/toptagging_finetune/logs/tt_ft_%j.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/toptagging_finetune/logs/tt_ft_%j.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

cd /home/jdezoort/coupled-particle-edge-networks/scans/jets
FT_ROOT="${FT_ROOT:-/scratch/gpfs/BHANIN/jdezoort/cpen_runs/toptagging_finetune}"
mkdir -p "$FT_ROOT/logs"

INIT_FROM="${INIT_FROM:?Set INIT_FROM to a JetClass pretrain Lightning .ckpt}"
DEPTH="${DEPTH:-4}"
WIDTH="${WIDTH:-384}"
HEAD="${HEAD:-12}"
BATCH="${BATCH:-128}"
N_GPUS="${N_GPUS:-4}"
NUM_PARTICLES="${NUM_PARTICLES:-80}"
STAR_R="${STAR_R:-0.2}"
KNN_K="${KNN_K:-6}"
MAX_STEPS="${MAX_STEPS:-20000}"
EPOCHS="${EPOCHS:-50}"
N_VAL="${N_VAL:-20000}"
SEED="${SEED:-0}"
RUN_TAG="${RUN_TAG:-ft-L${DEPTH}D${WIDTH}-b${BATCH}-g${N_GPUS}}"

ETA_ARGS=()
if [[ -n "${ETA0:-}" ]]; then
  ETA_ARGS=(--etas "$ETA0")
fi

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "init_from=$INIT_FROM"
echo "L=$DEPTH D=$WIDTH H=$HEAD P=$NUM_PARTICLES starR=$STAR_R knn=$KNN_K"
echo "eta0=${ETA0:-from-ckpt} max_steps=$MAX_STEPS root=$FT_ROOT"

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1

python - <<'PY'
import os, sys, torch
if not torch.cuda.is_available():
    sys.exit("No CUDA device visible")
print(
    f"GPU0 {torch.cuda.get_device_properties(0).name}  "
    f"count={torch.cuda.device_count()}  "
    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}"
)
PY

python test_cpen_toptagging_ft.py \
  --model capen-llama-att \
  --operators incidence \
  --init-from "$INIT_FROM" \
  --toptagging-live \
  --star-radius "$STAR_R" \
  --live-knn-k "$KNN_K" \
  --jetclass-edge-features part-interaction \
  --num-particles "$NUM_PARTICLES" \
  --mode sweep_lr \
  --optimizer adam \
  --t-epoch 1 \
  --residual-structure transformer-like \
  "${ETA_ARGS[@]}" \
  --epochs "$EPOCHS" \
  --max-steps "$MAX_STEPS" \
  --data-root /scratch/gpfs/BHANIN/jdezoort/datasets/toptagging \
  --depth "$DEPTH" --width "$WIDTH" --heads "$HEAD" \
  --dropout 0.0 \
  --root "$FT_ROOT" \
  --run-tag "$RUN_TAG" \
  --batch-size "$BATCH" \
  --n-gpus "$N_GPUS" \
  --num-workers 4 \
  --normalization uniform \
  --attention-normalization gamma \
  --n-val "$N_VAL" \
  --seed "$SEED" \
  --log-every-n-steps 50 \
  --layer-norm \
  --check-val-every-n-epoch 1 \
  --val-check-interval 500 \
  --checkpoint-every-n-steps 500 \
  --checkpoint-monitor val_roc_auc \
  --checkpoint-mode max \
  --no-torch-compile
