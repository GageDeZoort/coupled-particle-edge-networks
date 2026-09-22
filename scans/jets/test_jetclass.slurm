#!/bin/bash
# CAPEN-Llama-att full JetClass, 25k optimizer steps, 4×A100 gpu-test.
#
#   (L,D,h) = (3, 256, 8)   d_head = 32
#   η0      = 0.05, 0.25, 0.5, 1.0
#   no --n-train → Dfull; --max-steps 25000 → s25k
#
# --batch-size 512 is per GPU. 4 GPUs → global batch 2048 per optimizer step.
# One unique pass is ~48,828 steps; this job stops at 25k. Della gpu-test is
# 59 min (~5–6k steps at ~0.6 s/step), so resubmit the same array task until
# last.ckpt reaches 25k. Resume skips already-consumed jets.
#
#   sbatch scans/jets/test_jetclass.slurm
#   sbatch --array=2  # only η0=0.5
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-att-s25k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=a100
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=0:59:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_runs/jc_att_s25k_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_runs/jc_att_s25k_%A_%a.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

cd /home/jdezoort/coupled-particle-edge-networks/scans/jets

ETAS=(0.05 0.25 0.5 1.0)
TASK=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK < 0 || TASK >= ${#ETAS[@]} )); then
  echo "array index $TASK out of range (need 0-$((${#ETAS[@]} - 1)))" >&2
  exit 1
fi
ETA0=${ETAS[$TASK]}
EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-25000}"
N_VAL="${N_VAL:-25000}"
SEED="${SEED:-0}"

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  TASK=$TASK"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "eta0=$ETA0  arch=L3_D256_h8  n_train=full  max_steps=$MAX_STEPS  batch=512  n_gpus=4"
echo "model=capen-llama-att  graph=star-R=0.2  features=kin7  n_val=$N_VAL"

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1

python test_cpen_jetclass.py \
  --model capen-llama-att \
  --operators incidence \
  --jetclass-stream \
  --jetclass-raw-root /projects/j/jdezoort/jetclass \
  --jetclass-features kin7 \
  --jetclass-edge-features part-interaction \
  --star-radius 0.2 \
  --mode sweep_lr \
  --optimizer adam \
  --t-epoch 1 \
  --residual-structure transformer-like \
  --etas "$ETA0" \
  --epochs "$EPOCHS" \
  --max-steps "$MAX_STEPS" \
  --data-root /scratch/gpfs/BHANIN/jdezoort/datasets/jetclass \
  --depth 3 --width 256 --heads 8 \
  --dropout 0.0 \
  --root /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_runs \
  --batch-size 512 \
  --n-gpus 4 \
  --num-workers 4 \
  --normalization uniform \
  --attention-normalization gamma \
  --num-particles 80 \
  --n-val "$N_VAL" \
  --seed "$SEED" \
  --log-every-n-steps 100 \
  --layer-norm \
  --check-val-every-n-epoch 1 \
  --val-check-interval 2500 \
  --checkpoint-every-n-steps 200 \
  --checkpoint-monitor val_roc_auc \
  --checkpoint-mode max \
  --no-torch-compile
