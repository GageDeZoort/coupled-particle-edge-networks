#!/bin/bash
# Baby-model (4, 256, 8) full JetClass 1-epoch η0 sweep.
#
# ~10 h / epoch at this size; 24 h wall so a slow node still finishes.
# last.ckpt every 2000 steps (~6 min). Resubmitting the same array task
# resumes weights and skips already-seen jets.
#
#   sbatch scans/jets/test_jetclass.slurm
#   sbatch --array=0-0  # only η0=0.05
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-full-baby
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=a100
#SBATCH --qos=gpu-short
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=23:59:00
#SBATCH --array=0-4
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_runs/jc_full_baby_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_runs/jc_full_baby_%A_%a.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

cd /home/jdezoort/coupled-particle-edge-networks/scans/jets

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  TASK=${SLURM_ARRAY_TASK_ID:-0}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1

ETAS=(0.05 0.1 0.25 0.5 1.0)
TASK=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK < 0 || TASK >= ${#ETAS[@]} )); then
  echo "array index $TASK out of range (need 0-$((${#ETAS[@]} - 1)))" >&2
  exit 1
fi
ETA0=${ETAS[$TASK]}
echo "eta0=$ETA0  arch=L4_D256_h8  n_train=full  epochs=1"

python test_cpen_jetclass.py \
  --model capen-llama \
  --operators incidence \
  --jetclass-stream \
  --jetclass-raw-root /projects/j/jdezoort/jetclass \
  --jetclass-features kin7 \
  --star-radius 0.2 \
  --mode sweep_lr \
  --optimizer adam \
  --t-epoch 1 \
  --residual-structure transformer-like \
  --etas "$ETA0" \
  --epochs 1 \
  --data-root /scratch/gpfs/BHANIN/jdezoort/datasets/jetclass \
  --depth 4 --width 256 --heads 8 \
  --dropout 0.0 \
  --root /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_runs \
  --batch-size 512 \
  --n-gpus 4 \
  --num-workers 4 \
  --normalization uniform \
  --attention-normalization gamma \
  --num-particles 80 \
  --log-every-n-steps 100 \
  --layer-norm \
  --check-val-every-n-epoch 1 \
  --val-check-interval 5000 \
  --checkpoint-every-n-steps 2000 \
  --checkpoint-monitor val_roc_auc \
  --checkpoint-mode max \
  --no-torch-compile
