#!/bin/bash
# CAPEN-Llama-att full JetClass, 50k optimizer steps, 2×H200 (ailab).
#
#   (L,D,h) = (5, 1024, 16)   d_head = 64
#   η0      = 0.25
#   no --n-train → Dfull; --max-steps 50000 → s50k
#
# --batch-size 512 is per GPU. 2 GPUs → global batch 1024 per optimizer step.
# The stream is DDP-sharded (~97,660 steps / unique pass); this job stops at 50k.
#
#   sbatch scans/jets/test_jetclass_ailab_large.slurm
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-att-large-s50k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=16
#SBATCH --partition=ailab
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH --time=4-00:00:00
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_runs/jc_att_large_s50k_%j.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_runs/jc_att_large_s50k_%j.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

cd /home/jdezoort/coupled-particle-edge-networks/scans/jets

ETA0="${ETA0:-0.25}"
EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-50000}"
N_VAL="${N_VAL:-25000}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-512}"

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "eta0=$ETA0  arch=L5_D1024_h16  n_train=full  max_steps=$MAX_STEPS  batch=$BATCH_SIZE  n_gpus=2"
echo "model=capen-llama-att  graph=star-R=0.2  features=kin7  n_val=$N_VAL  partition=ailab"

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
  --depth 5 --width 1024 --heads 16 \
  --dropout 0.0 \
  --root /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_runs \
  --batch-size "$BATCH_SIZE" \
  --n-gpus 2 \
  --num-workers 4 \
  --normalization uniform \
  --attention-normalization gamma \
  --num-particles 80 \
  --n-val "$N_VAL" \
  --seed "$SEED" \
  --log-every-n-steps 100 \
  --layer-norm \
  --check-val-every-n-epoch 1 \
  --val-check-interval 5000 \
  --checkpoint-every-n-steps 2000 \
  --checkpoint-monitor val_roc_auc \
  --checkpoint-mode max \
  --no-torch-compile
