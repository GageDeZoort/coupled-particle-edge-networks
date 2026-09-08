#!/bin/bash
#SBATCH --account=bhanin
#SBATCH --job-name=transfer-scan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=a100
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=0:59:00
#SBATCH --array=0


module purge
module load anaconda3/2023.9
conda activate mamba-env

cd /home/jdezoort/coupled-particle-edge-networks/scans/common

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

export PYTHONFAULTHANDLER=1
# Prefer conda packages over ~/.local (avoids broken transformers/torchmetrics).
export PYTHONNOUSERSITE=1

# Pre-build star-R caches once on a login node, e.g.:
#   python scans/jets/build_toptagging_star_graphs.py \
#     --data-root /scratch/gpfs/BHANIN/jdezoort/datasets/toptagging \
#     --radius 0.25 --num-particles 100
#
# For cheap LR / architecture sweeps, add e.g.:
#   --n-train 100000 --n-val 20000

python test_cpen_toptagging.py \
  --model capen-llama \
  --operators incidence \
  --star-radius 0.15 \
  --mode sweep_lr \
  --optimizer adamw \
  --t-epoch 10 \
  --residual-structure transformer-like \
  --etas 0.25 \
  --epochs 35 \
  --data-root /scratch/gpfs/BHANIN/jdezoort/datasets/toptagging \
  --depth 4 --width 256 --heads 4 \
  --dropout 0.0 \
  --root /scratch/gpfs/BHANIN/jdezoort/cpen_runs \
  --batch-size 256 \
  --n-gpus 4 \
  --num-workers 4 \
  --normalization uniform \
  --num-particles 128 \
  --log-every-n-steps 100 \
  --attention-normalization gamma \
  --gamma-11 54 \
  --gamma-12 8.5 \
  --gamma-21 8.5 \
  --gamma-22 5.9 \
  --layer-norm \
  --check-val-every-n-epoch 1 \
  --no-torch-compile
