#!/bin/bash
#SBATCH --account=bhanin
#SBATCH --job-name=cpen-pascal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --partition=a100
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --array=0
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/slurm-%j_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/slurm-%j_%a.err

module purge
module load anaconda3/2023.9
conda activate mamba-env

cd /home/jdezoort/coupled-particle-edge-networks/scans/pascal

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

if [[ "${CUDA_VISIBLE_DEVICES:-}" == *MIG* ]]; then
  echo "ERROR: Pascal CPEN DDP requires full GPUs, but Slurm allocated MIG slices."
  exit 1
fi

export PYTHONFAULTHANDLER=1
# Prefer conda packages over ~/.local (avoids broken transformers/torchmetrics).
export PYTHONNOUSERSITE=1

# Pascal uses native cached edges (incidence-v2); do NOT pass --star-radius /
# --graph-construction. Rebuild once with:
#   sbatch scans/pascal/build_pascal_graphs.slurm

python test_cpen.py \
  --dataset pascalvoc-sp \
  --model capen-llama \
  --use-wire \
  --operators incidence \
  --mode sweep_lr \
  --optimizer adamw \
  --t-epoch 1000000 \
  --residual-structure transformer-like \
  --etas 0.25 \
  --dropout 0.25 \
  --epochs 500 \
  --data-root /scratch/gpfs/BHANIN/jdezoort/datasets \
  --depth 4 --width 256 --heads 8 \
  --root /scratch/gpfs/BHANIN/jdezoort/cpen_runs \
  --batch-size 8 \
  --n-gpus 4 \
  --num-workers 4 \
  --normalization uniform \
  --log-every-n-steps 100 \
  --operator-normalization degree \
  --gamma-11 7 \
  --gamma-12 1 \
  --gamma-21 1 \
  --gamma-22 1 \
  --layer-norm \
  --check-val-every-n-epoch 5 \
  --limit-val-batches 0.25
