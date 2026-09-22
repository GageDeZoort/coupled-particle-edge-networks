#!/bin/bash
# 1M unique-pass ablation: star-R hyperedges + part-interaction, NO kNN.
#   Completes the 2×2 with knn6+star / knn6-nostar / (this) star-only.
#   L=4, D=256, H=8, η₀=0.25, B=256, p=80, 1×A100 gpu-test
#
#   sbatch scans/jets/scaling_law_L4_ablate_star_noknn_1M.slurm
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-ablate-staronly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --qos=gpu-test
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=0:59:00
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_ablate_staronly_%j.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_ablate_staronly_%j.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

cd /home/jdezoort/coupled-particle-edge-networks/scans/jets
mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs

ETA0=0.25
WIDTH=256
HEAD=8
DEPTH=4
STAR_R=0.2
N_TRAIN=1000000
N_VAL=25000
SEED=0
BATCH=256
MAX_STEPS=$(( (N_TRAIN + BATCH - 1) / BATCH ))

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  label=partint+star${STAR_R}+noknn"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "L=$DEPTH D=$WIDTH H=$HEAD eta0=$ETA0 starR=$STAR_R knn=none"
echo "n_train=$N_TRAIN max_steps=$MAX_STEPS B=$BATCH"

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1

python test_cpen_jetclass.py \
  --model capen-llama-att \
  --operators incidence \
  --jetclass-stream \
  --jetclass-raw-root /projects/j/jdezoort/jetclass \
  --jetclass-features kin7 \
  --jetclass-edge-features part-interaction \
  --star-radius "$STAR_R" \
  --mode sweep_lr \
  --optimizer adam \
  --t-epoch 1 \
  --residual-structure transformer-like \
  --etas "$ETA0" \
  --epochs 1 \
  --max-steps "$MAX_STEPS" \
  --n-train "$N_TRAIN" \
  --data-root /scratch/gpfs/BHANIN/jdezoort/datasets/jetclass \
  --depth "$DEPTH" --width "$WIDTH" --heads "$HEAD" \
  --dropout 0.0 \
  --root /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law \
  --run-tag ablate-b256 \
  --batch-size "$BATCH" \
  --n-gpus 1 \
  --num-workers 4 \
  --normalization uniform \
  --attention-normalization gamma \
  --num-particles 80 \
  --n-val "$N_VAL" \
  --seed "$SEED" \
  --log-every-n-steps 100 \
  --layer-norm \
  --check-val-every-n-epoch 1 \
  --val-check-interval 1000 \
  --checkpoint-every-n-steps 200 \
  --checkpoint-monitor val_loss \
  --checkpoint-mode min \
  --no-torch-compile
