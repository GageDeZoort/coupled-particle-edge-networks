#!/bin/bash
#SBATCH --account=bhanin
#SBATCH --job-name=cpen-tt-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=a100
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=0:59:00
#SBATCH --array=0

# Plain CPEN (no attention / not CAPEN) on TopTagging star-R caches.
# Use this to isolate transfer / feature issues away from CAPEN-Llama.
#
# Requires ParT-kin star caches (n=128), e.g.:
#   sbatch --export=ALL,STAR_RADIUS=0.15,NUM_PARTICLES=128 \
#     scans/jets/build_star_graphs.slurm
#
# Optional overrides:
#   sbatch --export=ALL,ETA0=0.1,WIDTH=256,EPOCHS=10,N_TRAIN=100000 \
#     scans/jets/test_cpen_toptagging.slurm

module purge
module load anaconda3/2023.9
conda activate mamba-env

cd /home/jdezoort/coupled-particle-edge-networks/scans/jets

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1

python test_cpen_toptagging.py \
  --model cpen \
  --operators incidence \
  --star-radius "0.15" \
  --mode sweep_lr \
  --optimizer adam \
  --t-epoch 1 \
  --residual-structure transformer-like \
  --etas "0.005" \
  --epochs "20" \
  --data-root /scratch/gpfs/BHANIN/jdezoort/datasets/toptagging \
  --depth "6" --width "512" \
  --root /scratch/gpfs/BHANIN/jdezoort/cpen_runs \
  --batch-size 256 \
  --n-gpus 4 \
  --num-workers 4 \
  --normalization uniform \
  --num-particles 128 \
  --log-every-n-steps 100 \
  --operator-normalization gamma \
  --gamma-11 31.9 \
  --gamma-12 2.8 \
  --gamma-21 2.8 \
  --gamma-22 32.0 \
  --layer-norm \
  --check-val-every-n-epoch 1 \
  --no-torch-compile

