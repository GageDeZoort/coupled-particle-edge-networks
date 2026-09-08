#!/bin/bash
#SBATCH --account=bhanin
#SBATCH --job-name=transfer-knn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=a100
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=0:59:00
#SBATCH --array=0

# CAPEN-Llama on TopTagging directed kNN graphs (default: k=4, no pT cut).
# Mirrors scans/testing/test.slurm, swapping star-R for kNN.
#
# Pre-build caches first (sharded ~1h cputest jobs):
#   GRAPH_CONSTRUCTION=4-NN bash scans/jets/submit_knn_graphs_cputest.sh
# (writes processed/4nn/n128/...)
#
# Optional overrides:
#   sbatch --export=ALL,ETA0=0.25,WIDTH=256,EPOCHS=35 \
#     scans/jets/test_capen_llama_toptagging_kNN.slurm

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
  --model capen-llama \
  --operators incidence \
  --graph-construction "4-NN" \
  --mode sweep_lr \
  --optimizer adam \
  --identity-m22 \
  --residual-structure transformer-like \
  --etas "0.25" \
  --epochs "48" \
  --data-root /scratch/gpfs/BHANIN/jdezoort/datasets/toptagging \
  --depth "4" --width "128" --heads "4" \
  --dropout 0.1 \
  --root /scratch/gpfs/BHANIN/jdezoort/cpen_runs \
  --batch-size 256 \
  --n-gpus 4 \
  --num-workers 4 \
  --normalization uniform \
  --num-particles 128 \
  --log-every-n-steps 100 \
  --attention-normalization gamma \
  --gamma-11 54.14 \
  --gamma-12 1.75 \
  --gamma-21 7 \
  --gamma-22 5.02 \
  --layer-norm \
  --check-val-every-n-epoch 1 \
  --no-torch-compile
