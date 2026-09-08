#!/bin/bash
#SBATCH --account=bhanin
#SBATCH --job-name=cpen-profile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=a100
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=0:15:00
#SBATCH --array=0
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/slurm_output/slurm-%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/slurm_output/slurm-%A_%a.out

module purge
module load anaconda3/2023.9
conda activate mamba-env

cd /home/jdezoort/coupled-particle-edge-networks

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

export PYTHONFAULTHANDLER=1
export PYTHONPATH=src

for BACKEND in dense sparse; do
  echo ""
  echo "############################################################"
  echo "### operator_backend = ${BACKEND}"
  echo "############################################################"
  python scans/common/profile_cpen_step.py \
    --width 256 \
    --depth 4 \
    --batch-size 128 \
    --num-particles 100 \
    --num-edges 100 \
    --edge-degree 8 \
    --n-features 4 \
    --n-edge-features 4 \
    --iters 30 \
    --warmup 5 \
    --residual transformer-like \
    --norm gamma \
    --backend "${BACKEND}" \
    --precision bf16
done
