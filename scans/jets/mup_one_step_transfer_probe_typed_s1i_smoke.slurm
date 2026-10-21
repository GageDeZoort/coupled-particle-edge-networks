#!/bin/bash
#SBATCH --job-name=mup-typed-s1i
#SBATCH --account=bhanin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=ailab
#SBATCH --qos=gpu-test
#SBATCH --time=0:59:00
#SBATCH --array=0-2
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/mup_typed_s1i_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/mup_typed_s1i_%A_%a.err

# Quick width check after typed encoder init fix (σ=D^{-1/2}).
# OP graph: kin7 star + part-int kNN, k★=7 R★=0.1.

set -euo pipefail
REPO="${REPO:-$HOME/coupled-particle-edge-networks}"
cd "$REPO"
mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs
mkdir -p "$REPO/paper/data"

module purge
module load anaconda3/2023.9
conda activate mamba-env
export PYTHONPATH=src

ARCHS=(2:128 4:256 6:512)
TASK=${SLURM_ARRAY_TASK_ID:-0}
ARCH=${ARCHS[$TASK]}
WIDTH=${ARCH#*:}
OUT="$REPO/paper/data/stability_one_step_typed_s1i_D${WIDTH}.csv"

echo "task=$TASK arch=$ARCH typed-edge-streams cells=7:0.1 → $OUT"

python scans/jets/mup_one_step_transfer_probe.py \
  --synthetic \
  --typed-edge-streams \
  --archs "$ARCH" \
  --cells 7:0.10 \
  --etas 0.05,0.1,0.25 \
  --pools attn \
  --d-head 32 \
  --batch-size 8 \
  --n-particles 128 \
  --seed 0 \
  --n-seeds 5 \
  --n-batches 5 \
  --same-data-shard \
  --device cuda \
  --out "$OUT"

echo "done $ARCH"
