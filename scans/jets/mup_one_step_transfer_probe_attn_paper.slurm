#!/bin/bash
#SBATCH --job-name=mup-1s-attn-paper
#SBATCH --account=bhanin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
# Della routes time<=59m onto gputest; do NOT set partition=gputest.
#SBATCH --partition=a100
#SBATCH --time=0:59:00
#SBATCH --array=0-8
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/mup_one_step_attn_paper_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/mup_one_step_attn_paper_%A_%a.err

# Paper-quality attn-pool width ladder @ L=4, d_h=32 — one width per array task.
# 10 seeds × 10 shards. gpu-test only (≤59m). Merge CSVs after the array finishes.

set -euo pipefail
REPO="${REPO:-$HOME/coupled-particle-edge-networks}"
cd "$REPO"
mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs
mkdir -p "$REPO/paper/data" "$REPO/paper/figures"

module purge
module load anaconda3/2023.9
conda activate mamba-env
export PYTHONPATH=src

WIDTHS=(64 128 256 384 512 768 1024 1536 2048)
TASK=${SLURM_ARRAY_TASK_ID:-0}
WIDTH=${WIDTHS[$TASK]}
ARCH="4:${WIDTH}"
OUT="$REPO/paper/data/stability_one_step_attn_paper_D${WIDTH}.csv"

echo "task=$TASK L=4 D=$WIDTH d_h=32 pool=attn → $OUT"

python scans/jets/mup_one_step_transfer_probe.py \
  --synthetic \
  --archs "$ARCH" \
  --cells 12:0.10 \
  --etas 0.01,0.05,0.1,0.25,0.5,1.0 \
  --pools attn \
  --d-head 32 \
  --batch-size 8 \
  --n-particles 128 \
  --seed 0 \
  --n-seeds 10 \
  --n-batches 10 \
  --device cuda \
  --out "$OUT"

echo "done D=$WIDTH"
