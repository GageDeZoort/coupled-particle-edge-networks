#!/bin/bash
#SBATCH --job-name=attn-path
#SBATCH --account=bhanin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=a100
#SBATCH --time=0:59:00
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/toptagging_finetune/logs/attn_pathology_%j.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/toptagging_finetune/logs/attn_pathology_%j.err

# Per-head attention pathology probe on frozen L4/D256 JetClass cosine init.
# See scans/jets/probe_attn_pathology.py

set -euo pipefail
REPO="${REPO:-$HOME/coupled-particle-edge-networks}"
cd "$REPO"
module load anaconda3/2023.9
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mamba-env

OUT="${OUT:-/scratch/gpfs/BHANIN/jdezoort/cpen_runs/toptagging_finetune/attn_pathology}"
CKPT="${CKPT:-/scratch/gpfs/BHANIN/jdezoort/cpen_runs/toptagging_finetune/init_ckpts/L4D256-cosine-s781250-frozen.ckpt}"
mkdir -p "$OUT" "$(dirname "$OUT")/../logs" 2>/dev/null || mkdir -p "$OUT"

python scans/jets/probe_attn_pathology.py \
  --ckpt "$CKPT" \
  --out-dir "$OUT" \
  --device cuda \
  --n-batches "${N_BATCHES:-16}" \
  --batch-size "${BATCH:-8}" \
  --tag "${TAG:-L4D256-cosine}"
