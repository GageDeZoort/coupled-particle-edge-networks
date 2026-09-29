#!/bin/bash
# Paper-ready val eval from best.ckpt for matched graph-k arms.
#
#   # Discover + write manifest, then submit
#   python scans/jets/eval_graph_k_ckpt.py --discover \
#     --manifest /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/graph_k_val_eval/manifest.txt
#   N=$(wc -l < .../manifest.txt)
#   sbatch --array=0-$((N-1)) scans/jets/eval_graph_k_ckpt.slurm
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-gk-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --exclude=della-l02g6,della-l03g11
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=0:25:00
#SBATCH --array=0-13
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_gk_eval_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_gk_eval_%A_%a.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

REPO=/home/jdezoort/coupled-particle-edge-networks
cd "$REPO"

MANIFEST="${MANIFEST:-/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/graph_k_val_eval/manifest.txt}"
OUT_CSV="${OUT_CSV:-$REPO/paper/data/graph_k_val_from_ckpt.csv}"
OUT_JSON_DIR="${OUT_JSON_DIR:-/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/graph_k_val_eval}"
mkdir -p "$(dirname "$MANIFEST")" "$OUT_JSON_DIR" "$(dirname "$OUT_CSV")"

TASK=${SLURM_ARRAY_TASK_ID:-0}
N_LINES=$(grep -cve '^\s*$' "$MANIFEST" || true)
if (( TASK < 0 || TASK >= N_LINES )); then
  echo "array index $TASK out of range (manifest has $N_LINES lines)" >&2
  exit 1
fi

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  TASK=$TASK  manifest=$MANIFEST"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1

python scans/jets/eval_graph_k_ckpt.py \
  --manifest "$MANIFEST" \
  --manifest-index "$TASK" \
  --out-csv "$OUT_CSV" \
  --out-json-dir "$OUT_JSON_DIR" \
  --n-gpus 1 \
  --batch-size 128 \
  --n-val 25000 \
  --seed 0
