#!/bin/bash
# Attention pathology probe on typed-edge §1 ailab sweep ckpts (one η per array task).
#
#   sbatch scans/jets/probe_typed_edge_s1_attn_pathology.slurm
#
#SBATCH --account=bhanin
#SBATCH --job-name=attn-path-s1i
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=a100
#SBATCH --time=0:45:00
#SBATCH --array=0-4
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/attn_path_typed_s1_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/attn_path_typed_s1_%A_%a.err

set -euo pipefail
REPO="${REPO:-$HOME/coupled-particle-edge-networks}"
cd "$REPO"
module load anaconda3/2023.9
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mamba-env

SWEEP=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/jetclass_output/adam/sweep_lr
OUT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/attn_pathology_typed_s1
mkdir -p "$OUT" "$(dirname "$OUT")/../logs" 2>/dev/null || mkdir -p "$OUT"

ETAS=(0.01 0.05 0.1 0.25 0.5)
TASK=${SLURM_ARRAY_TASK_ID:-0}
ETA=${ETAS[$TASK]}
ET=$(python3 -c "e=float('${ETA}'); print(f'{e:g}'.replace('.','p'))")
TAG="typed-edge-b256-g1-s40000-dh32-k7-R0p1-s1i-ailab"

CKPT=$(ls -1d "$SWEEP"/capen-llama-att_4_256_${ET}_*${TAG}/last.ckpt 2>/dev/null | head -1)
if [[ -z "${CKPT:-}" || ! -f "$CKPT" ]]; then
  echo "missing ckpt for eta0=$ETA tag=$TAG" >&2
  exit 1
fi

STEP=$(python - <<PY
import torch
p=torch.load("$CKPT", map_location="cpu", weights_only=False)
print(int(p.get("global_step", 0) or 0))
PY
)

echo "Host=$(hostname) TASK=$TASK eta0=$ETA step=$STEP"
echo "CKPT=$CKPT"
nvidia-smi -L || true

python scans/jets/probe_attn_pathology.py \
  --ckpt "$CKPT" \
  --out-dir "$OUT" \
  --device cuda \
  --n-batches "${N_BATCHES:-16}" \
  --batch-size "${BATCH:-8}" \
  --depth 4 \
  --width 256 \
  --heads 8 \
  --knn-k 7 \
  --star-r 0.1 \
  --typed-edge-streams \
  --tag "typed-s1i-eta${ET}-step${STEP}"
