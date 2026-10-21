#!/bin/bash
# Fixed pathology probe (c13) on live typed -s1i snapshots.
# JetClass val kin7, ≥256 jets; multi-key collapse; logit spread; self-top.
#
# Array maps to available frozen snapshots under attn_pathology_typed_s1i/snapshots/
# (refresh snapshots before submit if you want newer steps).
#
#   sbatch scans/jets/probe_typed_edge_s1i_attn_pathology_fixed.slurm
#
#SBATCH --account=bhanin
#SBATCH --job-name=attn-path-fix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=ailab
#SBATCH --qos=gpu-test
#SBATCH --time=0:45:00
#SBATCH --array=0-2
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/attn_path_s1i_fix_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/attn_path_s1i_fix_%A_%a.err

set -euo pipefail
REPO="${REPO:-$HOME/coupled-particle-edge-networks}"
cd "$REPO"
module purge
module load anaconda3/2023.9
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mamba-env
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law
OUT="$ROOT/attn_pathology_typed_s1i_fixed"
SNAP="$ROOT/attn_pathology_typed_s1i/snapshots"
SWEEP="$ROOT/jetclass_output/adam/sweep_lr"
TAG="typed-edge-b256-g1-s40000-dh32-k7-R0p1-s1i-ailab"
mkdir -p "$OUT" "$SNAP" "$ROOT/logs"

ETAS=(0.01 0.05 0.1)
TASK=${SLURM_ARRAY_TASK_ID:-0}
ETA=${ETAS[$TASK]}
ET=$(python3 -c "e=float('${ETA}'); print(f'{e:g}'.replace('.','p'))")

# Prefer frozen snapshot; else copy current last.ckpt.
CKPT=$(ls -1t "$SNAP"/eta${ET}_step*_last.ckpt 2>/dev/null | head -1 || true)
if [[ -z "${CKPT:-}" || ! -f "$CKPT" ]]; then
  RUN=$(ls -1d "$SWEEP"/capen-llama-att_4_256_${ET}_*${TAG} 2>/dev/null | head -1)
  STEP=$(python - <<PY
import torch
p=torch.load("$RUN/last.ckpt", map_location="cpu", weights_only=False)
print(int(p.get("global_step", 0) or 0))
PY
)
  CKPT="$SNAP/eta${ET}_step${STEP}_last.ckpt"
  cp -f "$RUN/last.ckpt" "$CKPT"
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

N_JETS="${N_JETS:-256}"
BATCH="${BATCH:-16}"

python scans/jets/probe_attn_pathology.py \
  --ckpt "$CKPT" \
  --out-dir "$OUT" \
  --data jetclass \
  --jetclass-raw-root /projects/j/jdezoort/jetclass \
  --n-jets "$N_JETS" \
  --batch-size "$BATCH" \
  --device cuda \
  --depth 4 \
  --width 256 \
  --heads 8 \
  --knn-k 7 \
  --star-r 0.1 \
  --num-particles 80 \
  --typed-edge-streams \
  --tag "typed-s1i-fix-eta${ET}-step${STEP}"

echo "done eta0=$ETA step=$STEP"
