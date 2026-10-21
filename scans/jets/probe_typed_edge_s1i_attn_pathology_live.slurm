#!/bin/bash
# Pathology remeasure on live typed-edge -s1i-ailab last.ckpt snapshots.
# Array 0..2 → η₀ ∈ {0.01, 0.05, 0.1} (only cells with ckpts so far).
#
#   sbatch scans/jets/probe_typed_edge_s1i_attn_pathology_live.slurm
#
#SBATCH --account=bhanin
#SBATCH --job-name=attn-path-s1i
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
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/attn_path_typed_s1i_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/attn_path_typed_s1i_%A_%a.err

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
SWEEP="$ROOT/jetclass_output/adam/sweep_lr"
OUT="$ROOT/attn_pathology_typed_s1i"
SNAP="$OUT/snapshots"
mkdir -p "$OUT" "$SNAP" "$ROOT/logs"

ETAS=(0.01 0.05 0.1)
TASK=${SLURM_ARRAY_TASK_ID:-0}
ETA=${ETAS[$TASK]}
ET=$(python3 -c "e=float('${ETA}'); print(f'{e:g}'.replace('.','p'))")
TAG="typed-edge-b256-g1-s40000-dh32-k7-R0p1-s1i-ailab"

# Prefer a frozen snapshot (copied before submit) so training cannot tear the file.
CKPT=$(ls -1t "$SNAP"/eta${ET}_step*_last.ckpt 2>/dev/null | head -1 || true)
if [[ -z "${CKPT:-}" || ! -f "$CKPT" ]]; then
  RUN=$(ls -1d "$SWEEP"/capen-llama-att_4_256_${ET}_*${TAG} 2>/dev/null | head -1)
  if [[ -z "${RUN:-}" || ! -f "$RUN/last.ckpt" ]]; then
    echo "missing last.ckpt for eta0=$ETA tag=$TAG" >&2
    exit 1
  fi
  STEP=$(python - <<PY
import torch
p=torch.load("$RUN/last.ckpt", map_location="cpu", weights_only=False)
print(int(p.get("global_step", 0) or 0))
PY
)
  CKPT="$SNAP/eta${ET}_step${STEP}_last.ckpt"
  cp -f "$RUN/last.ckpt" "$CKPT"
  echo "SRC=$RUN/last.ckpt"
else
  STEP=$(python - <<PY
import torch
p=torch.load("$CKPT", map_location="cpu", weights_only=False)
print(int(p.get("global_step", 0) or 0))
PY
)
fi
echo "Host=$(hostname) TASK=$TASK eta0=$ETA step=$STEP"
echo "CKPT=$CKPT"
nvidia-smi -L || true

# Real TopTagging val slice (not synthetic). 16×B8 = 128 jets.
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

echo "done eta0=$ETA step=$STEP"
