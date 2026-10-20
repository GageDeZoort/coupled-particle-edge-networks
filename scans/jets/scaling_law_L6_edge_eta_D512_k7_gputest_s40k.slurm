#!/bin/bash
# Panel (c) edge CAPEN η₀ sweep — L=6 D=512 at frozen OP (k★=7, R★=0.1).
#
#   model   = capen-llama-att + all-seed star kin7 ∥ ΔR-kNN (noloop)
#   B_gpu=64 × 4 → B_global=256, Adam, 40k steps
#   η₀ ∈ {0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75}
#   n_train = 40k × 256 = 10.24M  (single-pass unique jets; no epoch repeat)
#   tag     = edge-L6D512-b64-g4-s40000-dh32-k7-R0p1
#
# Della routes time≤59m onto gputest. Lightning resumes last.ckpt under the
# same run_tag — the pool watchdog keeps cells rolling to 40k.
#
# Prefer the watchdog:
#   bash scans/jets/start_edge_eta_L6D512_k7_gputest_watchdogs.sh
# One cell:
#   sbatch --array=2 scans/jets/scaling_law_L6_edge_eta_D512_k7_gputest_s40k.slurm
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-edge-L6D512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=a100
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=0:59:00
#SBATCH --array=0-7
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_edge_L6D512_s40k_gt_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_edge_L6D512_s40k_gt_%A_%a.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

REPO=/home/jdezoort/coupled-particle-edge-networks
cd "$REPO/scans/jets"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs

TASK=${SLURM_ARRAY_TASK_ID:-0}
ETAS=(0.01 0.025 0.05 0.075 0.1 0.25 0.5 0.75)
if (( TASK < 0 || TASK >= ${#ETAS[@]} )); then
  echo "array index $TASK out of range (need 0-$(( ${#ETAS[@]} - 1 )))" >&2
  exit 1
fi

ETA0=${ETAS[$TASK]}
DEPTH=6
WIDTH=512
D_HEAD=32
HEAD=$((WIDTH / D_HEAD))
STAR_R=0.1
KNN_K=7
BATCH=64
N_GPUS=4
MAX_STEPS="${MAX_STEPS:-40000}"
SEED="${SEED:-0}"
N_VAL=25000
# One epoch = exactly MAX_STEPS × B_global unique jets (no repeat).
N_TRAIN=$((MAX_STEPS * BATCH * N_GPUS))
EPOCHS=1
ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law
R_TAG=$(python3 -c "e=float('${STAR_R}'); print(f'{e:g}'.replace('.','p'))")
RUN_TAG="edge-L6D512-b${BATCH}-g${N_GPUS}-s${MAX_STEPS}-dh${D_HEAD}-k${KNN_K}-R${R_TAG}"

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}  TASK=$TASK  partition=gpu-test (via a100≤59m)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
echo "study=edge-eta-L6D512-s40k-gputest  L=$DEPTH  D=$WIDTH  H=$HEAD  d_h=$D_HEAD"
echo "model=capen-llama-att  eta0=$ETA0  optimizer=adam  k=$KNN_K  R=$STAR_R  seed=$SEED"
echo "B_gpu=$BATCH  n_gpus=$N_GPUS  B_global=$((BATCH * N_GPUS))  max_steps=$MAX_STEPS"
echo "n_train=$N_TRAIN  (unique single-pass)  run_tag=$RUN_TAG"

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("No CUDA device visible")
print(f"n_cuda={torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU{i} {p.name} {p.total_memory/1024**3:.1f} GiB")
if torch.cuda.device_count() < 4:
    sys.exit(f"Need 4 GPUs; visible count={torch.cuda.device_count()}")
PY

# Canonical: all-seed star kin7 + ΔR-kNN noloop (defaults live_star_kin7=True,
# seed_min_pt=0; do NOT pass --live-knn-include-self).
python test_cpen_jetclass.py \
  --model capen-llama-att \
  --operators incidence \
  --jetclass-stream \
  --jetclass-raw-root /projects/j/jdezoort/jetclass \
  --jetclass-features kin7 \
  --jetclass-edge-features part-interaction \
  --live-knn-k "$KNN_K" \
  --star-radius "$STAR_R" \
  --live-star-kin7 \
  --mode sweep_lr \
  --optimizer adam \
  --t-epoch 1 \
  --residual-structure transformer-like \
  --etas "$ETA0" \
  --epochs "$EPOCHS" \
  --max-steps "$MAX_STEPS" \
  --n-train "$N_TRAIN" \
  --n-test 0 \
  --data-root /scratch/gpfs/BHANIN/jdezoort/datasets/jetclass \
  --depth "$DEPTH" --width "$WIDTH" --heads "$HEAD" \
  --dropout 0.0 \
  --root "$ROOT" \
  --run-tag "$RUN_TAG" \
  --batch-size "$BATCH" \
  --n-gpus "$N_GPUS" \
  --num-workers 4 \
  --normalization uniform \
  --attention-normalization gamma \
  --num-particles 80 \
  --n-val "$N_VAL" \
  --seed "$SEED" \
  --log-every-n-steps 50 \
  --layer-norm \
  --check-val-every-n-epoch 1 \
  --val-check-interval 500 \
  --checkpoint-every-n-steps 500 \
  --checkpoint-monitor val_loss \
  --checkpoint-mode min \
  --no-torch-compile

echo "done eta0=$ETA0 task=$TASK max_steps=$MAX_STEPS"
