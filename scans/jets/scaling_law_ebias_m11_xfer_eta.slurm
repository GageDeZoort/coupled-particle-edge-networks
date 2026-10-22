#!/bin/bash
# Edge-bias M11 (arm E) μP transfer η₀ sweep — typed §1 + --edge-bias-m11 dynamic.
#
# Env overrides (watchdogs set these):
#   DEPTH WIDTH D_HEAD BATCH N_GPUS MAX_STEPS TAG_SUFFIX SEED
# Defaults: L4/D256 Della gputest B=64×4 → 40k.
#
# η₀ array: 0→0.01  1→0.05  2→0.1  3→0.25  4→0.5
#
#   sbatch --array=2 scans/jets/scaling_law_ebias_m11_xfer_eta.slurm
#   sbatch --partition=ailab --qos=gpu-test --gpus-per-task=1 --gres=gpu:1 \
#     --mem=64G --cpus-per-task=8 --export=ALL,BATCH=256,N_GPUS=1,TAG_SUFFIX=ailab \
#     --array=2 scans/jets/scaling_law_ebias_m11_xfer_eta.slurm
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-ebX
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=a100
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=0:59:00
#SBATCH --array=0-4
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_ebX_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_ebX_%A_%a.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

REPO=/home/jdezoort/coupled-particle-edge-networks
cd "$REPO/scans/jets"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs

TASK=${SLURM_ARRAY_TASK_ID:-0}
ETAS=(0.01 0.05 0.1 0.25 0.5)
if (( TASK < 0 || TASK >= ${#ETAS[@]} )); then
  echo "array index $TASK out of range" >&2
  exit 1
fi

ETA0=${ETAS[$TASK]}
DEPTH="${DEPTH:-4}"
WIDTH="${WIDTH:-256}"
D_HEAD="${D_HEAD:-32}"
HEAD=$((WIDTH / D_HEAD))
if (( WIDTH % D_HEAD != 0 )); then
  echo "WIDTH=$WIDTH not divisible by D_HEAD=$D_HEAD" >&2
  exit 1
fi
STAR_R=0.1
KNN_K=7
BATCH="${BATCH:-64}"
N_GPUS="${N_GPUS:-4}"
MAX_STEPS="${MAX_STEPS:-40000}"
SEED="${SEED:-0}"
TAG_SUFFIX="${TAG_SUFFIX:-gt}"
N_VAL=25000
N_TRAIN=$((MAX_STEPS * BATCH * N_GPUS))
EPOCHS=2
ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law
# Short tag — sweep_common adds typed/eb-dyn/stream/… (avoid ENAMETOOLONG).
RUN_TAG="ebX-L${DEPTH}D${WIDTH}-${TAG_SUFFIX}"

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}  TASK=$TASK"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
echo "ebias-m11 xfer  L=$DEPTH D=$WIDTH H=$HEAD d_h=$D_HEAD eta0=$ETA0"
echo "B_gpu=$BATCH n_gpus=$N_GPUS B_global=$((BATCH*N_GPUS)) max_steps=$MAX_STEPS"
echo "run_tag=$RUN_TAG  (const LR; typed + edge-bias-m11 dynamic)"

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python - <<PY
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("No CUDA device visible")
n = torch.cuda.device_count()
need = int("${N_GPUS}")
print(f"n_cuda={n} need={need}")
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU{i} {p.name} {p.total_memory/1024**3:.1f} GiB")
if n < need:
    sys.exit(f"Need {need} GPUs; visible={n}")
PY

set +e
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
  --typed-edge-streams \
  --edge-bias-m11 dynamic \
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
rc=$?
set -e
echo "done ebX L=$DEPTH D=$WIDTH eta0=$ETA0 tag=$RUN_TAG rc=$rc"
exit 0
