#!/bin/bash
# Same μP check as gputest s1000, but 8×H200 on ailab with B_gpu=32
# (global batch 256 = same as gputest B=64×4).
#
#   (L,D) ∈ {(4,256), (8,1024)} × η₀ ∈ {0.05, 0.25, 0.75}
#   Note: Della `mig` only allows gres=gpu:1 — use ailab for 8-GPU.
#
#   sbatch scans/jets/scaling_law_mup_check_s1000_ailab8.slurm
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-mup-a8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=32
#SBATCH --partition=ailab
#SBATCH --mem=256G
#SBATCH --gres=gpu:8
#SBATCH --time=2:00:00
#SBATCH --array=0-5
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_mup_s1000_a8_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_mup_s1000_a8_%A_%a.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

REPO=/home/jdezoort/coupled-particle-edge-networks
cd "$REPO/scans/jets"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs

TASK=${SLURM_ARRAY_TASK_ID:-0}
ARCHS=( "4:256" "8:1024" )
# Override with e.g. --export=ALL,ETAS_CSV=0.01:0.075
# (use ':' — Slurm --export treats commas as var separators)
if [[ -n "${ETAS_CSV:-}" ]]; then
  IFS=':,' read -r -a ETAS <<< "$ETAS_CSV"
else
  ETAS=(0.05 0.25 0.75)
fi
N_ETA=${#ETAS[@]}
D_HEAD=32
STAR_R=0.1
KNN_K=6

# Single-cell continue: --export=ALL,DEPTH=4,WIDTH=256,ETA0=0.05,MAX_STEPS=5000
if [[ -z "${DEPTH:-}" || -z "${WIDTH:-}" || -z "${ETA0:-}" ]]; then
  ARCH_IDX=$((TASK / N_ETA))
  ETA_IDX=$((TASK % N_ETA))
  if (( ARCH_IDX < 0 || ARCH_IDX >= ${#ARCHS[@]} || ETA_IDX < 0 || ETA_IDX >= N_ETA )); then
    echo "array index $TASK out of range" >&2
    exit 1
  fi
  IFS=':' read -r DEPTH WIDTH <<< "${ARCHS[$ARCH_IDX]}"
  ETA0=${ETAS[$ETA_IDX]}
fi
HEAD=$((WIDTH / D_HEAD))
if (( HEAD * D_HEAD != WIDTH )); then
  echo "D=$WIDTH not divisible by d_h=$D_HEAD" >&2
  exit 1
fi

# Need n_train ≥ max_steps × B_global or the stream epoch ends early
# (1M / 256 ≈ 3906 steps). Override with e.g. N_TRAIN=2000000.
N_TRAIN="${N_TRAIN:-1000000}"
N_VAL=25000
SEED="${SEED:-0}"
BATCH="${BATCH:-32}"   # 32×8 = 256 global (matches gputest 64×4)
N_GPUS=8
MAX_STEPS="${MAX_STEPS:-1000}"
EPOCHS=1
ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law
R_TAG=$(python3 -c "e=float('${STAR_R}'); print(f'{e:g}'.replace('.','p'))")
ETA_TAG=$(python3 -c "e=float('${ETA0}'); print(f'{e:g}'.replace('.','p'))")
# Distinct tag so we don't collide with gputest run dirs.
# Override RUN_TAG when migrating s1k→s5k for Lightning resume.
RUN_TAG="${RUN_TAG:-mup-check-s${MAX_STEPS}-b${BATCH}-g${N_GPUS}-k${KNN_K}-R${R_TAG}-ailab8}"

echo "PWD=$(pwd)"
echo "Host: $(hostname) SLURM_JOB_ID=${SLURM_JOB_ID:-none} TASK=$TASK partition=ailab"
nvidia-smi -L || true
echo "study=mup-check-s1000-ailab8  L=$DEPTH  D=$WIDTH  H=$HEAD  d_h=$D_HEAD"
echo "eta0=$ETA0  optimizer=adam  k=$KNN_K  R=$STAR_R  seed=$SEED"
echo "B_gpu=$BATCH  n_gpus=$N_GPUS  B_global=$((BATCH * N_GPUS))  max_steps=$MAX_STEPS"
echo "run_tag=$RUN_TAG  eta_tag=$ETA_TAG"

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
if torch.cuda.device_count() < 8:
    sys.exit(f"Need 8 GPUs; visible count={torch.cuda.device_count()}")
PY

python test_cpen_jetclass.py \
  --model capen-llama-att \
  --operators incidence \
  --jetclass-stream \
  --jetclass-raw-root /projects/j/jdezoort/jetclass \
  --jetclass-features kin7 \
  --jetclass-edge-features part-interaction \
  --live-knn-k "$KNN_K" \
  --star-radius "$STAR_R" \
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

echo "done L=$DEPTH D=$WIDTH eta0=$ETA0"
