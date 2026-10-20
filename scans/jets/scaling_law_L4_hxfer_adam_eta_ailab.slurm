#!/bin/bash
# Hparam-xfer Arm 3 — Adam η₀×D fill-in on Della **ailab** (H200).
#
#   Panel (a) LR-transfer curves: Adam (WD=0), const LR, 10k steps
#   D ∈ {256,384} × η₀ ∈ {0.05,0.1,0.5,1.0}
#   plus task 8: D=256 η₀=0.25 (Arm 2 Adam skipped; D=384 η₀=0.25 from Arm 2b)
#   CAPEN k★=12 R★=0.1, L=4, d_h=32, B_gpu=128 × 4 → B_eff=512
#
#   Array layout (task = 4*d_idx + e_idx):
#     d_idx: 0→256, 1→384
#     e_idx: 0→0.05, 1→0.1, 2→0.5, 3→1.0
#   Extra: task 8 → D=256, η₀=0.25
#
#   sbatch scans/jets/scaling_law_L4_hxfer_adam_eta_ailab.slurm
#   sbatch --array=8-8 scans/jets/scaling_law_L4_hxfer_adam_eta_ailab.slurm
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-hxfer-aeta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=ailab
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=4:00:00
#SBATCH --array=0-7
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_hxfer_adam_eta_ailab_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_hxfer_adam_eta_ailab_%A_%a.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

cd /home/jdezoort/coupled-particle-edge-networks/scans/jets
mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs

TASK=${SLURM_ARRAY_TASK_ID:-0}

if (( TASK == 8 )); then
  # D=256 η₀=0.25 fill-in (missing from Arm 2 pause)
  WIDTH=256
  HEAD=8
  ETA0=0.25
else
  D_IDX=$((TASK / 4))
  E_IDX=$((TASK % 4))
  case "$D_IDX" in
    0) WIDTH=256; HEAD=8 ;;
    1) WIDTH=384; HEAD=12 ;;
    *) echo "d_idx $D_IDX out of range" >&2; exit 1 ;;
  esac
  case "$E_IDX" in
    0) ETA0=0.05 ;;
    1) ETA0=0.1 ;;
    2) ETA0=0.5 ;;
    3) ETA0=1.0 ;;
    *) echo "e_idx $E_IDX out of range" >&2; exit 1 ;;
  esac
fi

DEPTH=4
D_HEAD=32
OPT=adam
BATCH="${BATCH:-128}"
N_GPUS=4
MAX_STEPS="${MAX_STEPS:-10000}"
N_TRAIN="${N_TRAIN:-100000000}"
N_VAL="${N_VAL:-25000}"
SEED="${SEED:-0}"
STAR_R=0.1
KNN_K=12
ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law

ETA_TAG=$(python3 -c "e=float('${ETA0}'); print(f'{e:g}'.replace('.','p'))")
R_TAG=$(python3 -c "e=float('${STAR_R}'); print(f'{e:g}'.replace('.','p'))")
RUN_TAG="hxfer-adam-eta-b${BATCH}-g${N_GPUS}-s${MAX_STEPS}-k${KNN_K}-R${R_TAG}"

if (( WIDTH % HEAD != 0 || WIDTH / HEAD != D_HEAD )); then
  echo "D=$WIDTH H=$HEAD does not give d_h=$D_HEAD" >&2
  exit 1
fi

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  TASK=$TASK  partition=ailab"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
echo "study=hxfer-adam-eta-ailab  L=$DEPTH  D=$WIDTH  H=$HEAD  d_h=$D_HEAD"
echo "optimizer=$OPT  eta0=$ETA0  scheduler=none"
echo "k=$KNN_K  R=$STAR_R  B_gpu=$BATCH  n_gpus=$N_GPUS  B_eff=$((BATCH*N_GPUS))"
echo "max_steps=$MAX_STEPS  n_train=$N_TRAIN  run_tag=$RUN_TAG  root=$ROOT"

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
  --optimizer "$OPT" \
  --scheduler none \
  --residual-structure transformer-like \
  --etas "$ETA0" \
  --epochs 1 \
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
  --log-every-n-steps 100 \
  --layer-norm \
  --check-val-every-n-epoch 1 \
  --val-check-interval 1000 \
  --checkpoint-every-n-steps 500 \
  --checkpoint-monitor val_loss \
  --checkpoint-mode min \
  --no-torch-compile
