#!/bin/bash
# M11-only baseline (all-to-all attention, no edges) — η₀ × width grid on gpu-test.
#
# Matched to the edge Plot-1 global batch: 4×A100, B_gpu=96 → B_global=384.
# Fixed d_h=32 ⇒ H = D/32. Adam, 4000 steps (resumes 2500-step dirs).
#
# Array (15 tasks):
#   width ∈ {256, 384, 512} × η₀ ∈ {0.05, 0.1, 0.25, 0.5, 1.0}
#   task = width_idx*5 + eta_idx
#     0–4   D256 H=8
#     5–9   D384 H=12
#    10–14  D512 H=16
# D=768 lives on 1×A100-80GB: scaling_law_L4_m11_D768_eta_a10080.slurm
#
#   sbatch scans/jets/scaling_law_L4_m11_eta_gputest.slurm
#   sbatch --array=0 scans/jets/scaling_law_L4_m11_eta_gputest.slurm
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-m11
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
# Della routes time<=59m onto gputest; do NOT set partition=gputest.
#SBATCH --partition=a100
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=0:59:00
#SBATCH --array=0-14
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_m11_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_m11_%A_%a.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

cd /home/jdezoort/coupled-particle-edge-networks/scans/jets
mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs

TASK=${SLURM_ARRAY_TASK_ID:-0}
ETAS=(0.05 0.1 0.25 0.5 1.0)
WIDTHS=(256 384 512)
D_HEAD=32

WIDTH_IDX=$((TASK / 5))
ETA_IDX=$((TASK % 5))
if (( WIDTH_IDX < 0 || WIDTH_IDX >= ${#WIDTHS[@]} || ETA_IDX < 0 || ETA_IDX >= ${#ETAS[@]} )); then
  echo "array index $TASK out of range (need 0-14)" >&2
  exit 1
fi

WIDTH=${WIDTHS[$WIDTH_IDX]}
ETA0=${ETAS[$ETA_IDX]}
HEAD=$((WIDTH / D_HEAD))
if (( HEAD * D_HEAD != WIDTH )); then
  echo "D=$WIDTH not divisible by d_h=$D_HEAD" >&2
  exit 1
fi

DEPTH=4
MODEL=baseline-transformer
N_TRAIN=1000000
N_VAL=25000
SEED=0
BATCH=96
N_GPUS=4
MAX_STEPS="${MAX_STEPS:-4000}"
# One epoch ≈ n_train / (B*g) = 1e6/384 ≈ 2604 steps. Need ≥2 epochs to reach 4k.
EPOCHS="${EPOCHS:-2}"
ETA_TAG=$(python3 -c "e=float('${ETA0}'); print(f'{e:g}'.replace('.','p'))")
# Keep tag aligned with on-disk run dirs (renamed from s2500 → s4000 for resume).
RUN_TAG="m11-b${BATCH}-g${N_GPUS}-s${MAX_STEPS}-dh${D_HEAD}"

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  TASK=$TASK"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
echo "study=m11-eta-gputest  L=$DEPTH  D=$WIDTH  H=$HEAD  d_head=$D_HEAD"
echo "model=$MODEL  (M11 all-to-all, no edges)  eta0=$ETA0  optimizer=adam"
echo "B_gpu=$BATCH  n_gpus=$N_GPUS  B_global=$((BATCH * N_GPUS))  max_steps=$MAX_STEPS  epochs=$EPOCHS"
echo "run_tag=$RUN_TAG"

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1

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
  --model "$MODEL" \
  --operators incidence \
  --jetclass-stream \
  --jetclass-raw-root /projects/j/jdezoort/jetclass \
  --jetclass-features kin7 \
  --mode sweep_lr \
  --optimizer adam \
  --t-epoch 1 \
  --residual-structure transformer-like \
  --etas "$ETA0" \
  --epochs "$EPOCHS" \
  --max-steps "$MAX_STEPS" \
  --n-train "$N_TRAIN" \
  --data-root /scratch/gpfs/BHANIN/jdezoort/datasets/jetclass \
  --depth "$DEPTH" --width "$WIDTH" --heads "$HEAD" \
  --dropout 0.0 \
  --root /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law \
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
