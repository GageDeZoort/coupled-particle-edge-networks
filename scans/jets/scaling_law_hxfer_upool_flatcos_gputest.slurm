#!/bin/bash
# Uniform-pool hxfer flat→cosine (20k) on Della gpu-test — 4×A100.
#
# Matched to Lambda D=128 campaign (run_hxfer_upool_d128_flat_cosine_lambda8.sh):
#   B_global = 512  →  here B_gpu=128 × 4 GPUs (was 64×8 on Lambda)
#   flat 10k @ peak η=η₀/√D, then cosine 10k → 0.01×peak
#   n_train=1M, k★=6, R★=0.1, L=4, d_h=32, seed=0, uniform pool
#
# One cell per job. Set WIDTH + ETA0 (watchdog does this), or use array map:
#   CELLS env / default map below.
#
#   WIDTH=128 ETA0=0.25 sbatch scans/jets/scaling_law_hxfer_upool_flatcos_gputest.slurm
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-hxfc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
# Della routes time<=59m onto gputest; do NOT set partition=gputest.
#SBATCH --partition=a100
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=0:59:00
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_hxfc_%j.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_hxfc_%j.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

cd /home/jdezoort/coupled-particle-edge-networks/scans/jets
mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs

# Default cell map (array or single-task). Watchdog overrides WIDTH/ETA0.
# 0–2: D=128 remaining; 3–8: D=256 full sweep.
DEFAULT_WIDTHS=(128 128 128 256 256 256 256 256 256)
DEFAULT_ETAS=(0.25 0.5 1.0 0.01 0.05 0.1 0.25 0.5 1.0)

TASK=${SLURM_ARRAY_TASK_ID:-}
if [[ -n "${WIDTH:-}" && -n "${ETA0:-}" ]]; then
  : # env already set
elif [[ -n "$TASK" ]]; then
  if (( TASK < 0 || TASK >= ${#DEFAULT_WIDTHS[@]} )); then
    echo "array index $TASK out of range" >&2
    exit 1
  fi
  WIDTH=${DEFAULT_WIDTHS[$TASK]}
  ETA0=${DEFAULT_ETAS[$TASK]}
else
  echo "Set WIDTH+ETA0 or submit as --array" >&2
  exit 1
fi

DEPTH="${DEPTH:-4}"
D_HEAD=32
STAR_R=0.1
KNN_K=6
SEED="${SEED:-0}"
N_TRAIN=1000000
N_VAL=25000
BATCH=128
N_GPUS=4
FLAT_STEPS="${FLAT_STEPS:-10000}"
COSINE_STEPS="${COSINE_STEPS:-10000}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
MAX_STEPS="${MAX_STEPS:-$((FLAT_STEPS + COSINE_STEPS))}"
LR_MIN_FRAC="${LR_MIN_FRAC:-0.01}"
STEPS_PER_EPOCH=$(( (N_TRAIN + BATCH * N_GPUS - 1) / (BATCH * N_GPUS) ))
EPOCHS="${EPOCHS:-$(( (MAX_STEPS + STEPS_PER_EPOCH - 1) / STEPS_PER_EPOCH ))}"
ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law

WIDTH=$(printf '%d' "$WIDTH")
HEAD=$((WIDTH / D_HEAD))
if (( HEAD * D_HEAD != WIDTH )); then
  echo "D=$WIDTH not divisible by d_h=$D_HEAD" >&2
  exit 1
fi

ETA_TAG=$(python3 -c "e=float('${ETA0}'); print(f'{e:g}'.replace('.','p'))")
R_TAG=$(python3 -c "e=float('${STAR_R}'); print(f'{e:g}'.replace('.','p'))")
# L=4 tags stay identical to Lambda (`hxfer-d{W}-fc-...`) so finished cells resume.
# Non-default depth is tagged `hxfer-L{L}-d{W}-fc-...` so L=6 D=512 cannot resume L=4.
if [[ "$DEPTH" == "4" ]]; then
  RUN_TAG="hxfer-d${WIDTH}-fc-k${KNN_K}-R${R_TAG}"
else
  RUN_TAG="hxfer-L${DEPTH}-d${WIDTH}-fc-k${KNN_K}-R${R_TAG}"
fi

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none} TASK=${TASK:-env} partition=gpu-test (via a100≤59m)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
echo "study=hxfer-upool-flatcos-gputest  L=$DEPTH  D=$WIDTH  H=$HEAD  d_h=$D_HEAD"
echo "model=capen-llama-att  eta0=$ETA0  scheduler=warmup_flat_cosine  UNIFORM-pool"
echo "flat=$FLAT_STEPS cosine=$COSINE_STEPS max_steps=$MAX_STEPS min_frac=$LR_MIN_FRAC"
echo "B_gpu=$BATCH  n_gpus=$N_GPUS  B_global=$((BATCH * N_GPUS))  epochs=$EPOCHS"
echo "n_train=$N_TRAIN  seed=$SEED  k=$KNN_K  R=$STAR_R  run_tag=$RUN_TAG"

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

# Skip if already at target (watchdog may still submit; exit 0 quickly).
EXISTING=$(find "$ROOT/jetclass_output/adam/sweep_lr" -maxdepth 1 -type d \
  -name "*_${DEPTH}_${WIDTH}_${ETA_TAG}_*${RUN_TAG}*" 2>/dev/null | head -1 || true)
if [[ -n "$EXISTING" && -f "${EXISTING}.parquet" ]]; then
  STEP=$(python3 -c "
import pyarrow.parquet as pq
df=pq.read_table('${EXISTING}.parquet', columns=['global_step']).to_pandas()
print(int(df['global_step'].max()) if len(df) else 0)
" 2>/dev/null || echo 0)
  echo "existing=$EXISTING step=$STEP"
  if (( STEP >= MAX_STEPS )); then
    echo "SKIP already done (step=$STEP >= $MAX_STEPS)"
    exit 0
  fi
fi

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
  --scheduler warmup_flat_cosine \
  --warmup-steps "$WARMUP_STEPS" \
  --flat-steps "$FLAT_STEPS" \
  --cosine-steps "$COSINE_STEPS" \
  --lr-decay-min-frac "$LR_MIN_FRAC" \
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
  --uniform-output-pool \
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
