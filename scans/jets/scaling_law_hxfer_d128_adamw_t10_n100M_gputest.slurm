#!/bin/bash
# D=128 uniform-pool pretrain on the **full JetClass train stream**.
#
# Locked recipe (2026-10-06):
#   n_train = 100_000_000  → streaming (must be > RAM_SUBSET_LIMIT=2e6)
#   epochs = 1             → no wrap; 40k × B=512 = 20.48M unique jets
#   AdamW, t_epoch = 10    → λ = √D · B / (t η₀ n_train) with n_train=100M
#   const LR η = η₀/√D, 40k steps, B=128×4=512
#   k★=6, R★=0.1, L=4, d_h=32, seed=0, uniform pool
#
#   ETA0=0.1 sbatch --export=ALL,ETA0=0.1 \
#     scans/jets/scaling_law_hxfer_d128_adamw_t10_n100M_gputest.slurm
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-aw100M
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
# Della routes time<=59m onto gputest; do NOT set partition=gputest.
#SBATCH --partition=a100
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=0:59:00
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_aw100M_%j.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_aw100M_%j.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

cd /home/jdezoort/coupled-particle-edge-networks/scans/jets
export PYTHONPATH="/home/jdezoort/coupled-particle-edge-networks/src:${PYTHONPATH:-}"
mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs

DEPTH=4
WIDTH=128
D_HEAD=32
STAR_R=0.1
KNN_K=6
SEED="${SEED:-0}"
N_TRAIN=100000000
N_VAL=25000
BATCH=128
N_GPUS=4
MAX_STEPS="${MAX_STEPS:-40000}"
EPOCHS=1
T_EPOCH=10
ETA0="${ETA0:?Set ETA0}"
ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law
SWEEP="$ROOT/jetclass_output/adamw/sweep_lr"

HEAD=$((WIDTH / D_HEAD))
if (( HEAD * D_HEAD != WIDTH )); then
  echo "D=$WIDTH not divisible by d_h=$D_HEAD" >&2
  exit 1
fi

# Refuse ram-subset. 100M must stream.
python3 - <<'PY'
from cpen.apps.jets.jetclass_stream_datamodule import RAM_SUBSET_LIMIT
n = 100_000_000
if n <= RAM_SUBSET_LIMIT:
    raise SystemExit(f"n_train={n} would ram-subset (limit={RAM_SUBSET_LIMIT})")
print(f"[guard] n_train={n} > RAM_SUBSET_LIMIT={RAM_SUBSET_LIMIT} → STREAM")
PY

ETA_TAG=$(python3 -c "e=float('${ETA0}'); print(f'{e:g}'.replace('.','p'))")
R_TAG=$(python3 -c "e=float('${STAR_R}'); print(f'{e:g}'.replace('.','p'))")
RUN_TAG="hxfer-d${WIDTH}-aw-t${T_EPOCH}-n100M-s${MAX_STEPS}-k${KNN_K}-R${R_TAG}"

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none} partition=gpu-test (via a100≤59m)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
echo "study=hxfer-d128-adamw-t10-n100M"
echo "L=$DEPTH D=$WIDTH H=$HEAD d_h=$D_HEAD eta0=$ETA0 optimizer=adamw t_epoch=$T_EPOCH"
echo "n_train=$N_TRAIN (prescribed 100M, STREAM) epochs=$EPOCHS max_steps=$MAX_STEPS"
echo "B_gpu=$BATCH n_gpus=$N_GPUS B_global=$((BATCH * N_GPUS))"
echo "presentations=$((MAX_STEPS * BATCH * N_GPUS))  (must be < n_train: no wrap)"
echo "k=$KNN_K R=$STAR_R seed=$SEED UNIFORM-pool run_tag=$RUN_TAG"
if (( MAX_STEPS * BATCH * N_GPUS >= N_TRAIN )); then
  echo "ERROR: 40k×B would wrap the 100M pool" >&2
  exit 1
fi

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

mkdir -p "$SWEEP"
EXISTING=$(find "$SWEEP" -maxdepth 1 -type d \
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
  --optimizer adamw \
  --scheduler none \
  --t-epoch "$T_EPOCH" \
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
