#!/bin/bash
# Edge-bias M11 (arm E: --edge-bias-m11 dynamic) — L4/D256 on Della gputest.
# Typed §1 + short run_tag eb1E-gt. Budget 40k (warmup_flat_cosine),
# matched to typed-s1i / edge-bias xfer length.
# B_gpu=64 × 4 → B_global=256 (same as other Della gputest sweeps).
#
#   sbatch scans/jets/scaling_law_L4_edge_bias_m11_eta_D256_k7_gputest.slurm
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-eb1-gt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=a100
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=0:59:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_eb1_gt_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_eb1_gt_%A_%a.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

REPO=/home/jdezoort/coupled-particle-edge-networks
cd "$REPO/scans/jets"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs

TASK=${SLURM_ARRAY_TASK_ID:-0}
# Index: 0→0.05  1→0.1  2→0.25  3→0.5
# Ongoing jobs 15970682_{0,1,2} used the old map 0→0.1,1→0.25,2→0.5 — parse eta0= in logs.
ETAS=(0.05 0.1 0.25 0.5)
if (( TASK < 0 || TASK >= ${#ETAS[@]} )); then
  echo "array index $TASK out of range" >&2
  exit 1
fi

ETA0=${ETAS[$TASK]}
DEPTH=4
WIDTH=256
D_HEAD=32
HEAD=$((WIDTH / D_HEAD))
STAR_R=0.1
KNN_K=7
BATCH="${BATCH:-64}"
N_GPUS=4
MAX_STEPS="${MAX_STEPS:-40000}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
FLAT_STEPS="${FLAT_STEPS:-32000}"
COSINE_STEPS="${COSINE_STEPS:-7000}"
MIN_FRAC="${MIN_FRAC:-0.01}"
SEED="${SEED:-0}"
N_VAL=25000
N_TRAIN=$((MAX_STEPS * BATCH * N_GPUS))
EPOCHS=2
ROOT=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law
# Short tag — sweep_common adds typed/eb-dyn/sched tokens.
RUN_TAG="eb1E-gt"

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}  TASK=$TASK  (Della gputest via time≤59m)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
echo "edge-bias-m11 dynamic (arm E)  L=$DEPTH D=$WIDTH H=$HEAD eta0=$ETA0"
echo "B_gpu=$BATCH n_gpus=$N_GPUS B_global=$((BATCH*N_GPUS)) max_steps=$MAX_STEPS"
echo "sched=warmup_flat_cosine w=$WARMUP_STEPS f=$FLAT_STEPS c=$COSINE_STEPS min=$MIN_FRAC"
echo "run_tag=$RUN_TAG"

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("No CUDA device visible")
n = torch.cuda.device_count()
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU{i} {p.name} {p.total_memory/1024**3:.1f} GiB")
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
  --scheduler warmup_flat_cosine \
  --warmup-steps "$WARMUP_STEPS" \
  --flat-steps "$FLAT_STEPS" \
  --cosine-steps "$COSINE_STEPS" \
  --lr-decay-min-frac "$MIN_FRAC" \
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
echo "done eb1-armE-gt eta0=$ETA0 rc=$rc"
exit 0
