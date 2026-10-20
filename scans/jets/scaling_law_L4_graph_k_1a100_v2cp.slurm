#!/bin/bash
# Panel (a) k-sweep — CompleteP v2 on 1×A100 (non-ailab).
#
#   L=4 D=256 H=8 · η₀=0.1 · Adam · B=128 · 2500 steps · R★=0.1
#   k ∈ {2,3,4,5,6,7,8,10} × 3 shards × 2 seeds = 48 cells
#   Tag: v2cp
#
#   sbatch scans/jets/scaling_law_L4_graph_k_1a100_v2cp.slurm
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-gk-v2cp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
# 1:05 on a100 → regular queue (not 59m gputest)
#SBATCH --partition=a100
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=1:05:00
# Cap concurrency to avoid QOSMaxJobsPerUserLimit stalls
#SBATCH --array=0-47%8
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_gk_v2cp_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_gk_v2cp_%A_%a.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

REPO=/home/jdezoort/coupled-particle-edge-networks
cd "$REPO/scans/jets"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs
mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/graph_construction_completep_v2/k_R0p1

TASK=${SLURM_ARRAY_TASK_ID:-0}
KS=(2 3 4 5 6 7 8 10)
N_SHARD=3
N_INIT=2
N_PER_K=$((N_SHARD * N_INIT))

if (( TASK < 0 || TASK >= ${#KS[@]} * N_PER_K )); then
  echo "array index $TASK out of range (need 0-$(( ${#KS[@]} * N_PER_K - 1 )))" >&2
  exit 1
fi

K_IDX=$((TASK / N_PER_K))
REM=$((TASK % N_PER_K))
SHARD=$((REM / N_INIT))
INIT=$((REM % N_INIT))
KNN_K=${KS[$K_IDX]}

STAR_R=0.1
ETA0=0.1
WIDTH=256
HEAD=8
DEPTH=4
SEED=$INIT
BATCH=128
N_GPUS=1
MAX_STEPS=2500
# One unique pass over the shard at B_global=128.
N_TRAIN=$((MAX_STEPS * BATCH * N_GPUS))
TRAIN_SKIP=$((SHARD * N_TRAIN))
R_TAG=$(python3 -c "e=float('${STAR_R}'); print(f'{e:g}'.replace('.','p'))")
ETA_TAG=$(python3 -c "e=float('${ETA0}'); print(f'{e:g}'.replace('.','p'))")
RUN_TAG="graph-k-b${BATCH}-g${N_GPUS}-s${MAX_STEPS}-R${R_TAG}-shard${SHARD}-seed${SEED}-v2cp"

echo "PWD=$(pwd)"
echo "Host: $(hostname) SLURM_JOB_ID=${SLURM_JOB_ID:-none} TASK=$TASK"
echo "study=graph-k-completep-v2  L=$DEPTH  D=$WIDTH  H=$HEAD"
echo "eta0=$ETA0  optimizer=adam  starR=$STAR_R  knn=$KNN_K  seed=$SEED"
echo "n_train=$N_TRAIN  train_skip=$TRAIN_SKIP  max_steps=$MAX_STEPS"
echo "B_gpu=$BATCH  n_gpus=$N_GPUS  B_global=$((BATCH * N_GPUS))  run_tag=$RUN_TAG"
nvidia-smi -L || true

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("No CUDA device visible")
p = torch.cuda.get_device_properties(0)
print(f"n_cuda={torch.cuda.device_count()}  GPU0 {p.name} {p.total_memory/1024**3:.1f} GiB")
if torch.cuda.device_count() < 1:
    sys.exit("Need 1 GPU")
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
  --epochs 1 \
  --max-steps "$MAX_STEPS" \
  --n-train "$N_TRAIN" \
  --train-skip-jets "$TRAIN_SKIP" \
  --n-test 0 \
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
  --n-val 25000 \
  --seed "$SEED" \
  --log-every-n-steps 50 \
  --layer-norm \
  --check-val-every-n-epoch 1 \
  --val-check-interval 500 \
  --checkpoint-every-n-steps 500 \
  --checkpoint-monitor val_loss \
  --checkpoint-mode min \
  --no-torch-compile

echo "done k=$KNN_K shard=$SHARD seed=$SEED eta0=$ETA0"
