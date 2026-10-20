#!/bin/bash
# Graph construction Plot 1 (v6gt): kNN-k sweep with shard×init replicates on gpu-test.
#
#   Matched: 4×A100, B_gpu=96 → B_global=384, 1000 steps
#   (~384k jet-updates = one pass over a 384k train shard).
#   L=4 D=256 H=8 η₀=0.25 Adam, kin7+part-interaction, R★=0.125.
#
#   Per k: 3 disjoint train shards × 2 inits = 6 runs.
#   Array flat index 0..41:
#     k_idx = task // 6   → k ∈ {2,4,6,8,10,12,14}
#     shard = (task % 6) // 2  → 0,1,2   (skip = shard * 384000)
#     init  = (task % 6) % 2   → seed 0,1
#
#   Prefer the pool watchdog (keeps ≤3 concurrent under qos=gpu-test)::
#     bash scans/jets/start_graph_k_gputest_watchdogs.sh
#
#   Or one-shot (may queue all 42 and stall on MaxJobs)::
#     sbatch scans/jets/scaling_law_L4_graph_k_shard_gputest.slurm
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-gk-gt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
# Della routes time<=59m onto gputest; do NOT set partition=gputest.
#SBATCH --partition=a100
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=0:59:00
#SBATCH --array=0-41
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_gk_gt_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_gk_gt_%A_%a.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

cd /home/jdezoort/coupled-particle-edge-networks/scans/jets
mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs

TASK=${SLURM_ARRAY_TASK_ID:-0}
KS=(2 4 6 8 10 12 14)
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

STAR_R="${STAR_R:-0.125}"
ETA0=0.25
WIDTH=256
HEAD=8
DEPTH=4
SEED=$INIT
BATCH=96
N_GPUS=4
MAX_STEPS="${MAX_STEPS:-1000}"
# One unique pass over the shard at B_global=384.
N_TRAIN=$((MAX_STEPS * BATCH * N_GPUS))
TRAIN_SKIP=$((SHARD * N_TRAIN))
D_HEAD=$((WIDTH / HEAD))
R_TAG=$(python3 -c "e=float('${STAR_R}'); print(f'{e:g}'.replace('.','p'))")
RUN_TAG="graph-k-b${BATCH}-g${N_GPUS}-s${MAX_STEPS}-R${R_TAG}-shard${SHARD}-seed${SEED}-v6gt2"

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  TASK=$TASK  k=$KNN_K shard=$SHARD seed=$SEED"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
echo "study=graph-k-shard-v6gt  L=$DEPTH  D=$WIDTH  H=$HEAD  d_head=$D_HEAD"
echo "eta0=$ETA0  optimizer=adam  starR=$STAR_R  knn=$KNN_K"
echo "n_train=$N_TRAIN  train_skip=$TRAIN_SKIP  max_steps=$MAX_STEPS"
echo "B_gpu=$BATCH  n_gpus=$N_GPUS  B_global=$((BATCH * N_GPUS))  run_tag=$RUN_TAG"

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
  --optimizer adam \
  --t-epoch 1 \
  --residual-structure transformer-like \
  --etas "$ETA0" \
  --epochs 1 \
  --max-steps "$MAX_STEPS" \
  --n-train "$N_TRAIN" \
  --train-skip-jets "$TRAIN_SKIP" \
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
  --n-test 0 \
  --seed "$SEED" \
  --log-every-n-steps 50 \
  --layer-norm \
  --check-val-every-n-epoch 1 \
  --val-check-interval 500 \
  --checkpoint-every-n-steps 500 \
  --checkpoint-monitor val_loss \
  --checkpoint-mode min \
  --no-torch-compile
