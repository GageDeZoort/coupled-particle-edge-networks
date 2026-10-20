#!/bin/bash
# Graph construction Plot 2 (v6gt): R★ sweep matched to Plot 1 on gpu-test.
#
#   Matched: 4×A100, B_gpu=96 → B_global=384, 1000 steps, k★=12.
#   R★ ∈ {0.05, 0.1, 0.15, 0.2, 0.25, 0.075, 0.125}; 3 shards × 2 seeds.
#   (0.075 / 0.125 appended so tasks 0–29 keep their original indices.)
#   Array flat index 0..41:
#     r_idx = task // 6   → R★
#     shard = (task % 6) // 2
#     init  = (task % 6) % 2
#
#   Della routes time<=59m onto gputest; do NOT set partition=gputest/mig.
#   (mig only allows gres=gpu:1 and live-graph OOMs above ~B=32 on 1g.10gb.)
#
#   bash scans/jets/submit_graph_R_gputest.sh
#   sbatch scans/jets/scaling_law_L4_graph_R_shard_gputest.slurm
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-gR-gt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
#SBATCH --partition=a100
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=0:59:00
#SBATCH --array=0-41
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_gR_gt_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_gR_gt_%A_%a.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

cd /home/jdezoort/coupled-particle-edge-networks/scans/jets
mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs

TASK=${SLURM_ARRAY_TASK_ID:-0}
RADII=(0.05 0.1 0.15 0.2 0.25 0.075 0.125)
N_SHARD=3
N_INIT=2
N_PER_R=$((N_SHARD * N_INIT))

if (( TASK < 0 || TASK >= ${#RADII[@]} * N_PER_R )); then
  echo "array index $TASK out of range (need 0-$(( ${#RADII[@]} * N_PER_R - 1 )))" >&2
  exit 1
fi

R_IDX=$((TASK / N_PER_R))
REM=$((TASK % N_PER_R))
SHARD=$((REM / N_INIT))
INIT=$((REM % N_INIT))
STAR_R=${RADII[$R_IDX]}
KNN_K="${KNN_K:-12}"

ETA0=0.25
WIDTH=256
HEAD=8
DEPTH=4
SEED=$INIT
BATCH=96
N_GPUS=4
MAX_STEPS="${MAX_STEPS:-1000}"
N_TRAIN=$((MAX_STEPS * BATCH * N_GPUS))
TRAIN_SKIP=$((SHARD * N_TRAIN))
D_HEAD=$((WIDTH / HEAD))
R_TAG=$(python3 -c "e=float('${STAR_R}'); print(f'{e:g}'.replace('.','p'))")
# v6gt2: force streaming when train_skip>0 (v6gt ram-subset ignored skip → identical shards)
RUN_TAG="graph-R-b${BATCH}-g${N_GPUS}-s${MAX_STEPS}-k${KNN_K}-R${R_TAG}-shard${SHARD}-seed${SEED}-v6gt2"

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  TASK=$TASK  R=$STAR_R k=$KNN_K shard=$SHARD seed=$SEED"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
echo "study=graph-R-shard-v6gt  L=$DEPTH  D=$WIDTH  H=$HEAD  d_head=$D_HEAD"
echo "eta0=$ETA0  optimizer=adam  starR=$STAR_R  knn=$KNN_K"
echo "n_train=$N_TRAIN  train_skip=$TRAIN_SKIP  max_steps=$MAX_STEPS"
echo "B_gpu=$BATCH  n_gpus=$N_GPUS  B_global=$((BATCH * N_GPUS))  run_tag=$RUN_TAG"

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python - <<'PY'
import sys, torch
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
