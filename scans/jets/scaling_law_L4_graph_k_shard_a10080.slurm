#!/bin/bash
# Graph construction Plot 1 (v5): one Slurm job per k on 1×A100-80GB.
#
#   Each array task runs the full 3×2 (shard × init) grid for one k at
#   B=384, 1000 steps (~384k jet-updates / shard), R★=0.125.
#   Seven independent 1-GPU jobs (k ∈ {2,4,6,8,10,12,14}); submit all at once
#   so up to 7 of 8 free 80GB cards can run in parallel.
#
#   Wall ~10–15 min/cell → ~1–1.5 h for 6 cells; 2.5 h budget leaves margin
#   for k=14. Abort if landed on <70 GiB.
#
#   sbatch scans/jets/scaling_law_L4_graph_k_shard_a10080.slurm
#   sbatch --array=6 scans/jets/...   # k=14 only
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-gk-a80
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --constraint=gpu80
#SBATCH --qos=gpu-short
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=2:30:00
#SBATCH --array=0-6
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_gk_a80_%A_%a.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_gk_a80_%A_%a.err

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

if (( TASK < 0 || TASK >= ${#KS[@]} )); then
  echo "array index $TASK out of range (need 0-$(( ${#KS[@]} - 1 )))" >&2
  exit 1
fi

KNN_K=${KS[$TASK]}
STAR_R="${STAR_R:-0.125}"
ETA0=0.25
WIDTH=256
HEAD=8
DEPTH=4
BATCH=384
N_GPUS=1
MAX_STEPS=1000
N_TRAIN=$((MAX_STEPS * BATCH * N_GPUS))
D_HEAD=$((WIDTH / HEAD))
R_TAG=$(python3 -c "e=float('${STAR_R}'); print(f'{e:g}'.replace('.','p'))")

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  TASK=$TASK  k=$KNN_K"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
echo "study=graph-k-shard-v5a80  L=$DEPTH  D=$WIDTH  H=$HEAD  d_head=$D_HEAD"
echo "eta0=$ETA0  optimizer=adam  starR=$STAR_R  knn=$KNN_K"
echo "n_train=$N_TRAIN  max_steps=$MAX_STEPS  B_gpu=$BATCH  n_gpus=$N_GPUS"
echo "grid: ${N_SHARD} shards × ${N_INIT} inits (sequential in this job)"

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python - <<'PY'
import os, sys, torch
if not torch.cuda.is_available():
    sys.exit("No CUDA device visible")
props = torch.cuda.get_device_properties(0)
gib = props.total_memory / 1024**3
print(
    f"GPU0 {props.name} {gib:.1f} GiB  "
    f"count={torch.cuda.device_count()}  "
    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}"
)
if gib < 70:
    sys.exit(
        f"Need A100 80GB (#SBATCH --constraint=gpu80); this node has {gib:.1f} GiB."
    )
PY

fail=0
for SHARD in $(seq 0 $((N_SHARD - 1))); do
  for INIT in $(seq 0 $((N_INIT - 1))); do
    SEED=$INIT
    TRAIN_SKIP=$((SHARD * N_TRAIN))
    RUN_TAG="graph-k-b${BATCH}-g${N_GPUS}-s${MAX_STEPS}-R${R_TAG}-shard${SHARD}-seed${SEED}-v5a80"
    echo "======== k=$KNN_K shard=$SHARD seed=$SEED skip=$TRAIN_SKIP tag=$RUN_TAG ========"
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
    if (( rc != 0 )); then
      echo "FAIL k=$KNN_K shard=$SHARD seed=$SEED rc=$rc"
      fail=1
      # continue other cells so one OOM/timeout does not wipe the whole k
    else
      echo "OK k=$KNN_K shard=$SHARD seed=$SEED"
    fi
  done
done

if (( fail )); then
  echo "job finished with failures for k=$KNN_K"
  exit 1
fi
echo "ALL 6 cells OK for k=$KNN_K"
