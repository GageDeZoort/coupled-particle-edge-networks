#!/bin/bash
# Plot 1 extension: k=16 at R★=0.125 on gpu-test (matched B_global=384).
#
#   One job runs the full 3×2 (shard × seed) grid sequentially.
#   ~8 min/cell → ~50 min; fits gpu-test 59m.
#
#   sbatch scans/jets/scaling_law_L4_graph_k16_shard_gputest.slurm
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-gk16-gt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=16
# Della routes time<=59m onto gputest; do NOT set partition=gputest.
#SBATCH --partition=a100
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --time=0:59:00
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_gk16_gt_%j.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_gk16_gt_%j.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

cd /home/jdezoort/coupled-particle-edge-networks/scans/jets
mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs

KNN_K=16
STAR_R="${STAR_R:-0.125}"
ETA0=0.25
WIDTH=256
HEAD=8
DEPTH=4
BATCH=96
N_GPUS=4
MAX_STEPS="${MAX_STEPS:-1000}"
N_TRAIN=$((MAX_STEPS * BATCH * N_GPUS))
N_SHARD=3
N_INIT=2
# Re-run only true missing shards after v6gt ram-subset collapse (keep shard0).
SHARDS="${SHARDS:-1 2}"
D_HEAD=$((WIDTH / HEAD))
R_TAG=$(python3 -c "e=float('${STAR_R}'); print(f'{e:g}'.replace('.','p'))")

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  k=$KNN_K"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
echo "study=graph-k16-shard-v6gt2  L=$DEPTH  D=$WIDTH  H=$HEAD  d_head=$D_HEAD"
echo "eta0=$ETA0  starR=$STAR_R  knn=$KNN_K  B_global=$((BATCH * N_GPUS))  max_steps=$MAX_STEPS"
echo "shards=$SHARDS × ${N_INIT} inits (sequential)"

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

fail=0
for SHARD in $SHARDS; do
  for INIT in $(seq 0 $((N_INIT - 1))); do
    SEED=$INIT
    TRAIN_SKIP=$((SHARD * N_TRAIN))
    RUN_TAG="graph-k-b${BATCH}-g${N_GPUS}-s${MAX_STEPS}-R${R_TAG}-shard${SHARD}-seed${SEED}-v6gt2"
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
    rc=$?
    set -e
    # Exit 0 is not enough: empty stream plans used to end fit at step 0.
    SWEEP=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/jetclass_output/adam/sweep_lr
    ckpt=$(ls "$SWEEP"/*"${RUN_TAG}"/best*.ckpt 2>/dev/null | head -1 || true)
    if (( rc != 0 )) || [[ -z "$ckpt" ]]; then
      echo "FAIL k=$KNN_K shard=$SHARD seed=$SEED rc=$rc ckpt=${ckpt:-missing}"
      fail=1
    else
      echo "OK k=$KNN_K shard=$SHARD seed=$SEED ckpt=$(basename "$ckpt")"
    fi
  done
done

if (( fail )); then
  echo "job finished with failures for k=$KNN_K"
  exit 1
fi
echo "ALL 6 cells OK for k=$KNN_K"
