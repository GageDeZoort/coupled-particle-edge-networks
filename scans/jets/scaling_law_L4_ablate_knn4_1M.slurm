#!/bin/bash
# 1M unique-pass: star-R=0.2 part-interaction + directed ΔR-kNN (k=4) 2-edges.
#   L=4, D=256, H=8, η₀=0.25, p=80, 1×A100
#
# B=256 (B=512 OOM'd on 40GB with M=N+N·k edges). max_steps = ceil(1M/B)
# so a full unique pass matches the B=512 partint control's jet count.
# gpu-test (~59m): compare mid-run vals at equal wall time vs other ablations;
# resubmit same script to resume until max_steps if a full pass is needed.
# M22 stays hyperedge-only (deg>2); kNN 2-edges feed M12/M21 only (default
# capen-llama-att; do NOT pass --all-edge-m22).
#
#   sbatch scans/jets/scaling_law_L4_ablate_knn4_1M.slurm
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-ablate-knn4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --qos=gpu-test
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=0:59:00
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_ablate_knn4_%j.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_ablate_knn4_%j.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

cd /home/jdezoort/coupled-particle-edge-networks/scans/jets
mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs

ETA0=0.25
WIDTH=256
HEAD=8
DEPTH=4
STAR_R=0.2
KNN_K=4
N_TRAIN=1000000
N_VAL=25000
SEED=0
# Half the OOM batch; match 1M-jet unique pass via more steps.
BATCH=256
MAX_STEPS=$(( (N_TRAIN + BATCH - 1) / BATCH ))

echo "PWD=$(pwd)"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  label=partint+knn${KNN_K}  M22=hyperedge-only"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "L=$DEPTH D=$WIDTH H=$HEAD eta0=$ETA0 starR=$STAR_R knn=$KNN_K"
echo "n_train=$N_TRAIN max_steps=$MAX_STEPS B=$BATCH  (equiv 1M jets @ B=512 → 1954 steps)"

export PYTHONFAULTHANDLER=1
export PYTHONNOUSERSITE=1

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
  --data-root /scratch/gpfs/BHANIN/jdezoort/datasets/jetclass \
  --depth "$DEPTH" --width "$WIDTH" --heads "$HEAD" \
  --dropout 0.0 \
  --root /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law \
  --run-tag ablate-b256 \
  --batch-size "$BATCH" \
  --n-gpus 1 \
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
  --checkpoint-every-n-steps 200 \
  --checkpoint-monitor val_loss \
  --checkpoint-mode min \
  --no-torch-compile
