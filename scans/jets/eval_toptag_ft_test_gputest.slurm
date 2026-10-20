#!/bin/bash
# Full TopTagging test-set eval of a fine-tuned CAPEN ckpt (no training).
# Reports test AUC + bg rejection @50% / @30% (ParT Table 5 metrics).
#
# Default: η₀=0.025 full-FT best.ckpt (val AUC peak so far).
#
#   sbatch scans/jets/eval_toptag_ft_test_gputest.slurm
#   CKPT=/path/to/last.ckpt N_TEST=400000 sbatch --export=ALL,CKPT,N_TEST \
#     scans/jets/eval_toptag_ft_test_gputest.slurm
#
#SBATCH --account=bhanin
#SBATCH --job-name=tt-ft-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --constraint=gpu80
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/toptagging_finetune/logs/tt_ft_test_%j.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/toptagging_finetune/logs/tt_ft_test_%j.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

REPO=/home/jdezoort/coupled-particle-edge-networks
cd "$REPO/scans/jets"
export PYTHONPATH="${REPO}/src:${REPO}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

FT_ROOT="${FT_ROOT:-/scratch/gpfs/BHANIN/jdezoort/cpen_runs/toptagging_finetune}"
SWEEP="$FT_ROOT/toptagging_output/adam/sweep_lr"
DEFAULT_CKPT="$SWEEP/capen-llama-att_4_256_0p025_ln_op-incidence_onorm-degree_res-transformer-like_h8_nval20k_ntst0_nf7_gstar0p1_ft-initfrom-pstep781k-live-partint-p80-knn7-noloop-sk7-seed0-ft-L4D256-full-apool-cos781k-eta0p025-b256-g1-ep10-k7-R0p1/best.ckpt"
export CKPT="${CKPT:-$DEFAULT_CKPT}"
export N_TEST="${N_TEST:-400000}"
export BATCH="${BATCH:-256}"
export ETA0="${ETA0:-0.025}"
export FT_ROOT

echo "Host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "ckpt=$CKPT n_test=$N_TEST batch=$BATCH eta0=$ETA0"
test -f "$CKPT" || { echo "missing ckpt"; exit 1; }
nvidia-smi -L || true

# Real .py file (not `python -`) so DataLoader workers can spawn.
python eval_toptag_ft_test.py

echo "done toptag FT test eval"
