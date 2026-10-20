#!/bin/bash
#SBATCH --job-name=mup-1step
#SBATCH --account=bhanin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
# Della routes time<=59m onto gputest; do NOT set partition=gputest.
#SBATCH --partition=a100
#SBATCH --time=0:59:00
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/mup_one_step_%j.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/mup_one_step_%j.err

# One-step Adam μP: ρ / Δρ of X,E at encode / last residual / output
# across η₀ ∈ {0.01, 0.05, 0.1, 0.25, 0.5, 1}, k★=12 R★=0.1, attn pool.
# 4 widths × 6 η₀ × 10×10 trials — synthetic B=8 on 1×A100 (gputest 59m).
# Post 2026-10-02 CompleteP encoder fix; see paper/COMPLETEP_SCALES.md.
# Bad pre-fix uniform archive: paper/archive/stability_20261002_encoder_N01_bug/

set -euo pipefail
REPO="${REPO:-$HOME/coupled-particle-edge-networks}"
cd "$REPO"
mkdir -p /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs
mkdir -p "$REPO/paper/data" "$REPO/paper/figures"

# Archive prior QKV-style CSV if present (schema changed).
if [[ -f "$REPO/paper/data/stability_one_step.csv" ]]; then
  ts=$(date +%Y%m%d_%H%M%S)
  mkdir -p "$REPO/paper/data/archive_stability"
  mv -f "$REPO/paper/data/stability_one_step.csv" \
    "$REPO/paper/data/archive_stability/stability_one_step_pre_xe_${ts}.csv" || true
  mv -f "$REPO/paper/data/stability_one_step_trials.csv" \
    "$REPO/paper/data/archive_stability/stability_one_step_trials_pre_xe_${ts}.csv" || true
fi

python scans/jets/mup_one_step_transfer_probe.py \
  --synthetic \
  --archs 2:128,4:256,6:512,8:1024 \
  --cells 12:0.10 \
  --etas 0.01,0.05,0.1,0.25,0.5,1.0 \
  --pools attn \
  --batch-size 8 \
  --n-particles 128 \
  --seed 0 \
  --n-seeds 10 \
  --n-batches 10 \
  --device cuda \
  --out "$REPO/paper/data/stability_one_step.csv"

python scans/jets/plot_stability_nature.py --pool attn
