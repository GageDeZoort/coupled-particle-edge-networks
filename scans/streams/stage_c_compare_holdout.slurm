#!/bin/bash
#SBATCH --account=bhanin
#SBATCH --job-name=cpen-stagec
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-test
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=0:45:00
#SBATCH --output=scans/streams/slurm-%j.out
#SBATCH --error=scans/streams/slurm-%j.err

# Stage C head-to-head: mock vs last-block FT (optional full FT).
# Partition must be "gpu-test" (hyphen). "gputest" is rejected by the controller.
# Env:
#   MOCK_RUN_DIR   (required) mock pretrain run dir containing best.ckpt
#   LASTBLOCK_RUN_DIR (optional) — defaults to newest *lastblock* under RUN_ROOT
#   FULL_FT_RUN_DIR (optional) full maskholdout FT run dir

set -euo pipefail
module purge
module load anaconda3/2023.9
conda activate mamba-env
cd /home/jdezoort/coupled-particle-edge-networks

RUN_ROOT="${RUN_ROOT:-/scratch/gpfs/BHANIN/jdezoort/cpen_runs/blobs_real_mask_lastblock}"
MOCK_RUN_DIR="${MOCK_RUN_DIR:?Set MOCK_RUN_DIR to the mock pretrain run directory}"
LASTBLOCK_RUN_DIR="${LASTBLOCK_RUN_DIR:-}"
if [[ -z "${LASTBLOCK_RUN_DIR}" ]]; then
  LASTBLOCK_RUN_DIR=$(ls -td "${RUN_ROOT}"/stream_output/adamw/sweep_lr/*lastblock* 2>/dev/null | head -1 || true)
fi
if [[ -z "${LASTBLOCK_RUN_DIR}" || ! -f "${LASTBLOCK_RUN_DIR}/best.ckpt" ]]; then
  echo "No lastblock best.ckpt under ${RUN_ROOT}" >&2
  exit 1
fi

echo "Host=$(hostname) Job=${SLURM_JOB_ID} GPU=${CUDA_VISIBLE_DEVICES:-unset}"
echo "mock=${MOCK_RUN_DIR}"
echo "lastblock=${LASTBLOCK_RUN_DIR}"
echo "full_ft=${FULL_FT_RUN_DIR:-unset}"

EXTRA=()
if [[ -n "${FULL_FT_RUN_DIR:-}" && -f "${FULL_FT_RUN_DIR}/best.ckpt" ]]; then
  EXTRA+=(--ft-run-dir "${FULL_FT_RUN_DIR}")
fi

PYTHONPATH=src:. python scripts/stage_c_compare_holdout.py \
  --mock-run-dir "${MOCK_RUN_DIR}" \
  --lastblock-run-dir "${LASTBLOCK_RUN_DIR}" \
  "${EXTRA[@]}"

echo "Stage C compare done → notebooks/streams/figures/real_label_holdout_eval/"
