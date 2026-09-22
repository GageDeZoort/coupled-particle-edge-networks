#!/bin/bash
# Single-GPU memory ladder for the L=4 scaling-law width family.
# Tries B=512 at D=512 then D=768 (and 640/896/1024) in bf16 with live star-R.
#
#   sbatch scans/jets/probe_scaling_gpu_mem.slurm
#
#SBATCH --account=bhanin
#SBATCH --job-name=jc-scale-mem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu40
#SBATCH --time=0:25:00
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_scale_mem_%j.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_scale_mem_%j.err

set -euo pipefail

module purge
module load anaconda3/2023.9
conda activate mamba-env

cd /home/jdezoort/coupled-particle-edge-networks

echo "Host=$(hostname)  job=$SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true
python - <<'PY'
import torch
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
if torch.cuda.is_available():
    t = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"total_mem_gib {t:.2f}")
PY

export PYTHONPATH=src
export PYTHONFAULTHANDLER=1

for W in 512 640 768 896 1024; do
  echo
  echo "========== D=${W} B=512 =========="
  OUT=/tmp/probe_${W}.json
  python scans/jets/probe_scaling_gpu_mem.py --depth 4 --width "$W" --batch-size 512 --num-particles 80 --json-out "$OUT"
  if python -c "import json; print(json.load(open('$OUT')).get('oom'))" | grep -q True; then
    echo "Stopping width ladder after OOM at D=${W}"
    echo "Retrying D=${W} at B=256 then B=128"
    python scans/jets/probe_scaling_gpu_mem.py --depth 4 --width "$W" --batch-size 256 --num-particles 80
    python scans/jets/probe_scaling_gpu_mem.py --depth 4 --width "$W" --batch-size 128 --num-particles 80
    break
  fi
done
