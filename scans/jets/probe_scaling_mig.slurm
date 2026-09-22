#!/bin/bash
# Can a 10 GB Della MIG hold L=4 D=512 at any useful batch?
#SBATCH --account=bhanin
#SBATCH --job-name=jc-scale-mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --time=0:15:00
#SBATCH --output=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_scale_mig_%j.out
#SBATCH --error=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_scale_mig_%j.err

set -euo pipefail
module purge
module load anaconda3/2023.9
conda activate mamba-env
cd /home/jdezoort/coupled-particle-edge-networks
export PYTHONPATH=src
echo "Host=$(hostname)"
nvidia-smi -L || true
python - <<'PY'
import torch
p=torch.cuda.get_device_properties(0)
print(p.name, p.total_memory/1024**3)
PY
for B in 32 16 8; do
  echo "========== D=512 B=${B} on MIG =========="
  python scans/jets/probe_scaling_gpu_mem.py --depth 4 --width 512 --batch-size "$B" --num-particles 80
done
