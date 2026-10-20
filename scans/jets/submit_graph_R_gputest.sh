#!/bin/bash
# Submit Plot-2 R★ sweep on gpu-test (4×A100, B=96×4=384), one cell per array task.
set -euo pipefail
cd /home/jdezoort/coupled-particle-edge-networks
j=$(sbatch --parsable scans/jets/scaling_law_L4_graph_R_shard_gputest.slurm)
echo "submitted jc-gR-gt array=0-29 job=$j"
squeue -u "$USER" -n jc-gR-gt -o '%.18i %.9P %.12j %.2t %.10M %R'
