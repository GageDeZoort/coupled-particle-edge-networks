#!/bin/bash
# Submit k=16 Plot-1 extension (one packed 3×2 job on gpu-test).
set -euo pipefail
cd /home/jdezoort/coupled-particle-edge-networks
j=$(sbatch --parsable scans/jets/scaling_law_L4_graph_k16_shard_gputest.slurm)
echo "submitted jc-gk16-gt job=$j"
squeue -u "$USER" -n jc-gk16-gt -o '%.18i %.9P %.12j %.2t %.10M %R'
