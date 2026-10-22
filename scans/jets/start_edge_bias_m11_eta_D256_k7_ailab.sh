#!/usr/bin/env bash
# Submit arm-E edge-bias M11 η₀ sweep (L4/D256, 30k) on ailab qos=gpu-test.
# Does not stop the typed -s1i watchdog; new jobs will queue under QOSMaxJobs.
#
#   bash scans/jets/start_edge_bias_m11_eta_D256_k7_ailab.sh
#
set -euo pipefail
REPO=/home/jdezoort/coupled-particle-edge-networks
SLURM="$REPO/scans/jets/scaling_law_L4_edge_bias_m11_eta_D256_k7_ailab_gputest.slurm"
cd "$REPO"
sbatch "$SLURM"
echo "submitted edge-bias-m11 arm E (η₀ ∈ {0.1, 0.25, 0.5}, 30k, -eb1)"
