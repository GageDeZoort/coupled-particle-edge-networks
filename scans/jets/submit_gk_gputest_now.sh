#!/bin/bash
# One-shot: submit first 3 gk cells + start pool watchdog.
set -euo pipefail
cd /home/jdezoort/coupled-particle-edge-networks
rm -f "$HOME/.cache/jc_gk_watchdogs/watchdog_gk.lock" "$HOME/.cache/jc_gk_watchdogs/STOP"
pkill -f '/graph_k_gputest_watchdog.sh' 2>/dev/null || true
sleep 1
SLURM=scans/jets/scaling_law_L4_graph_k_shard_gputest.slurm
for a in 0 1 2; do
  j=$(sbatch --parsable --array="$a" --job-name=jc-gk-gt "$SLURM")
  echo "submitted array=$a job=$j"
done
bash scans/jets/start_graph_k_gputest_watchdogs.sh
sleep 2
squeue -u "$USER" -o '%.18i %.9P %.12j %.2t %.10M %R'
tail -20 "$HOME/.cache/jc_gk_watchdogs/watchdog_gk.log"
