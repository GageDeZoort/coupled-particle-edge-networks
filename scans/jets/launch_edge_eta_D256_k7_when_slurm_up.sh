#!/usr/bin/env bash
# Wait for Slurm controller, submit first η wave, then start the pool watchdog.
#
#   nohup bash scans/jets/launch_edge_eta_D256_k7_when_slurm_up.sh \
#     >> ~/.cache/jc_edge_k7_watchdogs/launch.out 2>&1 &
#
set -uo pipefail
REPO=/home/jdezoort/coupled-particle-edge-networks
CACHE="${HOME}/.cache/jc_edge_k7_watchdogs"
SLURM="$REPO/scans/jets/scaling_law_L4_edge_eta_D256_k7_gputest_s20k.slurm"
mkdir -p "$CACHE"
LOG="$CACHE/launch.out"

log() { echo "$(date -Iseconds) $*"; }

log "waiting for slurmctld…"
while true; do
  if timeout 20 scontrol ping 2>&1 | grep -qi 'UP'; then
    log "slurmctld UP"
    break
  fi
  log "slurmctld not up yet; sleep 30"
  sleep 30
done

# First wave: η ∈ {0.01, 0.05, 0.1} (tasks 0–2). Watchdog fills 3–4 and resumes.
for attempt in $(seq 1 30); do
  out=$(timeout 90 sbatch --parsable --job-name=jc-edge-k7 --array=0-2 \
    --export=ALL,MAX_STEPS=40000 \
    "$SLURM" 2>&1) && {
    log "submitted wave0 job=$out"
    echo "$out" > "$REPO/scans/jets/edge_eta_D256_k7_gputest_jobid.txt"
    break
  }
  log "sbatch attempt=$attempt failed: $out"
  sleep 20
  if (( attempt == 30 )); then
    log "giving up on initial wave; starting watchdog anyway"
  fi
done

bash "$REPO/scans/jets/start_edge_eta_D256_k7_gputest_watchdogs.sh"
log "launch complete"
