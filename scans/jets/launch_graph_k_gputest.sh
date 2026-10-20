#!/bin/bash
# Cancel leftover a10080 Plot-1 jobs, trash v5a80 runs, start gpu-test k-scan.
#
# Run on a Della *login* node (not from the Cursor agent sandbox — Slurm is
# unreachable there)::
#
#   bash scans/jets/launch_graph_k_gputest.sh
#
set -euo pipefail

ROOT=/home/jdezoort/coupled-particle-edge-networks
SWEEP=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/jetclass_output/adam/sweep_lr
TRASH="$SWEEP/_trash_v5a80"

cd "$ROOT"

echo "$(date -Iseconds) cancel jc-gk-a80"
scancel -u "$USER" -n jc-gk-a80 2>/dev/null || true
sleep 2
if squeue -u "$USER" -n jc-gk-a80 -h | grep -q .; then
  echo "WARN: jc-gk-a80 still in queue:" >&2
  squeue -u "$USER" -n jc-gk-a80
fi

echo "$(date -Iseconds) trash v5a80 under $TRASH"
mkdir -p "$TRASH"
shopt -s nullglob
moved=0
(
  cd "$SWEEP"
  for p in *v5a80*; do
    case "$p" in _trash_v5a80*) continue ;; esac
    mv -- "$p" "$TRASH/" && moved=$((moved+1)) || true
  done
  echo "moved=$moved"
)
# async cleanup so launch is not blocked on NFS unlink
nohup rm -rf "$TRASH" >/tmp/rm_v5a80_trash.log 2>&1 &
echo "trash_rm_pid=$!"

rm -f "$ROOT/paper/data/graph_k_val_from_trainlog.csv"

echo "$(date -Iseconds) start gk gpu-test watchdog (MAX_CONCURRENT=3)"
MAX_CONCURRENT="${MAX_CONCURRENT:-3}" TARGET="${TARGET:-950}" \
  bash "$ROOT/scans/jets/start_graph_k_gputest_watchdogs.sh"

sleep 2
echo "=== queue ==="
squeue -u "$USER" -o '%.18i %.9P %.12j %.2t %.10M %R' | head -40
echo "=== watchdog log (tail) ==="
tail -15 "$HOME/.cache/jc_gk_watchdogs/watchdog_gk.log" 2>/dev/null || true
echo "DONE — panel (a) refreshes via plot_graph_construction_nature.py as cells finish"
