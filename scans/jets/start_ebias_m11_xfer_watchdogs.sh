#!/usr/bin/env bash
# Start edge-bias M11 μP transfer watchdogs (ailab + Della gputest).
# Does NOT flood-submit — each watchdog owns at most MAX_SLOTS jobs and
# respects the shared qos=gpu-test cap (≈3).
#
# Grid: η₀ ∈ {0.01,0.05,0.1,0.25,0.5} × {L4/D256, L6/D512, L3/D192} → 40k
#   (D192 = standard L3 transfer width; "D194" treated as D192.)
#
#   bash scans/jets/start_ebias_m11_xfer_watchdogs.sh
# Stop one:
#   touch ~/.cache/jc_ebX_L4D256_gputest_watchdogs/STOP
# Stop all:
#   for d in ~/.cache/jc_ebX_*_watchdogs; do touch "$d/STOP"; done
#
set -euo pipefail
cd /home/jdezoort/coupled-particle-edge-networks
WD=scans/jets/ebias_m11_xfer_watchdog.sh
chmod +x "$WD" scans/jets/scaling_law_ebias_m11_xfer_eta.slurm

TARGET="${TARGET:-40000}"
MAX_SLOTS="${MAX_SLOTS:-1}"
SLEEP_SEC="${SLEEP_SEC:-90}"

ARCHS=(L4D256 L6D512 L3D192)
PARTS=(gputest ailab)

# Restart clean: stop any prior xfer watchdogs (not the 30k eb1E canvas watch).
mapfile -t _old < <(pgrep -f 'scans/jets/ebias_m11_xfer_watchdog.sh' || true)
for p in "${_old[@]:-}"; do
  [[ -z "$p" ]] && continue
  kill "$p" 2>/dev/null || true
done
sleep 2
# Drop stale flocks so a restart cannot lose to a dying holder.
rm -f "${HOME}"/.cache/jc_ebX_*_watchdogs/watchdog.lock \
  "${HOME}/.cache/jc_ebX_submit.lock" 2>/dev/null || true

for arch in "${ARCHS[@]}"; do
  for part in "${PARTS[@]}"; do
    CACHE="${HOME}/.cache/jc_ebX_${arch}_${part}_watchdogs"
    mkdir -p "$CACHE"
    rm -f "$CACHE/STOP" "$CACHE/watchdog.lock" \
      "/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/STOP_ebX_${arch}_${part}"

    nohup env \
      ARCH_KEY="$arch" PARTITION="$part" \
      TARGET="$TARGET" MAX_STEPS="$TARGET" MAX_SLOTS="$MAX_SLOTS" SLEEP_SEC="$SLEEP_SEC" \
      bash "$WD" >>"$CACHE/nohup.out" 2>&1 &
    echo "started ebX ${arch}/${part} pid=$!  MAX_SLOTS=$MAX_SLOTS  nohup=$CACHE/nohup.out"
  done
done

sleep 2
echo '=== ebX watchdog procs ==='
ps -u "$USER" -o pid,etime,cmd | grep -F 'ebias_m11_xfer_watchdog' | grep -v grep || true
echo '=== recent nohup lines ==='
for arch in "${ARCHS[@]}"; do
  for part in "${PARTS[@]}"; do
    f="${HOME}/.cache/jc_ebX_${arch}_${part}_watchdogs/nohup.out"
    echo "--- $arch/$part ---"
    tail -3 "$f" 2>/dev/null || echo '(empty)'
  done
done
echo
echo "Note: current jc-eb1-gt 30k probe jobs are separate (tag eb1E-gt)."
echo "Transfer watchdogs use ebX-L{3,4,6}D{192,256,512}-{gt,ailab} → 40k const LR."
