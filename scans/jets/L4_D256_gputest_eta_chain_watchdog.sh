#!/bin/bash
# DISABLED_BY_AGENT 2026-09-30: L4D256/wfc watchdogs clogging gpu-test.
echo "DISABLED: $0 — L4D256/wfc watchdogs retired. Remove this guard to re-enable." >&2
exit 0

#!/bin/bash
# Sequential L4 D=256 H=8 baseline gpu-test arms: finish each η₀ to TARGET, then pivot.
#
# Default chain fleshes out the η₀ wings after the mid-grid {0.1,0.25,0.5} is done:
#   η₀=1 → (on TARGET) → η₀=0.05
#
#   ETA_CHAIN="1 0.05" TARGET=45000 SLEEP_SEC=900 \
#     nohup bash scans/jets/L4_D256_gputest_eta_chain_watchdog.sh &
#
set -uo pipefail
cd /home/jdezoort/coupled-particle-edge-networks

ETA_CHAIN="${ETA_CHAIN:-1 0.05}"
TARGET="${TARGET:-45000}"
SLEEP_SEC="${SLEEP_SEC:-900}"
MAX_RESUBMITS="${MAX_RESUBMITS:-200}"
WD=scans/jets/L4_D256_gputest_watchdog.sh

LOG_CANDIDATES=(
  /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/watchdog_L4D256_gputest_eta_chain.log
  "${HOME}/.cache/jc_L4D256_watchdogs/watchdog_eta_chain.log"
)
LOG=""
for cand in "${LOG_CANDIDATES[@]}"; do
  mkdir -p "$(dirname "$cand")" 2>/dev/null || true
  if ( : >>"$cand" ) 2>/dev/null; then
    LOG="$cand"
    break
  fi
done
if [[ -z "$LOG" ]]; then
  echo "ERROR: cannot write chain watchdog log" >&2
  exit 1
fi

echo "$(date -Iseconds) chain start ETA_CHAIN=[$ETA_CHAIN] TARGET=$TARGET SLEEP_SEC=$SLEEP_SEC" | tee -a "$LOG"

for ETA0 in $ETA_CHAIN; do
  ETA_TAG=$(python3 -c "e=float('${ETA0}'); print(f'{e:g}'.replace('.','p'))")
  echo "$(date -Iseconds) === pivot/start ETA0=$ETA0 (tag=e${ETA_TAG}) ===" | tee -a "$LOG"
  ETA0="$ETA0" TARGET="$TARGET" SLEEP_SEC="$SLEEP_SEC" MAX_RESUBMITS="$MAX_RESUBMITS" \
    bash "$WD"
  ec=$?
  echo "$(date -Iseconds) === ETA0=$ETA0 finished exit=$ec ===" | tee -a "$LOG"
  if (( ec != 0 )); then
    echo "$(date -Iseconds) chain abort after ETA0=$ETA0 (non-zero exit)" | tee -a "$LOG"
    exit "$ec"
  fi
done

echo "$(date -Iseconds) chain complete: all of [$ETA_CHAIN] reached TARGET=$TARGET" | tee -a "$LOG"
exit 0
