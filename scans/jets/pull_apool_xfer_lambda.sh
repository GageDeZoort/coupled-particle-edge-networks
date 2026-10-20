#!/bin/bash
# Pull finished Lambda attentional-pool xfer validate runs (+ logs) to Della.
set -euo pipefail

HOST="${LAMBDA_HOST:-ubuntu@193.122.155.21}"
CM="${LAMBDA_CM:-$HOME/.cache/cm-lambda-193}"
RSH=(ssh -o BatchMode=yes -o ConnectTimeout=20)
if [[ -S "$CM" ]]; then
  RSH+=(-S "$CM")
fi

REMOTE_RUNS="${REMOTE_RUNS:-$HOST:~/runs/apool_xfer/jetclass_output/adam/sweep_lr/}"
REMOTE_LOGS="${REMOTE_LOGS:-$HOST:~/logs/apool_xfer/}"
LOCAL_SWEEP="${LOCAL_SWEEP:-/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/jetclass_output/adam/sweep_lr}"
LOCAL_LOGS="${LOCAL_LOGS:-/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/apool_xfer_lambda}"
GLOB="${APOOL_GLOB:-*apool-xfer-b48-g8-s1000*}"

mkdir -p "$LOCAL_SWEEP" "$LOCAL_LOGS"
echo "rsync runs $REMOTE_RUNS → $LOCAL_SWEEP ($GLOB)"
rsync -aP -e "${RSH[*]}" \
  --include="$GLOB" --include="${GLOB}/**" --exclude='*' \
  "$REMOTE_RUNS" "$LOCAL_SWEEP/"
echo "rsync logs $REMOTE_LOGS → $LOCAL_LOGS"
rsync -aP -e "${RSH[*]}" "$REMOTE_LOGS" "$LOCAL_LOGS/"
echo done
