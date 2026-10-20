#!/bin/bash
# Pull Lambda graph-R (panel b, k★=6) runs into Della SWEEP.
set -euo pipefail

HOST="${LAMBDA_HOST:-ubuntu@193.122.155.21}"
CM="${LAMBDA_CM:-$HOME/.cache/cm-lambda-193}"
RSH=(ssh -o BatchMode=yes -o ConnectTimeout=20)
if [[ -S "$CM" ]]; then
  RSH+=(-S "$CM")
fi
REMOTE_ROOT="${REMOTE_ROOT:-$HOST:~/runs/graph_R/jetclass_output/adam/sweep_lr/}"
LOCAL_SWEEP="${LOCAL_SWEEP:-/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/jetclass_output/adam/sweep_lr}"
GLOB="${GRAPH_R_GLOB:-*graph-R-b64-g8-s1000-k6*}"

mkdir -p "$LOCAL_SWEEP"
echo "rsync $REMOTE_ROOT → $LOCAL_SWEEP ($GLOB)"
rsync -aP -e "${RSH[*]}" \
  --include="$GLOB" --include="${GLOB}/**" --exclude='*' \
  "$REMOTE_ROOT" "$LOCAL_SWEEP/"
echo done
