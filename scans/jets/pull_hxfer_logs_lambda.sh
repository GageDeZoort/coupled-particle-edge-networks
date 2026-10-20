#!/bin/bash
# Pull hxfer stdout logs (attn + uniform) from Lambda into organized scratch + paper.
set -euo pipefail

HOST="${LAMBDA_HOST:-ubuntu@193.122.155.21}"
CM="${LAMBDA_CM:-$HOME/.cache/cm-lambda-193}"
RSH=(ssh -o BatchMode=yes -o ConnectTimeout=20)
if [[ -S "$CM" ]]; then
  RSH+=(-S "$CM")
fi

ORG=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/hxfer_constlr_v2stream
PAPER=/home/jdezoort/coupled-particle-edge-networks/paper/data/hxfer_constlr_v2stream
mkdir -p "$ORG"/{upool,apool}/{logs,runs,parquets,checkpoints} \
         "$PAPER"/{upool,apool}/{logs,summaries}

echo "==== logs ===="
rsync -aP -e "${RSH[*]}" "$HOST:~/logs/hxfer_upool_eta/" "$ORG/upool/logs/"
rsync -aP -e "${RSH[*]}" "$HOST:~/logs/hxfer_apool_eta/" "$ORG/apool/logs/"
# paper-side copies of masters + cell outs (small)
rsync -aP -e "${RSH[*]}" "$HOST:~/logs/hxfer_upool_eta/" "$PAPER/upool/logs/"
rsync -aP -e "${RSH[*]}" \
  --exclude='archive_*' --exclude='stopped_*' \
  "$HOST:~/logs/hxfer_apool_eta/" "$PAPER/apool/logs/"
echo done_logs
