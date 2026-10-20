#!/bin/bash
# Pull Lambda edge η₀×D runs into Della SWEEP + refresh Nature fig / edge CSV.
set -euo pipefail

HOST="${LAMBDA_HOST:-ubuntu@163.192.125.147}"
if [[ -z "${SSH_AUTH_SOCK:-}" ]]; then
  for s in /tmp/ssh-XXXXXXK9YkMy/agent.1667638 /tmp/ssh-agent-jg3607.sock; do
    [[ -S "$s" ]] || continue
    if SSH_AUTH_SOCK="$s" ssh-add -l >/dev/null 2>&1; then
      export SSH_AUTH_SOCK="$s"
      break
    fi
  done
fi
RSH="${RSYNC_RSH:-ssh -F /dev/null -o BatchMode=yes -o IdentitiesOnly=yes -o IdentityFile=$HOME/.ssh/id_ed25519}"
REMOTE_ROOT="${REMOTE_ROOT:-$HOST:~/runs/edge_eta/jetclass_output/adam/sweep_lr/}"
LOCAL_SWEEP="${LOCAL_SWEEP:-/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/jetclass_output/adam/sweep_lr}"
REPO="${REPO:-/home/jdezoort/coupled-particle-edge-networks}"
PYTHON="${PYTHON:-/home/jdezoort/.conda/envs/mamba-env/bin/python}"
# Match RUN_TAG_BASE from run_edge_eta_lambda8.sh (default R★=0.1).
GLOB="${EDGE_GLOB:-*edge-b96-g4-s4000-dh32*}"

mkdir -p "$LOCAL_SWEEP"
echo "rsync $REMOTE_ROOT → $LOCAL_SWEEP ($GLOB)"
rsync -aP -e "$RSH" \
  --include="$GLOB" --include="${GLOB}/**" --exclude='*' \
  "$REMOTE_ROOT" "$LOCAL_SWEEP/"

cd "$REPO"
"$PYTHON" scans/jets/plot_graph_construction_nature.py
# Keep the living figure notes in sync with CSV counts when present.
echo done
