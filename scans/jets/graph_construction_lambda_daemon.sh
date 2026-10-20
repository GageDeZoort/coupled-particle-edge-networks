#!/bin/bash
set -u
cd /home/jdezoort/coupled-particle-edge-networks
LOG=/home/jdezoort/.cursor/projects/home-jdezoort/terminals/graph_construction_lambda_loop.log
CM="$HOME/.cache/cm-lambda-193"
HOST=ubuntu@193.122.155.21
if [[ ! -S "$CM" ]]; then
  for a in /tmp/ssh-*/agent.*; do
    [[ -S "$a" ]] || continue
    export SSH_AUTH_SOCK="$a"
    if ssh-add -l >/dev/null 2>&1; then
      ssh -o ControlMaster=yes -o ControlPersist=8h -o ControlPath="$CM" -o BatchMode=yes -fN "$HOST" 2>/dev/null || true
      break
    fi
  done
fi
{
  while true; do
    echo "AGENT_LOOP_TICK_graph_construction $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    python scans/jets/update_graph_k_lambda_live_canvas.py || echo UPDATE_FAILED $?
    bash scans/jets/pull_graph_R_lambda.sh || echo PULL_R_FAILED $?
    DONE=$(python3 -c "import json; d=json.load(open('paper/data/graph_k_R0p1_lambda_live.json')); print(d.get('panel_b',{}).get('n_done',0))" 2>/dev/null || echo 0)
    echo "panel_b_done=${DONE}/30"
    if [[ "${DONE}" -ge 30 ]]; then
      echo "PANEL_B_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      python scans/jets/plot_graph_construction_nature.py || true
      break
    fi
    sleep 180
  done
} >>"$LOG" 2>&1
