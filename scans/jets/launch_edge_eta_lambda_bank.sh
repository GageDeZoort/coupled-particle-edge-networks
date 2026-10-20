#!/bin/bash
# Launch one bank of the edge η×D sweep on a Lambda multi-GPU host.
#
# Same training recipe as run_edge_eta_lambda8.sh (B_gpu=96, N_GPUS=4 →
# B_global=384 by default). Only the GPU set, NCCL port, and TASKS list change.
#
# Examples (run on the Lambda host, from ~/cpen):
#   # GPUs 0–3, first cell (D=256, η₀=0.05)
#   bash scans/jets/launch_edge_eta_lambda_bank.sh 0,1,2,3 29510 0
#
#   # GPUs 4–7, second cell (D=256, η₀=0.1)
#   bash scans/jets/launch_edge_eta_lambda_bank.sh 4,5,6,7 29511 1
#
#   # Continue remaining cells on freed GPUs 0–3
#   bash scans/jets/launch_edge_eta_lambda_bank.sh 0,1,2,3 29510 "2 3 4 5 6 7 8 9 10 11 12 13 14"
#
# Flat TASK index: width_idx*5 + eta_idx
#   WIDTHS=(256 384 512)  ETA0=(0.05 0.1 0.25 0.5 1.0)
#
set -euo pipefail

GPUS="${1:?usage: $0 CUDA_VISIBLE_DEVICES MASTER_PORT TASKS [LOG_TAG]}"
PORT="${2:?usage: $0 CUDA_VISIBLE_DEVICES MASTER_PORT TASKS [LOG_TAG]}"
TASKS="${3:?usage: $0 CUDA_VISIBLE_DEVICES MASTER_PORT TASKS [LOG_TAG]}"
TAG="${4:-bank}"

REPO="${REPO:-$HOME/cpen}"
LOGDIR="${LOG_DIR:-$HOME/logs/edge_eta}"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/master_${TAG}.log"

cd "$REPO"
echo "$(date -Iseconds) LAUNCH bank tag=$TAG gpus=$GPUS port=$PORT TASKS=$TASKS" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="$GPUS" MASTER_PORT="$PORT" TASKS="$TASKS" \
  nohup bash scans/jets/run_edge_eta_lambda8.sh >>"$LOG" 2>&1 &
echo "PID=$! log=$LOG"
