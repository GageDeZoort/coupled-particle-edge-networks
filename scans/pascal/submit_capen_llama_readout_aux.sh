#!/bin/bash
# Submit the CAPEN-Llama PascalVOC-SP readout-asymmetry grid (does not wait).
#
# Same knobs as scans/pascal/submit_readout_aux.sh, plus HEADS (must divide D).
#
# From repo root:
#   bash scans/pascal/submit_capen_llama_readout_aux.sh
#   DRY_RUN=1 bash scans/pascal/submit_capen_llama_readout_aux.sh
#   WIDTHS=256 ETAS=0.25 READOUTS=node,node+edge SEEDS=0 DEPTH=4 EPOCHS=100 \
#     BATCH_SIZE=128 bash scans/pascal/submit_capen_llama_readout_aux.sh
#
# Positionals on the slurm script (readout, width, eta0, seed) are the source of
# truth; --export is extra (Della sometimes drops --export=ALL).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/test_capen_llama_readout_aux.slurm"

WIDTHS="${WIDTHS:-256,512}"
ETAS="${ETAS:-0.05,0.25,1.0}"
READOUTS="${READOUTS:-node,node+edge}"
SEEDS="${SEEDS:-0}"
EPOCHS="${EPOCHS:-100}"
DEPTH="${DEPTH:-4}"
HEADS="${HEADS:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EDGE_LOSS_WEIGHT="${EDGE_LOSS_WEIGHT:-1.0}"
IDENTITY_M22="${IDENTITY_M22:-0}"
INCIDENCE_M22="${INCIDENCE_M22:-1}"
DRY_RUN="${DRY_RUN:-0}"

IFS=',' read -r -a WIDTH_ARR <<< "${WIDTHS}"
IFS=',' read -r -a ETA_ARR <<< "${ETAS}"
IFS=',' read -r -a READOUT_ARR <<< "${READOUTS}"
IFS=',' read -r -a SEED_ARR <<< "${SEEDS}"

n_jobs=$(( ${#WIDTH_ARR[@]} * ${#ETA_ARR[@]} * ${#READOUT_ARR[@]} * ${#SEED_ARR[@]} ))

echo "Pascal CAPEN-Llama readout-aux grid"
echo "  widths=${WIDTHS}"
echo "  etas=${ETAS}"
echo "  readouts=${READOUTS}"
echo "  seeds=${SEEDS}"
echo "  epochs=${EPOCHS} depth=${DEPTH} heads=${HEADS} bs=${BATCH_SIZE} lambda=${EDGE_LOSS_WEIGHT} identity_m22=${IDENTITY_M22} incidence_m22=${INCIDENCE_M22}"
echo "  n_jobs=${n_jobs}"
echo

node_ids=()
edge_ids=()

for mode in "${READOUT_ARR[@]}"; do
  mode="${mode// /}"
  for width in "${WIDTH_ARR[@]}"; do
    width="${width// /}"
    for eta in "${ETA_ARR[@]}"; do
      eta="${eta// /}"
      eta_tag="${eta//./p}"
      for seed in "${SEED_ARR[@]}"; do
        seed="${seed// /}"
        if [[ "${mode}" == "node+edge" ]]; then
          jname="llama-voc-eb-d${width}-e${eta_tag}-s${seed}"
        else
          jname="llama-voc-node-d${width}-e${eta_tag}-s${seed}"
        fi
        export_vars="ALL,READOUT_MODE=${mode},WIDTH=${width},ETAS=${eta},SEED=${seed},EPOCHS=${EPOCHS},DEPTH=${DEPTH},HEADS=${HEADS},BATCH_SIZE=${BATCH_SIZE},EDGE_LOSS_WEIGHT=${EDGE_LOSS_WEIGHT},IDENTITY_M22=${IDENTITY_M22},INCIDENCE_M22=${INCIDENCE_M22}"
        if [[ "${DRY_RUN}" == "1" ]]; then
          echo "DRY  ${jname}  ${mode}  D=${width}  eta0=${eta}  seed=${seed}"
          jid="dry"
        else
          jid=$(sbatch --parsable \
            --job-name="${jname}" \
            --export="${export_vars}" \
            "${SLURM_SCRIPT}" "${mode}" "${width}" "${eta}" "${seed}")
          echo "  ${jname} -> ${jid}"
        fi
        if [[ "${mode}" == "node+edge" ]]; then
          edge_ids+=("${jid}")
        else
          node_ids+=("${jid}")
        fi
      done
    done
  done
done

echo
echo "NODE_ONLY = ${node_ids[*]}"
echo "NODE_PLUS_BOUNDARY = ${edge_ids[*]}"
echo "Track with: squeue -u \$USER"
