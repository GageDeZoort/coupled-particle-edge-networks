#!/bin/bash
# Submit the MNISTSuperpixels graph-readout grid (does not wait).
#
#   D ∈ {128,256,512}  ×  η0 ∈ {0.05,0.25,1.0}
#     ×  {node, node+edge}  ×  seed ∈ {0}
#
# Attn is opt-in: READOUTS=attn bash scans/mnist/submit_readout.sh
#
#   bash scans/mnist/submit_readout.sh
#   DRY_RUN=1 bash scans/mnist/submit_readout.sh
#   READOUTS=node WIDTHS=256 ETAS=0.25 SEEDS=0 bash scans/mnist/submit_readout.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/test_cpen_readout.slurm"

WIDTHS="${WIDTHS:-128,256,512}"
ETAS="${ETAS:-0.05,0.25,1.0}"
READOUTS="${READOUTS:-node,node+edge}"
SEEDS="${SEEDS:-0}"
EPOCHS="${EPOCHS:-100}"
DEPTH="${DEPTH:-4}"
BATCH_SIZE="${BATCH_SIZE:-128}"
DRY_RUN="${DRY_RUN:-0}"

IFS=',' read -r -a WIDTH_ARR <<< "${WIDTHS}"
IFS=',' read -r -a ETA_ARR <<< "${ETAS}"
IFS=',' read -r -a READOUT_ARR <<< "${READOUTS}"
IFS=',' read -r -a SEED_ARR <<< "${SEEDS}"

n_jobs=$(( ${#WIDTH_ARR[@]} * ${#ETA_ARR[@]} * ${#READOUT_ARR[@]} * ${#SEED_ARR[@]} ))

echo "MNIST Superpixels readout grid"
echo "  widths=${WIDTHS}"
echo "  etas=${ETAS}"
echo "  readouts=${READOUTS}"
echo "  seeds=${SEEDS}"
echo "  epochs=${EPOCHS} depth=${DEPTH} bs=${BATCH_SIZE}"
echo "  n_jobs=${n_jobs}"
echo

for mode in "${READOUT_ARR[@]}"; do
  mode="${mode// /}"
  for width in "${WIDTH_ARR[@]}"; do
    width="${width// /}"
    for eta in "${ETA_ARR[@]}"; do
      eta="${eta// /}"
      eta_tag="${eta//./p}"
      for seed in "${SEED_ARR[@]}"; do
        seed="${seed// /}"
        jname="mnist-${mode}-d${width}-e${eta_tag}-s${seed}"
        export_vars="ALL,READOUT_MODE=${mode},WIDTH=${width},ETAS=${eta},SEED=${seed},EPOCHS=${EPOCHS},DEPTH=${DEPTH},BATCH_SIZE=${BATCH_SIZE}"
        if [[ "${DRY_RUN}" == "1" ]]; then
          echo "DRY  ${jname}  ${mode}  D=${width}  eta0=${eta}  seed=${seed}"
        else
          jid=$(sbatch --parsable \
            --job-name="${jname}" \
            --export="${export_vars}" \
            "${SLURM_SCRIPT}" "${mode}" "${width}" "${eta}" "${seed}")
          echo "  ${jname} -> ${jid}"
        fi
      done
    done
  done
done
echo "Track with: squeue -u \$USER"
