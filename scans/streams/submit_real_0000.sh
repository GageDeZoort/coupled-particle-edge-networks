#!/bin/bash
# Submit the mock-less train/0000 blob pipeline (stages 0→4).
#
#   bash scans/streams/submit_real_0000.sh
#   REBUILD=1 bash scans/streams/submit_real_0000.sh
set -euo pipefail

REPO_ROOT="/home/jdezoort/coupled-particle-edge-networks"
cd "${REPO_ROOT}"

EXPORT="ALL"
if [[ "${REBUILD:-0}" == "1" ]]; then
  EXPORT="${EXPORT},REBUILD=1"
fi
if [[ -n "${OUT_ROOT:-}" ]]; then
  EXPORT="${EXPORT},OUT_ROOT=${OUT_ROOT}"
fi

j=$(sbatch --parsable --export="${EXPORT}" scans/streams/real_galaxy_0000.slurm)
echo "real train/0000  ${j}"
echo "  out  /scratch/gpfs/BHANIN/jgdezoort/streams/galaxies_real/train/0000"
echo "  log  /scratch/gpfs/BHANIN/jdezoort/cpen_runs/streams/real0000_${j}.out"
