#!/bin/bash
# Compatibility redirect: this file moved to scans/pascal/build_pascal_graphs.slurm
# Resubmit with: sbatch scans/pascal/build_pascal_graphs.slurm "$@"
echo "ERROR: build_pascal_graphs.slurm moved to scans/pascal/build_pascal_graphs.slurm" >&2
echo "Resubmit: sbatch scans/pascal/build_pascal_graphs.slurm $*" >&2
exit 1
