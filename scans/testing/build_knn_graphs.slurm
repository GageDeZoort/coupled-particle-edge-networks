#!/bin/bash
# Compatibility redirect: this file moved to scans/jets/build_knn_graphs.slurm
# Resubmit with: sbatch scans/jets/build_knn_graphs.slurm "$@"
echo "ERROR: build_knn_graphs.slurm moved to scans/jets/build_knn_graphs.slurm" >&2
echo "Resubmit: sbatch scans/jets/build_knn_graphs.slurm $*" >&2
exit 1
