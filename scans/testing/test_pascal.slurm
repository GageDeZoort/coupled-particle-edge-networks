#!/bin/bash
# Compatibility redirect: this file moved to scans/pascal/test_pascal.slurm
# Resubmit with: sbatch scans/pascal/test_pascal.slurm "$@"
echo "ERROR: test_pascal.slurm moved to scans/pascal/test_pascal.slurm" >&2
echo "Resubmit: sbatch scans/pascal/test_pascal.slurm $*" >&2
exit 1
