#!/bin/bash
# Compatibility redirect: this file moved to scans/common/test.slurm
# Resubmit with: sbatch scans/common/test.slurm "$@"
echo "ERROR: test.slurm moved to scans/common/test.slurm" >&2
echo "Resubmit: sbatch scans/common/test.slurm $*" >&2
exit 1
