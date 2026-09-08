#!/bin/bash
# Compatibility redirect: this file moved to scans/common/profile.slurm
# Resubmit with: sbatch scans/common/profile.slurm "$@"
echo "ERROR: profile.slurm moved to scans/common/profile.slurm" >&2
echo "Resubmit: sbatch scans/common/profile.slurm $*" >&2
exit 1
