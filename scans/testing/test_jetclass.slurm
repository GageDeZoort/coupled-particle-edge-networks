#!/bin/bash
# Compatibility redirect: this file moved to scans/jets/test_jetclass.slurm
# Resubmit with: sbatch scans/jets/test_jetclass.slurm "$@"
echo "ERROR: test_jetclass.slurm moved to scans/jets/test_jetclass.slurm" >&2
echo "Resubmit: sbatch scans/jets/test_jetclass.slurm $*" >&2
exit 1
