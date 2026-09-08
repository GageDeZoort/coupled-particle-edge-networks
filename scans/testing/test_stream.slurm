#!/bin/bash
# Compatibility redirect: this file moved to scans/streams/test_stream.slurm
# Resubmit with: sbatch scans/streams/test_stream.slurm "$@"
echo "ERROR: test_stream.slurm moved to scans/streams/test_stream.slurm" >&2
echo "Resubmit: sbatch scans/streams/test_stream.slurm $*" >&2
exit 1
