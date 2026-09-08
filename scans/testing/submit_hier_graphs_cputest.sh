#!/bin/bash
# Shim — real script moved to scans/jets/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../jets/submit_hier_graphs_cputest.sh" "$@"
