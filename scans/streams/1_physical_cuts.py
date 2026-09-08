#!/usr/bin/env python
"""Stage 1: frozen CMD slab + π<1 mas on each DES cell.

Knobs were fit on train galaxy 0000 and are not refit.

    python scans/streams/1_physical_cuts.py --split train --array-task 0
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cpen.apps.streams.preprocess.pipeline import run_physical_cuts, run_stage_cli


if __name__ == "__main__":
    run_stage_cli(run_physical_cuts, description="Apply frozen CMD slab + parallax cuts per cell.")
