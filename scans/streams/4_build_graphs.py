#!/usr/bin/env python
"""Stage 4: sky kNN + sky/PM hyperedges, one .npz graph per blob.

    python scans/streams/4_build_graphs.py --split train --array-task 0
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cpen.apps.streams.preprocess.pipeline import run_build_graphs, run_stage_cli


if __name__ == "__main__":
    run_stage_cli(run_build_graphs, description="Build kNN + hyperedge graphs for every blob.")
