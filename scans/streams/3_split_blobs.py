#!/usr/bin/env python
"""Stage 3: drop w>W Gaussians, sky-split blobs with N>8000.

    python scans/streams/3_split_blobs.py --split train --array-task 0
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cpen.apps.streams.preprocess.pipeline import run_split_blobs, run_stage_cli


if __name__ == "__main__":
    run_stage_cli(
        run_split_blobs,
        description="Width-cut GMM blobs and sky-split pieces with N>8000.",
    )
