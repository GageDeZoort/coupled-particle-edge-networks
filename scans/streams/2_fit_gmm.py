#!/usr/bin/env python
"""Stage 2: k=64 GMM in proper motion on physically cut cells.

    python scans/streams/2_fit_gmm.py --split train --array-task 0
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cpen.apps.streams.preprocess.pipeline import run_gmm, run_stage_cli


if __name__ == "__main__":
    run_stage_cli(run_gmm, description="Fit k=64 PM GMM on cells after physical cuts.")
