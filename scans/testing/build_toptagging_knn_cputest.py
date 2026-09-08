#!/usr/bin/env python
"""Shim — real script moved to 'scans/jets/build_toptagging_knn_cputest.py'."""
from __future__ import annotations
import runpy
import sys
from pathlib import Path
TARGET = Path(__file__).resolve().parents[1] / "jets/build_toptagging_knn_cputest.py"
sys.argv[0] = str(TARGET)
runpy.run_path(str(TARGET), run_name="__main__")
