#!/usr/bin/env python
"""Shim — real script moved to 'scans/pascal/download_pascal_graphs.py'."""
from __future__ import annotations
import runpy
import sys
from pathlib import Path
TARGET = Path(__file__).resolve().parents[1] / "pascal/download_pascal_graphs.py"
sys.argv[0] = str(TARGET)
runpy.run_path(str(TARGET), run_name="__main__")
