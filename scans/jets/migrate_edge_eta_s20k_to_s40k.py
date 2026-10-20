#!/usr/bin/env python3
"""Rename panel-(c) edge η run dirs from 20k/5.12M → 40k/10.24M so Lightning resumes.

Must not race running jobs — scancel jc-edge-k7 first.
"""
from __future__ import annotations

import shutil
from pathlib import Path

SWEEP = Path(
    "/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/"
    "jetclass_output/adam/sweep_lr"
)
OLD = "ntr5120k_nval25k_ntst0_nf7_gstar0p1_stream-kin7-partint-p80-knn7-noloop-sk7-D5p12M-s20k-seed0-edge-b64-g4-s20000-dh32-k7-R0p1"
NEW = "ntr10240k_nval25k_ntst0_nf7_gstar0p1_stream-kin7-partint-p80-knn7-noloop-sk7-D10p24M-s40k-seed0-edge-b64-g4-s40000-dh32-k7-R0p1"


def main() -> None:
    moved = 0
    for p in sorted(SWEEP.glob("capen-llama-att_4_256_*")):
        name = p.name
        if OLD not in name:
            continue
        dest = SWEEP / name.replace(OLD, NEW)
        if dest.exists():
            print(f"skip exists: {dest.name}")
            continue
        print(f"mv {p.name}\n→  {dest.name}")
        shutil.move(str(p), str(dest))
        moved += 1
    print(f"migrated {moved} paths")


if __name__ == "__main__":
    main()
