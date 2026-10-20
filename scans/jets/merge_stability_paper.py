#!/usr/bin/env python3
"""Merge per-width paper probe shards and plot fig_stability_paper_{pool}*.

  python scans/jets/merge_stability_paper.py --pool uniform
  python scans/jets/merge_stability_paper.py --pool attn
  python scans/jets/merge_stability_paper.py --pool uniform --complete-only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "paper" / "data"
WIDTHS = (64, 128, 256, 384, 512, 768, 1024, 1536, 2048)
# Uniform: 6 channels × 6 η₀. Attn adds output_fused + output_logits → 8×6.
N_ETA = 6
COMPLETE_ROWS = {
    "uniform": N_ETA * 6,
    "attn": N_ETA * 8,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", choices=("uniform", "attn"), required=True)
    ap.add_argument("--plot", action="store_true", default=True)
    ap.add_argument("--no-plot", action="store_false", dest="plot")
    ap.add_argument(
        "--complete-only",
        action="store_true",
        help="Skip shards with fewer than the pool's complete agg-row count",
    )
    ap.add_argument(
        "--stem",
        type=str,
        default="fig_stability_paper",
        help="Figure stem passed to plot_stability_nature.py",
    )
    args = ap.parse_args()

    complete_rows = COMPLETE_ROWS[args.pool]
    stem = f"stability_one_step_{args.pool}_paper"
    shards: list[pd.DataFrame] = []
    missing: list[str] = []
    skipped: list[str] = []
    used_widths: list[int] = []
    for w in WIDTHS:
        p = DATA / f"{stem}_D{w}.csv"
        if not p.is_file():
            missing.append(p.name)
            continue
        df = pd.read_csv(p)
        if args.complete_only and len(df) < complete_rows:
            skipped.append(f"{p.name}({len(df)}<{complete_rows})")
            continue
        shards.append(df)
        used_widths.append(w)
    if missing:
        print(f"missing {len(missing)} shards: {missing}", file=sys.stderr)
    if skipped:
        print(f"skipped incomplete: {skipped}", file=sys.stderr)
    if not shards:
        print("no shards to merge", file=sys.stderr)
        return 1
    df = pd.concat(shards, ignore_index=True)
    out = DATA / f"{stem}.csv"
    trial_parts: list[pd.DataFrame] = []
    for w in used_widths:
        tp = DATA / f"{stem}_D{w}_trials.csv"
        if tp.is_file():
            trial_parts.append(pd.read_csv(tp))
    if trial_parts:
        trials = pd.concat(trial_parts, ignore_index=True)
        trials_out = DATA / f"{stem}_trials.csv"
        trials.to_csv(trials_out, index=False)
        print(f"wrote {trials_out} ({len(trials)} trials)")
    df.to_csv(out, index=False)
    print(
        f"wrote {out} ({len(df)} agg rows; widths={sorted(df.width.unique().tolist())})"
    )

    if args.plot:
        cmd = [
            sys.executable,
            str(REPO / "scans/jets/plot_stability_nature.py"),
            "--csv",
            str(out),
            "--pool",
            args.pool,
            "--stem",
            args.stem,
        ]
        print("+", " ".join(cmd), flush=True)
        return subprocess.call(cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
