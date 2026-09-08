"""Lightweight parquet run filters for hyperparameter transfer analysis.

For Slurm-job-driven LR / width transfer plots used by the notebooks, see
:mod:`cpen.utils.transfer_plots`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_parquet_runs(root: Path, pattern: str = "**/*.parquet") -> pd.DataFrame:
    """Load and concatenate parquet run logs under *root*."""
    paths = sorted(root.glob(pattern))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def filter_runs(
    df: pd.DataFrame,
    *,
    eta_0: float | None = None,
    depth: int | None = None,
    width: int | None = None,
) -> pd.DataFrame:
    """Filter logged runs by static hyperparameters."""
    out = df
    if eta_0 is not None:
        out = out[out["eta_0"] == eta_0]
    if depth is not None:
        out = out[out["depth"] == depth]
    if width is not None:
        out = out[out["width"] == width]
    return out


def final_epoch_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per run at the highest logged epoch."""
    if df.empty:
        return df
    idx = df.groupby(["run_dir"], dropna=False)["epoch"].idxmax()
    return df.loc[idx].reset_index(drop=True)
