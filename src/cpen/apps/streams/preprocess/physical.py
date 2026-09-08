"""Label-free CMD slab + parallax cuts (knobs frozen on train galaxy 0000)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cpen.apps.streams.preprocess.config import (
    CMD_G_COL,
    CMD_PARALLAX_COL,
    CMD_R_COL,
    CMD_SLAB_B,
    CMD_SLAB_HALF_WIDTH,
    CMD_SLAB_M,
    PI_MAX,
)
from cpen.apps.streams.preprocess.galaxy import ensure_is_mock_stream


def cmd_slab_signed_distance(g_r, r, m: float = CMD_SLAB_M, b: float = CMD_SLAB_B):
    """Orthogonal signed distance to ``r = m(g-r) + b``."""
    return (np.asarray(r, dtype=float) - float(m) * np.asarray(g_r, dtype=float) - float(b)) / np.sqrt(
        1.0 + float(m) * float(m)
    )


def physical_keep_mask(
    df: pd.DataFrame,
    *,
    m: float = CMD_SLAB_M,
    b: float = CMD_SLAB_B,
    half_width: float = CMD_SLAB_HALF_WIDTH,
    pi_max: float = PI_MAX,
    g_col: str = CMD_G_COL,
    r_col: str = CMD_R_COL,
    parallax_col: str = CMD_PARALLAX_COL,
) -> np.ndarray:
    """Keep stars in the frozen CMD slab with ``π < pi_max`` (missing π kept)."""
    g = df[g_col].to_numpy(float)
    r = df[r_col].to_numpy(float)
    g_r = g - r
    dist = np.abs(cmd_slab_signed_distance(g_r, r, m=m, b=b))
    cmd = np.isfinite(g_r) & np.isfinite(r) & (dist <= float(half_width))
    if parallax_col in df.columns:
        plx = df[parallax_col].to_numpy(float)
        pi_ok = (~np.isfinite(plx)) | (plx < float(pi_max))
    else:
        pi_ok = np.ones(len(df), dtype=bool)
    return cmd & pi_ok


def apply_physical_cuts(
    df: pd.DataFrame,
    *,
    verbose: bool = True,
    positive_name: str = "mocks",
    **kwargs,
) -> pd.DataFrame:
    """Filter one table by the frozen CMD slab + parallax cut."""
    df = ensure_is_mock_stream(df)
    keep = physical_keep_mask(df, **kwargs)
    out = df.loc[keep].copy()
    if verbose:
        n0, n1 = len(df), len(out)
        n_m0 = int(df["is_mock_stream"].sum())
        n_m1 = int(out["is_mock_stream"].sum()) if n1 else 0
        print(
            f"Physical cuts (CMD slab + π<{kwargs.get('pi_max', PI_MAX):g} mas): "
            f"{n0:,} → {n1:,} stars  "
            f"{positive_name} {n_m0:,} → {n_m1:,}"
        )
    return out


def apply_physical_cuts_to_cells(
    cell_dfs: dict,
    *,
    verbose: bool = True,
    positive_name: str = "mocks",
    **kwargs,
) -> dict:
    """Apply ``apply_physical_cuts`` to every cell (no per-cell refit)."""
    out = {}
    n0 = n1 = m0 = m1 = 0
    for cid, df in sorted(cell_dfs.items(), key=lambda kv: int(kv[0])):
        df = ensure_is_mock_stream(df)
        n0 += len(df)
        m0 += int(df["is_mock_stream"].sum())
        sub = apply_physical_cuts(df, verbose=False, **kwargs)
        n1 += len(sub)
        m1 += int(sub["is_mock_stream"].sum()) if len(sub) else 0
        if len(sub):
            out[int(cid)] = sub
    if verbose:
        print(
            f"Physical cuts on {len(cell_dfs)} cells → {len(out)} nonempty  "
            f"rows {n0:,} → {n1:,}  {positive_name} {m0:,} → {m1:,}"
        )
    return out
