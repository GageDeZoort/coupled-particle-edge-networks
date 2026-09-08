"""Write per-cell parquet files with rotated coordinates."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cpen.apps.streams.preprocess.geometry import rotate_to_patch_frame


def write_cell_parquets(df_out, centers, outdir, n_centers, verbose=True):
    """Write each cell as a parquet with phi, lam, pm_phi, pm_lam, star_id, cell_id."""
    cells_dir = Path(outdir) / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    df_out = df_out.copy()
    df_out["star_id"] = np.arange(len(df_out))
    for j in range(n_centers):
        col = f"in_cell_{j + 1}"
        sub_idx = np.where(df_out[col].to_numpy())[0]
        if len(sub_idx) == 0:
            continue
        sub = df_out.iloc[sub_idx].copy()
        ra0, dec0 = centers[j]
        phi, lam, pm_phi, pm_lam = rotate_to_patch_frame(
            sub["ra_des"].to_numpy(),
            sub["dec_des"].to_numpy(),
            sub["pmra_gaia"].to_numpy(),
            sub["pmdec_gaia"].to_numpy(),
            ra0,
            dec0,
        )
        sub["phi"] = phi
        sub["lam"] = lam
        sub["pm_phi"] = pm_phi
        sub["pm_lam"] = pm_lam
        sub["cell_id"] = j + 1
        sub.to_parquet(cells_dir / f"cell_{j + 1:04d}.parquet", index=False)
    if verbose:
        n_files = len(list(cells_dir.glob("cell_*.parquet")))
        print(f"\n  Wrote {n_files} cell parquets to {cells_dir}")
