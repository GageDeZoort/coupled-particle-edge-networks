"""
Apply GMM, parallax, and CMD ridgeline cuts per cell. Self-contained preprocessing.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import gaussian_filter1d

from cpen.apps.streams.preprocess.des_footprint_centers import DES_FOOTPRINT_CENTERS
from cpen.apps.streams.preprocess.galaxy import ensure_is_mock_stream
from cpen.apps.streams.preprocess.geometry import rotate_to_patch_frame as _rotate_to_patch_frame
from cpen.apps.streams.preprocess.gmm import GMM_N_COMPONENTS, fit_pm_gmm

CKPT_DIR = Path("/scratch/gpfs/BHANIN/jgdezoort/streams")
GMM_WIDTH_THRESHOLD = 3.0
PARALLAX_THRESHOLD = 0.5
CMD_DISTANCE_THRESHOLD = 0.65
MIN_MOCK_STARS = 100
MIN_TRAIN_STARS = MIN_MOCK_STARS  # backward-compatible alias



def _gmm_pm_cell(pm_phi, pm_lam, n_components=None, n_init=3, random_state=0, max_fit_samples=50_000):
    """Legacy wrapper: widths + labels aligned to the *finite*-PM subset only."""
    if n_components is None:
        n_components = GMM_N_COMPONENTS
    fit = fit_pm_gmm(
        pm_phi, pm_lam,
        n_components=n_components,
        n_init=n_init,
        random_state=random_state,
        max_fit_samples=max_fit_samples,
    )
    if fit.gmm is None:
        return None, None
    finite = fit.labels >= 0
    return fit.widths, fit.labels[finite]


def _extract_cmd_ridgeline(r, g_r, n_r_bins=40, n_color_bins=80, smooth_sigma=2):
    m = np.isfinite(r) & np.isfinite(g_r)
    r, g_r = r[m], g_r[m]
    if len(r) < 20:
        return np.array([]), np.array([])
    r_range = (np.percentile(r, 1), np.percentile(r, 99))
    color_range = (np.percentile(g_r, 1), np.percentile(g_r, 99))
    r_edges = np.linspace(r_range[0], r_range[1], n_r_bins + 1)
    color_edges = np.linspace(color_range[0], color_range[1], n_color_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    modes = []
    for i in range(n_r_bins):
        in_bin = (r >= r_edges[i]) & (r < r_edges[i + 1])
        if in_bin.sum() < 5:
            modes.append(np.nan)
            continue
        hist, _ = np.histogram(g_r[in_bin], bins=color_edges)
        imode = np.argmax(hist)
        mode_val = 0.5 * (color_edges[imode] + color_edges[imode + 1])
        modes.append(mode_val)
    modes = np.array(modes)
    valid = np.isfinite(modes)
    if valid.sum() < 3:
        return r_centers, modes
    modes_smooth = gaussian_filter1d(
        np.where(valid, modes, np.nanmean(modes[valid])), sigma=smooth_sigma, mode="nearest"
    )
    return r_centers, modes_smooth


def _distance_to_ridgeline(r, g_r, r_ridge, g_r_ridge):
    if len(r_ridge) < 2:
        return np.full_like(r, np.nan)
    g_r_interp = np.interp(r, r_ridge, g_r_ridge)
    return np.abs(g_r - g_r_interp)


def apply_three_cuts(df, centers, verbose=True):
    """
    Apply GMM, parallax, and CMD cuts per cell.
    Returns (df_out, report_rows) where df_out has phi, lam, pm_phi, pm_lam and passes_cuts.
    """
    n_centers = len(centers)
    cell_cols = [f"in_cell_{j + 1}" for j in range(n_centers)]

    df = df.copy()
    df["phi"] = np.nan
    df["lam"] = np.nan
    df["pm_phi"] = np.nan
    df["pm_lam"] = np.nan
    df["passes_cuts"] = False
    seen_for_coords = set()

    report_rows = []

    for j in range(n_centers):
        col = cell_cols[j]
        if col not in df.columns:
            continue
        sub_idx = np.where(df[col].to_numpy())[0]
        if len(sub_idx) == 0:
            continue

        sub = ensure_is_mock_stream(df.loc[sub_idx].copy())
        is_mock = sub["is_mock_stream"].to_numpy(bool)
        n_mock_before = int(is_mock.sum())
        if n_mock_before < MIN_MOCK_STARS:
            continue

        is_bg = ~is_mock
        n_bg_before = int(is_bg.sum())
        purity_before = (
            n_mock_before / (n_mock_before + n_bg_before)
            if (n_mock_before + n_bg_before) > 0
            else 0
        )
        ra0, dec0 = centers[j]
        phi, lam, pm_phi, pm_lam = _rotate_to_patch_frame(
            sub["ra_des"].to_numpy(),
            sub["dec_des"].to_numpy(),
            sub["pmra_gaia"].to_numpy(),
            sub["pmdec_gaia"].to_numpy(),
            ra0, dec0,
        )

        for ii, idx in enumerate(sub_idx):
            if idx not in seen_for_coords:
                df.loc[df.index[idx], "phi"] = phi[ii]
                df.loc[df.index[idx], "lam"] = lam[ii]
                df.loc[df.index[idx], "pm_phi"] = pm_phi[ii]
                df.loc[df.index[idx], "pm_lam"] = pm_lam[ii]
                seen_for_coords.add(idx)

        fit = fit_pm_gmm(pm_phi, pm_lam, n_components=GMM_N_COMPONENTS)
        if fit.gmm is None:
            continue
        mask_gmm = np.zeros(len(sub), dtype=bool)
        ok = fit.labels >= 0
        mask_gmm[ok] = fit.widths[fit.labels[ok]] <= GMM_WIDTH_THRESHOLD

        plx = sub["parallax_gaia"].to_numpy()
        mask_plx = (~np.isfinite(plx)) | (plx < PARALLAX_THRESHOLD)

        r = sub["psf_mag_aper_8_r_corrected_des"].to_numpy()
        g = sub["psf_mag_aper_8_g_corrected_des"].to_numpy()
        g_r = g - r
        m_cmd = np.isfinite(r) & np.isfinite(g_r) & (r >= 14) & (r <= 22) & (g_r >= -1) & (g_r <= 3)
        r_ridge, g_r_ridge = _extract_cmd_ridgeline(r[m_cmd], g_r[m_cmd])
        dist = _distance_to_ridgeline(r, g_r, r_ridge, g_r_ridge)
        mask_cmd = (dist <= CMD_DISTANCE_THRESHOLD) | ~m_cmd

        mask_pass = mask_gmm & mask_plx & mask_cmd

        n_mock_after = int((is_mock & mask_pass).sum())
        n_bg_after = int((is_bg & mask_pass).sum())
        purity_after = (
            n_mock_after / (n_mock_after + n_bg_after)
            if (n_mock_after + n_bg_after) > 0
            else 0
        )

        frac_signal_lost = (
            (n_mock_before - n_mock_after) / n_mock_before if n_mock_before > 0 else 0
        )
        frac_bg_lost = (n_bg_before - n_bg_after) / n_bg_before if n_bg_before > 0 else 0

        for ii, idx in enumerate(sub_idx):
            if mask_pass[ii]:
                df.loc[df.index[idx], "passes_cuts"] = True

        report_rows.append({
            "cell_id": j + 1,
            "n_signal_before": n_mock_before,
            "n_bg_before": n_bg_before,
            "n_signal_after": n_mock_after,
            "n_bg_after": n_bg_after,
            "purity_before": purity_before,
            "purity_after": purity_after,
            "frac_signal_lost": frac_signal_lost,
            "frac_bg_lost": frac_bg_lost,
        })

        if verbose:
            print(
                f"  Cell {j+1:3d}: mock {n_mock_before:,}->{n_mock_after:,} "
                f"(lost {frac_signal_lost*100:.1f}%)  |  bg {n_bg_before:,}->{n_bg_after:,} "
                f"(lost {frac_bg_lost*100:.1f}%)  |  purity {purity_before:.4f}->{purity_after:.4f}"
            )

    return df, pd.DataFrame(report_rows)


def main():
    df = pd.read_parquet(CKPT_DIR / "0000_checkpoint.parquet")
    meta = yaml.safe_load((CKPT_DIR / "0000_meta.yaml").read_text()) if (CKPT_DIR / "0000_meta.yaml").exists() else {}
    centers = DES_FOOTPRINT_CENTERS

    print("=" * 72)
    print("Apply 3 cuts: GMM width≤3.0, parallax<0.5, CMD dist≤0.65")
    print("=" * 72)
    print(f"Cells with ≥{MIN_MOCK_STARS} mock-stream stars:")
    df_out, report = apply_three_cuts(df, centers, verbose=True)

    if len(report) == 0:
        print("No cells processed.")
        return

    print("\n" + "=" * 72)
    print(f"AGGREGATE (all cells with ≥{MIN_MOCK_STARS} mock-stream stars)")
    print("=" * 72)
    tot_sig_before = report["n_signal_before"].sum()
    tot_bg_before = report["n_bg_before"].sum()
    tot_sig_after = report["n_signal_after"].sum()
    tot_bg_after = report["n_bg_after"].sum()
    agg_frac_sig_lost = (tot_sig_before - tot_sig_after) / tot_sig_before if tot_sig_before > 0 else 0
    agg_frac_bg_lost = (tot_bg_before - tot_bg_after) / tot_bg_before if tot_bg_before > 0 else 0
    purity_before = tot_sig_before / (tot_sig_before + tot_bg_before) if (tot_sig_before + tot_bg_before) > 0 else 0
    purity_after = tot_sig_after / (tot_sig_after + tot_bg_after) if (tot_sig_after + tot_bg_after) > 0 else 0

    print(f"  Mock stream: {tot_sig_before:,} -> {tot_sig_after:,}  (lost {agg_frac_sig_lost*100:.1f}%)")
    print(f"  Background:  {tot_bg_before:,} -> {tot_bg_after:,}  (lost {agg_frac_bg_lost*100:.1f}%)")
    print(f"  Purity:      {purity_before:.4f} -> {purity_after:.4f}")
    print("=" * 72)

    return df_out, report


if __name__ == "__main__":
    df_out, report = main()
