"""
Sky and diagnostic plotting utilities.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cpen.apps.streams.preprocess.geometry import radec_to_xyz as _radec_to_xyz, angular_distance_rad as _angular_distance_rad
from cpen.apps.streams.preprocess.galaxy import ensure_is_mock_stream


# Distinct colors for per-stream train stars (cycled if many streams)
_STREAM_COLORS = [
    "#d96c4a", "#2e7d32", "#6a1b9a", "#0277bd", "#ef6c00",
    "#c2185b", "#00796b", "#5d4037", "#7b1fa2", "#00838f",
    "#bf360c", "#4a148c",
]
# Hyperedge QC overlays: blue = overdensity, red = sky underdensity.
_HYPER_OVER = "#2563eb"
_HYPER_UNDER = "#dc2626"
_FIELD_SCATTER = "#9aa5b1"


def _shorten_stream_label(label, max_len=12):
    """Shorten stream label for legend; e.g. MOCK_123 -> M123."""
    s = str(label)
    if len(s) <= max_len:
        return s
    if s.startswith("MOCK_"):
        return "M" + s[5:]  # MOCK_123 -> M123
    return s[: max_len - 1] + "…"


def plot_cell_three_panel(
    cell_dfs,
    cell_ids,
    ra_col="ra_des",
    dec_col="dec_des",
    phi_col="phi",
    lam_col="lam",
    pm_phi_col="pm_phi",
    pm_lam_col="pm_lam",
    is_train_col="is_train",
    stream_label_col="stream_label",
    figsize=(14, 4.2),
    s_bg=0.6,
    s_train=4.0,
    alpha_bg=0.18,
    alpha_train=0.85,
    color_bg="#5a8ab0",
    dpi=200,
    show=True,
    savepath=None,
):
    """
    Plot N cells as separate 1×3 figures: (ra, dec) | (phi, λ) | (pm_φ, pm_λ).

    Highlighted points are mock-stream members when ``is_mock_stream`` is present;
    otherwise legacy ``is_train & stream_label != Background`` (restricted to
    ``is_mock`` when available so real/S5 streams are not shown as training).
    """
    from matplotlib.lines import Line2D

    col_labels = ["Sky (RA, Dec)", "Tangent plane (φ, λ)", "Proper motion (μ_φ, μ_λ)"]

    results = []
    for cid in cell_ids:
        fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi, squeeze=False)
        axes = axes[0]

        if cid not in cell_dfs:
            for j in range(3):
                axes[j].text(0.5, 0.5, f"Cell {cid} not in cell_dfs", ha="center", va="center", transform=axes[j].transAxes)
            results.append((fig, axes))
            if show:
                plt.show()
            continue

        df = ensure_is_mock_stream(cell_dfs[cid])
        # Prefer mock-stream members for highlighting (SC / DES+mocks setup).
        # Fall back to legacy is_train & non-Background for older cell files.
        if "is_mock_stream" in df.columns:
            train = df["is_mock_stream"].to_numpy(bool)
        else:
            train = (
                df[is_train_col].to_numpy(bool)
                & (df[stream_label_col] != "Background").to_numpy()
            )
            if "is_mock" in df.columns:
                train = train & df["is_mock"].to_numpy(bool)
        bg = ~train
        labels = df[stream_label_col].to_numpy(object)
        train_labels = np.unique(labels[train])
        stream_colors = {
            lab: _STREAM_COLORS[i % len(_STREAM_COLORS)]
            for i, lab in enumerate(train_labels)
        }

        def _scatter_panels(ax, x_vals, y_vals, x_fin, y_fin):
            m = x_fin & y_fin
            if bg[m].any():
                ax.scatter(x_vals[m & bg], y_vals[m & bg], s=s_bg, c=color_bg, alpha=alpha_bg, rasterized=True, linewidths=0)
            for lab in train_labels:
                mask = m & (labels == lab)
                if mask.any():
                    ax.scatter(x_vals[mask], y_vals[mask], s=s_train, c=stream_colors[lab], alpha=alpha_train, rasterized=True, zorder=5, linewidths=0, edgecolors="none")

        # Left: (ra, dec) — uniform box around center (RA wrap-safe: shift RA for plotting)
        ax = axes[0]
        ra, dec = df[ra_col].to_numpy(), df[dec_col].to_numpy()
        m = np.isfinite(ra) & np.isfinite(dec)
        if m.any():
            ra_m, dec_m = ra[m], dec[m]
            extent_ra = np.ptp(ra_m)
            extent_ra = 360 - extent_ra if extent_ra > 180 else extent_ra
            extent_dec = np.ptp(dec_m)
            box = max(extent_ra, extent_dec, 2.0) * 1.15
            ra_r = np.deg2rad(ra_m)
            ra_c = np.rad2deg(np.arctan2(np.mean(np.sin(ra_r)), np.mean(np.cos(ra_r))))
            dec_c = np.median(dec_m)
            # Shift RA so center is continuous (handles wrap at 0/360)
            ra_plot = ((ra - ra_c + 180) % 360 - 180) + ra_c
            _scatter_panels(ax, ra_plot, dec, np.isfinite(ra) & np.isfinite(dec), np.isfinite(dec))
            ax.set_xlim(ra_c - box / 2, ra_c + box / 2)
            ax.set_ylim(dec_c - box / 2, dec_c + box / 2)
            # Show true RA on axis when wrapped (e.g. -5 -> 355)
            def _ra_tick(x):
                return f"{x + 360:.0f}" if x < 0 else f"{x:.0f}"
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: _ra_tick(x)))
        else:
            _scatter_panels(ax, ra, dec, np.isfinite(ra), np.isfinite(dec))
        ax.set_xlabel("RA (deg)", fontsize=10)
        ax.set_ylabel("Dec (deg)", fontsize=10)
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.set_title(f"Cell {cid} — {col_labels[0]}", fontsize=11, fontweight="medium")

        # Center: (phi, λ)
        ax = axes[1]
        phi, lam = df[phi_col].to_numpy(), df[lam_col].to_numpy()
        _scatter_panels(ax, phi, lam, np.isfinite(phi), np.isfinite(lam))
        ax.set_xlabel("φ (deg)", fontsize=10)
        ax.set_ylabel("λ (deg)", fontsize=10)
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.set_title(col_labels[1], fontsize=11, fontweight="medium")

        # Right: (pm_φ, pm_λ)
        ax = axes[2]
        pm_phi, pm_lam = df[pm_phi_col].to_numpy(), df[pm_lam_col].to_numpy()
        _scatter_panels(ax, pm_phi, pm_lam, np.isfinite(pm_phi), np.isfinite(pm_lam))
        ax.set_xlabel("μ_φ (mas/yr)", fontsize=10)
        ax.set_ylabel("μ_λ (mas/yr)", fontsize=10)
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.set_title(col_labels[2], fontsize=11, fontweight="medium")

        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=color_bg, markersize=5, alpha=alpha_bg, label="Background"),
        ]
        for lab in train_labels:
            legend_elements.append(
                Line2D([0], [0], marker="o", color="w", markerfacecolor=stream_colors[lab], markersize=6, alpha=alpha_train, label=_shorten_stream_label(lab))
            )
        ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

        plt.tight_layout(pad=0.8, w_pad=0.6, h_pad=0.4)
        if savepath is not None:
            sp = Path(savepath)
            stem, ext = sp.stem, sp.suffix
            fig.savefig(sp.parent / f"{stem}_cell{cid}{ext}", dpi=dpi, bbox_inches="tight")
        results.append((fig, axes))
        if show:
            plt.show()

    return results if not show else None


def plot_cell_cmd(
    cell_dfs,
    cell_ids,
    g_col="psf_mag_aper_8_g_corrected_des",
    r_col="psf_mag_aper_8_r_corrected_des",
    is_train_col="is_train",
    stream_label_col="stream_label",
    figsize=(6, 6),
    s_bg=0.6,
    s_train=4.0,
    alpha_bg=0.18,
    alpha_train=0.85,
    color_bg="#5a8ab0",
    dpi=200,
    show=True,
    savepath=None,
):
    """
    Plot (g−r, r) color-magnitude diagrams for N cells, each as a separate figure.
    Mock-stream members are highlighted by ``stream_label`` (see ``is_mock_stream``).
    """
    from matplotlib.lines import Line2D

    results = []
    for cid in cell_ids:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        if cid not in cell_dfs:
            ax.text(0.5, 0.5, f"Cell {cid} not in cell_dfs", ha="center", va="center", transform=ax.transAxes)
            results.append((fig, ax))
            if show:
                plt.show()
            continue

        df = ensure_is_mock_stream(cell_dfs[cid])
        g = df[g_col].to_numpy()
        r = df[r_col].to_numpy()
        g_r = g - r

        # Restrict to sane magnitude range (avoids 1e10 offset from bad/outlier values)
        m_mag = (r >= 10) & (r <= 25) & (g_r >= -1) & (g_r <= 4)

        if "is_mock_stream" in df.columns:
            train = df["is_mock_stream"].to_numpy(bool)
        else:
            train = (
                df[is_train_col].to_numpy(bool)
                & (df[stream_label_col] != "Background").to_numpy()
            )
            if "is_mock" in df.columns:
                train = train & df["is_mock"].to_numpy(bool)
        bg = ~train
        labels = df[stream_label_col].to_numpy(object)
        train_labels = np.unique(labels[train])
        stream_colors = {
            lab: _STREAM_COLORS[i % len(_STREAM_COLORS)]
            for i, lab in enumerate(train_labels)
        }

        m = np.isfinite(g_r) & np.isfinite(r) & m_mag
        if bg[m].any():
            ax.scatter(g_r[m & bg], r[m & bg], s=s_bg, c=color_bg, alpha=alpha_bg, rasterized=True, linewidths=0)
        for lab in train_labels:
            mask = m & (labels == lab)
            if mask.any():
                ax.scatter(g_r[mask], r[mask], s=s_train, c=stream_colors[lab], alpha=alpha_train, rasterized=True, zorder=5, linewidths=0, edgecolors="none")

        ax.set_xlabel("g − r (mag)", fontsize=10)
        ax.set_ylabel("r (mag)", fontsize=10)
        ax.invert_yaxis()
        ax.ticklabel_format(style="plain", axis="y", useOffset=False)
        ax.grid(True, linestyle=":", alpha=0.4)

        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=color_bg, markersize=5, alpha=alpha_bg, label="Background"),
        ]
        for lab in train_labels:
            legend_elements.append(
                Line2D([0], [0], marker="o", color="w", markerfacecolor=stream_colors[lab], markersize=6, alpha=alpha_train, label=_shorten_stream_label(lab))
            )
        ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
        ax.set_title(f"Cell {cid} — CMD (g−r, r)", fontsize=11, fontweight="medium")

        plt.tight_layout(pad=0.8)
        if savepath is not None:
            sp = Path(savepath)
            stem, ext = sp.stem, sp.suffix
            fig.savefig(sp.parent / f"{stem}_cell{cid}{ext}", dpi=dpi, bbox_inches="tight")
        results.append((fig, ax))
        if show:
            plt.show()

    return results if not show else None


def _wrap_ra_mollweide(ra_deg, flip_ra=True):
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    ra = np.mod(ra, 2 * np.pi)
    ra[ra > np.pi] -= 2 * np.pi
    if flip_ra:
        ra = -ra
    return ra


def plot_sky_with_circles(
    df,
    centers,
    ra_col="ra_des",
    dec_col="dec_des",
    circle_radius=10.0,
    max_points=400_000,
    seed=0,
    figsize=(12, 6),
    s=0.15,
    alpha=0.15,
    flip_ra=True,
    use_precomputed_cells=True,
    color_by_assignment=False,
    stream_label_col="stream_label",
    is_train_col="is_train",  # unused; kept for call-site compatibility
    verbose=True,
):
    """
    Plot stars on Mollweide with circles at the given centers.
    If use_precomputed_cells=True, df must have in_cell_1, in_cell_2, ... (from checkpoint).
    If use_precomputed_cells=False, cells are computed from centers.
    If color_by_assignment=True: blue=in cluster, red=unassigned, and print
    mock-/real-stream assigned vs unassigned counts.
    If verbose=True, also print per-cell report. Generic summary always printed when color_by_assignment=True.
    Returns (df_out, centers).
    """
    del is_train_col  # legacy kw; summaries use mock/real stream membership
    if not centers:
        raise ValueError("centers must be a non-empty list of (ra, dec) tuples")

    centers = [(float(c[0]), float(c[1])) for c in centers]
    n_centers = len(centers)

    if use_precomputed_cells:
        cell_cols = [f"in_cell_{j + 1}" for j in range(n_centers)]
        missing = [c for c in cell_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Precomputed cells missing: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        df_out = df
    else:
        df_out = df.copy()
        centers_xyz = _radec_to_xyz(
            np.array([c[0] for c in centers]),
            np.array([c[1] for c in centers]),
        )
        xyz_all = _radec_to_xyz(df[ra_col].to_numpy(), df[dec_col].to_numpy())
        r_rad = np.deg2rad(circle_radius)
        for j in range(n_centers):
            dist = _angular_distance_rad(xyz_all, centers_xyz[j : j + 1])
            df_out[f"in_cell_{j + 1}"] = dist <= r_rad

    m = np.isfinite(df_out[ra_col]) & np.isfinite(df_out[dec_col])
    sub = df_out.loc[m].copy()
    if max_points is not None and len(sub) > max_points:
        rng = np.random.default_rng(seed)
        sub = sub.iloc[rng.choice(len(sub), size=max_points, replace=False)]

    ra = _wrap_ra_mollweide(sub[ra_col].to_numpy(), flip_ra=flip_ra)
    dec = np.deg2rad(sub[dec_col].to_numpy(dtype=float))

    fig = plt.figure(figsize=figsize, dpi=250)
    ax = fig.add_subplot(111, projection="mollweide")

    if color_by_assignment:
        in_any = np.zeros(len(df_out), dtype=bool)
        for j in range(n_centers):
            in_any |= df_out[f"in_cell_{j + 1}"].to_numpy()
        positions = df_out.index.get_indexer(sub.index)
        in_any_sub = in_any[positions]
        ax.scatter(ra[in_any_sub], dec[in_any_sub], s=0.2, c="#4a90d9", alpha=0.25, linewidths=0, marker=".", label="in cluster")
        ax.scatter(ra[~in_any_sub], dec[~in_any_sub], s=0.5, c="red", alpha=0.5, linewidths=0, marker=".", label="unassigned")
        ax.legend(loc="upper right")

        is_mock_stream, is_real_stream = _mock_real_stream_masks(
            df_out, stream_label_col=stream_label_col
        )
        if "is_mock" in df_out.columns:
            is_des_bg = (
                (~df_out["is_mock"].to_numpy(bool))
                & (df_out[stream_label_col].astype(str) == "Background").to_numpy()
            )
        else:
            is_des_bg = (~is_mock_stream) & (~is_real_stream)

        unassigned = ~in_any
        n_total = len(df_out)

        def _stream_assignment_line(name: str, mask: np.ndarray) -> str:
            n = int(mask.sum())
            n_asg = int((mask & in_any).sum())
            n_unasg = int((mask & unassigned).sum())
            return f"{name}: assigned {n_asg:,} / unassigned {n_unasg:,} / total {n:,}"

        print(
            f"\n  Total stars: {n_total:,}\n"
            f"  {_stream_assignment_line('Mock stream', is_mock_stream)}\n"
            f"  {_stream_assignment_line('DES background', is_des_bg)}\n"
            f"  {_stream_assignment_line('Real stream (expect 0)', is_real_stream)}"
        )
        if verbose:
            rows = []
            for j in range(n_centers):
                mask = df_out[f"in_cell_{j + 1}"].to_numpy()
                n_stars = int(mask.sum())
                n_mock_c = int((mask & is_mock_stream).sum())
                n_des_c = int((mask & is_des_bg).sum())
                n_real_c = int((mask & is_real_stream).sum())
                rows.append(
                    {
                        "cell": j + 1,
                        "n_stars": n_stars,
                        "n_mock_stream": n_mock_c,
                        "n_des_bg": n_des_c,
                        "n_real_stream": n_real_c,
                        "frac_mock_stream": (n_mock_c / n_stars) if n_stars else 0.0,
                    }
                )
            pd.set_option("display.max_rows", None)
            print(pd.DataFrame(rows).to_string(index=False))
    else:
        ax.scatter(ra, dec, s=s, c="#4a90d9", alpha=alpha, linewidths=0, marker=".")

    ra_c_plot = _wrap_ra_mollweide(np.array([c[0] for c in centers]), flip_ra=flip_ra)
    dec_c_plot = np.deg2rad(np.array([c[1] for c in centers]))
    theta = np.linspace(0, 2 * np.pi, 64)
    r_rad = np.deg2rad(circle_radius)
    for i in range(n_centers):
        lam = dec_c_plot[i] + r_rad * np.sin(theta)
        phi = ra_c_plot[i] + r_rad * np.cos(theta) / np.cos(dec_c_plot[i] + 1e-8)
        phi = np.where(phi > np.pi, phi - 2 * np.pi, phi)
        phi = np.where(phi < -np.pi, phi + 2 * np.pi, phi)
        ax.plot(phi, lam, "k-", lw=0.4, alpha=0.5)

    ax.grid(True, linestyle=":", alpha=0.6)
    ra_tick_deg = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210, 180])
    ax.set_xticks(_wrap_ra_mollweide(ra_tick_deg, flip_ra=flip_ra))
    ax.set_xticklabels([f"{d}°" for d in ra_tick_deg])
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    title = f"{n_centers} circles (R={circle_radius}°)"
    if color_by_assignment:
        title = "Clustered (blue) vs unassigned (red)  |  " + title
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

    return df_out, centers


def plot_named_streams_sky(
    df,
    centers=None,
    *,
    ra_col="ra_des",
    dec_col="dec_des",
    stream_col="stream_label",
    target_col="is_mock_stream",
    circle_radius=5.0,
    max_bg=80_000,
    seed=0,
    figsize=(12, 6),
    flip_ra=True,
    show=True,
    savepath=None,
):
    """Mollweide sky: field subsample in gray, named (S5) streams in color."""
    from matplotlib.lines import Line2D

    m_sky = np.isfinite(df[ra_col]) & np.isfinite(df[dec_col])
    work = df.loc[m_sky]
    target = (
        work[target_col].to_numpy(bool)
        if target_col in work.columns
        else (work[stream_col].astype(str) != "Background").to_numpy()
    )
    labels = work[stream_col].astype(str)
    names = [n for n in labels[target].value_counts().index if n != "Background"]

    fig = plt.figure(figsize=figsize, dpi=160)
    ax = fig.add_subplot(111, projection="mollweide")

    bg = work.loc[~target]
    if len(bg) > max_bg:
        bg = bg.sample(max_bg, random_state=seed)
    ra_bg = _wrap_ra_mollweide(bg[ra_col].to_numpy(), flip_ra=flip_ra)
    dec_bg = np.deg2rad(bg[dec_col].to_numpy(dtype=float))
    ax.scatter(ra_bg, dec_bg, s=0.15, c="#b0b0b0", alpha=0.25, linewidths=0, marker=".", zorder=1)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#b0b0b0",
               markersize=5, label="field")
    ]
    for i, name in enumerate(names):
        part = work.loc[labels.to_numpy() == name]
        color = _STREAM_COLORS[i % len(_STREAM_COLORS)]
        ra = _wrap_ra_mollweide(part[ra_col].to_numpy(), flip_ra=flip_ra)
        dec = np.deg2rad(part[dec_col].to_numpy(dtype=float))
        ax.scatter(ra, dec, s=8, c=color, alpha=0.85, linewidths=0, marker=".", zorder=2)
        handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                   markersize=7, label=f"{name} ({len(part):,})")
        )

    if centers:
        ra_c = _wrap_ra_mollweide(np.array([c[0] for c in centers]), flip_ra=flip_ra)
        dec_c = np.deg2rad(np.array([c[1] for c in centers], dtype=float))
        theta = np.linspace(0, 2 * np.pi, 64)
        r_rad = np.deg2rad(circle_radius)
        for i in range(len(centers)):
            lam = dec_c[i] + r_rad * np.sin(theta)
            phi = ra_c[i] + r_rad * np.cos(theta) / np.cos(dec_c[i] + 1e-8)
            phi = np.where(phi > np.pi, phi - 2 * np.pi, phi)
            phi = np.where(phi < -np.pi, phi + 2 * np.pi, phi)
            ax.plot(phi, lam, "k-", lw=0.35, alpha=0.35, zorder=0)

    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_title("Catalogued streams on the DES sky (mocks removed)")
    plt.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
        return None
    return fig, ax


def plot_stream_cutflow(
    table: pd.DataFrame,
    *,
    count_cols=None,
    figsize=None,
    dpi=140,
    show=True,
    savepath=None,
):
    """Grouped bars: remaining members of each named stream after sequential cuts."""
    if count_cols is None:
        skip = {"stream_label", "n_input"}
        count_cols = [
            c for c in table.columns
            if c not in skip and not str(c).endswith("_frac")
        ]
    stages = ["n_input"] + list(count_cols)
    names = table["stream_label"].astype(str).tolist()
    x = np.arange(len(names))
    n_st = len(stages)
    width = 0.8 / max(n_st, 1)
    if figsize is None:
        figsize = (max(8.0, 1.4 * len(names)), 4.2)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, n_st))
    for i, col in enumerate(stages):
        vals = table[col].to_numpy(float) if col in table.columns else np.zeros(len(names))
        ax.bar(x + (i - 0.5 * (n_st - 1)) * width, vals, width=width, color=cmap[i], label=col)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("stars")
    ax.set_title("Named-stream retention through the pipeline")
    ax.legend(fontsize=8, ncol=min(n_st, 4))
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
        return None
    return fig, ax


def _mock_real_stream_masks(
    df: pd.DataFrame,
    *,
    stream_label_col: str = "stream_label",
) -> tuple[np.ndarray, np.ndarray]:
    """Bool masks for mock-stream and real-stream members (not background)."""
    labels = df[stream_label_col].astype(str)

    if "is_mock_stream" in df.columns:
        is_mock_stream = df["is_mock_stream"].to_numpy(bool)
    elif "is_mock" in df.columns and "stream_id" in df.columns:
        is_mock = df["is_mock"].to_numpy(bool)
        sid = pd.to_numeric(df["stream_id"], errors="coerce").fillna(-999).astype("int64")
        is_mock_stream = is_mock & (sid.to_numpy() > 0)
    else:
        is_mock_stream = labels.str.startswith("MOCK_").to_numpy()

    if "is_real_stream" in df.columns:
        is_real_stream = df["is_real_stream"].to_numpy(bool)
    elif "is_mock" in df.columns:
        is_mock = df["is_mock"].to_numpy(bool)
        is_real_stream = (~is_mock) & (labels != "Background").to_numpy()
    else:
        is_real_stream = (~labels.str.startswith("MOCK_")) & (labels != "Background")
        is_real_stream = is_real_stream.to_numpy()

    return is_mock_stream, is_real_stream


def plot_aggregate_parallax_cmd(
    df,
    *,
    g_col="psf_mag_aper_8_g_corrected_des",
    r_col="psf_mag_aper_8_r_corrected_des",
    plx_col="parallax_gaia",
    in_any_cell=True,
    g_r_lim=(-0.6, 2.4),
    plx_lim=(-2.0, 2.5),
    gridsize=90,
    figsize=(10.5, 8.2),
    dpi=160,
    color_bg="#4c78a8",
    color_mock="#c0392b",
    cmap="YlGnBu",
    show=True,
    savepath=None,
):
    """Aggregate parallax vs. g−r for every DES cell, with mock streams on top.

    Stars in overlapping cells are counted once (union of ``in_cell_*``).
    Background is shown as a log-count hexbin; mocks are overplotted as points.
    Marginal histograms compare the two populations in each coordinate.
    """
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D

    df = ensure_is_mock_stream(df)
    cell_cols = [c for c in df.columns if str(c).startswith("in_cell_")]
    n_cells = len(cell_cols)
    if in_any_cell and cell_cols:
        in_any = np.zeros(len(df), dtype=bool)
        for c in cell_cols:
            in_any |= df[c].to_numpy(bool, na_value=False)
        work = df.loc[in_any]
    else:
        work = df

    g = work[g_col].to_numpy()
    r = work[r_col].to_numpy()
    g_r = g - r
    plx = work[plx_col].to_numpy()
    is_mock = work["is_mock_stream"].to_numpy(bool)

    finite = (
        np.isfinite(g_r)
        & np.isfinite(plx)
        & (g_r >= g_r_lim[0])
        & (g_r <= g_r_lim[1])
        & (plx >= plx_lim[0])
        & (plx <= plx_lim[1])
    )
    g_r, plx, is_mock = g_r[finite], plx[finite], is_mock[finite]
    other = ~is_mock

    n_mock = int(is_mock.sum())
    n_other = int(other.sum())
    n_streams = (
        work.loc[work["is_mock_stream"].to_numpy(bool), "stream_label"].nunique()
        if "stream_label" in work.columns
        else 0
    )

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[4.2, 0.12, 1.15],
        height_ratios=[1.05, 4.0],
        hspace=0.03,
        wspace=0.04,
        left=0.09,
        right=0.98,
        top=0.90,
        bottom=0.10,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0], sharex=ax_top)
    cax = fig.add_subplot(gs[1, 1])
    ax_right = fig.add_subplot(gs[1, 2], sharey=ax)

    if other.any():
        hb = ax.hexbin(
            g_r[other],
            plx[other],
            gridsize=gridsize,
            cmap=cmap,
            mincnt=1,
            bins="log",
            linewidths=0,
            rasterized=True,
        )
        cb = fig.colorbar(hb, cax=cax)
        cb.set_label("other stars / hex (log)", fontsize=9)
    else:
        cax.set_axis_off()
    if n_mock:
        ax.scatter(
            g_r[is_mock],
            plx[is_mock],
            s=7,
            c=color_mock,
            alpha=0.75,
            linewidths=0,
            zorder=5,
            rasterized=True,
            label="mock stream",
        )

    ax.set_xlabel(r"$g-r$ (mag)", fontsize=11)
    ax.set_ylabel("parallax (mas)", fontsize=11)
    ax.set_xlim(*g_r_lim)
    ax.set_ylim(*plx_lim)
    ax.axhline(0.0, color="0.35", lw=0.6, ls=":", zorder=3)
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(
        handles=[
            Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor=color_mock, markersize=7, label="mock stream",
            ),
        ],
        loc="upper right",
        frameon=True,
        fontsize=9,
    )

    bins_c = np.linspace(g_r_lim[0], g_r_lim[1], 70)
    bins_p = np.linspace(plx_lim[0], plx_lim[1], 70)
    if n_other:
        ax_top.hist(
            g_r[other], bins=bins_c, density=True, histtype="stepfilled",
            color=color_bg, alpha=0.45, label="other",
        )
        ax_right.hist(
            plx[other], bins=bins_p, density=True, histtype="stepfilled",
            color=color_bg, alpha=0.45, orientation="horizontal",
        )
    if n_mock:
        ax_top.hist(
            g_r[is_mock], bins=bins_c, density=True, histtype="step",
            color=color_mock, lw=1.6, label="mock stream",
        )
        ax_right.hist(
            plx[is_mock], bins=bins_p, density=True, histtype="step",
            color=color_mock, lw=1.6, orientation="horizontal",
        )
    ax_top.set_ylabel("density", fontsize=9)
    ax_top.legend(fontsize=8, loc="upper right", frameon=False)
    ax_top.tick_params(labelbottom=False)
    ax_top.grid(True, linestyle=":", alpha=0.3)
    ax_right.set_xlabel("density", fontsize=9)
    ax_right.tick_params(labelleft=False)
    ax_right.grid(True, linestyle=":", alpha=0.3)

    cell_note = f"{n_cells} overlapping DES cells" if n_cells else "full table"
    fig.suptitle(
        "All cells: parallax vs. $g-r$  "
        f"(union of {cell_note})",
        fontsize=13,
        fontweight="medium",
        y=0.97,
    )
    stats = (
        f"in-cell stars  {n_other + n_mock:,}\n"
        f"other          {n_other:,}\n"
        f"mock stream    {n_mock:,}  ({n_streams} streams)"
    )
    if n_mock:
        stats += f"\nmock median π  {np.median(plx[is_mock]):.3f} mas"
    ax.text(
        0.02,
        0.98,
        stats,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.82, edgecolor="0.8"),
    )

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
        return None
    return fig, (ax, ax_top, ax_right)


def fit_linear_cmd_isochrone(g_r, r, is_mock):
    """Fit a straight isochrone ``r = m (g-r) + b`` to mock stars via PCA/SVD.

    Two *parallel* diagonals are then ``r = m(g-r) + b ± w√(1+m²)``, i.e. a
    slab of orthogonal half-width ``w`` (mixed mag units).
    """
    g_r = np.asarray(g_r, dtype=float)
    r = np.asarray(r, dtype=float)
    is_mock = np.asarray(is_mock, dtype=bool)
    xy = np.column_stack([g_r[is_mock], r[is_mock]])
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) < 5:
        raise ValueError("Need ≥5 finite mock stars to fit a linear isochrone")
    mu = xy.mean(axis=0)
    _, _, vt = np.linalg.svd(xy - mu, full_matrices=False)
    direction = vt[0]
    if direction[0] < 0:
        direction = -direction
    m = float(direction[1] / direction[0])
    b = float(mu[1] - m * mu[0])
    return {"mu": mu, "direction": direction, "m": m, "b": b}


def cmd_slab_signed_distance(g_r, r, m, b):
    """Orthogonal signed distance to ``r = m(g-r) + b``."""
    return (np.asarray(r, dtype=float) - m * np.asarray(g_r, dtype=float) - b) / np.sqrt(1.0 + m * m)


def cmd_slab_intercepts(m, b, half_width):
    """Return ``(b_lo, b_hi)`` so the slab is ``b_lo ≤ r - m(g-r) ≤ b_hi``."""
    delta = half_width * np.sqrt(1.0 + m * m)
    return b - delta, b + delta


def sweep_cmd_slab(g_r, r, is_mock, m, b, half_widths):
    """Keep stars with |distance to line| ≤ w; return a sweep DataFrame."""
    dist = np.abs(cmd_slab_signed_distance(g_r, r, m, b))
    is_mock = np.asarray(is_mock, dtype=bool)
    n_mock_total = int(is_mock.sum())
    n_other_total = int((~is_mock).sum())
    rows = []
    for w in half_widths:
        keep = dist <= w
        n_mock = int((is_mock & keep).sum())
        n_other = int((~is_mock & keep).sum())
        rows.append({
            "half_width": w,
            "n_mock": n_mock,
            "n_other": n_other,
            "mock_frac": n_mock / n_mock_total if n_mock_total else np.nan,
            "other_frac": n_other / n_other_total if n_other_total else np.nan,
            "purity": n_mock / (n_mock + n_other) if (n_mock + n_other) else np.nan,
        })
    return pd.DataFrame(rows)


def plot_aggregate_cmd(
    df,
    *,
    g_col="psf_mag_aper_8_g_corrected_des",
    r_col="psf_mag_aper_8_r_corrected_des",
    in_any_cell=True,
    g_r_lim=(-0.6, 2.4),
    r_lim=(15.0, 24.0),
    g_r_cut=None,
    mark_mock_quantile=0.9,
    slab=None,
    gridsize=90,
    figsize=(10.5, 8.2),
    dpi=160,
    color_bg="#4c78a8",
    color_mock="#c0392b",
    cmap="YlGnBu",
    show=True,
    savepath=None,
):
    """Aggregate CMD (g−r, r) for every DES cell, with mock streams on top.

    Same layout as ``plot_aggregate_parallax_cmd``. Magnitude axis is inverted
    (bright at top). Optional ``g_r_cut`` draws a vertical color cut; if omitted
    and ``slab`` is not given, a dashed line at the mock ``mark_mock_quantile``
    of g−r is shown. ``slab=(m, b_lo, b_hi)`` draws the parallel diagonals
    ``r = m(g-r) + b``.
    """
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D

    df = ensure_is_mock_stream(df)
    cell_cols = [c for c in df.columns if str(c).startswith("in_cell_")]
    n_cells = len(cell_cols)
    if in_any_cell and cell_cols:
        in_any = np.zeros(len(df), dtype=bool)
        for c in cell_cols:
            in_any |= df[c].to_numpy(bool, na_value=False)
        work = df.loc[in_any]
    else:
        work = df

    g = work[g_col].to_numpy()
    r = work[r_col].to_numpy()
    g_r = g - r
    is_mock = work["is_mock_stream"].to_numpy(bool)

    finite = (
        np.isfinite(g_r)
        & np.isfinite(r)
        & (g_r >= g_r_lim[0])
        & (g_r <= g_r_lim[1])
        & (r >= r_lim[0])
        & (r <= r_lim[1])
    )
    g_r, r, is_mock = g_r[finite], r[finite], is_mock[finite]
    other = ~is_mock

    n_mock = int(is_mock.sum())
    n_other = int(other.sum())
    n_streams = (
        work.loc[work["is_mock_stream"].to_numpy(bool), "stream_label"].nunique()
        if "stream_label" in work.columns
        else 0
    )

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[4.2, 0.12, 1.15],
        height_ratios=[1.05, 4.0],
        hspace=0.03,
        wspace=0.04,
        left=0.09,
        right=0.98,
        top=0.90,
        bottom=0.10,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0], sharex=ax_top)
    cax = fig.add_subplot(gs[1, 1])
    ax_right = fig.add_subplot(gs[1, 2], sharey=ax)

    if other.any():
        hb = ax.hexbin(
            g_r[other],
            r[other],
            gridsize=gridsize,
            cmap=cmap,
            mincnt=1,
            bins="log",
            linewidths=0,
            rasterized=True,
        )
        cb = fig.colorbar(hb, cax=cax)
        cb.set_label("other stars / hex (log)", fontsize=9)
    else:
        cax.set_axis_off()
    if n_mock:
        ax.scatter(
            g_r[is_mock],
            r[is_mock],
            s=7,
            c=color_mock,
            alpha=0.75,
            linewidths=0,
            zorder=5,
            rasterized=True,
            label="mock stream",
        )

    cut_val = g_r_cut
    cut_label = None
    if slab is None and cut_val is None and n_mock and mark_mock_quantile is not None:
        cut_val = float(np.quantile(g_r[is_mock], mark_mock_quantile))
        cut_label = f"mock {int(mark_mock_quantile * 100)}% $g-r$={cut_val:.2f}"
    elif cut_val is not None:
        cut_label = f"$g-r$ cut = {cut_val:.2f}"
    if cut_val is not None:
        ax.axvline(cut_val, color=color_mock, lw=1.1, ls="--", zorder=4, label=cut_label)
        ax_top.axvline(cut_val, color=color_mock, lw=1.1, ls="--")

    slab_label = None
    if slab is not None:
        m_s, b_lo, b_hi = slab
        xs = np.linspace(g_r_lim[0], g_r_lim[1], 50)
        ax.plot(xs, m_s * xs + b_lo, color=color_mock, lw=1.15, ls="--", zorder=4)
        ax.plot(xs, m_s * xs + b_hi, color=color_mock, lw=1.15, ls="--", zorder=4)
        slab_label = rf"slab $r=m(g-r)+b$, $m={m_s:.2f}$"

    ax.set_xlabel(r"$g-r$ (mag)", fontsize=11)
    ax.set_ylabel("$r$ (mag)", fontsize=11)
    ax.set_xlim(*g_r_lim)
    ax.set_ylim(r_lim[1], r_lim[0])  # bright at top
    ax.grid(True, linestyle=":", alpha=0.35)
    legend_handles = [
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=color_mock, markersize=7, label="mock stream",
        ),
    ]
    if cut_label:
        legend_handles.append(
            Line2D([0], [0], color=color_mock, ls="--", lw=1.2, label=cut_label)
        )
    if slab_label:
        legend_handles.append(
            Line2D([0], [0], color=color_mock, ls="--", lw=1.2, label=slab_label)
        )
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=9)

    bins_c = np.linspace(g_r_lim[0], g_r_lim[1], 70)
    bins_r = np.linspace(r_lim[0], r_lim[1], 70)
    if n_other:
        ax_top.hist(
            g_r[other], bins=bins_c, density=True, histtype="stepfilled",
            color=color_bg, alpha=0.45, label="other",
        )
        ax_right.hist(
            r[other], bins=bins_r, density=True, histtype="stepfilled",
            color=color_bg, alpha=0.45, orientation="horizontal",
        )
    if n_mock:
        ax_top.hist(
            g_r[is_mock], bins=bins_c, density=True, histtype="step",
            color=color_mock, lw=1.6, label="mock stream",
        )
        ax_right.hist(
            r[is_mock], bins=bins_r, density=True, histtype="step",
            color=color_mock, lw=1.6, orientation="horizontal",
        )
    ax_top.set_ylabel("density", fontsize=9)
    ax_top.legend(fontsize=8, loc="upper right", frameon=False)
    ax_top.tick_params(labelbottom=False)
    ax_top.grid(True, linestyle=":", alpha=0.3)
    ax_right.set_xlabel("density", fontsize=9)
    ax_right.tick_params(labelleft=False)
    ax_right.grid(True, linestyle=":", alpha=0.3)

    cell_note = f"{n_cells} overlapping DES cells" if n_cells else "full table"
    fig.suptitle(
        "All cells: CMD ($g-r$, $r$)  "
        f"(union of {cell_note})",
        fontsize=13,
        fontweight="medium",
        y=0.97,
    )
    stats = (
        f"in-cell stars  {n_other + n_mock:,}\n"
        f"other          {n_other:,}\n"
        f"mock stream    {n_mock:,}  ({n_streams} streams)"
    )
    if n_mock:
        stats += (
            f"\nmock median g−r {np.median(g_r[is_mock]):.3f}"
            f"\nmock median r   {np.median(r[is_mock]):.2f}"
        )
    ax.text(
        0.02,
        0.02,
        stats,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=8,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.82, edgecolor="0.8"),
    )

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
        return None
    return fig, (ax, ax_top, ax_right)


def plot_parallax_upper_bound_sweep(
    df,
    *,
    plx_col="parallax_gaia",
    in_any_cell=True,
    chosen=1.0,
    thresholds=None,
    pi_true=0.04,
    plx_sigmas=(0.3, 0.5, 1.0),
    figsize=(11.2, 4.6),
    dpi=150,
    color_bg="#4c78a8",
    color_mock="#c0392b",
    show=True,
    savepath=None,
):
    """Foreground-only parallax cut: keep ``π < π_max`` (no lower bound).

    The parent sample is DES; ``parallax_gaia`` is the Gaia astrometric
    measurement on the DES×Gaia crossmatch. Background rejection is measured
    on the DES field. Mock keep is shown only as a diagnostic of unsmeared
    truth-π (do not tune on it). Smeared-stream efficiency is
    ``P(π_obs < π_max)`` for ``π_obs ~ N(π_true, σ)``, i.e. what a DES stream
    star would look like if its Gaia parallax had realistic errors.
    Default ``chosen=1 mas`` keeps a σ≈0.5 mas measurement at ~97% while still
    cutting the nearby-disk tail.
    """
    from scipy.stats import norm

    df = ensure_is_mock_stream(df)
    work = df
    if in_any_cell:
        cell_cols = [c for c in df.columns if str(c).startswith("in_cell_")]
        if cell_cols:
            in_any = np.zeros(len(df), dtype=bool)
            for c in cell_cols:
                in_any |= df[c].to_numpy(bool, na_value=False)
            work = df.loc[in_any]

    plx = work[plx_col].to_numpy(dtype=float)
    is_mock = work["is_mock_stream"].to_numpy(bool)
    finite = np.isfinite(plx)
    plx, is_mock = plx[finite], is_mock[finite]
    other = ~is_mock
    n_other = int(other.sum())
    n_mock = int(is_mock.sum())

    if thresholds is None:
        thresholds = np.linspace(0.15, 2.5, 48)

    rows = []
    for t in thresholds:
        keep = plx < t
        n_o = int((other & keep).sum())
        n_m = int((is_mock & keep).sum())
        rows.append({
            "threshold": t,
            "other_keep": n_o / n_other if n_other else np.nan,
            "other_rej": 1.0 - (n_o / n_other) if n_other else np.nan,
            "mock_keep": n_m / n_mock if n_mock else np.nan,
        })
    sweep = pd.DataFrame(rows)
    t_star = float(chosen)
    keep_star = plx < t_star
    other_rej_star = 1.0 - (other & keep_star).sum() / n_other if n_other else np.nan

    fig, axs = plt.subplots(1, 2, figsize=figsize, dpi=dpi, constrained_layout=True)

    ax = axs[0]
    ax.plot(
        sweep.threshold, sweep.other_rej, color=color_bg, lw=2.0,
        label="field rejection",
    )
    if n_mock:
        ax.plot(
            sweep.threshold, 1.0 - sweep.mock_keep, color=color_mock, lw=1.2,
            ls=":", alpha=0.85, label="unsmeared mock rejection (do not tune)",
        )
    for i, sig in enumerate(plx_sigmas):
        ls = ("--", "-.", ":")[i % 3]
        stream_loss = 1.0 - norm.cdf(sweep.threshold, loc=pi_true, scale=sig)
        ax.plot(
            sweep.threshold, stream_loss, ls=ls, color="0.25", alpha=0.9, lw=1.4,
            label=rf"smeared stream loss ($\sigma_\pi={sig:.1f}$ mas)",
        )
    ax.axvline(
        t_star, color="#c0392b", ls="--", lw=1.3,
        label=rf"chosen $\pi<{t_star:.1f}$ mas",
    )
    ax.set_xlabel(r"parallax upper bound $\pi_{\max}$ (mas)")
    ax.set_ylabel("fraction rejected")
    ax.set_xlim(float(np.min(thresholds)), float(np.max(thresholds)))
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_title(r"Foreground cut: keep $\pi < \pi_{\max}$")
    ax.legend(fontsize=8, loc="upper right", frameon=True)

    ax = axs[1]
    bins = np.linspace(-1.5, 2.5, 70)
    if n_other:
        ax.hist(
            plx[other], bins=bins, density=True, histtype="stepfilled",
            color=color_bg, alpha=0.45, label="other",
        )
    if n_mock:
        ax.hist(
            plx[is_mock], bins=bins, density=True, histtype="step",
            color=color_mock, lw=1.5, label="mock (unsmeared)",
        )
    xs = np.linspace(-1.5, 2.5, 400)
    ax.plot(
        xs, norm.pdf(xs, loc=pi_true, scale=0.5), color="0.2", lw=1.4, ls="-.",
        label=rf"smeared stream $\mathcal{{N}}({pi_true},0.5)$ mas",
    )
    ax.axvline(t_star, color="#c0392b", ls="--", lw=1.3)
    ax.axvspan(t_star, 2.5, color="0.7", alpha=0.25, label="rejected (nearby)")
    ax.set_xlabel("parallax (mas)")
    ax.set_ylabel("density")
    ax.set_xlim(-1.5, 2.5)
    ax.set_title("Why there is no lower bound")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.35)

    fig.suptitle(
        rf"Parallax as a nearby-disk veto  "
        rf"($\pi<{t_star:.1f}$ mas rejects {other_rej_star:.1%} of field)",
        fontsize=12,
        fontweight="medium",
    )

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
        return None
    return fig, axs, sweep


def _ra_for_plot(ra, ra0):
    """Shift RA so the window is continuous around ``ra0``."""
    return ((np.asarray(ra, dtype=float) - ra0 + 180.0) % 360.0) - 180.0 + ra0


def _local_field_mask(ra, dec, track, search_radius):
    from cpen.apps.streams.preprocess.tracks import _rough_sky_box

    length = 0.5 * (track.s_max - track.s_min)
    radius = max(float(search_radius), length + 0.6)
    return _rough_sky_box(ra, dec, track.ra0, track.dec0, radius)


def plot_mock_track_grid(
    df,
    tracks,
    *,
    labels=None,
    n_show=6,
    half_width=0.35,
    search_radius=2.0,
    ra_col="ra_des",
    dec_col="dec_des",
    figsize=None,
    dpi=150,
    cmap="YlGnBu",
    color_mock="#c0392b",
    show=True,
    savepath=None,
):
    """Small-multiples: DES field hexbin + mock members + PCA track + tube."""
    from matplotlib.lines import Line2D

    from cpen.apps.streams.preprocess.galaxy import ensure_is_mock_stream
    from cpen.apps.streams.preprocess.tracks import (
        project_on_track,
        track_curve_radec,
        track_tube_radec,
    )

    df = ensure_is_mock_stream(df)
    if not tracks:
        raise ValueError("No tracks to plot")

    if labels is None:
        by_n = sorted(tracks, key=lambda k: -tracks[k].n_mock)
        by_l = sorted(tracks, key=lambda k: -(tracks[k].s_max - tracks[k].s_min))
        labels = []
        for k in by_n + by_l:
            if k not in labels:
                labels.append(k)
            if len(labels) >= n_show:
                break
    labels = [lb for lb in labels if lb in tracks][:n_show]
    n = len(labels)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (12.4, 4.15 * nrows)

    ra_all = df[ra_col].to_numpy()
    dec_all = df[dec_col].to_numpy()
    is_mock_all = df["is_mock_stream"].to_numpy(bool)
    stream_all = df["stream_label"].astype(str).to_numpy()

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi, squeeze=False)
    for i, lab in enumerate(labels):
        ax = axes[i // ncols][i % ncols]
        track = tracks[lab]
        local = _local_field_mask(ra_all, dec_all, track, search_radius)
        finite = np.isfinite(ra_all) & np.isfinite(dec_all) & local
        ra, dec = ra_all[finite], dec_all[finite]
        is_this = is_mock_all[finite] & (stream_all[finite] == lab)
        other = ~is_this

        ra_p = _ra_for_plot(ra, track.ra0)
        if other.any():
            ax.hexbin(
                ra_p[other], dec[other], gridsize=55, cmap=cmap, mincnt=1,
                bins="log", linewidths=0, rasterized=True,
            )
        if is_this.any():
            ax.scatter(
                ra_p[is_this], dec[is_this], s=8, c=color_mock, zorder=5,
                linewidths=0, rasterized=True, alpha=0.9,
            )

        ra_t, dec_t = track_curve_radec(track)
        (ra_lo, dec_lo), (ra_hi, dec_hi) = track_tube_radec(track, half_width)
        ax.plot(_ra_for_plot(ra_t, track.ra0), dec_t, color="0.1", lw=1.6, zorder=6)
        ax.plot(_ra_for_plot(ra_lo, track.ra0), dec_lo, color=color_mock, lw=1.1, ls="--", zorder=6)
        ax.plot(_ra_for_plot(ra_hi, track.ra0), dec_hi, color=color_mock, lw=1.1, ls="--", zorder=6)

        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.set_xlabel("RA (deg)", fontsize=9)
        ax.set_ylabel("Dec (deg)", fontsize=9)
        ax.set_title(
            f"{lab}   N={track.n_mock}   "
            f"L={track.s_max - track.s_min:.1f}°   "
            rf"$\sigma_\perp$={track.rms_perp:.3f}°",
            fontsize=10,
        )
        s, d = project_on_track(ra[is_this], dec[is_this], track) if is_this.any() else (np.array([]), np.array([]))
        in_tube = float((np.abs(d) <= half_width).mean()) if len(d) else np.nan
        ax.text(
            0.02, 0.04,
            rf"in tube ({half_width:.2f}°): {in_tube:.0%}" if np.isfinite(in_tube) else "",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="0.85"),
        )

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_axis_off()

    fig.suptitle(
        rf"Minimal mock tracks: PCA line in the tangent plane  "
        rf"(dashed tube $\pm {half_width:.2f}$°)",
        fontsize=13,
        fontweight="medium",
        y=1.01,
    )
    fig.legend(
        handles=[
            Line2D([0], [0], color="0.1", lw=1.6, label="PCA track"),
            Line2D([0], [0], color=color_mock, lw=1.1, ls="--", label="tube"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=color_mock, markersize=7, label="mock members"),
        ],
        loc="upper right",
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.99, 1.02),
    )
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
        return None
    return fig, axes


def plot_mock_track_detail(
    df,
    track,
    *,
    half_width=0.35,
    search_radius=2.5,
    ra_col="ra_des",
    dec_col="dec_des",
    figsize=(11.4, 5.0),
    dpi=150,
    cmap="YlGnBu",
    color_mock="#c0392b",
    show=True,
    savepath=None,
):
    """One stream: sky tube (left) and along-track residuals (right)."""
    from matplotlib.patches import Polygon

    from cpen.apps.streams.preprocess.galaxy import ensure_is_mock_stream
    from cpen.apps.streams.preprocess.tracks import (
        project_on_track,
        track_curve_radec,
        track_tube_radec,
    )

    df = ensure_is_mock_stream(df)
    ra_all = df[ra_col].to_numpy()
    dec_all = df[dec_col].to_numpy()
    local = _local_field_mask(ra_all, dec_all, track, search_radius)
    finite = np.isfinite(ra_all) & np.isfinite(dec_all) & local
    ra, dec = ra_all[finite], dec_all[finite]
    is_this = (
        df["is_mock_stream"].to_numpy(bool)[finite]
        & (df["stream_label"].astype(str).to_numpy()[finite] == track.stream_label)
    )
    other = ~is_this
    s, d = project_on_track(ra, dec, track)
    on_seg = (s >= track.s_min) & (s <= track.s_max)

    fig, axs = plt.subplots(1, 2, figsize=figsize, dpi=dpi, constrained_layout=True)

    ax = axs[0]
    ra_p = _ra_for_plot(ra, track.ra0)
    if other.any():
        ax.hexbin(
            ra_p[other], dec[other], gridsize=70, cmap=cmap, mincnt=1,
            bins="log", linewidths=0, rasterized=True,
        )
    (ra_lo, dec_lo), (ra_hi, dec_hi) = track_tube_radec(track, half_width)
    poly_ra = np.concatenate([_ra_for_plot(ra_lo, track.ra0), _ra_for_plot(ra_hi, track.ra0)[::-1]])
    poly_dec = np.concatenate([dec_lo, dec_hi[::-1]])
    ax.add_patch(
        Polygon(
            np.column_stack([poly_ra, poly_dec]),
            closed=True, facecolor=color_mock, alpha=0.12, edgecolor="none", zorder=3,
        )
    )
    ra_t, dec_t = track_curve_radec(track)
    ax.plot(_ra_for_plot(ra_t, track.ra0), dec_t, color="0.1", lw=2.0, zorder=6, label="PCA track")
    ax.plot(_ra_for_plot(ra_lo, track.ra0), dec_lo, color=color_mock, lw=1.15, ls="--", zorder=6)
    ax.plot(_ra_for_plot(ra_hi, track.ra0), dec_hi, color=color_mock, lw=1.15, ls="--", zorder=6, label=rf"tube $\pm {half_width:.2f}$°")
    if is_this.any():
        ax.scatter(
            ra_p[is_this], dec[is_this], s=10, c=color_mock, zorder=7,
            linewidths=0, rasterized=True, label="mock members",
        )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.set_title(f"{track.stream_label} on the sky")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, linestyle=":", alpha=0.35)

    ax = axs[1]
    if other.any():
        ax.hexbin(
            s[other], d[other], gridsize=70, cmap=cmap, mincnt=1,
            bins="log", linewidths=0, rasterized=True,
        )
    if is_this.any():
        ax.scatter(
            s[is_this], d[is_this], s=10, c=color_mock, zorder=5,
            linewidths=0, rasterized=True,
        )
    ax.axhline(0.0, color="0.15", lw=1.4)
    ax.axhline(half_width, color=color_mock, ls="--", lw=1.15)
    ax.axhline(-half_width, color=color_mock, ls="--", lw=1.15)
    ax.axvline(track.s_min, color="0.5", ls=":", lw=0.9)
    ax.axvline(track.s_max, color="0.5", ls=":", lw=0.9)
    ax.set_xlabel(r"along track $s$ (deg)")
    ax.set_ylabel(r"perpendicular $d$ (deg)")
    ax.set_title("Unroll the stream")
    ax.grid(True, linestyle=":", alpha=0.35)
    n_mock = int(is_this.sum())
    n_mock_tube = int((is_this & (np.abs(d) <= half_width) & on_seg).sum())
    n_other_tube = int((other & (np.abs(d) <= half_width) & on_seg).sum())
    ax.text(
        0.02, 0.98,
        f"search window  {int(finite.sum()):,}\n"
        f"mock in tube   {n_mock_tube}/{n_mock}\n"
        f"other in tube  {n_other_tube:,}\n"
        rf"mock $\sigma_\perp$  {track.rms_perp:.3f}°",
        transform=ax.transAxes, va="top", ha="left", fontsize=8, family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="0.8"),
    )

    fig.suptitle(
        "Stage 2 (minimal): linear track + sky tube around a known mock",
        fontsize=13,
        fontweight="medium",
    )
    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
        return None
    return fig, axs


def plot_track_width_sweep(
    df,
    tracks,
    *,
    half_widths=None,
    search_radius=2.5,
    ra_col="ra_des",
    dec_col="dec_des",
    chosen=0.35,
    figsize=(11.0, 4.5),
    dpi=150,
    color_bg="#4c78a8",
    color_mock="#c0392b",
    show=True,
    savepath=None,
):
    """Mock kept vs field kept as a function of tube half-width (per-stream windows)."""
    from cpen.apps.streams.preprocess.galaxy import ensure_is_mock_stream
    from cpen.apps.streams.preprocess.tracks import project_on_track

    df = ensure_is_mock_stream(df)
    if half_widths is None:
        half_widths = np.linspace(0.05, 1.2, 24)

    ra_all = df[ra_col].to_numpy()
    dec_all = df[dec_col].to_numpy()
    is_mock_all = df["is_mock_stream"].to_numpy(bool)
    stream_all = df["stream_label"].astype(str).to_numpy()

    n_mock_tot = 0
    n_other_tot = 0
    mock_abs_d = []
    other_abs_d = []
    for track in tracks.values():
        local = _local_field_mask(ra_all, dec_all, track, search_radius)
        finite = np.isfinite(ra_all) & np.isfinite(dec_all) & local
        ra, dec = ra_all[finite], dec_all[finite]
        is_this = is_mock_all[finite] & (stream_all[finite] == track.stream_label)
        s, d = project_on_track(ra, dec, track)
        on_seg = (s >= track.s_min) & (s <= track.s_max)
        mock_abs_d.append(np.abs(d[is_this & on_seg]))
        other_abs_d.append(np.abs(d[(~is_this) & on_seg]))
        n_mock_tot += int((is_this & on_seg).sum())
        n_other_tot += int(((~is_this) & on_seg).sum())
    mock_abs_d = np.concatenate(mock_abs_d) if mock_abs_d else np.array([])
    other_abs_d = np.concatenate(other_abs_d) if other_abs_d else np.array([])

    rows = []
    for w in half_widths:
        n_m = int((mock_abs_d <= w).sum())
        n_o = int((other_abs_d <= w).sum())
        rows.append({
            "half_width": w,
            "mock_frac": n_m / n_mock_tot if n_mock_tot else np.nan,
            "other_frac": n_o / n_other_tot if n_other_tot else np.nan,
            "purity": n_m / (n_m + n_o) if (n_m + n_o) else np.nan,
        })
    sweep = pd.DataFrame(rows)

    fig, axs = plt.subplots(1, 2, figsize=figsize, dpi=dpi, constrained_layout=True)
    ax = axs[0]
    ax.plot(sweep.other_frac, sweep.mock_frac, "o-", color=color_bg, markersize=4, lw=1.6)
    pick = sweep.iloc[(sweep.half_width - chosen).abs().argmin()]
    ax.scatter([pick.other_frac], [pick.mock_frac], s=50, c=color_mock, zorder=5)
    ax.axvline(pick.other_frac, color=color_mock, ls="--", lw=1.1, label=rf"$w={chosen:.2f}$°")
    ax.set_xlabel("Other kept (in search windows)")
    ax.set_ylabel("Mock kept")
    ax.set_title("Sky tube vs. vertical-style grab")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(fontsize=9)

    ax = axs[1]
    ax.plot(sweep.half_width, sweep.other_frac, color=color_bg, lw=2.0, label="field in tube")
    ax.plot(sweep.half_width, 1.0 - sweep.mock_frac, color=color_mock, lw=1.4, ls="--", label="mock lost")
    ax.axvline(chosen, color=color_mock, ls=":", lw=1.2)
    ax.set_xlabel("tube half-width $w$ (deg)")
    ax.set_ylabel("fraction")
    ax.set_title("Conservative width: keep the stream, not the field")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(fontsize=9)

    fig.suptitle(
        f"Neighborhood grab around {len(tracks)} fitted mocks  "
        f"(windows along each track, not the whole galaxy)",
        fontsize=12,
        fontweight="medium",
    )
    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
        return None
    return fig, axs, sweep


def _gmm_cov_ellipse(mean, cov, n_std=2.0, **kwargs):
    from matplotlib.patches import Ellipse

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2.0 * n_std * np.sqrt(np.clip(vals, 0.0, None))
    return Ellipse(xy=mean, width=width, height=height, angle=theta, **kwargs)


def _gmm_component_colors(n_components, cmap_name=None):
    """Distinct colors for k Gaussians. tab20 repeats after 20; stack tab20b/c for k=32."""
    if cmap_name is not None:
        cmap = plt.get_cmap(cmap_name)
        return [cmap(i % cmap.N) for i in range(n_components)]
    if n_components <= 20:
        cmap = plt.get_cmap("tab20")
        return [cmap(i) for i in range(n_components)]
    colors = []
    for name in ("tab20", "tab20b", "tab20c"):
        cmap = plt.get_cmap(name)
        colors.extend(cmap(i) for i in range(cmap.N))
    if n_components > len(colors):
        hsv = plt.get_cmap("hsv")
        extra = n_components - len(colors)
        colors.extend(hsv(j / extra) for j in range(extra))
    return colors[:n_components]


def _wrapped_ra_plot(ra, dec):
    """Unwrap RA around the sample centroid so cell plots do not jump at 0/360."""
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    m_sky = np.isfinite(ra) & np.isfinite(dec)
    if m_sky.any():
        ra_r = np.deg2rad(ra[m_sky])
        ra_c = np.rad2deg(np.arctan2(np.mean(np.sin(ra_r)), np.mean(np.cos(ra_r))))
        ra_plot = ((ra - ra_c + 180.0) % 360.0 - 180.0) + ra_c
    else:
        ra_plot = ra
    return ra_plot, m_sky


def plot_cell_gmm(
    cell_dfs,
    fits,
    cell_ids,
    *,
    ra_col="ra_des",
    dec_col="dec_des",
    pm_phi_col="pm_phi",
    pm_lam_col="pm_lam",
    n_std=2.0,
    figsize_cell=(6.6, 5.8),
    figsize_host=(11.5, 4.6),
    dpi=150,
    s_comp=2.0,
    s_mock=18.0,
    alpha_comp=0.35,
    max_hosts=8,
    show=True,
    savepath=None,
):
    """Per cell: one PM overview of all Gaussians, then sky|PM for each mock host.

    Overview is the GMM tessellation in \((\mu_\phi,\mu_\lambda)\) with mocks
    overplotted. Each mock-containing Gaussian then gets its own sky (left) and
    PM (right) figure so hosts are not stacked in one panel.
    """
    from matplotlib.lines import Line2D

    from cpen.apps.streams.preprocess.galaxy import ensure_is_mock_stream
    from cpen.apps.streams.preprocess.gmm import GMM_LABEL_COL, GMM_N_COMPONENTS, GMM_WIDTH_COL

    results = []
    for cid in cell_ids:
        if cid not in cell_dfs:
            fig, ax = plt.subplots(figsize=figsize_cell, dpi=dpi)
            ax.text(0.5, 0.5, f"Cell {cid} not in cell_dfs", ha="center", va="center", transform=ax.transAxes)
            results.append((fig, ax, cid, None))
            if show:
                plt.show()
            continue

        df = ensure_is_mock_stream(cell_dfs[cid])
        fit = fits.get(cid) if fits else None
        labels = df[GMM_LABEL_COL].to_numpy(int)
        is_mock = df["is_mock_stream"].to_numpy(bool)
        present = np.unique(labels[labels >= 0])
        k_orig = int(fit.n_components) if fit is not None else int(GMM_N_COMPONENTS)
        n_color = max(k_orig, int(labels.max()) + 1 if (labels >= 0).any() else 1)
        colors = _gmm_component_colors(n_color)
        labeled = labels >= 0
        pm_phi = df[pm_phi_col].to_numpy(float)
        pm_lam = df[pm_lam_col].to_numpy(float)
        ra = df[ra_col].to_numpy(float)
        dec = df[dec_col].to_numpy(float)
        ra_plot, m_sky = _wrapped_ra_plot(ra, dec)

        fig, ax = plt.subplots(figsize=figsize_cell, dpi=dpi)
        for j in present:
            j = int(j)
            m = labeled & (labels == j)
            ax.scatter(
                pm_phi[m], pm_lam[m],
                s=s_comp, c=[colors[j]], alpha=alpha_comp, rasterized=True, linewidths=0, zorder=2,
            )
        if fit is not None and fit.gmm is not None:
            for j in present:
                j = int(j)
                if j >= len(fit.means) or not np.isfinite(fit.means[j]).all():
                    continue
                ell = _gmm_cov_ellipse(
                    fit.means[j], fit.covariances[j], n_std=n_std,
                    facecolor="none", edgecolor=colors[j], lw=1.2, alpha=0.95, zorder=4,
                )
                ax.add_patch(ell)
        if is_mock.any():
            ax.scatter(
                pm_phi[is_mock], pm_lam[is_mock],
                s=s_mock, c="#c0392b", marker="^", alpha=0.9, linewidths=0.3,
                edgecolors="k", zorder=6, label="mock stream",
            )
        fin = labeled & np.isfinite(pm_phi) & np.isfinite(pm_lam)
        if fin.any():
            q = np.nanpercentile(np.column_stack([pm_phi[fin], pm_lam[fin]]), [1, 99], axis=0)
            pad_x = 0.08 * (q[1, 0] - q[0, 0] + 1e-3)
            pad_y = 0.08 * (q[1, 1] - q[0, 1] + 1e-3)
            ax.set_xlim(q[0, 0] - pad_x, q[1, 0] + pad_x)
            ax.set_ylim(q[0, 1] - pad_y, q[1, 1] + pad_y)
        n_mock = int(is_mock.sum())
        n_lab = int(labeled.sum())
        mock_comps = np.unique(labels[is_mock & labeled])
        mock_comps = mock_comps[mock_comps >= 0]
        n_comp_mock = int(mock_comps.size)
        mock_frac = n_mock / n_lab if n_lab else 0.0
        ax.set_xlabel(r"$\mu_\phi$ (mas/yr)")
        ax.set_ylabel(r"$\mu_\lambda$ (mas/yr)")
        ax.set_title(
            f"Cell {cid} — all {len(present)} Gaussians"
            + (f" (k={k_orig} fit)" if k_orig != len(present) else "")
            + f"\n{n_lab:,} labeled, {n_mock} mocks ({100 * mock_frac:.2f}%) "
            + f"in {n_comp_mock} host Gaussian(s)"
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, linestyle=":", alpha=0.4)
        if is_mock.any():
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        fig.tight_layout()
        if savepath is not None:
            sp = Path(savepath)
            fig.savefig(sp.parent / f"{sp.stem}_cell{cid}{sp.suffix}", dpi=dpi, bbox_inches="tight")
        results.append((fig, ax, cid, None))
        if show:
            plt.show()

        hosts = []
        for j in mock_comps:
            j = int(j)
            m = labels == j
            n_m = int((m & is_mock).sum())
            n_s = int(m.sum())
            w = np.nan
            if GMM_WIDTH_COL in df.columns:
                w = float(np.nanmedian(df.loc[m, GMM_WIDTH_COL].to_numpy(float)))
            hosts.append((n_m, n_s, j, w))
        hosts.sort(key=lambda t: (-t[0], -t[1], t[2]))
        if max_hosts is not None and len(hosts) > int(max_hosts):
            print(f"Cell {cid}: {len(hosts)} mock-hosting Gaussians, showing top {int(max_hosts)} by n_mock")
            hosts = hosts[: int(max_hosts)]
        if not hosts:
            print(f"Cell {cid}: no mock-containing Gaussians")
            continue

        for n_m, n_s, j, w in hosts:
            m = labels == j
            color = colors[j]
            fig, axes = plt.subplots(1, 2, figsize=figsize_host, dpi=dpi)

            ax = axes[0]
            ctx = m_sky & ~m
            if ctx.any():
                ax.scatter(
                    ra_plot[ctx], dec[ctx],
                    s=1.2, c="0.75", alpha=0.08, rasterized=True, linewidths=0, zorder=1,
                )
            field_s = m_sky & m & ~is_mock
            mocks_s = m_sky & m & is_mock
            if field_s.any():
                ax.scatter(
                    ra_plot[field_s], dec[field_s],
                    s=s_comp, c=[color], alpha=alpha_comp, rasterized=True, linewidths=0, zorder=2,
                )
            if mocks_s.any():
                ax.scatter(
                    ra_plot[mocks_s], dec[mocks_s],
                    s=s_mock, c=[color], marker="^", alpha=0.95, linewidths=0.35,
                    edgecolors="k", zorder=6,
                )
            ax.legend(
                handles=[
                    Line2D(
                        [0], [0], marker="o", color="w", markerfacecolor=color,
                        markersize=6, linestyle="none", label=f"Gaussian {j} field",
                    ),
                    Line2D(
                        [0], [0], marker="^", color="w", markerfacecolor=color,
                        markeredgecolor="k", markersize=8, linestyle="none", label="mock stream",
                    ),
                ],
                loc="upper right", fontsize=8, framealpha=0.9,
            )
            ax.set_xlabel("RA (deg)")
            ax.set_ylabel("Dec (deg)")
            ax.set_title("Sky")
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True, linestyle=":", alpha=0.4)
            if m_sky.any():
                ax.xaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, _: f"{x + 360:.0f}" if x < 0 else f"{x:.0f}")
                )

            ax = axes[1]
            field = m & ~is_mock
            mocks = m & is_mock
            if field.any():
                ax.scatter(
                    pm_phi[field], pm_lam[field],
                    s=s_comp, c=[color], alpha=alpha_comp, rasterized=True, linewidths=0, zorder=2,
                )
            if mocks.any():
                ax.scatter(
                    pm_phi[mocks], pm_lam[mocks],
                    s=s_mock, c=[color], marker="^", alpha=0.95, linewidths=0.35,
                    edgecolors="k", zorder=6, label="mock stream",
                )
            if fit is not None and fit.gmm is not None and j < len(fit.means) and np.isfinite(fit.means[j]).all():
                ell = _gmm_cov_ellipse(
                    fit.means[j], fit.covariances[j], n_std=n_std,
                    facecolor="none", edgecolor=color, lw=1.4, alpha=0.95, zorder=4,
                )
                ax.add_patch(ell)
            fin_h = m & np.isfinite(pm_phi) & np.isfinite(pm_lam)
            if fin_h.any():
                q = np.nanpercentile(np.column_stack([pm_phi[fin_h], pm_lam[fin_h]]), [2, 98], axis=0)
                pad_x = 0.12 * (q[1, 0] - q[0, 0] + 1e-3)
                pad_y = 0.12 * (q[1, 1] - q[0, 1] + 1e-3)
                ax.set_xlim(q[0, 0] - pad_x, q[1, 0] + pad_x)
                ax.set_ylim(q[0, 1] - pad_y, q[1, 1] + pad_y)
            ax.set_xlabel(r"$\mu_\phi$ (mas/yr)")
            ax.set_ylabel(r"$\mu_\lambda$ (mas/yr)")
            ax.set_title("Proper motion")
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True, linestyle=":", alpha=0.4)
            if mocks.any():
                ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

            frac = n_m / n_s if n_s else 0.0
            wtxt = f",  w={w:.2f} mas/yr" if np.isfinite(w) else ""
            fig.suptitle(
                f"Cell {cid}  ·  Gaussian {j}   {n_s:,} stars, {n_m} mocks ({100 * frac:.2f}%){wtxt}",
                fontsize=12,
                fontweight="medium",
            )
            fig.tight_layout()
            if savepath is not None:
                sp = Path(savepath)
                fig.savefig(sp.parent / f"{sp.stem}_cell{cid}_g{j}{sp.suffix}", dpi=dpi, bbox_inches="tight")
            results.append((fig, axes, cid, j))
            if show:
                plt.show()
    return results if not show else None


def plot_gmm_components(
    cell_dfs,
    cell_ids,
    fits=None,
    *,
    min_mock=1,
    max_per_cell=8,
    ra_col="ra_des",
    dec_col="dec_des",
    pm_phi_col="pm_phi",
    pm_lam_col="pm_lam",
    n_std=2.0,
    figsize=(11.5, 4.6),
    dpi=150,
    s_comp=3.0,
    s_mock=22.0,
    alpha_comp=0.45,
    alpha_context=0.08,
    show_cell_context=True,
    show=True,
    savepath=None,
):
    """One PM|sky figure per GMM Gaussian (not all components overplotted).

    After the width cut, each remaining Gaussian is a separate chunk. Default
    ``min_mock=1`` plots mock-hosting Gaussians only; set 0 to see field chunks.
    Faint gray on the sky is the rest of the cell, for footprint context.
    """
    from matplotlib.lines import Line2D

    from cpen.apps.streams.preprocess.galaxy import ensure_is_mock_stream
    from cpen.apps.streams.preprocess.gmm import GMM_LABEL_COL, GMM_N_COMPONENTS, GMM_WIDTH_COL

    fits = fits or {}
    results = []
    for cid in cell_ids:
        if cid not in cell_dfs:
            continue
        df = ensure_is_mock_stream(cell_dfs[cid])
        labels = df[GMM_LABEL_COL].to_numpy(int)
        is_mock = df["is_mock_stream"].to_numpy(bool)
        fit = fits.get(cid)
        k_orig = int(fit.n_components) if fit is not None else int(GMM_N_COMPONENTS)
        n_color = max(k_orig, int(labels.max()) + 1 if (labels >= 0).any() else 1)
        colors = _gmm_component_colors(n_color)

        rows = []
        for j in np.unique(labels[labels >= 0]):
            j = int(j)
            m = labels == j
            n_m = int((m & is_mock).sum())
            if n_m < min_mock:
                continue
            n_s = int(m.sum())
            w = np.nan
            if GMM_WIDTH_COL in df.columns:
                w = float(np.nanmedian(df.loc[m, GMM_WIDTH_COL].to_numpy(float)))
            rows.append((n_m, n_s, j, w))
        rows.sort(key=lambda t: (-t[0], -t[1], t[2]))
        if max_per_cell is not None:
            rows = rows[: int(max_per_cell)]
        if not rows:
            print(f"Cell {cid}: no Gaussians with ≥{min_mock} mocks")
            continue

        ra = df[ra_col].to_numpy(float)
        dec = df[dec_col].to_numpy(float)
        m_sky = np.isfinite(ra) & np.isfinite(dec)
        if m_sky.any():
            ra_r = np.deg2rad(ra[m_sky])
            ra_c = np.rad2deg(np.arctan2(np.mean(np.sin(ra_r)), np.mean(np.cos(ra_r))))
            ra_plot = ((ra - ra_c + 180.0) % 360.0 - 180.0) + ra_c
        else:
            ra_plot = ra
        pm_phi = df[pm_phi_col].to_numpy(float)
        pm_lam = df[pm_lam_col].to_numpy(float)

        for n_m, n_s, j, w in rows:
            m = labels == j
            fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
            color = colors[j]

            ax = axes[0]
            field = m & ~is_mock
            mocks = m & is_mock
            if field.any():
                ax.scatter(
                    pm_phi[field], pm_lam[field],
                    s=s_comp, c=[color], alpha=alpha_comp, rasterized=True, linewidths=0, zorder=2,
                )
            if mocks.any():
                ax.scatter(
                    pm_phi[mocks], pm_lam[mocks],
                    s=s_mock, c=[color], marker="^", alpha=0.95, linewidths=0.35,
                    edgecolors="k", zorder=6, label="mock stream",
                )
            if fit is not None and fit.gmm is not None and j < len(fit.means) and np.isfinite(fit.means[j]).all():
                ell = _gmm_cov_ellipse(
                    fit.means[j], fit.covariances[j], n_std=n_std,
                    facecolor="none", edgecolor=color, lw=1.4, alpha=0.95, zorder=4,
                )
                ax.add_patch(ell)
            fin = m & np.isfinite(pm_phi) & np.isfinite(pm_lam)
            if fin.any():
                q = np.nanpercentile(np.column_stack([pm_phi[fin], pm_lam[fin]]), [2, 98], axis=0)
                pad_x = 0.12 * (q[1, 0] - q[0, 0] + 1e-3)
                pad_y = 0.12 * (q[1, 1] - q[0, 1] + 1e-3)
                ax.set_xlim(q[0, 0] - pad_x, q[1, 0] + pad_x)
                ax.set_ylim(q[0, 1] - pad_y, q[1, 1] + pad_y)
            ax.set_xlabel(r"$\mu_\phi$ (mas/yr)")
            ax.set_ylabel(r"$\mu_\lambda$ (mas/yr)")
            ax.set_title("Proper motion")
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True, linestyle=":", alpha=0.4)
            if mocks.any():
                ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

            ax = axes[1]
            if show_cell_context and m_sky.any():
                ctx = m_sky & ~m
                if ctx.any():
                    ax.scatter(
                        ra_plot[ctx], dec[ctx],
                        s=1.2, c="0.75", alpha=alpha_context, rasterized=True, linewidths=0, zorder=1,
                    )
            field_s = m_sky & m & ~is_mock
            mocks_s = m_sky & m & is_mock
            if field_s.any():
                ax.scatter(
                    ra_plot[field_s], dec[field_s],
                    s=s_comp, c=[color], alpha=alpha_comp, rasterized=True, linewidths=0, zorder=2,
                )
            if mocks_s.any():
                ax.scatter(
                    ra_plot[mocks_s], dec[mocks_s],
                    s=s_mock, c=[color], marker="^", alpha=0.95, linewidths=0.35,
                    edgecolors="k", zorder=6,
                )
            ax.legend(
                handles=[
                    Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=6, linestyle="none", label=f"Gaussian {j} field"),
                    Line2D([0], [0], marker="^", color="w", markerfacecolor=color, markeredgecolor="k", markersize=8, linestyle="none", label="mock stream"),
                ],
                loc="upper right", fontsize=8, framealpha=0.9,
            )
            ax.set_xlabel("RA (deg)")
            ax.set_ylabel("Dec (deg)")
            ax.set_title("Sky")
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True, linestyle=":", alpha=0.4)
            if m_sky.any():
                def _ra_tick(x):
                    return f"{x + 360:.0f}" if x < 0 else f"{x:.0f}"
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: _ra_tick(x)))

            frac = n_m / n_s if n_s else 0.0
            wtxt = f",  w={w:.2f} mas/yr" if np.isfinite(w) else ""
            fig.suptitle(
                f"Cell {cid}  ·  Gaussian {j}   {n_s:,} stars, {n_m} mocks ({100 * frac:.2f}%){wtxt}",
                fontsize=12,
                fontweight="medium",
            )
            fig.tight_layout()
            if savepath is not None:
                sp = Path(savepath)
                fig.savefig(sp.parent / f"{sp.stem}_cell{cid}_g{j}{sp.suffix}", dpi=dpi, bbox_inches="tight")
            results.append((fig, axes, cid, j))
            if show:
                plt.show()
    return results if not show else None


def plot_gmm_occupancy(
    occ,
    *,
    min_mock=30,
    figsize=(11.0, 4.2),
    dpi=150,
    color_bg="#5a8ab0",
    color_mock="#c0392b",
    show=True,
    savepath=None,
):
    """Component sizes and mock concentration (diagnostic, not a selection)."""
    fig, axs = plt.subplots(1, 3, figsize=figsize, dpi=dpi, constrained_layout=True)

    ax = axs[0]
    sizes = occ.median_comp_size.to_numpy(float)
    ax.hist(sizes[np.isfinite(sizes)], bins=20, color=color_bg, alpha=0.85, edgecolor="white")
    ax.set_xlabel("median stars / component")
    ax.set_ylabel("cells")
    ax.set_title("Chunk size")
    ax.grid(True, linestyle=":", alpha=0.4)

    ax = axs[1]
    rich = occ[occ.n_mock >= min_mock]
    if len(rich):
        ax.hist(
            100.0 * rich.richest_share.to_numpy(float),
            bins=np.linspace(0, 100, 11),
            color=color_mock, alpha=0.85, edgecolor="white",
        )
    ax.set_xlabel("% of cell mocks in one component")
    ax.set_ylabel("cells")
    ax.set_title(rf"Concentration ($\geq${min_mock} mocks)")
    ax.set_xlim(0, 100)
    ax.grid(True, linestyle=":", alpha=0.4)

    ax = axs[2]
    if len(rich):
        ax.hist(
            rich.n_comp_with_mock.to_numpy(float),
            bins=np.arange(0.5, rich.n_comp_with_mock.max() + 1.5, 1.0),
            color=color_mock, alpha=0.85, edgecolor="white",
        )
    ax.set_xlabel("# components with ≥1 mock")
    ax.set_ylabel("cells")
    ax.set_title("Split across Gaussians")
    ax.grid(True, linestyle=":", alpha=0.4)

    fig.suptitle(
        "GMM occupancy (keep every component; mocks are not a cut)",
        fontsize=12,
        fontweight="medium",
    )
    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
        return None
    return fig, axs


_QC_TAG_COLORS = {
    "localized": "#2e7d32",
    "partial": "#0277bd",
    "split": "#c0392b",
    "dilute": "#ef6c00",
    "sparse": "#7f8c8d",
    "empty": "#bdc3c7",
}


def plot_gmm_localization(
    occ,
    *,
    min_mock=30,
    figsize=(12.0, 4.2),
    dpi=150,
    show=True,
    savepath=None,
):
    """Fit QC: concentration, enrichment vs random, mock-rank profile."""
    from cpen.apps.streams.preprocess.gmm import QC_LOCALIZED_ENRICH, QC_LOCALIZED_SHARE

    rich = occ[occ.n_mock >= min_mock].copy()
    fig, axs = plt.subplots(1, 3, figsize=figsize, dpi=dpi, constrained_layout=True)

    ax = axs[0]
    if len(rich):
        tags = rich["qc_tag"].to_numpy(object) if "qc_tag" in rich.columns else np.array(["partial"] * len(rich))
        for tag, color in _QC_TAG_COLORS.items():
            m = tags == tag
            if not m.any():
                continue
            ax.scatter(
                100.0 * rich.loc[m, "richest_share"],
                rich.loc[m, "enrichment"],
                s=12 + 0.04 * rich.loc[m, "n_mock"].to_numpy(float),
                c=color, alpha=0.85, edgecolors="none", label=tag, zorder=3,
            )
    ax.axhline(1.0, color="0.5", ls=":", lw=1.0, label="random (1×)")
    ax.axhline(QC_LOCALIZED_ENRICH, color="#2e7d32", ls="--", lw=1.0, alpha=0.7)
    ax.axvline(100.0 * QC_LOCALIZED_SHARE, color="#2e7d32", ls="--", lw=1.0, alpha=0.7)
    ax.set_xlabel("% of cell mocks in one Gaussian")
    ax.set_ylabel("enrichment vs cell")
    ax.set_title("Did one Gaussian grab the stream?")
    ax.set_xlim(0, 105)
    ax.set_yscale("log")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    ax = axs[1]
    if len(rich) and "n_eff" in rich.columns:
        ax.hist(rich.n_eff.to_numpy(float), bins=np.arange(0.5, 8.5, 0.5),
                color="#c0392b", alpha=0.85, edgecolor="white")
    ax.axvline(1.0, color="0.3", ls=":", lw=1.0)
    ax.set_xlabel("effective # Gaussians  $1/\\sum p_i^2$")
    ax.set_ylabel("cells")
    ax.set_title("1 = all mocks in one blob")
    ax.grid(True, linestyle=":", alpha=0.4)

    ax = axs[2]
    if len(rich) and "mock_rms" in rich.columns:
        m = rich.mock_rms.notna() & rich.other_rms.notna()
        if m.any():
            ax.scatter(
                rich.loc[m, "other_rms"], rich.loc[m, "mock_rms"],
                s=18, c="#c0392b", alpha=0.8, edgecolors="none", zorder=3,
            )
            lo = 0.05
            hi = max(float(rich.loc[m, ["mock_rms", "other_rms"]].max().max()), 1.0)
            ax.plot([lo, hi], [lo, hi], color="0.5", ls=":", lw=1.0)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_xscale("log")
            ax.set_yscale("log")
    ax.set_xlabel("field RMS in host Gaussian (mas/yr)")
    ax.set_ylabel("mock RMS in host Gaussian (mas/yr)")
    ax.set_title("Cold spike inside the blob?")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.4)

    fig.suptitle(
        f"GMM localization QC  ({len(rich)} cells with ≥{min_mock} mocks; not a cut)",
        fontsize=12,
        fontweight="medium",
    )
    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
        return None
    return fig, axs


def plot_gmm_mock_bars(
    cell_dfs,
    occ=None,
    *,
    cell_ids=None,
    n_show=6,
    min_mock=30,
    n_components=None,
    figsize=None,
    dpi=150,
    show=True,
    savepath=None,
):
    """Per-component mock counts for the mock-richest cells (the actual localization picture)."""
    from cpen.apps.streams.preprocess.galaxy import ensure_is_mock_stream
    from cpen.apps.streams.preprocess.gmm import GMM_LABEL_COL, GMM_N_COMPONENTS

    if n_components is None:
        n_components = GMM_N_COMPONENTS

    if cell_ids is None:
        if occ is not None and len(occ):
            ranked = occ[occ.n_mock >= min_mock].sort_values("n_mock", ascending=False)
            cell_ids = [int(x) for x in ranked.cell_id.head(n_show).tolist()]
        else:
            cell_ids = []
            for cid, df in cell_dfs.items():
                n = int(ensure_is_mock_stream(df)["is_mock_stream"].sum())
                if n >= min_mock:
                    cell_ids.append((n, int(cid)))
            cell_ids = [c for _, c in sorted(cell_ids, reverse=True)[:n_show]]

    cell_ids = [c for c in cell_ids if c in cell_dfs][:n_show]
    if not cell_ids:
        print("No cells with enough mocks to plot GMM bars.")
        return None

    n = len(cell_ids)
    if figsize is None:
        figsize = (3.1 * n, 3.4)
    fig, axes = plt.subplots(1, n, figsize=figsize, dpi=dpi, sharey=True, squeeze=False)
    axes = axes[0]
    occ_map = {}
    if occ is not None and len(occ):
        occ_map = {int(r.cell_id): r for r in occ.itertuples()}

    ymax = 1
    for ax, cid in zip(axes, cell_ids):
        df = ensure_is_mock_stream(cell_dfs[cid])
        labels = df[GMM_LABEL_COL].to_numpy(int)
        is_mock = df["is_mock_stream"].to_numpy(bool)
        kmax = n_components
        counts = np.array([(is_mock & (labels == k)).sum() for k in range(kmax)])
        ymax = max(ymax, int(counts.max()) if counts.size else 1)
        colors = ["#c0392b" if c == counts.max() and counts.max() > 0 else "#5a8ab0" for c in counts]
        ax.bar(np.arange(kmax), counts, color=colors, width=0.85, edgecolor="white", linewidth=0.4)
        row = occ_map.get(int(cid))
        title = f"Cell {cid}"
        if row is not None:
            share = getattr(row, "richest_share", np.nan)
            enr = getattr(row, "enrichment", np.nan)
            tag = getattr(row, "qc_tag", "")
            title = f"Cell {cid}  {tag}\n{100 * share:.0f}%  {enr:.1f}×"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("GMM k")
        ax.set_xticks(np.arange(kmax)[::4])
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    axes[0].set_ylabel("# mock stars")
    for ax in axes:
        ax.set_ylim(0, ymax * 1.12)
    fig.suptitle("Mock stars per Gaussian (red = richest) — keep every component", fontsize=11, fontweight="medium")
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
        return None
    return fig, axes


def _gmm_width_hist_bins(widths, n_bins=40):
    w = np.asarray(widths, dtype=float)
    w = w[np.isfinite(w)]
    if w.size == 0:
        return np.linspace(0.0, 5.0, n_bins + 1)
    hi = float(np.nanpercentile(w, 99.5))
    hi = max(hi, 1.0)
    return np.linspace(0.0, hi, n_bins + 1)


def plot_gmm_width_sweep(
    sweep,
    *,
    chosen=None,
    width_mock=None,
    width_other=None,
    figsize=(12.2, 4.2),
    dpi=150,
    color_bg="#5a8ab0",
    color_mock="#c0392b",
    show=True,
    savepath=None,
):
    """Keep Gaussians with width ≤ W. Unique-star efficiencies over the DES footprint.

    ``chosen`` is the candidate threshold to carry to other galaxies. Mock
    keep is optimistic (unsmeared mock PMs); prefer a looser W than the
    mock-optimal point.
    """
    fig, axs = plt.subplots(1, 3, figsize=figsize, dpi=dpi, constrained_layout=True)

    ax = axs[0]
    ax.plot(sweep.other_frac, sweep.mock_frac, "o-", color=color_mock, markersize=4, lw=1.6)
    ax.set_xlabel("Other kept (unique stars)")
    ax.set_ylabel("Mock kept (unique stars)")
    ax.set_title(r"Keep $w_{\mathrm{GMM}}\leq W$")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.4)
    if chosen is not None and len(sweep):
        pick = sweep.iloc[(sweep.threshold - chosen).abs().argmin()]
        ax.axvline(pick.other_frac, color=color_mock, ls="--", lw=1.1,
                   label=rf"$W={pick.threshold:.2f}$ mas/yr")
        ax.scatter([pick.other_frac], [pick.mock_frac], s=50, c=color_mock, zorder=5)
        ax.legend(fontsize=8, loc="lower right")

    ax = axs[1]
    ax.plot(sweep.threshold, sweep.mock_frac, color=color_mock, lw=2.0, label="mock kept")
    ax.plot(sweep.threshold, sweep.other_frac, color=color_bg, lw=2.0, label="other kept")
    ax.set_xlabel(r"width upper bound $W$ (mas/yr)")
    ax.set_ylabel("efficiency")
    ax.set_title("Sky-wide unique-star efficiencies")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.4)
    if chosen is not None:
        ax.axvline(chosen, color=color_mock, ls="--", lw=1.1)
    ax.legend(fontsize=8, loc="best")

    ax = axs[2]
    bins = _gmm_width_hist_bins(
        np.concatenate([
            np.asarray(width_mock, float) if width_mock is not None else np.array([]),
            np.asarray(width_other, float) if width_other is not None else np.array([]),
        ])
    )
    if width_other is not None:
        ax.hist(np.asarray(width_other, float), bins=bins, density=True, alpha=0.75,
                color=color_bg, label="other", edgecolor="white", linewidth=0.4)
    if width_mock is not None:
        ax.hist(np.asarray(width_mock, float), bins=bins, density=True, alpha=0.65,
                color=color_mock, label="mock", edgecolor="white", linewidth=0.4)
    if chosen is not None:
        ax.axvline(chosen, color=color_mock, ls="--", lw=1.1)
    ax.set_xlabel(r"min $w_{\mathrm{GMM}}$ over cells (mas/yr)")
    ax.set_ylabel("density")
    ax.set_title("Unique-star width")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        "GMM width cut  (unique stars; overlapping cells counted once)",
        fontsize=12,
        fontweight="medium",
    )
    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
        return None
    return fig, axs


def plot_train_mock_fractions(
    tbl,
    *,
    min_mock=100,
    width_max=2.0,
    figsize=(8.5, 6.5),
    dpi=150,
    color_before="#5a8ab0",
    color_after="#c0392b",
    show=True,
    savepath=None,
):
    """Mock fraction before vs after \(w\le W\).

    ``min_mock`` only filters the *plot* to cells that host a substantial
    leftover mock count; it is not a training gate.
    """
    train = tbl[tbl.train_gate].sort_values("mock_frac_after", ascending=True)
    if not len(train):
        print(f"No cells with ≥{min_mock} mocks.")
        return None

    y = np.arange(len(train))
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)
    ax.barh(y, 100.0 * train.mock_frac.to_numpy(float), height=0.4,
            color=color_before, alpha=0.7, label="before \(w\) cut")
    ax.barh(y + 0.4, 100.0 * train.mock_frac_after.to_numpy(float), height=0.4,
            color=color_after, alpha=0.9, label=rf"after $w\leq{width_max}$")
    ax.set_yticks(y + 0.2)
    ax.set_yticklabels([str(int(c)) for c in train.cell_id], fontsize=8)
    ax.set_xlabel("mock fraction (%)")
    ax.set_ylabel("DES cell")
    ax.set_title(
        rf"Cells with $\geq${min_mock} mocks (occupancy, not a cut): mock fraction after dropping $w>{width_max}$"
    )
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    ax.legend(fontsize=9, loc="lower right")
    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
        return None
    return fig, ax


def plot_blob_knn(
    df,
    knn,
    *,
    hyper=None,
    title=None,
    ra_col="ra_des",
    dec_col="dec_des",
    max_edges=2500,
    figsize=(6.4, 5.6),
    dpi=150,
    s_field=4.0,
    s_mock=18.0,
    show=True,
    savepath=None,
    ax=None,
):
    """Sky view of one blob: kNN plus sky over/underdensities and PM-peak centers."""
    from matplotlib.collections import LineCollection, PatchCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon

    from cpen.apps.streams.preprocess.galaxy import ensure_is_mock_stream

    df = ensure_is_mock_stream(df)
    ra = df[ra_col].to_numpy(float)
    dec = df[dec_col].to_numpy(float)
    is_mock = df["is_mock_stream"].to_numpy(bool)
    m_sky = np.isfinite(ra) & np.isfinite(dec)
    if m_sky.any():
        ra_r = np.deg2rad(ra[m_sky])
        ra_c = np.rad2deg(np.arctan2(np.mean(np.sin(ra_r)), np.mean(np.cos(ra_r))))
        ra_plot = ((ra - ra_c + 180.0) % 360.0 - 180.0) + ra_c
    else:
        ra_plot = ra

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)
    else:
        fig = ax.figure
    n_e = knn.n_edges
    if n_e:
        rng = np.random.default_rng(0)
        take = np.arange(n_e) if n_e <= max_edges else rng.choice(n_e, max_edges, replace=False)
        segs = np.stack(
            [
                np.column_stack([ra_plot[knn.src[take]], dec[knn.src[take]]]),
                np.column_stack([ra_plot[knn.dst[take]], dec[knn.dst[take]]]),
            ],
            axis=1,
        )
        ax.add_collection(LineCollection(segs, colors="0.55", linewidths=0.35, alpha=0.35, zorder=1))

    field = m_sky & ~is_mock
    mocks = m_sky & is_mock
    if field.any():
        ax.scatter(ra_plot[field], dec[field], s=s_field, c=_FIELD_SCATTER, alpha=0.45, rasterized=True, linewidths=0, zorder=2)
    if mocks.any():
        ax.scatter(
            ra_plot[mocks], dec[mocks], s=s_mock, c="#c0392b", marker="^",
            alpha=0.95, linewidths=0.3, edgecolors="k", zorder=4, label="mock stream",
        )

    if hyper is not None and hyper.n_edges:
        from cpen.apps.streams.preprocess.blob_graph import (
            KIND_PM_PEAK, KIND_SKY_BG, KIND_SKY_PEAK,
        )
        hulls = []
        hull_colors = []
        spoke_segs = []
        spoke_cols = []
        kinds = getattr(hyper, "kind", np.zeros(hyper.n_edges, dtype=np.int8))
        if kinds.size != hyper.n_edges:
            kinds = np.zeros(hyper.n_edges, dtype=np.int8)
        for e, mem in enumerate(hyper.members):
            knd = int(kinds[e])
            if knd == KIND_SKY_PEAK:
                col, marker, draw_hull, draw_spokes = _HYPER_OVER, "o", True, True
            elif knd == KIND_SKY_BG:
                col, marker, draw_hull, draw_spokes = _HYPER_UNDER, "o", True, True
            elif knd == KIND_PM_PEAK:
                col, marker, draw_hull, draw_spokes = _HYPER_OVER, "s", False, False
            else:
                continue
            c = int(hyper.centers[e])
            pts = np.column_stack([ra_plot[mem], dec[mem]])
            finite_pts = pts[np.isfinite(pts).all(axis=1)]
            if draw_hull and finite_pts.shape[0] >= 3:
                try:
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(finite_pts)
                    hulls.append(Polygon(finite_pts[hull.vertices], closed=True))
                    hull_colors.append(col)
                except Exception:
                    pass
            if np.isfinite(ra_plot[c]) and np.isfinite(dec[c]):
                if draw_spokes:
                    for j in mem:
                        if j == c:
                            continue
                        if not (np.isfinite(ra_plot[j]) and np.isfinite(dec[j])):
                            continue
                        spoke_segs.append([[ra_plot[c], dec[c]], [ra_plot[j], dec[j]]])
                        spoke_cols.append(col)
                ax.scatter(
                    ra_plot[c], dec[c],
                    s=55,
                    marker=marker,
                    facecolors="none", edgecolors=col,
                    linewidths=1.4, zorder=5,
                )
        if hulls:
            ax.add_collection(PatchCollection(
                hulls, facecolors=hull_colors, edgecolors=hull_colors,
                alpha=0.12, linewidths=0.8, zorder=0,
            ))
        if spoke_segs:
            ax.add_collection(LineCollection(spoke_segs, colors=spoke_cols, linewidths=0.55, alpha=0.55, zorder=3))

    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.4)
    if title is None:
        title = f"sky kNN  k={knn.k}  N={knn.n_nodes:,}  E={knn.n_edges:,}  ⟨deg⟩={knn.mean_degree:.1f}"
        if hyper is not None:
            title += f"  H={hyper.n_edges} (sky {getattr(hyper, 'n_sky', 0)} / PM {getattr(hyper, 'n_pm', 0)})"
    ax.set_title(title)
    handles = []
    if mocks.any():
        handles.append(Line2D([0], [0], marker="^", color="w", markerfacecolor="#c0392b",
                              markeredgecolor="k", markersize=8, linestyle="none", label="mock stream"))
    if hyper is not None and hyper.n_edges:
        handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                              markeredgecolor=_HYPER_OVER, markersize=8, linestyle="none", label="sky overdensity"))
        handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                              markeredgecolor=_HYPER_UNDER, markersize=8, linestyle="none", label="sky underdensity"))
        if getattr(hyper, "n_pm_peak", 0):
            handles.append(Line2D([0], [0], marker="s", color="w", markerfacecolor="none",
                                  markeredgecolor=_HYPER_OVER, markersize=8, linestyle="none", label="PM peak (center)"))
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    if own_fig:
        if savepath is not None:
            fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
            return None
        return fig, ax
    return ax


def plot_blob_pm(
    df,
    hyper,
    *,
    title=None,
    pm_phi_col="pm_phi",
    pm_lam_col="pm_lam",
    figsize=(6.4, 5.6),
    dpi=150,
    s_field=4.0,
    s_mock=18.0,
    show=True,
    savepath=None,
    ax=None,
):
    """Proper-motion view of one blob with PM-peak hyperedges."""
    from matplotlib.collections import LineCollection, PatchCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon

    from cpen.apps.streams.preprocess.blob_graph import KIND_PM_PEAK
    from cpen.apps.streams.preprocess.galaxy import ensure_is_mock_stream

    df = ensure_is_mock_stream(df)
    px = df[pm_phi_col].to_numpy(float)
    py = df[pm_lam_col].to_numpy(float)
    is_mock = df["is_mock_stream"].to_numpy(bool)
    m_ok = np.isfinite(px) & np.isfinite(py)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)
    else:
        fig = ax.figure
    field = m_ok & ~is_mock
    mocks = m_ok & is_mock
    if field.any():
        ax.scatter(px[field], py[field], s=s_field, c=_FIELD_SCATTER, alpha=0.45, rasterized=True, linewidths=0, zorder=2)
    if mocks.any():
        ax.scatter(
            px[mocks], py[mocks], s=s_mock, c="#c0392b", marker="^",
            alpha=0.95, linewidths=0.3, edgecolors="k", zorder=4,
        )

    if hyper is not None and hyper.n_edges:
        kinds = getattr(hyper, "kind", np.zeros(hyper.n_edges, dtype=np.int8))
        hulls, hull_colors, spoke_segs, spoke_cols = [], [], [], []
        for e, mem in enumerate(hyper.members):
            knd = int(kinds[e]) if kinds.size == hyper.n_edges else KIND_PM_PEAK
            if knd != KIND_PM_PEAK:
                continue
            c = int(hyper.centers[e])
            col = _HYPER_OVER
            pts = np.column_stack([px[mem], py[mem]])
            finite_pts = pts[np.isfinite(pts).all(axis=1)]
            if finite_pts.shape[0] >= 3:
                try:
                    from scipy.spatial import ConvexHull
                    hull = ConvexHull(finite_pts)
                    hulls.append(Polygon(finite_pts[hull.vertices], closed=True))
                    hull_colors.append(col)
                except Exception:
                    pass
            if np.isfinite(px[c]) and np.isfinite(py[c]):
                for j in mem:
                    if j == c or not (np.isfinite(px[j]) and np.isfinite(py[j])):
                        continue
                    spoke_segs.append([[px[c], py[c]], [px[j], py[j]]])
                    spoke_cols.append(col)
                ax.scatter(
                    px[c], py[c], s=55, marker="s",
                    facecolors="none", edgecolors=col, linewidths=1.4, zorder=5,
                )
        if hulls:
            ax.add_collection(PatchCollection(
                hulls, facecolors=hull_colors, edgecolors=hull_colors,
                alpha=0.12, linewidths=0.8, zorder=0,
            ))
        if spoke_segs:
            ax.add_collection(LineCollection(spoke_segs, colors=spoke_cols, linewidths=0.55, alpha=0.55, zorder=3))

    ax.set_xlabel(r"$\mu_\phi$ (mas/yr)")
    ax.set_ylabel(r"$\mu_\lambda$ (mas/yr)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.4)
    if title is None:
        n_pm = int(getattr(hyper, "n_pm", 0) or 0) if hyper is not None else 0
        n_pm_pk = int(getattr(hyper, "n_pm_peak", 0) or 0) if hyper is not None else 0
        title = f"PM hypers  H_PM={n_pm}  peaks={n_pm_pk}"
    ax.set_title(title)
    handles = []
    if mocks.any():
        handles.append(Line2D([0], [0], marker="^", color="w", markerfacecolor="#c0392b",
                              markeredgecolor="k", markersize=8, linestyle="none", label="mock stream"))
    handles.append(Line2D([0], [0], marker="s", color="w", markerfacecolor="none",
                          markeredgecolor=_HYPER_OVER, markersize=8, linestyle="none", label="PM peak"))
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    if own_fig:
        if savepath is not None:
            fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
            return None
        return fig, ax
    return ax


def plot_blob_sky_pm(
    df,
    knn,
    hyper,
    *,
    title=None,
    show=True,
    savepath=None,
    figsize=(12.2, 5.4),
    dpi=150,
):
    """Sky kNN + PM hyperedges for one blob, side by side."""
    fig, axs = plt.subplots(1, 2, figsize=figsize, dpi=dpi, constrained_layout=True)
    plot_blob_knn(df, knn, hyper=hyper, ax=axs[0], show=False, title="sky")
    plot_blob_pm(df, hyper, ax=axs[1], show=False, title="PM")
    if title:
        fig.suptitle(title, fontsize=11, fontweight="medium")
    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
        return None
    return fig, axs


def plot_blob_examples(
    blobs,
    knns,
    hypers,
    example_tab,
    *,
    show=True,
    max_n_stars=25_000,
):
    """Sky+PM QC plots for the rows returned by ``pick_blob_examples``."""
    from cpen.apps.streams.preprocess.blob_graph import hyper_edge_y, knn_edge_y
    from cpen.apps.streams.preprocess.galaxy import ensure_is_mock_stream

    if example_tab is None or not len(example_tab):
        print("No blob examples to plot.")
        return
    n_plotted = 0
    for row in example_tab.itertuples(index=False):
        part = int(getattr(row, "blob_part", 0))
        key = (int(row.cell_id), int(row.gmm_label), part)
        if key not in blobs:
            key2 = (int(row.cell_id), int(row.gmm_label))
            if key2 in blobs:
                key = key2
            else:
                print(f"skip missing blob {key}")
                continue
        df_b = ensure_is_mock_stream(blobs[key])
        if len(df_b) > int(max_n_stars):
            print(f"skip huge blob {key}  N={len(df_b):,}")
            continue
        knn = knns.get(key)
        hyp = hypers.get(key)
        if knn is None or hyp is None:
            print(f"skip {key}: missing knn/hyper")
            continue
        m = df_b["is_mock_stream"].to_numpy(bool)
        n_e_pos = int(knn_edge_y(knn, m).sum()) if knn.n_edges else 0
        n_h_pos = int(hyper_edge_y(hyp, m).sum()) if hyp.n_edges else 0
        w_med = float(df_b["gmm_width"].median()) if "gmm_width" in df_b.columns else float("nan")
        role = getattr(row, "role", "")
        part_txt = f".{part}" if part else ""
        plot_blob_sky_pm(
            df_b,
            knn,
            hyp,
            title=(
                f"{role}  cell {key[0]}  Gaussian {key[1]}{part_txt}  w={w_med:.2f}  "
                f"N={knn.n_nodes:,}  mocks={int(m.sum())}  "
                f"E={knn.n_edges:,} ({n_e_pos}+)  "
                f"H={hyp.n_edges} (sky {hyp.n_sky} / PM {hyp.n_pm}, {n_h_pos}+)"
            ),
            show=show,
        )
        n_plotted += 1
    print(f"Plotted {n_plotted} blob examples (sky + PM).")

