"""Sky kNN 2-edges and hyperedges on one GMM blob.

Each remaining ``(cell, Gaussian[, part])`` is a graph (oversized Gaussians
are sky-split first), ready for CPEN node+edge training.

* **kNN** — undirected 2-edges on the sky (local cosine-dec plane).
* **Sky hyperedges** — overdensity peaks + underdensity k-balls in that sky plane.
* **PM hyperedges** — overdensity peaks only in \((\mu_\phi, \mu_\lambda)\) (the GMM
  already grouped the blob; these pick colder patches inside it).

Truth labels are training-only: ``edge_y`` is both endpoints mock; ``hyper_y`` is
member mock-fraction ≥ ``BLOB_HYPER_PURITY``. Do not use labels to *build* edges.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from cpen.apps.streams.preprocess.config import TRAIN_VARS
from cpen.apps.streams.preprocess.galaxy import ensure_is_mock_stream
from cpen.apps.streams.preprocess.utils import blob_key_ids, blob_stem

BLOB_KNN_K = 8
BLOB_HYPER_N = 64
BLOB_HYPER_K = 16
BLOB_HYPER_CONTRAST = 1.1
BLOB_HYPER_BG_MIN = 8
BLOB_HYPER_PURITY = 0.5
BLOB_TRAIN_BLOB_MOCK = 30
BLOB_TRAIN_CELL_MOCK = 100
KIND_SKY_PEAK = 0
KIND_SKY_BG = 1
KIND_PM_PEAK = 2
KIND_PM_BG = 3
KIND_PEAK = KIND_SKY_PEAK
KIND_BG = KIND_SKY_BG
_PEAK_KINDS = (KIND_SKY_PEAK, KIND_PM_PEAK)
_BG_KINDS = (KIND_SKY_BG, KIND_PM_BG)
_SKY_KINDS = (KIND_SKY_PEAK, KIND_SKY_BG)
_PM_KINDS = (KIND_PM_PEAK, KIND_PM_BG)


@dataclass
class BlobKNN:
    """Undirected kNN on a blob. ``src``/``dst`` are 0..n-1 positions in the blob frame."""

    src: np.ndarray
    dst: np.ndarray
    dist_deg: np.ndarray
    n_nodes: int
    k: int
    keep: np.ndarray

    @property
    def n_edges(self) -> int:
        return int(self.src.size)

    @property
    def mean_degree(self) -> float:
        if self.n_nodes == 0:
            return 0.0
        return 2.0 * self.n_edges / self.n_nodes


@dataclass
class BlobHyper:
    """Hyperedges on a blob. ``members[e]`` are blob-frame indices.

    ``kind`` is sky/PM × peak/background. ``n_hyper`` is the per-space peak cap.
    """

    centers: np.ndarray
    members: list[np.ndarray] = field(default_factory=list)
    r_k: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    kind: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    n_nodes: int = 0
    n_hyper: int = 0
    k_ball: int = 0
    contrast: float = BLOB_HYPER_CONTRAST
    keep: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    @property
    def n_edges(self) -> int:
        return len(self.members)

    @property
    def n_peak(self) -> int:
        return int(np.isin(self.kind, _PEAK_KINDS).sum()) if self.kind.size else 0

    @property
    def n_bg(self) -> int:
        return int(np.isin(self.kind, _BG_KINDS).sum()) if self.kind.size else 0

    @property
    def n_sky(self) -> int:
        return int(np.isin(self.kind, _SKY_KINDS).sum()) if self.kind.size else 0

    @property
    def n_pm(self) -> int:
        return int(np.isin(self.kind, _PM_KINDS).sum()) if self.kind.size else 0

    @property
    def n_sky_peak(self) -> int:
        return int((self.kind == KIND_SKY_PEAK).sum()) if self.kind.size else 0

    @property
    def n_pm_peak(self) -> int:
        return int((self.kind == KIND_PM_PEAK).sum()) if self.kind.size else 0

    @property
    def sizes(self) -> np.ndarray:
        return np.asarray([m.size for m in self.members], dtype=np.int32)

    @property
    def density(self) -> np.ndarray:
        r = np.asarray(self.r_k, dtype=float)
        out = np.full(r.shape, np.nan)
        ok = np.isfinite(r) & (r > 0)
        out[ok] = 1.0 / np.square(r[ok])
        return out


def _sky_xy(ra, dec) -> tuple[np.ndarray, float, float]:
    """Local plane: \(x = \Delta\alpha\cos\delta_0\), \(y = \Delta\delta\) (deg)."""
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    finite = np.isfinite(ra) & np.isfinite(dec)
    if not finite.any():
        return np.full((ra.size, 2), np.nan), np.nan, np.nan
    ra_r = np.deg2rad(ra[finite])
    ra0 = float(np.rad2deg(np.arctan2(np.mean(np.sin(ra_r)), np.mean(np.cos(ra_r)))))
    dec0 = float(np.mean(dec[finite]))
    dra = (ra - ra0 + 180.0) % 360.0 - 180.0
    xy = np.column_stack([dra * np.cos(np.deg2rad(dec0)), dec - dec0])
    return xy, ra0, dec0


def _fit_xy_nn(xy, k: int):
    """kd-tree on rows of ``xy`` (n, 2). ``dist``/``neigh`` include self at column 0."""
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy must be (n, 2); got {xy.shape}")
    keep = np.isfinite(xy).all(axis=1)
    idx_keep = np.flatnonzero(keep)
    n_fit = int(idx_keep.size)
    k_use = min(int(k), max(n_fit - 1, 0))
    if k_use < 1:
        return keep, idx_keep, k_use, None, None
    nn = NearestNeighbors(n_neighbors=k_use + 1, algorithm="kd_tree")
    nn.fit(xy[keep])
    dist, neigh = nn.kneighbors(xy[keep])
    return keep, idx_keep, k_use, dist, neigh


def _fit_sky_nn(ra, dec, k: int):
    """kd-tree on the local sky plane. ``dist``/``neigh`` include self at column 0."""
    xy, _, _ = _sky_xy(ra, dec)
    return _fit_xy_nn(xy, k)


def sky_knn_edges(ra, dec, k: int = BLOB_KNN_K) -> BlobKNN:
    """Undirected kNN among finite (RA, Dec) rows. Isolated / non-finite stars are omitted."""
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    n = ra.size
    keep, idx_keep, k_use, dist, neigh = _fit_sky_nn(ra, dec, k)
    empty = BlobKNN(
        src=np.zeros(0, dtype=np.int32),
        dst=np.zeros(0, dtype=np.int32),
        dist_deg=np.zeros(0, dtype=np.float64),
        n_nodes=n,
        k=int(k),
        keep=keep,
    )
    if k_use < 1 or dist is None:
        return empty

    n_fit = int(idx_keep.size)
    src_local = np.repeat(np.arange(n_fit), k_use)
    dst_local = neigh[:, 1:].reshape(-1)
    d = dist[:, 1:].reshape(-1)
    a = np.minimum(src_local, dst_local)
    b = np.maximum(src_local, dst_local)
    undirected = np.stack([a, b], axis=1)
    _, uniq = np.unique(undirected, axis=0, return_index=True)
    uniq = np.sort(uniq)
    src = idx_keep[undirected[uniq, 0]].astype(np.int32)
    dst = idx_keep[undirected[uniq, 1]].astype(np.int32)
    return BlobKNN(
        src=src,
        dst=dst,
        dist_deg=d[uniq].astype(np.float64),
        n_nodes=n,
        k=k_use,
        keep=keep,
    )


def blob_knn_edges(
    df: pd.DataFrame,
    k: int = BLOB_KNN_K,
    ra_col: str = "ra_des",
    dec_col: str = "dec_des",
) -> BlobKNN:
    """kNN on one GMM blob dataframe (row order = node index)."""
    return sky_knn_edges(df[ra_col].to_numpy(), df[dec_col].to_numpy(), k=k)


def build_blob_knns(
    blobs: dict,
    k: int = BLOB_KNN_K,
    ra_col: str = "ra_des",
    dec_col: str = "dec_des",
) -> dict[tuple[int, int], BlobKNN]:
    """Sky kNN for every ``(cell_id, gmm_label)`` blob."""
    return {
        key: blob_knn_edges(df, k=k, ra_col=ra_col, dec_col=dec_col)
        for key, df in blobs.items()
    }


def print_blob_knn_summary(graphs: dict, blobs: dict | None = None, n_show: int = 8):
    """Node / edge stats for ``build_blob_knns`` output."""
    if not graphs:
        print("No blob graphs.")
        return
    rows = []
    for key, knn in graphs.items():
        cid, j, part = blob_key_ids(key)
        n_mm = n_mf = n_mock = np.nan
        if blobs is not None and key in blobs and "is_mock_stream" in blobs[key].columns:
            m = blobs[key]["is_mock_stream"].to_numpy(bool)
            n_mock = int(m.sum())
            if knn.n_edges:
                src_m = m[knn.src]
                dst_m = m[knn.dst]
                n_mm = int((src_m & dst_m).sum())
                n_mf = int((src_m != dst_m).sum())
        rows.append({
            "cell_id": cid, "gmm_label": j, "blob_part": part,
            "n_nodes": knn.n_nodes, "n_edges": knn.n_edges,
            "mean_degree": knn.mean_degree, "k": knn.k,
            "n_mock": n_mock,
            "n_mock_mock_edges": n_mm,
            "n_mock_field_edges": n_mf,
        })
    tab = pd.DataFrame(rows)
    k_typ = int(tab.k.mode().iloc[0]) if len(tab) else 0
    print(f"Blob kNN: {len(tab):,} graphs  k={k_typ}")
    print(
        f"  N: median {tab.n_nodes.median():,.0f}  "
        f"E: median {tab.n_edges.median():,.0f}  "
        f"⟨deg⟩ median {tab.mean_degree.median():.1f}"
    )
    has_mock = tab["n_mock"].notna() & (tab["n_mock"] > 0)
    if has_mock.any():
        sub = tab[has_mock]
        mm = sub["n_mock_mock_edges"].to_numpy(float)
        mf = sub["n_mock_field_edges"].to_numpy(float)
        den = mm + mf
        frac = np.divide(mm, den, out=np.full_like(mm, np.nan), where=den > 0)
        print(
            f"  blobs with mocks: {int(has_mock.sum()):,}  "
            f"median mock–mock / (mock–mock + mock–field) = {np.nanmedian(frac):.2f}"
        )
    print("  largest graphs:")
    print(tab.sort_values("n_nodes", ascending=False).head(n_show).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if has_mock.any():
        print("  mock-richest graphs:")
        print(tab.sort_values(["n_mock", "n_nodes"], ascending=False).head(n_show).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return tab


def _empty_hyper(n: int, keep: np.ndarray, n_hyper: int, k_ball: int, contrast: float) -> BlobHyper:
    return BlobHyper(
        centers=np.zeros(0, dtype=np.int32),
        members=[],
        r_k=np.zeros(0, dtype=np.float64),
        kind=np.zeros(0, dtype=np.int8),
        n_nodes=n,
        n_hyper=int(n_hyper),
        k_ball=int(k_ball),
        contrast=float(contrast),
        keep=keep,
    )


def _nms_choose(order, neigh, banned: np.ndarray, n_take: int) -> list[int]:
    chosen: list[int] = []
    n_take = int(n_take)
    if n_take < 1:
        return chosen
    for i in order:
        i = int(i)
        if banned[i]:
            continue
        chosen.append(i)
        banned[neigh[i]] = True
        if len(chosen) >= n_take:
            break
    return chosen


def _hyper_from_local(
    idx_keep: np.ndarray,
    neigh: np.ndarray,
    r_k: np.ndarray,
    chosen: list[int],
    kind: np.ndarray,
    *,
    n: int,
    keep: np.ndarray,
    n_hyper: int,
    k_ball: int,
    contrast: float,
) -> BlobHyper:
    if not chosen:
        return _empty_hyper(n, keep, n_hyper, k_ball, contrast)
    loc = np.asarray(chosen, dtype=np.int32)
    return BlobHyper(
        centers=idx_keep[loc].astype(np.int32),
        members=[idx_keep[neigh[i]].astype(np.int32) for i in loc],
        r_k=r_k[loc].astype(np.float64),
        kind=np.asarray(kind, dtype=np.int8),
        n_nodes=n,
        n_hyper=int(n_hyper),
        k_ball=int(k_ball),
        contrast=float(contrast),
        keep=keep,
    )


def _density_hyperedges_xy(
    xy,
    n_hyper: int,
    k_ball: int,
    contrast: float,
    n_bg: int | None,
    bg_min: int,
    kind_peak: int,
    kind_bg: int,
) -> BlobHyper:
    """Peak + background k-balls in a 2-d feature plane (sky or PM)."""
    xy = np.asarray(xy, dtype=float)
    n = xy.shape[0]
    keep, idx_keep, k_use, dist, neigh = _fit_xy_nn(xy, k_ball)
    empty = _empty_hyper(n, keep, n_hyper, k_ball, contrast)
    if k_use < 2 or dist is None:
        return empty

    r_k = dist[:, -1].astype(np.float64)
    neigh_wo = neigh[:, 1:]
    is_peak = r_k <= r_k[neigh_wo].min(axis=1)
    med = float(np.median(r_k))
    if np.isfinite(med) and med > 0 and contrast > 0:
        is_peak &= r_k * float(contrast) <= med
    peaks = np.flatnonzero(is_peak)
    banned = np.zeros(int(idx_keep.size), dtype=bool)
    peak_order = peaks[np.argsort(r_k[peaks])] if peaks.size else np.zeros(0, dtype=int)
    chosen_peak = _nms_choose(peak_order, neigh, banned, n_hyper)

    n_bg_use = int(len(chosen_peak) if n_bg is None else n_bg)
    n_bg_use = max(n_bg_use, int(bg_min))
    chosen_bg = _nms_choose(np.argsort(-r_k), neigh, banned, n_bg_use)

    chosen = chosen_peak + chosen_bg
    kind = np.concatenate([
        np.full(len(chosen_peak), int(kind_peak), dtype=np.int8),
        np.full(len(chosen_bg), int(kind_bg), dtype=np.int8),
    ]) if chosen else np.zeros(0, dtype=np.int8)
    return _hyper_from_local(
        idx_keep, neigh, r_k, chosen, kind,
        n=n, keep=keep, n_hyper=n_hyper, k_ball=k_use, contrast=contrast,
    )


def concat_blob_hypers(*parts: BlobHyper) -> BlobHyper:
    """Stack hyperedge lists that share a blob frame."""
    parts = tuple(p for p in parts if p.n_edges)
    if not parts:
        return _empty_hyper(0, np.zeros(0, dtype=bool), 0, 0, BLOB_HYPER_CONTRAST)
    head = parts[0]
    if len(parts) == 1:
        return head
    n_nodes = int(head.n_nodes)
    members: list[np.ndarray] = []
    centers = []
    r_k = []
    kind = []
    keep = head.keep
    for p in parts:
        if int(p.n_nodes) != n_nodes:
            raise ValueError("concat_blob_hypers: n_nodes mismatch")
        members.extend(p.members)
        centers.append(p.centers)
        r_k.append(p.r_k)
        kind.append(p.kind)
    return BlobHyper(
        centers=np.concatenate(centers).astype(np.int32),
        members=members,
        r_k=np.concatenate(r_k).astype(np.float64),
        kind=np.concatenate(kind).astype(np.int8),
        n_nodes=n_nodes,
        n_hyper=int(head.n_hyper),
        k_ball=int(head.k_ball),
        contrast=float(head.contrast),
        keep=keep,
    )


def sky_density_hyperedges(
    ra,
    dec,
    n_hyper: int | None = None,
    k_ball: int | None = None,
    contrast: float | None = None,
) -> BlobHyper:
    """Sky peak hyperedges only (no background sample)."""
    return sky_blob_hyperedges(
        ra, dec, n_hyper=n_hyper, k_ball=k_ball, contrast=contrast, n_bg=0, bg_min=0,
    )


def sky_blob_hyperedges(
    ra,
    dec,
    n_hyper: int | None = None,
    k_ball: int | None = None,
    contrast: float | None = None,
    n_bg: int | None = None,
    bg_min: int | None = None,
) -> BlobHyper:
    """Sky-plane density peaks plus matched least-dense background k-balls."""
    xy, _, _ = _sky_xy(ra, dec)
    n_hyper = int(BLOB_HYPER_N if n_hyper is None else n_hyper)
    k_ball = int(BLOB_HYPER_K if k_ball is None else k_ball)
    contrast = float(BLOB_HYPER_CONTRAST if contrast is None else contrast)
    bg_min = int(BLOB_HYPER_BG_MIN if bg_min is None else bg_min)
    return _density_hyperedges_xy(
        xy, n_hyper, k_ball, contrast, n_bg, bg_min, KIND_SKY_PEAK, KIND_SKY_BG,
    )


def pm_blob_hyperedges(
    pm_phi,
    pm_lam,
    n_hyper: int | None = None,
    k_ball: int | None = None,
    contrast: float | None = None,
    n_bg: int | None = 0,
    bg_min: int | None = 0,
) -> BlobHyper:
    """Proper-motion density peaks only (no underdensity balls). Units mas/yr."""
    xy = np.column_stack([np.asarray(pm_phi, dtype=float), np.asarray(pm_lam, dtype=float)])
    n_hyper = int(BLOB_HYPER_N if n_hyper is None else n_hyper)
    k_ball = int(BLOB_HYPER_K if k_ball is None else k_ball)
    contrast = float(BLOB_HYPER_CONTRAST if contrast is None else contrast)
    n_bg = 0 if n_bg is None else int(n_bg)
    bg_min = 0 if bg_min is None else int(bg_min)
    return _density_hyperedges_xy(
        xy, n_hyper, k_ball, contrast, n_bg, bg_min, KIND_PM_PEAK, KIND_PM_BG,
    )


def blob_hyperedges(
    df: pd.DataFrame,
    n_hyper: int | None = None,
    k_ball: int | None = None,
    contrast: float | None = None,
    n_bg: int | None = None,
    bg_min: int | None = None,
    ra_col: str = "ra_des",
    dec_col: str = "dec_des",
    pm_phi_col: str = "pm_phi",
    pm_lam_col: str = "pm_lam",
    include_sky: bool = True,
    include_pm: bool = True,
) -> BlobHyper:
    """Sky over/underdensity hyperedges and/or PM peaks on one blob (row order = node index).

    Sky uses ``n_bg`` / ``bg_min``. PM peaks are constructed with no background
    balls regardless of those knobs.
    """
    parts: list[BlobHyper] = []
    sky_kw = dict(n_hyper=n_hyper, k_ball=k_ball, contrast=contrast, n_bg=n_bg, bg_min=bg_min)
    if include_sky:
        parts.append(sky_blob_hyperedges(df[ra_col].to_numpy(), df[dec_col].to_numpy(), **sky_kw))
    if include_pm:
        if pm_phi_col not in df.columns or pm_lam_col not in df.columns:
            raise KeyError(f"PM hyperedges need {pm_phi_col}/{pm_lam_col}")
        parts.append(pm_blob_hyperedges(
            df[pm_phi_col].to_numpy(), df[pm_lam_col].to_numpy(),
            n_hyper=n_hyper, k_ball=k_ball, contrast=contrast, n_bg=0, bg_min=0,
        ))
    if not parts:
        n = len(df)
        return _empty_hyper(n, np.ones(n, dtype=bool), int(BLOB_HYPER_N if n_hyper is None else n_hyper),
                            int(BLOB_HYPER_K if k_ball is None else k_ball),
                            float(BLOB_HYPER_CONTRAST if contrast is None else contrast))
    out = concat_blob_hypers(*parts)
    if out.n_nodes == 0:
        out.n_nodes = len(df)
    return out


def build_blob_hypers(
    blobs: dict,
    n_hyper: int | None = None,
    k_ball: int | None = None,
    contrast: float | None = None,
    n_bg: int | None = None,
    bg_min: int | None = None,
    ra_col: str = "ra_des",
    dec_col: str = "dec_des",
    include_sky: bool = True,
    include_pm: bool = True,
) -> dict[tuple[int, int], BlobHyper]:
    """Sky over/underdensity + PM-peak hyperedges for every ``(cell_id, gmm_label)`` blob."""
    return {
        key: blob_hyperedges(
            df, n_hyper=n_hyper, k_ball=k_ball, contrast=contrast,
            n_bg=n_bg, bg_min=bg_min, ra_col=ra_col, dec_col=dec_col,
            include_sky=include_sky, include_pm=include_pm,
        )
        for key, df in blobs.items()
    }


def hyper_incidence(hyper: BlobHyper, n_nodes: int | None = None) -> np.ndarray:
    """Boolean incidence ``(H, N)`` for one blob's hyperedges."""
    n = int(hyper.n_nodes if n_nodes is None else n_nodes)
    inc = np.zeros((hyper.n_edges, n), dtype=bool)
    for e, mem in enumerate(hyper.members):
        inc[e, mem] = True
    return inc


def knn_edge_y(knn: BlobKNN, is_mock) -> np.ndarray:
    """``edge_y = 1`` iff both endpoints are mock-stream stars."""
    m = np.asarray(is_mock, dtype=bool)
    if knn.n_edges == 0:
        return np.zeros(0, dtype=np.int32)
    return (m[knn.src] & m[knn.dst]).astype(np.int32)


def hyper_purity(hyper: BlobHyper, is_mock) -> np.ndarray:
    """Mock fraction of each hyperedge's members."""
    m = np.asarray(is_mock, dtype=bool)
    out = np.zeros(hyper.n_edges, dtype=np.float64)
    for e, mem in enumerate(hyper.members):
        if mem.size:
            out[e] = float(m[mem].mean())
    return out


def hyper_edge_y(hyper: BlobHyper, is_mock, tau: float | None = None) -> np.ndarray:
    """``hyper_y = 1`` iff member mock fraction ≥ ``tau`` (default ``BLOB_HYPER_PURITY``)."""
    tau = float(BLOB_HYPER_PURITY if tau is None else tau)
    return (hyper_purity(hyper, is_mock) >= tau).astype(np.int32)


def _blob_is_mock(df: pd.DataFrame) -> np.ndarray:
    if "is_mock_stream" not in df.columns:
        return np.zeros(len(df), dtype=bool)
    return df["is_mock_stream"].to_numpy(bool)


def blob_mock_table(blobs: dict) -> pd.DataFrame:
    rows = []
    for key, df in blobs.items():
        cid, j, part = blob_key_ids(key)
        n_m = int(_blob_is_mock(df).sum())
        rows.append({
            "cell_id": int(cid), "gmm_label": int(j), "blob_part": int(part),
            "n_stars": len(df), "n_mock": n_m,
        })
    tab = pd.DataFrame(rows)
    if len(tab):
        cell_n = tab.groupby("cell_id")["n_mock"].transform("sum")
        tab["cell_n_mock"] = cell_n
    return tab


def pick_blob_examples(
    blob_tab: pd.DataFrame,
    *,
    n_rich: int = 3,
    n_mid: int = 2,
    n_sparse: int = 2,
    n_empty: int = 4,
    max_n_stars: int = 25_000,
    verbose: bool = True,
) -> pd.DataFrame:
    """Diverse ``(cell, Gaussian)`` rows for sky+PM QC plots.

    Prefers blobs with ``n_stars <= max_n_stars`` so the notebook stays
    interactive. Roles: ``rich`` (most mocks), ``mid`` (near-median mock
    occupancy), ``sparse`` (few leftover mocks), ``empty`` (zero mocks).
    """
    cols = ["cell_id", "gmm_label", "blob_part", "role", "n_stars", "n_mock"]
    if blob_tab is None or not len(blob_tab):
        return pd.DataFrame(columns=cols)
    tab = blob_tab.copy()
    if "blob_part" not in tab.columns:
        tab["blob_part"] = 0
    skipped = tab[tab["n_stars"] > int(max_n_stars)]
    plottable = tab[tab["n_stars"] <= int(max_n_stars)]
    if not len(plottable):
        plottable = tab
    if verbose and len(skipped):
        print(
            f"Plot skip: {len(skipped)} blobs with N>{int(max_n_stars):,} "
            f"(max N={int(skipped.n_stars.max()):,})"
        )

    picked: list[dict] = []
    used: set[tuple[int, int, int]] = set()

    def _row_key(row) -> tuple[int, int, int]:
        return (int(row.cell_id), int(row.gmm_label), int(getattr(row, "blob_part", 0)))

    def _take(sub: pd.DataFrame, n: int, role: str, sort_cols, ascending) -> None:
        if n <= 0 or sub is None or not len(sub):
            return
        work = sub.sort_values(sort_cols, ascending=ascending)
        n_role = 0
        for row in work.itertuples(index=False):
            k = _row_key(row)
            if k in used:
                continue
            used.add(k)
            picked.append({
                "cell_id": k[0],
                "gmm_label": k[1],
                "blob_part": k[2],
                "role": role,
                "n_stars": int(row.n_stars),
                "n_mock": int(row.n_mock),
            })
            n_role += 1
            if n_role >= n:
                break

    nonempty = plottable[plottable["n_mock"] > 0]
    empty = plottable[plottable["n_mock"] == 0]
    _take(nonempty, n_rich, "rich", ["n_mock", "n_stars"], [False, False])
    if len(nonempty) and n_mid:
        med = float(nonempty["n_mock"].median())
        mid_sub = nonempty.assign(_d=(nonempty["n_mock"] - med).abs())
        _take(mid_sub, n_mid, "mid", ["_d", "n_stars"], [True, True])
    sparse_pool = nonempty
    if len(nonempty):
        sparse_pool = nonempty[nonempty["n_mock"] <= max(float(nonempty["n_mock"].median()), 1.0)]
    _take(sparse_pool, n_sparse, "sparse", ["n_mock", "n_stars"], [True, False])
    if n_empty:
        n_large = 1 if n_empty >= 2 else n_empty
        _take(empty, n_large, "empty", ["n_stars"], [False])
        n_got = sum(1 for p in picked if p["role"] == "empty")
        remain = int(n_empty) - n_got
        if remain > 0 and len(empty):
            med_n = float(empty["n_stars"].median())
            empty_typ = empty.assign(_d=(empty["n_stars"] - med_n).abs())
            _take(empty_typ, remain, "empty", ["_d", "n_stars"], [True, True])
    out = pd.DataFrame(picked, columns=cols)
    if verbose:
        print("Blob examples for sky+PM plots:")
        if len(out):
            print(out.to_string(index=False))
        else:
            print("  (none)")
    return out


def split_blobs_for_training(
    blobs: dict,
    min_cell_mock: int = BLOB_TRAIN_CELL_MOCK,
    min_blob_mock: int = BLOB_TRAIN_BLOB_MOCK,
    empty_per_pos: float = 1.0,
    rng: int = 0,
) -> dict:
    """Host blobs vs empty negatives vs skipped weak-signal graphs.

    Occupancy split for diagnostics only — the default graph pool keeps every
    remaining GMM blob. Tiny mock *streams* are dropped at load, not here.

    * **pos** — cell has ≥ ``min_cell_mock`` mocks and this Gaussian has
      ≥ ``min_blob_mock`` (a real host overdensity).
    * **empty** — ``n_mock == 0``, subsampled to ``empty_per_pos * n_pos``
      (prefer empty Gaussians inside gated cells).
    * **few_cell** — cell has 1…``min_cell_mock-1`` mocks.
    * **dilute** — gated cell but this Gaussian has 1…``min_blob_mock-1`` mocks.
    """
    tab = blob_mock_table(blobs)
    empty_keys: list[tuple[int, int]] = []
    pos_keys: list[tuple[int, int]] = []
    few_cell_keys: list[tuple[int, int]] = []
    dilute_keys: list[tuple[int, int]] = []
    if not len(tab):
        return {
            "pos": [], "empty": [], "few_cell": [], "dilute": [], "table": tab,
        }

    def _key(row) -> tuple[int, int, int]:
        return (int(row.cell_id), int(row.gmm_label), int(getattr(row, "blob_part", 0)))

    gated_empty = []
    other_empty = []
    for row in tab.itertuples(index=False):
        k = _key(row)
        if row.cell_n_mock <= 0:
            other_empty.append(k)
        elif row.cell_n_mock < int(min_cell_mock):
            few_cell_keys.append(k)
        elif row.n_mock >= int(min_blob_mock):
            pos_keys.append(k)
        elif row.n_mock == 0:
            gated_empty.append(k)
        else:
            dilute_keys.append(k)

    n_keep = int(np.ceil(float(empty_per_pos) * max(len(pos_keys), 1)))
    rng_np = np.random.default_rng(int(rng))
    ranked = gated_empty + other_empty
    if len(ranked) > n_keep:
        if len(gated_empty) >= n_keep:
            pick = rng_np.choice(len(gated_empty), n_keep, replace=False)
            empty_keys = [gated_empty[i] for i in pick]
        else:
            n_fill = n_keep - len(gated_empty)
            fill = []
            if other_empty and n_fill > 0:
                take = min(n_fill, len(other_empty))
                fill = [other_empty[i] for i in rng_np.choice(len(other_empty), take, replace=False)]
            empty_keys = list(gated_empty) + fill
    else:
        empty_keys = ranked

    return {
        "pos": pos_keys,
        "empty": empty_keys,
        "few_cell": few_cell_keys,
        "dilute": dilute_keys,
        "table": tab,
        "n_gated_empty": len(gated_empty),
        "n_other_empty": len(other_empty),
    }


def print_blob_hyper_summary(hypers: dict, blobs: dict | None = None, n_show: int = 8):
    """Size / peak vs background occupancy for ``build_blob_hypers`` output."""
    if not hypers:
        print("No blob hyperedges.")
        return
    rows = []
    for key, hyp in hypers.items():
        cid, j, part = blob_key_ids(key)
        n_mock = np.nan
        n_y = n_y_peak = n_y_bg = np.nan
        if blobs is not None and key in blobs:
            m = _blob_is_mock(blobs[key])
            n_mock = int(m.sum())
            if hyp.n_edges:
                hy = hyper_edge_y(hyp, m)
                n_y = int(hy.sum())
                n_y_peak = int(hy[np.isin(hyp.kind, _PEAK_KINDS)].sum()) if hyp.kind.size else 0
                n_y_bg = int(hy[np.isin(hyp.kind, _BG_KINDS)].sum()) if hyp.kind.size else 0
                n_y_sky = int(hy[np.isin(hyp.kind, _SKY_KINDS)].sum()) if hyp.kind.size else 0
                n_y_pm = int(hy[np.isin(hyp.kind, _PM_KINDS)].sum()) if hyp.kind.size else 0
            else:
                n_y_sky = n_y_pm = 0
        else:
            n_y_sky = n_y_pm = np.nan
        rows.append({
            "cell_id": cid, "gmm_label": j, "blob_part": part,
            "n_nodes": hyp.n_nodes,
            "n_sky": hyp.n_sky, "n_pm": hyp.n_pm,
            "n_peak": hyp.n_peak, "n_bg": hyp.n_bg,
            "n_hyper": hyp.n_edges, "n_mock": n_mock,
            "n_hyper_pos": n_y, "n_peak_pos": n_y_peak, "n_bg_pos": n_y_bg,
            "n_sky_pos": n_y_sky, "n_pm_pos": n_y_pm,
        })
    tab = pd.DataFrame(rows)
    any_h = next(iter(hypers.values()))
    print(
        f"Blob hyperedges: peak cap={any_h.n_hyper}  k_ball={any_h.k_ball}  "
        f"contrast={any_h.contrast:g}  bg_min={BLOB_HYPER_BG_MIN}  "
        f"purity τ={BLOB_HYPER_PURITY:g}"
    )
    print(
        f"  sky vs PM  (median H): sky {tab.n_sky.median():.0f}  PM {tab.n_pm.median():.0f}"
    )
    print(
        f"  graphs: {len(tab):,}  median peak {tab.n_peak.median():.0f}  "
        f"median bg {tab.n_bg.median():.0f}  median total H {tab.n_hyper.median():.0f}"
    )
    has_lab = tab["n_hyper_pos"].notna()
    if has_lab.any():
        sub = tab[has_lab]
        print(
            f"  hyper_y=1: median {sub.n_hyper_pos.median():.0f} / graph  "
            f"(peak pos median {sub.n_peak_pos.median():.0f}, "
            f"bg pos median {sub.n_bg_pos.median():.0f})"
        )
    print("  most peaks:")
    print(tab.sort_values(["n_peak", "n_mock"], ascending=False).head(n_show).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return tab


def print_blob_label_summary(
    knns: dict,
    hypers: dict,
    blobs: dict,
    tau: float | None = None,
    split: dict | None = None,
    n_show: int = 8,
):
    """Class balance for kNN ``edge_y`` and hyper ``hyper_y`` (training-only labels)."""
    tau = float(BLOB_HYPER_PURITY if tau is None else tau)
    rows = []
    for key, df in blobs.items():
        m = _blob_is_mock(df)
        knn = knns.get(key)
        hyp = hypers.get(key)
        n_e = knn.n_edges if knn is not None else 0
        n_e_pos = int(knn_edge_y(knn, m).sum()) if knn is not None and n_e else 0
        n_h = hyp.n_edges if hyp is not None else 0
        hy = hyper_edge_y(hyp, m, tau=tau) if hyp is not None and n_h else np.zeros(0, dtype=np.int32)
        n_h_pos = int(hy.sum()) if n_h else 0
        n_h_peak_pos = int(hy[np.isin(hyp.kind, _PEAK_KINDS)].sum()) if hyp is not None and n_h else 0
        n_h_sky_pos = int(hy[np.isin(hyp.kind, _SKY_KINDS)].sum()) if hyp is not None and n_h else 0
        n_h_pm_pos = int(hy[np.isin(hyp.kind, _PM_KINDS)].sum()) if hyp is not None and n_h else 0
        cid, j, part = blob_key_ids(key)
        rows.append({
            "cell_id": cid, "gmm_label": j, "blob_part": part,
            "n_stars": len(df), "n_mock": int(m.sum()),
            "n_knn": n_e, "n_knn_pos": n_e_pos,
            "n_hyper": n_h, "n_hyper_pos": n_h_pos, "n_peak_pos": n_h_peak_pos,
            "n_sky_pos": n_h_sky_pos, "n_pm_pos": n_h_pm_pos,
        })
    tab = pd.DataFrame(rows)
    print(f"Blob labels: edge_y = both mock;  hyper_y = purity ≥ {tau:g}")
    if split is not None:
        def _cover(name, keys):
            if not keys:
                print(f"  {name}: 0 graphs")
                return
            ix = pd.MultiIndex.from_tuples(list(keys), names=["cell_id", "gmm_label", "blob_part"])
            idx_cols = ["cell_id", "gmm_label", "blob_part"]
            if "blob_part" not in tab.columns:
                tab = tab.assign(blob_part=0)
            sub = tab.set_index(idx_cols).reindex(ix)
            print(
                f"  {name}: {len(keys):,} graphs  "
                f"median N={sub.n_stars.median():,.0f}  mocks={sub.n_mock.median():.0f}  "
                f"kNN+ {100 * sub.n_knn_pos.sum() / max(sub.n_knn.sum(), 1):.2f}%  "
                f"hyper+ {100 * sub.n_hyper_pos.sum() / max(sub.n_hyper.sum(), 1):.2f}%"
            )
        _cover("pos   (train hosts)", split["pos"])
        _cover("empty (train neg)", split["empty"])
        _cover("few_cell (skip)", split["few_cell"])
        _cover("dilute (skip)", split["dilute"])
    else:
        print(
            f"  kNN+ {100 * tab.n_knn_pos.sum() / max(tab.n_knn.sum(), 1):.2f}%  "
            f"hyper+ {100 * tab.n_hyper_pos.sum() / max(tab.n_hyper.sum(), 1):.2f}%"
        )
    print("  mock-richest:")
    print(tab.sort_values(["n_mock", "n_hyper_pos"], ascending=False).head(n_show).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return tab


def _node_features(df: pd.DataFrame, feature_cols: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    cols = list(feature_cols or TRAIN_VARS)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"blob missing node features {missing}")
    x = np.column_stack([df[c].to_numpy(np.float32) for c in cols]).astype(np.float32, copy=False)
    return x, cols


def _hyper_csr(hyper: BlobHyper) -> tuple[np.ndarray, np.ndarray]:
    ptr = [0]
    idx: list[int] = []
    for mem in hyper.members:
        idx.extend(np.asarray(mem, dtype=np.int32).tolist())
        ptr.append(len(idx))
    return np.asarray(ptr, dtype=np.int32), np.asarray(idx, dtype=np.int32)


def save_blob_graph(
    path,
    df: pd.DataFrame,
    knn: BlobKNN,
    hyper: BlobHyper,
    *,
    key,
    feature_cols: list[str] | None = None,
) -> dict:
    """Write one blob graph as ``.npz`` (nodes, kNN, hyperedges, labels)."""
    df = ensure_is_mock_stream(df)
    cid, j, part = blob_key_ids(key)
    x, cols = _node_features(df, feature_cols)
    y = df["is_mock_stream"].to_numpy(np.int8)
    edge_y = knn_edge_y(knn, y)
    hy = hyper_edge_y(hyper, y)
    h_ptr, h_idx = _hyper_csr(hyper)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=x,
        y=y,
        feature_cols=np.asarray(cols),
        cell_id=np.int32(cid),
        gmm_label=np.int32(j),
        blob_part=np.int32(part),
        knn_src=np.asarray(knn.src, dtype=np.int32),
        knn_dst=np.asarray(knn.dst, dtype=np.int32),
        knn_dist=np.asarray(knn.dist_deg, dtype=np.float32),
        knn_y=np.asarray(edge_y, dtype=np.int8),
        hyper_ptr=h_ptr,
        hyper_idx=h_idx,
        hyper_y=np.asarray(hy, dtype=np.int8),
        hyper_kind=np.asarray(hyper.kind, dtype=np.int8),
        hyper_r_k=np.asarray(hyper.r_k, dtype=np.float32),
        hyper_centers=np.asarray(hyper.centers, dtype=np.float32),
    )
    return {
        "cell_id": cid, "gmm_label": j, "blob_part": part,
        "n_stars": int(len(df)), "n_mock": int(y.sum()),
        "n_knn": int(knn.n_edges), "n_knn_pos": int(edge_y.sum()),
        "n_hyper": int(hyper.n_edges), "n_hyper_pos": int(hy.sum()),
        "file": path.name,
    }


def build_and_save_graphs(
    blobs: dict,
    outdir,
    *,
    subdir: str = "graphs",
    k: int = BLOB_KNN_K,
    n_hyper: int | None = None,
    k_ball: int | None = None,
    contrast: float | None = None,
    bg_min: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """kNN + hypers per blob, written one ``.npz`` at a time (no giant in-memory dict)."""
    out = Path(outdir) / subdir
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    n = len(blobs)
    for i, (key, df) in enumerate(blobs.items(), start=1):
        knn = blob_knn_edges(df, k=k)
        hyper = blob_hyperedges(
            df, n_hyper=n_hyper, k_ball=k_ball, contrast=contrast, bg_min=bg_min,
        )
        path = out / f"{blob_stem(key)}.npz"
        rows.append(save_blob_graph(path, df, knn, hyper, key=key))
        if verbose and (i % 50 == 0 or i == n):
            print(f"  graphs [{i}/{n}] {path.name}  N={len(df):,}", flush=True)
    tab = pd.DataFrame(rows)
    if len(tab):
        tab.to_parquet(out / "manifest.parquet", index=False)
    if verbose:
        print(
            f"Wrote {len(tab):,} graphs → {out}  "
            f"median N={tab.n_stars.median():,.0f}  max N={int(tab.n_stars.max()) if len(tab) else 0:,}"
        )
    return tab
