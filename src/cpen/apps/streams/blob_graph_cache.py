"""Stage-4 blob ``.npz`` graphs → CPEN padded COO incidence batches.

One graph per GMM blob (``blob_cXXXX_gYY_pZZ.npz``), written by
``scans/streams/4_build_graphs.py``. Every star carries a label, so supervision
is graph-level rather than node-level: a blob belongs entirely to train, val, or
test according to which galaxy split it came from.

Occupancy is extremely skewed — ~97% of blobs hold zero mock stars, since every
surviving Gaussian is kept as a negative. Training on that raw mix wastes almost
all steps on empty field, so the **train** pool is rebalanced:

* **rich** — ``n_mock >= rich_min_mock`` (default 100), a real host overdensity
* **empty** — ``n_mock == 0``, sampled 1:1 against the rich pool and matched to
  the rich size distribution so the negatives are not all tiny leftovers
* **dilute** — ``1 <= n_mock < rich_min_mock``, dropped by default (weak signal)

**val/test keep the natural occupancy**, since that is the prior the model runs
at in production. Only the train sampler is rebalanced.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from cpen.apps.streams.preprocess.config import DEFAULT_PIPELINE_ROOT
from cpen.graphs.graphs import scale_features_l2_sqrt_dim

GRAPHS_SUBDIR = "graphs"
MANIFEST_NAME = "manifest.parquet"
SPLITS = ("train", "val", "test")
OUT_DIM = 2

# Column order written by ``save_blob_graph`` (config.TRAIN_VARS).
RAW_FEATURE_NAMES = (
    "ra_des_sin",
    "ra_des_cos",
    "dec_des",
    "psf_mag_aper_8_g_corrected_des",
    "psf_mag_aper_8_r_corrected_des",
    "pmra_gaia",
    "pmdec_gaia",
    "parallax_gaia",
)
# Blob-local frame: absolute DES position is shared by every galaxy (same
# footprint, same cell centers), so it cannot generalize across splits.
LOCAL_FEATURE_NAMES = ("dphi", "dlam", "g", "r", "pmra", "pmdec", "parallax")
# Shared 4-vector for kNN 2-edges and hyperedges (same slots, analogous stats).
#   kNN:       (sky sep deg, |ΔPM|, PM dir cosine, |Δ(g-r)|)
#   hyperedge: (k-ball radius, member PM scatter, mean pairwise PM cosine, color scatter)
EDGE_FEATURE_NAMES = ("knn_dist", "dpm", "dir_cos", "dcolor")
N_EDGE_FEATURES = len(EDGE_FEATURE_NAMES)

_IDX_RA_SIN, _IDX_RA_COS, _IDX_DEC = 0, 1, 2
_IDX_G, _IDX_R = 3, 4
_IDX_PMRA, _IDX_PMDEC, _IDX_PLX = 5, 6, 7

# Mock stars are injected with essentially noiseless Gaia parallax
# (scatter ~0.002 mas vs ~0.27 mas for background). That is a catalog
# artifact, not a stream property, so parallax is withheld by default.
# Proper motion is kept: streams really are kinematically cold.
DEFAULT_DROP_FEATURES = ("parallax",)
_FEATURE_ALIASES: dict[str, frozenset[str]] = {
    "parallax": frozenset({"parallax", "parallax_gaia"}),
    "pm": frozenset({"pmra", "pmdec", "pmra_gaia", "pmdec_gaia"}),
    "pmra": frozenset({"pmra", "pmra_gaia"}),
    "pmdec": frozenset({"pmdec", "pmdec_gaia"}),
    "mag": frozenset(
        {"g", "r", "psf_mag_aper_8_g_corrected_des", "psf_mag_aper_8_r_corrected_des"}
    ),
    "sky": frozenset({"dphi", "dlam", "ra_des_sin", "ra_des_cos", "dec_des"}),
}
_NO_DROP_TOKENS = {"none", "keep-all", "all"}


def parse_drop_features(drop: str | Sequence[str] | None) -> tuple[str, ...]:
    """Normalize a comma/space-separated drop spec into sorted alias tokens.

    ``None`` means "caller did not choose"; use ``DEFAULT_DROP_FEATURES``.
    ``"none"`` explicitly keeps every column.
    """
    if drop is None:
        return tuple(DEFAULT_DROP_FEATURES)
    items = re.split(r"[,\s]+", drop.strip()) if isinstance(drop, str) else list(drop)
    tokens: list[str] = []
    for raw in items:
        tok = str(raw).strip().lower()
        if not tok:
            continue
        if tok in _NO_DROP_TOKENS:
            return ()
        known = tok in _FEATURE_ALIASES or any(
            tok in names for names in _FEATURE_ALIASES.values()
        )
        if not known:
            raise ValueError(
                f"unknown feature {tok!r}; expected 'none', an explicit column "
                f"name, or one of {sorted(_FEATURE_ALIASES)}"
            )
        tokens.append(tok)
    return tuple(sorted(set(tokens)))


def kept_feature_columns(
    node_frame: str, drop_features: str | Sequence[str] | None = None
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """``(column indices, names)`` of node features surviving ``drop_features``."""
    names = LOCAL_FEATURE_NAMES if node_frame == "local" else RAW_FEATURE_NAMES
    dropped: set[str] = set()
    for tok in parse_drop_features(drop_features):
        dropped |= _FEATURE_ALIASES.get(tok, frozenset({tok}))
    keep = tuple(i for i, n in enumerate(names) if n not in dropped)
    if not keep:
        raise ValueError(f"drop_features {drop_features!r} removed every node feature")
    return keep, tuple(names[i] for i in keep)


def n_node_features(
    node_frame: str, drop_features: str | Sequence[str] | None = None
) -> int:
    return len(kept_feature_columns(node_frame, drop_features)[0])


# Edge features are derived from node quantities, so an ablation that withholds
# a node column has to withhold the edge columns built from it too. Parallax
# feeds no edge feature, so the default drop leaves all four intact.
_EDGE_FEATURE_SOURCES: dict[str, frozenset[str]] = {
    "knn_dist": frozenset(),
    "dpm": frozenset({"pm", "pmra", "pmdec"}),
    "dir_cos": frozenset({"pm", "pmra", "pmdec"}),
    "dcolor": frozenset({"mag"}),
}


def kept_edge_columns(
    drop_features: str | Sequence[str] | None = None,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    tokens = set(parse_drop_features(drop_features))
    keep = tuple(
        i
        for i, name in enumerate(EDGE_FEATURE_NAMES)
        if not (_EDGE_FEATURE_SOURCES[name] & tokens)
    )
    if not keep:
        raise ValueError(f"drop_features {drop_features!r} removed every edge feature")
    return keep, tuple(EDGE_FEATURE_NAMES[i] for i in keep)


def n_edge_features(drop_features: str | Sequence[str] | None = None) -> int:
    return len(kept_edge_columns(drop_features)[0])


# --------------------------------------------------------------------------
# manifests
# --------------------------------------------------------------------------


def load_split_manifest(
    out_root: str | Path,
    split: str,
    *,
    galaxy_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Concatenate every ``{out_root}/{split}/{gid}/graphs/manifest.parquet``.

    Adds ``galaxy_id``, ``split``, ``path``, and ``is_empty``.
    """
    root = Path(out_root) / split
    if not root.is_dir():
        raise FileNotFoundError(f"No blob graphs for split {split!r} under {root}")
    wanted = None if galaxy_ids is None else {str(g).strip() for g in galaxy_ids}
    frames = []
    for gdir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit()):
        if wanted is not None and gdir.name not in wanted:
            continue
        man = gdir / GRAPHS_SUBDIR / MANIFEST_NAME
        if not man.is_file():
            continue
        tab = pd.read_parquet(man)
        if not len(tab):
            continue
        tab["galaxy_id"] = gdir.name
        tab["split"] = split
        tab["path"] = [str(gdir / GRAPHS_SUBDIR / f) for f in tab["file"]]
        frames.append(tab)
    if not frames:
        raise FileNotFoundError(
            f"No graphs/manifest.parquet under {root} — run pipeline stage 4 first"
        )
    out = pd.concat(frames, ignore_index=True)
    out["is_empty"] = out["n_mock"] == 0
    return out


# --------------------------------------------------------------------------
# split selection
# --------------------------------------------------------------------------


def _match_size_bins(
    rich: pd.DataFrame,
    empty: pd.DataFrame,
    n_take: int,
    *,
    rng: np.random.Generator,
    n_bins: int = 8,
) -> np.ndarray:
    """Sample ``n_take`` empty rows whose ``n_stars`` tracks the rich distribution.

    Uniform sampling would fill the negative class with tiny leftover blobs; the
    model would then separate hosts from negatives on size alone.
    """
    if n_take <= 0 or not len(empty):
        return np.zeros(0, dtype=np.int64)
    if n_take >= len(empty):
        return empty.index.to_numpy()

    edges = np.unique(
        np.quantile(rich["n_stars"].to_numpy(float), np.linspace(0.0, 1.0, n_bins + 1))
    )
    if edges.size < 2:
        return rng.choice(empty.index.to_numpy(), n_take, replace=False)
    edges[0], edges[-1] = -np.inf, np.inf

    rich_bin = np.digitize(rich["n_stars"].to_numpy(float), edges[1:-1])
    empty_bin = np.digitize(empty["n_stars"].to_numpy(float), edges[1:-1])
    target = np.bincount(rich_bin, minlength=edges.size - 1).astype(float)
    target = np.floor(target / target.sum() * n_take).astype(int)

    picked: list[np.ndarray] = []
    leftover = 0
    for b, want in enumerate(target):
        pool = empty.index.to_numpy()[empty_bin == b]
        take = min(int(want), pool.size)
        if take:
            picked.append(rng.choice(pool, take, replace=False))
        leftover += int(want) - take
    chosen = np.concatenate(picked) if picked else np.zeros(0, dtype=np.int64)

    # Bins that ran dry (plus the floor() remainder) are filled at random.
    short = n_take - chosen.size
    if short > 0:
        rest = np.setdiff1d(empty.index.to_numpy(), chosen, assume_unique=False)
        if rest.size:
            chosen = np.concatenate([chosen, rng.choice(rest, min(short, rest.size), replace=False)])
    return chosen


def select_train_graphs(
    man: pd.DataFrame,
    *,
    rich_min_mock: int = 100,
    empty_per_rich: float = 1.0,
    include_dilute: bool = False,
    min_stars: int = 16,
    seed: int = 0,
) -> pd.DataFrame:
    """Balanced train pool: mock-rich hosts plus size-matched empty negatives."""
    tab = man[man["n_stars"] >= int(min_stars)].copy()
    rich = tab[tab["n_mock"] >= int(rich_min_mock)]
    empty = tab[tab["is_empty"]]
    dilute = tab[(~tab["is_empty"]) & (tab["n_mock"] < int(rich_min_mock))]
    if not len(rich):
        raise ValueError(
            f"No train blobs with n_mock >= {rich_min_mock}; lower --blob-rich-min-mock"
        )

    rng = np.random.default_rng(int(seed))
    n_take = int(round(float(empty_per_rich) * len(rich)))
    empty_idx = _match_size_bins(rich, empty, n_take, rng=rng)

    parts = [rich.assign(role="rich"), tab.loc[empty_idx].assign(role="empty")]
    if include_dilute and len(dilute):
        parts.append(dilute.assign(role="dilute"))
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)


def select_eval_graphs(
    man: pd.DataFrame,
    *,
    max_graphs: int | None = None,
    min_stars: int = 16,
    seed: int = 0,
) -> pd.DataFrame:
    """Natural-occupancy eval pool, optionally capped while preserving the ratio."""
    tab = man[man["n_stars"] >= int(min_stars)].copy()
    tab["role"] = np.where(tab["is_empty"], "empty", "host")
    if max_graphs is None or max_graphs <= 0 or len(tab) <= max_graphs:
        return tab.reset_index(drop=True)

    rng = np.random.default_rng(int(seed))
    frac = float(max_graphs) / len(tab)
    picked = []
    for _, sub in tab.groupby("is_empty", sort=False):
        take = max(1, int(round(frac * len(sub))))
        take = min(take, len(sub))
        picked.append(sub.iloc[rng.choice(len(sub), take, replace=False)])
    return pd.concat(picked, ignore_index=True).reset_index(drop=True)


# --------------------------------------------------------------------------
# class weights (manifest arithmetic — no need to open any .npz)
# --------------------------------------------------------------------------


def _inverse_frequency(n_neg: float, n_pos: float) -> torch.Tensor:
    counts = torch.tensor([max(n_neg, 1.0), max(n_pos, 1.0)], dtype=torch.float32)
    return counts.sum() / (OUT_DIM * counts)


def node_class_weights(man: pd.DataFrame) -> torch.Tensor:
    n_pos = float(man["n_mock"].sum())
    return _inverse_frequency(float(man["n_stars"].sum()) - n_pos, n_pos)


def edge_class_weights(man: pd.DataFrame) -> torch.Tensor:
    n_pos = float(man["n_knn_pos"].sum())
    n_tot = float(man["n_knn"].sum())
    if "n_hyper" in man.columns:
        n_pos += float(man["n_hyper_pos"].sum())
        n_tot += float(man["n_hyper"].sum())
    return _inverse_frequency(n_tot - n_pos, n_pos)


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------


def _local_node_features(raw: np.ndarray) -> np.ndarray:
    """Blob-centered tangent-plane sky coords; photometry / PM / parallax as-is."""
    ra_sin = raw[:, _IDX_RA_SIN]
    ra_cos = raw[:, _IDX_RA_COS]
    dec = raw[:, _IDX_DEC]
    ra = np.degrees(np.arctan2(ra_sin, ra_cos))
    ra0 = np.degrees(np.arctan2(ra_sin.mean(), ra_cos.mean()))
    dec0 = float(dec.mean())
    dra = (ra - ra0 + 180.0) % 360.0 - 180.0
    dphi = dra * np.cos(np.radians(dec0))
    dlam = dec - dec0
    return np.column_stack(
        [
            dphi,
            dlam,
            raw[:, _IDX_G],
            raw[:, _IDX_R],
            raw[:, _IDX_PMRA],
            raw[:, _IDX_PMDEC],
            raw[:, _IDX_PLX],
        ]
    ).astype(np.float32, copy=False)


def _edge_features(raw: np.ndarray, src: np.ndarray, dst: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Symmetric kNN 2-edge features — undirected edges must be endpoint-order invariant."""
    pmra, pmdec = raw[:, _IDX_PMRA], raw[:, _IDX_PMDEC]
    dpmra = pmra[src] - pmra[dst]
    dpmdec = pmdec[src] - pmdec[dst]
    dpm = np.hypot(dpmra, dpmdec)
    n_s = np.hypot(pmra[src], pmdec[src])
    n_d = np.hypot(pmra[dst], pmdec[dst])
    dir_cos = (pmra[src] * pmra[dst] + pmdec[src] * pmdec[dst]) / np.clip(n_s * n_d, 1e-6, None)
    color = raw[:, _IDX_G] - raw[:, _IDX_R]
    dcolor = np.abs(color[src] - color[dst])
    return np.column_stack([dist, dpm, dir_cos, dcolor]).astype(np.float32, copy=False)


def _hyper_features(
    raw: np.ndarray,
    ptr: np.ndarray,
    idx: np.ndarray,
    r_k: np.ndarray,
) -> np.ndarray:
    """Hyperedge features in the same 4 slots as kNN 2-edges.

    ``r_k`` is the k-ball radius in the space the hyperedge was built in
    (sky degrees for sky peak/bg, mas/yr for PM peaks). PM scatter / mean
    pairwise PM cosine / colour scatter are always computed in node space.
    """
    n_h = int(ptr.size) - 1
    if n_h <= 0:
        return np.zeros((0, 4), dtype=np.float32)
    pmra = raw[:, _IDX_PMRA]
    pmdec = raw[:, _IDX_PMDEC]
    color = raw[:, _IDX_G] - raw[:, _IDX_R]
    out = np.zeros((n_h, 4), dtype=np.float32)
    out[:, 0] = np.asarray(r_k, dtype=np.float32).reshape(-1)[:n_h]
    for e in range(n_h):
        mem = idx[int(ptr[e]) : int(ptr[e + 1])]
        if mem.size == 0:
            continue
        out[e, 1] = float(np.hypot(pmra[mem].std(), pmdec[mem].std()))
        nrm = np.clip(np.hypot(pmra[mem], pmdec[mem]), 1e-6, None)
        ux, uy = pmra[mem] / nrm, pmdec[mem] / nrm
        n_mem = int(mem.size)
        if n_mem == 1:
            out[e, 2] = 1.0
        else:
            # Mean pairwise cosine = (|Σu|² − n) / (n(n−1)).
            sx, sy = float(ux.sum()), float(uy.sum())
            out[e, 2] = (sx * sx + sy * sy - n_mem) / (n_mem * (n_mem - 1))
        out[e, 3] = float(color[mem].std()) if n_mem > 1 else 0.0
    return out


def _load_graph_arrays(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        out = {
            "raw": z["x"].astype(np.float32),
            "y": z["y"].astype(np.int64),
            "src": z["knn_src"].astype(np.int64),
            "dst": z["knn_dst"].astype(np.int64),
            "dist": z["knn_dist"].astype(np.float32),
            "knn_y": z["knn_y"].astype(np.int64),
        }
        if "hyper_ptr" in z.files:
            out["hyper_ptr"] = z["hyper_ptr"].astype(np.int64)
            out["hyper_idx"] = z["hyper_idx"].astype(np.int64)
            out["hyper_y"] = z["hyper_y"].astype(np.int64)
            out["hyper_r_k"] = z["hyper_r_k"].astype(np.float32)
        return out


def _combined_edge_features(
    arrays: dict[str, np.ndarray],
    *,
    include_hyperedges: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    knn_x = _edge_features(arrays["raw"], arrays["src"], arrays["dst"], arrays["dist"])
    n_knn = int(knn_x.shape[0])
    if not include_hyperedges or "hyper_ptr" not in arrays:
        return knn_x, arrays["knn_y"], n_knn
    hyp_x = _hyper_features(
        arrays["raw"], arrays["hyper_ptr"], arrays["hyper_idx"], arrays["hyper_r_k"]
    )
    hyp_y = arrays["hyper_y"]
    if hyp_x.shape[0] != hyp_y.shape[0]:
        raise ValueError(
            f"hyper feature/label length mismatch: {hyp_x.shape[0]} vs {hyp_y.shape[0]}"
        )
    if hyp_x.shape[0] == 0:
        return knn_x, arrays["knn_y"], n_knn
    return (
        np.concatenate([knn_x, hyp_x], axis=0),
        np.concatenate([arrays["knn_y"], hyp_y], axis=0),
        n_knn,
    )


def raw_features(
    path: str | Path,
    *,
    node_frame: str,
    drop_features: str | Sequence[str] | None = None,
    include_hyperedges: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """``(node_features, edge_features)`` before standardization, after drops."""
    arrays = _load_graph_arrays(path)
    x = _local_node_features(arrays["raw"]) if node_frame == "local" else arrays["raw"]
    e, _, _ = _combined_edge_features(arrays, include_hyperedges=include_hyperedges)
    node_keep, _ = kept_feature_columns(node_frame, drop_features)
    edge_keep, _ = kept_edge_columns(drop_features)
    return x[:, node_keep], e[:, edge_keep]


def fit_feature_stats(
    paths: Sequence[str | Path],
    *,
    node_frame: str,
    drop_features: str | Sequence[str] | None = None,
    include_hyperedges: bool = True,
    max_graphs: int = 256,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Per-feature mean/std over a sample of **train** graphs only."""
    paths = list(paths)
    if len(paths) > max_graphs:
        rng = np.random.default_rng(int(seed))
        paths = [paths[i] for i in rng.choice(len(paths), max_graphs, replace=False)]
    xs, es = [], []
    for p in paths:
        x, e = raw_features(
            p,
            node_frame=node_frame,
            drop_features=drop_features,
            include_hyperedges=include_hyperedges,
        )
        xs.append(x)
        es.append(e)
    x_all = np.concatenate(xs, axis=0)
    e_all = np.concatenate(es, axis=0)
    return {
        "x_mean": torch.tensor(x_all.mean(0), dtype=torch.float32),
        "x_std": torch.tensor(x_all.std(0), dtype=torch.float32).clamp_min(1e-6),
        "edge_mean": torch.tensor(e_all.mean(0), dtype=torch.float32),
        "edge_std": torch.tensor(e_all.std(0), dtype=torch.float32).clamp_min(1e-6),
    }


# --------------------------------------------------------------------------
# payloads
# --------------------------------------------------------------------------


def blob_to_incidence_payload(
    path: str | Path,
    *,
    role: str,
    node_frame: str = "local",
    drop_features: str | Sequence[str] | None = None,
    stats: dict[str, torch.Tensor] | None = None,
    include_hyperedges: bool = True,
) -> dict[str, torch.Tensor]:
    """One blob ``.npz`` → CPEN COO incidence payload (unbatched).

    ``role`` is the split this graph belongs to (``train``/``val``/``test``).
    Every star is labeled, so the whole graph sits in exactly one split mask;
    the other two masks are all-False. That keeps the discovery pool
    (``~train_mask``) equal to the full graph during val/test.

    Incidence concatenates kNN 2-edges and hyperedges. Each hyperedge is one
    incidence row of degree ``k_ball`` (~17); CPEN's sparse backend already
    handles degree > 2.
    """
    if role not in SPLITS:
        raise ValueError(f"role must be train|val|test; got {role!r}")
    arrays = _load_graph_arrays(path)
    raw = arrays["raw"]
    y_np = arrays["y"]
    src, dst = arrays["src"], arrays["dst"]
    e_np, y_e, n_knn = _combined_edge_features(arrays, include_hyperedges=include_hyperedges)

    x_np = _local_node_features(raw) if node_frame == "local" else raw
    node_keep, _ = kept_feature_columns(node_frame, drop_features)
    edge_keep, _ = kept_edge_columns(drop_features)
    x_np = x_np[:, node_keep]
    e_np = e_np[:, edge_keep]
    x = torch.from_numpy(np.ascontiguousarray(x_np))
    edge_x = torch.from_numpy(np.ascontiguousarray(e_np))
    if stats is not None:
        x = (x - stats["x_mean"]) / stats["x_std"]
        if edge_x.size(0):
            edge_x = (edge_x - stats["edge_mean"]) / stats["edge_std"]
    x = scale_features_l2_sqrt_dim(x)
    if edge_x.size(0):
        edge_x = scale_features_l2_sqrt_dim(edge_x)

    n_nodes = int(x.size(0))
    n_edges = int(edge_x.size(0))
    n_hyper = n_edges - int(n_knn)

    knn_node = np.stack([src, dst], axis=1).reshape(-1).astype(np.int64)
    knn_edge = np.repeat(np.arange(n_knn, dtype=np.int64), 2)
    knn_deg_inv = np.full(n_knn, 0.5, dtype=np.float32)
    if include_hyperedges and "hyper_ptr" in arrays and n_hyper > 0:
        ptr = arrays["hyper_ptr"]
        sizes = np.diff(ptr).astype(np.int64)
        hyp_node = arrays["hyper_idx"]
        hyp_edge = np.repeat(np.arange(n_hyper, dtype=np.int64) + n_knn, sizes)
        hyp_deg_inv = np.where(sizes > 0, 1.0 / sizes.astype(np.float32), 0.0).astype(
            np.float32
        )
        incidence_node = torch.from_numpy(np.concatenate([knn_node, hyp_node]))
        incidence_edge = torch.from_numpy(np.concatenate([knn_edge, hyp_edge]))
        edge_degree_inv = torch.from_numpy(np.concatenate([knn_deg_inv, hyp_deg_inv]))
    else:
        incidence_node = torch.from_numpy(knn_node)
        incidence_edge = torch.from_numpy(knn_edge)
        edge_degree_inv = torch.from_numpy(knn_deg_inv)

    node_degree = torch.bincount(incidence_node, minlength=n_nodes).to(torch.float32)
    y = torch.from_numpy(y_np)
    edge_y = torch.from_numpy(np.ascontiguousarray(y_e))
    node_all = torch.ones(n_nodes, dtype=torch.bool)
    node_none = torch.zeros(n_nodes, dtype=torch.bool)
    edge_all = torch.ones(n_edges, dtype=torch.bool)
    edge_none = torch.zeros(n_edges, dtype=torch.bool)
    is_hyper = torch.zeros(n_edges, dtype=torch.bool)
    if n_hyper > 0:
        is_hyper[n_knn:] = True

    return {
        "x": x,
        "edge_x": edge_x,
        "y": y,
        "edge_y": edge_y,
        "mask": node_all,
        "train_mask": node_all if role == "train" else node_none,
        "val_mask": node_all if role == "val" else node_none,
        "test_mask": node_all if role == "test" else node_none,
        "edge_train_mask": edge_all if role == "train" else edge_none,
        "edge_val_mask": edge_all if role == "val" else edge_none,
        "edge_test_mask": edge_all if role == "test" else edge_none,
        "incidence_node": incidence_node,
        "incidence_edge": incidence_edge,
        "incidence_nnz": torch.tensor(int(incidence_node.numel()), dtype=torch.int64),
        "node_degree_inv": node_degree.clamp_min(1).reciprocal(),
        "edge_degree_inv": edge_degree_inv,
        "n_nodes": torch.tensor(n_nodes, dtype=torch.int64),
        "n_edges": torch.tensor(n_edges, dtype=torch.int64),
        "n_knn": torch.tensor(int(n_knn), dtype=torch.int64),
        "n_hyper": torch.tensor(int(n_hyper), dtype=torch.int64),
        "is_hyper": is_hyper,
    }


class BlobGraphDataset(Dataset):
    """Lazy per-blob dataset. One sample = one graph = one optimizer step."""

    def __init__(
        self,
        paths: Sequence[str],
        *,
        split: str,
        node_frame: str = "local",
        drop_features: str | Sequence[str] | None = None,
        stats: dict[str, torch.Tensor] | None = None,
        include_incidence: bool = True,
        include_edge_features: bool = True,
        include_hyperedges: bool = True,
        max_cache: int = 4096,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be train|val|test; got {split!r}")
        self.paths = list(paths)
        self.split = split
        self.node_frame = node_frame
        self.drop_features = parse_drop_features(drop_features)
        self.stats = stats
        self.include_incidence = include_incidence
        self.include_edge_features = include_edge_features
        self.include_hyperedges = bool(include_hyperedges)
        self.max_cache = int(max_cache)
        self._cache: dict[int, dict[str, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> int:
        if index < 0 or index >= len(self.paths):
            raise IndexError(index)
        return index

    def _payload(self, index: int) -> dict[str, torch.Tensor]:
        hit = self._cache.get(index)
        if hit is not None:
            return hit
        payload = blob_to_incidence_payload(
            self.paths[index],
            role=self.split,
            node_frame=self.node_frame,
            drop_features=self.drop_features,
            stats=self.stats,
            include_hyperedges=self.include_hyperedges,
        )
        if len(self._cache) < self.max_cache:
            self._cache[index] = payload
        return payload

    def collate_samples(self, indices: list[int]) -> dict[str, torch.Tensor]:
        if len(indices) != 1:
            raise ValueError(
                f"blob graphs have different sizes; batch_size must be 1, got {len(indices)}"
            )
        p = self._payload(indices[0])
        batch: dict[str, torch.Tensor] = {
            "x": p["x"].unsqueeze(0),
            "y": p["y"].unsqueeze(0),
            "edge_y": p["edge_y"].unsqueeze(0),
            "mask": p["mask"].unsqueeze(0),
            "loss_mask": p[f"{self.split}_mask"].unsqueeze(0),
            "edge_loss_mask": p[f"edge_{self.split}_mask"].unsqueeze(0),
            "train_mask": p["train_mask"].unsqueeze(0),
            "val_mask": p["val_mask"].unsqueeze(0),
            "test_mask": p["test_mask"].unsqueeze(0),
            "edge_train_mask": p["edge_train_mask"].unsqueeze(0),
            "edge_val_mask": p["edge_val_mask"].unsqueeze(0),
            "edge_test_mask": p["edge_test_mask"].unsqueeze(0),
            "n_knn": p["n_knn"].unsqueeze(0),
            "n_hyper": p["n_hyper"].unsqueeze(0),
            "is_hyper": p["is_hyper"].unsqueeze(0),
        }
        if self.include_edge_features:
            batch["edge_x"] = p["edge_x"].unsqueeze(0)
        if self.include_incidence:
            batch["incidence_node"] = p["incidence_node"].unsqueeze(0)
            batch["incidence_edge"] = p["incidence_edge"].unsqueeze(0)
            batch["incidence_nnz"] = p["incidence_nnz"].unsqueeze(0)
            batch["node_degree_inv"] = p["node_degree_inv"].unsqueeze(0)
            batch["edge_degree_inv"] = p["edge_degree_inv"].unsqueeze(0)
        return batch


DEFAULT_BLOB_DATA_DIR = Path(DEFAULT_PIPELINE_ROOT)
