"""Score real-galaxy blob graphs and tag stars for discovery visualization.

Used by ``notebooks/streams/real_stream_discovery.ipynb`` to:

* rebuild the cell-level train / val / test split used at fine-tune time
* run a fine-tuned ``LitCPENStream`` over every blob in ``galaxies_real/train/0000``
* classify each star as TP / FP / FN / TN at a chosen probability cut, and
  attribute hits/misses to catalogued stream names
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from cpen.apps.streams.blob_graph_cache import (
    blob_to_incidence_payload,
    fit_feature_stats,
    load_split_manifest,
    n_edge_features,
    n_node_features,
    parse_drop_features,
    select_train_graphs,
)
from cpen.apps.streams.cell_stream_split import (
    assign_cells_by_train_streams,
    parse_stream_name_list,
    scan_cell_stream_occupancy,
)
from cpen.apps.streams.preprocess.config import (
    DEFAULT_REAL_PIPELINE_ROOT,
    TEST_REAL_STREAMS,
)

BACKGROUND = "Background"


@dataclass(frozen=True)
class DiscoveryConfig:
    real_root: Path = Path(DEFAULT_REAL_PIPELINE_ROOT)
    galaxy_id: str = "0000"
    split: str = "train"
    node_frame: str = "local"
    drop_features: str = "parallax"
    include_hyperedges: bool = True
    min_stars: int = 16
    rich_min_mock: int = 10
    empty_per_rich: float = 1.0
    cell_val_frac: float = 0.15
    split_seed: int = 0
    min_train_stream_stars: int = 1
    holdout_streams: frozenset[str] = frozenset(TEST_REAL_STREAMS)
    # If True, rebuild the mask-holdout cell assignment (all non-val = train).
    mask_holdout: bool = False


def galaxy_dir(cfg: DiscoveryConfig) -> Path:
    return Path(cfg.real_root) / cfg.split / cfg.galaxy_id


def rebuild_cell_split(
    cfg: DiscoveryConfig,
    *,
    train_streams: str | Sequence[str] | None = None,
    holdout_streams: str | Sequence[str] | None = None,
):
    """Reproduce the fine-tune cell assignment from on-disk cell parquets.

    For ``cfg.mask_holdout``: val carved from train-stream cells; every other
    cell is train; ``test_cells`` = all cells (evaluation universe).
    """
    from cpen.apps.streams.cell_stream_split import (
        CellSplit,
        resolve_train_holdout_streams,
    )

    occ = scan_cell_stream_occupancy(galaxy_dir(cfg), galaxy_id=cfg.galaxy_id)
    holdout = parse_stream_name_list(holdout_streams)
    if holdout is None:
        holdout = set(cfg.holdout_streams)
    train_req = parse_stream_name_list(train_streams)

    if not cfg.mask_holdout:
        return assign_cells_by_train_streams(
            occ,
            train_streams=train_req,
            holdout_streams=holdout,
            min_train_stream_stars=cfg.min_train_stream_stars,
            val_frac=cfg.cell_val_frac,
            seed=cfg.split_seed,
        )

    train, hold = resolve_train_holdout_streams(
        occ, train_streams=train_req, holdout_streams=holdout
    )
    cells_with_train = occ.cells_with_streams(
        train, min_stars=cfg.min_train_stream_stars
    )
    all_cells = set(occ.cell_streams)
    train_list = sorted(cells_with_train)
    rng = np.random.default_rng(int(cfg.split_seed))
    val_frac = float(cfg.cell_val_frac)
    if val_frac > 0.0 and len(train_list) >= 2:
        n_val = int(round(val_frac * len(train_list)))
        n_val = max(1, min(n_val, len(train_list) - 1))
        val_idx = set(rng.choice(len(train_list), n_val, replace=False).tolist())
        val_cells = {train_list[i] for i in val_idx}
    else:
        val_cells = set()
    train_cells = all_cells - val_cells
    return CellSplit(
        train_streams=frozenset(train),
        holdout_streams=frozenset(hold),
        train_cells=frozenset(train_cells),
        val_cells=frozenset(val_cells),
        test_cells=frozenset(all_cells),
        occupancy=occ,
    )


def cell_role(cell_id: int, split, *, mask_holdout: bool = False) -> str:
    """Map a cell to a fit-split role.

    Legacy cell-split: ``train`` / ``val`` / ``test`` (test = discovery cells).
    Mask-holdout: only ``fit_train`` / ``fit_val`` — there is no discovery *cell*
    split; holdout is defined by stream labels.
    """
    cid = int(cell_id)
    if mask_holdout:
        if cid in split.val_cells:
            return "fit_val"
        return "fit_train"
    if cid in split.train_cells:
        return "train"
    if cid in split.val_cells:
        return "val"
    return "test"


def sky_ra_dec_from_raw(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``raw`` columns follow TRAIN_VARS: sin(ra), cos(ra), dec, …"""
    ra = np.degrees(np.arctan2(raw[:, 0], raw[:, 1]))
    ra = (ra + 360.0) % 360.0
    dec = raw[:, 2].astype(np.float64)
    return ra.astype(np.float64), dec


def load_stream_labels(
    cfg: DiscoveryConfig,
    *,
    cell_id: int,
    gmm_label: int,
    blob_part: int,
    n_stars: int,
) -> np.ndarray:
    """Named labels from the blob parquet sidecar (not stored in the npz)."""
    stem = f"blob_c{int(cell_id):04d}_g{int(gmm_label):02d}_p{int(blob_part):02d}"
    path = galaxy_dir(cfg) / "blobs" / f"{stem}.parquet"
    if not path.is_file():
        return np.full(n_stars, BACKGROUND, dtype=object)
    tab = pd.read_parquet(path, columns=["stream_label"])
    labels = tab["stream_label"].astype(str).to_numpy()
    if labels.size != n_stars:
        # Fall back rather than crash if a rare length mismatch appears.
        out = np.full(n_stars, BACKGROUND, dtype=object)
        n = min(n_stars, labels.size)
        out[:n] = labels[:n]
        return out
    return labels


def pick_ckpt(run_dir: str | Path) -> Path:
    run = Path(run_dir)
    best, last = run / "best.ckpt", run / "last.ckpt"
    if best.is_file():
        return best
    if last.is_file():
        return last
    raise FileNotFoundError(f"No best.ckpt/last.ckpt under {run}")


def parse_run_hparams(run_dir: str | Path) -> dict[str, Any]:
    """Recover depth/width/dropout/… from the sweep run-directory name.

    When the run name omits ``drop0p*``, try the sibling parquet's ``dropout``
    column (recent mask-holdout sweeps use dropout=0 and drop the tag).
    """
    run_dir = Path(run_dir)
    name = run_dir.name
    hp: dict[str, Any] = {
        "depth": 4,
        "width": 256,
        "dropout": 0.0,
        "eta_0": 0.01,
        "residual_structure": "transformer-like",
        "operator_normalization": "degree",
        "layer_norm": True,
    }
    m = re.search(r"cpen_(\d+)_(\d+)_", name)
    if m:
        hp["depth"] = int(m.group(1))
        hp["width"] = int(m.group(2))
    m = re.search(r"_0p(\d+)_t", name)
    if m:
        # etas tag: 0p01 → 0.01
        digits = m.group(1)
        hp["eta_0"] = float(f"0.{digits}")
    m = re.search(r"drop0p(\d+)", name)
    if m:
        hp["dropout"] = float(f"0.{m.group(1)}")
    else:
        pq = run_dir.parent / f"{name}.parquet"
        if pq.is_file():
            try:
                tab = pd.read_parquet(pq, columns=["dropout"])
                if len(tab) and pd.notna(tab["dropout"].iloc[-1]):
                    hp["dropout"] = float(tab["dropout"].iloc[-1])
            except Exception:
                pass
    return hp


def load_lit_from_run(
    run_dir: str | Path,
    *,
    cfg: DiscoveryConfig,
    device: str | torch.device | None = None,
):
    """Build CPEN + LitCPENStream matching the fine-tune recipe and load weights."""
    from cpen.lit_models.lit_cpen_stream import LitCPENStream
    from cpen.models.cpen import CPEN

    run_dir = Path(run_dir)
    ckpt = pick_ckpt(run_dir)
    hp = parse_run_hparams(run_dir)
    drop = parse_drop_features(cfg.drop_features)
    nf = n_node_features(cfg.node_frame, drop)
    ne = n_edge_features(drop)
    model = CPEN(
        n_features=nf,
        n_edge_features=ne,
        out_dim=2,
        depth=int(hp["depth"]),
        width=int(hp["width"]),
        operators="incidence",
        normalization="uniform",
        residual_structure=str(hp["residual_structure"]),
        operator_normalization=str(hp["operator_normalization"]),
        operator_backend="sparse",
        layer_norm=bool(hp["layer_norm"]),
        dropout=float(hp["dropout"]),
        readout_mode="node+edge",
    )
    lit = LitCPENStream.load_from_checkpoint(
        str(ckpt),
        model=model,
        eta_0=float(hp["eta_0"]),
        model_name="cpen",
        optimizer="adamw",
        dataset="stream",
        depth=int(hp["depth"]),
        width=int(hp["width"]),
        map_location="cpu",
        strict=False,
    )
    lit.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    lit.to(device)
    return lit, ckpt, hp, device


def fit_stats_for_cfg(cfg: DiscoveryConfig):
    """Feature stats from the same train-cell rich/empty pool used at fine-tune."""
    root = Path(cfg.real_root)
    man = load_split_manifest(root, cfg.split, galaxy_ids=[cfg.galaxy_id])
    split = rebuild_cell_split(cfg)
    train_raw = man[man["cell_id"].astype(int).isin(split.train_cells)]
    train_man = select_train_graphs(
        train_raw,
        rich_min_mock=cfg.rich_min_mock,
        empty_per_rich=cfg.empty_per_rich,
        min_stars=cfg.min_stars,
        seed=cfg.split_seed,
    )
    return fit_feature_stats(
        list(train_man["path"]),
        node_frame=cfg.node_frame,
        drop_features=cfg.drop_features,
        include_hyperedges=cfg.include_hyperedges,
        seed=cfg.split_seed,
    ), split, train_man


@torch.inference_mode()
def score_blob(
    path: str | Path,
    *,
    lit,
    stats,
    cfg: DiscoveryConfig,
    device: str | torch.device,
    cell_split,
) -> dict[str, Any]:
    """Forward one blob graph; attach sky coords, S5 labels, and cell role."""
    path = Path(path)
    with np.load(path) as z:
        raw = z["x"].astype(np.float32)
        y = z["y"].astype(np.int64)
        cell_id = int(z["cell_id"])
        gmm_label = int(z["gmm_label"])
        blob_part = int(z["blob_part"])
    ra, dec = sky_ra_dec_from_raw(raw)
    labels = load_stream_labels(
        cfg,
        cell_id=cell_id,
        gmm_label=gmm_label,
        blob_part=blob_part,
        n_stars=int(y.size),
    )
    payload = blob_to_incidence_payload(
        path,
        role="test",
        node_frame=cfg.node_frame,
        drop_features=cfg.drop_features,
        stats=stats,
        include_hyperedges=cfg.include_hyperedges,
    )
    batch = {
        k: payload[k].unsqueeze(0).to(device)
        for k in (
            "x",
            "edge_x",
            "incidence_node",
            "incidence_edge",
            "incidence_nnz",
            "node_degree_inv",
            "edge_degree_inv",
            "mask",
        )
        if k in payload
    }
    node_logits, _edge_logits = lit.model(batch["x"], **lit._model_kwargs(batch))
    p_node = F.softmax(node_logits[0], dim=-1)[:, 1].detach().cpu().numpy()
    role = cell_role(cell_id, cell_split, mask_holdout=bool(cfg.mask_holdout))
    is_s5 = labels != BACKGROUND
    return {
        "path": str(path),
        "cell_id": cell_id,
        "gmm_label": gmm_label,
        "blob_part": blob_part,
        "role": role,
        "n_stars": int(y.size),
        "n_s5": int(is_s5.sum()),
        "y": y,
        "p_node": p_node.astype(np.float32),
        "ra": ra,
        "dec": dec,
        "labels": labels,
        "mean_p": float(p_node.mean()),
        "mean_p_s5": float(p_node[is_s5].mean()) if is_s5.any() else float("nan"),
    }


def score_manifest(
    cfg: DiscoveryConfig,
    *,
    lit,
    stats,
    cell_split,
    device: str | torch.device,
    max_graphs: int | None = None,
    progress: bool = True,
) -> list[dict[str, Any]]:
    man = load_split_manifest(
        Path(cfg.real_root), cfg.split, galaxy_ids=[cfg.galaxy_id]
    )
    man = man[man["n_stars"] >= int(cfg.min_stars)].reset_index(drop=True)
    if max_graphs is not None and max_graphs > 0:
        man = man.iloc[: int(max_graphs)].copy()
    rows: list[dict[str, Any]] = []
    iterator: Iterable[Any] = man.itertuples(index=False)
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(list(man.itertuples(index=False)), desc="score blobs")
        except Exception:
            iterator = man.itertuples(index=False)
    for row in iterator:
        rows.append(
            score_blob(
                row.path,
                lit=lit,
                stats=stats,
                cfg=cfg,
                device=device,
                cell_split=cell_split,
            )
        )
    return rows


def apply_threshold(
    scored: Sequence[dict[str, Any]],
    threshold: float,
) -> pd.DataFrame:
    """Per-star table with discovery labels at ``threshold``."""
    frames = []
    t = float(threshold)
    for r in scored:
        pred = r["p_node"] >= t
        y = r["y"].astype(bool)
        labels = r["labels"]
        frames.append(
            pd.DataFrame(
                {
                    "cell_id": r["cell_id"],
                    "gmm_label": r["gmm_label"],
                    "blob_part": r["blob_part"],
                    "role": r["role"],
                    "ra": r["ra"],
                    "dec": r["dec"],
                    "p": r["p_node"],
                    "y": y,
                    "stream_label": labels,
                    "pred": pred,
                    "tp": pred & y,
                    "fp": pred & ~y,
                    "fn": (~pred) & y,
                    "tn": (~pred) & ~y,
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def stream_recovery_table(stars: pd.DataFrame, *, roles: Sequence[str] | None = None) -> pd.DataFrame:
    """Per-stream recall / precision-like counts, optionally restricted by cell role."""
    tab = stars if roles is None else stars[stars["role"].isin(list(roles))]
    rows = []
    for name, sub in tab.groupby("stream_label", sort=False):
        if name == BACKGROUND:
            continue
        n = len(sub)
        tp = int(sub["tp"].sum())
        fn = int(sub["fn"].sum())
        rows.append(
            {
                "stream_label": name,
                "n_members": n,
                "n_tp": tp,
                "n_fn": fn,
                "recall": tp / max(n, 1),
                "n_cells": int(sub["cell_id"].nunique()),
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("n_members", ascending=False).reset_index(drop=True)
    return out


def choose_threshold_on_val(
    scored: Sequence[dict[str, Any]],
    *,
    grid: np.ndarray | None = None,
    default_t: float = 0.5,
) -> tuple[float, pd.DataFrame]:
    """Max-F1 threshold on val-cell stars only (no peeking at discovery test)."""
    from sklearn.metrics import f1_score

    grid = np.linspace(0.05, 0.95, 19) if grid is None else np.asarray(grid)

    def _pack(roles: set[str]) -> tuple[np.ndarray, np.ndarray]:
        ps = [r["p_node"] for r in scored if r["role"] in roles]
        ys = [r["y"] for r in scored if r["role"] in roles]
        if not ps:
            return np.array([], dtype=np.float32), np.array([], dtype=int)
        return np.concatenate(ps), np.concatenate(ys).astype(int)

    p, y = _pack({"val"})
    if p.size == 0 or y.sum() == 0 or y.sum() == y.size:
        # Degenerate / empty val (e.g. dry-run): fall back to supervised cells.
        p, y = _pack({"train", "val"})
    if p.size == 0:
        empty = pd.DataFrame(
            [{"t": float(t), "f1": 0.0, "prec": 0.0, "rec": 0.0} for t in grid]
        )
        return float(default_t), empty

    rows = []
    for t in grid:
        pred = (p >= float(t)).astype(int)
        rows.append(
            {
                "t": float(t),
                "f1": float(f1_score(y, pred, zero_division=0)),
                "prec": float(((pred == 1) & (y == 1)).sum() / max(int(pred.sum()), 1)),
                "rec": float(((pred == 1) & (y == 1)).sum() / max(int(y.sum()), 1)),
            }
        )
    sweep = pd.DataFrame(rows)
    t_star = float(sweep.loc[sweep["f1"].idxmax(), "t"])
    return t_star, sweep


def choose_threshold_supervised(
    stars_or_scored: pd.DataFrame | Sequence[dict[str, Any]],
    *,
    holdout_streams: Sequence[str] | set[str] | frozenset[str],
    roles: Sequence[str] = ("fit_val", "val"),
    grid: np.ndarray | None = None,
    default_t: float = 0.5,
) -> tuple[float, pd.DataFrame]:
    """Max-F1 threshold on non-holdout stars in ``roles`` (usually fit_val/val)."""
    from sklearn.metrics import f1_score

    hold = {str(s) for s in holdout_streams}
    grid = np.linspace(0.05, 0.95, 19) if grid is None else np.asarray(grid)

    if isinstance(stars_or_scored, pd.DataFrame):
        tab = stars_or_scored[stars_or_scored["role"].isin(list(roles))]
        keep = ~tab["stream_label"].astype(str).isin(hold)
        p = tab.loc[keep, "p"].to_numpy(dtype=np.float32)
        y = tab.loc[keep, "y"].astype(int).to_numpy()
    else:
        ps, ys = [], []
        for r in stars_or_scored:
            if r["role"] not in set(roles):
                continue
            labels = np.asarray(r["labels"]).astype(str)
            m = ~np.isin(labels, list(hold))
            if not m.any():
                continue
            ps.append(r["p_node"][m])
            ys.append(r["y"][m])
        if not ps:
            p, y = np.array([], dtype=np.float32), np.array([], dtype=int)
        else:
            p = np.concatenate(ps)
            y = np.concatenate(ys).astype(int)

    if p.size == 0 or y.sum() == 0 or y.sum() == y.size:
        empty = pd.DataFrame(
            [{"t": float(t), "f1": 0.0, "prec": 0.0, "rec": 0.0} for t in grid]
        )
        return float(default_t), empty

    rows = []
    for t in grid:
        pred = (p >= float(t)).astype(int)
        rows.append(
            {
                "t": float(t),
                "f1": float(f1_score(y, pred, zero_division=0)),
                "prec": float(((pred == 1) & (y == 1)).sum() / max(int(pred.sum()), 1)),
                "rec": float(((pred == 1) & (y == 1)).sum() / max(int(y.sum()), 1)),
            }
        )
    sweep = pd.DataFrame(rows)
    return float(sweep.loc[sweep["f1"].idxmax(), "t"]), sweep


def cell_has_other_streams(
    occupancy,
    cell_id: int,
    stream_label: str,
) -> bool:
    counts = occupancy.cell_streams.get(int(cell_id), {})
    return any(k != stream_label and int(v) > 0 for k, v in counts.items())


def holdout_recovery_breakdown(
    stars: pd.DataFrame,
    *,
    holdout_streams: Sequence[str] | set[str] | frozenset[str],
    occupancy,
) -> pd.DataFrame:
    """Per-holdout recall split into solo-cell vs co-located members."""
    hold = {str(s) for s in holdout_streams}
    rows = []
    for name in sorted(hold):
        sub = stars[stars["stream_label"].astype(str) == name]
        if not len(sub):
            continue
        coloc = sub["cell_id"].map(
            lambda c, n=name: cell_has_other_streams(occupancy, int(c), n)
        )
        for tag, part in (("solo", sub[~coloc]), ("colocated", sub[coloc]), ("all", sub)):
            n = len(part)
            if n == 0 and tag != "all":
                rows.append(
                    {
                        "stream_label": name,
                        "subset": tag,
                        "n_members": 0,
                        "n_tp": 0,
                        "n_fn": 0,
                        "recall": float("nan"),
                        "mean_p": float("nan"),
                        "n_cells": 0,
                    }
                )
                continue
            tp = int(part["tp"].sum()) if "tp" in part.columns else 0
            fn = int(part["fn"].sum()) if "fn" in part.columns else 0
            rows.append(
                {
                    "stream_label": name,
                    "subset": tag,
                    "n_members": n,
                    "n_tp": tp,
                    "n_fn": fn,
                    "recall": tp / max(n, 1),
                    "mean_p": float(part["p"].mean()) if n else float("nan"),
                    "n_cells": int(part["cell_id"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def open_discovery_candidates(
    stars: pd.DataFrame,
    *,
    threshold: float,
    catalog_streams: Sequence[str] | set[str] | frozenset[str] | None = None,
    min_stars: int = 20,
    min_mean_p: float = 0.3,
) -> pd.DataFrame:
    """High-score islands among non-catalog stars (candidate new structure).

    Groups predicted-positive field stars by cell; does not treat them as FPs.
    """
    catalog = (
        {str(s) for s in catalog_streams}
        if catalog_streams is not None
        else set(stars.loc[stars["y"].astype(bool), "stream_label"].astype(str).unique())
        - {BACKGROUND}
    )
    field = stars[
        (~stars["stream_label"].astype(str).isin(catalog))
        & (stars["stream_label"].astype(str) != BACKGROUND)
    ]
    # Prefer pure Background predicted positives
    bg_pred = stars[
        (stars["stream_label"].astype(str) == BACKGROUND) & (stars["p"] >= float(threshold))
    ]
    rows = []
    for cid, sub in bg_pred.groupby("cell_id"):
        rows.append(
            {
                "cell_id": int(cid),
                "n_pred": len(sub),
                "mean_p": float(sub["p"].mean()),
                "max_p": float(sub["p"].max()),
                "n_catalog_in_cell": int(
                    stars[
                        (stars["cell_id"] == cid)
                        & stars["stream_label"].astype(str).isin(catalog)
                    ].shape[0]
                ),
            }
        )
    out = pd.DataFrame(rows)
    if not len(out):
        return out
    out = out[(out["n_pred"] >= int(min_stars)) & (out["mean_p"] >= float(min_mean_p))]
    return out.sort_values(["mean_p", "n_pred"], ascending=False).reset_index(drop=True)
