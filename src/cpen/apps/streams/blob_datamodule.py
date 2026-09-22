"""Blob-graph datamodule: balanced train pool, natural-occupancy val/test."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset

from cpen.apps.streams.blob_graph_cache import (
    DEFAULT_BLOB_DATA_DIR,
    OUT_DIM,
    BlobGraphDataset,
    edge_class_weights,
    fit_feature_stats,
    kept_edge_columns,
    kept_feature_columns,
    load_split_manifest,
    n_edge_features,
    n_node_features,
    node_class_weights,
    parse_drop_features,
    select_eval_graphs,
    select_train_graphs,
)
from cpen.apps.streams.cell_stream_split import (
    CellSplit,
    assign_cells_by_train_streams,
    describe_cell_split,
    parse_stream_name_list,
    partition_manifest_by_cells,
    resolve_train_holdout_streams,
    scan_cell_stream_occupancy,
)
from cpen.datamodules.base_datamodule import BaseDatamodule
from cpen.utils.log_utils import log_info


class BlobStreamDatamodule(BaseDatamodule):
    """
    One GMM blob per step, split by galaxy (train/val/test galaxies are disjoint).

    Train is rebalanced — mock-rich hosts (``n_mock >= rich_min_mock``) against a
    size-matched sample of empty blobs — because the raw pool is ~97% empty and
    almost every step would otherwise carry no signal. Val and test keep the
    natural occupancy, which is the prior the model actually runs at.

    Every star is labeled, so a whole graph lands in one split: ``train_mask`` is
    all-True on train graphs and all-False on val/test graphs. The discovery pool
    ``~train_mask`` is therefore the entire val/test graph.

    Real-stream fine-tune modes
      ``cell_split=True``
        Supervise only cells that contain *train* streams; other cells = discovery
        test. Holdout members that co-occupy train cells remain ``y=1`` (legacy).

      ``mask_holdout=True`` (preferred)
        Load **all** cells; carve val from train-stream cells for early stopping.
        Holdout stream members are excluded from CE via per-star masks (labels
        withheld, sky kept). Discovery AUROC = recovery on ``~train_mask``.
    """

    OUT_DIM = OUT_DIM

    def __init__(
        self,
        data_root: str | None = None,
        batch_size: int = 1,
        *,
        num_workers: int | None = 0,
        node_frame: str = "local",
        drop_features: str | None = None,
        rich_min_mock: int = 100,
        empty_per_rich: float = 1.0,
        include_dilute: bool = False,
        min_stars: int = 16,
        max_val_graphs: int | None = 2000,
        max_test_graphs: int | None = None,
        galaxy_ids: str | None = None,
        split_seed: int = 0,
        include_incidence: bool = True,
        include_edge_features: bool = True,
        include_hyperedges: bool = True,
        max_cache: int = 4096,
        cell_split: bool = False,
        mask_holdout: bool = False,
        train_streams: str | None = None,
        holdout_streams: str | None = None,
        min_train_stream_stars: int = 1,
        cell_val_frac: float = 0.15,
    ) -> None:
        root = str(data_root or DEFAULT_BLOB_DATA_DIR)
        if batch_size != 1:
            raise ValueError(
                f"blob graphs vary in size; batch_size must be 1, got {batch_size}"
            )
        if node_frame not in {"local", "raw"}:
            raise ValueError(f"node_frame must be 'local' or 'raw'; got {node_frame!r}")
        if cell_split and mask_holdout:
            raise ValueError(
                "Pass only one of cell_split / mask_holdout "
                "(mask_holdout is the preferred real fine-tune protocol)"
            )
        super().__init__(root, batch_size, num_workers=num_workers)

        self.node_frame = node_frame
        self.drop_features = parse_drop_features(drop_features)
        _, self.feature_names = kept_feature_columns(node_frame, self.drop_features)
        _, self.edge_feature_names = kept_edge_columns(self.drop_features)
        self.rich_min_mock = int(rich_min_mock)
        self.empty_per_rich = float(empty_per_rich)
        self.include_dilute = bool(include_dilute)
        self.min_stars = int(min_stars)
        self.split_seed = int(split_seed)
        self.include_incidence = bool(include_incidence)
        self.include_edge_features = bool(include_edge_features)
        self.include_hyperedges = bool(include_hyperedges)
        self.max_cache = int(max_cache)
        self.cell_split = bool(cell_split)
        self.mask_holdout = bool(mask_holdout)
        self.min_train_stream_stars = int(min_train_stream_stars)
        self.cell_val_frac = float(cell_val_frac)
        self.task = "node-edge-cls"
        self.graph_construction = "blob-knn8-hyper" if self.include_hyperedges else "blob-knn8"
        self.cell_split_info: CellSplit | None = None
        self.holdout_stream_set: frozenset[str] | None = None

        ids = None
        if galaxy_ids:
            ids = [g for g in re.split(r"[,\s]+", str(galaxy_ids).strip()) if g]
        self.galaxy_ids = ids

        if self.mask_holdout:
            self._init_mask_holdout_pools(
                root,
                ids,
                train_streams=parse_stream_name_list(train_streams),
                holdout_streams=parse_stream_name_list(holdout_streams),
                max_val_graphs=max_val_graphs,
                max_test_graphs=max_test_graphs,
            )
        elif self.cell_split:
            self._init_cell_split_pools(
                root,
                ids,
                train_streams=parse_stream_name_list(train_streams),
                holdout_streams=parse_stream_name_list(holdout_streams),
                max_val_graphs=max_val_graphs,
                max_test_graphs=max_test_graphs,
            )
        else:
            self._train_man = select_train_graphs(
                load_split_manifest(root, "train", galaxy_ids=ids),
                rich_min_mock=self.rich_min_mock,
                empty_per_rich=self.empty_per_rich,
                include_dilute=self.include_dilute,
                min_stars=self.min_stars,
                seed=self.split_seed,
            )
            self._val_man = select_eval_graphs(
                load_split_manifest(root, "val", galaxy_ids=ids),
                max_graphs=max_val_graphs,
                min_stars=self.min_stars,
                seed=self.split_seed,
            )
            self._test_man = select_eval_graphs(
                load_split_manifest(root, "test", galaxy_ids=ids),
                max_graphs=max_test_graphs,
                min_stars=self.min_stars,
                seed=self.split_seed,
            )

        self.feature_stats = fit_feature_stats(
            list(self._train_man["path"]),
            node_frame=self.node_frame,
            drop_features=self.drop_features,
            seed=self.split_seed,
            include_hyperedges=self.include_hyperedges,
        )
        self.class_weights = node_class_weights(self._train_man)
        self.edge_class_weights = edge_class_weights(self._train_man)
        self.n_particles = int(self._train_man["n_stars"].max())
        self.n_val = len(self._val_man)
        self.n_test = len(self._test_man)

        self._log_pools()

    def _resolve_single_galaxy(
        self, root: str, galaxy_ids: list[str] | None
    ) -> tuple[str, Path]:
        ids = galaxy_ids or ["0000"]
        if len(ids) != 1:
            raise ValueError(
                f"real fine-tune expects exactly one galaxy id, got {ids!r}"
            )
        gid = ids[0]
        galaxy_dir = Path(root) / "train" / gid
        if not galaxy_dir.is_dir():
            raise FileNotFoundError(
                f"galaxy dir missing: {galaxy_dir} "
                f"(expected real pipeline under data_root/train/{gid})"
            )
        return gid, galaxy_dir

    def _init_mask_holdout_pools(
        self,
        root: str,
        galaxy_ids: list[str] | None,
        *,
        train_streams: set[str] | None,
        holdout_streams: set[str] | None,
        max_val_graphs: int | None,
        max_test_graphs: int | None,
    ) -> None:
        """All cells in train/test; holdout stream *labels* masked from loss."""
        import numpy as np

        gid, galaxy_dir = self._resolve_single_galaxy(root, galaxy_ids)
        occupancy = scan_cell_stream_occupancy(galaxy_dir, galaxy_id=gid)
        train, holdout = resolve_train_holdout_streams(
            occupancy,
            train_streams=train_streams,
            holdout_streams=holdout_streams,
        )
        self.holdout_stream_set = frozenset(holdout)

        cells_with_train = occupancy.cells_with_streams(
            train, min_stars=self.min_train_stream_stars
        )
        all_cells = set(occupancy.cell_streams)
        if not cells_with_train:
            raise ValueError(
                f"No cells with ≥{self.min_train_stream_stars} members of "
                f"{sorted(train)} in galaxy {gid}"
            )

        train_list = sorted(cells_with_train)
        rng = np.random.default_rng(int(self.split_seed))
        val_frac = float(self.cell_val_frac)
        if val_frac < 0.0 or val_frac >= 1.0:
            raise ValueError(f"cell_val_frac must be in [0, 1); got {val_frac}")
        if val_frac > 0.0 and len(train_list) >= 2:
            n_val = int(round(val_frac * len(train_list)))
            n_val = max(1, min(n_val, len(train_list) - 1))
            val_idx = set(rng.choice(len(train_list), n_val, replace=False).tolist())
            val_cells = {train_list[i] for i in val_idx}
        else:
            val_cells = set()
        # Train on every non-val cell (incl. holdout-only + empty).
        train_cells = all_cells - val_cells
        # Lightning test = full galaxy (notebook is the real discovery eval).
        test_cells = set(all_cells)

        split = CellSplit(
            train_streams=frozenset(train),
            holdout_streams=frozenset(holdout),
            train_cells=frozenset(train_cells),
            val_cells=frozenset(val_cells),
            test_cells=frozenset(test_cells),
            occupancy=occupancy,
        )
        self.cell_split_info = split
        log_info(
            f"[blobs] mask-holdout: {describe_cell_split(split)} | "
            f"train_cells={len(train_cells)} (all non-val) val={len(val_cells)} "
            f"test=all({len(test_cells)}) | labels masked: {sorted(holdout)}"
        )

        full = load_split_manifest(root, "train", galaxy_ids=[gid])
        rich_cells = cells_with_train - val_cells
        train_raw = full[full["cell_id"].astype(int).isin(train_cells)].copy()
        train_rich_pref = train_raw[train_raw["cell_id"].astype(int).isin(rich_cells)]
        if len(train_rich_pref):
            preferred_paths = set(train_rich_pref["path"])
            empty = train_raw[train_raw["n_mock"] == 0]
            richish = train_raw[
                (train_raw["n_mock"] >= self.rich_min_mock)
                & train_raw["path"].isin(preferred_paths)
            ]
            other = train_raw[
                ~train_raw["path"].isin(set(richish["path"]) | set(empty["path"]))
            ]
            pool_for_balance = (
                pd.concat([richish, empty, other], ignore_index=True)
                if len(richish)
                else train_raw
            )
        else:
            pool_for_balance = train_raw

        self._train_man = select_train_graphs(
            pool_for_balance,
            rich_min_mock=self.rich_min_mock,
            empty_per_rich=self.empty_per_rich,
            include_dilute=self.include_dilute,
            min_stars=self.min_stars,
            seed=self.split_seed,
        )
        val_raw = full[full["cell_id"].astype(int).isin(val_cells)].copy()
        if len(val_raw):
            self._val_man = select_eval_graphs(
                val_raw,
                max_graphs=max_val_graphs,
                min_stars=self.min_stars,
                seed=self.split_seed,
            )
        else:
            self._val_man = select_eval_graphs(
                train_raw,
                max_graphs=min(int(max_val_graphs or 200), max(32, len(train_raw) // 10)),
                min_stars=self.min_stars,
                seed=self.split_seed + 1,
            )
            log_info(
                "[blobs] mask-holdout: no val cells; using a train-cell subsample"
            )
        self._test_man = select_eval_graphs(
            full,
            max_graphs=max_test_graphs,
            min_stars=self.min_stars,
            seed=self.split_seed,
        )
        log_info(
            f"[blobs] mask-holdout pools: train_graphs={len(self._train_man):,} "
            f"val={len(self._val_man):,} test={len(self._test_man):,}"
        )

    def _init_cell_split_pools(
        self,
        root: str,
        galaxy_ids: list[str] | None,
        *,
        train_streams: set[str] | None,
        holdout_streams: set[str] | None,
        max_val_graphs: int | None,
        max_test_graphs: int | None,
    ) -> None:
        """Legacy: supervise train-stream cells only; rest = discovery test."""
        gid, galaxy_dir = self._resolve_single_galaxy(root, galaxy_ids)
        occupancy = scan_cell_stream_occupancy(galaxy_dir, galaxy_id=gid)
        split = assign_cells_by_train_streams(
            occupancy,
            train_streams=train_streams,
            holdout_streams=holdout_streams,
            min_train_stream_stars=self.min_train_stream_stars,
            val_frac=self.cell_val_frac,
            seed=self.split_seed,
        )
        self.cell_split_info = split
        self.holdout_stream_set = frozenset(split.holdout_streams)
        log_info(f"[blobs] cell-split: {describe_cell_split(split)}")

        full = load_split_manifest(root, "train", galaxy_ids=[gid])
        train_raw, val_raw, test_raw = partition_manifest_by_cells(full, split)
        if not len(train_raw):
            raise ValueError(
                f"No graph rows in train cells {sorted(split.train_cells)[:12]}…"
            )
        self._train_man = select_train_graphs(
            train_raw,
            rich_min_mock=self.rich_min_mock,
            empty_per_rich=self.empty_per_rich,
            include_dilute=self.include_dilute,
            min_stars=self.min_stars,
            seed=self.split_seed,
        )
        if len(val_raw):
            self._val_man = select_eval_graphs(
                val_raw,
                max_graphs=max_val_graphs,
                min_stars=self.min_stars,
                seed=self.split_seed,
            )
        else:
            self._val_man = select_eval_graphs(
                train_raw,
                max_graphs=min(int(max_val_graphs or 200), max(32, len(train_raw) // 10)),
                min_stars=self.min_stars,
                seed=self.split_seed + 1,
            )
            log_info(
                "[blobs] cell-split: no val cells carved out; "
                "using a train-cell subsample for val"
            )
        self._test_man = select_eval_graphs(
            test_raw,
            max_graphs=max_test_graphs,
            min_stars=self.min_stars,
            seed=self.split_seed,
        )
        log_info(
            f"[blobs] cell-split pools: train_graphs={len(self._train_man):,} "
            f"val={len(self._val_man):,} test={len(self._test_man):,} "
            f"(test cells dominate discovery)"
        )

    @staticmethod
    def _describe(man: pd.DataFrame) -> str:
        n_empty = int(man["is_empty"].sum())
        return (
            f"{len(man):,} graphs  empty {100 * n_empty / max(len(man), 1):.0f}%  "
            f"median N={man['n_stars'].median():,.0f}  "
            f"node+ {100 * man['n_mock'].sum() / max(man['n_stars'].sum(), 1):.2f}%  "
            f"knn+ {100 * man['n_knn_pos'].sum() / max(man['n_knn'].sum(), 1):.2f}%"
            + (
                f"  hyper+ {100 * man['n_hyper_pos'].sum() / max(man['n_hyper'].sum(), 1):.2f}%"
                if "n_hyper" in man.columns
                else ""
            )
        )

    def _log_pools(self) -> None:
        roles = self._train_man["role"].value_counts().to_dict()
        mode = (
            "mask-holdout real fine-tune"
            if self.mask_holdout
            else ("cell-split real fine-tune" if self.cell_split else "")
        )
        log_info(
            f"[blobs] node features ({len(self.feature_names)}): "
            f"{', '.join(self.feature_names)}"
            + (f"  [withheld: {', '.join(self.drop_features)}]" if self.drop_features else "")
        )
        log_info(
            f"[blobs] edge features ({len(self.edge_feature_names)}, "
            f"{'kNN+hyper' if self.include_hyperedges else 'kNN only'}): "
            f"{', '.join(self.edge_feature_names)}"
        )
        log_info(
            f"[blobs] root={Path(self.data_root).name} frame={self.node_frame} "
            f"galaxies train/val/test="
            f"{self._train_man.galaxy_id.nunique()}/"
            f"{self._val_man.galaxy_id.nunique()}/"
            f"{self._test_man.galaxy_id.nunique()}"
            + (f"  [{mode}]" if mode else "")
        )
        log_info(
            f"[blobs] train (balanced, rich = n_mock>={self.rich_min_mock}, "
            f"roles={roles}): {self._describe(self._train_man)}"
        )
        log_info(f"[blobs] val   (natural occupancy): {self._describe(self._val_man)}")
        log_info(f"[blobs] test  (natural occupancy): {self._describe(self._test_man)}")
        man = self._train_man
        node_pos = man["n_mock"].sum() / max(man["n_stars"].sum(), 1)
        knn_pos = man["n_knn_pos"].sum() / max(man["n_knn"].sum(), 1)
        extra = f"kNN {100 * knn_pos:.2f}% positive"
        if self.include_hyperedges and "n_hyper" in man.columns:
            hyp_pos = man["n_hyper_pos"].sum() / max(man["n_hyper"].sum(), 1)
            extra += f", hyper {100 * hyp_pos:.2f}% positive"
        log_info(
            f"[blobs] train balance: graphs 50/50 rich:empty, "
            f"nodes {100 * node_pos:.2f}% positive, {extra}"
        )
        log_info(
            f"[blobs] inverse-frequency CE weights "
            f"node=[{self.class_weights[0]:.3f}, {self.class_weights[1]:.3f}] "
            f"edge=[{self.edge_class_weights[0]:.3f}, {self.edge_class_weights[1]:.3f}]"
        )

    def dataset_name(self) -> str:
        return "blob"

    def _split_dataset(self, split: str, man: pd.DataFrame) -> Dataset:
        return BlobGraphDataset(
            list(man["path"]),
            split=split,
            node_frame=self.node_frame,
            drop_features=self.drop_features,
            stats=self.feature_stats,
            include_incidence=self.include_incidence,
            include_edge_features=self.include_edge_features,
            include_hyperedges=self.include_hyperedges,
            max_cache=self.max_cache,
            holdout_streams=self.holdout_stream_set if self.mask_holdout else None,
        )

    def build_datasets(self) -> tuple[Dataset, Dataset, Dataset]:
        return (
            self._split_dataset("train", self._train_man),
            self._split_dataset("val", self._val_man),
            self._split_dataset("test", self._test_man),
        )

    def _build_train_dataset(self) -> Dataset:
        return self._split_dataset("train", self._train_man)

    def _build_val_dataset(self) -> Dataset:
        return self._split_dataset("val", self._val_man)

    def _build_test_dataset(self) -> Dataset:
        return self._split_dataset("test", self._test_man)

    def get_dims(self) -> tuple[int, int]:
        return n_node_features(self.node_frame, self.drop_features), self.OUT_DIM

    def get_edge_dim(self) -> int:
        return n_edge_features(self.drop_features)

    def planned_graph_mode(self) -> tuple[str, None]:
        return "cache", None

    def probe_split_sizes(self) -> dict[str, int]:
        return {
            "n_train": len(self._train_man),
            "n_val": len(self._val_man),
            "n_test": len(self._test_man),
            "n_train_rich": int((self._train_man["role"] == "rich").sum()),
            "n_train_empty": int((self._train_man["role"] == "empty").sum()),
            "n_nodes": int(self._train_man["n_stars"].sum()),
            "n_edges": int(self._train_man["n_knn"].sum()),
            "n_edge_train_pos": int(self._train_man["n_knn_pos"].sum()),
            "n_hyper": int(self._train_man["n_hyper"].sum()) if "n_hyper" in self._train_man.columns else 0,
            "n_hyper_pos": int(self._train_man["n_hyper_pos"].sum()) if "n_hyper_pos" in self._train_man.columns else 0,
            "n_train_cells": len(self.cell_split_info.train_cells) if self.cell_split_info else 0,
            "n_val_cells": len(self.cell_split_info.val_cells) if self.cell_split_info else 0,
            "n_test_cells": len(self.cell_split_info.test_cells) if self.cell_split_info else 0,
        }

    def run_metadata(self, split_sizes: dict[str, int]) -> dict:
        if self.mask_holdout:
            supervision = (
                "real fine-tune label-holdout: all non-val cells in train; "
                "holdout stream members masked from node/edge CE; discovery "
                "AUROC on ~train_mask (= holdout members)"
            )
        elif self.cell_split:
            supervision = (
                "real fine-tune: supervise blobs in cells that contain train "
                f"streams; discovery on all other cells (val carved from "
                f"{self.cell_val_frac:.0%} of train-stream cells)"
            )
        else:
            supervision = (
                "graph-level split by galaxy; node CE + kNN/hyper edge CE "
                f"(hyper_y = member purity ≥ 0.5) on train graphs (balanced: "
                f"rich n_mock>={self.rich_min_mock} vs size-matched empty); "
                "val/test at natural occupancy, discovery AUROC over the whole "
                "held-out graph"
            )
        meta = {
            "dataset": "blob",
            "benchmark": "stellar-stream-blobs",
            "task": self.task,
            "graph_path": str(Path(self.data_root)),
            "graph_construction": self.graph_construction,
            "graph_cache": False,
            "graph_loading": "npz-lazy-blob",
            "edge_mode": "knn+hyper" if self.include_hyperedges else "knn",
            "include_hyperedges": self.include_hyperedges,
            "uses_edge_features": self.include_edge_features,
            "n_edge_features": n_edge_features(self.drop_features),
            "n_features": n_node_features(self.node_frame, self.drop_features),
            "node_frame": self.node_frame,
            "node_features": ",".join(self.feature_names),
            "edge_features": ",".join(self.edge_feature_names),
            "drop_features": ",".join(self.drop_features),
            "rich_min_mock": self.rich_min_mock,
            "empty_per_rich": self.empty_per_rich,
            "include_dilute": self.include_dilute,
            "min_stars": self.min_stars,
            "split_seed": self.split_seed,
            "val_empty_frac": float(self._val_man["is_empty"].mean()),
            "test_empty_frac": float(self._test_man["is_empty"].mean()),
            "cell_split": self.cell_split,
            "mask_holdout": self.mask_holdout,
            "supervision": supervision,
            "num_workers": self.num_workers,
            **split_sizes,
        }
        if self.cell_split_info is not None:
            info = self.cell_split_info
            meta.update(
                {
                    "train_streams": ",".join(sorted(info.train_streams)),
                    "holdout_streams": ",".join(sorted(info.holdout_streams)),
                    "min_train_stream_stars": self.min_train_stream_stars,
                    "cell_val_frac": self.cell_val_frac,
                }
            )
        return meta

    def metadata(self) -> dict:
        return self.run_metadata(self.probe_split_sizes())
