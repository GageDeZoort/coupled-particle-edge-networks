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
    ) -> None:
        root = str(data_root or DEFAULT_BLOB_DATA_DIR)
        if batch_size != 1:
            raise ValueError(
                f"blob graphs vary in size; batch_size must be 1, got {batch_size}"
            )
        if node_frame not in {"local", "raw"}:
            raise ValueError(f"node_frame must be 'local' or 'raw'; got {node_frame!r}")
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
        self.task = "node-edge-cls"
        self.graph_construction = "blob-knn8-hyper" if self.include_hyperedges else "blob-knn8"

        ids = None
        if galaxy_ids:
            ids = [g for g in re.split(r"[,\s]+", str(galaxy_ids).strip()) if g]
        self.galaxy_ids = ids

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

        # Standardization statistics come from the train pool only.
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

    # -- logging ----------------------------------------------------------

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
        )
        log_info(
            f"[blobs] train (balanced, rich = n_mock>={self.rich_min_mock}, "
            f"roles={roles}): {self._describe(self._train_man)}"
        )
        log_info(f"[blobs] val   (natural occupancy): {self._describe(self._val_man)}")
        log_info(f"[blobs] test  (natural occupancy): {self._describe(self._test_man)}")
        # Graphs are balanced 50/50 by construction, but stars inside a host are
        # still a minority, so the CE weights carry the within-graph imbalance.
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

    # -- BaseDatamodule ---------------------------------------------------

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
        }

    def run_metadata(self, split_sizes: dict[str, int]) -> dict:
        return {
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
            "supervision": (
                "graph-level split by galaxy; node CE + kNN/hyper edge CE "
                f"(hyper_y = member purity ≥ 0.5) on train graphs (balanced: "
                f"rich n_mock>={self.rich_min_mock} vs size-matched empty); "
                "val/test at natural occupancy, discovery AUROC over the whole "
                "held-out graph"
            ),
            "num_workers": self.num_workers,
            **split_sizes,
        }

    def metadata(self) -> dict:
        return self.run_metadata(self.probe_split_sizes())
