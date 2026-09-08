"""Stellar-stream semisupervised multi-MOCK datamodule."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

from cpen.datamodules.base_datamodule import BaseDatamodule
from cpen.utils.log_utils import log_info
from cpen.utils.stream_graph_cache import (
    DEFAULT_STREAM_DATA_DIR,
    N_EDGE_FEATURES,
    N_FEATURES,
    OUT_DIM,
    MultiStreamGraphDataset,
    aggregate_class_weights,
    aggregate_edge_class_weights,
    load_stream_pyg,
    pyg_to_incidence_payload,
    resolve_stream_graph_path,
    resolve_stream_names,
)


class StreamDatamodule(BaseDatamodule):
    """
    Stream graphs (MOCK by default, optionally real too), one graph per step.

    Within each graph:
      - train: CE on ``train_mask`` nodes and ``edge_train_mask`` edges
      - val/test: CE on holdout masks; discovery AUROC on ``~train_mask``

    By default, on-disk obvious stream stars (``train_mask ∧ y=1``) are
    re-partitioned ~⅓/⅓/⅓ into train/val/test with a deterministic seed.
    Obvious background stays in train; former hard val/test members are
    unlabeled for discovery.

    Graphs differ in size, so ``batch_size`` must stay 1 (gradient after each graph).
    """

    N_FEATURES = N_FEATURES
    N_EDGE_FEATURES = N_EDGE_FEATURES
    OUT_DIM = OUT_DIM

    def __init__(
        self,
        data_root: str | None = None,
        batch_size: int = 1,
        *,
        stream_name: str = "all",
        include_real_streams: bool = False,
        num_workers: int | None = 0,
        include_incidence: bool = True,
        include_edge_features: bool = True,
        repartition_obvious: bool = True,
        mask_seed: int = 0,
    ) -> None:
        root = str(data_root or DEFAULT_STREAM_DATA_DIR)
        if batch_size != 1:
            raise ValueError(
                f"StreamDatamodule uses one graph per step; batch_size must be 1, got {batch_size}"
            )
        super().__init__(root, batch_size, num_workers=num_workers)
        self.stream_name = stream_name
        self.include_real_streams = bool(include_real_streams)
        self.include_incidence = include_incidence
        self.include_edge_features = include_edge_features
        self.repartition_obvious = bool(repartition_obvious)
        self.mask_seed = int(mask_seed)
        self.task = "node-edge-cls"
        self.graph_construction = "native-edges"

        self.stream_names = resolve_stream_names(
            self.data_root,
            stream_name,
            include_real=self.include_real_streams,
        )
        self._graph_paths = [
            resolve_stream_graph_path(self.data_root, stream_name=name)
            for name in self.stream_names
        ]
        self._payloads: list[dict] = []
        self._stream_labels: list[str] = []
        self._edge_variants: list[str] = []
        obvious_tr = obvious_va = obvious_te = hard_u = 0
        for path, name in zip(self._graph_paths, self.stream_names):
            raw = load_stream_pyg(path)
            payload = pyg_to_incidence_payload(
                raw,
                repartition_obvious=self.repartition_obvious,
                mask_seed=self.mask_seed,
                stream_key=name,
            )
            self._payloads.append(payload)
            self._stream_labels.append(str(getattr(raw, "stream_label", name)))
            self._edge_variants.append(str(getattr(raw, "edge_variant", "")))
            stats = payload.get("repartition_stats")
            if isinstance(stats, dict):
                obvious_tr += int(stats.get("obvious_stream_train", 0))
                obvious_va += int(stats.get("obvious_stream_val", 0))
                obvious_te += int(stats.get("obvious_stream_test", 0))
                hard_u += int(stats.get("hard_unlabeled", 0))

        self.class_weights = aggregate_class_weights(self._payloads)
        self.edge_class_weights = aggregate_edge_class_weights(self._payloads)
        self.n_particles = max(int(p["n_nodes"].item()) for p in self._payloads)
        self.n_val = len(self._payloads)
        self.n_test = len(self._payloads)
        self.stream_label = ",".join(self._stream_labels[:3]) + (
            f",+{len(self._stream_labels) - 3}" if len(self._stream_labels) > 3 else ""
        )
        self.edge_variant = self._edge_variants[0] if self._edge_variants else ""

        n_train_nodes = sum(int(p["train_mask"].sum()) for p in self._payloads)
        n_val_nodes = sum(int(p["val_mask"].sum()) for p in self._payloads)
        n_test_nodes = sum(int(p["test_mask"].sum()) for p in self._payloads)
        n_edge_train = sum(int(p["edge_train_mask"].sum()) for p in self._payloads)
        log_info(
            f"[streams] loaded {len(self._payloads)} graphs from {Path(self.data_root).name} "
            f"(max N={self.n_particles}); "
            f"supervised nodes train/val/test={n_train_nodes}/{n_val_nodes}/{n_test_nodes}; "
            f"supervised edges(train both-ends)={n_edge_train}"
        )
        if self.repartition_obvious:
            log_info(
                f"[streams] obvious-stream repartition seed={self.mask_seed}: "
                f"y=1 train/val/test={obvious_tr}/{obvious_va}/{obvious_te}; "
                f"hard unlabeled={hard_u}"
            )

    def dataset_name(self) -> str:
        return "stream"

    def _split_dataset(self, split: str) -> Dataset:
        return MultiStreamGraphDataset(
            self._payloads,
            self.stream_names,
            split=split,
            include_incidence=self.include_incidence,
            include_edge_features=self.include_edge_features,
        )

    def build_datasets(self) -> tuple[Dataset, Dataset, Dataset]:
        return (
            self._split_dataset("train"),
            self._split_dataset("val"),
            self._split_dataset("test"),
        )

    def _build_train_dataset(self) -> Dataset:
        return self._split_dataset("train")

    def _build_val_dataset(self) -> Dataset:
        return self._split_dataset("val")

    def _build_test_dataset(self) -> Dataset:
        return self._split_dataset("test")

    def probe_split_sizes(self) -> dict[str, int]:
        return {
            "n_train": sum(int(p["train_mask"].sum()) for p in self._payloads),
            "n_val": sum(int(p["val_mask"].sum()) for p in self._payloads),
            "n_test": sum(int(p["test_mask"].sum()) for p in self._payloads),
            "n_edge_train": sum(int(p["edge_train_mask"].sum()) for p in self._payloads),
            "n_nodes": sum(int(p["n_nodes"].item()) for p in self._payloads),
            "n_edges": sum(int(p["n_edges"].item()) for p in self._payloads),
            "n_graphs": len(self._payloads),
        }

    def get_dims(self) -> tuple[int, int]:
        return self.N_FEATURES, self.OUT_DIM

    def get_edge_dim(self) -> int:
        return self.N_EDGE_FEATURES

    def planned_graph_mode(self) -> tuple[str, None]:
        return "cache", None

    def run_metadata(self, split_sizes: dict[str, int]) -> dict:
        return {
            "dataset": "stream",
            "benchmark": (
                "stellar-stream-mock+real"
                if self.include_real_streams
                or (
                    isinstance(self.stream_name, str)
                    and self.stream_name.strip().lower()
                    in {"real", "reals", "mock+real", "real+mock", "both", "all+real", "all_streams"}
                )
                else "stellar-stream-mock"
            ),
            "task": self.task,
            "stream_name": self.stream_name,
            "include_real_streams": self.include_real_streams,
            "stream_names": ",".join(self.stream_names),
            "stream_label": self.stream_label,
            "edge_variant": self.edge_variant,
            "graph_path": str(Path(self.data_root)),
            "graph_construction": self.graph_construction,
            "graph_cache": False,
            "graph_loading": "pyg-live-multi",
            "edge_mode": "native-edges",
            "uses_edge_features": self.include_edge_features,
            "n_edge_features": self.N_EDGE_FEATURES,
            "n_features": self.N_FEATURES,
            "repartition_obvious": self.repartition_obvious,
            "mask_seed": self.mask_seed,
            "supervision": (
                "per-mock: node CE on train_mask, edge CE on edge_train_mask; "
                "val/test on holdout masks; discovery AUROC on ~train_mask"
                + (
                    "; obvious-stream stars split ~1/3 across train/val/test"
                    if self.repartition_obvious
                    else ""
                )
            ),
            "num_workers": self.num_workers,
            **split_sizes,
        }

    def metadata(self) -> dict:
        return self.run_metadata(self.probe_split_sizes())
