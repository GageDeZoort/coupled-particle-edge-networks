"""MNISTSuperpixels graph-classification datamodule using cached incidence."""

from __future__ import annotations

from torch.utils.data import Dataset

from cpen.apps.mnist.mnist_graph_cache import (
    CachedMNISTGraphDataset,
    load_mnist_meta,
    missing_mnist_cache_message,
    mnist_cache_available,
    mnist_split_size,
)
from cpen.datamodules.base_datamodule import BaseDatamodule
from cpen.utils.log_utils import log_info


class MNISTSuperpixelsDatamodule(BaseDatamodule):
    N_FEATURES = 3
    N_EDGE_FEATURES = 2
    OUT_DIM = 10
    MAX_NODES = 75

    def __init__(
        self,
        data_root: str,
        batch_size: int = 128,
        *,
        num_workers: int | None = None,
        n_train: int | None = None,
        n_val: int | None = None,
        n_test: int | None = None,
        data_seed: int = 42,
        include_incidence: bool = True,
        include_edge_features: bool = True,
    ) -> None:
        super().__init__(data_root, batch_size, num_workers=num_workers)
        self.n_train_limit = n_train
        self.n_val_limit = n_val
        self.n_test_limit = n_test
        self.data_seed = data_seed
        self.include_incidence = include_incidence
        self.include_edge_features = include_edge_features
        self.n_particles = self.MAX_NODES
        self.n_val = 0
        self.n_test = 0
        self.task = "graph-cls"
        self.graph_construction = "native-edges"
        meta = load_mnist_meta(data_root)
        if meta.get("node_features"):
            self.N_FEATURES = int(meta["node_features"])
        if meta.get("edge_features"):
            self.N_EDGE_FEATURES = int(meta["edge_features"])
        if meta.get("out_dim"):
            self.OUT_DIM = int(meta["out_dim"])

    def dataset_name(self) -> str:
        return "mnist"

    def _build_split(self, split: str, limit: int | None) -> Dataset:
        if not mnist_cache_available(self.data_root, split):
            raise FileNotFoundError(missing_mnist_cache_message(self.data_root))
        log_info(
            f"[graphs] split={split} mode=mnist-cache native edges "
            f"edge_features={self.include_edge_features}"
        )
        return CachedMNISTGraphDataset(
            data_root=self.data_root,
            split=split,
            max_graphs=limit,
            seed=self.data_seed,
            include_incidence=self.include_incidence,
            include_edge_features=self.include_edge_features,
        )

    def build_datasets(self) -> tuple[Dataset, Dataset, Dataset]:
        return (
            self._build_split("train", self.n_train_limit),
            self._build_split("val", self.n_val_limit),
            self._build_split("test", self.n_test_limit),
        )

    def _build_train_dataset(self) -> Dataset:
        return self._build_split("train", self.n_train_limit)

    def _build_val_dataset(self) -> Dataset:
        return self._build_split("val", self.n_val_limit)

    def _build_test_dataset(self) -> Dataset:
        return self._build_split("test", self.n_test_limit)

    def setup(self, stage: str | None = None) -> None:
        super().setup(stage)
        self.n_val = len(self._val) if self._val is not None else 0
        self.n_test = len(self._test) if self._test is not None else 0
        if self.num_workers > 0:
            log_info(f"[graphs] MNIST mmap cache: num_workers={self.num_workers} with spawn")

    def _effective_size(self, split: str, limit: int | None) -> int:
        total = mnist_split_size(self.data_root, split)
        return min(total, limit) if limit is not None else total

    def probe_split_sizes(self) -> dict[str, int]:
        return {
            "n_train": self._effective_size("train", self.n_train_limit),
            "n_val": self._effective_size("val", self.n_val_limit),
            "n_test": self._effective_size("test", self.n_test_limit),
        }

    def get_dims(self) -> tuple[int, int]:
        return self.N_FEATURES, self.OUT_DIM

    def get_edge_dim(self) -> int:
        return self.N_EDGE_FEATURES

    def planned_graph_mode(self) -> tuple[str, None]:
        return "cache", None

    def run_info(self, split_sizes: dict[str, int] | None = None) -> dict:
        sizes = split_sizes if split_sizes is not None else self.probe_split_sizes()
        return {
            "dataset": "mnist",
            "benchmark": "MNISTSuperpixels",
            "task": self.task,
            "graph_construction": self.graph_construction,
            "graph_cache": True,
            "graph_loading": "cache",
            "edge_mode": "native-edges",
            "uses_edge_features": self.include_edge_features,
            "n_edge_features": self.N_EDGE_FEATURES,
            "max_nodes": self.MAX_NODES,
            "num_workers": self.num_workers,
            **sizes,
        }

    def run_metadata(self, split_sizes: dict[str, int] | None = None) -> dict:
        return self.run_info(split_sizes)

    def metadata(self) -> dict:
        return self.run_info()
