"""TopTagging with pre-built hierarchical (kNN+DBSCAN+virtual) graph caches."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

from cpen.datamodules.base_datamodule import BaseDatamodule
from cpen.utils.graph_hierarchical import hierarchical_construction_tag
from cpen.utils.hier_graph_cache import (
    CachedHierJetDataset,
    missing_hier_graph_cache_message,
    read_hier_graph_cache_split_counts,
)
from cpen.utils.log_utils import log_info
from cpen.utils.part_kin import N_PART_KIN_FEATURES
from cpen.utils.preprocessing import get_dataset_stats
from cpen.utils.toptagging import DEFAULT_NUM_PARTICLES


class TopTaggingHierDatamodule(BaseDatamodule):
    """TopTagging benchmark on memory-mapped hierarchical hypergraph caches."""

    DEFAULT_N_PARTICLES = DEFAULT_NUM_PARTICLES
    N_FEATURES = N_PART_KIN_FEATURES
    N_EDGE_FEATURES = 4
    OUT_DIM = 2

    def __init__(
        self,
        data_root: str,
        batch_size: int = 128,
        *,
        k: int = 8,
        eps: float = 0.08,
        min_samples: int = 2,
        n_virtual_nodes: int = 1,
        n_virtual_edges: int = 1,
        max_dbscan_edges: int = 32,
        num_workers: int | None = None,
        num_particles: int = DEFAULT_N_PARTICLES,
        n_train: int | None = None,
        n_val: int | None = None,
        n_test: int | None = None,
        data_seed: int = 42,
    ) -> None:
        super().__init__(data_root, batch_size, num_workers=num_workers)
        self.k = int(k)
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.n_virtual_nodes = int(n_virtual_nodes)
        self.n_virtual_edges = int(n_virtual_edges)
        self.max_dbscan_edges = int(max_dbscan_edges)
        self.num_particles = int(num_particles)
        self.n_train_limit = n_train
        self.n_val_limit = n_val
        self.n_test_limit = n_test
        self.data_seed = int(data_seed)
        self._uses_graph_cache = False
        self.n_val = 0
        self.n_test = 0

    def dataset_name(self) -> str:
        return "toptagging"

    @property
    def graph_construction(self) -> str:
        return hierarchical_construction_tag(
            k=self.k,
            eps=self.eps,
            min_samples=self.min_samples,
            n_virtual_nodes=self.n_virtual_nodes,
            n_virtual_edges=self.n_virtual_edges,
            max_dbscan_edges=self.max_dbscan_edges,
        )

    def _build_split(
        self,
        split: str,
        max_jets: int | None,
        *,
        seed: int,
    ) -> Dataset:
        log_info(
            f"[graphs] split={split} mode=hier-cache "
            f"{self.graph_construction} n_particles={self.num_particles}"
        )
        try:
            return CachedHierJetDataset(
                data_root=self.data_root,
                split=split,
                k=self.k,
                eps=self.eps,
                min_samples=self.min_samples,
                n_virtual_nodes=self.n_virtual_nodes,
                n_virtual_edges=self.n_virtual_edges,
                max_dbscan_edges=self.max_dbscan_edges,
                num_particles=self.num_particles,
                max_jets=max_jets,
                seed=seed,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                missing_hier_graph_cache_message(
                    data_root=self.data_root,
                    split=split,
                    k=self.k,
                    eps=self.eps,
                    min_samples=self.min_samples,
                    n_virtual_nodes=self.n_virtual_nodes,
                    n_virtual_edges=self.n_virtual_edges,
                    max_dbscan_edges=self.max_dbscan_edges,
                    num_particles=self.num_particles,
                )
            ) from exc

    def build_datasets(self) -> tuple[Dataset, Dataset, Dataset]:
        self._ensure_dataset_stats()
        train = self._build_split("train", self.n_train_limit, seed=self.data_seed)
        val = self._build_split("val", self.n_val_limit, seed=self.data_seed)
        test = self._build_split("test", self.n_test_limit, seed=self.data_seed)
        return train, val, test

    def _ensure_dataset_stats(self) -> None:
        if getattr(self, "_stats_loaded", False):
            return
        stats = get_dataset_stats("toptagging")
        self.corr_adam = stats.corr_adam
        self.corr_sgd = stats.corr_sgd
        Path(self.data_root).mkdir(parents=True, exist_ok=True)
        self._stats_loaded = True

    def _build_train_dataset(self) -> Dataset:
        self._ensure_dataset_stats()
        return self._build_split("train", self.n_train_limit, seed=self.data_seed)

    def _build_val_dataset(self) -> Dataset:
        self._ensure_dataset_stats()
        return self._build_split("val", self.n_val_limit, seed=self.data_seed)

    def _build_test_dataset(self) -> Dataset:
        self._ensure_dataset_stats()
        return self._build_split("test", self.n_test_limit, seed=self.data_seed)

    def probe_split_sizes(self) -> dict[str, int]:
        if self._train is not None:
            sizes = {"n_train": len(self._train)}
            if self._val is not None:
                sizes["n_val"] = len(self._val)
            if self._test is not None:
                sizes["n_test"] = len(self._test)
            return sizes

        stored = read_hier_graph_cache_split_counts(
            self.data_root,
            k=self.k,
            eps=self.eps,
            min_samples=self.min_samples,
            n_virtual_nodes=self.n_virtual_nodes,
            n_virtual_edges=self.n_virtual_edges,
            max_dbscan_edges=self.max_dbscan_edges,
            num_particles=self.num_particles,
        )
        if stored is not None:

            def _effective(split: str, max_jets: int | None) -> int:
                n_total = stored[split]
                if max_jets is not None and max_jets < n_total:
                    return max_jets
                return n_total

            return {
                "n_train": _effective("train", self.n_train_limit),
                "n_val": _effective("val", self.n_val_limit),
                "n_test": _effective("test", self.n_test_limit),
            }

        self.setup()
        return self.split_sizes()

    def setup(self, stage: str | None = None) -> None:
        super().setup(stage)
        self.n_val = len(self._val) if self._val is not None else 0
        self.n_test = len(self._test) if self._test is not None else 0
        self._uses_graph_cache = isinstance(self._train, CachedHierJetDataset)
        if self._uses_graph_cache and self.num_workers > 0:
            log_info(
                f"[graphs] hier-cache mmap: num_workers={self.num_workers} with spawn"
            )

    def planned_graph_mode(self) -> tuple[str, None]:
        return "cache", None

    def run_metadata(self, split_sizes: dict[str, int]) -> dict:
        return {
            "graph_construction": self.graph_construction,
            "hier_k": self.k,
            "hier_eps": self.eps,
            "hier_min_samples": self.min_samples,
            "hier_vn": self.n_virtual_nodes,
            "hier_ve": self.n_virtual_edges,
            "hier_mdb": self.max_dbscan_edges,
            "graph_cache": True,
            "graph_loading": "cache",
            "edge_mode": "hierarchical",
            "num_workers": self.num_workers,
            "num_particles": self.num_particles,
            "n_edge_features": self.N_EDGE_FEATURES,
            "cache_tag": self.graph_construction,
            **split_sizes,
        }

    def get_dims(self) -> tuple[int, int]:
        return self.N_FEATURES, self.OUT_DIM

    def get_edge_dim(self) -> int:
        return self.N_EDGE_FEATURES

    def split_sizes(self) -> dict[str, int]:
        return {
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
        }

    def metadata(self) -> dict:
        meta = super().metadata()
        meta["graph_construction"] = self.graph_construction
        meta["graph_cache"] = self._uses_graph_cache
        meta["edge_mode"] = "hierarchical"
        meta["graph_loading"] = "cache"
        meta["n_edge_features"] = self.N_EDGE_FEATURES
        return meta
