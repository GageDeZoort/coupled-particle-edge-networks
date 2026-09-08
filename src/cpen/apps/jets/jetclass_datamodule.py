"""JetClassLite datamodule with ParT-full features and star-$R$ caches."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

from cpen.datamodules.base_datamodule import BaseDatamodule
from cpen.utils.jetclass import DEFAULT_NUM_PARTICLES, N_CLASSES, N_PART_FEATURES
from cpen.utils.jetclass_star_cache import (
    CachedJetClassStarDataset,
    missing_jetclass_star_cache_message,
    jetclass_star_cache_available,
    read_jetclass_star_split_counts,
)
from cpen.utils.log_utils import log_info
from cpen.utils.preprocessing import get_dataset_stats


class JetClassLiteStarDatamodule(BaseDatamodule):
    """
    JetClassLite multi-class benchmark with memory-mapped star-$R$ caches.

    Particle inputs follow the Particle Transformer ``full`` feature set (17-D).
    Graphs must be pre-built with ``build_jetclass_lite_star_graphs.py``.
    """

    DEFAULT_N_PARTICLES = DEFAULT_NUM_PARTICLES
    N_FEATURES = N_PART_FEATURES
    N_EDGE_FEATURES = 4
    OUT_DIM = N_CLASSES

    def __init__(
        self,
        data_root: str,
        batch_size: int = 128,
        *,
        radius: float,
        num_workers: int | None = None,
        num_particles: int = DEFAULT_N_PARTICLES,
        n_train: int | None = None,
        n_val: int | None = None,
        n_test: int | None = None,
        data_seed: int = 42,
    ) -> None:
        super().__init__(data_root, batch_size, num_workers=num_workers)
        self.radius = float(radius)
        self.num_particles = num_particles
        self.n_train_limit = n_train
        self.n_val_limit = n_val
        self.n_test_limit = n_test
        self.data_seed = data_seed
        self.n_val: int = 0
        self.n_test: int = 0

    def dataset_name(self) -> str:
        return "jetclass"

    @property
    def graph_construction(self) -> str:
        return f"star-R={self.radius:g}"

    def _build_split(self, split: str, max_jets: int | None) -> Dataset:
        if not jetclass_star_cache_available(
            cache_root=self.data_root,
            split=split,
            radius=self.radius,
            num_particles=self.num_particles,
            max_jets=None,
            seed=self.data_seed,
        ):
            raise FileNotFoundError(
                missing_jetclass_star_cache_message(
                    cache_root=self.data_root,
                    split=split,
                    radius=self.radius,
                    num_particles=self.num_particles,
                )
            )
        log_info(
            f"[graphs] split={split} mode=star-cache R={self.radius:g} "
            f"n_particles={self.num_particles}"
        )
        return CachedJetClassStarDataset(
            cache_root=self.data_root,
            split=split,
            radius=self.radius,
            num_particles=self.num_particles,
            max_jets=max_jets,
            seed=self.data_seed,
        )

    def build_datasets(self) -> tuple[Dataset, Dataset, Dataset]:
        self._ensure_dataset_stats()
        train = self._build_split("train", self.n_train_limit)
        val = self._build_split("val", self.n_val_limit)
        test = self._build_split("test", self.n_test_limit)
        return train, val, test

    def _ensure_dataset_stats(self) -> None:
        if getattr(self, "_stats_loaded", False):
            return
        stats = get_dataset_stats("jetclass")
        self.corr_adam = stats.corr_adam
        self.corr_sgd = stats.corr_sgd
        Path(self.data_root).mkdir(parents=True, exist_ok=True)
        self._stats_loaded = True

    def _build_train_dataset(self) -> Dataset:
        self._ensure_dataset_stats()
        return self._build_split("train", self.n_train_limit)

    def _build_val_dataset(self) -> Dataset:
        self._ensure_dataset_stats()
        return self._build_split("val", self.n_val_limit)

    def _build_test_dataset(self) -> Dataset:
        self._ensure_dataset_stats()
        return self._build_split("test", self.n_test_limit)

    def probe_split_sizes(self) -> dict[str, int]:
        if self._train is not None:
            sizes = {"n_train": len(self._train)}
            if self._val is not None:
                sizes["n_val"] = len(self._val)
            if self._test is not None:
                sizes["n_test"] = len(self._test)
            return sizes

        stored = read_jetclass_star_split_counts(
            self.data_root,
            radius=self.radius,
            num_particles=self.num_particles,
        )
        if stored is None:
            return {}

        def _cap(n_full: int, limit: int | None) -> int:
            return n_full if limit is None else min(n_full, limit)

        return {
            "n_train": _cap(int(stored.get("train", 0)), self.n_train_limit),
            "n_val": _cap(int(stored.get("val", 0)), self.n_val_limit),
            "n_test": _cap(int(stored.get("test", 0)), self.n_test_limit),
        }

    def get_dims(self) -> tuple[int, int]:
        return self.N_FEATURES, self.OUT_DIM

    def get_edge_dim(self) -> int:
        return self.N_EDGE_FEATURES
