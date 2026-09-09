"""TopTagging with pre-built star-$R$ PyG-compatible hypergraph caches."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

from cpen.datamodules.base_datamodule import BaseDatamodule
from cpen.utils.log_utils import log_info
from cpen.utils.part_kin import N_PART_KIN_FEATURES
from cpen.utils.preprocessing import get_dataset_stats
from cpen.utils.star_graph_cache import (
    CachedStarGraphJetDataset,
    missing_star_graph_cache_message,
    read_star_graph_cache_split_counts,
    star_graph_cache_available,
    star_radius_tag,
)
from cpen.utils.toptagging import DEFAULT_NUM_PARTICLES


class TopTaggingStarDatamodule(BaseDatamodule):
    """
    TopTagging benchmark with memory-mapped star-$R$ hypergraph caches.

    Graphs must be pre-built with ``build_toptagging_star_graphs.py``. Each jet
    is stored with dense incidence ``(N, N)`` and hyperedge features ``(N, 4)``
    (row ``c`` = star centered on particle ``c``), compatible with CPEN and
    convertible to PyG via :meth:`CachedStarGraphJetDataset.get_pyg_data`.
    """

    DEFAULT_N_PARTICLES = DEFAULT_NUM_PARTICLES
    # ParT TopLandscape / JetClass_kin (7-D, affine only — no row-L2).
    N_FEATURES = N_PART_KIN_FEATURES
    N_EDGE_FEATURES = 4
    OUT_DIM = 2

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
        rebuild_graph_cache: bool = False,
    ) -> None:
        super().__init__(data_root, batch_size, num_workers=num_workers)
        self.radius = float(radius)
        self.num_particles = num_particles
        self.n_train_limit = n_train
        self.n_val_limit = n_val
        self.n_test_limit = n_test
        self.data_seed = data_seed
        self.rebuild_graph_cache = rebuild_graph_cache
        self._uses_graph_cache = False

        self.n_val: int = 0
        self.n_test: int = 0

    def dataset_name(self) -> str:
        return "toptagging"

    @property
    def graph_construction(self) -> str:
        return f"star-R={self.radius:g}"

    def _build_split(
        self,
        split: str,
        max_jets: int | None,
        *,
        seed: int,
    ) -> Dataset:
        if not star_graph_cache_available(
            data_root=self.data_root,
            split=split,
            radius=self.radius,
            num_particles=self.num_particles,
            max_jets=max_jets,
            seed=seed,
        ) and not self.rebuild_graph_cache:
            raise FileNotFoundError(
                missing_star_graph_cache_message(
                    data_root=self.data_root,
                    split=split,
                    radius=self.radius,
                    num_particles=self.num_particles,
                )
            )

        log_info(
            f"[graphs] split={split} mode=star-cache R={self.radius:g} "
            f"n_particles={self.num_particles}"
        )
        return CachedStarGraphJetDataset(
            data_root=self.data_root,
            split=split,
            radius=self.radius,
            num_particles=self.num_particles,
            max_jets=max_jets,
            seed=seed,
            rebuild_cache=self.rebuild_graph_cache,
        )

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

        stored = read_star_graph_cache_split_counts(
            self.data_root,
            radius=self.radius,
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
        self._uses_graph_cache = isinstance(self._train, CachedStarGraphJetDataset)
        if self._uses_graph_cache and self.num_workers > 0:
            log_info(
                f"[graphs] star-cache mmap: num_workers={self.num_workers} with spawn"
            )

    def planned_graph_mode(self) -> tuple[str, None]:
        return "cache", None

    def run_metadata(self, split_sizes: dict[str, int]) -> dict:
        return {
            "graph_construction": self.graph_construction,
            "star_radius": self.radius,
            "graph_cache": True,
            "graph_loading": "cache",
            "edge_mode": "star-radius",
            "num_workers": self.num_workers,
            "num_particles": self.num_particles,
            "n_edge_features": self.N_EDGE_FEATURES,
            "cache_tag": star_radius_tag(self.radius),
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
        meta["star_radius"] = self.radius
        meta["graph_cache"] = self._uses_graph_cache
        meta["edge_mode"] = "star-radius"
        meta["graph_loading"] = "cache"
        meta["n_edge_features"] = self.N_EDGE_FEATURES
        return meta
