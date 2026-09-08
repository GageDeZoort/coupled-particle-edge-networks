"""JetClass datamodule reading raw ROOT, with graphs built live on the GPU."""

from __future__ import annotations

from typing import Any

import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset

from cpen.apps.jets.jetclass_stream import (
    DEFAULT_FEATURE_CONFIG,
    JetClassStreamDataset,
    JetClassSubsetDataset,
    feature_names,
    n_features,
    steps_per_epoch,
)
from cpen.training.base_datamodule import BaseDatamodule
from cpen.utils.jetclass import N_CLASSES
from cpen.utils.log_utils import log_info
from cpen.utils.preprocessing import get_dataset_stats

#: Above this many training jets the split is streamed rather than held in RAM.
#: At ~11 kB per jet the crossover sits near 22 GB resident, comfortably inside
#: a 256 GB allocation while keeping the repetition regime ($D \\le 10^6$) in
#: memory where re-decoding ROOT every epoch would dominate.
RAM_SUBSET_LIMIT = 2_000_000

DEFAULT_N_VAL = 250_000
DEFAULT_N_TEST = 1_000_000


class JetClassStreamDatamodule(BaseDatamodule):
    """
    JetClass straight from the published ROOT files.

    Emits particle four-vectors as ``x_raw`` and no edges, so the Lightning
    module builds the star-$R$ hypergraph on the GPU once per batch. This keeps
    the dataset-size axis free: nothing is materialized, so a change of radius,
    edge featurization, feature configuration or particle count costs no disk.

    ``n_train`` is the $D$ of a scaling-law point. Budgets are split evenly over
    the ten classes and consume each class's files in order, so a given
    ``n_train`` always selects the same jets and smaller budgets nest inside
    larger ones.
    """

    N_EDGE_FEATURES = 4
    OUT_DIM = N_CLASSES

    def __init__(
        self,
        data_root: str,
        batch_size: int = 128,
        *,
        num_workers: int | None = None,
        num_particles: int = 128,
        feature_config: str = DEFAULT_FEATURE_CONFIG,
        n_train: int | None = None,
        n_val: int | None = None,
        n_test: int | None = None,
        shuffle_seed: int = 0,
        sort_by_pt: bool = False,
    ) -> None:
        super().__init__(data_root, batch_size, num_workers=num_workers)
        self.num_particles = int(num_particles)
        self.feature_config = feature_config
        self.n_train_limit = n_train
        self.n_val_limit = n_val if n_val is not None else DEFAULT_N_VAL
        self.n_test_limit = n_test if n_test is not None else DEFAULT_N_TEST
        self.shuffle_seed = int(shuffle_seed)
        self.sort_by_pt = bool(sort_by_pt)
        self.n_particles = self.num_particles
        # Validate eagerly so a typo fails before any ROOT file is opened.
        feature_names(self.feature_config)

    def dataset_name(self) -> str:
        return "jetclass"

    @property
    def graph_construction(self) -> str:
        return "live"

    @property
    def streaming(self) -> bool:
        """Whether the training split is streamed rather than resident."""
        return self.n_train_limit is None or self.n_train_limit > RAM_SUBSET_LIMIT

    def get_dims(self) -> tuple[int, int]:
        return n_features(self.feature_config), self.OUT_DIM

    def get_edge_dim(self) -> int:
        return self.N_EDGE_FEATURES

    def _ensure_dataset_stats(self) -> None:
        if getattr(self, "_stats_loaded", False):
            return
        stats = get_dataset_stats("jetclass")
        self.corr_adam = stats.corr_adam
        self.corr_sgd = stats.corr_sgd
        self._stats_loaded = True

    def _subset(self, split: str, n_jets: int) -> Dataset:
        return JetClassSubsetDataset(
            self.data_root,
            split,
            n_jets=n_jets,
            num_particles=self.num_particles,
            feature_config=self.feature_config,
            sort_by_pt=self.sort_by_pt,
        )

    def _build_train_dataset(self) -> Dataset:
        self._ensure_dataset_stats()
        if self.streaming:
            dataset = JetClassStreamDataset(
                self.data_root,
                "train",
                n_jets=self.n_train_limit,
                num_particles=self.num_particles,
                feature_config=self.feature_config,
                shuffle_seed=self.shuffle_seed,
                sort_by_pt=self.sort_by_pt,
            )
            log_info(
                f"[graphs] split=train mode=stream (single pass) "
                f"n_jets={dataset.n_jets:,} features={self.feature_config} "
                f"n_particles={self.num_particles}"
            )
            return dataset
        dataset = self._subset("train", int(self.n_train_limit))
        log_info(
            f"[graphs] split=train mode=ram-subset (repeatable) "
            f"n_jets={len(dataset):,} features={self.feature_config} "
            f"n_particles={self.num_particles}"
        )
        return dataset

    def _build_val_dataset(self) -> Dataset:
        self._ensure_dataset_stats()
        return self._subset("val", int(self.n_val_limit))

    def _build_test_dataset(self) -> Dataset:
        self._ensure_dataset_stats()
        return self._subset("test", int(self.n_test_limit))

    def build_datasets(self) -> tuple[Dataset, Dataset, Dataset]:
        return (
            self._build_train_dataset(),
            self._build_val_dataset(),
            self._build_test_dataset(),
        )

    def setup(self, stage: str | None = None) -> None:
        super().setup(stage)
        if self._train is not None:
            # IterableDataset has no len() contract for Lightning; n_train drives
            # the muP learning-rate scaling, so take it from the read plan.
            self.n_train = int(getattr(self._train, "n_jets", len(self._train)))

    def _stream_dataloader_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": True,
        }
        if self.num_workers > 0:
            kwargs["multiprocessing_context"] = mp.get_context("spawn")
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 2
        return kwargs

    def train_dataloader(self) -> DataLoader:
        assert self._train is not None
        if isinstance(self._train, JetClassStreamDataset):
            # Shuffling happens inside the worker's buffer; DataLoader must not
            # try to shuffle an IterableDataset.
            return DataLoader(self._train, shuffle=False, **self._stream_dataloader_kwargs())
        return DataLoader(self._train, shuffle=True, **self._dataloader_kwargs(self._train))

    def train_steps_per_epoch(self) -> int:
        """Batches per training epoch, for schedulers and progress bars."""
        assert self._train is not None
        n = int(getattr(self._train, "n_jets", len(self._train)))
        if isinstance(self._train, JetClassStreamDataset):
            return steps_per_epoch(n, self.batch_size, self.num_workers)
        return -(-n // self.batch_size)

    def probe_split_sizes(self) -> dict[str, int]:
        sizes: dict[str, int] = {}
        for key, dataset in (
            ("n_train", self._train),
            ("n_val", self._val),
            ("n_test", self._test),
        ):
            if dataset is not None:
                sizes[key] = int(getattr(dataset, "n_jets", len(dataset)))
        return sizes

    def run_info(self) -> dict[str, Any]:
        return {
            "jetclass_source": "raw-root-stream",
            "jetclass_features": self.feature_config,
            "jetclass_n_features": n_features(self.feature_config),
            "jetclass_num_particles": self.num_particles,
            "jetclass_streaming": self.streaming,
            "jetclass_particle_normalization": "part-full-affine",
        }
