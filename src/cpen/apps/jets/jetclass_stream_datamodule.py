"""JetClass datamodule reading raw ROOT, with graphs built live on the GPU."""

from __future__ import annotations

from typing import Any

import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset

from cpen.apps.jets.jetclass_stream import (
    DEFAULT_FEATURE_CONFIG,
    JetClassStreamDataset,
    JetClassSubsetDataset,
    chunk_size_for_workers,
    feature_names,
    n_features,
    plan_shards,
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
        shuffle_buffer: int | None = None,
        sort_by_pt: bool = False,
        train_skip_jets: int = 0,
        expected_world_size: int = 1,
    ) -> None:
        super().__init__(data_root, batch_size, num_workers=num_workers)
        self.num_particles = int(num_particles)
        self.feature_config = feature_config
        self.n_train_limit = n_train
        self.n_val_limit = n_val if n_val is not None else DEFAULT_N_VAL
        self.n_test_limit = n_test if n_test is not None else DEFAULT_N_TEST
        self.shuffle_seed = int(shuffle_seed)
        # Streaming jobs walk ROOT in class-interleaved file order so a
        # preempted run can skip the consumed prefix and continue. RAM subsets
        # still shuffle via the map-style DataLoader.
        self.shuffle_buffer = 0 if shuffle_buffer is None else int(shuffle_buffer)
        self.sort_by_pt = bool(sort_by_pt)
        self.train_skip_jets = max(0, int(train_skip_jets))
        # DDP world size used to size stream chunks *before* the Lightning
        # trainer is attached (γ estimate / early setup). Under-sharding relative
        # to world×workers makes every replica empty and fit ends at step 0.
        self.expected_world_size = max(1, int(expected_world_size))
        self._stream_replicas_planned: int | None = None
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
        """Whether the training split is streamed rather than resident.

        Force streaming whenever ``train_skip_jets > 0``: the RAM subset path
        ignores skip and would make every "shard" identical (panel-b collapse).
        """
        if self.train_skip_jets > 0:
            return True
        return self.n_train_limit is None or self.n_train_limit > RAM_SUBSET_LIMIT

    def get_dims(self) -> tuple[int, int]:
        return n_features(self.feature_config), self.OUT_DIM

    def get_edge_dim(self) -> int:
        return self.N_EDGE_FEATURES

    def _ensure_dataset_stats(self) -> None:
        if getattr(self, "_stats_loaded", False):
            return
        # DatasetStats.corr_* are legacy defaults (always 1.0 for JetClass).
        # CLI ``--corr`` is written onto the DM in create_jetclass_datamodule
        # *before* bytecode train_one_run calls setup(); do not clobber it here
        # or LitCPEN is built with corr=1 and embedding LR correction is a no-op.
        get_dataset_stats("jetclass")
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

    def _ddp_identity(self) -> tuple[int, int]:
        """(global_rank, world_size) from the Lightning trainer, else single-process."""
        # ``self.trainer`` raises if the datamodule is not attached. Prefer the
        # slot Lightning sets on attach; tests may also set a plain ``trainer``.
        trainer = getattr(self, "_trainer", None) or vars(self).get("trainer")
        if trainer is None:
            return 0, 1
        world = int(getattr(trainer, "world_size", 1) or 1)
        rank = int(getattr(trainer, "global_rank", 0) or 0)
        return rank, max(1, world)

    def _stream_replica_count(self) -> int:
        """DataLoader replicas the stream plan must cover (workers × DDP world)."""
        _, world = self._ddp_identity()
        world = max(world, self.expected_world_size)
        return max(1, int(self.num_workers or 0)) * max(1, world)

    def _apply_ddp_to_stream(self, dataset: JetClassStreamDataset) -> None:
        """Stamp rank/world on the dataset before DataLoader workers pickle it."""
        rank, world = self._ddp_identity()
        dataset.ddp_rank = rank
        dataset.ddp_world_size = world

    def _build_train_dataset(self) -> Dataset:
        self._ensure_dataset_stats()
        if self.streaming:
            rank, world = self._ddp_identity()
            # Prefer the live trainer world once attached; otherwise the CLI
            # ``--n-gpus`` hint so γ-estimate setup does not under-shard.
            world_for_plan = max(world, self.expected_world_size)
            n_replicas = self._stream_replica_count()
            dataset = JetClassStreamDataset(
                self.data_root,
                "train",
                n_jets=self.n_train_limit,
                num_particles=self.num_particles,
                feature_config=self.feature_config,
                chunk_size=chunk_size_for_workers(self.n_train_limit, n_replicas),
                shuffle_seed=self.shuffle_seed,
                shuffle_buffer=self.shuffle_buffer,
                sort_by_pt=self.sort_by_pt,
                skip_jets=self.train_skip_jets,
                ddp_rank=rank,
                ddp_world_size=world_for_plan,
            )
            self._stream_replicas_planned = n_replicas
            log_info(
                f"[graphs] split=train mode=stream (single pass) "
                f"n_jets={dataset.n_jets:,} skip={self.train_skip_jets:,} "
                f"features={self.feature_config} "
                f"n_particles={self.num_particles} shuffle_buffer={self.shuffle_buffer} "
                f"replicas={n_replicas} (workers={self.num_workers} world={world_for_plan})"
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
        n = int(self.n_val_limit) if self.n_val_limit is not None else 0
        # Train-only / checkpoint-later runs pass --n-val 0 (no val ROOT needed).
        if n <= 0:
            from torch.utils.data import TensorDataset
            import torch

            return TensorDataset(torch.empty(0, dtype=torch.long))
        return self._subset("val", n)

    def _build_test_dataset(self) -> Dataset:
        self._ensure_dataset_stats()
        n = int(self.n_test_limit) if self.n_test_limit is not None else 0
        # Fast sweeps pass --n-test 0 to skip post-fit test; empty subset
        # previously crashed np.concatenate. Return a 0-length subset only if
        # the builder can handle it; otherwise build a tiny 1-jet placeholder
        # is wrong — prefer empty-capable path via n_jets=max(n,0) guard in
        # JetClassSubsetDataset. When n<=0, avoid building altogether by using
        # a 0-jet stream subset that short-circuits.
        if n <= 0:
            from torch.utils.data import TensorDataset
            import torch

            # Lightning still asks for a test dataloader; a zero-length
            # TensorDataset makes test finish immediately with no batches.
            return TensorDataset(torch.empty(0, dtype=torch.long))
        return self._subset("test", n)

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

    def _resume_skip_jets(self) -> int:
        """Jets already consumed on this rank, from a restored Lightning trainer.

        Uses ``global_step`` (optimizer steps across the whole run), not the
        in-epoch batch counter. Within one data-epoch a preempted job must
        continue through the ROOT files (not rewind). Across epochs the skip
        wraps in *jet* space: ``train_steps_per_epoch`` is slightly above
        ``n/(B·world)`` because of per-worker partial batches, so a step-modulo
        at the epoch boundary still lands past EOF and Lightning spins empty
        epochs at a frozen global_step.
        """
        trainer = getattr(self, "_trainer", None) or vars(self).get("trainer")
        if trainer is None:
            return 0
        step = int(getattr(trainer, "global_step", 0) or 0)
        raw = max(0, step) * int(self.batch_size)
        if isinstance(self._train, JetClassStreamDataset):
            n = int(getattr(self._train, "n_jets", 0) or 0)
            _, world = self._ddp_identity()
            world = max(1, int(world), int(getattr(self, "expected_world_size", 1) or 1))
            period = max(1, n // world) if n > 0 else 0
            if period > 0:
                raw = raw % period
        return raw

    def train_dataloader(self) -> DataLoader:
        assert self._train is not None
        if isinstance(self._train, JetClassStreamDataset):
            needed = self._stream_replica_count()
            if self._stream_replicas_planned != needed:
                # Trainer world now known (or changed): rebuild so chunk sizing
                # matches world×workers. Otherwise replicas can be empty and
                # Lightning ends the epoch at global_step=0 with exit 0.
                self._train = self._build_train_dataset()
                assert isinstance(self._train, JetClassStreamDataset)
            self._apply_ddp_to_stream(self._train)
            skip = self._resume_skip_jets()
            self._train.skip_jets = skip
            if skip:
                rank, world = self._ddp_identity()
                log_info(
                    f"[graphs] stream resume skip_jets={skip:,} "
                    f"(batch_size={self.batch_size} rank={rank}/{world})"
                )
            # Fail fast: empty replica plans used to look like a successful 0-step fit.
            rank, world = self._ddp_identity()
            probe = sum(s.n_jets for s in self._train._worker_shards())
            if probe <= 0:
                raise RuntimeError(
                    f"[graphs] stream plan empty on rank {rank}/{world} "
                    f"(workers={self.num_workers}, skip={self.train_skip_jets}, "
                    f"replicas_planned={self._stream_replicas_planned}). "
                    "Increase expected_world_size / shrink chunk size so every "
                    "replica receives jets."
                )
            # Shuffling happens inside the worker's buffer; DataLoader must not
            # try to shuffle an IterableDataset.
            return DataLoader(self._train, shuffle=False, **self._stream_dataloader_kwargs())
        return DataLoader(self._train, shuffle=True, **self._dataloader_kwargs(self._train))

    def train_steps_per_epoch(self) -> int:
        """Batches per training epoch on each DDP rank."""
        assert self._train is not None
        n = int(getattr(self._train, "n_jets", len(self._train)))
        if isinstance(self._train, JetClassStreamDataset):
            _, world = self._ddp_identity()
            return steps_per_epoch(n, self.batch_size, self.num_workers, world)
        return -(-n // self.batch_size)

    def _planned_size(self, split: str, limit: int | None) -> int:
        """
        Split size from the read plan, without opening any ROOT file.

        The sweep driver asks for these before ``setup``, so this has to answer
        from the file listing alone.
        """
        return sum(
            s.n_jets for s in plan_shards(self.data_root, split, n_jets=limit)
        )

    def probe_split_sizes(self) -> dict[str, int]:
        sizes: dict[str, int] = {}
        for key, split, limit, dataset in (
            ("n_train", "train", self.n_train_limit, self._train),
            ("n_val", "val", self.n_val_limit, self._val),
            ("n_test", "test", self.n_test_limit, self._test),
        ):
            if dataset is not None:
                sizes[key] = int(getattr(dataset, "n_jets", len(dataset)))
            elif limit is not None and int(limit) <= 0:
                # Train-only / no-test runs must not require val_5M / test_20M
                # ROOT trees on disk (probe runs before setup builds empties).
                sizes[key] = 0
            else:
                sizes[key] = self._planned_size(split, limit)
        return sizes

    def split_sizes(self) -> dict[str, int]:
        return self.probe_split_sizes()

    def _stream_metadata(self) -> dict[str, Any]:
        """Keys shared by the parquet run record and the banner."""
        return {
            "jetclass_source": "raw-root-stream",
            "jetclass_features": self.feature_config,
            "jetclass_n_features": n_features(self.feature_config),
            "jetclass_num_particles": self.num_particles,
            "jetclass_streaming": self.streaming,
            "jetclass_particle_normalization": "part-full-affine",
            "graph_construction": self.graph_construction,
            "graph_cache": False,
            "graph_loading": "live",
            "edge_mode": "star-radius",
            "n_edge_features": self.get_edge_dim(),
        }

    def run_metadata(self, split_sizes: dict[str, int]) -> dict[str, Any]:
        return {
            **self._stream_metadata(),
            "num_workers": self.num_workers,
            "num_particles": self.num_particles,
            **split_sizes,
        }

    def metadata(self) -> dict[str, Any]:
        meta = super().metadata()
        meta.update(self._stream_metadata())
        # ``super().setup`` cannot len() an IterableDataset, and the banner plus
        # the muP LR scaling both read this, so answer from the read plan.
        meta["n_train"] = self.probe_split_sizes()["n_train"]
        return meta
