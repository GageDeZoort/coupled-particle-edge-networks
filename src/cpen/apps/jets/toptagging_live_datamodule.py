"""TopTagging datamodule with live GPU star-$R$ (+ optional kNN) graphs.

Mirrors ``JetClassStreamDatamodule``: HDF5 jets ship ``x`` / ``x_raw`` / ``mask``
and no edges; ``BaseLitCPEN.on_after_batch_transfer`` builds the same live
star+knn graph used at JetClass pretrain time.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from cpen.apps.jets.part_kin import N_PART_KIN_FEATURES
from cpen.apps.jets.toptagging import (
    DEFAULT_NUM_PARTICLES,
    TopTaggingJetDataset,
    normalize_data_root,
    preprocess_particle_features,
    raw_hdf5_path,
)
from cpen.training.base_datamodule import BaseDatamodule
from cpen.utils.log_utils import log_info
from cpen.utils.preprocessing import get_dataset_stats


class TopTaggingLiveJetDataset(Dataset):
    """Map-style TopTagging split returning dict batches for live graphs."""

    def __init__(
        self,
        *,
        data_root: str,
        split: str,
        num_particles: int = DEFAULT_NUM_PARTICLES,
        max_jets: int | None = None,
        seed: int = 42,
    ) -> None:
        # Reuse index / HDF5 loading from the scratch TopTagging dataset.
        self._base = TopTaggingJetDataset(
            data_root=data_root,
            split=split,
            num_particles=num_particles,
            max_jets=max_jets,
            seed=seed,
            download=False,
        )
        self.num_particles = int(num_particles)

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Reach past TopTaggingJetDataset's (x_model, y) to raw four-vectors.
        base_idx = int(self._base._indices[idx])
        particles, jet = self._base._base[base_idx]
        x_raw, x_model, mask = preprocess_particle_features(particles.float())
        label = int(jet[0].item())
        return {
            "x": x_model,
            "x_raw": x_raw,
            "mask": mask.to(dtype=torch.bool),
            "y": torch.tensor(label, dtype=torch.long),
        }

    @staticmethod
    def collate_samples(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            "x": torch.stack([b["x"] for b in batch], dim=0),
            "x_raw": torch.stack([b["x_raw"] for b in batch], dim=0),
            "mask": torch.stack([b["mask"] for b in batch], dim=0),
            "y": torch.stack([b["y"] for b in batch], dim=0),
        }


class TopTaggingLiveDatamodule(BaseDatamodule):
    """
    TopLandscape HDF5 → live star-$R$ (+ kNN) graphs on GPU.

    ``OUT_DIM = 2`` (QCD / top). Particle features are ParT kin7 — the same
    seven channels as JetClass ``--jetclass-features kin7``.
    """

    N_EDGE_FEATURES = 4
    OUT_DIM = 2

    def __init__(
        self,
        data_root: str,
        batch_size: int = 128,
        *,
        num_workers: int | None = None,
        num_particles: int = DEFAULT_NUM_PARTICLES,
        n_train: int | None = None,
        n_val: int | None = None,
        n_test: int | None = None,
        data_seed: int = 42,
    ) -> None:
        root = str(normalize_data_root(data_root))
        super().__init__(root, batch_size, num_workers=num_workers)
        self.num_particles = int(num_particles)
        self.n_train_limit = n_train
        self.n_val_limit = n_val
        self.n_test_limit = n_test
        self.data_seed = int(data_seed)
        self.n_particles = self.num_particles
        for split in ("train", "val", "test"):
            path = raw_hdf5_path(self.data_root, split)
            if not path.is_file():
                raise FileNotFoundError(
                    f"TopTagging live FT requires {path}. "
                    "Download HDF5 splits before launching."
                )

    def dataset_name(self) -> str:
        return "toptagging"

    @property
    def graph_construction(self) -> str:
        return "live"

    def get_dims(self) -> tuple[int, int]:
        return N_PART_KIN_FEATURES, self.OUT_DIM

    def get_edge_dim(self) -> int:
        return self.N_EDGE_FEATURES

    def _ensure_dataset_stats(self) -> None:
        if getattr(self, "_stats_loaded", False):
            return
        stats = get_dataset_stats("toptagging")
        self.corr_adam = stats.corr_adam
        self.corr_sgd = stats.corr_sgd
        self._stats_loaded = True

    def _split_ds(self, split: str, max_jets: int | None) -> TopTaggingLiveJetDataset:
        return TopTaggingLiveJetDataset(
            data_root=self.data_root,
            split=split,
            num_particles=self.num_particles,
            max_jets=max_jets,
            seed=self.data_seed,
        )

    def _build_train_dataset(self) -> Dataset:
        self._ensure_dataset_stats()
        return self._split_ds("train", self.n_train_limit)

    def _build_val_dataset(self) -> Dataset:
        self._ensure_dataset_stats()
        return self._split_ds("val", self.n_val_limit)

    def _build_test_dataset(self) -> Dataset:
        self._ensure_dataset_stats()
        return self._split_ds("test", self.n_test_limit)

    def build_datasets(self) -> tuple[Dataset, Dataset, Dataset]:
        return (
            self._build_train_dataset(),
            self._build_val_dataset(),
            self._build_test_dataset(),
        )

    def setup(self, stage: str | None = None) -> None:
        super().setup(stage)
        if self._train is not None:
            log_info(
                f"[toptagging-live] train={len(self._train)} "
                f"val={len(self._val) if self._val is not None else 0} "
                f"test={len(self._test) if self._test is not None else 0} "
                f"P={self.num_particles}"
            )

    def state_dict(self) -> dict[str, Any]:
        return {
            "num_particles": self.num_particles,
            "n_train_limit": self.n_train_limit,
            "n_val_limit": self.n_val_limit,
            "n_test_limit": self.n_test_limit,
            "data_seed": self.data_seed,
        }
