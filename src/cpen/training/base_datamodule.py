"""Shared LightningDataModule base for jet datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import lightning as L
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset


class BaseDatamodule(L.LightningDataModule, ABC):
    """Standard train/val/test dataloaders with shared worker settings."""

    NUM_WORKERS = 8

    def __init__(
        self,
        data_root: str,
        batch_size: int = 128,
        *,
        num_workers: int | None = None,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else self.NUM_WORKERS

        self.in_dim: int | None = None
        self.out_dim: int | None = None
        self.n_particles: int | None = None
        self.task: str = "classification"
        self.n_train: int = 0
        self.corr_sgd: float = 1.0
        self.corr_adam: float = 1.0

        self._train: Dataset | None = None
        self._val: Dataset | None = None
        self._test: Dataset | None = None

    @abstractmethod
    def dataset_name(self) -> str:
        """Short dataset identifier used in output paths."""

    @abstractmethod
    def build_datasets(self) -> tuple[Dataset, Dataset, Dataset]:
        """Return (train, val, test) datasets."""

    @abstractmethod
    def get_dims(self) -> tuple[int, int]:
        """Return (in_dim, out_dim)."""

    def setup(self, stage: str | None = None) -> None:
        """Load only the splits needed for ``stage`` (fit/validate/test)."""
        if stage in (None, "fit", "validate"):
            if self._train is None:
                self._train = self._build_train_dataset()
            if self._val is None:
                self._val = self._build_val_dataset()
        if stage in (None, "test"):
            if self._test is None:
                self._test = self._build_test_dataset()
        if self._train is not None:
            self.in_dim, self.out_dim = self.get_dims()
            self.n_train = len(self._train)

    def _build_train_dataset(self) -> Dataset:
        train, _, _ = self.build_datasets()
        return train

    def _build_val_dataset(self) -> Dataset:
        _, val, _ = self.build_datasets()
        return val

    def _build_test_dataset(self) -> Dataset:
        _, _, test = self.build_datasets()
        return test

    def _dataloader_kwargs(self, dataset: Dataset | None = None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": True,
        }
        collate_fn = getattr(dataset, "collate_samples", None) if dataset is not None else None
        if collate_fn is not None:
            kwargs["collate_fn"] = collate_fn
            if self.num_workers > 0:
                kwargs["multiprocessing_context"] = mp.get_context("spawn")
                kwargs["persistent_workers"] = True
                kwargs["prefetch_factor"] = 2
        elif self.num_workers > 0:
            kwargs["persistent_workers"] = True
        return kwargs

    def train_dataloader(self) -> DataLoader:
        assert self._train is not None
        return DataLoader(self._train, shuffle=True, **self._dataloader_kwargs(self._train))

    def val_dataloader(self) -> DataLoader:
        assert self._val is not None
        return DataLoader(self._val, shuffle=False, **self._dataloader_kwargs(self._val))

    def test_dataloader(self) -> DataLoader:
        assert self._test is not None
        return DataLoader(self._test, shuffle=False, **self._dataloader_kwargs(self._test))

    def metadata(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_name(),
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "n_particles": self.n_particles,
            "task": self.task,
            "n_train": self.n_train,
            "corr_sgd": self.corr_sgd,
            "corr_adam": self.corr_adam,
        }
