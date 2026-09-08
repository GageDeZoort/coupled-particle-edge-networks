"""Dataset statistics and correction factors for transfer theory."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetStats:
    """Static dataset metadata used by datamodules and weight-decay proxies."""

    name: str
    n_train: int
    corr_sgd: float = 1.0
    corr_adam: float = 1.0


# Placeholder values; replace when real dataset paths are wired in.
DATASET_STATS = {
    "toptagging": DatasetStats(name="toptagging", n_train=1_211_000),
    "jetclass": DatasetStats(name="jetclass", n_train=5_000_000),
}


def get_dataset_stats(dataset: str) -> DatasetStats:
    key = dataset.lower()
    if key not in DATASET_STATS:
        raise KeyError(f"Unknown dataset {dataset!r}. Known: {list(DATASET_STATS)}")
    return DATASET_STATS[key]
