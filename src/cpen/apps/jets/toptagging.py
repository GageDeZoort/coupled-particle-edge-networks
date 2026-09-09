"""TopTagging dataset loading helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from cpen.utils.log_utils import log_info
from cpen.utils.part_kin import (
    N_PART_KIN_FEATURES,
    build_part_kin_features,
)

SPLIT_TO_JETNET = {"train": "train", "val": "valid", "test": "test"}
SPLIT_TO_HDF5 = {"train": "train.h5", "val": "val.h5", "test": "test.h5"}
N_PARTICLE_FEATURES = 4  # raw HDF5 layout [E, px, py, pz]
N_MODEL_FEATURES = N_PART_KIN_FEATURES  # ParT TopLandscape kin
MAX_NUM_PARTICLES = 200  # HDF5 storage width; model default pad is 128
DEFAULT_NUM_PARTICLES = 128  # ParT TopLandscape pad/truncate length


def normalize_data_root(data_root: str | Path) -> Path:
    """
    TopTagging dataset root: contains ``train.h5`` and a ``processed/`` subdir.

    Accepts either ``.../toptagging`` or ``.../toptagging/processed`` and returns
    the former so cache paths are not doubled as ``processed/processed/...``.
    """
    root = Path(data_root).expanduser()
    if root.name == "processed":
        parent = root.parent
        if any((parent / name).is_file() for name in SPLIT_TO_HDF5.values()):
            return parent
    return root


def particle_mask_from_four_vectors(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """True for real particles (pT > 0), false for zero-padding."""
    px, py = x[..., 1], x[..., 2]
    pt = torch.sqrt(px * px + py * py)
    return pt > eps


def zero_masked_particles(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        mask = particle_mask_from_four_vectors(x)
    return x * mask.unsqueeze(-1).to(dtype=x.dtype)


def preprocess_particle_features(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Mask padding, retain raw four-vectors, and build ParT-kin model inputs.

    Returns (x_raw, x_model, mask) where ``x_model`` holds the seven
    TopLandscape / JetClass_kin features with ParT affine standardization
    (no row-\(L^2\)). ``x_raw`` keeps physical ``[E, px, py, pz]`` for graph
    construction and edge features.
    """
    mask = particle_mask_from_four_vectors(x)
    x_raw = zero_masked_particles(x, mask)
    x_model = build_part_kin_features(x_raw, mask)
    return x_raw, x_model, mask


def raw_hdf5_path(data_root: str | Path, split: str) -> Path:
    if split not in SPLIT_TO_HDF5:
        raise ValueError(f"split must be one of {list(SPLIT_TO_HDF5)}; got {split!r}")
    return Path(data_root) / SPLIT_TO_HDF5[split]


def raw_splits_available(data_root: str | Path) -> bool:
    root = Path(data_root)
    return all((root / name).is_file() for name in SPLIT_TO_HDF5.values())


class RawTopTaggingSplit(Dataset):
    """
    Read TopTagging HDF5 directly from disk without contacting Zenodo.

    Mirrors jetnet's TopTagging layout for particle/jet features.
    """

    def __init__(
        self,
        *,
        data_root: str,
        split: str,
        num_particles: int = DEFAULT_NUM_PARTICLES,
    ) -> None:
        if split not in SPLIT_TO_HDF5:
            raise ValueError(f"split must be one of {list(SPLIT_TO_HDF5)}; got {split!r}")

        hdf5_path = raw_hdf5_path(data_root, split)
        if not hdf5_path.is_file():
            raise FileNotFoundError(
                f"Missing TopTagging split file {hdf5_path}. "
                "Download on a login node with --download."
            )

        data = np.array(pd.read_hdf(hdf5_path, key="table"))
        total_particle_features = MAX_NUM_PARTICLES * N_PARTICLE_FEATURES
        self.particle_data = data[:, :total_particle_features].reshape(
            -1, MAX_NUM_PARTICLES, N_PARTICLE_FEATURES
        )[:, :num_particles]
        self.jet_data = np.concatenate(
            (data[:, -1:], data[:, total_particle_features : total_particle_features + 4]),
            axis=-1,
        )
        log_info(
            f"[data] Loaded {len(self.jet_data)} jets from {hdf5_path.name} "
            f"({num_particles} particles/jet, {N_PARTICLE_FEATURES} raw features)"
        )

    def __len__(self) -> int:
        return len(self.jet_data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        particles = torch.from_numpy(self.particle_data[idx].astype(np.float32))
        jet = torch.from_numpy(self.jet_data[idx].astype(np.float32))
        return particles, jet


# Backwards-compatible alias
_LocalTopTaggingBackend = RawTopTaggingSplit


class TopTaggingJetDataset(Dataset):
    """
    TopTagging split with optional random subsampling.

    Returns ParT-kin constituent features ``(num_particles, 7)`` and a class
    label (0=qcd, 1=top).
    """

    def __init__(
        self,
        *,
        data_root: str,
        split: str,
        num_particles: int = DEFAULT_NUM_PARTICLES,
        max_jets: int | None = None,
        seed: int = 42,
        download: bool = False,
    ) -> None:
        if split not in SPLIT_TO_JETNET:
            raise ValueError(f"split must be one of {list(SPLIT_TO_JETNET)}; got {split!r}")

        self.split = split
        self.num_particles = num_particles

        hdf5_path = raw_hdf5_path(data_root, split)
        if hdf5_path.is_file():
            self._base: Dataset = RawTopTaggingSplit(
                data_root=data_root,
                split=split,
                num_particles=num_particles,
            )
        else:
            if not download:
                raise FileNotFoundError(
                    f"Missing {hdf5_path} and download=False. "
                    "Fetch raw TopTagging HDF5 on a login node first."
                )
            try:
                from jetnet.datasets import TopTagging
            except ImportError as exc:
                raise ImportError(
                    "jetnet is required for TopTagging loading. Install with: pip install jetnet"
                ) from exc
            self._base = TopTagging(
                jet_type="all",
                data_dir=data_root,
                particle_features=["E", "px", "py", "pz"],
                jet_features=["type"],
                num_particles=num_particles,
                split=SPLIT_TO_JETNET[split],
                download=True,
            )

        n_total = len(self._base)
        if max_jets is not None and max_jets < n_total:
            split_offset = {"train": 0, "val": 1, "test": 2}[split]
            rng = np.random.default_rng(seed + split_offset)
            self._indices = rng.choice(n_total, size=max_jets, replace=False)
        else:
            self._indices = np.arange(n_total)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        particles, jet = self._base[int(self._indices[idx])]
        _, x_model, _ = preprocess_particle_features(particles.float())
        label = int(jet[0].item())
        return x_model, torch.tensor(label, dtype=torch.long)


def ensure_raw_splits(
    data_root: str,
    *,
    num_particles: int = DEFAULT_NUM_PARTICLES,
    download: bool = False,
) -> None:
    """Touch all TopTagging splits so HDF5 files exist locally."""
    for split in SPLIT_TO_HDF5:
        path = raw_hdf5_path(data_root, split)
        if path.is_file():
            continue
        if not download:
            raise FileNotFoundError(f"Missing raw split {path}; rerun with --download on a login node.")
        TopTaggingJetDataset(
            data_root=data_root,
            split=split,
            num_particles=num_particles,
            download=True,
        )
