"""
Streaming JetClass reader: raw ROOT to ParT features with no materialized cache.

The star-$R$ cache path (:mod:`cpen.apps.jets.jetclass_star_cache`) writes about
46.5 kB per jet, so the full 100M-jet training split would need several TB per
radius and per feature configuration. Scaling-law studies need the dataset-size
axis to reach $10^8$ and need several input configurations, which makes
materialization impractical. The raw ROOT files are 1.6 kB per jet and already
on disk, ``uproot`` reads them at ~87k jets/s per worker, and star-$R$ graph
construction on an A100 runs at ~255k jets/s -- one to eight percent of a CPEN
training step. So this module reads ROOT directly and leaves graph building to
:meth:`cpen.training.base_lit_cpen.BaseLitCPEN.on_after_batch_transfer`.

Two dataset flavours are provided:

``JetClassSubsetDataset``
    Map-style, held in RAM. For the data-repetition regime, where $D$ is small
    (the reference study goes to $D = 10^6$, about 11 GB) and the same jets are
    revisited for many epochs.

``JetClassStreamDataset``
    Iterable, single pass, class-interleaved. For the compute-optimal regime,
    where each jet is seen exactly once and $D$ can reach the full 100M.

Unlike the ``jetclass-lite`` archives this applies only the ParT/Weaver
per-feature affine standardization and *not* the additional row-$L^2$
renormalization: dividing each particle's 17-vector by its own norm makes the
particle-ID one-hots depend on the particle's kinematics, and the affine
constants already deliver $\\Theta(1)$ inputs on their own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from cpen.utils.jetclass import (
    CLASS_FILE_PREFIXES,
    N_CLASSES,
    PART_FEATURE_NAMES,
    _PART_PREPROCESS,
)

#: ROOT branches needed to build the ParT feature set. Deliberately excludes the
#: ten ``label_*`` branches: every file holds a single class, so the label comes
#: from the filename (``test_stream_labels_match_branches`` checks that).
RAW_BRANCHES: tuple[str, ...] = (
    "part_px",
    "part_py",
    "part_pz",
    "part_energy",
    "part_deta",
    "part_dphi",
    "part_d0val",
    "part_d0err",
    "part_dzval",
    "part_dzerr",
    "part_charge",
    "part_isChargedHadron",
    "part_isNeutralHadron",
    "part_isPhoton",
    "part_isElectron",
    "part_isMuon",
    "jet_pt",
    "jet_energy",
)

SPLIT_DIRS: dict[str, str] = {
    "train": "train_100M",
    "val": "val_5M",
    "test": "test_20M",
}

#: Files per class per split, and jets per file. The published JetClass release
#: is uniform: 100k jets per file, 100/5/20 files per class.
FILES_PER_CLASS: dict[str, int] = {"train": 100, "val": 5, "test": 20}
JETS_PER_FILE = 100_000

#: Input configurations from the scaling-law study, as subsets of the 17 ParT
#: features so that every configuration shares one standardization convention.
#: ``kin`` is the reference "kinematic variables only" arm
#: $(\\Delta\\eta, \\Delta\\phi, \\log p_T)$.
FEATURE_CONFIGS: dict[str, tuple[str, ...]] = {
    "full": tuple(PART_FEATURE_NAMES),
    "kin": ("part_deta", "part_dphi", "part_pt_log"),
    "kin7": (
        "part_pt_log",
        "part_e_log",
        "part_logptrel",
        "part_logerel",
        "part_deltaR",
        "part_deta",
        "part_dphi",
    ),
}

DEFAULT_FEATURE_CONFIG = "full"


def feature_names(config: str) -> tuple[str, ...]:
    """Ordered feature names for an input configuration."""
    if config not in FEATURE_CONFIGS:
        raise ValueError(
            f"feature_config must be one of {sorted(FEATURE_CONFIGS)}; got {config!r}"
        )
    return FEATURE_CONFIGS[config]


def n_features(config: str) -> int:
    """Number of particle features for an input configuration."""
    return len(feature_names(config))


def _pad(array, num_particles: int):
    """Clip to ``num_particles`` and zero-pad shorter jets."""
    import awkward as ak

    return ak.to_numpy(
        ak.fill_none(ak.pad_none(array, num_particles, clip=True), 0.0)
    )


def _apply_part_preprocess(name: str, values: np.ndarray) -> np.ndarray:
    """
    ParT/Weaver per-feature affine standardization.

    Deliberately *not* followed by a row-$L^2$ step; see the module docstring.
    """
    subtract, multiply, clip_min, clip_max = _PART_PREPROCESS.get(
        name, (0.0, 1.0, None, None)
    )
    out = (values.astype(np.float32) - np.float32(subtract)) * np.float32(multiply)
    if clip_min is not None or clip_max is not None:
        out = np.clip(out, clip_min, clip_max)
    return out


def featurize_chunk(
    arrays,
    *,
    num_particles: int,
    feature_config: str = DEFAULT_FEATURE_CONFIG,
    label: int,
    sort_by_pt: bool = False,
) -> dict[str, np.ndarray]:
    """
    Turn a chunk of raw ROOT branches into model inputs.

    Returns ``x`` ``(n, P, F)``, ``x_raw`` ``(n, P, 4)`` holding
    ``[E, px, py, pz]`` for live graph construction, ``mask`` ``(n, P)``,
    ``z`` ``(n, P)`` $p_T$ fractions, and ``y`` ``(n,)``.

    ``sort_by_pt`` reorders constituents by decreasing $p_T$ so that truncation
    keeps the hardest ones. The published JetClass files already ship in that
    order (verified over 20k jets), so it defaults to off; enable it for other
    sources such as Aspen Open Jets.
    """
    import awkward as ak

    px, py = arrays["part_px"], arrays["part_py"]
    pt = np.hypot(px, py)

    if sort_by_pt:
        order = ak.argsort(pt, axis=-1, ascending=False)
        arrays = {k: (v[order] if k.startswith("part_") else v) for k, v in arrays.items()}
        px, py = arrays["part_px"], arrays["part_py"]
        pt = np.hypot(px, py)

    energy = arrays["part_energy"]
    raw: dict[str, np.ndarray] = {
        "part_pt_log": _pad(np.log(pt), num_particles),
        "part_e_log": _pad(np.log(energy), num_particles),
        "part_logptrel": _pad(np.log(pt / arrays["jet_pt"]), num_particles),
        "part_logerel": _pad(np.log(energy / arrays["jet_energy"]), num_particles),
        "part_deltaR": _pad(
            np.hypot(arrays["part_deta"], arrays["part_dphi"]), num_particles
        ),
        "part_charge": _pad(arrays["part_charge"], num_particles),
        "part_isChargedHadron": _pad(arrays["part_isChargedHadron"], num_particles),
        "part_isNeutralHadron": _pad(arrays["part_isNeutralHadron"], num_particles),
        "part_isPhoton": _pad(arrays["part_isPhoton"], num_particles),
        "part_isElectron": _pad(arrays["part_isElectron"], num_particles),
        "part_isMuon": _pad(arrays["part_isMuon"], num_particles),
        "part_d0": _pad(np.tanh(arrays["part_d0val"]), num_particles),
        "part_d0err": _pad(arrays["part_d0err"], num_particles),
        "part_dz": _pad(np.tanh(arrays["part_dzval"]), num_particles),
        "part_dzerr": _pad(arrays["part_dzerr"], num_particles),
        "part_deta": _pad(arrays["part_deta"], num_particles),
        "part_dphi": _pad(arrays["part_dphi"], num_particles),
    }

    x_raw = np.stack(
        [
            _pad(energy, num_particles),
            _pad(px, num_particles),
            _pad(py, num_particles),
            _pad(arrays["part_pz"], num_particles),
        ],
        axis=-1,
    ).astype(np.float32)

    # Padded slots carry log(0) = -inf before masking; the mask zeroes them.
    mask = np.hypot(x_raw[..., 1], x_raw[..., 2]) > 1e-8
    names = feature_names(feature_config)
    x = np.stack([_apply_part_preprocess(n, raw[n]) for n in names], axis=-1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) * mask[..., None]
    x_raw = x_raw * mask[..., None]

    pt_kept = np.where(mask, np.hypot(x_raw[..., 1], x_raw[..., 2]), 0.0)
    z = pt_kept / np.clip(pt_kept.sum(axis=-1, keepdims=True), 1e-12, None)

    n = x.shape[0]
    return {
        "x": x.astype(np.float32),
        "x_raw": x_raw,
        "mask": mask,
        "z": (z * mask).astype(np.float32),
        "y": np.full(n, label, dtype=np.int64),
    }


@dataclass(frozen=True)
class Shard:
    """One contiguous read: ``[entry_start, entry_stop)`` of a single-class file."""

    path: Path
    label: int
    entry_start: int
    entry_stop: int

    @property
    def n_jets(self) -> int:
        return self.entry_stop - self.entry_start


def class_files(data_root: str | Path, split: str, label: int) -> list[Path]:
    """Sorted ROOT files for one class, e.g. ``ZJetsToNuNu_000.root`` upward."""
    if split not in SPLIT_DIRS:
        raise ValueError(f"split must be one of {sorted(SPLIT_DIRS)}; got {split!r}")
    prefix = CLASS_FILE_PREFIXES[label][0]
    directory = Path(data_root) / SPLIT_DIRS[split]
    files = sorted(directory.glob(f"{prefix}_*.root"))
    if not files:
        raise FileNotFoundError(
            f"No {prefix}_*.root files under {directory}. Expected the published "
            f"JetClass layout with {FILES_PER_CLASS.get(split, '?')} files per class."
        )
    return files


def plan_shards(
    data_root: str | Path,
    split: str,
    *,
    n_jets: int | None,
    chunk_size: int = 20_000,
) -> list[Shard]:
    """
    Class-balanced read plan.

    Splits ``n_jets`` equally across the ten classes and walks each class's files
    in order, so a given ``n_jets`` always selects the same jets and smaller
    budgets are nested inside larger ones. Shards are interleaved by class so
    that consecutive reads cycle through all ten labels.
    """
    per_class_lists: list[list[Shard]] = []
    for label in range(N_CLASSES):
        files = class_files(data_root, split, label)
        budget = (
            len(files) * JETS_PER_FILE
            if n_jets is None
            else _per_class_quota(n_jets, label)
        )
        shards: list[Shard] = []
        remaining = budget
        for path in files:
            if remaining <= 0:
                break
            start = 0
            while remaining > 0 and start < JETS_PER_FILE:
                take = min(chunk_size, JETS_PER_FILE - start, remaining)
                shards.append(Shard(path, label, start, start + take))
                start += take
                remaining -= take
        per_class_lists.append(shards)

    # Round-robin across classes so any prefix of the plan is class balanced.
    interleaved: list[Shard] = []
    for i in range(max(len(s) for s in per_class_lists)):
        for shards in per_class_lists:
            if i < len(shards):
                interleaved.append(shards[i])
    return interleaved


def _per_class_quota(n_jets: int, label: int) -> int:
    """Split ``n_jets`` over ten classes, giving the remainder to low labels."""
    base, extra = divmod(n_jets, N_CLASSES)
    return base + (1 if label < extra else 0)


def read_shard(
    shard: Shard,
    *,
    num_particles: int,
    feature_config: str,
    sort_by_pt: bool = False,
) -> dict[str, np.ndarray]:
    """Read and featurize one shard."""
    import uproot

    with uproot.open(shard.path) as handle:
        arrays = handle["tree"].arrays(
            list(RAW_BRANCHES),
            entry_start=shard.entry_start,
            entry_stop=shard.entry_stop,
        )
    return featurize_chunk(
        arrays,
        num_particles=num_particles,
        feature_config=feature_config,
        label=shard.label,
        sort_by_pt=sort_by_pt,
    )


class JetClassSubsetDataset(Dataset):
    """
    A fixed, class-balanced JetClass subset held in RAM.

    For the data-repetition regime: the same $D$ jets are revisited every epoch,
    so reading them once and keeping the tensors resident avoids re-decoding
    ROOT on each pass. At ``num_particles=128`` and the full feature set this
    costs about 11 kB per jet, so $D = 10^6$ needs roughly 11 GB.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        *,
        n_jets: int,
        num_particles: int = 128,
        feature_config: str = DEFAULT_FEATURE_CONFIG,
        sort_by_pt: bool = False,
    ) -> None:
        self.split = split
        self.num_particles = num_particles
        self.feature_config = feature_config

        shards = plan_shards(data_root, split, n_jets=n_jets)
        chunks = [
            read_shard(
                s,
                num_particles=num_particles,
                feature_config=feature_config,
                sort_by_pt=sort_by_pt,
            )
            for s in shards
        ]
        self._data = {
            key: torch.from_numpy(np.concatenate([c[key] for c in chunks], axis=0))
            for key in ("x", "x_raw", "mask", "z", "y")
        }
        self.n_jets = int(self._data["y"].shape[0])

    def __len__(self) -> int:
        return self.n_jets

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self._data.items()}


class JetClassStreamDataset(IterableDataset):
    """
    Single-pass, class-interleaved stream over raw JetClass ROOT files.

    For the compute-optimal regime, where every jet is seen exactly once. Shards
    are distributed across DataLoader workers, and each worker shuffles within a
    buffer so batches mix classes without ever holding the split in memory.

    ``shuffle_seed`` permutes the shard order (not the class-balanced plan
    itself), so the set of jets seen for a given ``n_jets`` is reproducible
    while their order varies.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        *,
        n_jets: int | None = None,
        num_particles: int = 128,
        feature_config: str = DEFAULT_FEATURE_CONFIG,
        chunk_size: int = 20_000,
        shuffle_buffer: int = 50_000,
        shuffle_seed: int = 0,
        sort_by_pt: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.num_particles = num_particles
        self.feature_config = feature_config
        self.shuffle_buffer = shuffle_buffer
        self.shuffle_seed = shuffle_seed
        self.sort_by_pt = sort_by_pt
        self.shards = plan_shards(
            data_root, split, n_jets=n_jets, chunk_size=chunk_size
        )
        self.n_jets = sum(s.n_jets for s in self.shards)

    def __len__(self) -> int:
        return self.n_jets

    def _worker_shards(self) -> list[Shard]:
        """Interleaved slice of the plan for this worker, preserving class balance."""
        info = get_worker_info()
        if info is None:
            return list(self.shards)
        return list(self.shards[info.id :: info.num_workers])

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        shards = self._worker_shards()
        info = get_worker_info()
        seed = self.shuffle_seed + (0 if info is None else info.id)
        rng = np.random.default_rng(seed)

        buffer: list[dict[str, torch.Tensor]] = []
        for shard in shards:
            chunk = read_shard(
                shard,
                num_particles=self.num_particles,
                feature_config=self.feature_config,
                sort_by_pt=self.sort_by_pt,
            )
            tensors = {k: torch.from_numpy(v) for k, v in chunk.items()}
            for i in range(shard.n_jets):
                buffer.append({k: v[i] for k, v in tensors.items()})
            if len(buffer) >= self.shuffle_buffer:
                for index in rng.permutation(len(buffer)):
                    yield buffer[index]
                buffer = []
        for index in rng.permutation(len(buffer)):
            yield buffer[index]


def steps_per_epoch(n_jets: int, batch_size: int, num_workers: int) -> int:
    """
    Batches a streaming epoch yields.

    Each worker batches its own shard slice independently, so the last partial
    batch per worker is kept and the total exceeds ``n_jets // batch_size``.
    """
    workers = max(1, num_workers)
    per_worker = n_jets / workers
    return workers * math.ceil(per_worker / batch_size)
