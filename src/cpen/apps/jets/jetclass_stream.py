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

#: Jets per contiguous ROOT read. The branches sit in ~200-entry baskets, so
#: this is not about basket alignment: it amortizes the fixed per-call cost of
#: ``uproot.arrays`` over the 17 branches. A worker holds one chunk per class,
#: so this also sets the resident set: ~0.55 GB at 128 particles, full features.
DEFAULT_CHUNK_SIZE = 5_000

#: Jets held per worker for shuffling. Arrivals are already class round-robin,
#: so this only has to break the within-class file ordering.
DEFAULT_SHUFFLE_BUFFER = 20_000

#: Files per class per split, and jets per file. The published JetClass release
#: is uniform: 100k jets per file, 100/5/20 files per class.
FILES_PER_CLASS: dict[str, int] = {"train": 100, "val": 5, "test": 20}
JETS_PER_FILE = 100_000

#: Input configurations from the scaling-law study, as subsets of the 17 ParT
#: features so that every configuration shares one standardization convention.
#:
#: ``kin7`` is the Particle Transformer *transfer* recipe: the same seven
#: channels as ``JetClass_kin.yaml`` / ``top_kin.yaml`` (affine only; no
#: row-\(L^2\)). Use this when pretraining on JetClass for TopTagging finetune.
#: ``kin`` is a smaller 3-D ablation \((\\Delta\\eta, \\Delta\\phi, \\log p_T)\),
#: not the ParT ``kin`` yaml.
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
    chunk_size: int = DEFAULT_CHUNK_SIZE,
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

    return _interleave_classes(per_class_lists)


def _interleave_classes(per_class_lists: list[list[Shard]]) -> list[Shard]:
    """Round-robin across classes so any prefix of the plan is class balanced."""
    interleaved: list[Shard] = []
    depth = max((len(s) for s in per_class_lists), default=0)
    for i in range(depth):
        for shards in per_class_lists:
            if i < len(shards):
                interleaved.append(shards[i])
    return interleaved


def split_plan_over_workers(
    shards: Sequence[Shard], worker_id: int, num_workers: int
) -> list[Shard]:
    """
    One worker's slice of the plan, keeping every class in every worker.

    Striding the interleaved plan directly (``shards[id::num_workers]``) aliases
    with the class cycle whenever ``gcd(num_workers, N_CLASSES) > 1``: with six
    workers a worker would only ever see the five even labels. Dealing out each
    class's shards separately and re-interleaving keeps all ten labels, and the
    prefix of each worker's stream class balanced.
    """
    if num_workers <= 1:
        return list(shards)
    per_class: list[list[Shard]] = [[] for _ in range(N_CLASSES)]
    for shard in shards:
        per_class[shard.label].append(shard)
    return _interleave_classes(
        [shards_of_class[worker_id::num_workers] for shards_of_class in per_class]
    )


def split_plan_over_replicas(
    shards: Sequence[Shard],
    *,
    rank: int = 0,
    world_size: int = 1,
    worker_id: int = 0,
    num_workers: int = 1,
) -> list[Shard]:
    """
    One (DDP rank, DataLoader worker) slice of the plan.

    Without this, every rank iterates the full stream and an epoch has
    ``world_size`` times as many optimizer steps as a unique pass at global
    batch ``batch_size * world_size``. Replica slices are class-balanced the
    same way as workers. When ``world_size > 1`` every replica is truncated to
    the shortest replica so DDP ranks exhaust on the same step.
    """
    world_size = max(1, int(world_size))
    num_workers = max(1, int(num_workers))
    rank = int(rank)
    worker_id = int(worker_id)
    if world_size == 1:
        return split_plan_over_workers(shards, worker_id, num_workers)
    n_replicas = world_size * num_workers
    replica_id = rank * num_workers + worker_id
    plan = split_plan_over_workers(shards, replica_id, n_replicas)
    quota = min(
        sum(s.n_jets for s in split_plan_over_workers(shards, i, n_replicas))
        for i in range(n_replicas)
    )
    return take_round_robin_shards(plan, quota)


def chunk_size_for_workers(n_jets: int | None, num_workers: int) -> int:
    """
    Largest chunk size that still gives every worker a shard of every class.

    Workers are dealt each class's shards round-robin, so a class with fewer
    shards than workers leaves some workers without it and their batches
    class-imbalanced. Shrinking the chunk raises the shard count per class.
    """
    workers = max(1, num_workers)
    if n_jets is None:
        per_class = FILES_PER_CLASS["train"] * JETS_PER_FILE
    else:
        per_class = _per_class_quota(n_jets, N_CLASSES - 1)
    return max(1, min(DEFAULT_CHUNK_SIZE, per_class // workers))


def _per_class_quota(n_jets: int, label: int) -> int:
    """Split ``n_jets`` over ten classes, giving the remainder to low labels."""
    base, extra = divmod(n_jets, N_CLASSES)
    return base + (1 if label < extra else 0)


def skip_jets_for_worker(skip_jets: int, *, worker_id: int, num_workers: int) -> int:
    """Split a global skip count across DataLoader workers, remainder first."""
    skip_jets = max(0, int(skip_jets))
    workers = max(1, int(num_workers))
    if workers == 1:
        return skip_jets
    base, rem = divmod(skip_jets, workers)
    return base + (1 if worker_id < rem else 0)


def _trim_class_shards(shards: list[Shard], n_jets: int) -> list[Shard]:
    """Drop the first ``n_jets`` jets from one class's ordered shard list."""
    remaining = max(0, int(n_jets))
    out = list(shards)
    while remaining > 0 and out:
        shard = out[0]
        if shard.n_jets <= remaining:
            remaining -= shard.n_jets
            out.pop(0)
            continue
        out[0] = Shard(
            shard.path, shard.label, shard.entry_start + remaining, shard.entry_stop
        )
        remaining = 0
    return out


def _keep_class_shards(shards: Sequence[Shard], n_jets: int) -> list[Shard]:
    """Keep the first ``n_jets`` jets of one class's ordered shard list."""
    remaining = max(0, int(n_jets))
    out: list[Shard] = []
    for shard in shards:
        if remaining <= 0:
            break
        if shard.n_jets <= remaining:
            out.append(shard)
            remaining -= shard.n_jets
            continue
        out.append(
            Shard(
                shard.path,
                shard.label,
                shard.entry_start,
                shard.entry_start + remaining,
            )
        )
        remaining = 0
    return out


def take_round_robin_shards(shards: Sequence[Shard], n_take: int) -> list[Shard]:
    """
    Keep a class-interleaved prefix of ``n_take`` jets without reading ROOT.

    Matches ``_stream_jets`` emission order. Used to equalize DDP replica
    lengths so every rank exhausts on the same optimizer step.
    """
    pending: list[list[Shard]] = [[] for _ in range(N_CLASSES)]
    for shard in shards:
        pending[shard.label].append(shard)
    left = [sum(s.n_jets for s in pending[i]) for i in range(N_CLASSES)]
    total = sum(left)
    n_take = max(0, min(int(n_take), total))
    if n_take == 0:
        return []
    if n_take == total:
        return list(shards)
    keep = [0] * N_CLASSES
    remaining = n_take
    while remaining > 0:
        active = [label for label in range(N_CLASSES) if keep[label] < left[label]]
        if not active:
            break
        n_active = len(active)
        per_class, leftover = divmod(remaining, n_active)
        if per_class:
            take = min(per_class, min(left[i] - keep[i] for i in active))
            take = max(1, take)
            for label in active:
                keep[label] += take
            remaining -= take * n_active
            continue
        for label in active[:leftover]:
            keep[label] += 1
        remaining = 0
    return _interleave_classes(
        [_keep_class_shards(pending[label], keep[label]) for label in range(N_CLASSES)]
    )


def skip_round_robin_shards(shards: Sequence[Shard], n_skip: int) -> list[Shard]:
    """
    Advance a class-interleaved plan by ``n_skip`` jets without reading ROOT.

    Matches ``_stream_jets`` emission order: one jet from each still-active
    class per cycle. Used to resume a stream after ``last.ckpt``.
    """
    pending: list[list[Shard]] = [[] for _ in range(N_CLASSES)]
    for shard in shards:
        pending[shard.label].append(shard)
    remaining = max(0, int(n_skip))
    while remaining > 0:
        active = [label for label in range(N_CLASSES) if pending[label]]
        if not active:
            break
        n_active = len(active)
        per_class, leftover = divmod(remaining, n_active)
        if per_class:
            take = min(per_class, min(sum(s.n_jets for s in pending[i]) for i in active))
            take = max(1, take)
            for label in active:
                pending[label] = _trim_class_shards(pending[label], take)
            remaining -= take * n_active
            continue
        for label in active[:leftover]:
            pending[label] = _trim_class_shards(pending[label], 1)
        remaining = 0
    out: list[Shard] = []
    for group in pending:
        out.extend(group)
    return out


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
    are distributed across DDP ranks and DataLoader workers, and each worker
    holds one open chunk per class and emits round-robin over them, so every
    batch is class balanced without ever holding the split in memory.

    ``shuffle_seed`` varies the emission order (not the class-balanced plan
    itself) when ``shuffle_buffer > 1``. The set of jets for a given ``n_jets``
    is always the same. ``shuffle_buffer <= 1`` yields the class-interleaved
    ROOT order, so a resumed job that skips already-consumed jets continues
    straight through the files instead of reshuffling the same prefix.
    ``ddp_rank`` / ``ddp_world_size`` must be set on the dataset *before*
    workers are spawned (DataLoader workers do not inherit the process group).
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        *,
        n_jets: int | None = None,
        num_particles: int = 128,
        feature_config: str = DEFAULT_FEATURE_CONFIG,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        shuffle_buffer: int = DEFAULT_SHUFFLE_BUFFER,
        shuffle_seed: int = 0,
        sort_by_pt: bool = False,
        skip_jets: int = 0,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.num_particles = num_particles
        self.feature_config = feature_config
        self.chunk_size = chunk_size
        self.shuffle_buffer = shuffle_buffer
        self.shuffle_seed = shuffle_seed
        self.sort_by_pt = sort_by_pt
        self.skip_jets = max(0, int(skip_jets))
        self.ddp_rank = max(0, int(ddp_rank))
        self.ddp_world_size = max(1, int(ddp_world_size))
        self.shards = plan_shards(
            data_root, split, n_jets=n_jets, chunk_size=chunk_size
        )
        self.n_jets = sum(s.n_jets for s in self.shards)

    def __len__(self) -> int:
        return self.n_jets

    def _replica_ids(self) -> tuple[int, int, int, int]:
        """(rank, world_size, worker_id, num_workers) for this iterator."""
        info = get_worker_info()
        n_workers = 1 if info is None else info.num_workers
        worker_id = 0 if info is None else info.id
        rank = max(0, int(getattr(self, "ddp_rank", 0) or 0))
        world = max(1, int(getattr(self, "ddp_world_size", 1) or 1))
        return rank, world, worker_id, n_workers

    def _worker_shards(self) -> list[Shard]:
        """Slice of the plan for this DDP rank and worker, class balanced."""
        rank, world, worker_id, n_workers = self._replica_ids()
        return split_plan_over_replicas(
            self.shards,
            rank=rank,
            world_size=world,
            worker_id=worker_id,
            num_workers=n_workers,
        )

    def _read_shard_jets(self, shard: Shard) -> list[dict[str, torch.Tensor]]:
        chunk = read_shard(
            shard,
            num_particles=self.num_particles,
            feature_config=self.feature_config,
            sort_by_pt=self.sort_by_pt,
        )
        tensors = {k: torch.from_numpy(v) for k, v in chunk.items()}
        return [{k: v[i] for k, v in tensors.items()} for i in range(shard.n_jets)]

    def _stream_jets(self, shards: Sequence[Shard]) -> Iterator[dict[str, torch.Tensor]]:
        """
        Jets in class round-robin order, one open chunk per class.

        Interleaving has to happen at jet granularity rather than shard
        granularity. A shard holds ``chunk_size`` jets of a single class, so a
        shuffle buffer would have to span a whole class cycle to undo that block
        structure, and even then arrivals leak: a jet inserted into a reservoir
        can be drawn again immediately, so a block of one class raises that
        class's share of the next few thousand emissions. Cycling per-class
        cursors makes every window of ``N_CLASSES`` jets exactly balanced.
        """
        pending: list[list[Shard]] = [[] for _ in range(N_CLASSES)]
        for shard in shards:
            pending[shard.label].append(shard)
        cursors: dict[int, Iterator[dict[str, torch.Tensor]]] = {}
        active = [label for label in range(N_CLASSES) if pending[label]]
        for label in active:
            cursors[label] = iter(self._read_shard_jets(pending[label].pop(0)))

        while active:
            still_active: list[int] = []
            for label in active:
                jet = next(cursors[label], None)
                if jet is None:
                    if not pending[label]:
                        continue
                    cursors[label] = iter(
                        self._read_shard_jets(pending[label].pop(0))
                    )
                    jet = next(cursors[label], None)
                    if jet is None:
                        continue
                yield jet
                still_active.append(label)
            active = still_active

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        rank, _world, worker_id, n_workers = self._replica_ids()
        worker_skip = skip_jets_for_worker(
            self.skip_jets, worker_id=worker_id, num_workers=n_workers
        )
        shards = skip_round_robin_shards(self._worker_shards(), worker_skip)
        stream = self._stream_jets(shards)
        # Sequential pass: skip-ahead lands on the next unread jet in file
        # order. A shuffle buffer would re-permute that suffix with a reset
        # RNG and is not needed for class balance (arrivals are round-robin).
        if int(self.shuffle_buffer) <= 1:
            yield from stream
            return
        seed = self.shuffle_seed + rank * 1024 + worker_id
        rng = np.random.default_rng(seed)
        buffer: list[dict[str, torch.Tensor]] = []
        for jet in stream:
            buffer.append(jet)
            if len(buffer) >= self.shuffle_buffer:
                for index in rng.permutation(len(buffer)):
                    yield buffer[index]
                buffer = []
        for index in rng.permutation(len(buffer)):
            yield buffer[index]


def steps_per_epoch(
    n_jets: int, batch_size: int, num_workers: int, world_size: int = 1
) -> int:
    """
    Optimizer steps a streaming epoch yields on each DDP rank.

    Each DataLoader worker batches its own replica slice independently, so the
    last partial batch per worker is kept and the total can exceed
    ``n_jets // (batch_size * world_size)``. ``world_size`` is the DDP replica
    count: ranks do not re-walk each other's shards.
    """
    world = max(1, int(world_size))
    workers = max(1, int(num_workers))
    per_replica = n_jets / (world * workers)
    return workers * math.ceil(per_replica / batch_size)
