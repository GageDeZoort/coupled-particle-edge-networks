"""JetClass constants + helpers (replaces lost ``jetclass.pyc``).

Values match ParT/Weaver ``JetClass_full.yaml`` / ``JetClass_kin.yaml`` and the
filename↔label layout used by ``jetclass_stream``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

#: ParT pad/truncate length on JetClass.
DEFAULT_NUM_PARTICLES = 128

N_CLASSES = 10

#: One-hot truth branches in the published ROOT files (label index order).
LABEL_NAMES: tuple[str, ...] = (
    "label_QCD",
    "label_Hbb",
    "label_Hcc",
    "label_Hgg",
    "label_H4q",
    "label_Hqql",
    "label_Zqq",
    "label_Wqq",
    "label_Tbqq",
    "label_Tbl",
)

#: Filename prefixes per class label. ``[label][0]`` is the primary stem.
CLASS_FILE_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("ZJetsToNuNu",),
    ("HToBB",),
    ("HToCC",),
    ("HToGG",),
    ("HToWW4Q",),
    ("HToWW2Q1L",),
    ("ZToQQ",),
    ("WToQQ",),
    ("TTBar",),
    ("TTBarLep",),
)

#: Full 17-D ParT particle feature stack (Weaver ``pf_features`` order).
PART_FEATURE_NAMES: tuple[str, ...] = (
    "part_pt_log",
    "part_e_log",
    "part_logptrel",
    "part_logerel",
    "part_deltaR",
    "part_charge",
    "part_isChargedHadron",
    "part_isNeutralHadron",
    "part_isPhoton",
    "part_isElectron",
    "part_isMuon",
    "part_d0",
    "part_d0err",
    "part_dz",
    "part_dzerr",
    "part_deta",
    "part_dphi",
)

N_PART_FEATURES = len(PART_FEATURE_NAMES)

#: ``(subtract, multiply, clip_min, clip_max)`` — Weaver manual preprocess.
#: ``None`` clips mean “no clip” (see ``jetclass_stream._apply_part_preprocess``).
_PART_PREPROCESS: dict[str, tuple[float, float, float | None, float | None]] = {
    "part_pt_log": (1.7, 0.7, -5.0, 5.0),
    "part_e_log": (2.0, 0.7, -5.0, 5.0),
    "part_logptrel": (-4.7, 0.7, -5.0, 5.0),
    "part_logerel": (-4.7, 0.7, -5.0, 5.0),
    "part_deltaR": (0.2, 4.0, -5.0, 5.0),
    "part_d0err": (0.0, 1.0, 0.0, 1.0),
    "part_dzerr": (0.0, 1.0, 0.0, 1.0),
}


def build_part_features(*args: Any, **kwargs: Any) -> Any:
    """Delegate to the streaming featurizer (RAM-subset / lite path)."""
    from cpen.apps.jets.jetclass_stream import featurize_chunk

    return featurize_chunk(*args, **kwargs)


def read_jetclass_root_file(
    path: str | Path,
    *,
    entry_start: int = 0,
    entry_stop: int | None = None,
    branches: list[str] | tuple[str, ...] | None = None,
) -> dict[str, np.ndarray]:
    """Read a JetClass ROOT file tree into numpy/awkward-friendly arrays."""
    import uproot

    from cpen.apps.jets.jetclass_stream import RAW_BRANCHES

    path = Path(path)
    keys = list(branches) if branches is not None else list(RAW_BRANCHES)
    with uproot.open(path) as handle:
        tree = handle["tree"]
        return tree.arrays(
            keys,
            entry_start=entry_start,
            entry_stop=entry_stop,
            library="np",
        )
