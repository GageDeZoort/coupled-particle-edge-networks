"""Compatibility shim — prefer ``cpen.apps.jets.part_kin``."""
from __future__ import annotations

import importlib as _importlib

_m = _importlib.import_module("cpen.apps.jets.part_kin")
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
