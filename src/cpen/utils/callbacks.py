"""Compatibility shim — prefer ``cpen.training.callbacks``."""
from __future__ import annotations

import importlib as _importlib

_m = _importlib.import_module("cpen.training.callbacks")
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
