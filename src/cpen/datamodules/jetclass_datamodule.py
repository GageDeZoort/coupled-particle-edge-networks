"""Compatibility shim — prefer ``cpen.apps.jets.jetclass_datamodule``."""
from __future__ import annotations

import importlib as _importlib

_m = _importlib.import_module("cpen.apps.jets.jetclass_datamodule")
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
