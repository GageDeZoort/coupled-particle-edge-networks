"""Compatibility shim — prefer ``cpen.apps.pascal.pascal_datamodule``."""
from __future__ import annotations

import importlib as _importlib

_m = _importlib.import_module("cpen.apps.pascal.pascal_datamodule")
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
