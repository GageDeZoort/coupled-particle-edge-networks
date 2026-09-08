"""Compatibility shim — prefer ``cpen.apps.streams.lit_cpen_stream``."""
from __future__ import annotations

import importlib as _importlib

_m = _importlib.import_module("cpen.apps.streams.lit_cpen_stream")
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
