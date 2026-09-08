"""Compatibility shim — prefer ``cpen.apps.streams.stream_graph_cache``."""
from __future__ import annotations

import importlib as _importlib

_m = _importlib.import_module("cpen.apps.streams.stream_graph_cache")
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
