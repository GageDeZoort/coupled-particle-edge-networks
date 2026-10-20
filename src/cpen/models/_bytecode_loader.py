"""Load gitignored model bytecode or reconstructed ``*_bytecode.py`` sources."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_FALLBACKS = {
    "cpen.pyc": "cpen_bytecode.py",
    "capen.pyc": "capen_bytecode.py",
    "capen_llama.pyc": "capen_llama_bytecode.py",
}


def load_backend(*, pyc_name: str, module_key: str) -> ModuleType:
    """Exec ``pyc_name`` in ``models/``, or the reconstructed ``*_bytecode.py``."""
    here = Path(__file__).resolve().parent
    pyc = here / pyc_name
    if pyc.is_file():
        path: Path = pyc
    else:
        fb = _FALLBACKS.get(pyc_name)
        candidate = here / fb if fb else None
        if candidate is None or not candidate.is_file():
            raise ImportError(
                f"Missing bytecode backend {pyc}"
                + (f" (and no reconstructed source {candidate})" if candidate else "")
            )
        path = candidate
    spec = importlib.util.spec_from_file_location(module_key, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load backend for {pyc_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(module_key, mod)
    spec.loader.exec_module(mod)
    return mod
