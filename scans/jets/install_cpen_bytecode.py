#!/usr/bin/env python3
"""Validate recovered backend source and install cpen.pyc / capen.pyc.

Usage (on Della, after copying Mac ``Downloads/cpen.py``):

  scp ~/Downloads/cpen.py \\
      della:~/coupled-particle-edge-networks/src/cpen/models/cpen_bytecode.py

  PYTHONPATH=src python scans/jets/install_cpen_bytecode.py
  PYTHONPATH=src pytest tests/test_capen_llama_att.py::test_writeup_pooling_geometry -q
"""

from __future__ import annotations

import ast
import py_compile
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_MODELS = _REPO / "src" / "cpen" / "models"
_SRC = _MODELS / "cpen_bytecode.py"
_CPEN_PYC = _MODELS / "cpen.pyc"
_CAPEN_PYC = _MODELS / "capen.pyc"
_LLAMA_SRC = _MODELS / "capen_llama_bytecode.py"
_LLAMA_PYC = _MODELS / "capen_llama.pyc"

_REQUIRED_ANY = ("CAPEN", "CPEN")
_REQUIRED_ALL = ("SupportAttention",)
_FORBIDDEN_SNIPPETS = (
    "self._readout_scale = 1.0",
    "_readout_scale = 1.0  #",
)


def _validate_source(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for bad in _FORBIDDEN_SNIPPETS:
        if bad in text:
            raise SystemExit(f"refusing install: suspicious snippet {bad!r} in {path}")
    tree = ast.parse(text, filename=str(path))
    defined = {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }
    if not any(n in defined for n in _REQUIRED_ANY):
        raise SystemExit(
            f"{path} must define one of {_REQUIRED_ANY}; "
            f"found classes: {sorted(defined)[:20]}…"
        )
    missing = [n for n in _REQUIRED_ALL if n not in defined]
    if missing:
        raise SystemExit(
            f"{path} missing class(es) {missing}; "
            f"found classes: {sorted(defined)[:20]}…"
        )
    if "readout_scale" not in text and "_readout_scale" not in text:
        raise SystemExit(f"{path}: no _readout_scale / readout_scale — wrong file?")


def _compile(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    py_compile.compile(str(src), cfile=str(dest), doraise=True)
    print(f"  wrote {dest} ({dest.stat().st_size} bytes)")


def main() -> int:
    if not _SRC.is_file():
        raise SystemExit(
            f"Missing {_SRC}\n"
            "Copy your recovered backend to that path (not the thin cpen.py wrapper)."
        )
    print(f"[install] validating {_SRC}")
    _validate_source(_SRC)
    print("[install] compiling cpen.pyc + capen.pyc (same module)")
    _compile(_SRC, _CPEN_PYC)
    shutil.copy2(_CPEN_PYC, _CAPEN_PYC)

    if _LLAMA_SRC.is_file():
        print(f"[install] compiling capen_llama.pyc from {_LLAMA_SRC}")
        _compile(_LLAMA_SRC, _LLAMA_PYC)
    elif not _LLAMA_PYC.is_file():
        print(
            "[install] WARNING: capen_llama.pyc still missing. "
            "Add capen_llama_bytecode.py or restore capen_llama.pyc before "
            "CAPEN-Llama-att probes.",
            file=sys.stderr,
        )
    else:
        print(f"[install] keeping existing {_LLAMA_PYC}")

    print("[install] smoke import …")
    sys.path.insert(0, str(_REPO / "src"))
    from cpen.models.capen import CAPEN, SupportAttention  # noqa: F401

    print("[install] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
