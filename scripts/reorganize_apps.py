#!/usr/bin/env python3
"""One-shot package reorganization: shared core + apps/{jets,streams,pascal}.

Run from repo root. Idempotent-ish: skips missing sources.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if False else Path.cwd()
SRC = ROOT / "src" / "cpen"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def move(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"  skip missing {src.relative_to(ROOT)}")
        return
    ensure_dir(dst.parent)
    if dst.exists():
        print(f"  skip exists {dst.relative_to(ROOT)}")
        return
    print(f"  {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")
    shutil.move(str(src), str(dst))


def write(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text)
    print(f"  write {path.relative_to(ROOT)}")


def shim(old: Path, new_mod: str, *, star: bool = True) -> None:
    """Replace old path with a re-export shim."""
    body = (
        f'"""Compatibility shim — prefer ``{new_mod}``."""\n'
        f"from {new_mod} import *  # noqa: F403\n"
        if star
        else f'"""Compatibility shim — prefer ``{new_mod}``."""\n'
        f"from {new_mod} import *  # noqa: F403\n"
    )
    # Always overwrite old location with shim (after move, old is gone).
    write(old, body)


def main() -> int:
    print("=== creating packages ===")
    for p in [
        SRC / "graphs",
        SRC / "training",
        SRC / "apps",
        SRC / "apps" / "jets",
        SRC / "apps" / "streams",
        SRC / "apps" / "pascal",
        ROOT / "scans" / "jets",
        ROOT / "scans" / "streams",
        ROOT / "scans" / "pascal",
        ROOT / "scans" / "common",
        ROOT / "notebooks" / "jets",
        ROOT / "notebooks" / "streams",
        ROOT / "notebooks" / "pascal",
        ROOT / "notebooks" / "shared",
    ]:
        ensure_dir(p)

    # Package inits
    write(SRC / "apps" / "__init__.py", '"""Physics application packages (jets, streams, pascal)."""\n')
    write(SRC / "apps" / "jets" / "__init__.py", '"""Jet tagging application (TopTagging, JetClass)."""\n')
    write(SRC / "apps" / "streams" / "__init__.py", '"""Stellar-stream application."""\n')
    write(SRC / "apps" / "pascal" / "__init__.py", '"""PascalVOC-SP application."""\n')
    write(SRC / "graphs" / "__init__.py", '"""Shared graph / incidence primitives."""\n')
    write(SRC / "training" / "__init__.py", '"""Shared Lightning training utilities."""\n')

    print("=== moving graphs ===")
    g = SRC / "graphs"
    u = SRC / "utils"
    for name in [
        "sparse_incidence.py",
        "graphs.py",
        "graphs.pyc",
        "graph_star.py",
        "graph_hypergraph.py",
        "graph_radius.py",
        "wire_coordinates.py",
        "operator_gamma.py",
    ]:
        move(u / name, g / name)

    print("=== moving training ===")
    t = SRC / "training"
    for name in [
        "lightning.py",
        "callbacks.pyc",
        "metrics.py",
        "run_tags.py",
        "log_utils.py",
        "sweep_sizes.py",
        "preprocessing.py",
        "attention_temperature.pyc",
        "transfer_plots.py",
        "plotting.py",
    ]:
        move(u / name, t / name)
    move(SRC / "datamodules" / "base_datamodule.py", t / "base_datamodule.py")
    move(SRC / "lit_models" / "base_lit_cpen.py", t / "base_lit_cpen.py")

    print("=== moving apps/jets ===")
    j = SRC / "apps" / "jets"
    for name in [
        "toptagging.py",
        "part_kin.py",
        "graph_cache.py",
        "graph_cache.pyc",
        "star_graph_cache.pyc",
        "jetclass.pyc",
        "jetclass_star_cache.pyc",
        "hier_graph_cache.py",
        "graph_hierarchical.py",
        "wire_cache.py",
        "graph_plotting.py",
        "graph_radius_viz.py",
    ]:
        move(u / name, j / name)
    move(SRC / "datamodules" / "toptagging_datamodule.pyc", j / "toptagging_datamodule.pyc")
    move(SRC / "datamodules" / "toptagging_star_datamodule.py", j / "toptagging_star_datamodule.py")
    move(SRC / "datamodules" / "jetclass_datamodule.py", j / "jetclass_datamodule.py")
    move(SRC / "lit_models" / "lit_cpen_toptagging.py", j / "lit_cpen_toptagging.py")
    move(SRC / "lit_models" / "lit_cpen_jetclass.py", j / "lit_cpen_jetclass.py")

    print("=== moving apps/streams ===")
    s = SRC / "apps" / "streams"
    move(u / "stream_graph_cache.py", s / "stream_graph_cache.py")
    move(SRC / "datamodules" / "stream_datamodule.py", s / "stream_datamodule.py")
    move(SRC / "lit_models" / "lit_cpen_stream.py", s / "lit_cpen_stream.py")

    print("=== moving apps/pascal ===")
    p = SRC / "apps" / "pascal"
    move(u / "pascal_graph_cache.py", p / "pascal_graph_cache.py")
    move(SRC / "datamodules" / "pascal_datamodule.py", p / "pascal_datamodule.py")
    move(SRC / "lit_models" / "lit_cpen_pascal.py", p / "lit_cpen_pascal.py")

    print("=== writing compatibility shims (utils) ===")
    shim_map_utils = {
        # graphs
        "sparse_incidence": "cpen.graphs.sparse_incidence",
        "graphs": "cpen.graphs.graphs",
        "graph_star": "cpen.graphs.graph_star",
        "graph_hypergraph": "cpen.graphs.graph_hypergraph",
        "graph_radius": "cpen.graphs.graph_radius",
        "wire_coordinates": "cpen.graphs.wire_coordinates",
        "operator_gamma": "cpen.graphs.operator_gamma",
        # training
        "lightning": "cpen.training.lightning",
        "callbacks": "cpen.training.callbacks",
        "metrics": "cpen.training.metrics",
        "run_tags": "cpen.training.run_tags",
        "log_utils": "cpen.training.log_utils",
        "sweep_sizes": "cpen.training.sweep_sizes",
        "preprocessing": "cpen.training.preprocessing",
        "attention_temperature": "cpen.training.attention_temperature",
        "transfer_plots": "cpen.training.transfer_plots",
        "plotting": "cpen.training.plotting",
        # jets
        "toptagging": "cpen.apps.jets.toptagging",
        "part_kin": "cpen.apps.jets.part_kin",
        "graph_cache": "cpen.apps.jets.graph_cache",
        "star_graph_cache": "cpen.apps.jets.star_graph_cache",
        "jetclass": "cpen.apps.jets.jetclass",
        "jetclass_star_cache": "cpen.apps.jets.jetclass_star_cache",
        "hier_graph_cache": "cpen.apps.jets.hier_graph_cache",
        "graph_hierarchical": "cpen.apps.jets.graph_hierarchical",
        "wire_cache": "cpen.apps.jets.wire_cache",
        "graph_plotting": "cpen.apps.jets.graph_plotting",
        "graph_radius_viz": "cpen.apps.jets.graph_radius_viz",
        # streams / pascal
        "stream_graph_cache": "cpen.apps.streams.stream_graph_cache",
        "pascal_graph_cache": "cpen.apps.pascal.pascal_graph_cache",
    }
    for name, mod in shim_map_utils.items():
        shim(u / f"{name}.py", mod)

    write(
        u / "__init__.py",
        '"""Legacy utils namespace — prefer ``cpen.graphs``, ``cpen.training``, ``cpen.apps.*``."""\n',
    )

    print("=== writing compatibility shims (datamodules / lit_models) ===")
    shim(SRC / "datamodules" / "base_datamodule.py", "cpen.training.base_datamodule")
    shim(SRC / "datamodules" / "toptagging_datamodule.py", "cpen.apps.jets.toptagging_datamodule")
    shim(SRC / "datamodules" / "toptagging_star_datamodule.py", "cpen.apps.jets.toptagging_star_datamodule")
    shim(SRC / "datamodules" / "jetclass_datamodule.py", "cpen.apps.jets.jetclass_datamodule")
    shim(SRC / "datamodules" / "stream_datamodule.py", "cpen.apps.streams.stream_datamodule")
    shim(SRC / "datamodules" / "pascal_datamodule.py", "cpen.apps.pascal.pascal_datamodule")

    shim(SRC / "lit_models" / "base_lit_cpen.py", "cpen.training.base_lit_cpen")
    shim(SRC / "lit_models" / "lit_cpen_toptagging.py", "cpen.apps.jets.lit_cpen_toptagging")
    shim(SRC / "lit_models" / "lit_cpen_jetclass.py", "cpen.apps.jets.lit_cpen_jetclass")
    shim(SRC / "lit_models" / "lit_cpen_stream.py", "cpen.apps.streams.lit_cpen_stream")
    shim(SRC / "lit_models" / "lit_cpen_pascal.py", "cpen.apps.pascal.lit_cpen_pascal")

    # pyc-only loaders for training + jets
    print("=== writing pyc loaders ===")
    pyc_loader = '''\
"""Load sibling ``{pyc_name}`` bytecode as this module."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PYC = Path(__file__).resolve().with_name("{pyc_name}")
if not _PYC.is_file():
    raise ImportError(f"Missing bytecode backend {{_PYC}}")

_spec = importlib.util.spec_from_file_location("{bc_name}", _PYC)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load {{_PYC}}")
_bc = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("{bc_name}", _bc)
_spec.loader.exec_module(_bc)

_PUBLIC = __name__
for _name, _value in vars(_bc).items():
    if _name.startswith("__"):
        continue
    if isinstance(_value, type) or callable(_value):
        try:
            _value.__module__ = _PUBLIC
        except (AttributeError, TypeError):
            pass
    globals()[_name] = _value
'''
    for leaf, pyc, bc in [
        ("callbacks", "callbacks.pyc", "_cpen_callbacks_bc"),
        ("attention_temperature", "attention_temperature.pyc", "_cpen_attn_temp_bc"),
    ]:
        write(t / f"{leaf}.py", pyc_loader.format(pyc_name=pyc, bc_name=bc))

    for leaf, pyc, bc in [
        ("star_graph_cache", "star_graph_cache.pyc", "_cpen_star_graph_cache_bc"),
        ("jetclass", "jetclass.pyc", "_cpen_jetclass_bc"),
        ("jetclass_star_cache", "jetclass_star_cache.pyc", "_cpen_jetclass_star_cache_bc"),
        ("toptagging_datamodule", "toptagging_datamodule.pyc", "_cpen_toptag_dm_bc"),
    ]:
        write(j / f"{leaf}.py", pyc_loader.format(pyc_name=pyc, bc_name=bc))

    print("=== moving scans ===")
    st = ROOT / "scans" / "testing"
    sj = ROOT / "scans" / "jets"
    ss = ROOT / "scans" / "streams"
    sp = ROOT / "scans" / "pascal"
    sc = ROOT / "scans" / "common"

    jet_scans = [
        "build_toptagging_graphs.py",
        "build_toptagging_knn_cputest.py",
        "build_toptagging_hier_cputest.py",
        "build_toptagging_star_graphs.py",
        "build_toptagging_wire_coordinates.py",
        "build_jetclass_lite.py",
        "build_jetclass_lite_star_graphs.py",
        "test_cpen_toptagging.py",
        "test_cpen_jetclass.py",
        "build_graphs.slurm",
        "build_knn_graphs.slurm",
        "build_knn_graphs_cputest.slurm",
        "submit_knn_graphs_cputest.sh",
        "build_hier_graphs_cputest.slurm",
        "submit_hier_graphs_cputest.sh",
        "build_star_graphs.slurm",
        "build_jetclass_lite.slurm",
        "build_jetclass_lite_cputest.slurm",
        "submit_jetclass_lite_cputest.sh",
        "test_cpen_toptagging.slurm",
        "test_capen_llama_toptagging_kNN.slurm",
        "test_jetclass.slurm",
    ]
    for name in jet_scans:
        move(st / name, sj / name)

    for name in ["test_cpen_stream.py", "test_stream.slurm"]:
        move(st / name, ss / name)

    for name in [
        "build_pascal_graphs.py",
        "download_pascal_graphs.py",
        "build_pascal_graphs.slurm",
        "download_pascal_graphs.slurm",
        "test_pascal.slurm",
    ]:
        move(st / name, sp / name)

    for name in [
        "sweep_common.py",
        "_ddp_metrics_check.py",
        "profile_cpen_step.py",
        "profile.slurm",
        "test_cpen.py",
        "test.slurm",
        "__init__.py",
    ]:
        move(st / name, sc / name)

    print("=== moving notebooks ===")
    nb = ROOT / "notebooks"
    for name in [
        "attention_temperature_gamma.ipynb",
        "attention_temperature_gamma_knn.ipynb",
        "attention_temperature_gamma_jetclass.ipynb",
        "dbscan_jet_study.ipynb",
        "estimate-gamma.ipynb",
        "jet_graph_study.ipynb",
        "knn_graph_study.ipynb",
        "radius_hypergraph_study.ipynb",
        "star_graph_viz.ipynb",
        "transfer_scaling_11only.ipynb",
        "transfer_scaling_11only.png",
        "transfer_scaling_11only_final.png",
        "transfer_scaling_eta0_sweep.png",
    ]:
        move(nb / name, nb / "jets" / name)
    if (nb / "outputs").exists():
        move(nb / "outputs", nb / "jets" / "outputs")

    for name in ["stream_checkpoint_viz.ipynb", "stream_label_sanity.ipynb"]:
        move(nb / name, nb / "streams" / name)
    if (nb / "figures").exists():
        move(nb / "figures", nb / "streams" / "figures")

    for name in ["run_result_plots.ipynb", "mlp_transfer_scaling_talk.ipynb", "Untitled.ipynb", ".gitkeep"]:
        move(nb / name, nb / "shared" / name)

    print("=== done moves ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
