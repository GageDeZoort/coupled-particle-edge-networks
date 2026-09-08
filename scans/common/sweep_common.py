"""Sweep helpers.

The full implementation currently lives in ``sweep_common.pyc`` (source was
lost in a restore). This module loads that bytecode and wraps:

* ``run_options_from_args`` so ``operator_normalization`` is tagged
  (``onorm-degree`` / ``onorm-gamma``)
* ``add_common_args`` / ``configure_graph_args`` for ``--knn-min-pt``, which
  rewrites ``--graph-construction 8-NN`` into ``8-NN@pt1`` (etc.)
* defaults ``--heavy-metrics-frac`` to ``0`` (ROC AUC / bg rejection every val)
* optional ``--val-check-interval`` passed through to Lightning
* ``build_model`` forwards ``--dropout`` into CPEN MLP residuals
* stream obvious-star mask re-partition (``--stream-mask-seed``)
* ``--include-real-streams`` to train on non-MOCK ``stream_*.pt`` as well
* ``--edge-loss-weight`` / ``--checkpoint-monitor`` for stream edge learning
* ``--identity-m22`` skips CAPEN/CAPEN-Llama edge–edge attention (``f_22 = Id``)
* ``--incidence-m22`` replaces it with linear ``f_22 = S S^T h_e`` (no ``M×M`` attn)
* ``--hyperedge-only`` drops all pairwise 2-edges (kNN, vn_link, deg≤2) at train time
* ``capen-llama-att`` + hierarchical TopTagging caches (``--hier-k`` …)
* attention ``γ_rs`` estimated from the train cache before each fit
  (``--attention-normalization gamma``; opt out with ``--no-estimate-gamma-rs``)
"""

from __future__ import annotations

import contextvars
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

_PYC = Path(__file__).resolve().with_name("sweep_common.pyc")
if not _PYC.is_file():
    raise ImportError(f"Missing bytecode backend {_PYC}")

_spec = importlib.util.spec_from_file_location("_cpen_sweep_common_bc", _PYC)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load {_PYC}")
_bc = importlib.util.module_from_spec(_spec)
# Register before exec so relative/circular imports inside the pyc resolve.
sys.modules.setdefault("_cpen_sweep_common_bc", _bc)
_spec.loader.exec_module(_bc)

# Re-export the bytecode API under this module's name.
# Keep single-underscore helpers (e.g. ``_default_operators``); skip dunders.
for _name, _value in vars(_bc).items():
    if _name.startswith("__"):
        continue
    globals()[_name] = _value

_orig_run_options_from_args = _bc.run_options_from_args
_orig_add_common_args = _bc.add_common_args
_orig_configure_graph_args = _bc.configure_graph_args
_orig_train_one_run = _bc.train_one_run
_orig_resolve_attention_gammas = _bc.resolve_attention_gammas
_orig_build_trainer = _bc.build_trainer
_orig_build_model = _bc.build_model
_orig_create_toptagging_datamodule = _bc.create_toptagging_datamodule
_orig_create_jetclass_datamodule = _bc.create_jetclass_datamodule

_CAPEN_LLAMA_ATT = "capen-llama-att"
_orig_create_stream_datamodule = _bc.create_stream_datamodule

# train_one_run (bytecode) does not know about val_check_interval; inject via ctx.
_val_check_interval_ctx: contextvars.ContextVar[float | int | None] = contextvars.ContextVar(
    "cpen_val_check_interval", default=None
)
_checkpoint_monitor_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "cpen_checkpoint_monitor", default=None
)
_checkpoint_mode_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "cpen_checkpoint_mode", default=None
)


def _format_count(n: int) -> str:
    """Compact dataset-size tag: 1000 -> 1k, 2_500_000 -> 2p5M."""
    for scale, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "k")):
        if n >= scale:
            value = n / scale
            text = f"{value:g}".replace(".", "p")
            return f"{text}{suffix}"
    return str(n)


def run_options_from_args(args):  # type: ignore[no-untyped-def]
    """Like the bytecode helper, but always tag operator normalization."""
    from cpen.utils.run_tags import format_eta

    restore_dataset = None
    alias_dataset = None
    dataset_arg = str(getattr(args, "dataset", "") or "")
    if dataset_arg in {"mnist-sp", "mnist"}:
        alias_dataset = "mnist"
    elif dataset_arg == "qm9":
        alias_dataset = "qm9"
    if alias_dataset is not None:
        restore_dataset = args.dataset
        # Bytecode run-dir helper only knows the original three datasets.
        args.dataset = "toptagging"
    try:
        opts = _orig_run_options_from_args(args)
    finally:
        if restore_dataset is not None:
            args.dataset = restore_dataset
    kwargs = {
        "operator_normalization": getattr(args, "operator_normalization", "degree"),
    }
    if alias_dataset is not None and hasattr(opts, "dataset"):
        kwargs["dataset"] = alias_dataset
    tags: list[str] = []
    # Blob runs reuse dataset=stream, so without a tag they would resume the
    # per-MOCK stream run dir (and each other) at equal depth/width.
    if bool(getattr(args, "blob_run", False)):
        tags.append("blobs")
        rich = int(getattr(args, "blob_rich_min_mock", 100) or 100)
        if rich != 100:
            tags.append(f"rich{rich}")
        empty_per_rich = float(getattr(args, "blob_empty_per_rich", 1.0) or 1.0)
        if abs(empty_per_rich - 1.0) > 1e-12:
            tags.append(f"emp{format_eta(empty_per_rich)}")
        if bool(getattr(args, "blob_include_dilute", False)):
            tags.append("dilute")
        if str(getattr(args, "blob_node_frame", "local") or "local") != "local":
            tags.append("rawframe")
        # Feature ablations must not resume a run trained on other columns.
        from cpen.apps.streams.blob_graph_cache import parse_drop_features

        dropped = parse_drop_features(getattr(args, "blob_drop_features", None))
        tags.append("drop" + "-".join(dropped) if dropped else "allfeat")
        if bool(getattr(args, "blob_no_hyperedges", False)):
            tags.append("nohyper")
        else:
            tags.append("hyper")
    # Tag mock+real / real-only stream runs so they don't collide with MOCK-only dirs.
    stream_name = str(getattr(args, "stream_name", "all") or "all").strip().lower()
    include_real = bool(getattr(args, "include_real_streams", False))
    if stream_name in {"real", "reals"}:
        tags.append("streams-real")
    elif include_real or stream_name in {
        "mock+real",
        "real+mock",
        "both",
        "all+real",
        "all_streams",
    }:
        tags.append("streams-mock+real")
    edge_aux = str(getattr(args, "edge_aux", "hard-ce") or "hard-ce").strip().lower().replace("_", "-")
    if edge_aux in {"product", "consistency", "product-cons", "pu-pv"}:
        edge_aux = "product-consistency"
    if edge_aux in {"hard+product", "both", "ce+cons"}:
        edge_aux = "hard+product"
    if edge_aux in {"product-consistency", "hard+product"}:
        tags.append("edgecons")
    # Tag non-default edge loss weight so w≠1 does not resume a w=1 run dir.
    edge_w = float(getattr(args, "edge_loss_weight", 1.0) or 1.0)
    if abs(edge_w - 1.0) > 1e-12:
        tags.append(f"edgew{format_eta(edge_w)}")
    if bool(getattr(args, "identity_m22", False)):
        tags.append("idm22")
    if bool(getattr(args, "incidence_m22", False)):
        tags.append("incm22")
    if bool(getattr(args, "hyperedge_only", False)):
        tags.append("nopair")
    dataset = str(getattr(args, "dataset", "") or "")
    readout = str(getattr(args, "readout_mode", "") or "")
    if readout == "node+edge" and dataset == "pascalvoc-sp":
        tags.append("edgeboundary")
    if dataset in {"mnist-sp", "mnist"}:
        if readout == "attn":
            tags.append("gattn")
        elif readout == "node":
            tags.append("gx")
        else:
            tags.append("gxe")
        tags.append(f"seed{int(getattr(args, 'seed', 0) or 0)}")
    # Pascal readout study: seed in the run name so repeats don't resume
    # each other and the notebook can collate error bars.
    if dataset == "pascalvoc-sp":
        tags.append(f"seed{int(getattr(args, 'seed', 0) or 0)}")
    # One QM9 cache serves every target column, so the target has to be in the
    # run name or a mu run would resume a gap run at equal depth/width.
    if dataset == "qm9":
        tags.append(str(getattr(args, "qm9_target", "mu") or "mu"))
        tags.append(f"seed{int(getattr(args, 'seed', 0) or 0)}")
    # Scaling-law runs vary the dataset size D at equal depth/width, so D has to
    # be in the run name or a D=1e5 point would resume a D=1e6 one. Same for the
    # input configuration and the edge featurization.
    if jetclass_stream_active(args):
        tags.append("stream")
        features = str(getattr(args, "jetclass_features", "full") or "full")
        if features != "full":
            tags.append(features)
        edge_features = str(
            getattr(args, "jetclass_edge_features", "logdot-dp") or "logdot-dp"
        )
        if edge_features != "logdot-dp":
            tags.append("partint")
        n_particles = int(getattr(args, "num_particles", 128) or 128)
        if n_particles != 128:
            tags.append(f"p{n_particles}")
        n_train = getattr(args, "n_train", None)
        tags.append("Dfull" if n_train is None else f"D{_format_count(int(n_train))}")
        tags.append(f"seed{int(getattr(args, 'seed', 0) or 0)}")
    if tags:
        kwargs["extra_tag"] = "_".join(tags)
    return replace(opts, **kwargs)


def add_common_args(parser):  # type: ignore[no-untyped-def]
    """Bytecode CLI plus kNN / validation metric knobs."""
    _orig_add_common_args(parser)
    # Override bytecode default (0.1 ≈ every 10% of epochs) → every validation.
    for action in parser._actions:
        if getattr(action, "dest", None) == "heavy_metrics_frac":
            action.default = 0.0
            action.help = (
                "Compute ROC AUC / bg rejection on this fraction of epochs "
                "(plus final). <=0 means every validation (default: 0)."
            )
        elif getattr(action, "dest", None) == "stream_name":
            action.help = (
                "Stellar-stream selection: 'all'/'mock' (every stream_MOCK_*.pt), "
                "'real' (non-MOCK stream_*.pt), 'mock+real'/'both', a single stem "
                "(e.g. MOCK_1863 or Chenab), or a comma-separated list. "
                "With --include-real-streams, 'all' expands to MOCK+real."
            )
        elif getattr(action, "dest", None) == "model" and action.choices is not None:
            # Bytecode choices omit CAPEN-Llama-att.
            choices = list(action.choices)
            if _CAPEN_LLAMA_ATT not in choices:
                choices.append(_CAPEN_LLAMA_ATT)
            action.choices = choices
        elif getattr(action, "dest", None) == "estimate_gamma_rs":
            action.default = True
            action.help = (
                "Estimate gamma_{rs} from a training-cache subsample before fit "
                "(default: on). For CAPEN attention this is the mean support row "
                "degree; hierarchical jobs use the same M11/M12/M21/M22 policy as "
                "training (all-to-all nodes, hyperedge M22, drop pairwise 2-edges "
                "when --hyperedge-only). Logged to stdout and parquet."
            )
        elif getattr(action, "dest", None) == "dataset" and action.choices is not None:
            choices = list(action.choices)
            for name in ("mnist-sp", "qm9"):
                if name not in choices:
                    choices.append(name)
            action.choices = choices
        elif getattr(action, "dest", None) == "readout_mode":
            action.default = None
            action.choices = ("graph", "node", "node+edge", "attn")
            action.help = (
                "CPEN decoder: 'graph' pooled z_X+z_E, 'node' decode X only, "
                "'node+edge' decode X and E, 'attn' class-token MHSA over "
                "nodes+edges (off unless set). Pascal defaults to node "
                "(node+edge = boundary co-train). MNIST defaults to node "
                "(pooled z_X); node+edge mixes pooled z_X and z_E."
            )
        elif getattr(action, "dest", None) == "seed":
            action.help = (
                "Global RNG seed (Lightning seed_everything, including "
                "DataLoader workers). Tagged in Pascal run names as seed{N}."
            )
    parser.add_argument(
        "--no-estimate-gamma-rs",
        dest="estimate_gamma_rs",
        action="store_false",
        help=(
            "Do not estimate gamma_{rs} from the cache; use explicit "
            "--gamma-11/12/21/22 instead."
        ),
    )
    parser.add_argument(
        "--knn-min-pt",
        type=float,
        default=None,
        help=(
            "For --graph-construction k-NN caches: keep a directed edge only if "
            "at least one endpoint has pT >= this value (GeV). Encodes into the "
            "construction string as e.g. 8-NN@pt1. Default: no pT cut."
        ),
    )
    from cpen.apps.qm9.qm9_graph_cache import DEFAULT_TARGET, QM9_TARGETS

    parser.add_argument(
        "--qm9-target",
        type=str,
        default=DEFAULT_TARGET,
        choices=sorted(QM9_TARGETS),
        help=(
            "QM9 regression target. One cache holds every column, so this "
            "selects at train time: mu = dipole moment (Debye), gap = "
            "HOMO-LUMO gap (eV), u0 = internal energy at 0K (eV). "
            f"Default: {DEFAULT_TARGET}."
        ),
    )
    parser.add_argument(
        "--qm9-loss",
        type=str,
        default="l1",
        choices=("l1", "mse", "huber"),
        help=(
            "QM9 regression loss on the standardized target. MAE is always "
            "logged in the target's physical unit. Default: l1."
        ),
    )
    parser.add_argument(
        "--jetclass-stream",
        action="store_true",
        help=(
            "Read JetClass straight from the published ROOT files and build "
            "star-R graphs live on the GPU, instead of using materialized "
            "caches. Required for dataset sizes beyond a few million: the star "
            "cache costs ~46.5 kB/jet, so 100M jets would need several TB per "
            "radius. Uses --star-radius as the live radius and expects "
            "--jetclass-raw-root (or --data-root) to be the raw ROOT tree."
        ),
    )
    parser.add_argument(
        "--jetclass-raw-root",
        type=str,
        default=None,
        help=(
            "Directory holding train_100M/ val_5M/ test_20M/ of raw JetClass "
            "ROOT files. Only used with --jetclass-stream. Default: --data-root."
        ),
    )
    parser.add_argument(
        "--jetclass-features",
        type=str,
        default="full",
        choices=("full", "kin", "kin7"),
        help=(
            "Particle input configuration: 'full' = 17 ParT features, "
            "'kin' = (delta_eta, delta_phi, log pT) only, 'kin7' = the 7 ParT "
            "kinematic features. Reproduces the input-feature ablation, which "
            "moves the asymptotic loss rather than the scaling exponent. "
            "Default: full."
        ),
    )
    parser.add_argument(
        "--jetclass-edge-features",
        type=str,
        default="logdot-dp",
        choices=("logdot-dp", "part-interaction"),
        help=(
            "Star-R hyperedge featurization against the support centroid. "
            "'logdot-dp' = Minkowski dot plus lab-frame momentum difference "
            "(3 of its 4 components are not rotation or boost invariant); "
            "'part-interaction' = the ParT set (ln Delta, ln k_T, ln z, ln m^2), "
            "all four invariant. Default: logdot-dp."
        ),
    )
    parser.add_argument(
        "--jetclass-centroid-weight",
        type=str,
        default=None,
        choices=("energy", "pt"),
        help=(
            "Weight for the star-R support centroid. Defaults to 'energy' for "
            "logdot-dp (matching existing caches) and 'pt' for part-interaction, "
            "where pT weighting is required for exact boost invariance."
        ),
    )
    parser.add_argument(
        "--jetclass-sort-by-pt",
        action="store_true",
        help=(
            "Re-sort constituents by decreasing pT before truncation. The "
            "published JetClass files already ship pT-sorted, so this is off by "
            "default; enable it for other sources such as Aspen Open Jets."
        ),
    )
    parser.add_argument(
        "--val-check-interval",
        type=float,
        default=None,
        help=(
            "Lightning val_check_interval: int = every N training steps, "
            "float in (0,1] = fraction of an epoch. Default: None (epoch-end "
            "only, via --check-val-every-n-epoch). Full val is expensive; "
            "consider --limit-val-batches with step-based checks."
        ),
    )
    parser.add_argument(
        "--stream-mask-seed",
        type=int,
        default=0,
        help=(
            "Deterministic seed for re-partitioning obvious stream stars "
            "across train/val/test (combined with each stream name). "
            "Default: 0."
        ),
    )
    parser.add_argument(
        "--no-repartition-obvious",
        action="store_true",
        help=(
            "Keep on-disk stream train/val/test masks as-is (skip equal split "
            "of obvious stream stars)."
        ),
    )
    parser.add_argument(
        "--include-real-streams",
        action="store_true",
        help=(
            "When --stream-name is all/mock, also load non-MOCK stream_*.pt "
            "graphs (e.g. Chenab, Orphan). Equivalent to --stream-name mock+real. "
            "Tags the run dir with streams-mock+real."
        ),
    )
    parser.add_argument(
        "--edge-loss-weight",
        type=float,
        default=1.0,
        help=(
            "Stream training: multiply edge aux in L = node_CE + w * L_edge. "
            "Applies to hard-ce or product-consistency. Default: 1.0."
        ),
    )
    parser.add_argument(
        "--edge-aux",
        type=str,
        default="hard-ce",
        choices=("hard-ce", "product-consistency", "hard+product"),
        help=(
            "Edge term in L = node_CE + w * L_edge. "
            "'hard-ce' = CE on edge_y; "
            "'product-consistency' = BCE(p_e, stopgrad(p_u p_v)) for 2-edges "
            "and BCE(p_e, stopgrad(mean p_i)) for hyperedges; "
            "'hard+product' = both. Tags run dirs with edgecons when consistency is on."
        ),
    )
    parser.add_argument(
        "--checkpoint-monitor",
        type=str,
        default=None,
        help=(
            "Metric name for best.ckpt (e.g. val_loss, val_auroc, val_edge_auroc). "
            "Default: val_auroc for product-consistency, else val_edge_auroc for "
            "streams, else val_loss."
        ),
    )
    parser.add_argument(
        "--checkpoint-mode",
        type=str,
        default=None,
        choices=("min", "max"),
        help=(
            "min/max for --checkpoint-monitor. Default: min for *loss*, else max."
        ),
    )
    parser.add_argument(
        "--identity-m22",
        action="store_true",
        help=(
            "CAPEN / CAPEN-Llama: skip edge–edge (relation 22) SupportAttention and "
            "use f_22 = identity on LayerNorm'd edge states. Same residual layout "
            "otherwise; tags run dirs with idm22. Mutually exclusive with "
            "--incidence-m22."
        ),
    )
    parser.add_argument(
        "--incidence-m22",
        action="store_true",
        help=(
            "CAPEN / CAPEN-Llama: replace edge–edge SupportAttention with "
            "f_22 = S S^T h_e (two incidence matmuls, no dense M×M attention). "
            "Tags run dirs with incm22. Mutually exclusive with --identity-m22."
        ),
    )
    parser.add_argument(
        "--hyperedge-only",
        action="store_true",
        help=(
            "Hierarchical caches: drop all pairwise 2-edges at train time "
            "(kNN, virtual-node links, and any incidence degree ≤ 2, including "
            "2-member DBSCAN) and compact the edge axis. True hyperedges remain "
            "(DBSCAN with ≥3 members + virtual hypers). Does not drop virtual "
            "nodes from X and does not rebuild the cache. Tags run dirs with nopair."
        ),
    )
    parser.add_argument(
        "--hier-k",
        type=int,
        default=None,
        help=(
            "Use hierarchical TopTagging caches (kNN+DBSCAN+virtual). Sets k for "
            "the cache tag hier_k{K}_eps…. Mutually exclusive with "
            "--graph-construction / --star-radius. Default companion flags: "
            "--hier-eps 0.08 --hier-min-samples 2 --hier-vn 1 --hier-ve 1 --hier-mdb 32."
        ),
    )
    parser.add_argument("--hier-eps", type=float, default=0.08, help="Hierarchical DBSCAN eps.")
    parser.add_argument(
        "--hier-min-samples",
        type=int,
        default=2,
        help="Hierarchical DBSCAN min_samples.",
    )
    parser.add_argument("--hier-vn", type=int, default=1, help="Virtual nodes in hier caches.")
    parser.add_argument("--hier-ve", type=int, default=1, help="Virtual hyperedges in hier caches.")
    parser.add_argument(
        "--hier-mdb",
        type=int,
        default=32,
        help="Max DBSCAN hyperedges per jet in hier caches.",
    )
    if not any(getattr(a, "dest", None) == "readout_mode" for a in parser._actions):
        parser.add_argument(
            "--readout-mode",
            type=str,
            default=None,
            choices=("graph", "node", "node+edge", "attn"),
            help=(
                "CPEN decoder: 'graph' pooled z_X+z_E, 'node' decode X only, "
                "'node+edge' decode X and E. Pascal defaults to node; "
                "node+edge enables boundary co-training (1[y_u != y_v])."
            ),
        )
    if not any(getattr(a, "dest", None) == "seed" for a in parser._actions):
        parser.add_argument(
            "--seed",
            type=int,
            default=0,
            help=(
                "Global RNG seed (Lightning seed_everything, including "
                "DataLoader workers). Pascal run dirs are tagged seed{N}."
            ),
        )
    return parser


def _hier_args_active(args) -> bool:  # type: ignore[no-untyped-def]
    if getattr(args, "hier_k", None) is not None:
        return True
    construction = str(getattr(args, "graph_construction", None) or "")
    return construction.startswith("hier_")


def configure_graph_args(args):  # type: ignore[no-untyped-def]
    """Validate graph CLI options; fold ``--knn-min-pt`` into graph_construction."""
    from cpen.utils.graph_hierarchical import hierarchical_construction_tag

    # Map CAPEN-Llama-att → capen-llama for bytecode graph/dropout guards.
    restore_model = None
    if getattr(args, "model", None) == _CAPEN_LLAMA_ATT:
        restore_model = _CAPEN_LLAMA_ATT
        args.model = "capen-llama"
    cli_readout = getattr(args, "readout_mode", None)

    if _hier_args_active(args):
        if getattr(args, "star_radius", None) is not None:
            raise ValueError("--hier-k / hier caches cannot combine with --star-radius")
        knn_like = getattr(args, "graph_construction", None)
        if knn_like is not None and not str(knn_like).startswith("hier_"):
            raise ValueError(
                "--hier-k cannot combine with --graph-construction "
                f"{knn_like!r} (omit --graph-construction for hierarchical caches)"
            )
        k = int(getattr(args, "hier_k") or 8)
        args.hier_k = k
        args.graph_construction = hierarchical_construction_tag(
            k=k,
            eps=float(getattr(args, "hier_eps", 0.08)),
            min_samples=int(getattr(args, "hier_min_samples", 2)),
            n_virtual_nodes=int(getattr(args, "hier_vn", 1)),
            n_virtual_edges=int(getattr(args, "hier_ve", 1)),
            max_dbscan_edges=int(getattr(args, "hier_mdb", 32)),
        )

    # Bytecode still errors: "--dropout is currently supported only with
    # --model capen or capen-llama". CPEN now uses dropout for MLP residuals,
    # so temporarily clear it for that check, then restore.
    dropout = float(getattr(args, "dropout", 0.0) or 0.0)
    bypass_dropout_guard = getattr(args, "model", None) == "cpen" and dropout > 0.0
    saved_dropout = None
    if bypass_dropout_guard:
        saved_dropout = args.dropout
        args.dropout = 0.0
    # Bytecode configure only knows toptagging / pascalvoc-sp / stream.
    restore_dataset = None
    if str(getattr(args, "dataset", "") or "") in {"mnist-sp", "mnist", "qm9"}:
        restore_dataset = args.dataset
        args.dataset = "pascalvoc-sp"
    restore_readout = None
    if str(getattr(args, "readout_mode", "") or "") == "attn":
        restore_readout = args.readout_mode
        args.readout_mode = "graph"
    try:
        _orig_configure_graph_args(args)
    finally:
        if bypass_dropout_guard:
            args.dropout = saved_dropout
        if restore_model is not None:
            args.model = restore_model
        if restore_dataset is not None:
            args.dataset = restore_dataset
        if restore_readout is not None:
            args.readout_mode = restore_readout

    # CLI wins. If omitted, force dataset defaults even when bytecode
    # already filled graph/node (Pascal must not silently stay graph).
    if cli_readout is not None:
        args.readout_mode = cli_readout
    else:
        dataset = str(getattr(args, "dataset", "") or "")
        if dataset == "pascalvoc-sp":
            args.readout_mode = "node"
        elif dataset in {"mnist-sp", "mnist"}:
            args.readout_mode = "node"
        elif dataset == "stream":
            args.readout_mode = "node+edge"
        elif dataset == "qm9":
            # Molecular property is graph-level; the Pascal alias above would
            # otherwise leave this on the per-node readout.
            args.readout_mode = "graph"
        elif getattr(args, "readout_mode", None) is None:
            args.readout_mode = "graph"

    min_pt = getattr(args, "knn_min_pt", None)
    construction = getattr(args, "graph_construction", None)
    star = getattr(args, "star_radius", None)
    if min_pt is not None and min_pt > 0.0:
        if star is not None:
            raise ValueError("--knn-min-pt is only valid with --graph-construction (kNN), not --star-radius")
        if _hier_args_active(args):
            raise ValueError("--knn-min-pt is not valid with hierarchical caches")
        if not construction:
            raise ValueError("--knn-min-pt requires --graph-construction (e.g. 8-NN)")
        from cpen.utils.graphs import compose_knn_spec, parse_knn_spec

        k, existing = parse_knn_spec(construction)
        if existing is not None and abs(existing - float(min_pt)) > 1e-12:
            raise ValueError(
                f"graph_construction already sets min_pt={existing:g} but "
                f"--knn-min-pt={min_pt:g}"
            )
        args.graph_construction = compose_knn_spec(k, float(min_pt))
    return args


def create_mnist_datamodule(args, *, data_root):  # type: ignore[no-untyped-def]
    """Cached MNISTSuperpixels incidence graphs (digit labels)."""
    from cpen.apps.mnist.mnist_datamodule import MNISTSuperpixelsDatamodule

    operators = _default_operators(args)
    include_incidence = (getattr(args, "model", None) == "cpen") or (operators == "adjacency")
    include_edge_features = getattr(args, "model", None) == "cpen"
    num_workers = args.num_workers if args.num_workers is not None else 0
    return MNISTSuperpixelsDatamodule(
        data_root=data_root,
        batch_size=int(getattr(args, "batch_size", 128) or 128),
        num_workers=num_workers,
        n_train=getattr(args, "n_train", None),
        n_val=getattr(args, "n_val", None),
        n_test=getattr(args, "n_test", None),
        data_seed=int(getattr(args, "data_seed", 42) or 42),
        include_incidence=include_incidence,
        include_edge_features=include_edge_features,
    )


def create_qm9_datamodule(args, *, data_root):  # type: ignore[no-untyped-def]
    """Cached QM9 bond graphs (graph-level scalar regression targets)."""
    from cpen.apps.qm9.qm9_datamodule import QM9Datamodule

    operators = _default_operators(args)
    include_incidence = (getattr(args, "model", None) == "cpen") or (operators == "adjacency")
    include_edge_features = getattr(args, "model", None) == "cpen"
    num_workers = args.num_workers if args.num_workers is not None else 0
    return QM9Datamodule(
        data_root=data_root,
        batch_size=int(getattr(args, "batch_size", 128) or 128),
        num_workers=num_workers,
        target=getattr(args, "qm9_target", None),
        n_train=getattr(args, "n_train", None),
        n_val=getattr(args, "n_val", None),
        n_test=getattr(args, "n_test", None),
        data_seed=int(getattr(args, "data_seed", 42) or 42),
        include_incidence=include_incidence,
        include_edge_features=include_edge_features,
    )


def jetclass_stream_active(args) -> bool:  # type: ignore[no-untyped-def]
    """Whether this run reads raw JetClass ROOT instead of star-$R$ caches."""
    return str(getattr(args, "dataset", "") or "") == "jetclass" and bool(
        getattr(args, "jetclass_stream", False)
    )


def jetclass_live_graph_kwargs(args) -> dict:  # type: ignore[no-untyped-def]
    """Live star-$R$ settings for the Lightning module (streaming runs only)."""
    if not jetclass_stream_active(args):
        return {}
    radius = getattr(args, "star_radius", None)
    if radius is None:
        raise ValueError("--jetclass-stream requires --star-radius for live graphs")
    return {
        "live_star_radius": float(radius),
        "live_edge_features": str(
            getattr(args, "jetclass_edge_features", "logdot-dp") or "logdot-dp"
        ),
        "live_centroid_weight": getattr(args, "jetclass_centroid_weight", None),
    }


def create_jetclass_datamodule(args, *, data_root):  # type: ignore[no-untyped-def]
    """Star-$R$ caches, or raw ROOT streaming under ``--jetclass-stream``."""
    if not jetclass_stream_active(args):
        return _orig_create_jetclass_datamodule(args, data_root=data_root)

    from cpen.apps.jets.jetclass_stream_datamodule import JetClassStreamDatamodule

    raw_root = getattr(args, "jetclass_raw_root", None) or data_root
    num_workers = args.num_workers if args.num_workers is not None else 0
    return JetClassStreamDatamodule(
        data_root=raw_root,
        batch_size=int(getattr(args, "batch_size", 128) or 128),
        num_workers=num_workers,
        num_particles=int(getattr(args, "num_particles", 128) or 128),
        feature_config=str(getattr(args, "jetclass_features", "full") or "full"),
        n_train=getattr(args, "n_train", None),
        n_val=getattr(args, "n_val", None),
        n_test=getattr(args, "n_test", None),
        shuffle_seed=int(getattr(args, "seed", 0) or 0),
        sort_by_pt=bool(getattr(args, "jetclass_sort_by_pt", False)),
    )


def create_stream_datamodule(args, data_root):  # type: ignore[no-untyped-def]
    """Like bytecode helper, with obvious-stream mask re-partition options."""
    from cpen.datamodules.stream_datamodule import StreamDatamodule

    operators = _default_operators(args)
    include_incidence = (getattr(args, "model", None) == "cpen") or (operators == "adjacency")
    include_edge_features = getattr(args, "model", None) == "cpen"
    num_workers = args.num_workers if args.num_workers is not None else 0
    return StreamDatamodule(
        data_root=data_root,
        batch_size=1,
        stream_name=getattr(args, "stream_name", "all"),
        include_real_streams=bool(getattr(args, "include_real_streams", False)),
        num_workers=num_workers,
        include_incidence=include_incidence,
        include_edge_features=include_edge_features,
        repartition_obvious=not bool(getattr(args, "no_repartition_obvious", False)),
        mask_seed=int(getattr(args, "stream_mask_seed", 0) or 0),
    )


def build_model(args, *, n_features, n_edge_features, out_dim, depth, width, heads=1):  # type: ignore[no-untyped-def]
    """Like bytecode ``build_model``, but forwards ``--dropout`` into CPEN MLPs.

    CAPEN / CAPEN-Llama are intentionally untouched here: they still go through
    ``_orig_build_model``, where ``dropout`` is attention-only
    (``SupportAttention``), not MLP dropout. ``capen-llama-att`` is built here.
    """
    if getattr(args, "model", None) == "cpen":
        from cpen.models.cpen import CPEN

        dataset = str(getattr(args, "dataset", "") or "")
        cli_readout = str(getattr(args, "readout_mode", "graph") or "graph")
        # MNIST / QM9 mean-pool arms use graph decoders; attn keeps readout_mode=attn.
        if dataset in {"mnist-sp", "mnist", "qm9"} and cli_readout != "attn":
            construct_readout = "graph"
        else:
            construct_readout = cli_readout
        extra = {}
        if construct_readout == "attn":
            extra["output_attention_heads"] = getattr(args, "heads", None)

        return CPEN(
            n_features=n_features,
            n_edge_features=n_edge_features,
            out_dim=out_dim,
            depth=depth,
            width=width,
            operators=args.operators,
            normalization=args.normalization,
            energy_index=args.energy_index,
            optimizer=args.optimizer,
            residual_structure=getattr(args, "residual_structure", "standard"),
            operator_normalization=getattr(args, "operator_normalization", "degree"),
            operator_backend=getattr(args, "operator_backend", "dense"),
            layer_norm=bool(getattr(args, "layer_norm", False)),
            gamma_rs=getattr(args, "gamma_rs", None),
            readout_mode=construct_readout,
            dropout=float(getattr(args, "dropout", 0.0) or 0.0),
            **extra,
        )
    if getattr(args, "model", None) == _CAPEN_LLAMA_ATT:
        from cpen.models.capen_llama_att import CAPENLlamaAtt

        attention_gammas = _attention_gammas_from_args(args)
        dataset = getattr(args, "dataset", "toptagging")
        readout_mode = getattr(args, "readout_mode", "graph")
        all_to_all = (dataset != "pascalvoc-sp") and (readout_mode != "node")
        model = CAPENLlamaAtt(
            n_features=n_features,
            n_edge_features=n_edge_features,
            out_dim=out_dim,
            depth=depth,
            width=width,
            heads=heads,
            dropout=float(getattr(args, "dropout", 0.0) or 0.0),
            normalization=args.normalization,
            energy_index=args.energy_index,
            optimizer=args.optimizer,
            readout_mode=readout_mode,
            attention_normalization=getattr(args, "attention_normalization", "none"),
            attention_gammas=attention_gammas,
            use_wire=bool(getattr(args, "use_wire", False)),
            wire_coordinate_dim=int(getattr(args, "wire_coordinate_dim", 8)),
            wire_share_across_heads=bool(getattr(args, "wire_share_across_heads", False)),
            wire_frequency_init_std=getattr(args, "wire_frequency_init_std", None),
            wire_sign_augmentation=bool(getattr(args, "wire_sign_augmentation", False)),
            wire_normalized_laplacian=not bool(
                getattr(args, "wire_combinatorial_laplacian", False)
            ),
            all_to_all_particle_attention=all_to_all,
            identity_m22=bool(getattr(args, "identity_m22", False)),
            incidence_m22=bool(getattr(args, "incidence_m22", False)),
            hyperedge_m22_only=True,
            ignore_knn_edges=bool(getattr(args, "hyperedge_only", False)),
            n_class_tokens=int(out_dim),
        )
        return model
    # capen / capen-llama / particle-only: original factory (attention dropout only for CAPEN*).
    model = _orig_build_model(
        args,
        n_features=n_features,
        n_edge_features=n_edge_features,
        out_dim=out_dim,
        depth=depth,
        width=width,
        heads=heads,
    )
    if bool(getattr(args, "identity_m22", False)) and bool(
        getattr(args, "incidence_m22", False)
    ):
        raise ValueError("--identity-m22 and --incidence-m22 cannot both be set")
    if bool(getattr(args, "identity_m22", False)):
        if not hasattr(model, "identity_m22"):
            raise TypeError(
                "--identity-m22 requires CAPEN / CAPEN-Llama "
                f"(got {type(model).__name__})"
            )
        model.identity_m22 = True
    if bool(getattr(args, "incidence_m22", False)):
        if not hasattr(model, "incidence_m22"):
            raise TypeError(
                "--incidence-m22 requires CAPEN / CAPEN-Llama "
                f"(got {type(model).__name__})"
            )
        model.incidence_m22 = True
    if bool(getattr(args, "hyperedge_only", False)):
        model.ignore_knn_edges = True
    return model


def create_toptagging_datamodule(args, *, data_root, dense_pairs):  # type: ignore[no-untyped-def]
    """Bytecode helper plus hierarchical cache branch (``--hier-k``)."""
    if _hier_args_active(args):
        from cpen.datamodules.toptagging_hier_datamodule import TopTaggingHierDatamodule

        num_workers = args.num_workers if args.num_workers is not None else 0
        return TopTaggingHierDatamodule(
            data_root=data_root,
            batch_size=args.batch_size,
            k=int(getattr(args, "hier_k") or 8),
            eps=float(getattr(args, "hier_eps", 0.08)),
            min_samples=int(getattr(args, "hier_min_samples", 2)),
            n_virtual_nodes=int(getattr(args, "hier_vn", 1)),
            n_virtual_edges=int(getattr(args, "hier_ve", 1)),
            max_dbscan_edges=int(getattr(args, "hier_mdb", 32)),
            num_workers=num_workers,
            num_particles=int(getattr(args, "num_particles", 128)),
            n_train=getattr(args, "n_train", None),
            n_val=getattr(args, "n_val", None),
            n_test=getattr(args, "n_test", None),
            data_seed=int(getattr(args, "data_seed", 42) or 42),
        )
    return _orig_create_toptagging_datamodule(
        args, data_root=data_root, dense_pairs=dense_pairs
    )


def _attention_all_to_all_m11(args) -> bool:  # type: ignore[no-untyped-def]
    model = str(getattr(args, "model", "") or "")
    if model not in ("capen-llama", _CAPEN_LLAMA_ATT):
        return False
    dataset = getattr(args, "dataset", "toptagging")
    readout_mode = getattr(args, "readout_mode", "graph")
    return (dataset != "pascalvoc-sp") and (readout_mode != "node")


def _attention_gammas_from_args(args):  # type: ignore[no-untyped-def]
    """Build ``AttentionGammas`` for CAPEN-Llama-att without ``float(None)``.

    Bytecode ``train_one_run`` only calls ``resolve_attention_gammas`` for
    ``capen`` / ``capen-llama``, so ``capen-llama-att`` arrives here with CLI
    defaults ``gamma_*=None``. Estimate (or require explicit ``--gamma-*``)
    before constructing the dataclass.
    """
    if getattr(args, "attention_normalization", "none") != "gamma":
        return None
    from cpen.models.capen import AttentionGammas

    existing = getattr(args, "attention_gammas", None)
    if existing is not None:
        return existing
    values = tuple(getattr(args, name, None) for name in ("gamma_11", "gamma_12", "gamma_21", "gamma_22"))
    if all(v is not None for v in values):
        return AttentionGammas(
            gamma_11=float(values[0]),
            gamma_12=float(values[1]),
            gamma_21=float(values[2]),
            gamma_22=float(values[3]),
        )
    data_root = getattr(args, "data_root", None) or getattr(args, "root", None)
    gammas = resolve_attention_gammas(args, data_root)
    if gammas is None:
        raise ValueError(
            "attention_normalization='gamma' requires --estimate-gamma-rs "
            "with a graph cache, or explicit --gamma-11/12/21/22"
        )
    return gammas


def _apply_attention_gammas_to_args(args, gammas) -> None:  # type: ignore[no-untyped-def]
    args.gamma_11 = float(gammas.gamma_11)
    args.gamma_12 = float(gammas.gamma_12)
    args.gamma_21 = float(gammas.gamma_21)
    args.gamma_22 = float(gammas.gamma_22)
    args.gamma_estimated = True
    args.attention_gammas = gammas


def _log_attention_gammas(gammas, summaries, *, extra: str = "") -> None:  # type: ignore[no-untyped-def]
    print(
        "[attn-gamma] mean support row degrees: "
        f"gamma_11={gammas.gamma_11:.4f} "
        f"gamma_12={gammas.gamma_12:.4f} "
        f"gamma_21={gammas.gamma_21:.4f} "
        f"gamma_22={gammas.gamma_22:.4f}"
        + (f"  ({extra})" if extra else ""),
        flush=True,
    )
    if not summaries:
        return
    for rel in ("11", "12", "21", "22"):
        stats = summaries.get(rel) or {}
        print(
            f"[attn-gamma] M_{rel}: n_rows={int(stats.get('n_rows', 0)):,}  "
            f"mean={float(stats.get('mean', 0)):.3f}  "
            f"median={float(stats.get('median', 0)):.3f}  "
            f"std={float(stats.get('std', 0)):.3f}  "
            f"min={float(stats.get('min', 0)):.0f}  "
            f"max={float(stats.get('max', 0)):.0f}",
            flush=True,
        )


def resolve_attention_gammas(args, data_root):  # type: ignore[no-untyped-def]
    """Estimate CAPEN attention γ from the train cache unless opted out.

    Hierarchical + CAPEN-Llama-att uses all-to-all ``M_{11}``, incidence
    ``M_{12}/M_{21}``, hyperedge-only ``M_{22}``, and drops pairwise 2-edges
    when ``--hyperedge-only``. Values are written back onto ``args`` so the
    Slurm banner and parquet metadata record what was actually used.
    """
    if getattr(args, "attention_normalization", "none") != "gamma":
        return None
    if getattr(args, "gamma_estimated", False) and getattr(args, "gamma_11", None) is not None:
        from cpen.models.capen import AttentionGammas

        return AttentionGammas(
            gamma_11=float(args.gamma_11),
            gamma_12=float(args.gamma_12),
            gamma_21=float(args.gamma_21),
            gamma_22=float(args.gamma_22),
        )
    estimate = bool(getattr(args, "estimate_gamma_rs", True))
    if not estimate:
        return _orig_resolve_attention_gammas(args, data_root=data_root)

    from cpen.utils.attention_temperature import (
        estimate_attention_gammas_from_cache,
        estimate_attention_gammas_from_hier_cache,
    )

    all_to_all = _attention_all_to_all_m11(args)
    hyperedge_m22 = str(getattr(args, "model", "") or "") == _CAPEN_LLAMA_ATT
    drop_pairwise = bool(getattr(args, "hyperedge_only", False))
    n_jets = int(getattr(args, "gamma_estimate_jets", 512) or 512)
    seed = int(getattr(args, "data_seed", 42) or 42)
    num_particles = int(getattr(args, "num_particles", 128))
    summaries = None
    extra = (
        f"n_jets={n_jets} all_to_all_M11={all_to_all} "
        f"hyperedge_M22={hyperedge_m22} drop_pairwise={drop_pairwise}"
    )
    try:
        if _hier_args_active(args):
            print(f"[attn-gamma] estimating from hierarchical cache  {extra}", flush=True)
            gammas, summaries = estimate_attention_gammas_from_hier_cache(
                data_root=data_root,
                split="train",
                k=int(getattr(args, "hier_k") or 8),
                eps=float(getattr(args, "hier_eps", 0.08)),
                min_samples=int(getattr(args, "hier_min_samples", 2)),
                n_virtual_nodes=int(getattr(args, "hier_vn", 1)),
                n_virtual_edges=int(getattr(args, "hier_ve", 1)),
                max_dbscan_edges=int(getattr(args, "hier_mdb", 32)),
                num_particles=num_particles,
                n_jets=n_jets,
                seed=seed,
                all_to_all_particle_attention=all_to_all,
                hyperedge_m22_only=hyperedge_m22,
                drop_pairwise_edges=drop_pairwise,
            )
        else:
            print(f"[attn-gamma] estimating from star/kNN cache  {extra}", flush=True)
            radius = getattr(args, "star_radius", None)
            result = estimate_attention_gammas_from_cache(
                data_root=data_root,
                split="train",
                num_particles=num_particles,
                n_jets=n_jets,
                seed=seed,
                radius=radius,
                graph_construction=None if radius is not None else getattr(args, "graph_construction", None),
                all_to_all_particle_attention=all_to_all,
            )
            if isinstance(result, tuple):
                gammas, summaries = result
            else:
                gammas, summaries = result, None
    except FileNotFoundError as exc:
        has_cli = all(
            getattr(args, name, None) is not None
            for name in ("gamma_11", "gamma_12", "gamma_21", "gamma_22")
        )
        if not has_cli:
            raise
        print(
            f"[attn-gamma] cache estimate failed ({exc}); using CLI --gamma-*",
            flush=True,
        )
        args.gamma_estimated = False
        return _orig_resolve_attention_gammas(args, data_root=data_root)

    _apply_attention_gammas_to_args(args, gammas)
    _log_attention_gammas(gammas, summaries, extra=extra)
    return gammas


def _resolve_checkpoint_settings(args) -> tuple[str, str]:  # type: ignore[no-untyped-def]
    monitor = getattr(args, "checkpoint_monitor", None)
    mode = getattr(args, "checkpoint_mode", None)
    if not monitor:
        if getattr(args, "dataset", None) == "stream":
            edge_aux = str(getattr(args, "edge_aux", "hard-ce") or "hard-ce").lower()
            # Product consistency is an aux for nodes → checkpoint on node discovery.
            if "product" in edge_aux or "cons" in edge_aux:
                monitor = "val_auroc"
            else:
                monitor = "val_edge_auroc"
            mode = mode or "max"
        elif getattr(args, "dataset", None) == "pascalvoc-sp":
            monitor = "val_f1"
            mode = mode or "max"
        elif getattr(args, "dataset", None) in {"mnist-sp", "mnist"}:
            monitor = "val_acc"
            mode = mode or "max"
        elif getattr(args, "dataset", None) == "qm9":
            monitor = "val_mae"
            mode = mode or "min"
        else:
            monitor = "val_loss"
            mode = mode or "min"
    if not mode:
        mode = "min" if "loss" in str(monitor).lower() else "max"
    return str(monitor), str(mode)


def build_trainer(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Pass optional val_check_interval / checkpoint monitor from train_one_run ctx."""
    if "val_check_interval" not in kwargs:
        vci = _val_check_interval_ctx.get()
        if vci is not None:
            kwargs["val_check_interval"] = vci
    if "checkpoint_monitor" not in kwargs:
        monitor = _checkpoint_monitor_ctx.get()
        if monitor is not None:
            kwargs["checkpoint_monitor"] = monitor
    if "checkpoint_mode" not in kwargs:
        mode = _checkpoint_mode_ctx.get()
        if mode is not None:
            kwargs["checkpoint_mode"] = mode
    return _orig_build_trainer(*args, **kwargs)


def train_one_run(*, args, **kwargs):  # type: ignore[no-untyped-def]
    """Like bytecode ``train_one_run``, with stream checkpoint / val-interval support."""
    import lightning as L

    seed = int(getattr(args, "seed", 0) or 0)
    L.seed_everything(seed, workers=True)
    raw = getattr(args, "val_check_interval", None)
    if raw is None:
        vci = None
    else:
        vci = int(raw) if float(raw).is_integer() and float(raw) >= 1.0 else float(raw)
    monitor, mode = _resolve_checkpoint_settings(args)
    token_vci = _val_check_interval_ctx.set(vci)
    token_mon = _checkpoint_monitor_ctx.set(monitor)
    token_mode = _checkpoint_mode_ctx.set(mode)
    # Bytecode only resolves attention γ for capen / capen-llama. Fill args
    # before build_model so capen-llama-att does not float(None).
    if (
        getattr(args, "model", None) == _CAPEN_LLAMA_ATT
        and getattr(args, "attention_normalization", "none") == "gamma"
        and getattr(args, "gamma_11", None) is None
    ):
        data_root = getattr(args, "data_root", None) or getattr(args, "root", None)
        resolve_attention_gammas(args, data_root)
    # Bytecode train_one_run only forwards live_graph_k / live_dense_pairs, so
    # live star-R settings have to ride in on the lit_factory.
    live_kwargs = jetclass_live_graph_kwargs(args)
    if live_kwargs and "lit_factory" in kwargs:
        base_factory = kwargs["lit_factory"]

        def lit_factory(*factory_args, **factory_kwargs):
            factory_kwargs.update(live_kwargs)
            return base_factory(*factory_args, **factory_kwargs)

        kwargs["lit_factory"] = lit_factory
    try:
        return _orig_train_one_run(args=args, **kwargs)
    finally:
        _val_check_interval_ctx.reset(token_vci)
        _checkpoint_monitor_ctx.reset(token_mon)
        _checkpoint_mode_ctx.reset(token_mode)


# Ensure in-module callers see the wrapped helpers.
_bc.run_options_from_args = run_options_from_args
_bc.add_common_args = add_common_args
_bc.configure_graph_args = configure_graph_args
_bc.build_trainer = build_trainer
_bc.train_one_run = train_one_run
_bc.resolve_attention_gammas = resolve_attention_gammas
_bc.build_model = build_model
_bc.create_stream_datamodule = create_stream_datamodule
_bc.create_toptagging_datamodule = create_toptagging_datamodule
_bc.create_mnist_datamodule = create_mnist_datamodule
_bc.create_qm9_datamodule = create_qm9_datamodule
_bc.create_jetclass_datamodule = create_jetclass_datamodule
globals()["run_options_from_args"] = run_options_from_args
globals()["add_common_args"] = add_common_args
globals()["configure_graph_args"] = configure_graph_args
globals()["build_trainer"] = build_trainer
globals()["train_one_run"] = train_one_run
globals()["resolve_attention_gammas"] = resolve_attention_gammas
globals()["build_model"] = build_model
globals()["create_stream_datamodule"] = create_stream_datamodule
globals()["create_toptagging_datamodule"] = create_toptagging_datamodule
globals()["create_mnist_datamodule"] = create_mnist_datamodule
globals()["create_qm9_datamodule"] = create_qm9_datamodule
globals()["create_jetclass_datamodule"] = create_jetclass_datamodule

# registry.pyc / sweep bytecode bind model classes at import time; keep in sync.
try:
    import cpen.models.registry as _registry
    from cpen.models.capen import CAPEN as _CAPEN
    from cpen.models.cpen import CPEN as _CPEN

    _registry.build_model = build_model
    _registry.CPEN = _CPEN
    _registry.CAPEN = _CAPEN
except Exception:
    pass

try:
    from cpen.models.capen import CAPEN as _CAPEN_WRAP
    from cpen.models.capen_llama import CAPENLlama as _CAPEN_LLAMA_WRAP

    _orig_build_model.__globals__["CAPEN"] = _CAPEN_WRAP
    _orig_build_model.__globals__["CAPENLlama"] = _CAPEN_LLAMA_WRAP
    _orig_train_one_run.__globals__["resolve_attention_gammas"] = resolve_attention_gammas
except Exception:
    pass
