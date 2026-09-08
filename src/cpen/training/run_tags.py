"""Format eta_0 and other hyperparameters for deterministic output filenames."""

from __future__ import annotations

from dataclasses import dataclass


def format_eta(eta_0: float) -> str:
    """Convert eta_0 to filename token, e.g. 0.3 -> 0p3."""
    text = f"{eta_0:g}"
    return text.replace(".", "p")


def format_t_epoch(t_epoch: float) -> str:
    text = f"{t_epoch:g}"
    return text.replace(".", "p")


def format_count(n: int) -> str:
    """Compact integer token for filenames, e.g. 50000 -> 50k."""
    if n >= 1_000_000 and n % 1_000_000 == 0:
        return f"{n // 1_000_000}M"
    if n >= 1_000 and n % 1_000 == 0:
        return f"{n // 1_000}k"
    return str(n)


@dataclass
class RunOptions:
    """Non-default architecture and training flags that affect output names."""

    layer_norm: bool = False
    sigma_output: str | None = None  # e.g. "Lp1-width" for alternate output scaling
    encoder: str = "linear"
    decoder: str = "linear"
    scheduler: str = "none"
    heads: int | None = None
    dropout: float = 0.0
    operators: str = "identity"
    normalization: str = "uniform"
    residual_structure: str = "standard"
    alpha: float | None = None
    gamma: float | None = None
    n_train: int | None = None
    n_val: int | None = None
    n_test: int | None = None
    graph_construction: str | None = None
    star_radius: float | None = None
    n_features: int | None = None
    use_wire: bool = False
    wire_coordinate_dim: int = 8
    # CPEN / particle-only incidence scaling. Always emitted when set so
    # ``degree`` and ``gamma`` runs never share an output directory.
    operator_normalization: str | None = None
    # Optional free-form tag (e.g. streams-mock+real) appended to the run name.
    extra_tag: str | None = None

    def suffix(self) -> str:
        parts: list[str] = []
        if self.layer_norm:
            parts.append("ln")
        if self.sigma_output is not None:
            parts.append(f"sigma-{self.sigma_output}")
        if self.encoder != "linear":
            parts.append(f"{self.encoder.replace('_', '-')}-enc")
        if self.decoder != "linear":
            parts.append(f"{self.decoder.replace('_', '-')}-dec")
        if self.scheduler == "cosine":
            parts.append("cosine")
        if self.operators != "identity":
            op_token = self.operators.replace(",", "-").replace("_", "-")
            parts.append(f"op-{op_token}")
        if self.operator_normalization is not None:
            parts.append(f"onorm-{self.operator_normalization.replace('_', '-')}")
        if self.normalization != "uniform":
            norm_token = self.normalization.replace("_", "-")
            parts.append(f"norm-{norm_token}")
        if self.residual_structure != "standard":
            parts.append(f"res-{self.residual_structure.replace('_', '-')}")
        if self.heads is not None:
            parts.append(f"h{self.heads}")
        if self.dropout > 0.0:
            parts.append(f"drop{format_eta(self.dropout)}")
        if self.alpha is not None:
            alpha_token = f"{self.alpha:g}".replace(".", "p")
            parts.append(f"aa{alpha_token}")
        if self.gamma is not None:
            gamma_token = f"{self.gamma:g}".replace(".", "p")
            parts.append(f"gamma{gamma_token}")
        if self.n_train is not None:
            parts.append(f"ntr{format_count(self.n_train)}")
        if self.n_val is not None:
            parts.append(f"nval{format_count(self.n_val)}")
        if self.n_test is not None:
            parts.append(f"ntst{format_count(self.n_test)}")
        if self.n_features is not None and self.n_features != 4:
            parts.append(f"nf{self.n_features}")
        if self.star_radius is not None:
            parts.append(f"gstar{format_eta(self.star_radius)}")
        elif self.graph_construction is not None:
            gtag = (
                self.graph_construction.replace("-", "")
                .replace("@", "_")
                .lower()
            )
            parts.append(f"g{gtag}")
        if self.use_wire:
            parts.append(f"wirem{self.wire_coordinate_dim}")
        if self.extra_tag:
            parts.append(self.extra_tag.replace("_", "-"))
        if not parts:
            return ""
        return "_" + "_".join(parts)


def run_basename(
    model: str,
    depth: int,
    width: int,
    eta_0: float,
    *,
    t_epoch: float | None = None,
    options: RunOptions | None = None,
) -> str:
    """Build deterministic run basename without extension."""
    name = f"{model}_{depth}_{width}_{format_eta(eta_0)}"
    if t_epoch is not None:
        name += f"_t{format_t_epoch(t_epoch)}"
    if options is not None:
        name += options.suffix()
    return name
