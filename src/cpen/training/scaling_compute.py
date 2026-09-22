"""Measured training FLOPs for JetClass CAPEN-att scaling-law plots.

Iso-compute axes should use **net + incidence** FLOPs after the star-$R$ graph
already exists, then multiply by optimizer steps:

    C(S) = flops_per_optimizer_step * S

``flops_per_optimizer_step`` is forward+backward on one global batch, i.e.
``n_gpus * (fwd+bwd FLOPs at per-GPU batch)``. Graph construction is measured
separately and is **not** included in C(S).

Kaplan's ``6 * P * n_p * n`` is printed as a reference only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from cpen.models.capen_llama_att import CAPENLlamaAtt
from cpen.training.base_lit_cpen import BaseLitCPEN


@dataclass(frozen=True)
class FlopReport:
    depth: int
    width: int
    heads: int
    n_params: int
    num_particles: int
    per_gpu_batch: int
    n_gpus: int
    global_batch: int
    flops_graph_per_gpu: float
    flops_train_per_gpu: float  # fwd+bwd, graphs already built
    flops_per_optimizer_step: float
    kaplan_6Pnp_per_jet: float
    device: str

    @property
    def flops_per_jet_train(self) -> float:
        return self.flops_train_per_gpu / float(self.per_gpu_batch)

    # Star-R construction is mostly elementwise / indexing. FlopCounterMode
    # therefore often reports 0 graph FLOPs even though wall-clock is 1–8% of a
    # training step. That is expected: do not put construction in C(S).

    def C(self, steps: int) -> float:
        """Total training FLOPs for a unique-pass run of *steps* optimizer steps."""
        return float(self.flops_per_optimizer_step) * float(steps)


def _synthetic_four_vectors(
    batch: int, n_particles: int, *, occupancy: float = 0.5, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random massive 4-vectors with a JetClass-like occupancy (~40/80)."""
    g = torch.Generator().manual_seed(seed)
    n_live = max(1, int(round(occupancy * n_particles)))
    px = torch.randn(batch, n_particles, generator=g)
    py = torch.randn(batch, n_particles, generator=g)
    pz = torch.randn(batch, n_particles, generator=g)
    mass = 0.1 + 0.2 * torch.rand(batch, n_particles, generator=g)
    energy = torch.sqrt(px * px + py * py + pz * pz + mass * mass)
    x_raw = torch.stack([energy, px, py, pz], dim=-1)
    mask = torch.zeros(batch, n_particles, dtype=torch.bool)
    mask[:, :n_live] = True
    x_raw = x_raw * mask.unsqueeze(-1)
    return x_raw, mask


def build_lit(
    *,
    depth: int,
    width: int,
    heads: int,
    n_features: int = 7,
    n_edge_features: int = 4,
    out_dim: int = 10,
    star_radius: float = 0.2,
    edge_features: str = "part-interaction",
) -> BaseLitCPEN:
    model = CAPENLlamaAtt(
        n_features=n_features,
        n_edge_features=n_edge_features,
        out_dim=out_dim,
        depth=depth,
        width=width,
        heads=heads,
        dropout=0.0,
        normalization="uniform",
        energy_index=0,
        optimizer="adam",
        readout_mode="graph",
        attention_normalization="none",
        all_to_all_particle_attention=True,
        hyperedge_m22_only=True,
        n_class_tokens=out_dim,
    )
    return BaseLitCPEN(
        model,
        eta_0=0.25,
        model_name="capen-llama-att",
        optimizer="adam",
        dataset="jetclass",
        depth=depth,
        width=width,
        heads=heads,
        live_star_radius=star_radius,
        live_edge_features=edge_features,
    )


def _raw_batch(
    lit: BaseLitCPEN,
    *,
    batch_size: int,
    num_particles: int,
    n_features: int = 7,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    x_raw, mask = _synthetic_four_vectors(batch_size, num_particles, seed=seed)
    g = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(batch_size, num_particles, n_features, generator=g) * mask.unsqueeze(-1)
    y = torch.randint(0, int(lit.model.out_dim), (batch_size,), generator=g)
    return {"x": x, "y": y, "x_raw": x_raw, "mask": mask}


def _total_flops(flop_counter) -> float:
    if hasattr(flop_counter, "get_total_flops"):
        return float(flop_counter.get_total_flops())
    counts = flop_counter.get_flop_counts()
    return float(sum(sum(v.values()) for v in counts.values()))


def measure_flops(
    *,
    depth: int,
    width: int,
    heads: int,
    per_gpu_batch: int = 8,
    n_gpus: int = 4,
    num_particles: int = 80,
    device: str | torch.device | None = None,
) -> FlopReport:
    """One-shot FLOP count. Use a small *per_gpu_batch* then scale linearly in B."""
    from torch.utils.flop_counter import FlopCounterMode

    if device is None:
        # Counting does not need a GPU; CUDA init on a login node can hang.
        device = torch.device("cpu")
    else:
        device = torch.device(device)

    lit = build_lit(depth=depth, width=width, heads=heads)
    lit.eval()
    lit.to(device)
    raw = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in _raw_batch(lit, batch_size=per_gpu_batch, num_particles=num_particles).items()
    }

    with FlopCounterMode(display=False) as fc_graph:
        batch = lit.on_after_batch_transfer({k: v.clone() if torch.is_tensor(v) else v for k, v in raw.items()}, 0)
    flops_graph = _total_flops(fc_graph)

    lit.train()
    # Count a training step on an already-built batch (no construction).
    batch = {
        k: v.detach() if torch.is_tensor(v) else v for k, v in batch.items()
    }
    for v in batch.values():
        if torch.is_tensor(v) and v.is_floating_point():
            v.requires_grad_(False)

    with FlopCounterMode(display=False) as fc_train:
        x, y = lit._unpack_batch(batch)
        logits = lit.forward(x, **lit._model_kwargs(batch))
        loss = F.cross_entropy(logits, y)
        loss.backward()
    flops_train = _total_flops(fc_train)

    n_params = sum(p.numel() for p in lit.parameters() if p.requires_grad)
    return FlopReport(
        depth=depth,
        width=width,
        heads=heads,
        n_params=n_params,
        num_particles=num_particles,
        per_gpu_batch=per_gpu_batch,
        n_gpus=n_gpus,
        global_batch=per_gpu_batch * n_gpus,
        flops_graph_per_gpu=flops_graph,
        flops_train_per_gpu=flops_train,
        flops_per_optimizer_step=flops_train * n_gpus,
        kaplan_6Pnp_per_jet=6.0 * n_params * num_particles,
        device=str(device),
    )


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4, help="per-GPU batch for the counter")
    p.add_argument("--n-gpus", type=int, default=4)
    p.add_argument("--num-particles", type=int, default=80)
    p.add_argument("--device", default=None)
    p.add_argument("--steps", type=int, default=25000, help="print C at this unique-pass S")
    args = p.parse_args(argv)

    report = measure_flops(
        depth=args.depth,
        width=args.width,
        heads=args.heads,
        per_gpu_batch=args.batch_size,
        n_gpus=args.n_gpus,
        num_particles=args.num_particles,
        device=args.device,
    )
    payload = asdict(report)
    payload["flops_per_jet_train"] = report.flops_per_jet_train
    payload[f"C_at_S{args.steps}"] = report.C(args.steps)
    payload["graph_frac_of_gpu_step"] = report.flops_graph_per_gpu / max(
        report.flops_graph_per_gpu + report.flops_train_per_gpu, 1.0
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
