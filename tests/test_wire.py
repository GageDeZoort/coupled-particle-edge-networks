"""Unit tests for Wave-Induced Rotary Encodings (WIRE)."""

from __future__ import annotations

import math
import warnings

import pytest
import torch

from cpen.models.capen import SupportAttention
from cpen.models.capen_llama import CAPENLlama
from cpen.models.wire import WIRERotaryEmbedding, apply_paired_rotation
from cpen.utils.wire_coordinates import (
    augment_wire_coordinate_signs,
    compute_wire_coordinates,
    compute_wire_coordinates_batched,
    particle_adjacency_from_incidence,
)


def _rotate_by_relative(
    q: torch.Tensor,
    k: torch.Tensor,
    omega: torch.Tensor,
    r_i: torch.Tensor,
    r_j: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply R(r_j - r_i) to k in the WIRE pairing convention (for identity checks)."""
    delta = (r_j - r_i).float()
    # omega: (H, A, m); theta: (H, A)
    theta = torch.einsum("m,ham->ha", delta, omega.float())
    cos, sin = theta.cos(), theta.sin()
    # Broadcast over leading dims of k if present.
    while cos.dim() < k.dim() - 1:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    return q, apply_paired_rotation(k.float(), cos, sin).to(dtype=k.dtype)


# ---------------------------------------------------------------------------
# Rotation correctness
# ---------------------------------------------------------------------------


def test_wire_output_shapes() -> None:
    wire = WIRERotaryEmbedding(head_dim=8, num_heads=2, coordinate_dim=3)
    q = torch.randn(5, 2, 8)
    k = torch.randn(7, 2, 8)
    rt = torch.randn(5, 3)
    rs = torch.randn(7, 3)
    q_out, k_out = wire(q, k, rt, rs)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


def test_wire_batched_shapes() -> None:
    wire = WIRERotaryEmbedding(head_dim=4, num_heads=3, coordinate_dim=2)
    q = torch.randn(2, 5, 3, 4)
    k = torch.randn(2, 6, 3, 4)
    rt = torch.randn(2, 5, 2)
    rs = torch.randn(2, 6, 2)
    q_out, k_out = wire(q, k, rt, rs)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


def test_wire_preserves_qk_norms() -> None:
    wire = WIRERotaryEmbedding(head_dim=8, num_heads=4, coordinate_dim=5)
    q = torch.randn(6, 4, 8)
    k = torch.randn(9, 4, 8)
    rt = torch.randn(6, 5)
    rs = torch.randn(9, 5)
    q_out, k_out = wire(q, k, rt, rs)
    assert torch.allclose(q_out.norm(dim=-1), q.norm(dim=-1), atol=1e-5)
    assert torch.allclose(k_out.norm(dim=-1), k.norm(dim=-1), atol=1e-5)


def test_wire_zero_frequencies_noop() -> None:
    wire = WIRERotaryEmbedding(head_dim=6, num_heads=2, coordinate_dim=4)
    with torch.no_grad():
        wire.omega.zero_()
    q = torch.randn(3, 2, 6)
    k = torch.randn(4, 2, 6)
    rt = torch.randn(3, 4)
    rs = torch.randn(4, 4)
    q_out, k_out = wire(q, k, rt, rs)
    assert torch.equal(q_out, q)
    assert torch.equal(k_out, k)


def test_wire_zero_coordinates_noop() -> None:
    wire = WIRERotaryEmbedding(head_dim=6, num_heads=2, coordinate_dim=4)
    q = torch.randn(3, 2, 6)
    k = torch.randn(4, 2, 6)
    rt = torch.zeros(3, 4)
    rs = torch.zeros(4, 4)
    q_out, k_out = wire(q, k, rt, rs)
    assert torch.allclose(q_out, q)
    assert torch.allclose(k_out, k)


def test_wire_common_translation_invariance() -> None:
    wire = WIRERotaryEmbedding(head_dim=8, num_heads=2, coordinate_dim=3)
    q = torch.randn(4, 2, 8)
    k = torch.randn(5, 2, 8)
    rt = torch.randn(4, 3)
    rs = torch.randn(5, 3)
    q1, k1 = wire(q, k, rt, rs)
    c = torch.randn(3)
    q2, k2 = wire(q, k, rt + c, rs + c)
    dots1 = torch.einsum("ihd,jhd->hij", q1, k1)
    dots2 = torch.einsum("ihd,jhd->hij", q2, k2)
    assert torch.allclose(dots1, dots2, atol=1e-5)


def test_wire_relative_rotation_identity() -> None:
    torch.manual_seed(0)
    wire = WIRERotaryEmbedding(head_dim=4, num_heads=1, coordinate_dim=2)
    q = torch.randn(2, 1, 4)
    k = torch.randn(2, 1, 4)
    r = torch.randn(2, 2)
    q_rot, k_rot = wire(q, k, r, r)
    # Compare pairwise dots to q_i^T R(r_j - r_i) k_j
    for i in range(2):
        for j in range(2):
            left = (q_rot[i] * k_rot[j]).sum()
            _, k_rel = _rotate_by_relative(q[i], k[j], wire.omega, r[i], r[j])
            right = (q[i] * k_rel).sum()
            assert torch.allclose(left, right, atol=1e-5)


def test_wire_rejects_odd_head_dim() -> None:
    with pytest.raises(ValueError, match="even head_dim"):
        WIRERotaryEmbedding(head_dim=5, num_heads=1, coordinate_dim=2)


# ---------------------------------------------------------------------------
# Attention integration
# ---------------------------------------------------------------------------


def test_wire_disabled_matches_prewire_attention() -> None:
    torch.manual_seed(0)
    attn = SupportAttention(width=16, heads=4, dropout=0.0)
    target = torch.randn(2, 5, 16)
    source = torch.randn(2, 7, 16)
    support = torch.ones(2, 5, 7, dtype=torch.bool)
    out_a = attn(target, source, support)
    out_b = attn(
        target,
        source,
        support,
        target_coordinates=None,
        source_coordinates=None,
        wire=None,
    )
    assert torch.equal(out_a, out_b)


def test_wire_enabled_matches_dense_reference() -> None:
    torch.manual_seed(1)
    width, heads = 16, 4
    head_dim = width // heads
    attn = SupportAttention(width=width, heads=heads, dropout=0.0)
    wire = WIRERotaryEmbedding(head_dim, heads, coordinate_dim=3)
    batch, n_t, n_s = 2, 4, 5
    target = torch.randn(batch, n_t, width)
    source = torch.randn(batch, n_s, width)
    support = torch.randint(0, 2, (batch, n_t, n_s), dtype=torch.bool)
    support[:, :, 0] = True  # nonempty rows
    rt = torch.randn(batch, n_t, 3)
    rs = torch.randn(batch, n_s, 3)

    out = attn(
        target,
        source,
        support,
        target_coordinates=rt,
        source_coordinates=rs,
        wire=wire,
    )

    # Dense reference with the same weights / WIRE module.
    scale = 1.0 / math.sqrt(width)
    logit_scale = 1.0 / head_dim
    q = (attn.wq(target) * scale).view(batch, n_t, heads, head_dim)
    k = (attn.wk(source) * scale).view(batch, n_s, heads, head_dim)
    v = (attn.wv(source) * scale).view(batch, n_s, heads, head_dim)
    q, k = wire(q, k, rt, rs)
    logits = torch.einsum("bihd,bjhd->bhij", q, k) * logit_scale
    neg = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(~support.unsqueeze(1), neg)
    attn_w = logits.softmax(dim=-1) * support.unsqueeze(1).to(logits.dtype)
    row = attn_w.sum(dim=-1, keepdim=True)
    attn_w = torch.where(row > 0, attn_w / row.clamp_min(1e-12), attn_w)
    ctx = torch.einsum("bhij,bjhd->bihd", attn_w, v).reshape(batch, n_t, width)
    ref = attn.wo(ctx) * scale
    assert torch.allclose(out, ref, atol=1e-5)


def test_wire_logit_scale_is_one_over_d() -> None:
    torch.manual_seed(2)
    width, heads = 8, 2
    d = width // heads
    attn = SupportAttention(width=width, heads=heads, dropout=0.0)
    wire = WIRERotaryEmbedding(d, heads, coordinate_dim=2)
    # Spy on einsum product magnitude via a single pair.
    target = torch.randn(1, 1, width)
    source = torch.randn(1, 1, width)
    support = torch.ones(1, 1, 1, dtype=torch.bool)
    rt = torch.zeros(1, 1, 2)
    rs = torch.zeros(1, 1, 2)

    captured: dict[str, torch.Tensor] = {}

    real_einsum = torch.einsum

    def _capture(equation, *operands):
        result = real_einsum(equation, *operands)
        if equation == "bihd,bjhd->bhij":
            captured["raw"] = result.detach().clone()
            captured["scaled_path"] = result * attn._logit_scale
        return result

    import cpen.models.capen as capen_mod

    original = capen_mod.torch.einsum
    capen_mod.torch.einsum = _capture  # type: ignore[method-assign]
    try:
        _ = attn(
            target,
            source,
            support,
            target_coordinates=rt,
            source_coordinates=rs,
            wire=wire,
        )
    finally:
        capen_mod.torch.einsum = original  # type: ignore[method-assign]

    assert "raw" in captured
    assert abs(attn._logit_scale - 1.0 / d) < 1e-12
    assert abs(attn._logit_scale - 1.0 / math.sqrt(d)) > 1e-6


def test_wire_does_not_rotate_values() -> None:
    torch.manual_seed(3)
    width, heads = 8, 2
    attn = SupportAttention(width=width, heads=heads, dropout=0.0)
    wire = WIRERotaryEmbedding(width // heads, heads, coordinate_dim=2)
    target = torch.randn(1, 2, width)
    source = torch.randn(1, 3, width)
    support = torch.ones(1, 2, 3, dtype=torch.bool)
    rt = torch.randn(1, 2, 2)
    rs = torch.randn(1, 3, 2)

    scale = attn._proj_scale
    v_expected = (attn.wv(source.float()) * scale).view(
        1, 3, heads, width // heads
    )

    real_einsum = torch.einsum
    contexts: list[torch.Tensor] = []

    def einsum_hook(eq, *ops):
        result = real_einsum(eq, *ops)
        if eq == "bhij,bjhd->bihd":
            contexts.append(ops[1].detach().clone())
        return result

    import cpen.models.capen as capen_mod

    original = capen_mod.torch.einsum
    capen_mod.torch.einsum = einsum_hook  # type: ignore[method-assign]
    try:
        _ = attn(
            target,
            source,
            support,
            target_coordinates=rt,
            source_coordinates=rs,
            wire=wire,
        )
    finally:
        capen_mod.torch.einsum = original  # type: ignore[method-assign]

    assert contexts
    assert torch.allclose(contexts[0], v_expected, atol=1e-5)


def test_wire_gradients_to_projections_and_frequencies() -> None:
    torch.manual_seed(4)
    width, heads = 8, 2
    attn = SupportAttention(width=width, heads=heads, dropout=0.0)
    wire = WIRERotaryEmbedding(width // heads, heads, coordinate_dim=2)
    target = torch.randn(1, 3, width, requires_grad=True)
    source = torch.randn(1, 4, width, requires_grad=True)
    support = torch.ones(1, 3, 4, dtype=torch.bool)
    rt = torch.randn(1, 3, 2)
    rs = torch.randn(1, 4, 2)
    out = attn(
        target,
        source,
        support,
        target_coordinates=rt,
        source_coordinates=rs,
        wire=wire,
    )
    out.sum().backward()
    assert attn.wq.weight.grad is not None and attn.wq.weight.grad.abs().sum() > 0
    assert attn.wk.weight.grad is not None and attn.wk.weight.grad.abs().sum() > 0
    assert wire.omega.grad is not None and wire.omega.grad.abs().sum() > 0


def test_wire_rotation_does_not_allocate_nt_ns() -> None:
    """WIRE itself only touches Q/K; it must not build an N_t x N_s matrix."""
    wire = WIRERotaryEmbedding(head_dim=4, num_heads=2, coordinate_dim=2)
    n_t, n_s = 11, 13
    q = torch.randn(n_t, 2, 4)
    k = torch.randn(n_s, 2, 4)
    rt = torch.randn(n_t, 2)
    rs = torch.randn(n_s, 2)
    allocated: list[torch.Size] = []
    real_empty = torch.empty

    def tracking_empty(*args, **kwargs):
        t = real_empty(*args, **kwargs)
        allocated.append(t.shape)
        return t

    torch.empty = tracking_empty  # type: ignore[assignment]
    try:
        q_out, k_out = wire(q, k, rt, rs)
    finally:
        torch.empty = real_empty  # type: ignore[assignment]
    assert q_out.shape == q.shape and k_out.shape == k.shape
    for shape in allocated:
        assert shape != torch.Size([n_t, n_s])
        assert shape != torch.Size([n_t, n_s, 2])


# ---------------------------------------------------------------------------
# Spectral coordinates
# ---------------------------------------------------------------------------


def _path_graph_edges(n: int) -> torch.Tensor:
    src = torch.arange(n - 1)
    dst = src + 1
    return torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)


def test_coords_shape() -> None:
    edge_index = _path_graph_edges(5)
    coords = compute_wire_coordinates(edge_index, 5, num_frequencies=3)
    assert coords.shape == (5, 3)


def test_coords_omit_constant_mode() -> None:
    edge_index = _path_graph_edges(4)
    # Connected path: one combinatorial zero mode. Requesting 4 frequencies
    # must drop that mode and zero-pad the last column.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        coords = compute_wire_coordinates(
            edge_index,
            4,
            num_frequencies=4,
            normalized_laplacian=False,
            standardize=False,
            canonicalize_sign=False,
        )
    assert coords.shape == (4, 4)
    assert coords[:, :3].abs().sum() > 0
    assert coords[:, 3].abs().sum() == 0
    assert any("padding" in str(w.message).lower() for w in caught)
    # Retained modes are not constant (nonzero variance).
    for k in range(3):
        assert float(coords[:, k].var()) > 1e-8


def test_disconnected_multiple_zero_modes() -> None:
    # Two disjoint edges: 2 zero modes; request m=2 nontrivial (may pad).
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        coords = compute_wire_coordinates(
            edge_index,
            4,
            num_frequencies=3,
            standardize=True,
            canonicalize_sign=True,
        )
    assert coords.shape == (4, 3)
    # At most 2 nontrivial modes for 4 nodes with 2 components → pad ≥1.
    assert any("padding" in str(w.message).lower() for w in caught) or (
        coords[:, 2].abs().sum() == 0
    )


def test_sign_canonicalization_deterministic() -> None:
    edge_index = _path_graph_edges(6)
    a = compute_wire_coordinates(edge_index, 6, 3, canonicalize_sign=True)
    b = compute_wire_coordinates(edge_index, 6, 3, canonicalize_sign=True)
    assert torch.equal(a, b)
    for k in range(3):
        col = a[:, k]
        if col.abs().sum() == 0:
            continue
        idx = int(col.abs().argmax())
        assert col[idx] > 0


def test_sign_augmentation_flips_whole_columns() -> None:
    coords = torch.randn(2, 5, 3)
    mask = torch.ones(2, 5, dtype=torch.bool)
    g = torch.Generator().manual_seed(0)
    out = augment_wire_coordinate_signs(coords, mask, generator=g)
    for b in range(2):
        for k in range(3):
            ratio = out[b, :, k] / coords[b, :, k].clamp_min(1e-12)
            # All finite ratios in a column share the same ±1.
            signs = torch.sign(out[b, :, k] * coords[b, :, k])
            assert torch.unique(signs).numel() == 1
            assert abs(float(signs[0])) == 1.0


def test_batched_graphs_independent() -> None:
    # Graph 0: path of 3; graph 1: path of 3 with different edges after pad.
    adj = torch.zeros(2, 4, 4, dtype=torch.bool)
    adj[0, 0, 1] = adj[0, 1, 0] = True
    adj[0, 1, 2] = adj[0, 2, 1] = True
    adj[1, 0, 1] = adj[1, 1, 0] = True
    adj[1, 1, 2] = adj[1, 2, 1] = True
    adj[1, 2, 3] = adj[1, 3, 2] = True
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
    coords = compute_wire_coordinates_batched(adj, mask, num_frequencies=2)
    assert coords.shape == (2, 4, 2)
    assert torch.equal(coords[0, 3], torch.zeros(2))
    # Recompute graph 0 alone and match live slots.
    ei = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    solo = compute_wire_coordinates(ei, 3, 2)
    assert torch.allclose(coords[0, :3], solo, atol=1e-5)


def test_standardized_columns_zero_mean_unit_rms() -> None:
    edge_index = _path_graph_edges(8)
    coords = compute_wire_coordinates(
        edge_index, 8, num_frequencies=3, standardize=True, canonicalize_sign=True
    )
    for k in range(3):
        col = coords[:, k]
        if col.abs().sum() == 0:
            continue
        assert abs(float(col.mean())) < 1e-5
        rms = col.pow(2).mean().sqrt()
        assert abs(float(rms) - 1.0) < 1e-4


def test_particle_adjacency_from_incidence() -> None:
    # One edge connecting particles 0-1; second edge 1-2.
    s = torch.zeros(1, 2, 4, dtype=torch.bool)
    s[0, 0, 0] = s[0, 0, 1] = True
    s[0, 1, 1] = s[0, 1, 2] = True
    mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    adj = particle_adjacency_from_incidence(s, mask=mask)
    assert adj[0, 0, 1] and adj[0, 1, 0]
    assert adj[0, 1, 2] and adj[0, 2, 1]
    assert not adj[0, 0, 2]  # only via co-occurrence of shared edges? 0 and 2
    # 0 and 2 do not share an edge → no direct co-occurrence in S^T S for
    # distinct edges... actually S^T S[i,j] counts shared edges; 0 and 2 share none.
    assert not adj[0, 0, 2]
    assert not adj[0, 3].any()


# ---------------------------------------------------------------------------
# CAPEN-Llama integration (11 only)
# ---------------------------------------------------------------------------


def test_capen_llama_wire_forward() -> None:
    torch.manual_seed(0)
    model = CAPENLlama(
        n_features=3,
        n_edge_features=4,
        out_dim=2,
        depth=2,
        width=16,
        heads=4,
        use_wire=True,
        wire_coordinate_dim=2,
        readout_mode="graph",
    )
    assert model.wire_11 is not None
    assert len(model.wire_11) == 2
    n, m = 4, 2
    x = torch.randn(1, n, 3)
    edge_x = torch.randn(1, m, 4)
    incidence = torch.zeros(1, m, n, dtype=torch.bool)
    incidence[0, 0, 0] = incidence[0, 0, 1] = True
    incidence[0, 1, 1] = incidence[0, 1, 2] = True
    mask = torch.ones(1, n, dtype=torch.bool)
    out = model(x, edge_x, incidence, mask=mask)
    assert out.shape == (1, 2)
    assert torch.isfinite(out).all()


def test_capen_llama_wire_disabled_has_no_modules() -> None:
    model = CAPENLlama(
        n_features=3,
        n_edge_features=4,
        out_dim=2,
        depth=1,
        width=8,
        heads=2,
        use_wire=False,
    )
    assert model.wire_11 is None
    assert model.use_wire is False


def test_capen_llama_wire_requires_even_head_dim() -> None:
    with pytest.raises(ValueError, match="even head_dim"):
        CAPENLlama(
            n_features=3,
            n_edge_features=4,
            out_dim=2,
            depth=1,
            width=10,
            heads=2,  # head_dim=5 odd
            use_wire=True,
        )
