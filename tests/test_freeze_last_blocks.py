"""Unit tests for last-block freeze helper."""

from __future__ import annotations

from cpen.apps.streams.init_from import freeze_except_last_blocks
from cpen.models.cpen import CPEN


def test_freeze_except_last_block_trains_only_tail_and_decoders() -> None:
    model = CPEN(
        n_features=6,
        n_edge_features=4,
        out_dim=2,
        depth=4,
        width=64,
        operators="incidence",
        residual_structure="transformer-like",
        operator_normalization="degree",
        layer_norm=True,
        dropout=0.0,
        readout_mode="node+edge",
    )
    info = freeze_except_last_blocks(model, n_blocks=1)
    assert info["trainable_block_indices"] == [3]
    # Early block frozen
    assert not any(p.requires_grad for p in model.f11_weights[0].parameters())
    assert not any(p.requires_grad for p in model.encoder_x.parameters())
    # Last block + decoders trainable
    assert all(p.requires_grad for p in model.f11_weights[3].parameters())
    assert all(p.requires_grad for p in model.mlp_x_w1[3].parameters())
    assert all(p.requires_grad for p in model.decoder_x.parameters())
    assert all(p.requires_grad for p in model.decoder_e.parameters())
    assert 0 < info["trainable_frac"] < 0.5
