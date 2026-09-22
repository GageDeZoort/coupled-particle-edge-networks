"""Unit tests for JetClass → TopTagging fine-tune helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from cpen.training.init_from import (
    assert_ft_root_not_pretrain,
    is_readout_param,
    load_init_checkpoint,
    read_ckpt_global_step,
)
from scans.common.sweep_common import run_options_from_args, toptagging_live_active


class _TinyLit(nn.Module):
    """Minimal stand-in with trunk + readout-shaped params."""

    def __init__(self, out_dim: int = 2, width: int = 8) -> None:
        super().__init__()
        self.encoder_x = nn.Linear(7, width)
        self.trunk = nn.Linear(width, width)
        self.decoder_x = nn.Linear(width, out_dim)
        self.output_attention_pool = nn.Linear(width, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder_x(self.trunk(self.encoder_x(x)))


def test_is_readout_param_names() -> None:
    assert is_readout_param("model.decoder_x.weight")
    assert is_readout_param("model.output_attention_pool.class_tokens")
    assert is_readout_param("model.output_attention_pool.class_unembed")
    assert not is_readout_param("model.encoder_x.weight")
    assert not is_readout_param("model.trunk.weight")


def test_load_init_skips_readout_and_copies_trunk(tmp_path: Path) -> None:
    src = _TinyLit(out_dim=10)
    with torch.no_grad():
        src.encoder_x.weight.fill_(0.3)
        src.trunk.weight.fill_(0.7)
        src.decoder_x.weight.fill_(1.5)

    ckpt = tmp_path / "pretrain.ckpt"
    torch.save({"state_dict": src.state_dict(), "global_step": 1234}, ckpt)

    dst = _TinyLit(out_dim=2)
    with torch.no_grad():
        dst.encoder_x.weight.zero_()
        dst.trunk.weight.zero_()
        dst.decoder_x.weight.fill_(-1.0)

    load_init_checkpoint(dst, ckpt, skip_readout=True, min_load_frac=0.2)
    assert torch.allclose(dst.encoder_x.weight, src.encoder_x.weight)
    assert torch.allclose(dst.trunk.weight, src.trunk.weight)
    # Head stays randomly / freshly initialized (not copied from 10-way).
    assert not torch.allclose(dst.decoder_x.weight, torch.full_like(dst.decoder_x.weight, 1.5))
    assert read_ckpt_global_step(ckpt) == 1234


def test_assert_ft_root_rejects_jetclass_trees() -> None:
    assert_ft_root_not_pretrain("/scratch/cpen_runs/toptagging_finetune")
    with pytest.raises(ValueError, match="JetClass"):
        assert_ft_root_not_pretrain(
            "/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law"
        )


def test_toptagging_live_ft_tags() -> None:
    args = SimpleNamespace(
        dataset="toptagging",
        toptagging_live=True,
        blob_run=False,
        init_from="/tmp/fake.ckpt",
        init_from_step=8500,
        train_last_blocks=0,
        star_radius=0.2,
        live_knn_k=6,
        jetclass_edge_features="part-interaction",
        num_particles=80,
        no_star_hyperedges=False,
        max_steps=20000,
        seed=0,
        run_tag="ft-pilot",
        operator_normalization="degree",
        model="capen-llama-att",
        use_rope=False,
        rope_theta=100.0,
        corr=1.0,
        all_edge_m22=False,
        uniform_output_pool=False,
        hyperedge_only=False,
        identity_m22=False,
        incidence_m22=False,
        edge_aux="hard-ce",
        edge_loss_weight=1.0,
        include_real_streams=False,
        stream_name="all",
        readout_mode="graph",
        layer_norm=True,
        sigma_output=None,
        encoder="linear",
        decoder="linear",
        scheduler="none",
        heads=12,
        dropout=0.0,
        operators="incidence",
        normalization="uniform",
        residual_structure="transformer-like",
        alpha=None,
        gamma=None,
        n_train=None,
        n_val=20000,
        n_test=None,
        graph_construction="live",
        n_features=7,
        use_wire=False,
        wire_coordinate_dim=8,
        attention_normalization="gamma",
        jetclass_stream=False,
        jetclass_features="kin7",
        jetclass_sort_by_pt=False,
        jetclass_centroid_weight=None,
        blob_rich_min_mock=100,
        blob_empty_per_rich=1.0,
        blob_include_dilute=False,
        blob_node_frame="local",
        blob_drop_features=None,
        blob_no_hyperedges=False,
        blob_cell_split=False,
        blob_mask_holdout=False,
        blob_galaxies="0000",
        blob_holdout_streams="",
    )
    assert toptagging_live_active(args)
    opts = run_options_from_args(args)
    tag = opts.extra_tag or ""
    assert "ft" in tag
    assert "initfrom" in tag
    assert "live" in tag
    assert "partint" in tag
    assert "knn6" in tag
    assert "p80" in tag
    assert "pstep8p5k" in tag or "pstep8500" in tag
