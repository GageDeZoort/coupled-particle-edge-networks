"""DDP launch-mode selection for Slurm vs single-process spawn."""

from __future__ import annotations

import pytest
import torch

from cpen.training.lightning import _trainer_parallel_settings


def test_single_task_uses_ddp_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLURM_NTASKS", "1")
    monkeypatch.delenv("SLURM_NTASKS_PER_NODE", raising=False)
    devices, _strategy, num_nodes, mode = _trainer_parallel_settings(
        n_gpus=4, use_gpu=True
    )
    assert devices == 4
    assert num_nodes == 1
    assert mode == "ddp-spawn"


def test_srun_multitask_uses_one_device_per_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLURM_NTASKS", "4")
    monkeypatch.setenv("SLURM_NTASKS_PER_NODE", "4")
    monkeypatch.setenv("SLURM_NNODES", "1")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    devices, _strategy, num_nodes, mode = _trainer_parallel_settings(
        n_gpus=4, use_gpu=True
    )
    assert devices == 1
    assert num_nodes == 1
    assert mode == "slurm-multitask"


def test_srun_with_all_gpus_visible_is_a_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLURM_NTASKS", "4")
    monkeypatch.setenv("SLURM_NTASKS_PER_NODE", "4")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    with pytest.raises(RuntimeError, match="gpus-per-task=1"):
        _trainer_parallel_settings(n_gpus=4, use_gpu=True)
