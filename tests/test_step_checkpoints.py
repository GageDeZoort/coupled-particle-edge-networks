from pathlib import Path

import torch

from cpen.training.lightning import last_checkpoint_callback, last_checkpoint_path


def test_last_checkpoint_defaults_to_each_epoch(tmp_path: Path):
    cb = last_checkpoint_callback(tmp_path)
    assert cb._every_n_epochs == 1
    assert cb._every_n_train_steps == 0
    assert cb.save_last is True
    assert cb._enable_version_counter is False


def test_last_checkpoint_can_fire_every_n_steps(tmp_path: Path):
    cb = last_checkpoint_callback(tmp_path, every_n_train_steps=2000)
    assert cb._every_n_train_steps == 2000
    assert cb._every_n_epochs == 0
    assert cb.save_last is True
    assert cb._enable_version_counter is False


def test_last_checkpoint_path_prefers_highest_global_step(tmp_path: Path):
    stale = tmp_path / "last.ckpt"
    newer = tmp_path / "last-v1.ckpt"
    torch.save({"global_step": 400}, stale)
    torch.save({"global_step": 3500}, newer)
    assert last_checkpoint_path(tmp_path) == newer
