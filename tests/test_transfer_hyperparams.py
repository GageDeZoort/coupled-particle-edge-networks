"""Transfer prescriptions for Adam LR and AdamW weight decay."""

from __future__ import annotations

import math
from types import SimpleNamespace

from cpen.models.base import adam_lr, lambda_0_from_t_epoch, t_epoch_weight_decay
from cpen.models.registry import weight_decay_for_run


def test_adam_lr_scales_with_sqrt_width() -> None:
    eta_0 = 0.3
    assert adam_lr(eta_0, 256) == eta_0 / math.sqrt(256)
    assert adam_lr(eta_0, 512) == eta_0 / math.sqrt(512)


def test_t_epoch_weight_decay_transfer() -> None:
    eta_0 = 0.5
    batch_size = 128
    n_gpus = 4
    n_train = 1_211_000
    t_epoch = 10.0

    lambda_0 = lambda_0_from_t_epoch(
        eta_0=eta_0,
        batch_size=batch_size,
        n_gpus=n_gpus,
        n_train=n_train,
        t_epoch=t_epoch,
    )
    assert math.isclose(
        lambda_0,
        batch_size * n_gpus / (t_epoch * eta_0 * n_train),
    )

    wd_256 = t_epoch_weight_decay(
        eta_0=eta_0,
        width=256,
        batch_size=batch_size,
        n_gpus=n_gpus,
        n_train=n_train,
        t_epoch=t_epoch,
    )
    wd_512 = t_epoch_weight_decay(
        eta_0=eta_0,
        width=512,
        batch_size=batch_size,
        n_gpus=n_gpus,
        n_train=n_train,
        t_epoch=t_epoch,
    )

    lr_256 = adam_lr(eta_0, 256)
    lr_512 = adam_lr(eta_0, 512)
    assert math.isclose(wd_256, lambda_0 * math.sqrt(256))
    assert math.isclose(wd_512, lambda_0 * math.sqrt(512))
    assert math.isclose(wd_256 * lr_256, wd_512 * lr_512)


def test_weight_decay_for_run_uses_t_epoch_schedule() -> None:
    """AdamW + t_epoch must not silently collapse to weight_decay=0."""
    args = SimpleNamespace(
        optimizer="adamw",
        weight_decay=0.0,
        batch_size=128,
        n_gpus=4,
    )
    eta_0 = 0.1
    n_train = 100_000_000
    t_epoch = 10.0
    for width in (128, 256, 512):
        got = weight_decay_for_run(
            args,
            width=width,
            n_train=n_train,
            t_epoch=t_epoch,
            eta_0=eta_0,
        )
        expected = t_epoch_weight_decay(
            eta_0=eta_0,
            width=width,
            batch_size=128,
            n_gpus=4,
            n_train=n_train,
            t_epoch=t_epoch,
        )
        assert got > 0.0
        assert math.isclose(got, expected)

    adam_args = SimpleNamespace(optimizer="adam", weight_decay=0.0, batch_size=128, n_gpus=4)
    assert weight_decay_for_run(adam_args, width=512, n_train=n_train, t_epoch=t_epoch, eta_0=eta_0) == 0.0
