"""Transfer prescriptions for Adam LR and AdamW weight decay."""

from __future__ import annotations

import math

from cpen.models.base import adam_lr, lambda_0_from_t_epoch, t_epoch_weight_decay


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
