import pandas as pd

from cpen.training.transfer_plots import (
    format_n_train,
    metrics_at_epoch,
    plot_dataset_size_lr,
)


def _epoch_row(**kwargs):
    base = dict(
        record_type="epoch",
        epoch=0,
        global_step=10,
        train_loss=1.2,
        val_acc=0.5,
        val_roc_auc=0.9,
        val_bg_rejection=100.0,
        val_loss=1.1,
        eta_0=0.1,
        depth=4,
        width=256,
        heads=8,
        n_train=100_000,
        epochs=5,
        role="100k",
        job_id="1",
        note="",
        run_name="capen-llama_4_256",
    )
    base.update(kwargs)
    return base


def test_format_n_train():
    assert format_n_train(100_000) == "100k"
    assert format_n_train(1_000_000) == "1M"
    assert format_n_train(5_000_000) == "5M"


def test_metrics_at_epoch_keeps_first_epoch_only():
    df = pd.DataFrame(
        [
            _epoch_row(job_id="a", epoch=0, train_loss=1.4, global_step=10),
            _epoch_row(job_id="a", epoch=1, train_loss=0.9, global_step=20),
            _epoch_row(job_id="b", epoch=0, eta_0=0.5, train_loss=1.3, global_step=10),
        ]
    )
    out = metrics_at_epoch(df, epoch=0)
    assert list(out["job_id"]) == ["a", "b"]
    assert list(out["train_loss"]) == [1.4, 1.3]


def test_plot_dataset_size_lr(tmp_path):
    rows = []
    for n_train, role in ((100_000, "100k"), (1_000_000, "1M")):
        for width, heads in ((256, 8), (512, 16)):
            for eta in (0.05, 0.1, 0.25):
                rows.append(
                    _epoch_row(
                        job_id=f"{role}_{width}_{eta}",
                        n_train=n_train,
                        role=role,
                        width=width,
                        heads=heads,
                        eta_0=eta,
                        train_loss=1.0 + 0.1 * eta,
                        val_acc=0.5 + 0.01 * (width / 256),
                        val_roc_auc=0.9,
                        val_bg_rejection=80 + n_train / 1e5,
                    )
                )
    df = pd.DataFrame(rows)
    fig = plot_dataset_size_lr(
        df, stem="unit_dataset_size_lr", fig_dir=tmp_path, show=False
    )
    assert fig is not None
    assert len(fig.axes) == 4
    assert (tmp_path / "unit_dataset_size_lr.pdf").is_file()
