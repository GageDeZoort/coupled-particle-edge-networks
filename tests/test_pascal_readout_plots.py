import pandas as pd

from cpen.training.transfer_plots import (
    normalize_model_family,
    pascal_readout_summary,
    plot_pascal_readout_lr,
)


def _epoch_row(**kwargs):
    base = dict(
        record_type="epoch",
        epoch=1,
        global_step=10,
        val_loss=1.0,
        val_f1=0.1,
        val_acc=0.4,
        train_loss=0.8,
        eta_0=0.25,
        depth=4,
        width=256,
        heads=8,
        epochs=10,
        model="cpen",
        role="node",
        job_id="1",
        note="",
        run_name="cpen_4_256",
    )
    base.update(kwargs)
    return base


def test_normalize_model_family():
    assert normalize_model_family("capen-llama") == "capen-llama"
    assert normalize_model_family("cpen") == "cpen"


def test_pascal_summary_and_family_plot(tmp_path):
    rows = [
        _epoch_row(job_id="c1", model="cpen", role="node", eta_0=0.05, val_f1=0.10),
        _epoch_row(job_id="c2", model="cpen", role="node+edge", eta_0=0.05, val_f1=0.12),
        _epoch_row(
            job_id="l1",
            model="capen-llama",
            role="node",
            eta_0=0.05,
            val_f1=0.11,
            run_name="capen-llama_4_256",
        ),
        _epoch_row(
            job_id="l2",
            model="capen-llama",
            role="node+edge",
            eta_0=0.05,
            val_f1=0.13,
            run_name="capen-llama_4_256",
        ),
    ]
    df = pd.DataFrame(rows)
    df["model_family"] = df["model"]
    summary = pascal_readout_summary(df)
    assert set(summary["model_family"]) == {"cpen", "capen-llama"}
    assert "val_acc_best" in summary.columns
    assert "train_loss_min" in summary.columns
    fig = plot_pascal_readout_lr(
        df, stem="unit_pascal_family", fig_dir=tmp_path, show=False
    )
    assert fig is not None
    assert len(fig.axes) == 2
    assert (tmp_path / "unit_pascal_family.pdf").is_file()
