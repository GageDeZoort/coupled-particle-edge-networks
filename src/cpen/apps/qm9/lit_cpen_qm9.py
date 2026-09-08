"""Graph-regression Lightning module for QM9 molecular properties."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torchmetrics import MeanMetric

from cpen.training.base_lit_cpen import BaseLitCPEN

LOSSES = ("l1", "mse", "huber")


class LitCPENQM9(BaseLitCPEN):
    """Single-target regression on the pooled graph readout.

    The loss is computed on the standardized target, so it is scale-free across
    QM9 properties whose magnitudes span four decades (\\(\\mu\\sim2.7\\,\\mathrm{D}\\)
    versus \\(G\\sim-11180\\,\\mathrm{eV}\\)). MAE is rescaled by ``target_std`` and
    logged in the target's physical unit, which is the number to compare against
    published QM9 results.
    """

    def __init__(
        self,
        model,
        *args,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        target_name: str = "mu",
        target_unit: str = "D",
        loss: str = "l1",
        readout_mode: str = "graph",
        **kwargs,
    ) -> None:
        mode = str(readout_mode or "graph")
        if mode not in {"graph", "node"}:
            raise ValueError(
                f"QM9 regression needs a pooled graph readout ('graph' mixes "
                f"z_X and z_E, 'node' pools z_X only), got {mode!r}"
            )
        out_dim = int(getattr(model, "out_dim", 0))
        if out_dim != 1:
            raise ValueError(f"QM9 regression needs out_dim=1, got {out_dim}")
        loss_name = str(loss or "l1").strip().lower()
        if loss_name not in LOSSES:
            raise ValueError(f"QM9 loss must be one of {LOSSES}, got {loss!r}")
        std = float(target_std)
        if not std > 0.0:
            raise ValueError(f"target_std must be positive, got {target_std!r}")
        # Molecules have variable bond counts, so pooling has to honour
        # edge_mask. Setting this flag routes the forward through the
        # mask-aware graph readout instead of the unmasked base pooling.
        model.x_only_graph_readout = mode == "node"
        super().__init__(model, *args, **kwargs)
        self.readout_mode = mode
        self.loss_name = loss_name
        self.target_name = str(target_name)
        self.target_unit = str(target_unit)
        self.target_mean = float(target_mean)
        self.target_std = std

    def _build_task_metrics(self, out_dim: int) -> None:
        self.train_loss_running = MeanMetric()
        self.train_mae = MeanMetric()
        self.val_mae = MeanMetric()
        self.test_mae = MeanMetric()

    def running_train_metrics(self, *, sync: bool = True) -> dict[str, float]:
        metrics = super().running_train_metrics(sync=sync)
        mae = self._mean_metric_value(self.train_mae, sync=sync)
        if mae is not None:
            metrics["train_mae"] = mae
        return metrics

    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_name == "mse":
            return F.mse_loss(pred, target)
        if self.loss_name == "huber":
            return F.huber_loss(pred, target)
        return F.l1_loss(pred, target)

    def standardize(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.target_mean) / self.target_std

    def unstandardize(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.target_std + self.target_mean

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        x, y = self._unpack_batch(batch)
        pred = self.forward(x, **self._model_kwargs(batch)).reshape(-1)
        target = self.standardize(y.reshape(-1).to(pred.dtype))
        loss = self._loss(pred, target)
        # Physical-unit MAE: the standardized residual times the target scale.
        mae = (pred - target).abs().mean().detach() * self.target_std
        batch_size = x.size(0)

        if stage == "train":
            self.train_loss_running.update(loss.detach(), weight=batch_size)
            self.train_mae(mae, weight=batch_size)
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
            self.log("train_mae", self.train_mae, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
        elif stage == "val":
            self.val_mae(mae, weight=batch_size)
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)
            self.log("val_mae", self.val_mae, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
        elif stage == "test":
            self.test_mae(mae, weight=batch_size)
            self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=batch_size)
            self.log("test_mae", self.test_mae, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
        return loss

    def on_validation_epoch_end(self) -> None:
        """No ROC AUC / background rejection for a regression target."""

    def on_test_epoch_end(self) -> None:
        """No ROC AUC / background rejection for a regression target."""

    def static_metadata(self) -> dict[str, Any]:
        meta = super().static_metadata()
        meta.update(
            {
                "task": "graph-reg",
                "loss_fn": self.loss_name,
                "qm9_target": self.target_name,
                "qm9_target_unit": self.target_unit,
                "qm9_target_mean": self.target_mean,
                "qm9_target_std": self.target_std,
            }
        )
        return meta
