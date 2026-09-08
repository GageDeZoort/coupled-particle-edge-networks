"""Node-classification Lightning module for PascalVOC-SP."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import BinaryAUROC, MulticlassF1Score

from cpen.lit_models.base_lit_cpen import BaseLitCPEN


PASCAL_CLASS_WEIGHTS = (
    0.0682,
    5.1760,
    5.3622,
    5.3225,
    6.6348,
    8.2225,
    3.6330,
    2.3136,
    1.5351,
    3.7823,
    8.1427,
    4.8259,
    1.7081,
    5.0962,
    4.1840,
    0.6375,
    7.8190,
    7.1679,
    4.0963,
    3.6047,
    6.4206,
)

# Val incidence-v2: 91.8% interior / 8.2% boundary.
PASCAL_BOUNDARY_POS_WEIGHT = 0.918 / 0.082


def boundary_edge_targets(
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """``edge_y = 1[y_u != y_v]`` and a boolean edge mask, shapes ``(B, M)``."""
    inc = batch["incidence_node"]
    y = batch["y"]
    if inc.dim() != 2 or y.dim() != 2:
        raise ValueError(
            f"expected incidence_node (B, nnz) and y (B, N); "
            f"got {tuple(inc.shape)} / {tuple(y.shape)}"
        )
    src = inc[:, 0::2].to(torch.long).clamp(min=0, max=y.size(1) - 1)
    dst = inc[:, 1::2].to(torch.long).clamp(min=0, max=y.size(1) - 1)
    yu = torch.gather(y, 1, src)
    yv = torch.gather(y, 1, dst)
    edge_y = (yu != yv).to(torch.long)
    if "edge_mask" in batch:
        valid = batch["edge_mask"].to(torch.bool)
    elif "n_edges" in batch:
        n_edges = batch["n_edges"].to(torch.long).reshape(-1, 1)
        idx = torch.arange(src.size(1), device=y.device).unsqueeze(0)
        valid = idx < n_edges
    else:
        valid = torch.ones_like(edge_y, dtype=torch.bool)
    if valid.shape != edge_y.shape:
        valid = valid[:, : edge_y.size(1)]
    return edge_y, valid


def _resize_edge_decoder(model: nn.Module, n_edge_classes: int = 2) -> None:
    """CPEN builds ``decoder_e`` with the node ``out_dim``; boundary needs 2 logits."""
    decoder = getattr(model, "decoder_e", None)
    if decoder is None or not isinstance(decoder, nn.Linear):
        raise TypeError("boundary aux requires CPEN.decoder_e as nn.Linear")
    width = int(decoder.in_features)
    if decoder.out_features == n_edge_classes:
        return
    new = nn.Linear(width, n_edge_classes, bias=False)
    nn.init.normal_(new.weight, mean=0.0, std=1.0 / math.sqrt(width))
    model.decoder_e = new


class LitCPENPascal(BaseLitCPEN):
    """Weighted node CE with PascalVOC-SP's benchmark macro-F1 metric.

    Optional boundary co-training (``readout_mode='node+edge'``): decode
    ``E^{(L)}`` with a 2-way edge readout and CE on ``1[y_u != y_v]``.
    """

    def __init__(
        self,
        *args,
        edge_loss_weight: float = 1.0,
        edge_aux: str = "none",
        **kwargs,
    ):
        kwargs["heavy_metrics_frac"] = 0.0
        super().__init__(*args, **kwargs)
        self.register_buffer(
            "class_weights",
            torch.tensor(PASCAL_CLASS_WEIGHTS, dtype=torch.float32),
        )
        aux = str(edge_aux or "none").strip().lower().replace("_", "-")
        if aux in {"off", "none", "node-only", ""}:
            aux = "none"
        if aux not in {"none", "hard-ce"}:
            raise ValueError(
                f"Pascal edge_aux must be 'none' or 'hard-ce'; got {edge_aux!r}"
            )
        readout = str(getattr(self.model, "readout_mode", "node") or "node")
        self.edge_aux = aux
        self.edge_loss_weight = float(edge_loss_weight)
        self._use_edge_readout = readout == "node+edge" and aux == "hard-ce"
        if hasattr(self.model, "x_only_node_readout"):
            # Skip stock (z_X + T21 z_E)/√2 so node CE is W_X X only.
            self.model.x_only_node_readout = True
        if self._use_edge_readout:
            _resize_edge_decoder(self.model, n_edge_classes=2)
            self.register_buffer(
                "edge_class_weights",
                torch.tensor(
                    [1.0, PASCAL_BOUNDARY_POS_WEIGHT], dtype=torch.float32
                ),
            )
            self.val_edge_auroc = BinaryAUROC()
            self.test_edge_auroc = BinaryAUROC()
        self.train_acc = MulticlassF1Score(num_classes=self.model.out_dim, average="macro")
        self.val_acc = MulticlassF1Score(num_classes=self.model.out_dim, average="macro")
        self.test_acc = MulticlassF1Score(num_classes=self.model.out_dim, average="macro")
        self.train_f1 = MulticlassF1Score(num_classes=self.model.out_dim, average="macro")
        self.val_f1 = MulticlassF1Score(num_classes=self.model.out_dim, average="macro")
        self.test_f1 = MulticlassF1Score(num_classes=self.model.out_dim, average="macro")

    def running_train_metrics(self, *, sync: bool = True) -> dict[str, float]:
        """Step logging only needs the exactly reducible running loss."""
        metrics: dict[str, float] = {}
        loss_updated = bool(
            getattr(self.train_loss_running, "update_called", False)
            or getattr(self.train_loss_running, "_update_called", False)
        )
        if loss_updated:
            numer_t = getattr(self.train_loss_running, "mean_value", None)
            if numer_t is None:
                numer_t = self.train_loss_running.value
            numer = numer_t.detach()
            denom = self.train_loss_running.weight.detach()
            if sync:
                numer = self._all_reduce(numer)
                denom = self._all_reduce(denom)
            if float(denom) > 0:
                metrics["train_loss"] = float((numer / denom).cpu())
        return metrics

    @staticmethod
    def _unpack_node_logits(
        output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(output, tuple):
            if len(output) != 2:
                raise ValueError(f"expected (node_logits, edge_logits); got len={len(output)}")
            return output[0], output[1]
        return output, None

    def _model_kwargs(self, batch: dict[str, torch.Tensor] | tuple) -> dict[str, torch.Tensor]:
        """CPEN.forward does not take WIRE / edge_mask; CAPEN-Llama does."""
        from cpen.models.capen import CAPEN
        from cpen.models.cpen import CPEN

        kwargs = super()._model_kwargs(batch)
        model = self._unwrap_model(self.model)
        if isinstance(model, CAPEN):
            return kwargs
        if isinstance(model, CPEN):
            allowed = {
                "edge_x",
                "incidence",
                "incidence_node",
                "incidence_edge",
                "incidence_nnz",
                "node_degree_inv",
                "edge_degree_inv",
                "mask",
                "z",
            }
            return {key: value for key, value in kwargs.items() if key in allowed}
        return kwargs

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        x = batch["x"]
        mask = batch["mask"].to(torch.bool)
        node_logits, edge_logits = self._unpack_node_logits(
            self.forward(x, **self._model_kwargs(batch))
        )
        node_logits = node_logits[mask]
        targets = batch["y"][mask].to(torch.long)
        loss = F.cross_entropy(node_logits, targets, weight=self.class_weights)
        if self._use_edge_readout:
            if edge_logits is None:
                raise ValueError("boundary aux requires readout_mode='node+edge'")
            edge_y, edge_mask = boundary_edge_targets(batch)
            edge_logits = edge_logits[edge_mask]
            edge_y = edge_y[edge_mask]
            if edge_y.numel() > 0:
                loss = loss + self.edge_loss_weight * F.cross_entropy(
                    edge_logits, edge_y, weight=self.edge_class_weights
                )
        probs = F.softmax(node_logits, dim=-1)
        n_nodes = targets.numel()

        if stage == "train":
            self.train_acc(probs, targets)
            self.train_f1(probs, targets)
            self.train_loss_running.update(loss, weight=n_nodes)
            self.log("train_loss", loss, on_epoch=True, batch_size=n_nodes, sync_dist=True)
            self.log("train_acc", self.train_acc, on_epoch=True, batch_size=n_nodes, sync_dist=True)
            self.log("train_f1", self.train_f1, on_epoch=True, batch_size=n_nodes, sync_dist=True)
        elif stage == "val":
            self.val_acc(probs, targets)
            self.val_f1(probs, targets)
            self.log("val_loss", loss, on_epoch=True, batch_size=n_nodes, sync_dist=True)
            self.log("val_acc", self.val_acc, on_epoch=True, batch_size=n_nodes, sync_dist=True)
            self.log("val_f1", self.val_f1, on_epoch=True, batch_size=n_nodes, sync_dist=True)
            if self._use_edge_readout and edge_logits is not None and edge_y.numel() > 0:
                self.val_edge_auroc(F.softmax(edge_logits, dim=-1)[:, 1], edge_y)
                self.log(
                    "val_edge_auroc",
                    self.val_edge_auroc,
                    on_epoch=True,
                    batch_size=int(edge_y.numel()),
                    sync_dist=True,
                )
        else:
            self.test_acc(probs, targets)
            self.test_f1(probs, targets)
            self.log("test_loss", loss, on_epoch=True, batch_size=n_nodes, sync_dist=True)
            self.log("test_acc", self.test_acc, on_epoch=True, batch_size=n_nodes, sync_dist=True)
            self.log("test_f1", self.test_f1, on_epoch=True, batch_size=n_nodes, sync_dist=True)
            if self._use_edge_readout and edge_logits is not None and edge_y.numel() > 0:
                self.test_edge_auroc(F.softmax(edge_logits, dim=-1)[:, 1], edge_y)
                self.log(
                    "test_edge_auroc",
                    self.test_edge_auroc,
                    on_epoch=True,
                    batch_size=int(edge_y.numel()),
                    sync_dist=True,
                )
        return loss

    def on_validation_epoch_end(self) -> None:
        self._val_probs.clear()
        self._val_targets.clear()

    def on_test_epoch_end(self) -> None:
        self._test_probs.clear()
        self._test_targets.clear()
