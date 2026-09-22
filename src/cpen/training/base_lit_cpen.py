"""Shared LightningModule base for CPEN classification benchmarks."""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics import MeanMetric
from torchmetrics.functional.classification.accuracy import _accuracy_reduce

from cpen.utils.graphs import adjacency_from_batch, build_dense_pairs, build_knn_graph, pairwise_from_batch
from cpen.utils.sparse_incidence import attach_sparse_incidence_batch
from cpen.utils.metrics import should_compute_heavy_metrics


def background_rejection_at_efficiency(
    y_true: torch.Tensor,
    y_score: torch.Tensor,
    *,
    signal_efficiency: float = 0.5,
    positive_class: int = 1,
) -> float:
    """
    Background rejection (1 / background efficiency) at fixed signal efficiency.

    For binary top tagging, *positive_class* is typically the signal (top) label.
    Returns NaN when undefined (e.g. single-class batch).
    """
    y_true = y_true.detach().cpu().numpy()
    if y_score.ndim == 2:
        y_score = y_score[:, positive_class].detach().cpu().numpy()
    else:
        y_score = y_score.detach().cpu().numpy()

    if len(set(y_true.tolist())) < 2:
        return float("nan")

    order = y_score.argsort()[::-1]
    y_sorted = y_true[order]
    signal_mask = y_sorted == positive_class
    n_signal = signal_mask.sum()
    if n_signal == 0:
        return float("nan")

    threshold_idx = max(int(signal_efficiency * n_signal) - 1, 0)
    signal_positions = signal_mask.nonzero()[0]
    if len(signal_positions) <= threshold_idx:
        cutoff = len(y_sorted)
    else:
        cutoff = int(signal_positions[threshold_idx]) + 1

    selected = y_sorted[:cutoff]
    n_bg_selected = (selected != positive_class).sum()
    n_bg_total = (y_true != positive_class).sum()
    if n_bg_total == 0:
        return float("nan")
    bg_eff = n_bg_selected / n_bg_total
    if bg_eff <= 0:
        return float("inf")
    return float(1.0 / bg_eff)


class BaseLitCPEN(L.LightningModule):
    """Classification LightningModule; LR comes from model.get_lr()."""

    # Suppress Adam/AdamW epsilon effects; eps may otherwise need its own
    # transfer prescription (Dey et al. 2025).
    ADAM_EPS = 1e-14

    def __init__(
        self,
        model: nn.Module,
        *,
        eta_0: float,
        model_name: str = "particle-only",
        optimizer: str = "adam",
        scheduler: str = "none",
        lr_decay_min_frac: float = 0.005,
        weight_decay: float = 0.0,
        corr: float = 1.0,
        dataset: str = "",
        depth: int = 4,
        width: int = 128,
        heads: int = 1,
        t_epoch: float | None = None,
        lambda_0: float | None = None,
        run_options: dict[str, Any] | None = None,
        bg_rejection_signal_class: int = 1,
        heavy_metrics_frac: float = 0.0,
        live_graph_k: int | None = None,
        live_dense_pairs: bool = False,
        live_star_radius: float | None = None,
        live_edge_features: str = "part-interaction",
        live_centroid_weight: str | None = None,
        live_no_star_hyperedges: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.eta_0 = eta_0
        self.optimizer_name = optimizer.lower()
        self.scheduler_name = scheduler.lower()
        self.lr_decay_min_frac = lr_decay_min_frac
        self.weight_decay = weight_decay
        self.corr = corr
        self.dataset = dataset
        self.depth = depth
        self.width = width
        self.heads = heads
        self.t_epoch = t_epoch
        self.lambda_0 = lambda_0
        self.run_options = run_options or {}
        self.bg_rejection_signal_class = bg_rejection_signal_class
        self.heavy_metrics_frac = heavy_metrics_frac
        self.live_graph_k = live_graph_k
        self.live_dense_pairs = live_dense_pairs
        self.live_star_radius = live_star_radius
        self.live_edge_features = live_edge_features
        self.live_centroid_weight = live_centroid_weight
        self.live_no_star_hyperedges = bool(live_no_star_hyperedges)
        # live_star_radius + live_graph_k together ⇒ star hyperedges ∥ kNN 2-edges.
        # live_no_star_hyperedges + live_graph_k ⇒ ΔR-kNN 2-edges only.

        self._build_task_metrics(int(model.out_dim))

        self._val_probs: list[torch.Tensor] = []
        self._val_targets: list[torch.Tensor] = []
        self._test_probs: list[torch.Tensor] = []
        self._test_targets: list[torch.Tensor] = []

    def _build_task_metrics(self, out_dim: int) -> None:
        """Task metrics. Regression subclasses override; ``out_dim`` may be 1."""
        self.train_acc = MulticlassAccuracy(num_classes=out_dim)
        self.val_acc = MulticlassAccuracy(num_classes=out_dim)
        self.test_acc = MulticlassAccuracy(num_classes=out_dim)
        self.train_loss_running = MeanMetric()

    def on_train_epoch_start(self) -> None:
        self.train_loss_running.reset()

    @staticmethod
    def _all_reduce(t: torch.Tensor) -> torch.Tensor:
        if dist.is_available() and dist.is_initialized():
            out = t.clone()
            dist.all_reduce(out)
            return out
        return t

    def _mean_metric_value(self, metric: MeanMetric, *, sync: bool = True) -> float | None:
        """Current value of a MeanMetric, or None when it has no updates yet."""
        updated = bool(
            getattr(metric, "update_called", False)
            or getattr(metric, "_update_called", False)
        )
        if not updated:
            return None
        # torchmetrics>=1.x renamed MeanMetric.value -> mean_value (sum of
        # weighted values); weight is still the total weight.
        numer_t = getattr(metric, "mean_value", None)
        if numer_t is None:
            numer_t = metric.value
        numer = numer_t.detach()
        denom = metric.weight.detach()
        if sync:
            numer = self._all_reduce(numer)
            denom = self._all_reduce(denom)
        if float(denom) <= 0:
            return None
        return float((numer / denom).cpu())

    def running_train_metrics(self, *, sync: bool = True) -> dict[str, float]:
        """Running train loss / accuracy within the current epoch.

        Reads torchmetrics internal state directly instead of compute(), which
        breaks under DDP with older torchmetrics (sync_on_compute unsupported,
        double-compute raises "already synced").
        """
        metrics: dict[str, float] = {}
        train_loss = self._mean_metric_value(self.train_loss_running, sync=sync)
        if train_loss is not None:
            metrics["train_loss"] = train_loss

        train_acc = getattr(self, "train_acc", None)
        acc_updated = train_acc is not None and bool(
            getattr(train_acc, "update_called", False)
            or getattr(train_acc, "_update_called", False)
        )
        if acc_updated:
            tp = self.train_acc.tp.detach()
            fp = self.train_acc.fp.detach()
            tn = self.train_acc.tn.detach()
            fn_attr = getattr(self.train_acc, "fn", None)
            fn = fn_attr.detach() if fn_attr is not None else torch.zeros_like(tp)
            if sync:
                tp = self._all_reduce(tp)
                fp = self._all_reduce(fp)
                tn = self._all_reduce(tn)
                fn = self._all_reduce(fn)
            average = getattr(self.train_acc, "average", "micro")
            metrics["train_acc"] = float(
                _accuracy_reduce(tp, fp, tn, fn, average=average).cpu()
            )
        return metrics

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """
        Build kNN / dense-pair edge features batched on GPU.

        DataLoader workers only fetch particle four-vectors; graph construction
        runs here once per batch instead of once per jet in worker processes.
        """
        if not isinstance(batch, dict):
            return batch
        model = self._unwrap_model(self.model)
        if "x_raw" in batch:
            x_raw = batch.pop("x_raw")
            mask = batch["mask"]
            if bool(getattr(model, "m11_only", False)):
                n_ef = int(model.encoder_e.weight.size(1))
                n_nodes = int(mask.size(-1))
                n_batch = int(mask.size(0))
                device = batch["x"].device
                dtype = batch["x"].dtype
                batch["edge_x"] = torch.zeros(
                    n_batch, 1, n_ef, device=device, dtype=dtype
                )
                batch["incidence"] = torch.zeros(
                    n_batch, 1, n_nodes, dtype=torch.bool, device=device
                )
                batch["edge_mask"] = torch.zeros(
                    n_batch, 1, dtype=torch.bool, device=device
                )
            elif self.live_no_star_hyperedges and self.live_graph_k is not None:
                from cpen.graphs.graph_star import build_delta_r_knn_edges

                edge_x, incidence = build_delta_r_knn_edges(
                    x_raw,
                    k=int(self.live_graph_k),
                    mask=mask,
                    edge_features=self.live_edge_features,
                    radius=float(self.live_star_radius or 0.2),
                )
                batch["edge_x"] = edge_x
                batch["incidence"] = incidence
            elif self.live_star_radius is not None and self.live_graph_k is not None:
                from cpen.graphs.graph_star import build_star_plus_knn_graph

                edge_x, incidence = build_star_plus_knn_graph(
                    x_raw,
                    radius=float(self.live_star_radius),
                    k=int(self.live_graph_k),
                    mask=mask,
                    edge_features=self.live_edge_features,
                    centroid_weight=self.live_centroid_weight,
                )
                batch["edge_x"] = edge_x
                batch["incidence"] = incidence
            elif self.live_star_radius is not None:
                from cpen.graphs.graph_star import build_star_radius_graph

                edge_x, incidence, _ = build_star_radius_graph(
                    x_raw,
                    radius=float(self.live_star_radius),
                    mask=mask,
                    edge_features=self.live_edge_features,
                    centroid_weight=self.live_centroid_weight,
                )
                batch["edge_x"] = edge_x
                batch["incidence"] = incidence
            elif self.live_graph_k is not None:
                _, edge_x, incidence = build_knn_graph(x_raw, k=self.live_graph_k, mask=mask)
                batch["edge_x"] = edge_x
                batch["incidence"] = incidence
            elif self.live_dense_pairs:
                batch["edge_x"] = build_dense_pairs(x_raw, mask)
            else:
                raise RuntimeError("batch contains x_raw but no live graph mode is configured")
        else:
            x_raw = None
        if x_raw is not None and bool(getattr(model, "use_rope", False)):
            from cpen.apps.jets.part_kin import jet_centered_deta_dphi

            batch["rope_coordinates"] = jet_centered_deta_dphi(x_raw, batch["mask"])
        if bool(getattr(model, "ignore_knn_edges", False)):
            from cpen.apps.jets.graph_hierarchical import drop_pairwise_edges_from_batch
            from cpen.utils.log_utils import log_info

            n_before = int(batch["edge_x"].size(1)) if "edge_x" in batch else -1
            batch = drop_pairwise_edges_from_batch(batch)
            if not getattr(self, "_logged_ignore_knn", False):
                self._logged_ignore_knn = True
                n_after = int(batch["edge_x"].size(1)) if "edge_x" in batch else -1
                n_live = int(batch["edge_mask"].sum().item()) if "edge_mask" in batch else -1
                log_info(
                    f"[graphs] --hyperedge-only: dropped pairwise 2-edges "
                    f"(kNN, vn_link, deg≤2) M {n_before} → {n_after} "
                    f"(live={n_live} in this batch)"
                )
        return attach_sparse_incidence_batch(batch)

    def _compute_heavy_metrics_this_epoch(self) -> bool:
        if self.trainer is None:
            return True
        return should_compute_heavy_metrics(
            self.trainer.current_epoch,
            self.trainer.max_epochs,
            self.heavy_metrics_frac,
        )

    @staticmethod
    def _unwrap_model(model: nn.Module) -> nn.Module:
        """Return the inner module when wrapped by torch.compile."""
        return getattr(model, "_orig_mod", model)

    def _model_kwargs(self, batch: dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> dict[str, torch.Tensor]:
        from cpen.models.capen import CAPEN
        from cpen.models.cpen import CPEN
        from cpen.models.particle_only import ParticleOnlyNetwork

        if isinstance(batch, dict):
            x = batch["x"]
        else:
            x, _ = batch

        model = self._unwrap_model(self.model)
        if isinstance(model, (CPEN, CAPEN)):
            if not isinstance(batch, dict):
                raise ValueError("CPEN/CAPEN requires graph batches with edge_x and incidence")
            kwargs = {
                "edge_x": batch["edge_x"],
                "incidence": batch.get("incidence"),
                "incidence_node": batch.get("incidence_node"),
                "incidence_edge": batch.get("incidence_edge"),
                "incidence_nnz": batch.get("incidence_nnz"),
                "node_degree_inv": batch.get("node_degree_inv"),
                "edge_degree_inv": batch.get("edge_degree_inv"),
                "mask": batch.get("mask"),
                "z": batch.get("z"),
            }
            if batch.get("wire_coordinates") is not None:
                kwargs["wire_coordinates"] = batch["wire_coordinates"]
            if batch.get("rope_coordinates") is not None:
                kwargs["rope_coordinates"] = batch["rope_coordinates"]
            if batch.get("x_raw") is not None:
                kwargs["x_raw"] = batch["x_raw"]
            # Hierarchical / typed caches (CAPEN-Llama-att).
            if batch.get("edge_type") is not None:
                kwargs["edge_type"] = batch["edge_type"]
            if batch.get("edge_mask") is not None:
                kwargs["edge_mask"] = batch["edge_mask"]
            if batch.get("node_mask") is not None:
                kwargs["node_mask"] = batch["node_mask"]
            return kwargs

        if isinstance(model, ParticleOnlyNetwork) and "adjacency" in model.operator_names:
            if not isinstance(batch, dict):
                raise ValueError(
                    "Particle-only adjacency requires graph batches from a cached "
                    "star-R or kNN datamodule."
                )
            if model.operator_normalization == "gamma":
                return {"pairwise": pairwise_from_batch(batch)}
            return {"adjacency": adjacency_from_batch(batch)}
        return {}

    def _unpack_batch(
        self,
        batch: dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, dict):
            return batch["x"], batch["y"]
        return batch

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # torch.compile does not reliably thread **kwargs into positional edge_x.
        if "edge_x" in kwargs:
            edge_x = kwargs.pop("edge_x")
            return self.model(x, edge_x, **kwargs)
        return self.model(x, **kwargs)

    def _shared_step(
        self,
        batch: dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor],
        stage: str,
    ) -> torch.Tensor:
        x, y = self._unpack_batch(batch)
        logits = self.forward(x, **self._model_kwargs(batch))
        loss = F.cross_entropy(logits, y)
        probs = F.softmax(logits, dim=-1)

        if stage == "train":
            self.train_acc(probs, y)
            self.train_loss_running.update(loss, weight=x.size(0))
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=x.size(0))
            self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=False, batch_size=x.size(0))
        elif stage == "val":
            self.val_acc(probs, y)
            if self._compute_heavy_metrics_this_epoch():
                self._val_probs.append(probs.detach())
                self._val_targets.append(y.detach())
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=x.size(0))
            self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=x.size(0))
        elif stage == "test":
            self.test_acc(probs, y)
            self._test_probs.append(probs.detach())
            self._test_targets.append(y.detach())
            self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=x.size(0))
            self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True, batch_size=x.size(0))
        return loss

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def _epoch_classification_metrics(
        self,
        probs_list: list[torch.Tensor],
        targets_list: list[torch.Tensor],
        prefix: str,
    ) -> None:
        if not probs_list:
            return
        probs = torch.cat(probs_list, dim=0)
        targets = torch.cat(targets_list, dim=0)

        try:
            if self.model.out_dim == 2:
                auc = roc_auc_score(
                    targets.cpu().numpy(),
                    probs[:, 1].cpu().numpy(),
                )
            else:
                auc = roc_auc_score(
                    targets.cpu().numpy(),
                    probs.cpu().numpy(),
                    multi_class="ovr",
                    average="macro",
                )
        except ValueError:
            auc = float("nan")

        bg_rej_05 = background_rejection_at_efficiency(
            targets,
            probs,
            signal_efficiency=0.5,
            positive_class=self.bg_rejection_signal_class,
        )
        bg_rej_03 = background_rejection_at_efficiency(
            targets,
            probs,
            signal_efficiency=0.3,
            positive_class=self.bg_rejection_signal_class,
        )
        self.log(f"{prefix}_roc_auc", auc, prog_bar=False, sync_dist=True)
        self.log(f"{prefix}_bg_rejection", bg_rej_05, prog_bar=False, sync_dist=True)
        self.log(f"{prefix}_bg_rejection_0p3", bg_rej_03, prog_bar=False, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        if self._compute_heavy_metrics_this_epoch():
            self._epoch_classification_metrics(self._val_probs, self._val_targets, "val")
        else:
            # Avoid carrying forward stale AUC / rejection from the last heavy-metrics epoch.
            self.log("val_roc_auc", float("nan"), prog_bar=False, sync_dist=True)
            self.log("val_bg_rejection", float("nan"), prog_bar=False, sync_dist=True)
            self.log("val_bg_rejection_0p3", float("nan"), prog_bar=False, sync_dist=True)
        self._val_probs.clear()
        self._val_targets.clear()

    def on_test_epoch_end(self) -> None:
        self._epoch_classification_metrics(self._test_probs, self._test_targets, "test")
        self._test_probs.clear()
        self._test_targets.clear()

    def configure_optimizers(self):
        lr = self.model.get_lr(eta_0=self.eta_0, corr=self.corr)
        # Decoupled weight decay is AdamW-only; plain Adam uses L2-in-loss if wd>0.
        wd = self.weight_decay if self.optimizer_name == "adamw" else 0.0
        params = [p for p in self.parameters() if p.requires_grad]
        if not params:
            raise ValueError("configure_optimizers: no trainable parameters")
        if self.optimizer_name == "adam":
            opt = torch.optim.Adam(params, lr=lr, weight_decay=wd, eps=self.ADAM_EPS)
        elif self.optimizer_name == "adamw":
            opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd, eps=self.ADAM_EPS)
        elif self.optimizer_name == "sgd":
            opt = torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer_name!r}")

        if self.scheduler_name == "cosine":
            scheduler = CosineAnnealingLR(
                opt, T_max=self.trainer.max_epochs, eta_min=self.lr_decay_min_frac * lr
            )
            return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        return opt

    def static_metadata(self) -> dict[str, Any]:
        meta = {
            "model": self.model_name,
            "dataset": self.dataset,
            "optimizer": self.optimizer_name,
            "depth": self.depth,
            "width": self.width,
            "heads": self.heads,
            "eta_0": self.eta_0,
            "weight_decay": self.weight_decay,
            "adam_eps": self.ADAM_EPS if self.optimizer_name in {"adam", "adamw"} else None,
            "scheduler": self.scheduler_name,
            "lr_decay_min_frac": self.lr_decay_min_frac if self.scheduler_name == "cosine" else None,
            "corr": self.corr,
            "heavy_metrics_frac": self.heavy_metrics_frac,
        }
        if self.t_epoch is not None:
            meta["t_epoch"] = self.t_epoch
        if self.lambda_0 is not None:
            meta["lambda_0"] = self.lambda_0
        meta.update(self.run_options)
        return meta
