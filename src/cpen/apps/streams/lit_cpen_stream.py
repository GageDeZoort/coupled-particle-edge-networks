"""Semisupervised node+edge Lightning module for multi-MOCK stellar streams."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchmetrics.classification import BinaryAUROC, MulticlassAccuracy

from cpen.lit_models.base_lit_cpen import BaseLitCPEN


class LitCPENStream(BaseLitCPEN):
    """
    Per-MOCK step: CE on supervised nodes; optional edge auxiliary; discovery AUROC.

    Training loss
      L = node_CE(train_mask) + edge_loss_weight * L_edge

    Edge auxiliary (``edge_aux``)
      - ``hard-ce``: CE on ``edge_y`` over ``edge_train_mask``
      - ``product-consistency``: soft target from node membership
        * 2-edges: BCE(p_e, stopgrad(p_u p_v))
        * hyperedges: BCE(p_e, stopgrad(mean_i p_i))  (matches hyper_y = purity ≥ 0.5)
      - ``hard+product``: hard CE plus the consistency term (blob default)

    Discovery (logged every val/test pass)
      - node AUROC on ``~train_mask``
      - edge AUROC on ``~edge_train_mask`` (kNN + hyperedges together)
      - hyper AUROC on hyperedges in that pool, when present
    """

    def __init__(
        self,
        *args,
        class_weights: torch.Tensor | None = None,
        edge_class_weights: torch.Tensor | None = None,
        edge_loss_weight: float = 1.0,
        edge_aux: str = "hard-ce",
        **kwargs,
    ):
        kwargs["heavy_metrics_frac"] = 0.0
        super().__init__(*args, **kwargs)
        out_dim = self.model.out_dim
        if class_weights is None:
            class_weights = torch.ones(out_dim, dtype=torch.float32)
        if edge_class_weights is None:
            edge_class_weights = torch.ones(out_dim, dtype=torch.float32)
        self.register_buffer("class_weights", class_weights.to(torch.float32))
        self.register_buffer("edge_class_weights", edge_class_weights.to(torch.float32))
        self.edge_loss_weight = float(edge_loss_weight)
        aux = str(edge_aux or "hard-ce").strip().lower().replace("_", "-")
        if aux in {"product", "consistency", "product-cons", "pu-pv"}:
            aux = "product-consistency"
        if aux in {"hard+product", "both", "ce+cons", "hard-ce+product-consistency"}:
            aux = "hard+product"
        if aux not in {"hard-ce", "product-consistency", "hard+product"}:
            raise ValueError(
                f"edge_aux must be 'hard-ce', 'product-consistency', or 'hard+product'; "
                f"got {edge_aux!r}"
            )
        self.edge_aux = aux

        self.train_acc = MulticlassAccuracy(num_classes=out_dim)
        self.val_acc = MulticlassAccuracy(num_classes=out_dim)
        self.test_acc = MulticlassAccuracy(num_classes=out_dim)
        self.train_edge_acc = MulticlassAccuracy(num_classes=out_dim)
        self.val_edge_acc = MulticlassAccuracy(num_classes=out_dim)
        self.test_edge_acc = MulticlassAccuracy(num_classes=out_dim)
        self.val_auroc = BinaryAUROC()
        self.test_auroc = BinaryAUROC()
        self.val_edge_auroc = BinaryAUROC()
        self.test_edge_auroc = BinaryAUROC()
        self.val_hyper_auroc = BinaryAUROC()
        self.test_hyper_auroc = BinaryAUROC()

    def running_train_metrics(self, *, sync: bool = True) -> dict[str, float]:
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
    def _unpack_logits(
        output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(output, tuple):
            if len(output) != 2:
                raise ValueError(f"expected (node_logits, edge_logits); got len={len(output)}")
            node_logits, edge_logits = output
            return node_logits, edge_logits
        raise ValueError(
            "stream training requires readout_mode='node+edge' returning "
            f"(node_logits, edge_logits); got {type(output).__name__} "
            f"shape={tuple(output.shape) if torch.is_tensor(output) else 'n/a'}"
        )

    def _masked_ce(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        mask = mask.to(torch.bool)
        supervised = logits[mask]
        y = targets[mask].to(torch.long)
        if y.numel() == 0:
            zero = logits.sum() * 0.0
            empty_probs = logits.new_zeros((0, logits.size(-1)))
            return zero, empty_probs, y, 0
        loss = F.cross_entropy(supervised, y, weight=weight)
        probs = F.softmax(supervised, dim=-1)
        return loss, probs, y, int(y.numel())

    def _edge_endpoints(
        self, batch: dict[str, torch.Tensor], n_edges: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(src, dst)`` node indices of length ``n_edges`` from COO incidence.

        Only valid when every edge has degree 2 (legacy per-MOCK stream graphs).
        """
        if "incidence_node" not in batch:
            raise ValueError("product-consistency requires batch['incidence_node']")
        inc = batch["incidence_node"]
        if inc.dim() == 2:
            inc = inc[0]
        if "incidence_nnz" in batch:
            nnz = int(batch["incidence_nnz"].reshape(-1)[0].item())
        else:
            nnz = int(inc.numel())
        flat = inc[:nnz].to(torch.long)
        if flat.numel() != 2 * n_edges:
            raise ValueError(
                f"incidence_node length {flat.numel()} != 2*n_edges={2 * n_edges}"
            )
        return flat[0::2], flat[1::2]

    def _soft_edge_targets(
        self,
        node_logits: torch.Tensor,
        batch: dict[str, torch.Tensor],
        n_edges: int,
    ) -> torch.Tensor:
        """Stop-grad soft targets for the edge head.

        * kNN 2-edges (first ``n_knn``): ``p_u p_v`` — P(both endpoints stream)
        * hyperedges: mean member probability — matches ``hyper_y`` = purity ≥ 0.5
        """
        p_node = F.softmax(node_logits[0], dim=-1)[:, 1]
        inc_n = batch["incidence_node"]
        inc_e = batch["incidence_edge"]
        if inc_n.dim() == 2:
            inc_n = inc_n[0]
            inc_e = inc_e[0]
        nnz = int(batch["incidence_nnz"].reshape(-1)[0].item()) if "incidence_nnz" in batch else int(inc_n.numel())
        nodes = inc_n[:nnz].to(torch.long)
        edges = inc_e[:nnz].to(torch.long)
        vals = p_node[nodes]
        deg = torch.zeros(n_edges, device=p_node.device, dtype=vals.dtype)
        sum_p = torch.zeros(n_edges, device=p_node.device, dtype=vals.dtype)
        log_prod = torch.zeros(n_edges, device=p_node.device, dtype=vals.dtype)
        ones = torch.ones_like(vals)
        deg.scatter_add_(0, edges, ones)
        sum_p.scatter_add_(0, edges, vals)
        log_prod.scatter_add_(0, edges, vals.clamp_min(1e-6).log())
        mean_p = sum_p / deg.clamp_min(1.0)
        prod_p = log_prod.exp()
        if "n_knn" in batch:
            n_knn = int(batch["n_knn"].reshape(-1)[0].item())
        else:
            n_knn = n_edges
        target = prod_p.clone()
        if n_knn < n_edges:
            target[n_knn:] = mean_p[n_knn:]
        return target.detach().clamp(1e-6, 1.0 - 1e-6)

    def _product_consistency_loss(
        self,
        node_logits: torch.Tensor,
        edge_logits: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, int]:
        """BCE-with-logits against stopgrad node-derived soft targets (B=1)."""
        if node_logits.size(0) != 1 or edge_logits.size(0) != 1:
            raise ValueError(
                "product-consistency currently expects batch_size=1 stream graphs; "
                f"got B={node_logits.size(0)}"
            )
        n_edges = int(edge_logits.size(1))
        if n_edges == 0:
            return edge_logits.sum() * 0.0, 0
        target = self._soft_edge_targets(node_logits, batch, n_edges)
        edge_logit = edge_logits[0, :, 1] - edge_logits[0, :, 0]
        loss = F.binary_cross_entropy_with_logits(edge_logit, target)
        return loss, n_edges

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        x = batch["x"]
        node_logits, edge_logits = self._unpack_logits(
            self.forward(x, **self._model_kwargs(batch))
        )
        if node_logits.dim() != 3 or edge_logits.dim() != 3:
            raise ValueError(
                "expected node/edge logits (B, N|E, C); got "
                f"{tuple(node_logits.shape)} / {tuple(edge_logits.shape)}"
            )

        if stage == "train":
            node_mask = batch["train_mask"]
            edge_mask = batch["edge_train_mask"]
        else:
            node_mask = batch["loss_mask"]
            edge_mask = batch["edge_loss_mask"]

        node_loss, node_probs, node_targets, n_nodes = self._masked_ce(
            node_logits, batch["y"], node_mask, self.class_weights
        )
        if stage == "train" and n_nodes == 0:
            raise ValueError("train train_mask is empty — no supervised nodes")

        # Hard-CE edge head outputs (for acc / diagnostics) always computed on the
        # supervised edge mask when present; training objective may replace CE.
        edge_ce_loss, edge_probs, edge_targets, n_edges_sup = self._masked_ce(
            edge_logits, batch["edge_y"], edge_mask, self.edge_class_weights
        )

        use_cons = self.edge_aux in {"product-consistency", "hard+product"}
        cons_loss, n_edges_cons = (
            self._product_consistency_loss(node_logits, edge_logits, batch)
            if use_cons
            else (edge_logits.sum() * 0.0, 0)
        )
        if self.edge_aux == "product-consistency":
            edge_loss, n_edges_aux = cons_loss, n_edges_cons
        elif self.edge_aux == "hard+product":
            edge_loss, n_edges_aux = edge_ce_loss + cons_loss, n_edges_sup
        else:
            edge_loss, n_edges_aux = edge_ce_loss, n_edges_sup

        loss = node_loss + self.edge_loss_weight * edge_loss
        # Weight running average by supervised nodes (primary task scale).
        loss_weight = max(n_nodes, 1)

        train_mask = batch["train_mask"].to(torch.bool)
        pool = ~train_mask
        pool_probs = F.softmax(node_logits[pool], dim=-1)[:, 1]
        pool_targets = batch["y"][pool].to(torch.long)

        edge_train_mask = batch["edge_train_mask"].to(torch.bool)
        edge_pool = ~edge_train_mask
        edge_pool_probs = F.softmax(edge_logits[edge_pool], dim=-1)[:, 1]
        edge_pool_targets = batch["edge_y"][edge_pool].to(torch.long)

        if stage == "train":
            if n_nodes > 0:
                self.train_acc(node_probs, node_targets)
                self.log(
                    "train_acc",
                    self.train_acc,
                    on_epoch=True,
                    batch_size=n_nodes,
                    sync_dist=True,
                )
            if n_edges_sup > 0:
                self.train_edge_acc(edge_probs, edge_targets)
                self.log(
                    "train_edge_acc",
                    self.train_edge_acc,
                    on_epoch=True,
                    batch_size=n_edges_sup,
                    sync_dist=True,
                )
            self.train_loss_running.update(loss.detach(), weight=loss_weight)
            self.log("train_loss", loss, on_epoch=True, batch_size=loss_weight, sync_dist=True)
            self.log(
                "train_node_loss",
                node_loss,
                on_epoch=True,
                batch_size=max(n_nodes, 1),
                sync_dist=True,
            )
            self.log(
                "train_edge_loss",
                edge_loss,
                on_epoch=True,
                batch_size=max(n_edges_aux, 1),
                sync_dist=True,
            )
            if self.edge_aux in {"product-consistency", "hard+product"}:
                self.log(
                    "train_edge_ce_loss",
                    edge_ce_loss,
                    on_epoch=True,
                    batch_size=max(n_edges_sup, 1),
                    sync_dist=True,
                )
                self.log(
                    "train_edge_cons_loss",
                    cons_loss,
                    on_epoch=True,
                    batch_size=max(n_edges_cons, 1),
                    sync_dist=True,
                )
        elif stage == "val":
            if n_nodes > 0:
                self.val_acc(node_probs, node_targets)
                self.log(
                    "val_acc",
                    self.val_acc,
                    on_epoch=True,
                    batch_size=n_nodes,
                    sync_dist=True,
                )
                self.log(
                    "val_mean_p_stream",
                    node_probs[:, 1].mean(),
                    on_epoch=True,
                    batch_size=n_nodes,
                    sync_dist=True,
                )
            if n_edges_sup > 0:
                self.val_edge_acc(edge_probs, edge_targets)
                self.log(
                    "val_edge_acc",
                    self.val_edge_acc,
                    on_epoch=True,
                    batch_size=n_edges_sup,
                    sync_dist=True,
                )
            if pool_targets.numel() > 0:
                self.val_auroc(pool_probs, pool_targets)
                self.log(
                    "val_auroc",
                    self.val_auroc,
                    on_epoch=True,
                    batch_size=int(pool_targets.numel()),
                    sync_dist=True,
                )
                # Alias for parquet METRIC_KEYS that expect val_roc_auc.
                self.log(
                    "val_roc_auc",
                    self.val_auroc,
                    on_epoch=True,
                    batch_size=int(pool_targets.numel()),
                    sync_dist=True,
                )
            if edge_pool_targets.numel() > 0:
                self.val_edge_auroc(edge_pool_probs, edge_pool_targets)
                self.log(
                    "val_edge_auroc",
                    self.val_edge_auroc,
                    on_epoch=True,
                    batch_size=int(edge_pool_targets.numel()),
                    sync_dist=True,
                )
            if "is_hyper" in batch:
                hyper_pool = edge_pool & batch["is_hyper"].to(torch.bool)
                hp = F.softmax(edge_logits[hyper_pool], dim=-1)[:, 1]
                hy = batch["edge_y"][hyper_pool].to(torch.long)
                if hp.numel() > 0:
                    self.val_hyper_auroc(hp, hy)
                    self.log(
                        "val_hyper_auroc",
                        self.val_hyper_auroc,
                        on_epoch=True,
                        batch_size=int(hp.numel()),
                        sync_dist=True,
                    )
            self.log("val_loss", loss, on_epoch=True, batch_size=loss_weight, sync_dist=True)
            self.log(
                "val_node_loss",
                node_loss,
                on_epoch=True,
                batch_size=max(n_nodes, 1),
                sync_dist=True,
            )
            self.log(
                "val_edge_loss",
                edge_loss,
                on_epoch=True,
                batch_size=max(n_edges_aux, 1),
                sync_dist=True,
            )
        else:
            if n_nodes > 0:
                self.test_acc(node_probs, node_targets)
                self.log(
                    "test_acc",
                    self.test_acc,
                    on_epoch=True,
                    batch_size=n_nodes,
                    sync_dist=True,
                )
                self.log(
                    "test_mean_p_stream",
                    node_probs[:, 1].mean(),
                    on_epoch=True,
                    batch_size=n_nodes,
                    sync_dist=True,
                )
            if n_edges_sup > 0:
                self.test_edge_acc(edge_probs, edge_targets)
                self.log(
                    "test_edge_acc",
                    self.test_edge_acc,
                    on_epoch=True,
                    batch_size=n_edges_sup,
                    sync_dist=True,
                )
            if pool_targets.numel() > 0:
                self.test_auroc(pool_probs, pool_targets)
                self.log(
                    "test_auroc",
                    self.test_auroc,
                    on_epoch=True,
                    batch_size=int(pool_targets.numel()),
                    sync_dist=True,
                )
                self.log(
                    "test_roc_auc",
                    self.test_auroc,
                    on_epoch=True,
                    batch_size=int(pool_targets.numel()),
                    sync_dist=True,
                )
            if edge_pool_targets.numel() > 0:
                self.test_edge_auroc(edge_pool_probs, edge_pool_targets)
                self.log(
                    "test_edge_auroc",
                    self.test_edge_auroc,
                    on_epoch=True,
                    batch_size=int(edge_pool_targets.numel()),
                    sync_dist=True,
                )
            if "is_hyper" in batch:
                hyper_pool = edge_pool & batch["is_hyper"].to(torch.bool)
                hp = F.softmax(edge_logits[hyper_pool], dim=-1)[:, 1]
                hy = batch["edge_y"][hyper_pool].to(torch.long)
                if hp.numel() > 0:
                    self.test_hyper_auroc(hp, hy)
                    self.log(
                        "test_hyper_auroc",
                        self.test_hyper_auroc,
                        on_epoch=True,
                        batch_size=int(hp.numel()),
                        sync_dist=True,
                    )
            self.log("test_loss", loss, on_epoch=True, batch_size=loss_weight, sync_dist=True)
            self.log(
                "test_node_loss",
                node_loss,
                on_epoch=True,
                batch_size=max(n_nodes, 1),
                sync_dist=True,
            )
            self.log(
                "test_edge_loss",
                edge_loss,
                on_epoch=True,
                batch_size=max(n_edges_aux, 1),
                sync_dist=True,
            )
        return loss

    def on_validation_epoch_end(self) -> None:
        self._val_probs.clear()
        self._val_targets.clear()

    def on_test_epoch_end(self) -> None:
        self._test_probs.clear()
        self._test_targets.clear()
