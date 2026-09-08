"""Graph-classification Lightning module for MNISTSuperpixels."""

from __future__ import annotations

from cpen.lit_models.base_lit_cpen import BaseLitCPEN


class LitCPENMNIST(BaseLitCPEN):
    """Digit CE + accuracy. node pools z_X; node+edge mixes pooled heads."""

    def __init__(self, model, *args, readout_mode: str = "node", **kwargs):
        mode = str(readout_mode or "node")
        if mode not in {"node", "node+edge", "graph", "attn"}:
            raise ValueError(
                f"MNIST readout_mode must be node, node+edge, or attn, got {mode!r}"
            )
        if mode == "attn":
            if getattr(model, "output_attention_pool", None) is None:
                raise ValueError(
                    "readout_mode=attn requires CPEN built with readout_mode='attn'"
                )
        else:
            # CLI ``graph`` is the mixed pool (same logits as node+edge).
            model.x_only_graph_readout = mode == "node"
        super().__init__(model, *args, **kwargs)
