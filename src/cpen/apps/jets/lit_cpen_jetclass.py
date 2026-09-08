"""LightningModule for JetClass."""

from cpen.lit_models.base_lit_cpen import BaseLitCPEN


class LitCPENJetClass(BaseLitCPEN):
    """Multi-class JetClass benchmark."""

    def __init__(self, *args, **kwargs):
        # Background rejection helper defaults to class 0 until a signal class is chosen.
        kwargs.setdefault("bg_rejection_signal_class", 0)
        super().__init__(*args, **kwargs)
