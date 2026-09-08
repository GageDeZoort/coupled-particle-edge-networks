"""LightningModule for TopTagging."""

from cpen.lit_models.base_lit_cpen import BaseLitCPEN


class LitCPENTopTagging(BaseLitCPEN):
    """Binary top vs QCD tagging; background rejection uses class 1 as signal."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("bg_rejection_signal_class", 1)
        super().__init__(*args, **kwargs)
