"""Lightning modules — prefer ``cpen.training`` / ``cpen.apps.*``.

Compatibility re-exports for existing imports.
"""

from cpen.training.base_lit_cpen import BaseLitCPEN
from cpen.apps.jets.lit_cpen_jetclass import LitCPENJetClass
from cpen.apps.jets.lit_cpen_toptagging import LitCPENTopTagging

__all__ = ["BaseLitCPEN", "LitCPENTopTagging", "LitCPENJetClass"]
