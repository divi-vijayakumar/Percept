"""SenseProbe — the eval suite's uniform handle on `percept`'s eight independent senses.

Lives in `tools/` and not in `percept/` on purpose: unifying the senses in the package would break
public API on a qualified 1.0.0 wheel for zero runtime gain. `percept` must never import this.
"""

from .probe import REGISTRY, SenseProbe, SenseSpec, probe

__all__ = ["SenseSpec", "SenseProbe", "REGISTRY", "probe"]
