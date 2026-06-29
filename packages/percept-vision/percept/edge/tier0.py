"""Public shim for the Python tier-0 edge gate."""

from __future__ import annotations

try:
    from percept.harness.tier0 import *  # noqa: F403
    from percept.harness.tier0 import __all__ as _ALL
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "percept.edge.tier0 needs the Python edge extra. Install with "
        '`pip install "percept-vision[edge]"` or `pip install "percept-vision[runtime]"`.'
    ) from exc

__all__ = list(_ALL)
