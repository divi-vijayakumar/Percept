"""Public Python edge namespace.

The implementation currently lives in the internal ``percept.harness`` package so it can be tested and
published independently during the preview. Users should import through ``percept.edge``:

    from percept.edge.tier0 import GatedWatchLoop
    from percept.edge.detectors import register_detector
"""

from __future__ import annotations

try:
    from percept.harness import *  # noqa: F403
    from percept.harness import __all__ as _HARNESS_ALL
except ModuleNotFoundError as exc:  # pragma: no cover - only when the optional extra is absent
    raise ModuleNotFoundError(
        "percept.edge needs the Python edge extra. Install with "
        '`pip install "percept-vision[edge]"` or `pip install "percept-vision[runtime]"`.'
    ) from exc

__all__ = list(_HARNESS_ALL)
