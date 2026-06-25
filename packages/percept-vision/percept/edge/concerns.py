"""Public shim for the Python edge concern registry."""

from __future__ import annotations

try:
    from percept_harness.concerns import *  # noqa: F403
    from percept_harness.concerns import __all__ as _ALL
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "percept.edge.concerns needs the Python edge extra. Install with "
        '`pip install "percept-vision[edge]"`.'
    ) from exc

__all__ = list(_ALL)
