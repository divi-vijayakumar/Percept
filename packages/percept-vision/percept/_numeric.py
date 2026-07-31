"""Shared numeric firewalls for untrusted scalars — one rule, one implementation.

Every seam that accepts a number it did not compute (a goal-state hint, a caller timestamp, a
pluggable adapter's return value) needs the same two answers, and the package had been answering
them in three private copies that disagreed:

* :func:`finite` — **refuse** a value that is not a real, finite number. Lifted verbatim from
  ``percept.interest._finite`` (15 call sites there), which was the only shared-quality helper in
  the package and was module-private. ``bool`` is rejected on purpose: ``True`` passes ``float()``
  and silently becomes ``1.0``, which is how a typo'd hint saturates a cutoff instead of failing.
* :func:`clamp01` — **absorb** it. Total, never raises: anything non-numeric or non-finite reads as
  ``0.0``, then the value is pinned to ``[0, 1]``.

:func:`clamp01` deliberately matches the JavaScript edge's ``clamp01`` in
``packages/percept-edge/src/skills.js`` (``typeof v !== "number" || !Number.isFinite(v) -> 0``).
Before this module the Python and JS gates forked on exactly one input — a NaN — with JS reading it
as 0.0 and Python passing it straight through into a comparison that is False against *every*
cutoff, including ``0.0``. A NaN must not be able to admit less than a zero does; the two runtimes
must not disagree about what a broken channel is worth.

Pure stdlib and dependency-free, so the low-level modules (``percept.clock``) can import it.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = ["finite", "clamp01"]


def finite(name: str, value: Any) -> float:
    """``value`` as a float, or :class:`ValueError` naming ``name``. The refusing firewall.

    Rejects ``bool`` (an ``int`` subclass that ``float()`` happily saturates), anything that is not
    an ``int``/``float``, and any non-finite float (``nan``/``inf``/``-inf``). ``name`` is the
    caller's attribution — the field, and where useful the goal — so the refusal says *what* was
    wrong rather than surfacing an unattributed stdlib ``TypeError``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be an int or float")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return 0.0 if result == 0.0 else result


def clamp01(value: Any) -> float:
    """``value`` pinned to ``[0, 1]``; non-numeric / non-finite reads as ``0.0``. Never raises.

    The absorbing firewall, for the channels where a bad reading must degrade to "saw nothing"
    rather than take the frame down — a salience score, a threshold, a fused wake weight. Matches
    the edge's JS ``clamp01`` exactly, including ``bool``: JS ``typeof true !== "number"``, so a
    boolean is not a reading, it is a wrong type, and it reads as ``0.0`` rather than saturating to
    ``1.0``. Same judgement :func:`finite` makes, absorbed instead of raised.
    """
    if isinstance(value, bool):
        return 0.0
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(result):
        return 0.0
    if result < 0.0:
        return 0.0
    if result > 1.0:
        return 1.0
    return result
