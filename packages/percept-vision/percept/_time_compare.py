"""Float-safe comparisons for the one place floats always bite us: a time HORIZON boundary.

Every time seam in this package is fed wall-clock or media timestamps that arrive as IEEE-754
doubles, and a uniformly-sampled stream does NOT produce uniform differences: at 10 Hz,
``0.4 - 0.3 == 0.10000000000000003``, which is strictly greater than a ``0.1`` horizon while
``0.3 - 0.2 == 0.09999999999999998`` is strictly less. A bare ``dt <= horizon`` therefore flips the
documented INCLUSIVE boundary to exclusive on SOME steps and not others, so the same cadence
fragments arbitrarily — phantom gaps, and a total like ``0.7999999999999998`` where the client
authored a horizon exactly equal to their cadence and expected ``0.8``.

The rest of the package already answers this at comparable seams with
``math.isclose(rel_tol=0.0, abs_tol=...)`` (``events.py``, ``fused_sync.py``, ``audio_input.py`` —
all of them EQUALITY checks). :func:`within` is the ordered-boundary form of the same idea, kept
here so the tolerance policy has ONE home rather than being re-derived at each call site.

**The fix does not belong in the client's config.** Telling a caller "author the horizon strictly
greater than your cadence" pushes a floating-point subtlety onto them and silently punishes the
most natural configuration there is (horizon == cadence). The boundary is inclusive as documented;
this module makes it inclusive in fact.

**Direction matters.** The tolerance widens the admitted side, so use :func:`within` only where
erring INCLUSIVE is the safe error. Two known future candidates, both in ``percept/gate.py`` and
both deliberately untouched here:

* ``gate.py:627`` (dwell, ``now - t0 >= hold_s``) already errs LATE — a dwell that needs one more
  tick is conservative.
* ``gate.py:597`` (refractory, ``(now - last_fire) < effective_refractory``) errs by ending
  suppression one tick EARLY, which is the unsafe direction (a duplicate fire); that one would
  genuinely benefit from this helper.
"""

from __future__ import annotations

import math

__all__ = ["within"]


def within(dt: float, horizon: float, *, abs_tol: float = 1e-9) -> bool:
    """Is ``dt`` inside ``horizon``, treating the boundary itself as INSIDE?

    ``abs_tol`` is absolute by design (``rel_tol=0.0``): these are durations in seconds, where a
    fixed nanosecond-scale slack is meaningful and a relative one would grow without bound over a
    long-running stream. ``1e-9`` is far below any real sampling cadence and far above the
    accumulated representation error of summed decimal timestamps.
    """
    return dt <= horizon or math.isclose(dt, horizon, rel_tol=0.0, abs_tol=abs_tol)
