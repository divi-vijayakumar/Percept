"""String sugar for ``watch()`` — ``watch(stream, "the milk is boiling over")``.

**Pass-through is not compilation.** This module is deliberately the dumbest possible layer: it
takes the plain-language string the caller typed and puts it, BYTE-FOR-BYTE, into
:attr:`Goal.condition`. It does not parse it, normalise it, stem it, lowercase it, extract an
object/attribute out of it, or map it onto any vocabulary. There is exactly one firing path and the
judge sees the caller's own words — the cardinal rule stated in ``percept.contracts`` ("the Vision
seam judges a goal's plain-language *condition* ... it never compiles the goal to a rigid
``attr_key=value`` and exact-matches") is preserved here by construction, because there is nowhere
in this module where the condition text is read for meaning. The only thing derived from the text is
the goal's **id**, and that derivation is a pure content hash, not an interpretation.

Documented id scheme (stable — treat it as part of the contract):

    ``g_<slug>-<h8>``

* ``<slug>``  — the condition lowercased, with every run of non-``[a-z0-9]`` characters collapsed to
  a single ``_``, trimmed of leading/trailing ``_``, truncated to 32 characters. Purely cosmetic: it
  exists so ids are readable in logs. A condition with no ascii alphanumerics slugs to ``goal``.
* ``<h8>``    — the first 8 hex characters of ``blake2s`` over the **verbatim, unmodified** condition
  (utf-8). This, not the slug, is what makes the id unique and deterministic.

Consequences a caller can rely on:

* the same condition string always desugars to the same id, in this process and the next (no
  counters, no clock, no randomness) — so re-arming the same watch re-uses the same goal;
* two *different* conditions never collide, even when their slugs are identical (``"a dog"`` vs
  ``"A DOG!"`` share a slug but not a hash);
* nothing about the id is fed back into judging — it is a handle, not a signal.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable

from .adapters import Goal

#: Maximum characters of the (cosmetic) readable slug portion of a desugared id.
SLUG_MAX = 32
#: Hex characters of the blake2s content digest appended to every desugared id.
DIGEST_HEX = 8

_NON_SLUG = re.compile(r"[^a-z0-9]+")


def _slug(condition: str) -> str:
    """The readable, cosmetic half of a desugared id. Never used for anything but display."""
    s = _NON_SLUG.sub("_", condition.lower()).strip("_")
    s = s[:SLUG_MAX].strip("_")
    return s or "goal"


def goal_id_for(condition: str) -> str:
    """The documented, deterministic id for a bare-string goal: ``g_<slug>-<h8>``.

    Pure function of the verbatim condition — no counters, clock, or randomness — so the same string
    yields the same id across processes and restarts. The hash is taken over the ORIGINAL text (not
    the slug), so conditions that differ only in case/punctuation still get distinct ids.
    """
    h = hashlib.blake2s(condition.encode("utf-8"), digest_size=8).hexdigest()[:DIGEST_HEX]
    return f"g_{_slug(condition)}-{h}"


def desugar_goal(spec: Any, *, source: str = "camera") -> Goal:
    """Coerce one ``goals=`` entry to a :class:`Goal`.

    * a :class:`Goal` is returned **unchanged and unwrapped** (identity — not a copy), so nothing a
      caller authored explicitly is ever rewritten by the sugar layer;
    * a ``str`` becomes ``Goal(id=goal_id_for(s), condition=s, say=s, source=source)`` with the
      condition (and the spoken line) carrying the caller's exact characters — no strip, no case
      change, no punctuation fixing.

    Anything else raises :class:`TypeError`. An empty/whitespace-only string raises
    :class:`ValueError`: there is no condition to judge, and silently arming it would create a goal
    that can never fire.
    """
    if isinstance(spec, Goal):
        return spec
    if isinstance(spec, str):
        if not spec.strip():
            raise ValueError(
                "a goal condition must be non-empty plain language, e.g. "
                'watch(stream, "the milk is boiling over")'
            )
        return Goal(id=goal_id_for(spec), condition=spec, say=spec, source=source)
    raise TypeError(
        f"goal must be a str (a plain-language condition) or a Goal, not {type(spec).__name__}"
    )


def desugar_goals(goals: Any, *, source: str = "camera") -> list[Goal]:
    """Normalise the ``goals`` argument of :func:`percept.watch.watch` to ``list[Goal]``.

    Accepts a bare string, a bare :class:`Goal`, or any iterable mixing the two (a list of strings
    is the headline case). Order is preserved exactly; a bare scalar becomes a one-element list.
    ``None`` is not a goal set and raises :class:`TypeError` like any other unsupported type.
    """
    if isinstance(goals, (str, Goal)):
        return [desugar_goal(goals, source=source)]
    if isinstance(goals, Iterable):
        return [desugar_goal(g, source=source) for g in goals]
    raise TypeError(
        "goals must be a str, a Goal, or an iterable of them, not " f"{type(goals).__name__}"
    )


__all__ = ["desugar_goal", "desugar_goals", "goal_id_for", "SLUG_MAX", "DIGEST_HEX"]
