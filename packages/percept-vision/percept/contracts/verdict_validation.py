"""The ONE verdict-validation authority — the "single boundary" ADR-001 §7 names.

ADR-001 §7 says a backend fails in exactly two ways: it raises, or it returns something that is
not a well-formed :class:`Verdict`. Both are "normalized at a single boundary; see §10 invariant
4". Invariant 4 then says why that matters more than it reads: the code downstream dereferences
``.confidence`` / ``.satisfied`` with no guard of its own, so **an unvalidated malformed verdict is
not a crash, it is a confident wrong notification** — the worst output this system can produce.

"A single boundary" was, until this module existed, three boundaries that disagreed:

- ``percept.agent._is_valid_verdict`` checked three terms (exact type, ``satisfied`` is a bool,
  ``confidence`` is numeric).
- ``percept.backends.judge_pool._valid_verdict`` checked six (duck-typed, and additionally: the
  confidence is not a ``bool``, is finite, is inside ``[0, 1]``, and the evidence is a ``str``).
- ``percept.conformance.assert_verdict`` checked eight — the only complete one, and the only one
  that also constrained the optional ``usage`` / ``events`` accounting fields.

The gap was reachable from a live provider, not theoretical. Every shipped backend builds
``Verdict(confidence=float(data.get("confidence", 0.0)))`` straight off decoded provider JSON, and
``json.loads`` accepts bare ``NaN`` / ``Infinity`` tokens by default. Feed the agent-side validator
a ``nan`` confidence and it answers "well-formed"; ``percept.gate.classify`` then evaluates
``nan < threshold`` — which is ``False``, because every comparison against NaN is ``False`` — falls
through the freshness arm the same way, and returns ``"yes"``. A garbled number became a confident
FIRE, on a rising edge, delivered to a human. The pool-side twin already refused all five of
``nan`` / ``inf`` / ``7.5`` / ``-0.5`` / ``True``; the agent-side twin accepted all five.

So: one implementation, and the callers differ only in the one axis they legitimately differ on.

**The two modes, and why the flag is per-call-site rather than global.**

- ``require_exact_type=True`` — ``isinstance(value, Verdict)``. This is the contract as
  :mod:`percept.conformance` states it to an adapter author: import the dataclass, do not
  duck-type a look-alike, because the gate reads the dataclass. It is what the agent's own
  cognition path demands of anything that reaches ``_VerdictSnapshot``.
- ``require_exact_type=False`` — duck-typed via :func:`getattr`. The judge pool does this
  deliberately at the provider seam: a deployment may hand the pool a ``Verdict``-shaped object of
  its own (including a ``Verdict`` subclass, which passes either way), and the pool's job there is
  to classify a route's answer as decisive / uncertain / malformed, not to police the type. The
  asymmetry is fail-closed — the stricter side is the one guarding the gate.

Neither mode is "the" default for the package: making one global would either break every
duck-typed fake at the pool seam or silently loosen the agent's boundary. It is an argument
precisely so each call site states which contract it is enforcing.

**A token, not a bool.** :func:`verdict_defect` returns ``None`` when conformant, else exactly one
stable string from :data:`VERDICT_DEFECTS`. The callers all want to say WHICH term failed:
``agent.py`` stamps a ``malformed_response`` ``_ProviderFailure``, the pool records a ``malformed``
route outcome, and ``conformance.assert_verdict`` raises a long human message naming the field. A
bare bool would have forced each of them to re-derive the reason — which is how three
implementations of one rule happened in the first place. :func:`is_valid_verdict` is the bool
projection for the call sites that only branch on it.

The tokens are a CLOSED, ordered set and are checked in the order they are listed: a value with
several defects always reports the first one, so the token is deterministic rather than dependent
on evaluation order. They are stable identifiers safe to switch on and to put in telemetry.

**What this deliberately does NOT do.** It does not clamp, coerce, or repair. A malformed verdict
is refused, never quietly rounded into range — a 7.5 silently clamped to 1.0 would turn a backend
emitting a 0-100 scale into a permanent confident FIRE, which is the exact defect this module
exists to prevent. Normalizing is the caller's decision (the agent substitutes an explicit
abstention; the pool fails the route over), and it is made where the caller's failure policy lives.

This module is stdlib-only and imports nothing but the frozen :class:`Verdict`, so it is reachable
from the zero-dependency consumers ``percept.contracts`` exists for. It defines no DTO and changes
no wire shape, so ``CONTRACT_VERSION`` does not move.
"""

from __future__ import annotations

import math
from typing import Any

from . import Verdict

#: The CLOSED set of defect tokens :func:`verdict_defect` may return, in the order the terms are
#: checked. A value with more than one defect reports the FIRST in this order, so the token a given
#: value produces is deterministic. Safe to switch on and to record in telemetry.
VERDICT_DEFECTS: tuple[str, ...] = (
    "not_a_verdict",
    "satisfied_not_bool",
    "confidence_is_bool",
    "confidence_not_numeric",
    "confidence_not_finite",
    "confidence_out_of_range",
    "evidence_not_str",
    "usage_not_dict",
    "events_not_list",
)

#: Sentinel for "this object does not carry the attribute at all", distinct from "carries ``None``".
#: Only the two OPTIONAL accounting fields use it — see :func:`verdict_defect`.
_ABSENT = object()


def _confidence_defect(confidence: Any) -> str | None:
    """The confidence terms, in order — the ones the gate dereferences without a guard.

    ``bool`` is rejected before the numeric test rather than after it, because ``isinstance(True,
    int)`` is ``True`` in Python: a naive numeric check reads ``True`` as the confidence ``1.0``,
    which is a maximally confident fire produced by a backend that emitted a flag where a
    probability belongs.

    Finiteness is asked ONLY of a real ``float``. A Python ``int`` is finite by construction and
    cannot be NaN or infinite, and ``math.isfinite`` on a large one raises ``OverflowError``
    converting it to a float — so an oversized integer confidence must fall through to the range
    term (which refuses it honestly) instead of blowing up the validator.
    """
    if isinstance(confidence, bool):
        return "confidence_is_bool"
    if not isinstance(confidence, (int, float)):
        return "confidence_not_numeric"
    if isinstance(confidence, float) and not math.isfinite(confidence):
        return "confidence_not_finite"
    if not (0.0 <= confidence <= 1.0):
        return "confidence_out_of_range"
    return None


def verdict_defect(value: Any, *, require_exact_type: bool = True) -> str | None:
    """Name the ONE way ``value`` fails the :class:`Verdict` contract, or ``None`` if it does not.

    The single authority behind ``agent._is_valid_verdict``, ``judge_pool._valid_verdict`` and
    ``conformance.assert_verdict`` (ADR-001 §7, §10 invariant 4). Call this BEFORE any code
    dereferences ``.confidence`` / ``.satisfied``.

    ``require_exact_type=True`` (the default) demands ``isinstance(value, Verdict)`` — the contract
    as stated to an adapter author. ``False`` duck-types every term through :func:`getattr`, which
    is what the judge pool wants at the provider seam; see the module docstring for why the choice
    belongs to the call site.

    The three REQUIRED terms — ``satisfied``, ``confidence``, ``evidence`` — are always checked; a
    duck-typed object missing one reports that term's defect, because an absent field is not a
    conformant field. The two OPTIONAL accounting terms — ``usage`` and ``events`` — are checked
    only when the object actually carries them. On a real ``Verdict`` they always exist (the
    dataclass defaults them), so this is not a loophole there; it is what keeps a minimal duck-typed
    look-alike, which the pool has always accepted, from being newly refused for omitting fields the
    backend contract never required it to carry.

    Returns one token from :data:`VERDICT_DEFECTS`, or ``None`` when the value is conformant.
    """
    if require_exact_type:
        if not isinstance(value, Verdict):
            return "not_a_verdict"
    elif value is None:
        # Duck-typing still has a floor. ``None`` is the pool's "no verdict at all" (a route that
        # raised), and naming it `not_a_verdict` says that, where reading `.satisfied` off it would
        # report the far less useful `satisfied_not_bool`.
        return "not_a_verdict"

    if not isinstance(getattr(value, "satisfied", None), bool):
        # A truthy string or a 0/1 int is NOT conformant: the gate branches on this, and
        # ``"no" if verdict.satisfied else ...`` reads a non-empty "false" string as True.
        return "satisfied_not_bool"

    defect = _confidence_defect(getattr(value, "confidence", None))
    if defect is not None:
        return defect

    if not isinstance(getattr(value, "evidence", None), str):
        return "evidence_not_str"

    usage = getattr(value, "usage", _ABSENT)
    if usage is not _ABSENT and not isinstance(usage, dict):
        return "usage_not_dict"

    events = getattr(value, "events", _ABSENT)
    if events is not _ABSENT and not isinstance(events, list):
        return "events_not_list"

    return None


def is_valid_verdict(value: Any, *, require_exact_type: bool = True) -> bool:
    """Bool projection of :func:`verdict_defect` — ``True`` iff the value is a conformant Verdict.

    For the call sites that only branch on conformance and never report which term failed. Anything
    that DOES report a reason should call :func:`verdict_defect` directly rather than re-deriving it.
    """
    return verdict_defect(value, require_exact_type=require_exact_type) is None


__all__ = ["VERDICT_DEFECTS", "verdict_defect", "is_valid_verdict"]
