"""Concern registry — the standing-concern twin of the detector registry.

FUTURE (public API, exercised only by tests): this registry + :func:`compile_registered_watchspec`
are a complete, semver-stable surface, but NOTHING in ``percept/runtime/`` calls them today — the
live path is still the per-call ``tier0.compile_watchspec``. Kept as the forward seam for standing
(registered) concerns; wire a runtime consumer before treating registration as load-bearing. Verified
2026-07-12: ``grep -rn 'compile_registered_watchspec\\|register_concern' percept/runtime/`` is empty.

Concerns were previously composed per-call only (the conductor authors goals, ``tier0.compile_watchspec``
turns each watch goal into a :class:`~percept.adapters.WatchCondition` on the fly). This module adds the
missing *registration lifecycle* for concerns, mirroring ``detectors.py``:

    register_concern(concern)  ->  registered_concerns() / get_concern(id)  ->  WatchSpec selection
                                                                            ->  unregister_concern() / clear_concerns()

A registered :class:`~percept.adapters.Concern` is a standing unit of consequence-weighted attention that
the brain can compile DOWN to a :class:`~percept.adapters.WatchSpec` (:func:`compile_registered_watchspec`)
**alongside** the existing per-call path — registering concerns does not touch ``compile_watchspec`` or its
byte-for-byte behavior. The registry is a process-global singleton keyed by ``Concern.id``; this is a
PUBLIC, semver-stable surface (see ``docs/registration.md``).

Why a separate registry from detectors: a *detector* is a cheap reusable signal skill (one impl, many
goals); a *concern* is a specific standing thing-to-watch (one per id, carrying its own condition +
consequence). They share the register/teardown shape but not the keyspace.
"""

from __future__ import annotations

from typing import Iterable

from percept.adapters import Concern, Goal, WatchCondition, WatchSpec

from .detectors import coerce_authored_detectors, partition_by_capability
from .tier0 import _clamp01, _coerce_box

# ---------------------------------------------------------------------------
# The registry — register a Concern; select it into a WatchSpec by id
# ---------------------------------------------------------------------------

_CONCERNS: dict[str, Concern] = {}


def register_concern(concern: Concern) -> Concern:
    """Register a standing :class:`~percept.adapters.Concern` under its ``id`` (last-writer-wins).

    The deliberate-registration lifecycle for concerns: after this, the concern is returned by
    :func:`registered_concerns`, fetchable via :func:`get_concern`, and selectable into a
    :class:`~percept.adapters.WatchSpec` by :func:`compile_registered_watchspec`. Re-registering the same
    ``id`` replaces the prior concern (how a refinement updates a standing concern in place). Returns the
    concern for convenient chaining.

    Accepts a :class:`~percept.adapters.Goal` too (via :meth:`Concern.from_goal`) so a goal-shaped caller
    can register without first building a full Concern — the compat shim keeps today's firing unchanged.
    """
    if isinstance(concern, Goal):
        concern = Concern.from_goal(concern)
    _CONCERNS[concern.id] = concern
    return concern


def registered_concerns() -> list[Concern]:
    """All registered concerns, ordered by ``id`` (the standing set the brain watches)."""
    return [_CONCERNS[k] for k in sorted(_CONCERNS)]


def registered_concern_ids() -> list[str]:
    """Registered concern ids (sorted)."""
    return sorted(_CONCERNS)


def get_concern(concern_id: str) -> Concern:
    """Fetch a registered concern by id (raises ``KeyError`` if absent)."""
    if concern_id not in _CONCERNS:
        raise KeyError(f"unknown concern {concern_id!r}; registered: {sorted(_CONCERNS)}")
    return _CONCERNS[concern_id]


def unregister_concern(concern_id: str) -> bool:
    """Tear down one concern by id. Returns ``True`` if it was registered, ``False`` if absent.

    The teardown half of the lifecycle: after this the concern is no longer selected into a
    :class:`~percept.adapters.WatchSpec`. A WatchSpec already compiled keeps the conditions it captured —
    teardown affects *future* compilation, not specs already produced.
    """
    return _CONCERNS.pop(concern_id, None) is not None


def clear_concerns() -> None:
    """Empty the concern registry (test isolation / a clean re-registration pass)."""
    _CONCERNS.clear()


# ---------------------------------------------------------------------------
# Selection — registered concerns DOWN to a WatchSpec (alongside compile_watchspec)
# ---------------------------------------------------------------------------


def detectors_for_concern(concern: Concern) -> list[str]:
    """The edge skills a concern selects — its ``detectors`` override (validated), else empty.

    Mirrors :func:`~percept.harness.detectors.detectors_for` for the goal path. A concern carries no
    ``state`` dict (it is a frozen DTO), so the selection reads an OPTIONAL ``detectors`` attribute a
    caller may attach to the concern before registering; absent -> ``[]`` (byte-for-byte: an
    un-annotated concern selects no detectors). Names are run through the registry firewall
    (:func:`~percept.harness.detectors.coerce_authored_detectors`), so an unknown skill is dropped.
    """
    return coerce_authored_detectors(getattr(concern, "detectors", None))


def concern_to_condition(
    concern: Concern,
    *,
    threshold: float,
    edge_capabilities: Iterable[str] | None = None,
) -> WatchCondition:
    """Compile one :class:`~percept.adapters.Concern` to a :class:`~percept.adapters.WatchCondition`.

    The concern analog of a watch goal's row in :func:`~percept.harness.tier0.compile_watchspec`:

    - ``goal_id`` <- ``concern.id``
    - ``anchor_text`` <- ``concern.condition`` (the plain-language anchor the edge salience-matches)
    - ``threshold`` <- the gate default ``threshold`` (clamped to ``0..1``)
    - ``region_hint`` <- an optional ``region_hint`` attribute attached to the concern, coerced to a
      :class:`~percept.adapters.Box` (absent -> ``None``, like a goal without ``region_hint``)
    - ``detectors`` <- :func:`detectors_for_concern`, filtered to ``edge_capabilities`` (the same
      capability handshake the goal path uses)
    """
    supported, _dropped = partition_by_capability(
        detectors_for_concern(concern), edge_capabilities
    )
    return WatchCondition(
        goal_id=concern.id,
        anchor_text=concern.condition,
        threshold=_clamp01(threshold),
        region_hint=_coerce_box(getattr(concern, "region_hint", None)),
        detectors=supported,
    )


def compile_registered_watchspec(
    *,
    threshold: float,
    version: int = 1,
    edge_capabilities: Iterable[str] | None = None,
    concerns: Iterable[Concern] | None = None,
) -> WatchSpec:
    """Compile the REGISTERED concerns DOWN to a :class:`~percept.adapters.WatchSpec`.

    The concern-registry counterpart of :func:`~percept.harness.tier0.compile_watchspec` (which compiles
    a per-call list of goals). One :class:`~percept.adapters.WatchCondition` per concern, in id order.
    ``concerns`` defaults to :func:`registered_concerns` — pass an explicit iterable to compile a subset
    without registering. ``edge_capabilities`` applies the same capability handshake as the goal path.

    This runs ALONGSIDE the goal path; it does not replace or modify ``compile_watchspec``. A product can
    register standing concerns once and call this each tick instead of (or in addition to) composing goals
    per call.
    """
    src = registered_concerns() if concerns is None else list(concerns)
    conditions = [
        concern_to_condition(c, threshold=threshold, edge_capabilities=edge_capabilities)
        for c in src
    ]
    return WatchSpec(conditions=conditions, version=version)


__all__ = [
    "register_concern",
    "registered_concerns",
    "registered_concern_ids",
    "get_concern",
    "unregister_concern",
    "clear_concerns",
    "detectors_for_concern",
    "concern_to_condition",
    "compile_registered_watchspec",
]
