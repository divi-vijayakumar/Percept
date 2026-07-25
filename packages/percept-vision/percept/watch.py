"""The public ``watch()`` entrypoint — the frozen seam every downstream capability depends on.

``watch(stream, goals, profile=None) -> AsyncIterator[WatchEvent]`` is the ONE stable, versioned way
to run the existing fovea -> judge -> gate path (:meth:`~percept.agent.Percept.perceive_judged`) over a
stream of frames. It is a thin wrapper, not a new firing path: **no detection behavior is hardcoded
here** — this module adds zero judge/gate logic, it only adapts the existing per-frame loop to a
stream-in/stream-out shape and maps its output to the frozen :class:`~percept.contracts.WatchEvent`.

``profile`` is resolved for real: a :class:`~percept.watchprofile.WatchProfile` binds
(environment, goal-type, state) to the runtime's knobs (``salience_stat``, ``plan_mode``, cadence, gate
config, ...) — the free variables of ``Q(p | environment, goal, state)`` — and that resolution
(:func:`percept.profile_wiring.resolve_watch_profile`) is applied when :meth:`Percept.create` builds the
core (never a mid-run mutation). The documented default is still ``profile=None`` -> today's
:class:`~percept.config.Settings`-driven behavior (:func:`~percept.config.load_settings`), BYTE-FOR-BYTE
unchanged from the legacy default. See :func:`_settings_from_profile` and :func:`_build_core`.

``goals`` additionally accepts bare plain-language strings —
``watch(stream, "the milk is boiling over")``. That is SUGAR, not a second entrypoint and not a
compilation step: :func:`percept.goalsugar.desugar_goals` wraps the string, verbatim, in a
``Goal`` with a deterministic documented id, and the identical firing path runs. The signature,
streaming semantics, and ``WATCH_VERSION`` are unchanged — only the accepted shape of an existing
argument widens, which no existing caller can observe.

``goals`` additionally accepts a :class:`~percept.contracts.watchplan.WatchPlan` (ADR-008) — a
typed, portable statement of intent that may carry ``explicit`` per-field overrides. A ``WatchPlan``
entry is resolved (:class:`percept.resolver.ConfigResolver`) and armed
(:func:`percept.watchplan_apply.apply_watch_plan`, which also runs the existing ``coerce_authored_*``
firewall over its compiled state) exactly as if the caller had done that themselves before calling
``watch()``; the bare-string/``Goal`` path (above) is DELIBERATELY left untouched by this widening —
it keeps calling :func:`~percept.goalsugar.desugar_goals` directly, with no resolver in between, so
that a caller who never constructs a ``WatchPlan`` gets byte-for-byte pre-ADR-008 behavior with no
new env read and no new provenance bookkeeping. Wiring the string-sugar path through the resolver
too (so its filled SDK defaults carry provenance) needs a receipt-return channel this frozen
signature does not have; left for a later card.
"""

from __future__ import annotations

from typing import Any, AsyncIterable, AsyncIterator, Iterable

from .adapters import Goal
from .agent import FireEvent, Percept
from .config import Settings
from .contracts import WatchEvent
from .contracts.watchplan import WatchPlan
from .goalsugar import desugar_goals
from .profile_wiring import resolve_watch_profile
from .watchplan_apply import apply_watch_plan
from .watchprofile import WatchProfile

#: Semver of the ``watch()`` entrypoint contract itself — the function signature + streaming
#: semantics — distinct from ``percept.contracts.CONTRACT_VERSION`` (the DTO wire shapes it returns).
#: Bump MAJOR on a breaking change to the signature or streaming contract; MINOR for an additive
#: (new optional kwarg); PATCH otherwise.
WATCH_VERSION = "1.1.0"


def _settings_from_profile(profile: Any) -> Settings | None:
    """Resolve ``profile`` to a :class:`Settings` override, or ``None`` for the default.

    A real :class:`~percept.watchprofile.WatchProfile` resolves for real (via
    :func:`percept.profile_wiring.resolve_watch_profile`). Two minimal, non-behavior-changing
    pass-throughs remain for callers that aren't authoring a ``WatchProfile``: a bare ``Settings``
    instance is used directly (handy for tests/callers that already build one), and any other object's
    ``.settings`` attribute is read if present. Anything else (including ``None``) resolves to ``None``,
    so :meth:`Percept.create` falls back to ``load_settings()`` — today's unchanged default.
    """
    if profile is None:
        return None
    if isinstance(profile, WatchProfile):
        return resolve_watch_profile(profile).settings
    if isinstance(profile, Settings):
        return profile
    return getattr(profile, "settings", None)


def _build_core(profile: Any) -> Percept:
    """Build the default core for ``watch()`` when no ``percept=`` is supplied — the ONE place a
    ``WatchProfile`` is applied at CORE CREATION time (cadence -> a real ``SensePolicy``, ``plan_mode``
    threaded through, never a mid-run mutation). Any other ``profile`` shape (``None``, a bare
    ``Settings``, an opaque ``.settings``-bearing object) carries no cadence/plan_mode override — same
    as the legacy default."""
    if isinstance(profile, WatchProfile):
        resolved = resolve_watch_profile(profile)
        return Percept.create(
            settings=resolved.settings,
            sense_policy=resolved.sense_policy,
            plan_mode=resolved.plan_mode,
        )
    return Percept.create(settings=_settings_from_profile(profile))


def _as_watch_event(fired: FireEvent, *, t: float) -> WatchEvent:
    return WatchEvent(
        goal_id=fired.goal_id,
        action=fired.action,
        t=t,
        text=fired.text,
        entity_id=fired.entity_id,
        verdict=fired.verdict,
    )


async def _aiter(stream: "AsyncIterable[Any] | Iterable[Any]") -> AsyncIterator[Any]:
    """Adapt a sync OR async iterable of frames to one async-iteration protocol."""
    if hasattr(stream, "__aiter__"):
        async for item in stream:  # type: ignore[union-attr]
            yield item
        return
    for item in stream:  # type: ignore[union-attr]
        yield item


def _arm_goals(core: Percept, goals: Any, *, source: str) -> None:
    """Arm every entry in ``goals`` on ``core``, in order.

    A bare string / :class:`Goal` runs through the EXISTING, untouched
    :func:`~percept.goalsugar.desugar_goals` -> :meth:`Percept.add_goal` path — no resolver, no new
    env read, byte-for-byte pre-ADR-008. A :class:`~percept.contracts.watchplan.WatchPlan` runs
    through :func:`percept.watchplan_apply.apply_watch_plan` instead (resolve + compile + coerce +
    arm). The two kinds may be freely mixed in one ``goals`` iterable; legacy entries are batched
    through ``desugar_goals`` between ``WatchPlan`` entries so relative order is preserved without
    calling ``desugar_goals`` once per item.
    """
    items = [goals] if isinstance(goals, (str, Goal, WatchPlan)) else list(goals)
    legacy_batch: list[Any] = []

    def _flush_legacy() -> None:
        if not legacy_batch:
            return
        for goal in desugar_goals(list(legacy_batch), source=source):
            core.add_goal(goal)
        legacy_batch.clear()

    for item in items:
        if isinstance(item, WatchPlan):
            _flush_legacy()
            _require_applied(apply_watch_plan(core, item), item)
        else:
            legacy_batch.append(item)
    _flush_legacy()


async def watch(
    stream: "AsyncIterable[Any] | Iterable[Any]",
    goals: "str | Goal | WatchPlan | Iterable[str | Goal | WatchPlan]",
    profile: Any = None,
    *,
    percept: Percept | None = None,
    source: str = "camera",
) -> AsyncIterator[WatchEvent]:
    """Watch ``stream`` (frames) for ``goals``, yielding one :class:`WatchEvent` per gate action.

    The frozen public entrypoint: every downstream capability — WatchProfiles, resilience/SLOs, the
    SDK front door — is built on this seam. It wires directly to
    the existing runtime (``Percept.create`` + ``Percept.perceive_judged``) with no detection behavior
    added or changed; a ``WatchProfile`` passed as ``profile`` IS resolved into runtime knobs (salience
    stat/thresholds, gate config, cadence, plan_mode) when ``watch()`` builds its own core (see
    the module docstring and :func:`_build_core`). The signature and streaming semantics here remain
    unchanged — this is additive wiring behind the existing seam, not a contract change.

    Args:
        stream: An (async or sync) iterable of frames, oldest to newest. A frame is whatever the
            configured ``vision`` backend accepts (an opaque per-backend type — e.g. a numpy array,
            a PIL image, or a backend-specific handle).
        goals: What to watch for; registered on the core before the first frame is consumed. Either
            :class:`~percept.adapters.Goal`\\ s, or — the one-liner form — bare plain-language
            condition strings: ``watch(stream, "the milk is boiling over")``. A bare string (or a
            list mixing strings and Goals) is desugared by :func:`percept.goalsugar.desugar_goals`
            into ``Goal(condition=<your exact string>, say=<your exact string>)`` with a
            deterministic, documented id. That desugaring is **pass-through, not compilation**: the
            string is never parsed, normalised, or mapped to a vocabulary — the judge sees the
            caller's own words, exactly as typed. A ``Goal`` passed in is used unchanged. A
            :class:`~percept.contracts.watchplan.WatchPlan` (ADR-008) is resolved and armed via
            :func:`percept.watchplan_apply.apply_watch_plan` (see the module docstring); strings,
            ``Goal``\\ s, and ``WatchPlan``\\ s may be freely mixed in one iterable.
        profile: OPTIONAL. ``None`` (the default) is today's ``Settings``-driven behavior — the
            documented default profile, BYTE-FOR-BYTE unchanged. A :class:`~percept.watchprofile.WatchProfile`
            is resolved for real; other shapes (a bare ``Settings``, an opaque
            ``.settings``-bearing object) keep their legacy pass-through. See the module docstring.
        percept: OPTIONAL, a pre-built :class:`Percept` core to reuse (e.g. one already carrying
            state across calls, or wired with specific backends). When omitted, one is created via
            :func:`_build_core` (applying any ``WatchProfile``-derived knobs at construction time).
        source: The frame source key; multi-camera setups pass distinct sources per ``watch()`` call.

    Yields:
        One :class:`WatchEvent` per FIRE or ASK the gate produces this tick, in frame order. A frame
        that changes nothing yields zero events (today's internal wiring never surfaces SILENT to
        callers — see :class:`WatchEvent`'s docstring for why the type still carries it).
    """
    core = percept if percept is not None else _build_core(profile)
    _arm_goals(core, goals, source=source)
    async for frame in _aiter(stream):
        now = core.clock()
        fired = await core.perceive_judged(frame, source=source, t=now)
        for ev in fired:
            yield _as_watch_event(ev, t=now)


def _require_applied(receipt: Any, plan: WatchPlan) -> None:
    """Raise at the public front door when a WatchPlan failed before arming."""
    if receipt.status == "rejected":
        detail = "; ".join(d.message for d in receipt.dropped_fields)
        raise ValueError(f"WatchPlan {plan.id or plan.condition!r} was rejected: {detail}")


__all__ = ["watch", "WATCH_VERSION"]
