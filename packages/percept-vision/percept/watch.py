"""The public ``watch()`` entrypoint — the M1 frozen seam every downstream MVP card depends on.

``watch(stream, goals, profile=None) -> AsyncIterator[WatchEvent]`` is the ONE stable, versioned way
to run the existing fovea -> judge -> gate path (:meth:`~percept.agent.Percept.perceive_judged`) over a
stream of frames. It is a thin wrapper, not a new firing path: **no detection behavior is hardcoded
here** — this module adds zero judge/gate logic, it only adapts the existing per-frame loop to a
stream-in/stream-out shape and maps its output to the frozen :class:`~percept.contracts.WatchEvent`.

``profile`` is deliberately **opaque** today. Per the implementation plan (E2), a future ``WatchProfile``
(P1) will bind (environment, goal-type, state) to the runtime's knobs (``salience_stat``, ``plan_mode``,
cadence, gate config, ...) — the free variables of ``Q(p | environment, goal, state)``. This card does
not build that resolution; it only reserves the seam. The documented default is ``profile=None`` ->
today's :class:`~percept.config.Settings`-driven behavior (:func:`~percept.config.load_settings`),
unchanged. See :func:`_settings_from_profile` for the one (deliberately minimal) pass-through rule.
"""

from __future__ import annotations

from typing import Any, AsyncIterable, AsyncIterator, Iterable

from .adapters import Goal
from .agent import FireEvent, Percept
from .config import Settings
from .contracts import WatchEvent

#: Semver of the ``watch()`` entrypoint contract itself — the function signature + streaming
#: semantics — distinct from ``percept.contracts.CONTRACT_VERSION`` (the DTO wire shapes it returns).
#: Bump MAJOR on a breaking change to the signature or streaming contract; MINOR for an additive
#: (new optional kwarg); PATCH otherwise.
WATCH_VERSION = "1.0.0"


def _settings_from_profile(profile: Any) -> Settings | None:
    """Resolve ``profile`` to a :class:`Settings` override, or ``None`` for the default.

    ``profile`` is OPAQUE today — P1 has not shipped the real ``WatchProfile`` yet, and this card makes
    no resolution decisions (that is P1's job). Two minimal, non-behavior-changing pass-throughs so
    early callers aren't blocked: a bare ``Settings`` instance is used directly (handy for tests/callers
    that already build one), and any other object's ``.settings`` attribute is read if present. Anything
    else (including ``None``) resolves to ``None``, so :meth:`Percept.create` falls back to
    ``load_settings()`` — today's unchanged default. This function is the one seam P1 will replace with
    real ``(environment, goal, state) -> knobs`` resolution.
    """
    if profile is None:
        return None
    if isinstance(profile, Settings):
        return profile
    return getattr(profile, "settings", None)


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


async def watch(
    stream: "AsyncIterable[Any] | Iterable[Any]",
    goals: list[Goal],
    profile: Any = None,
    *,
    percept: Percept | None = None,
    source: str = "camera",
) -> AsyncIterator[WatchEvent]:
    """Watch ``stream`` (frames) for ``goals``, yielding one :class:`WatchEvent` per gate action.

    The frozen public entrypoint (implementation plan E1/M1): every downstream card — WatchProfiles
    (E2), resilience/SLOs (E1), the SDK front door (E6) — is built on this seam. It wires directly to
    the existing runtime (``Percept.create`` + ``Percept.perceive_judged``) with no detection behavior
    added or changed; ``profile`` is accepted and threaded through but not yet resolved into runtime
    knobs (see the module docstring and :func:`_settings_from_profile`).

    Args:
        stream: An (async or sync) iterable of frames, oldest to newest. A frame is whatever the
            configured ``vision`` backend accepts (an opaque per-backend type — e.g. a numpy array,
            a PIL image, or a backend-specific handle).
        goals: The :class:`~percept.adapters.Goal`\\ s to watch; registered on the core before the
            first frame is consumed.
        profile: OPTIONAL, opaque. ``None`` (the default) is today's ``Settings``-driven behavior —
            the documented default profile. See the module docstring.
        percept: OPTIONAL, a pre-built :class:`Percept` core to reuse (e.g. one already carrying
            state across calls, or wired with specific backends). When omitted, one is created via
            ``Percept.create()`` (using ``profile``-derived settings when available).
        source: The frame source key; multi-camera setups pass distinct sources per ``watch()`` call.

    Yields:
        One :class:`WatchEvent` per FIRE or ASK the gate produces this tick, in frame order. A frame
        that changes nothing yields zero events (today's internal wiring never surfaces SILENT to
        callers — see :class:`WatchEvent`'s docstring for why the type still carries it).
    """
    core = percept if percept is not None else Percept.create(settings=_settings_from_profile(profile))
    for goal in goals:
        core.add_goal(goal)
    async for frame in _aiter(stream):
        now = core.clock()
        fired = await core.perceive_judged(frame, source=source, t=now)
        for ev in fired:
            yield _as_watch_event(ev, t=now)


__all__ = ["watch", "WATCH_VERSION"]
