"""The runtime worker — ``run(core, source) -> incidents``.

The IRA's #1 ask, made generic: own the camera→incidents lifecycle so apps stop writing the capture +
threading + feed glue. Per frame it drives the tier-0 :class:`~percept.edge.tier0.MotionGate`, and on
the rising edge spends a cloud judgment (``core.perceive_judged``), turning each fire into a durable
:class:`~percept_contracts.Incident` (with a pre-roll snippet) handed to the notification sink and yielded.

The **cognition is unchanged** — this *drives* ``vision.judge`` / the three-state gate, it doesn't edit
them. ``source`` is a URI string (file / RTSP / ``webcam:0``), opened on a capture thread, **or** an
iterable of ``(t, gate_frame, judge_frame)`` (tests / a pre-decoded clip).
"""
from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable

from percept.events import FrameWindow
from percept.gate import FIRE, ASK
from percept.incidents import make_incident, snippet_from_window


@dataclass
class RuntimeStats:
    """Mutable runtime counters for health dashboards and tests."""

    frames: int = 0
    escalations: int = 0
    judge_calls: int = 0
    fires: int = 0
    asks: int = 0
    incidents: int = 0
    sink_errors: int = 0


@dataclass(frozen=True)
class RuntimeHealthEvent:
    """Structured runtime health event emitted through ``on_health``."""

    kind: str
    ts: float | None = None
    data: dict[str, Any] | None = None


async def _aiter(source: Any, fps: float) -> AsyncIterator[tuple]:
    """Normalize ``source`` to an async iterator of ``(t, gate_frame, judge_frame)``."""
    if isinstance(source, str):
        from percept.runtime.source import open_source

        async for item in open_source(source, fps=fps):
            yield item
    elif hasattr(source, "__aiter__"):
        async for item in source:  # already an async iterator
            yield item
    else:  # a sync iterable (a pre-decoded clip / a test stream)
        for item in source:
            yield item
            await asyncio.sleep(0)  # cooperate with the event loop


async def _call_sink(fn: Callable, *args) -> None:
    result = fn(*args)
    if inspect.isawaitable(result):
        await result


async def _call_sink_with_policy(
    fn: Callable,
    *args,
    errors: str = "raise",
    stats: RuntimeStats | None = None,
    on_health: Callable | None = None,
    health_kind: str = "sink_error",
) -> None:
    try:
        await _call_sink(fn, *args)
    except Exception as exc:
        if stats is not None:
            stats.sink_errors += 1
        if on_health is not None:
            event = RuntimeHealthEvent(
                health_kind,
                data={"error": repr(exc), "sink": getattr(fn, "__name__", fn.__class__.__name__)},
            )
            await _call_sink(on_health, event)
        if errors == "ignore":
            return
        raise


async def run(
    core: Any,
    *,
    source: Any,
    on_incident: Callable | None = None,
    on_ask: Callable | None = None,
    on_frame: Callable | None = None,
    on_health: Callable | None = None,
    fps: float = 4.0,
    snippet_pre_s: float = 5.0,
    snippet_size: int = 48,
    snippet_kind: str = "frame",
    sink_errors: str = "raise",
    stats: RuntimeStats | None = None,
    gate_source: str = "camera",
) -> AsyncIterator:
    """Drive a live ``source`` through the gate + brain, yielding an :class:`Incident` on each rising edge.

    - ``on_incident(incident)`` — the notification sink (SMS / Slack / webhook lives here).
    - ``on_ask(event)`` — optional refusal sink for low-confidence reads; asks are not incidents.
    - ``on_frame(t, judge_frame)`` — the live-feed hook (wire an MJPEG streamer here).
    - ``on_health(event)`` — optional structured health/telemetry hook.
    - ``snippet_pre_s`` — seconds of pre-roll captured into each incident's snippet.
    - ``snippet_kind`` — ``"frame"`` keeps decoded frames; ``"clip"`` emits a portable encoded bundle.
    - ``sink_errors`` — ``"raise"`` (default) or ``"ignore"`` for notification/feed hook failures.
    - ``stats`` — optional mutable :class:`RuntimeStats` counters for health dashboards.

    Owns capture off the consumer thread; the per-goal cadence + the three-state gate decide *when* and
    *whether* to fire. Yields incidents lazily, so an ``async for incident in run(...)`` is the front door.
    """
    from percept.config import load_settings
    from percept.edge.tier0 import MotionGate, compile_watchspec

    settings = getattr(core, "settings", None) or load_settings()
    spec = compile_watchspec(list(core.goals.active()), threshold=settings.edge.motion_gate)
    gate = MotionGate(spec, settings=settings)
    clips = FrameWindow(size=snippet_size)  # rolling pre-roll buffer for the incident snippet

    if on_health is not None:
        await _call_sink(on_health, RuntimeHealthEvent("runtime_started", data={"source": repr(source)}))
    try:
        async for t, gate_frame, judge_frame in _aiter(source, fps):
            if stats is not None:
                stats.frames += 1
            clips.push(t, judge_frame)
            if on_frame is not None:
                await _call_sink_with_policy(
                    on_frame, t, judge_frame, errors=sink_errors, stats=stats, on_health=on_health, health_kind="frame_sink_error"
                )
            decision = gate.evaluate(gate_frame, ts=t)
            if not decision.escalate:
                continue  # still / sub-cadence -> no cloud call, no incident
            if stats is not None:
                stats.escalations += 1
                stats.judge_calls += 1
            if on_health is not None:
                await _call_sink(on_health, RuntimeHealthEvent("escalation", ts=t, data={"source": gate_source}))
            fires = await core.perceive_judged(judge_frame, gate_source, t=t)
            for fire in fires or []:
                if getattr(fire, "action", None) == ASK:
                    if stats is not None:
                        stats.asks += 1
                    if on_health is not None:
                        await _call_sink(on_health, RuntimeHealthEvent("ask", ts=t, data={"goal_id": fire.goal_id}))
                    if on_ask is not None:
                        await _call_sink_with_policy(
                            on_ask, fire, errors=sink_errors, stats=stats, on_health=on_health, health_kind="ask_sink_error"
                        )
                    continue
                if getattr(fire, "action", None) != FIRE:
                    continue
                if stats is not None:
                    stats.fires += 1
                snippet = snippet_from_window(clips, kind=snippet_kind, pre_s=snippet_pre_s)
                incident = make_incident(fire, ts=t, snippet=snippet)
                if on_health is not None:
                    await _call_sink(
                        on_health,
                        RuntimeHealthEvent("incident", ts=t, data={"incident_id": incident.id, "goal_id": incident.concern_id}),
                    )
                if on_incident is not None:
                    await _call_sink_with_policy(
                        on_incident,
                        incident,
                        errors=sink_errors,
                        stats=stats,
                        on_health=on_health,
                        health_kind="incident_sink_error",
                    )
                if stats is not None:
                    stats.incidents += 1
                yield incident
    except Exception as exc:
        if on_health is not None:
            await _call_sink(on_health, RuntimeHealthEvent("runtime_error", data={"error": repr(exc)}))
        raise
    finally:
        if on_health is not None:
            await _call_sink(on_health, RuntimeHealthEvent("runtime_ended"))
