"""The runtime worker — ``run(core, source) -> incidents``.

The IRA's #1 ask, made generic: own the camera→incidents lifecycle so apps stop writing the capture +
threading + feed glue. Per frame it drives the tier-0 :class:`~percept.edge.tier0.MotionGate`, and on
the rising edge spends a cloud judgment (``core.perceive_judged``), turning each fire into a durable
:class:`~percept.contracts.Incident` (with a pre-roll snippet) handed to the notification sink and yielded.

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


async def _aiter(source: Any, fps: float, **source_kwargs) -> AsyncIterator[tuple]:
    """Normalize ``source`` to an async iterator of ``(t, gate_frame, judge_frame)``."""
    if isinstance(source, str):
        from percept.runtime.source import open_source

        async for item in open_source(source, fps=fps, **source_kwargs):
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


# Cap so a tiny brief-window prior can't explode the capture rate (mirrors the bench
# _CAPTURE_HEADROOM_FPS): the fovea coast still drives JUDGING, but DECODE never exceeds this.
_BRIEF_CAPTURE_CAP_FPS = 8.0


def _capture_plan(goals: list, base_fps: float) -> tuple[float, Any]:
    """Derive ``(capture_fps, coast_controller)`` from active goals' sensor plans (TODO 1 / Build 5).

    A brief-action *judge* goal (one carrying a ``sense.brief_window_s`` prior) compiles a SENSITIVE
    above-Nyquist cadence via :func:`~percept.sensorplan.sensor_plan_for`; ordinary goals compile no
    cadence floor. So with no brief goal active this returns ``(base_fps, None)`` and the worker is
    byte-for-byte the fixed-fps default. When a brief goal IS active we (a) raise the capture rate to
    cover its cadence (capped) and (b) return a fovea :class:`~percept.sampling.SamplingController` whose
    coast floor forces a judge on quiet, sub-threshold frames the motion gate would skip — so a
    sub-second window is still covered. The coast tracks the MOST sensitive active goal."""
    from percept.sampling import SamplingController, SensePolicy
    from percept.sensorplan import sensor_plan_for

    capture = float(base_fps)
    policy: SensePolicy | None = None
    for g in goals:
        try:
            cad = sensor_plan_for(g).cadence
        except Exception:
            continue  # a goal whose sense doesn't compile a plan never raises the floor
        base = cad.get("base_fps")
        if not base:
            continue  # ordinary (non-brief) goal: no cadence floor
        mx = cad.get("max_fps") or base
        capture = max(capture, min(_BRIEF_CAPTURE_CAP_FPS, float(mx)))
        et = cad.get("escalate_threshold")
        pol = SensePolicy(
            base_fps=float(base), max_fps=float(mx),
            escalate_threshold=float(et) if et is not None else SensePolicy().escalate_threshold,
        )
        if policy is None or pol.base_fps > policy.base_fps:
            policy = pol
    return capture, (SamplingController(policy) if policy is not None else None)


async def run(
    core: Any,
    *,
    source: Any,
    on_incident: Callable | None = None,
    on_ask: Callable | None = None,
    on_frame: Callable | None = None,
    on_health: Callable | None = None,
    fps: float = 4.0,
    source_queue_size: int = 64,
    source_reconnects: int = 0,
    source_reconnect_backoff_s: float = 0.5,
    source_rtsp_tcp: bool = True,
    snippet_pre_s: float = 5.0,
    snippet_size: int = 48,
    snippet_kind: str = "frame",
    sink_errors: str = "raise",
    stats: RuntimeStats | None = None,
    gate_source: str = "camera",
    episode_inactivity_s: float | None = None,
) -> AsyncIterator:
    """Drive a live ``source`` through the gate + brain, yielding an :class:`Incident` on each rising edge.

    - ``on_incident(incident)`` — the notification sink (SMS / Slack / webhook lives here).
    - ``on_ask(event)`` — optional refusal sink for low-confidence reads; asks are not incidents.
    - ``on_frame(t, judge_frame)`` — the live-feed hook (wire an MJPEG streamer here).
    - ``on_health(event)`` — optional structured health/telemetry hook.
    - ``source_*`` — capture queue/reconnect/RTSP options for URI sources.
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
    # TODO 1 / Build 5: a brief-action goal raises the capture rate + arms a fovea coast so a sub-second
    # window is covered above its Nyquist; no brief goal -> (fps, None) -> byte-for-byte the fixed-fps path.
    capture_fps, coast = _capture_plan(list(core.goals.active()), fps)
    clips = FrameWindow(size=snippet_size)  # rolling pre-roll buffer for the incident snippet

    if on_health is not None:
        await _call_sink(on_health, RuntimeHealthEvent("runtime_started", data={"source": repr(source)}))
    try:
        source_kwargs = {
            "maxsize": source_queue_size,
            "reconnects": source_reconnects,
            "reconnect_backoff_s": source_reconnect_backoff_s,
            "rtsp_tcp": source_rtsp_tcp,
        }
        # Item 4 episode boundary: a process activity ends precisely on QUIET frames (no escalation), so the
        # boundary check rides the per-frame cadence — only spending a summarize call when it actually fires.
        check_boundary = getattr(core, "check_episode_boundary", None)

        async def _emit_fires(fires, *, at):
            """Turn a list of FireEvents (from perceive_judged OR the boundary) into incidents/asks."""
            for fire in fires or []:
                if getattr(fire, "action", None) == ASK:
                    if stats is not None:
                        stats.asks += 1
                    if on_health is not None:
                        await _call_sink(on_health, RuntimeHealthEvent("ask", ts=at, data={"goal_id": fire.goal_id}))
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
                incident = make_incident(fire, ts=at, snippet=snippet)
                if on_health is not None:
                    await _call_sink(
                        on_health,
                        RuntimeHealthEvent("incident", ts=at, data={"incident_id": incident.id, "goal_id": incident.concern_id}),
                    )
                if on_incident is not None:
                    await _call_sink_with_policy(
                        on_incident, incident, errors=sink_errors, stats=stats, on_health=on_health, health_kind="incident_sink_error",
                    )
                if stats is not None:
                    stats.incidents += 1
                yield incident

        async for t, gate_frame, judge_frame in _aiter(source, capture_fps, **source_kwargs):
            if stats is not None:
                stats.frames += 1
            clips.push(t, judge_frame)
            if on_frame is not None:
                await _call_sink_with_policy(
                    on_frame, t, judge_frame, errors=sink_errors, stats=stats, on_health=on_health, health_kind="frame_sink_error"
                )

            # boundary first: it must see quiet frames too (the episode ends when activity stops)
            if check_boundary is not None:
                async for incident in _emit_fires(await check_boundary(t=t, inactivity_s=episode_inactivity_s), at=t):
                    yield incident

            decision = gate.evaluate(gate_frame, ts=t)
            # TODO 1: a brief-action goal's fovea coast forces a judge on a quiet, in-window frame the
            # motion gate would skip (above-Nyquist coverage). No brief goal -> coast is None -> this is
            # byte-for-byte the gate-only path.
            coast_admit = bool(coast and coast.should_process("__coast__", t, float(decision.signal.motion)))
            if not (decision.escalate or coast_admit):
                continue  # still / sub-cadence -> no cloud call, no incident
            if stats is not None:
                stats.escalations += 1
                stats.judge_calls += 1
            if on_health is not None:
                await _call_sink(on_health, RuntimeHealthEvent("escalation", ts=t, data={"source": gate_source}))
            async for incident in _emit_fires(await core.perceive_judged(judge_frame, gate_source, t=t), at=t):
                yield incident
    except Exception as exc:
        if on_health is not None:
            await _call_sink(on_health, RuntimeHealthEvent("runtime_error", data={"error": repr(exc)}))
        raise
    finally:
        if on_health is not None:
            await _call_sink(on_health, RuntimeHealthEvent("runtime_ended"))
