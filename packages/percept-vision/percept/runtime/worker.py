"""The runtime worker — ``run(core, source) -> incidents``.

The IRA's #1 ask, made generic: own the camera→incidents lifecycle so apps stop writing the capture +
threading + feed glue. Per frame it drives the tier-0 :class:`~percept_harness.tier0.MotionGate`, and on
the rising edge spends a cloud judgment (``core.perceive_judged``), turning each fire into a durable
:class:`~percept_contracts.Incident` (with a pre-roll snippet) handed to the notification sink and yielded.

The **cognition is unchanged** — this *drives* ``vision.judge`` / the three-state gate, it doesn't edit
them. ``source`` is a URI string (file / RTSP / ``webcam:0``), opened on a capture thread, **or** an
iterable of ``(t, gate_frame, judge_frame)`` (tests / a pre-decoded clip).
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Callable

from percept.events import FrameWindow
from percept.incidents import make_incident, snippet_from_window


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


async def run(
    core: Any,
    *,
    source: Any,
    on_incident: Callable | None = None,
    on_frame: Callable | None = None,
    fps: float = 4.0,
    snippet_pre_s: float = 5.0,
    snippet_size: int = 48,
    gate_source: str = "camera",
) -> AsyncIterator:
    """Drive a live ``source`` through the gate + brain, yielding an :class:`Incident` on each rising edge.

    - ``on_incident(incident)`` — the notification sink (SMS / Slack / webhook lives here).
    - ``on_frame(t, judge_frame)`` — the live-feed hook (wire an MJPEG streamer here).
    - ``snippet_pre_s`` — seconds of pre-roll captured into each incident's snippet.

    Owns capture off the consumer thread; the per-goal cadence + the three-state gate decide *when* and
    *whether* to fire. Yields incidents lazily, so an ``async for incident in run(...)`` is the front door.
    """
    from percept.config import load_settings
    from percept_harness.tier0 import MotionGate, compile_watchspec

    settings = getattr(core, "settings", None) or load_settings()
    spec = compile_watchspec(list(core.goals.active()), threshold=settings.edge.motion_gate)
    gate = MotionGate(spec, settings=settings)
    clips = FrameWindow(size=snippet_size)  # rolling pre-roll buffer for the incident snippet

    async for t, gate_frame, judge_frame in _aiter(source, fps):
        clips.push(t, judge_frame)
        if on_frame is not None:
            on_frame(t, judge_frame)
        if not gate.evaluate(gate_frame, ts=t).escalate:
            continue  # still / sub-cadence -> no cloud call, no incident
        fires = await core.perceive_judged(judge_frame, gate_source, t=t)
        for fire in fires or []:
            snippet = snippet_from_window(clips, pre_s=snippet_pre_s)
            incident = make_incident(fire, ts=t, snippet=snippet)
            if on_incident is not None:
                on_incident(incident)
            yield incident
