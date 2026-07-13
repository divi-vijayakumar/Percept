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
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable

from percept.events import FrameWindow
from percept.gate import FIRE, ASK
from percept.incidents import make_incident, snippet_from_window
from percept.observability import get_logger

_log = get_logger("percept.runtime.worker")


@dataclass
class RuntimeStats:
    """Mutable runtime counters for health dashboards and tests."""

    frames: int = 0
    frames_dropped: int = 0  # frames the bounded capture queue discarded (slow consumer)
    escalations: int = 0
    judge_calls: int = 0
    fires: int = 0
    asks: int = 0
    incidents: int = 0
    sink_errors: int = 0
    audio_chunks: int = 0  # audio chunks pumped into the AudioWindow (fused/acoustic perception)
    audio_wakes: int = 0  # audio onsets that admitted a frame the visual gate would have skipped
    boundaries: int = 0  # edge-detector boundaries (rep peaks / onsets) fed to perceive_boundary
    utterances: int = 0  # complete spoken utterances endpointed and handed to on_utterance


@dataclass(frozen=True)
class RuntimeHealthEvent:
    """Structured runtime health event emitted through ``on_health``."""

    kind: str
    ts: float | None = None
    data: dict[str, Any] | None = None


def _log_disposition(log: Any, *, incident_id: str, fire: Any, disposition: str, ts: float) -> None:
    """Append one gate disposition to the :class:`~percept.labelstore.DispositionLog` (§4.3).

    Best-effort: a disposition-log IO error must never kill the watch (it is telemetry, not the trust
    surface). ``incident_id`` is ``""`` for an ASK (asks are not incidents)."""
    from percept.labelstore import verdict_ref as _vr

    try:
        log.append(
            incident_id=incident_id,
            verdict_ref=_vr(getattr(fire, "verdict", None)),
            disposition=disposition,
            ts=float(ts),
        )
    except Exception:  # noqa: BLE001 - telemetry write must not crash the run
        pass


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


async def _audio_aiter(source: Any, **source_kwargs) -> AsyncIterator[tuple]:
    """Normalize an audio ``source`` to an async iterator of ``(t, chunk, energy)`` — the audio twin
    of :func:`_aiter`. A ``str`` is a mic/wav URI; anything else is an (async) iterable (tests/replay)."""
    if isinstance(source, str):
        from percept.runtime.audio import open_audio_source

        async for item in open_audio_source(source, **source_kwargs):
            yield item
    elif hasattr(source, "__aiter__"):
        async for item in source:
            yield item
    else:
        for item in source:
            yield item
            await asyncio.sleep(0)


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


def _utterance_clip(core: Any, source: str, dur_s: float) -> Any:
    """Assemble ONE audio clip covering the last ``dur_s`` seconds of the source's AudioWindow — the
    endpointed utterance handed to ``on_utterance``. The window holds typed :class:`AudioFrame` chunks
    (from :func:`percept.runtime.audio.open_audio_source`), so we concatenate their PCM into one frame;
    for a raw-bytes/tuple test source we fall back to :func:`percept.agent._concat_audio`. Returns
    ``None`` when the window is empty (nothing to send)."""
    from percept.agent import _concat_audio

    try:
        chunks = core._audio_window(source).clip(max(0.0, dur_s))  # noqa: SLF001 - worker drives core internals
    except Exception:  # noqa: BLE001 - a missing/odd window must not kill the pump
        return None
    if not chunks:
        return None
    pcm_frames = [c for c in chunks if hasattr(c, "pcm")]
    if not pcm_frames:
        return _concat_audio(chunks)  # raw bytes / (mime, bytes) test source
    from percept.contracts import AudioFrame

    first = pcm_frames[0]
    return AudioFrame(
        pcm=b"".join(bytes(f.pcm) for f in pcm_frames),
        sample_rate=first.sample_rate,
        channels=first.channels,
        sample_format=first.sample_format,
    )


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


def _as_goal(g: Any, i: int) -> Any:
    """Coerce a convenience `goals=[...]` entry into a Goal: a plain condition string, a dict of Goal
    fields, or a Goal passthrough. A string `"the milk is boiling over"` -> Goal(id=f"g{i}", condition=,
    say=) so the one-liner `percept.run(source=, goals=["..."])` needs no Goal import."""
    from percept.adapters import Goal

    if isinstance(g, Goal):
        return g
    if isinstance(g, str):
        return Goal(id=f"g{i}", condition=g, say=g)
    if isinstance(g, dict):
        d = dict(g)
        d.setdefault("id", f"g{i}")
        d.setdefault("say", d.get("condition", ""))
        return Goal(**d)
    raise TypeError(f"goals[{i}] must be a str, dict, or Goal, not {type(g).__name__}")


async def run(
    core: Any = None,
    *,
    source: Any,
    goals: Any = None,
    vision: Any = None,
    on_incident: Callable | None = None,
    on_ask: Callable | None = None,
    on_frame: Callable | None = None,
    on_health: Callable | None = None,
    on_trace: Callable | None = None,
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
    coverage_staleness_s: float | None = None,
    audio_source: Any = None,
    audio_gate: Any = "acoustic-onset",
    audio_wake: dict | None = None,
    on_utterance: Callable | None = None,
    vad: Any = None,
    utterance_threshold: float = 0.01,
    utterance_hangover_frames: int = 8,
    utterance_min_speech_frames: int = 3,
    deaf_until: Callable[[], float] | None = None,
    stale_audio_s: float | None = None,
    wall_clock: Callable[[], float] = time.monotonic,
    wake_policy: Any = None,
    regate_on_goal_change: bool = True,
    paused: Callable | None = None,
    muted: Callable | None = None,
    clock: Callable | None = None,
    persistence: Any = None,
    restore: bool = False,
    autosave_every_s: float | None = None,
    label_store: Any = None,
    disposition_log: Any = None,
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
    - ``audio_source`` — optional mic/wav URI (``"mic:0"``) **or** an (async) iterable of
      ``(t, chunk, energy)`` (see :func:`percept.runtime.audio.open_audio_source`). When set, the worker
      pumps each chunk into ``core.push_audio`` — **filling the ``AudioWindow`` that ``modality="fused"/
      "audio"`` goals judge over** (otherwise it stays empty and fused judges degrade to vision-only) —
      and feeds ``energy`` to the gate, so an off-camera sound (a cry, glass breaking) can escalate a
      frame the visual motion gate would skip. Default ``None`` → the video-only path is byte-for-byte.
    - ``audio_gate`` — the tier-0 audio detector: a name (``"acoustic-onset"``), an instance with
      ``push(t, v) -> [Boundary]``, or ``None`` to only fill the window + feed energy (no onset wake).
    - ``audio_wake`` — kwargs for the onset detector, e.g. ``{"high": 0.3, "low": 0.15, "refractory_s": 5.0}``.
    - ``on_utterance(audio, t)`` — the **voice-escalation seam**. When the audio pump endpoints a complete
      spoken utterance (energy rose, then fell for ``utterance_hangover_frames``), the utterance's PCM clip
      (an :class:`~percept.contracts.AudioFrame`) is handed here — the hook the voice layer turns into an
      ephemeral reply. Percept owns the mic + the "when is a turn done" decision; the app owns the reply.
      Default ``None`` → no endpointing, the audio path is byte-for-byte. ``vad`` overrides the endpointer
      (any object with ``push(energy) -> event.{started,endpoint}``); else an :class:`EnergyVad` built from
      the ``utterance_*`` knobs is used.

      **The handler runs OFF the pump** (spawned as a tracked task, cancelled on exit): a handler that
      takes real time — STT + a voice turn + playback — must not block capture, or the backlog it creates
      is replayed through the endpointer the instant it returns and Percept endpoints its own reply
      (self-echo). Two utterances can therefore overlap; the runtime does NOT serialize them — the app
      owns concurrency (guard at capture time, e.g. a ``speaking`` flag). ``on_utterance`` opens an
      **un-AEC'd** mic and hands raw PCM; if the app plays audio into the same room it MUST call
      ``deaf_until``/``session.deafen(...)`` around playback or the mic will hear it. For a speakers-on,
      always-on deployment prefer ``attach_transport`` (the transport owns AEC).
    - ``deaf_until`` — optional ``() -> deadline`` (in the ``wall_clock`` timebase). While
      ``wall_clock() < deaf_until()`` the endpointer is suppressed (the app is about to / is speaking):
      chunks still fill the AudioWindow (fused judging is unaffected) but are not fed to the VAD, and the
      VAD is reset so a half-built utterance cannot straddle the window. Turn-taking, not perception —
      checked immediately before the VAD, never above ``push_audio``. Default ``None`` → never deafened.
    - ``stale_audio_s`` — optional staleness guard for the VAD path. When the pump has fallen behind
      real time (a slow turn outran the drop-oldest capture queue) draining the backlog would replay
      minutes-old audio through the endpointer as if live. When pump lag (wall-elapsed minus stream-elapsed
      since the first chunk) exceeds this, the chunk is dropped from the VAD path and a
      ``RuntimeHealthEvent("audio_stale")`` is emitted — it is still ``push_audio``'d (the AudioWindow
      wants the history). Default ``None`` → off (byte-for-byte).
    - ``wall_clock`` — monotonic time source for ``deaf_until`` + ``stale_audio_s`` (default
      :func:`time.monotonic`; injectable for tests). Distinct from ``clock`` (civil time for schedules).

    - ``label_store`` — optional :class:`~percept.labelstore.LabelStore` (L-1). When set, every yielded
      incident is bound to it, so ``incident.record_label(good, ...)`` writes a LOCAL
      :class:`~percept.contracts.LabelRecord` (the flywheel intake). Default ``None`` → ``record_label``
      raises (nothing bound); the firing path is byte-for-byte either way.
    - ``disposition_log`` — optional :class:`~percept.labelstore.DispositionLog` (§4.3). When set, every
      gate ``FIRE``/``ASK`` appends a ``{incident_id, verdict_ref, disposition, ts}`` row (the escalation
      meter's raw feed, distinct from ``Verdict.usage``). Default ``None`` → nothing written, byte-for-byte.

    Owns capture off the consumer thread; the per-goal cadence + the three-state gate decide *when* and
    *whether* to fire. Yields incidents lazily, so an ``async for incident in run(...)`` is the front door.
    """
    from percept.labelstore import verdict_ref as _verdict_ref
    from percept.config import load_settings
    from percept.edge.detectors import DetectorBank
    from percept.edge.tier0 import MotionGate, compile_watchspec

    # R-0 convenience: `percept.run(source=..., goals=[...])` with no pre-built core. Build a Percept
    # here and arm each goal, so the one-line quickstart is real. Passing an explicit `core` (the
    # canonical form) is unchanged and takes precedence — `goals`/`vision` are ignored when core is given.
    if core is None:
        if not goals:
            raise TypeError(
                "percept.run needs either a core (positional) or goals=[...]; got neither"
            )
        from percept.agent import Percept

        core = Percept.create(vision=vision) if vision is not None else Percept.create()
        for i, g in enumerate(goals):
            core.add_goal(_as_goal(g, i))

    settings = getattr(core, "settings", None) or load_settings()

    # PR 2.2: restart durability. Rehydrate goals/entities/action-log from a persistence store BEFORE
    # compiling the gate, so a restarted 24/7 watch resumes its armed goals + refractory state instead
    # of re-arming edges (which would double-fire). Default (persistence=None) is byte-for-byte.
    if persistence is not None and restore:
        load = getattr(core, "load_state", None)
        if load is not None:
            load(persistence)

    def _build_gate():
        """(Re)compile the WatchSpec → gate + detector banks from the core's CURRENT active goals.
        Called once at start, and again on a goal-registry revision change when regate_on_goal_change
        is set (so a goal armed mid-run — e.g. by voice — actually reaches the tier-0 gate)."""
        s = compile_watchspec(
            list(core.goals.active()),
            threshold=settings.edge.motion_gate,
            # Deployment-wide salience-mode defaults (per-goal goal.state may override); resolved onto
            # each WatchCondition so the gate/edge honor the effective value.
            salience_stat=settings.edge.salience_stat,
            ego_compensate=settings.edge.ego_compensate,
            # Per-stat base cutoffs — block_max/p95 need a higher threshold than the mean-tuned default.
            stat_thresholds={
                "block_max": settings.edge.motion_gate_block_max,
                "p95": settings.edge.motion_gate_p95,
            },
        )
        # clock (PR 2.2): a civil-time source arms per-goal Schedules (C1); None -> always-armed (default).
        g = MotionGate(s, settings=settings, wake_policy=wake_policy, clock=clock)
        # PR 1.3: one detector bank per goal that selected edge detectors (state['detectors']). The bank
        # emits timed boundaries (rep peaks, acoustic onsets) from the CHEAP motion/energy signals at
        # native fps — never spends a cloud judge. No detectors -> no banks -> byte-for-byte pre-PR.
        b = {c.goal_id: DetectorBank(c.detectors) for c in s.conditions if getattr(c, "detectors", None)}
        return s, g, b

    spec, gate, banks = _build_gate()
    perceive_boundary = getattr(core, "perceive_boundary", None)
    goals_revision = getattr(core.goals, "revision", None)  # PR 2.1: watch for mid-run goal changes
    # TODO 1 / Build 5: a brief-action goal raises the capture rate + arms a fovea coast so a sub-second
    # window is covered above its Nyquist; no brief goal -> (fps, None) -> byte-for-byte the fixed-fps path.
    capture_fps, coast = _capture_plan(list(core.goals.active()), fps)
    clips = FrameWindow(size=snippet_size)  # rolling pre-roll buffer for the incident snippet

    # This run's tracer takes precedence; a tracer set via Percept.create(on_trace=) is preserved when
    # run() gets none. Set immediately before the try so the finally ALWAYS restores it — a tracer never
    # leaks into the next run(core), even if the body below raises. (Setup above runs first, so a setup
    # failure happens before the tracer is wired and leaks nothing.)
    _prev_on_trace = getattr(core, "on_trace", None)
    if on_trace is not None:
        core.on_trace = on_trace  # the per-frame decision trace flows out of perceive_judged
    audio_task = None  # bound before try so the finally can always cancel it, even on setup failure
    autosave_task = None
    # R1: in-flight on_utterance handlers, spawned OFF the pump. Tracked so the finally can cancel them —
    # a voice turn must not outlive the run (a dangling STT/Live turn after stop() is a leak).
    _utt_tasks: set = set()

    def _utt_done(task: "asyncio.Task") -> None:
        # Discard from the set AND retrieve the result: a raising handler (sink_errors="raise") re-raises
        # inside this off-pump task; without retrieval asyncio logs "exception never retrieved". The error
        # was already surfaced via on_health in _call_sink_with_policy — the pump stays alive regardless.
        _utt_tasks.discard(task)
        if not task.cancelled():
            task.exception()

    try:
        if on_health is not None:
            await _call_sink(on_health, RuntimeHealthEvent("runtime_started", data={"source": repr(source)}))
        dropped: list[int] = [0]  # shared with the capture thread; surfaced as stats.frames_dropped
        source_health: list = []  # A-4: capture thread appends (kind, data); drained → on_health below
        source_kwargs = {
            "maxsize": source_queue_size,
            "reconnects": source_reconnects,
            "reconnect_backoff_s": source_reconnect_backoff_s,
            "rtsp_tcp": source_rtsp_tcp,
            "dropped": dropped,
            "health": source_health,
        }
        # Item 4 episode boundary: a process activity ends precisely on QUIET frames (no escalation), so the
        # boundary check rides the per-frame cadence — only spending a summarize call when it actually fires.
        check_boundary = getattr(core, "check_episode_boundary", None)
        # PR 1.4: the absence-native differentiators, driven every frame (they fire on QUIET frames the
        # gate skips). Both are perception-robust (fire on the LACK of evidence, never a guess) and opt-in
        # per concern — check_observability only for miss_cost=='severe' ("I can't see the baby"), and
        # check_deadlines only for rearm=='deadline' ("overdue"). Default concerns -> both silent,
        # byte-for-byte. Sync (unlike the async boundary check).
        check_coverage = getattr(core, "check_observability", None)
        check_deadlines = getattr(core, "check_deadlines", None)

        # --- autosave (PR 2.2): periodically persist state so a crash/restart resumes, not restarts. ----
        if persistence is not None and autosave_every_s and getattr(core, "save_state", None):
            async def _autosave():
                while True:
                    await asyncio.sleep(autosave_every_s)
                    try:
                        core.save_state(persistence)
                        if on_health is not None:
                            await _call_sink(on_health, RuntimeHealthEvent("state_saved"))
                    except Exception as exc:  # noqa: BLE001 - a save failure must not kill the watch
                        if on_health is not None:
                            await _call_sink(on_health, RuntimeHealthEvent("state_save_error", data={"error": repr(exc)}))
            autosave_task = asyncio.ensure_future(_autosave())

        # --- audio ingress (PR 1.2): fill the AudioWindow + feed the gate an energy/onset signal --------
        # `audio` holds the latest energy and a one-shot onset-wake flag, shared with the pump task below
        # (same event loop, cooperative — no lock). Default (audio_source=None) leaves it inert: energy
        # stays None and wake stays False, so gate.evaluate and the admit test are byte-for-byte the
        # video-only path.
        audio = {"energy": None, "wake": False}
        audio_task = None
        if audio_source is not None:
            onset = None
            if audio_gate is not None:
                if isinstance(audio_gate, str):
                    from percept.edge.detectors import AcousticOnset
                    onset = AcousticOnset(**(audio_wake or {})) if audio_gate == "acoustic-onset" else None
                else:
                    onset = audio_gate  # a caller-supplied detector instance

            # Voice-escalation endpointer (PR 3.x): only built when on_utterance is wired — default None
            # leaves the audio path byte-for-byte. `vad` can be any object with push(energy)->ev.endpoint.
            _vad = None
            _utt = {"start": None}  # start time of the utterance currently being spoken
            _wall_anchor: list = [None]  # (wall0, stream0) at the first VAD chunk — R4 lag reference
            if on_utterance is not None:
                if vad is not None:
                    _vad = vad
                else:
                    from percept.runtime._vad import EnergyVad
                    _vad = EnergyVad(
                        threshold=utterance_threshold,
                        hangover_frames=utterance_hangover_frames,
                        min_speech_frames=utterance_min_speech_frames,
                    )

            async def _pump_audio():
                # Pump every chunk into the AudioWindow (un-strands modality="fused"/"audio") and track
                # energy + onsets. Never lets a bad audio chunk kill the run — the watch survives a mic drop.
                async for at, chunk, energy in _audio_aiter(audio_source):
                    # MUTED (RuntimeSession.detach_audio_source): keep draining the mic stream but stop
                    # FEEDING it to perception — the fused judge goes video-only and the energy/onset gate
                    # signals freeze, without ending the session. Default (muted=None) is byte-for-byte.
                    if muted is not None and muted():
                        continue
                    try:
                        core.push_audio(chunk, source=gate_source, t=at)
                    except Exception:  # noqa: BLE001 - audio ingress must not crash the video watch
                        pass
                    if stats is not None:
                        stats.audio_chunks += 1
                    audio["energy"] = float(energy) if energy is not None else audio["energy"]
                    if onset is not None and energy is not None:
                        for b in onset.push(at, float(energy)):
                            if getattr(b, "kind", None) == "onset":
                                audio["wake"] = True

                    # Voice escalation: endpoint a complete utterance and hand its clip to on_utterance.
                    # Runs AFTER push_audio so the just-heard chunk is already in the window we clip from.
                    if _vad is not None and energy is not None:
                        # R2/R4: suppress the endpointer while the app is about to speak (deafen) or while
                        # the pump is replaying a stale capture backlog (staleness). The chunk still filled
                        # the AudioWindow above — this is turn-taking, not perception. Reset the VAD when
                        # suppressed so a half-built utterance cannot straddle (survive) the window.
                        suppress = False
                        if stale_audio_s is not None or deaf_until is not None:
                            wnow = wall_clock()
                            if stale_audio_s is not None:
                                if _wall_anchor[0] is None:
                                    _wall_anchor[0] = (wnow, at)
                                w0, s0 = _wall_anchor[0]
                                if (wnow - w0) - (at - s0) > stale_audio_s:  # pump lag vs real time
                                    suppress = True
                                    if on_health is not None:
                                        await _call_sink(on_health, RuntimeHealthEvent(
                                            "audio_stale", ts=at, data={"lag_s": (wnow - w0) - (at - s0)}))
                            if deaf_until is not None and wnow < deaf_until():
                                suppress = True
                        if suppress:
                            _vad.reset()
                            _utt["start"] = None
                        else:
                            ev = _vad.push(float(energy))
                            if getattr(ev, "started", False):
                                _utt["start"] = at
                            if getattr(ev, "endpoint", False):
                                start_t = _utt["start"] if _utt["start"] is not None else at
                                _utt["start"] = None
                                clip = _utterance_clip(core, gate_source, at - start_t)
                                if clip is not None:
                                    if stats is not None:
                                        stats.utterances += 1
                                    if on_health is not None:
                                        await _call_sink(on_health, RuntimeHealthEvent(
                                            "utterance", ts=at, data={"dur_s": at - start_t}))
                                    # R1: hand off OFF the pump — a real handler (STT + voice turn + playback)
                                    # blocking here guarantees a capture backlog replayed as self-echo the
                                    # instant it returns. The app owns concurrency (overlaps are possible;
                                    # the runtime does not serialize). Errors still route through the policy.
                                    task = asyncio.ensure_future(_call_sink_with_policy(
                                        on_utterance, clip, at, errors=sink_errors, stats=stats,
                                        on_health=on_health, health_kind="utterance_sink_error",
                                    ))
                                    _utt_tasks.add(task)
                                    task.add_done_callback(_utt_done)

            audio_task = asyncio.ensure_future(_pump_audio())

        async def _emit_fires(fires, *, at, from_boundary=False):
            """Turn a list of FireEvents (from perceive_judged OR the boundary) into incidents/asks."""
            for fire in fires or []:
                if getattr(fire, "action", None) == ASK:
                    if stats is not None:
                        stats.asks += 1
                    if disposition_log is not None:  # §4.3: escalation-meter raw feed (asks are not incidents)
                        _log_disposition(disposition_log, incident_id="", fire=fire, disposition="ask", ts=at)
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
                if label_store is not None:  # L-1: bind the local label sink so incident.record_label() writes
                    incident.bind_labeler(
                        label_store.append,
                        verdict_ref=_verdict_ref(getattr(fire, "verdict", None)),
                        goal_id=getattr(fire, "goal_id", None),
                    )
                if disposition_log is not None:  # §4.3: escalation-meter raw feed
                    _log_disposition(disposition_log, incident_id=incident.id, fire=fire, disposition="fire", ts=at)
                emit_trace = getattr(core, "_emit_trace", None)
                if getattr(core, "on_trace", None) is not None and emit_trace is not None:
                    # Routed through the safe core emitter (awaits async, swallows tracer errors); guarded
                    # with getattr since `core` is duck-typed (a custom core may lack _emit_trace).
                    # from_boundary flags incidents that did NOT pass through perceive_judged, so their
                    # trace_id has no upstream judge stages (an honest marker, not an orphan).
                    await emit_trace(
                        f"{gate_source}@{at}", at, "incident.created",
                        goal_id=incident.concern_id, incident_id=incident.id, from_boundary=from_boundary,
                    )
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
            # PR 2.1 / E5: pick up goals armed/cancelled mid-run (e.g. by voice via RuntimeSession.arm)
            # by recompiling the gate+banks when the registry revision moved. Default ON (a goal armed
            # mid-run MUST reach the gate — the proactive premise). Capture fps is fixed for the open
            # source; only the escalation logic re-arms (the important part for arm-at-runtime). When
            # regate is explicitly OFF and goals STILL move, that is the silent-default trap the shipped
            # realtime demo hit (goal lands in core, gate never re-arms, nothing ever fires) — so we fail
            # LOUD (warning + health event) instead of doing nothing.
            if goals_revision is not None:
                rev = getattr(core.goals, "revision", goals_revision)
                if rev != goals_revision:
                    if regate_on_goal_change:
                        spec, gate, banks = _build_gate()
                    else:
                        _log.warning(
                            "goals_changed_without_regate: core.goals.revision moved %s->%s while "
                            "regate_on_goal_change=False — the running gate will NOT re-arm, so the "
                            "new/changed goal cannot fire. Pass regate_on_goal_change=True "
                            "(RuntimeSession.start does this for you).",
                            goals_revision, rev,
                        )
                        if on_health is not None:
                            await _call_sink(on_health, RuntimeHealthEvent(
                                "goals_changed_without_regate", ts=t,
                                data={"from_revision": goals_revision, "to_revision": rev}))
                    goals_revision = rev
            # A-4: surface any capture-thread health events (e.g. a source whose DELIVERED fps diverges
            # from its reported CAP_PROP_FPS — the previously-silent half-rate decimation). Drained on the
            # frame cadence so it fails LOUD through on_health instead of sitting in an unread list.
            if source_health and on_health is not None:
                while source_health:
                    _hk, _hd = source_health.pop(0)
                    await _call_sink(on_health, RuntimeHealthEvent(_hk, ts=t, data=_hd))
            if stats is not None:
                stats.frames += 1
                stats.frames_dropped = dropped[0]
            clips.push(t, judge_frame)
            if on_frame is not None:
                await _call_sink_with_policy(
                    on_frame, t, judge_frame, errors=sink_errors, stats=stats, on_health=on_health, health_kind="frame_sink_error"
                )

            # SLEEP (RuntimeSession.pause): keep capturing + streaming frames to subscribers (the camera
            # panel stays live) but spend NO gate/judge/incident this tick — a cost-free pause that resumes
            # instantly. Default (paused=None) is byte-for-byte.
            if paused is not None and paused():
                continue

            # boundary first: it must see quiet frames too (the episode ends when activity stops)
            if check_boundary is not None:
                async for incident in _emit_fires(await check_boundary(t=t, inactivity_s=episode_inactivity_s), at=t, from_boundary=True):
                    yield incident
            # PR 1.4: absence-native fires — coverage loss ("I can't see the baby") + overdue deadlines.
            # Sync; ride every frame so they fire on quiet frames the gate skips. from_boundary=True marks
            # that they did NOT pass through perceive_judged (no upstream judge stage in their trace).
            if check_coverage is not None:
                async for incident in _emit_fires(check_coverage(t=t, staleness_s=coverage_staleness_s), at=t, from_boundary=True):
                    yield incident
            if check_deadlines is not None:
                async for incident in _emit_fires(check_deadlines(t=t), at=t, from_boundary=True):
                    yield incident

            # Feed the latest audio energy so a goal with a WakePolicy can fuse it; None (no audio_source)
            # leaves the gate byte-for-byte the visual-only path.
            decision = gate.evaluate(gate_frame, ts=t, energy=audio["energy"])

            # PR 1.3: run edge detectors on the cheap motion/energy signals at native fps. Each emitted
            # boundary (a rep peak, an acoustic onset) is a distinct occurrence fed to the brain via
            # perceive_boundary (edge proposes · brain counts) — firing per the goal's accumulate.fire_when
            # WITHOUT a cloud judge. No banks -> this whole block is skipped (byte-for-byte the pre-PR path).
            if banks and perceive_boundary is not None:
                for goal_id, bank in banks.items():
                    for b in bank.push(t, motion=decision.signal.motion, frame=judge_frame, energy=audio["energy"]):
                        if stats is not None:
                            stats.boundaries += 1
                        fired = perceive_boundary(goal_id, t=b.t)
                        if fired is not None:
                            async for incident in _emit_fires([fired], at=b.t):
                                yield incident
            # TODO 1: a brief-action goal's fovea coast forces a judge on a quiet, in-window frame the
            # motion gate would skip (above-Nyquist coverage). No brief goal -> coast is None -> this is
            # byte-for-byte the gate-only path.
            coast_admit = bool(coast and coast.should_process("__coast__", t, float(decision.signal.motion)))
            # An audio onset admits a frame the visual gate would skip (off-camera cry / glass break). One-shot:
            # consumed here so a single onset causes one judge, not a run of them (AcousticOnset hysteresis +
            # refractory already debounce the signal itself).
            audio_admit = audio["wake"]
            audio["wake"] = False
            if not (decision.escalate or coast_admit or audio_admit):
                continue  # still / sub-cadence -> no cloud call, no incident
            if stats is not None:
                stats.escalations += 1
                stats.judge_calls += 1
                if audio_admit and not (decision.escalate or coast_admit):
                    stats.audio_wakes += 1
            if on_health is not None:
                cause = "audio" if (audio_admit and not (decision.escalate or coast_admit)) else \
                    ("both" if (audio_admit and decision.escalate) else ("coast" if coast_admit and not decision.escalate else "motion"))
                await _call_sink(on_health, RuntimeHealthEvent("escalation", ts=t, data={"source": gate_source, "cause": cause}))
            async for incident in _emit_fires(await core.perceive_judged(judge_frame, gate_source, t=t), at=t):
                yield incident
    except Exception as exc:
        if on_health is not None:
            await _call_sink(on_health, RuntimeHealthEvent("runtime_error", data={"error": repr(exc)}))
        raise
    finally:
        if audio_task is not None:  # stop the audio pump; a dangling capture thread outlives the run otherwise
            audio_task.cancel()
        for _t in list(_utt_tasks):  # R1: an in-flight voice turn must not outlive the run
            _t.cancel()
        if autosave_task is not None:
            autosave_task.cancel()
        # PR 2.2: one last save on the way out so a clean shutdown loses nothing since the last autosave.
        if persistence is not None and getattr(core, "save_state", None):
            try:
                core.save_state(persistence)
            except Exception:  # noqa: BLE001 - teardown; a failed final save must not mask the real exit
                pass
        core.on_trace = _prev_on_trace  # restore — don't leak this run's tracer into the next run(core)
        if on_health is not None:
            await _call_sink(on_health, RuntimeHealthEvent("runtime_ended"))
