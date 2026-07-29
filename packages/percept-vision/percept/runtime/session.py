"""RuntimeSession — the input seam over a running ``run()``.

``run()`` is a pure OUTPUT generator (incidents/asks/frames out); an interactive agent also needs to push
things IN: ask a question about what's on camera, arm/disarm goals at runtime, and interrupt a spoken turn
(the barge-in boundary). ``RuntimeSession`` wraps a background ``run()``, stashes the latest frame it
captured, and exposes exactly those inputs — while the watch keeps running.

The key thing it fixes: a spoken vision question ("what do you see?") MUST carry the watch's current
frame. The voice layer can't reach the camera — the persistent watch owns it — so ``ask`` routes the
utterance through the **conductor** (``core.on_speech_endpoint``) WITH ``latest_frame()``. A demo that
piped audio straight to a voice model with no frame is exactly why it answered "I can't see anything".

STT / TTS / barge-in *detection* stay OUTSIDE the core (the voice layer's job); this only supplies the
frame-scoped answer, the runtime goal ops, and a clean cancellation point.

**Voice embodiment moved app-side (v0.6 diet).** Binding a speech-to-speech transport — command tools,
grounding, fire delivery, A2V metrics, floor arbitration — is no longer the SDK's job. Wire a
``VoiceEngine`` to Percept through the public seams below, using a ``PerceptBridge``/``Floor``
arbitration pattern of your own — see ``docs/sources/voice-and-watchdog.md``. ``attach_transport``/
``deliver_fire`` remain here only as raising stubs pointing at that doc; the ``VoiceTransport`` contract
stays (deprecated) in ``percept.contracts`` for this release.
"""
from __future__ import annotations

import asyncio
import inspect
import os
import time
from collections import deque
from typing import Any, Callable

from percept.goals import GoalOp
from percept.runtime.worker import run


def _env_int(name: str, default: int) -> int:
    """Read a positive int env override, falling back to ``default`` on absent/malformed (never raises)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


class RuntimeSession:
    """A live session wrapping a background ``run()``: `ask`/`arm`/`disarm`/`interrupt`/`latest_frame`."""

    #: Default cap on retained incidents — the steepest 24/7 leak (each incident pins an IncidentSnippet
    #: with up to ~48 DECODED frames, so an unbounded list is MB/day). ``label()`` needs recent history, not
    #: unbounded history. Overridable per-session or via ``PERCEPT_MAX_INCIDENTS``.
    DEFAULT_MAX_INCIDENTS = 500

    def __init__(self, core: Any, *, store: Any = None, label_store: Any = None,
                 max_incidents: int | None = None) -> None:
        self._core = core
        # The local LabelStore for out-of-band triage (`session.label(...)`) and the sink bound to
        # each yielded incident (so `incident.record_label(...)` writes). Injected; None -> labeling off.
        self._label_store = label_store
        # STUB / DEFERRED (control-plane): a shared store (EvidenceMemory / WatchPlan) is INJECTED at
        # construction — the session *uses* it, does not *own* it (multi-device shared memory). The
        # constructor argument is added now, while RuntimeSession is unreleased, so wiring it later is
        # not a signature-compat exercise; the store's actual semantics await the control-plane design.
        self.store = store
        self._latest: tuple[float, Any] | None = None
        self._run_task: asyncio.Task | None = None
        self._turn_task: asyncio.Task | None = None
        self._frame_subscribers: list[Callable] = []
        self._paused = False
        self._muted = False
        self._deaf_until = 0.0            # wall-clock deadline through which utterance endpointing is suppressed
        self._wall_clock = time.monotonic  # timebase for deafen (matches run()'s default wall_clock)
        # BOUNDED incident history (24/7 leak fix). A ``deque(maxlen=N)`` keeps the N most-recent incidents
        # and DROPS the oldest — bounding both the count and the decoded snippet frames they pin. Precedence:
        # explicit arg > PERCEPT_MAX_INCIDENTS > DEFAULT_MAX_INCIDENTS. deque supports append + reversed()
        # (label lookup).
        if max_incidents is None:
            max_incidents = _env_int("PERCEPT_MAX_INCIDENTS", self.DEFAULT_MAX_INCIDENTS)
        self.max_incidents = max(1, int(max_incidents))
        self.incidents: deque[Any] = deque(maxlen=self.max_incidents)

    # -- power ladder: SLEEP (pause judging, keep the socket + frame stream) ----
    def pause(self) -> None:
        """SLEEP: stop spending gate/judge on frames (cost-free) while keeping the session alive and the
        frame stream flowing. NOT :meth:`stop` (which is terminal). Resume with :meth:`resume`."""
        self._paused = True

    def resume(self) -> None:
        """Wake from :meth:`pause` — judging resumes on the next frame."""
        self._paused = False

    def paused(self) -> bool:
        return self._paused

    # -- power ladder: MUTED (detach audio from perception, keep watching) ------
    def detach_audio_source(self) -> None:
        """MUTED: stop feeding the mic to perception — the fused judge goes video-only and the
        energy/onset gate signals freeze — without ending the session. Only meaningful when the session
        was started with an ``audio_source``. Re-enable with :meth:`attach_audio_source`."""
        self._muted = True

    def attach_audio_source(self) -> None:
        """Un-MUTE — resume feeding the session's mic stream to perception."""
        self._muted = False

    def muted(self) -> bool:
        return self._muted

    def mute(self) -> None:
        """Alias for :meth:`detach_audio_source` — the power-ladder verb. MUTED: drop the mic
        from perception (video-only judging, frozen audio gate) without ending the session."""
        self.detach_audio_source()

    def unmute(self) -> None:
        """Alias for :meth:`attach_audio_source` — resume feeding the mic to perception."""
        self.attach_audio_source()

    # -- turn-taking: DEAFEN (suppress endpointing while the app speaks) ---------
    def deafen(self, seconds: float) -> None:
        """Suppress utterance endpointing for ``seconds`` — the app is **about to speak**.

        ``on_utterance`` opens an un-AEC'd mic; if the app plays audio into the same room, the tail of its
        own speech is captured (before any app-side ``speaking`` flag can fall) and would be endpointed as
        a user turn — Percept answers itself, and loops. A flag set at *delivery* cannot express "ignore
        what I'm about to say"; a deadline can. Call this with ``playback_duration + margin`` (margin >
        the endpointer hangover) immediately before playback.

        Chunks captured in the window still fill the AudioWindow (fused judging is unaffected); they are
        simply not fed to the VAD, and the VAD is reset on exit so a half-built utterance cannot survive
        the window. Idempotent-ish: the LATEST deadline wins (a longer/overlapping window extends it)."""
        deadline = self._wall_clock() + max(0.0, float(seconds))
        self._deaf_until = max(self._deaf_until, deadline)

    def deaf_until(self) -> float:
        """The current deafen deadline in the ``wall_clock`` timebase (wired into ``run(deaf_until=...)``)."""
        return self._deaf_until

    # -- read side --------------------------------------------------------
    def latest_frame(self) -> Any | None:
        """The most recent judge frame the watch captured (what ``ask`` scopes a vision answer to)."""
        return self._latest[1] if self._latest is not None else None

    def subscribe(self, cb: Callable) -> Callable:
        """Register a frame consumer ``cb(t, frame)`` (sync or async), called on every captured frame.

        The command seam is **multi-consumer**: a UI panel AND a voice command adapter can each read the
        watch's frames without stepping on the single ``on_frame`` callback. Returns an ``unsubscribe()``
        closure. A subscriber that raises is dropped for that frame — a bad consumer never crashes the watch."""
        self._frame_subscribers.append(cb)

        def _unsub() -> None:
            try:
                self._frame_subscribers.remove(cb)
            except ValueError:
                pass

        return _unsub

    async def _notify_frame(self, t: float, frame: Any) -> None:
        """Stash the latest frame and fan it out to all subscribers (wired as ``run(on_frame=...)``)."""
        self._latest = (t, frame)
        for cb in list(self._frame_subscribers):
            try:
                res = cb(t, frame)
                if inspect.isawaitable(res):
                    await res
            except Exception:  # noqa: BLE001 - a frame consumer must never crash the watch
                pass

    # -- input side -------------------------------------------------------
    async def ask(self, utterance: str, *, t: float | None = None) -> Any:
        """Answer a spoken turn about the CURRENT scene, through the conductor, with the latest frame.

        Runs as a tracked task so :meth:`interrupt` can cancel it mid-reply (barge-in). Returns the
        core's :class:`~percept.adapters.TurnResult` (route + reply)."""
        self._turn_task = asyncio.ensure_future(
            self._core.on_speech_endpoint(utterance, frame=self.latest_frame(), t=t)
        )
        try:
            return await self._turn_task
        finally:
            self._turn_task = None

    def interrupt(self) -> bool:
        """Cancel an in-flight :meth:`ask` (the barge-in boundary — detecting barge-in is the voice
        layer's job). Returns True if a turn was actually cancelled."""
        task = self._turn_task
        if task is not None and not task.done():
            task.cancel()
            return True
        return False

    async def arm(self, goal: Any) -> Any:
        """Arm a new watch goal at runtime (e.g. from a voice command). Bumps the goal-registry
        revision, so a ``run()`` started with ``regate_on_goal_change=True`` re-arms the gate for it."""
        return self._core.add_goal(goal)

    async def disarm(self, goal_id: str) -> None:
        """Cancel a goal at runtime (stops watching for it); also re-arms the gate under regate."""
        self._core.apply_goal_ops([GoalOp(op="cancel", goal_id=goal_id)])

    def on_goal_change(self, cb: Callable[[], Any]) -> Callable[[], None]:
        """Register ``cb()`` to fire on EVERY goal-set change; returns an ``unsubscribe()`` closure.

        Subscribes at the AUTHORITATIVE mutation point — the :class:`~percept.goals.GoalRegistry`
        revision change — not at :meth:`arm`/:meth:`disarm`. That is the whole point: the rich command
        path ``core.arm_from_language`` / ``apply_goal_ops`` mutates the registry WITHOUT going through
        ``session.arm``, so a callback wired to ``arm`` would silently miss every voice-authored watch.
        Binding to the registry catches all of it: structured arm, language-authored add/update/remove,
        disarm, restore, and same-id replacement.

        Purpose is **snapshot invalidation** (mark a grounding cache dirty), NOT wire injection. The
        callback takes no arguments; callbacks are additive + ISOLATED — one raising subscriber can never
        break another (the registry swallows it)."""
        registry = self._core.goals

        def _on_revision(_revision: int) -> None:
            cb()

        subscribe = getattr(registry, "subscribe_revision", None)
        if subscribe is None:  # a registry without the hook (older core) -> a no-op unsubscribe
            return lambda: None
        return subscribe(_on_revision)

    def _goal_entry(self, g: Any) -> dict:
        """Map a :class:`~percept.contracts.Goal` to its bounded, read-only public view (no Goal leakage).

        Carries the safe display fields (id/condition/status) PLUS the durable ``origin`` (as its wire
        string) and the goal's authored **requested delivery** (parsed once through the single interpreter,
        :func:`~percept.incidents.parse_requested_delivery`). The requested delivery is what an application
        needs to resolve an incident back to its goal's authored ttl/verbatim/preempt HINTS while the
        attention policy still clamps the effective ``preempt`` by origin."""
        from percept.incidents import parse_requested_delivery

        registry = self._core.goals
        origin = getattr(g, "origin", None)
        origin_str = origin.value if hasattr(origin, "value") else (None if origin is None else str(origin))
        requested = parse_requested_delivery((getattr(g, "state", None) or {}).get("delivery"))
        return {
            "id": g.id,
            "condition": getattr(g, "condition", ""),
            "status": registry.status(g.id) or "active",
            "origin": origin_str,
            "delivery": {
                "preempt": requested.preempt,
                "ttl_s": requested.ttl_s,
                "verbatim": requested.verbatim,
            },
        }

    def list_goals(self) -> list[dict]:
        """A bounded, public read of the ACTIVE goals — one :meth:`_goal_entry` view each.

        So an application (e.g. a ``whats_armed`` voice tool) can answer "what am I watching?" and resolve
        a fire back to its goal's authored delivery WITHOUT reaching into ``core.goals``. Each entry is a
        plain dict (``{id, condition, status, origin, delivery}``) — no goal objects, no core references;
        inactive (completed/cancelled) goals are excluded."""
        registry = self._core.goals
        return [self._goal_entry(g) for g in registry.active()]

    def goal_view(self, goal_id: str) -> dict | None:
        """Resolve a single goal by id to its bounded read-only view (:meth:`_goal_entry`), or ``None``.

        The seam an application uses to map an :class:`~percept.contracts.Incident` back to its goal so
        ``attention_intent_for`` can honor the goal's authored ttl/verbatim — a language-authored goal has
        no other public handle (it never passed through ``session.arm``). Returns the same shape as a
        :meth:`list_goals` entry; ``None`` for an unknown id. Read-only: never leaks the Goal object."""
        registry = self._core.goals
        get = getattr(registry, "get", None)
        g = get(goal_id) if get is not None else None
        if g is None:
            return None
        return self._goal_entry(g)

    # -- feedback intake: out-of-band triage of a past fire ----
    def label(self, incident_id: str, good: bool, note: str | None = None,
              tags: tuple[str, ...] | list[str] = (), *,
              source: str = "api") -> Any:
        """Record a ground-truth label for a PAST incident by id — out-of-band triage.

        Looks the incident up in this session's history (so ``goal_id`` / ``verdict_ref`` come along) and
        delegates to :meth:`Incident.record_label`; if the id is unknown it still writes a minimal
        :class:`~percept.contracts.LabelRecord` (id only). A LOCAL write — no key, no network. Requires a
        ``label_store`` (pass one to :func:`start` / the constructor); raises :class:`RuntimeError` if
        none is bound."""
        if self._label_store is None:
            raise RuntimeError(
                "RuntimeSession has no label_store — pass label_store= to percept.runtime.start() "
                "(or construct RuntimeSession(label_store=...)). label() is a LOCAL write."
            )
        for inc in reversed(self.incidents):
            if getattr(inc, "id", None) == incident_id:
                return inc.record_label(good, note=note, tags=tags, source=source)  # type: ignore[arg-type]
        # Unknown incident (e.g. from a prior process): record what we can.
        from percept.contracts import LabelRecord

        rec = LabelRecord(
            incident_id=incident_id, verdict_ref="", goal_id="", ts=time.time(),
            good=bool(good), note=note, tags=list(tags), source=source,  # type: ignore[arg-type]
        )
        return self._label_store.append(rec)

    # -- voice embodiment: MOVED app-side (v0.6 diet) — raising stubs ----------
    def attach_transport(self, *args: Any, **kwargs: Any) -> Any:
        """REMOVED — binding a voice transport is now the application's job (v0.6 SDK voice diet).

        Command tools, pull-grounding, fire delivery, and A2V metrics moved out of the SDK: wire a voice
        engine app-side using your own ``PerceptBridge``-style adapter to bind it to a ``Floor``. See
        ``docs/sources/voice-and-watchdog.md`` for the seam. The ``VoiceTransport`` contract stays
        (deprecated) in ``percept.contracts`` for this release. This stub raises so an old call fails
        loud, not silent."""
        raise RuntimeError(
            "RuntimeSession.attach_transport was removed in the v0.6 SDK voice diet. The voice "
            "embodiment (transport wiring, pull-grounding, fire delivery, A2V metrics) moved app-side — "
            "wire a voice engine app-side using your own PerceptBridge/Floor arbitration; see "
            "docs/sources/voice-and-watchdog.md. The VoiceTransport contract remains (deprecated) in "
            "percept.contracts."
        )

    def deliver_fire(self, *args: Any, **kwargs: Any) -> Any:
        """REMOVED — fire delivery + floor arbitration are now the application's job (v0.6 voice diet).

        Routing a fire to speech/act (interrupt / queue / drop, TTL, verbatim) moved app-side: wire
        ``percept.start(..., on_incident=bridge.on_incident)`` and let your own Floor +
        ``attention_intent_for``-style policy decide. See ``docs/sources/voice-and-watchdog.md``. This
        stub raises so an old call fails loud, not silent."""
        raise RuntimeError(
            "RuntimeSession.deliver_fire was removed in the v0.6 SDK voice diet. Fire delivery + floor "
            "arbitration moved app-side — wire percept.start(..., on_incident=bridge.on_incident) and let "
            "your own Floor + attention_intent_for-style policy decide; see "
            "docs/sources/voice-and-watchdog.md."
        )

    async def stop(self) -> None:
        """Stop the background watch. Cancels an in-flight :meth:`ask` turn, then the run task."""
        self.interrupt()
        if self._run_task is not None and not self._run_task.done():
            self._run_task.cancel()
            try:
                await self._run_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001 - teardown; swallow
                pass


async def start(core: Any, *, source: Any, on_frame: Callable | None = None,
                on_incident: Callable | None = None, on_utterance: Callable | None = None,
                store: Any = None, label_store: Any = None, max_incidents: int | None = None,
                allow_scripted_brain: bool = False,
                **run_kwargs: Any) -> RuntimeSession:
    """Start a background watch and return a :class:`RuntimeSession` to drive it.

    Wraps ``run(core, source=..., regate_on_goal_change=True, ...)`` in a task. Incidents still flow to
    ``on_incident`` (pass it here); ``on_frame`` is wrapped so the session always holds the latest frame
    for :meth:`RuntimeSession.ask`. Pass ``audio_source="mic:0"`` + ``on_utterance`` (peer of
    ``on_frame``) for the ambient voice entry — Percept owns the mic and hands each endpointed utterance
    to the voice layer, so there's no push-to-talk / fixed record window.

    ``on_utterance`` gives you an **un-AEC'd mic**. The handler runs off the pump, so a slow voice turn
    no longer blocks capture — but if the app plays audio into the same room it **must** call
    :meth:`RuntimeSession.deafen` around playback (with ``playback_duration + margin``), or Percept will
    hear itself and loop. For a speakers-on, always-on deployment with AEC, wire a voice engine app-side
    (see ``docs/sources/voice-and-watchdog.md``). ``store`` is the injected shared store (deferred; see
    :class:`RuntimeSession`). ``async for incident in run(...)`` remains the headless front door; this is
    the interactive layer on top."""
    # FATAL seam: guard SYNCHRONOUSLY at the start() call (not inside the drain task, where a
    # ModeError would surface late/awkwardly) when a live voice path (on_utterance) is wired behind a
    # scripted brain. Passing on_utterance later is caught by run()'s own guard.
    if on_utterance is not None:
        from percept.agent import _guard_live_language_path

        _guard_live_language_path(core, allow_scripted_brain=allow_scripted_brain)

    session = RuntimeSession(core, store=store, label_store=label_store, max_incidents=max_incidents)
    if on_frame is not None:  # the start-provided consumer is just the first subscriber (back-compat)
        session.subscribe(on_frame)

    async def _drain() -> None:
        async for incident in run(
            core, source=source, on_frame=session._notify_frame, on_incident=on_incident,  # noqa: SLF001
            on_utterance=on_utterance, label_store=label_store, allow_scripted_brain=allow_scripted_brain,
            regate_on_goal_change=True, paused=session.paused, muted=session.muted,
            deaf_until=session.deaf_until, **run_kwargs,
        ):
            session.incidents.append(incident)

    session._run_task = asyncio.ensure_future(_drain())  # noqa: SLF001 - same module
    await asyncio.sleep(0)  # let the run task start (open the source, stash a first frame)
    return session


__all__ = ["RuntimeSession", "start"]
