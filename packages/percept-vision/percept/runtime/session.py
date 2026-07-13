"""RuntimeSession — the input seam over a running ``run()`` (PR 2.1).

``run()`` is a pure OUTPUT generator (incidents/asks/frames out); a voice/interactive agent also needs
to push things IN: ask a question about what's on camera, arm/disarm goals at runtime, and interrupt a
spoken turn (the barge-in boundary). ``RuntimeSession`` wraps a background ``run()``, stashes the
latest frame it captured, and exposes exactly those inputs — while the watch keeps running.

The key thing it fixes: a spoken vision question ("what do you see?") MUST carry the watch's current
frame. The voice layer can't reach the camera — the persistent watch owns it — so ``ask`` routes the
utterance through the **conductor** (``core.on_speech_endpoint``) WITH ``latest_frame()``. A demo that
piped audio straight to a voice model with no frame is exactly why it answered "I can't see anything".

STT / TTS / barge-in *detection* stay OUTSIDE the core (the voice layer's job); this only supplies the
frame-scoped answer, the runtime goal ops, and a clean cancellation point.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import os
import time
from collections import deque
from typing import Any, Callable

from percept.adapters import Reply
from percept.goals import GoalOp
from percept.runtime.worker import run

#: JSON schema for the thin ``arm_goal`` tool (a transport structures the goal itself). Kept for
#: back-compat; the realtime path prefers ``watch`` (raw language → the conductor authors the watchspec).
ARM_GOAL_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "description": "short kebab-case id (optional)"},
        "condition": {"type": "string", "description": "plain-language condition to watch the camera for"},
        "say": {"type": "string", "description": "short message to speak once, when it fires"},
    },
    "required": ["condition"],
}

#: The RICH command verbs a realtime transport exposes to its model. Each passes RAW language down so
#: Percept's **conductor** authors the full watchspec (the levers) — not a pre-structured goal.
WATCH_SCHEMA = {
    "type": "object",
    "properties": {"command": {"type": "string",
                               "description": "the user's watch request, verbatim, e.g. 'tell me the second my baby cries'"}},
    "required": ["command"],
}
STOP_WATCHING_SCHEMA = {
    "type": "object",
    "properties": {"what": {"type": "string", "description": "which watch to stop (a word or phrase from it)"}},
    "required": ["what"],
}
WHATS_ARMED_SCHEMA = {"type": "object", "properties": {}}


def _env_int(name: str, default: int) -> int:
    """Read a positive int env override, falling back to ``default`` on absent/malformed (never raises)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _arm_slug(raw_id: Any, condition: str) -> str:
    """Derive a STABLE, COLLISION-RESISTANT goal id for a voice-armed watch.

    An explicit ``id`` (from the tool args) is honored verbatim (kebab-cased, bounded). Otherwise the
    id is derived from the condition — but a plain ``condition[:40]`` slug silently COLLIDES for two
    watches sharing a 40-char prefix ("the red car is parked in the driveway near the mailbox on
    monday" vs "…tuesday"), and ``GoalRegistry.create`` is last-writer-wins, so one instruction is lost
    with no signal. We suffix a short content hash of the FULL condition so distinct conditions get
    distinct ids even when their prefixes match (D3b)."""
    explicit = str(raw_id or "").strip()
    if explicit:
        return explicit.lower().replace(" ", "-")[:40] or "goal"
    slug = condition.strip().lower().replace(" ", "-")
    digest = hashlib.sha1(condition.encode("utf-8")).hexdigest()[:6]
    return f"{slug[:32]}-{digest}" if slug else f"goal-{digest}"


class RuntimeSession:
    """A live session wrapping a background ``run()``: `ask`/`arm`/`disarm`/`interrupt`/`latest_frame`."""

    #: Default cap on retained incidents — the steepest 24/7 leak (each incident pins an IncidentSnippet
    #: with up to ~48 DECODED frames, so an unbounded list is MB/day). ``label()`` needs recent history, not
    #: unbounded history; the pulse reads only the last 3. Overridable per-session or via ``PERCEPT_MAX_INCIDENTS``.
    DEFAULT_MAX_INCIDENTS = 500

    def __init__(self, core: Any, *, store: Any = None, label_store: Any = None,
                 max_incidents: int | None = None) -> None:
        self._core = core
        # L-1: the local LabelStore for out-of-band triage (`session.label(...)`) and the sink bound to
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
        self._transport: Any = None       # the attached VoiceTransport, if any
        self._transport_stt: Any = None   # optional STT for audio-only utterances (a transport that
        self._act_sink: Any = None        # the ACT channel for fires (webhook/callback/log)
        self._user_speaking = False       # transport's turn-in-progress signal (fire timing)
        # A2V latency: the wall time a response was TRIGGERED (a user turn ended, or a fire was enqueued to
        # the transport) and its kind; on the transport's on_bot_speaking(True) edge (audio-out actually
        # began) we record `now - since` into metrics. Measuring to say()/enqueue instead understates true
        # latency by the model's TTS first-token time — the number the realtime demo/paid pitch rests on.
        self._pending_response_since: float | None = None
        self._pending_response_kind: str | None = None
        self._queued_fires: list[tuple] = []  # (incident, delivery, queued_at) held behind a live turn
        self._pulsed = ""                 # last brain-state summary sent (dedup the pulse)
        self._arm_tasks: set[asyncio.Task] = set()  # in-flight async watch-arming (watch verb, §3 #1)
        # BOUNDED incident history (24/7 leak fix). A ``deque(maxlen=N)`` keeps the N most-recent incidents
        # and DROPS the oldest — bounding both the count and the decoded snippet frames they pin. Precedence:
        # explicit arg > PERCEPT_MAX_INCIDENTS > DEFAULT_MAX_INCIDENTS. deque supports append + reversed()
        # (label lookup); the pulse takes list(...)[-3:] since a deque isn't sliceable.
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
        """Alias for :meth:`detach_audio_source` — the spec's power-ladder verb. MUTED: drop the mic
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
        result = self._core.add_goal(goal)
        await self._pulse_state()   # goal changed → re-ground the wire (§4.1)
        return result

    async def disarm(self, goal_id: str) -> None:
        """Cancel a goal at runtime (stops watching for it); also re-arms the gate under regate."""
        self._core.apply_goal_ops([GoalOp(op="cancel", goal_id=goal_id)])
        await self._pulse_state()   # goal changed → re-ground the wire (§4.1)

    # -- flywheel intake: out-of-band triage of a past fire (L-1, §4.3) ----
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

    # -- voice transport: bind an AEC'd mic+speaker pipe (Pipecat/LiveKit/browser/S2S) ----
    async def attach_transport(self, transport: Any, *, arm_tool: bool = True, watch_tools: bool = True,
                               stt: Any = None, act_sink: Any = None,
                               allow_scripted_brain: bool = False) -> Any:
        """Bind a :class:`~percept.contracts.VoiceTransport` and wire it to cognition — the ONE call an
        app makes for voice. The transport owns mic + speaker + AEC + endpointing + barge-in.

        Two archetypes, one call:
          * **component** (STT transport that emits discrete utterances) — each ``on_utterance`` runs a
            frame-grounded conductor turn (:meth:`ask`) and the reply is streamed back via ``say``.
          * **realtime speech-to-speech** (the model handles the conversation itself) — it does NOT emit
            ``on_utterance``; instead the model calls the command tools (``watch``/``stop_watching``/
            ``whats_armed``), the brain keeps the wire GROUNDED via ``sync_state``, mines ``on_transcript``
            for evidence, and speaks proactive fires via :meth:`deliver_fire`.

        - ``watch_tools`` — expose the rich ``watch``/``stop_watching``/``whats_armed`` verbs (raw language
          → the **conductor** authors the watchspec). ``arm_tool`` also exposes the thin ``arm_goal``.
        - ``act_sink`` — the ACT channel for fires (webhook/callback/log). A ``speak`` fire with no
          transport falls back to it (assistant-class: notify when no one's listening).
        - ``stt`` — transcriber for a component transport that ships audio-only utterances.

        On THIS path Percept never touches raw audio — the transport owns mic + speaker + AEC, so there is
        no echo to hand-cancel. (This does NOT hold for the ``on_utterance`` path — see :func:`start` —
        which opens the mic itself with no AEC.) Returns the started transport."""
        # FATAL seam (RFC §5): an attached transport IS a live language path (it emits utterances and
        # exposes watch/arm tools that route raw language to the conductor). Behind a scripted brain
        # those tools silently author nothing — refuse it here, at the wiring call.
        from percept.agent import _guard_live_language_path

        _guard_live_language_path(self._core, allow_scripted_brain=allow_scripted_brain)
        self._transport = transport
        self._transport_stt = stt
        self._act_sink = act_sink
        transport.set_frame_provider(self.latest_frame)
        if arm_tool:
            transport.register_tool("arm_goal", dict(ARM_GOAL_SCHEMA), self._arm_goal_tool)
        if watch_tools:
            transport.register_tool("watch", dict(WATCH_SCHEMA), self._watch_tool)
            transport.register_tool("stop_watching", dict(STOP_WATCHING_SCHEMA), self._stop_watching_tool)
            transport.register_tool("whats_armed", dict(WHATS_ARMED_SCHEMA), self._whats_armed_tool)
        transport.on_barge_in(self._on_transport_barge_in)
        transport.on_utterance(self._on_transport_utterance)      # component archetype
        transport.on_user_speaking(self._set_user_speaking)        # fire timing (§4.3)
        if hasattr(transport, "on_bot_speaking"):                  # optional/additive: audio-out edge -> A2V
            transport.on_bot_speaking(self._on_bot_speaking)
        transport.on_transcript(self._on_transport_transcript)     # evidence tee (§4.6)
        await transport.start()
        await self._pulse_state(force=True)                        # ground the wire at connect (§4.1)
        return transport

    async def detach_transport(self) -> None:
        """Unbind + stop the attached transport (idempotent). The watch keeps running."""
        transport, self._transport = self._transport, None
        if transport is not None:
            try:
                await transport.stop()
            except Exception:  # noqa: BLE001 - teardown must not raise
                pass

    async def _on_transport_utterance(self, ev: Any) -> Any:
        """A complete (AEC'd, endpointed) user turn from the transport -> frame-grounded reply."""
        utterance = ev.text
        if not utterance and getattr(ev, "audio", None) is not None:
            utterance = await self._transcribe(ev.audio)
        if not utterance:
            return None
        result = await self.ask(utterance, t=getattr(ev, "t", None))
        reply = getattr(result, "reply", None)
        if self._transport is not None and reply:
            await self._transport.say(Reply(text=reply))  # no-op for a speech-to-speech transport
        return result

    async def _on_transport_barge_in(self) -> None:
        """User cut in during playout: cancel the in-flight turn (cognition) + stop playout (transport)."""
        self.interrupt()
        if self._transport is not None:
            try:
                await self._transport.interrupt()
            except Exception:  # noqa: BLE001 - a transport that can't interrupt is fine
                pass

    async def _arm_goal_tool(self, args: dict) -> dict:
        """The ``arm_goal`` tool handler — a spoken watch-command becomes an armed goal."""
        from percept.adapters import Goal

        args = args or {}
        condition = str(args.get("condition") or "").strip()
        if not condition:
            return {"error": "condition required"}
        gid = _arm_slug(args.get("id"), condition)
        say = str(args.get("say") or f"Heads up — {condition}.").strip()
        await self.arm(Goal(id=gid, condition=condition, say=say))
        return {"output": f"now watching for {condition}"}

    # -- realtime command verbs: raw language → the CONDUCTOR authors the watchspec (§3 #1) ----
    async def _watch_tool(self, args: dict) -> dict:
        """``watch(command)`` — author the watchspec from raw language and arm it, ASYNC.

        Returns instantly ("On it.") so the voice turn never blocks on the conductor, and — critically —
        routes through :meth:`Percept.arm_from_language`, NOT :meth:`ask`, so the specialist's frame answer
        is never streamed back into the held wire (that both poisoned the session and added ~8s, §4.2). The
        armed state re-grounds the wire via ``sync_state`` when authoring completes; ``whats_armed`` then
        reflects it."""
        command = str((args or {}).get("command") or "").strip()
        if not command:
            return {"error": "command required"}

        async def _arm() -> None:
            try:
                await self._core.arm_from_language(command)   # conductor authors + arms; no spoken reply
                await self._pulse_state()                     # re-ground the wire once the goal is live (§4.1)
            except Exception:  # noqa: BLE001 - background arming must never crash the wire
                pass

        task = asyncio.ensure_future(_arm())
        self._arm_tasks.add(task)
        task.add_done_callback(self._arm_tasks.discard)
        return {"output": "On it."}

    async def _stop_watching_tool(self, args: dict) -> dict:
        what = str((args or {}).get("what") or "").strip().lower()
        if not what:
            return {"error": "what required"}
        stopped = []
        for g in list(self._core.goals.active()):
            hay = f"{g.id} {getattr(g, 'condition', '')}".lower()
            if what in hay or any(w in hay for w in what.split() if len(w) > 2):
                await self.disarm(g.id)
                stopped.append(g.id)
        await self._pulse_state()
        return {"output": f"stopped {', '.join(stopped)}" if stopped else f"nothing matching '{what}'"}

    async def _whats_armed_tool(self, _args: dict) -> dict:
        return {"armed": [{"id": g.id, "watching_for": getattr(g, "condition", "")}
                          for g in self._core.goals.active()]}

    # -- fire timing + evidence (§4.3, §4.6) -----------------------------
    async def _set_user_speaking(self, speaking: bool) -> None:
        self._user_speaking = bool(speaking)
        if not speaking:
            # Turn ended → the A2V clock starts: an S2S wire (Gemini Live) now renders its reply, and the
            # honest latency is until on_bot_speaking(True) (audio-out), not until say()/enqueue.
            self._mark_pending_response("reactive")
            await self._flush_queued_fires()   # turn ended → deliver anything queued behind it

    def _mark_pending_response(self, kind: str) -> None:
        """Stamp the moment a spoken response was TRIGGERED (a user turn ended, or a fire was enqueued).
        The transport's on_bot_speaking(True) edge closes the window and records the A2V latency."""
        self._pending_response_since = self._wall_clock()
        self._pending_response_kind = kind

    async def _on_bot_speaking(self, speaking: bool) -> None:
        """The agent's voice output actually began (``True``) / ended (``False``) — the HONEST A2V end.
        Record ``now - trigger`` into ``core.metrics`` ONCE per turn (subsequent audio frames of the same
        turn find no pending stamp and are ignored). Only the transport can see its own playout, so a wire
        that never emits this simply yields no A2V metric — everything else is unaffected."""
        if not speaking or self._pending_response_since is None:
            return
        dt = max(0.0, self._wall_clock() - self._pending_response_since)
        kind = self._pending_response_kind or "reply"
        metrics = getattr(self._core, "metrics", None)
        if metrics is not None:
            metrics.add("voice.a2v_latency_s.total", dt)
            metrics.incr("voice.a2v_latency_s.count")
            metrics.add(f"voice.a2v_latency_s.{kind}.total", dt)
            metrics.incr(f"voice.a2v_latency_s.{kind}.count")
        self._pending_response_since = None
        self._pending_response_kind = None

    async def _on_transport_transcript(self, text: str) -> None:
        """A user-turn transcript (free from a realtime model) → low-priority evidence; nobody waits."""
        text = (text or "").strip()
        note = getattr(self._core, "note_mention", None)
        if text and note is not None:
            try:
                note(text)   # records into the evidence/mention path; the conductor mines it later
            except Exception:  # noqa: BLE001 - evidence is best-effort, never blocks the wire
                pass

    # -- fire delivery: modality + delivery policy (§3, §4.3) ------------
    async def deliver_fire(self, incident: Any) -> None:
        """Route a watch FIRE. Wire ``percept.run(on_incident=session.deliver_fire)``. Always records to
        the act sink (nothing lost; verbatim); and — per goal ``modality`` + ``delivery`` lever — speaks
        it through the transport (interrupt / queue-behind-turn / drop-if-stale). A ``speak`` fire with no
        transport degrades to the act sink (assistant-class: notify when no one's listening)."""
        goal = self._goal_for(getattr(incident, "concern_id", None))
        modality = getattr(goal, "modality", None) or "speak"
        delivery = dict((getattr(goal, "state", None) or {}).get("delivery", {})) if goal else {}
        if self._act_sink is not None:
            res = self._act_sink(incident)
            if inspect.isawaitable(res):
                await res
        if "speak" in modality and self._transport is not None:
            await self._speak_fire(incident, delivery)
        await self._pulse_state()

    async def _speak_fire(self, incident: Any, delivery: dict) -> None:
        if self._user_speaking:
            if delivery.get("preempt") == "interrupt":   # severe → cut into the turn
                try:
                    await self._transport.interrupt()
                except Exception:  # noqa: BLE001
                    pass
            else:                                        # gentle → hold behind the turn
                self._queued_fires.append((incident, delivery, time.monotonic()))
                return
        self._mark_pending_response("proactive")  # A2V clock for a spoken fire: enqueue -> audio-out
        await self._transport.say(Reply(text=getattr(incident, "message", "")),
                                  verbatim=bool(delivery.get("verbatim", False)))

    async def _flush_queued_fires(self) -> None:
        queued, self._queued_fires = self._queued_fires, []
        now = time.monotonic()
        for incident, delivery, queued_at in queued:
            ttl = delivery.get("ttl_s")
            if ttl is not None and (now - queued_at) > float(ttl):
                continue   # went stale while the user talked → drop (a late nudge is worse than none)
            if self._transport is not None:
                self._mark_pending_response("proactive")  # A2V clock for the (now-flushed) spoken fire
                await self._transport.say(Reply(text=getattr(incident, "message", "")),
                                          verbatim=bool(delivery.get("verbatim", False)))

    # -- brain-state pulse: keep the wire grounded (§4.1) ----------------
    def _brain_state_summary(self) -> str:
        watching = "; ".join(f"{g.id}: {getattr(g, 'condition', '')}"
                             for g in self._core.goals.active()) or "nothing yet"
        recent = list(self.incidents)[-3:]  # deque isn't sliceable; materialize the last 3
        fired = "; ".join(getattr(i, "message", "") for i in recent) if recent else "none"
        return f"[percept state] watching for → {watching}. recent alerts → {fired}."

    async def _pulse_state(self, *, force: bool = False) -> None:
        """Push the brain's current state into the transport's context (no response triggered) so the
        wire's reactive answers stay coherent with what the brain knows. Deduped unless forced."""
        if self._transport is None:
            return
        summary = self._brain_state_summary()
        if not force and summary == self._pulsed:
            return
        self._pulsed = summary
        sync = getattr(self._transport, "sync_state", None)
        if sync is not None:
            try:
                await sync(summary)
            except Exception:  # noqa: BLE001 - grounding is best-effort
                pass

    def _goal_for(self, gid: Any) -> Any:
        if gid is None:
            return None
        get = getattr(self._core.goals, "get", None)
        if get is not None:
            try:
                g = get(gid)
                if g is not None:
                    return g
            except Exception:  # noqa: BLE001
                pass
        for g in self._core.goals.active():
            if getattr(g, "id", None) == gid:
                return g
        return None

    async def _transcribe(self, audio: Any) -> str | None:
        """Transcribe an audio-only utterance via the optional ``stt`` given to :meth:`attach_transport`.
        Accepts a plain async/sync ``callable(audio) -> str`` OR an :class:`STT` seam (``.stream``)."""
        stt = self._transport_stt
        if stt is None:
            return None
        if hasattr(stt, "stream"):  # an STT seam: take the last final event's text
            text = ""
            async for e in stt.stream(audio):
                if getattr(e, "is_final", False):
                    text = e.text
            return text or None
        res = stt(audio)
        res = await res if inspect.isawaitable(res) else res
        return res or None

    async def stop(self) -> None:
        """Stop the background watch (and the attached transport, if any)."""
        self.interrupt()
        for task in list(self._arm_tasks):   # cancel any in-flight async watch-arming
            task.cancel()
        await self.detach_transport()
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
    hear itself and loop. For a speakers-on, always-on deployment prefer :meth:`RuntimeSession.attach_transport`
    — that path owns AEC. ``store`` is the injected shared store (deferred; see :class:`RuntimeSession`).
    ``async for incident in run(...)`` remains the headless front door; this is the interactive layer on top."""
    # FATAL seam (RFC §5): guard SYNCHRONOUSLY at the start() call (not inside the drain task, where a
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
