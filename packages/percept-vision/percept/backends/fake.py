"""Deterministic, offline, key-free backends.

Fakes are load-bearing: they make ``pip install`` -> run work with no keys, power the L1 CI
lane, and isolate *our logic* from *the vendor*. They are **scripted, never random**.

The hard rule (standing rule 2): a fake proves seam wiring and our logic — it must **never**
be used to assert backend *intelligence*. Real-model behaviour is tested in L2/L3 only. So
``FakeVision`` reads a token off the frame and returns a pre-scripted verdict; it does not
"understand" anything.
"""

from __future__ import annotations

import inspect
from typing import Any, AsyncIterator

from ..adapters import Decision, Goal, Observation, Reply, STTEvent, UtteranceEvent, Verdict

# frame-token -> scripted verdict. Tests pass a string frame like "sip-coffee".
_SCRIPT: dict[str, Verdict] = {
    "sip-coffee": Verdict(satisfied=True, confidence=0.9, evidence="scripted: coffee"),
    "sip-water": Verdict(satisfied=False, confidence=0.9, evidence="scripted: water"),
    "maybe-coffee": Verdict(satisfied=True, confidence=0.4, evidence="scripted: unsure"),
    "empty": Verdict(satisfied=False, confidence=0.95, evidence="scripted: nothing"),
}
_UNKNOWN = Verdict(satisfied=False, confidence=0.2, evidence="scripted: unknown frame")


class FakeVision:
    """Returns a scripted verdict keyed off a string frame token.

    ``judge`` is **scripted wiring, not intelligence**: it looks the frame token up in a dict.
    When ``corrections`` are present it first tries a ``"<frame>|corrected"`` key — this lets a
    test prove that a user-asserted fact actually *flows into* the judge and changes the outcome
    (the assertion-beats-observation wiring), without claiming the fake understands anything.

    ``detect`` returns scripted observations from ``detect_script`` so the entity graph has
    something to track in offline tests.
    """

    #: The L1 fake is treated as claim-channel-parsing (scripted, offline): a ``claim_slot`` plan armed
    #: on a fake/scripted backend must NOT raise the "backend unverified" receipt warning, so the keyless
    #: recipe test drives the real fold without a spurious diagnostic (mirrors the registry's fake/scripted
    #: parity). A recipe double that feeds model-reply JSON through the real parser inherits this marker.
    PARSES_CLAIM_CHANNEL = True

    def __init__(
        self,
        script: dict[str, Verdict] | None = None,
        detect_script: dict[str, list[Observation]] | None = None,
        events_script: dict[str, list[tuple[str, float]]] | None = None,
    ) -> None:
        self._script = {**_SCRIPT, **(script or {})}
        self._detect = detect_script or {}
        # frame token -> [(event_type, confidence), ...] emitted at that frame's timestamp
        self._events = events_script or {}

    async def judge(
        self,
        condition: str,
        frame: Any,
        *,
        roi: Any = None,
        exemplar: Any = None,
        corrections: dict | None = None,
        frames: list | None = None,
        audio: Any = None,
        context: dict | None = None,
    ) -> Verdict:
        token = str(frame)
        # Item 4 logsum: an observe-step call returns this clip's step delta (scripted via "<frame>|step").
        if context and context.get("observe_step"):
            return self._script.get(f"{token}|step", Verdict(satisfied=False, confidence=0.5, evidence=""))
        # Caller-supplied ``context`` can route to a distinct scripted verdict
        # (a ``"<frame>|enriched"`` key), proving specialist facts actually *reach* judge — the
        # signal→specialist→judge-context seam — without claiming the fake understands anything.
        if context and f"{token}|enriched" in self._script:
            return self._script[f"{token}|enriched"]
        if corrections:
            corrected = f"{token}|corrected"
            if corrected in self._script:
                return self._script[corrected]
        # Perception-escalation wiring: a strip or audio clip can route to a distinct scripted verdict,
        # proving the PerceptionSpec assembly actually *reaches* judge (wiring, not intelligence). Audio
        # takes precedence over a strip when both are present (the fused/escalated read).
        if audio is not None and f"{token}|audio" in self._script:
            return self._script[f"{token}|audio"]
        if frames and f"{token}|strip" in self._script:
            return self._script[f"{token}|strip"]
        return self._script.get(token, _UNKNOWN)

    async def detect(self, frame: Any, goals: list[Goal]) -> list[Observation]:
        return list(self._detect.get(str(frame), []))

    async def detect_events(self, frames: list, goals: list[Goal]) -> list:
        """Emit scripted events off the newest ``(t, frame)`` in the window (wiring, not intelligence)."""
        if not frames:
            return []
        from ..events import Event

        t, frame = frames[-1]
        return [
            Event(type=etype, t=float(t), confidence=conf)
            for etype, conf in self._events.get(str(frame), [])
        ]

    async def summarize_process(self, condition: str, log: list[str]) -> str:
        """Item 4 logsum L1: a trivial order-preserving dedup of the log (wiring, not intelligence)."""
        seen: list[str] = []
        for s in log:
            if s and s not in seen:
                seen.append(s)
        return "; ".join(seen)


class FakeSpecialist:
    """Streams a fixed token sequence so reply wiring is testable offline."""

    def __init__(self, reply: str = "ok.") -> None:
        self._reply = reply

    async def stream(
        self, prompt: str, *, frame: Any = None, fast: bool = False
    ) -> AsyncIterator[str]:
        for token in self._reply.split(" "):
            yield token + " "


class FakeConductor:
    """Returns a scripted Decision (default: ignore). Authors no goals unless told to."""

    def __init__(self, decision: Decision | None = None) -> None:
        self._decision = decision or Decision(route="ignore")

    async def run(self, event: str, utterance: str, context: dict) -> Decision:
        return self._decision


class ScriptedConductor:
    """A scripted conductor double: maps an utterance SUBSTRING to a :class:`Decision` that authors a
    real goal, exactly as :class:`FakeVision` maps a frame token to a :class:`Verdict`.

    It exists because the goal-authoring path had no keyless test — :class:`FakeConductor` returns
    ``Decision(route="ignore")`` for every input, which is the ONE value indistinguishable from a real
    conductor correctly declining, so *"'tell me when the kettle boils' -> a goal exists"* could never
    be asserted offline. That gap is plausibly why the watch-authoring bug shipped: nothing failed.

    Wiring, not intelligence (standing rule 2): it does a lowercased substring match; it does not
    understand language. Its **no-match fallback is** ``Decision(route="respond_direct")`` — an honest
    "I heard you and authored nothing", **never** ``route="ignore"``: a scripted backend must never
    return the value that is indistinguishable from success.
    """

    def __init__(self, script: dict[str, Goal] | None = None) -> None:
        # utterance-substring -> the Goal to author when the utterance contains it (matched lowercased).
        self._script = dict(script or {})

    async def run(self, event: str, utterance: str, context: dict) -> Decision:
        from ..goals import GoalOp

        text = (utterance or "").lower()
        for key, goal in self._script.items():
            if key.lower() in text:
                return Decision(
                    route="respond_direct",
                    say=getattr(goal, "say", "") or f"Watching: {getattr(goal, 'condition', key)}",
                    goal_ops=[GoalOp(op="create", goal=goal)],
                )
        # HONEST no-author — NOT route="ignore" (which is indistinguishable from real success).
        return Decision(route="respond_direct")


class FakeSTT:
    """Emits scripted final STT events from a list of utterances."""

    def __init__(self, utterances: list[str] | None = None) -> None:
        self._utterances = utterances or []

    async def stream(self, audio: Any) -> AsyncIterator[STTEvent]:
        for i, text in enumerate(self._utterances):
            yield STTEvent(text=text, is_final=True, ts=float(i))


class FakeTTS:
    """Records pushed text + flushes so audio-out wiring is assertable."""

    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.flushes: int = 0

    def push(self, text: str) -> None:
        self.spoken.append(text)

    def flush(self) -> None:
        self.flushes += 1


class FakeVoiceTransport:
    """A deterministic, vendor-free :class:`~percept.contracts.VoiceTransport` — the reference impl and
    the test double for the (deprecated) voice seam. It owns no real audio; a test **drives** it and it
    **records** what the brain sent back (rule 2: proves wiring, not model intelligence).

    .. deprecated:: 0.6
        The voice embodiment moved app-side in the SDK diet — ``RuntimeSession.attach_transport`` is a
        raising stub, so nothing in the SDK orchestrates this fake anymore. It stays as the reference impl
        of the still-defined ``VoiceTransport`` contract and the registry ``make_transport("fake")`` seam.
        A voice application uses its own ``VoiceEngine`` fake instead — see
        ``docs/sources/voice-and-watchdog.md``.

    Drive (transport -> brain):
      * ``await emit_utterance(text=...)`` — deliver a completed user turn to the on_utterance sink.
      * ``await emit_barge_in()`` — fire barge-in.
      * ``await call_tool("arm_goal", {...})`` — simulate the transport's model invoking a tool.

    Record (brain -> transport): ``said`` (Reply objects from :meth:`say`), ``events``, ``interrupts``,
    ``tools`` (registered), plus ``started``/``stopped``. ``self_reply=True`` puts it in speech-to-speech
    mode where :meth:`say` is a no-op (the model already spoke)."""

    def __init__(self, *, self_reply: bool = False) -> None:
        self.self_reply = self_reply
        self.started = False
        self.stopped = False
        self.said: list[Reply] = []
        self.spoke_verbatim: list[bool] = []   # parallel to `said`: the verbatim flag per say()
        self.events: list[dict] = []
        self.state_pulses: list[str] = []      # sync_state summaries pushed by the brain
        self.interrupts = 0
        self.tools: dict[str, tuple[dict, Any]] = {}
        self.frame_provider: Any = None
        self._on_utterance: Any = None
        self._on_barge_in: Any = None
        self._on_event: Any = None
        self._on_user_speaking: Any = None
        self._on_bot_speaking: Any = None
        self._on_transcript: Any = None

    # -- lifecycle --------------------------------------------------------
    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    # -- up-channel registration -----------------------------------------
    def on_utterance(self, cb: Any) -> None:
        self._on_utterance = cb

    def on_barge_in(self, cb: Any) -> None:
        self._on_barge_in = cb

    def on_event(self, cb: Any) -> None:
        self._on_event = cb

    def on_user_speaking(self, cb: Any) -> None:
        self._on_user_speaking = cb

    def on_bot_speaking(self, cb: Any) -> None:
        self._on_bot_speaking = cb

    def on_transcript(self, cb: Any) -> None:
        self._on_transcript = cb

    # -- down-channel -----------------------------------------------------
    async def say(self, reply: Reply, *, verbatim: bool = False) -> None:
        # Records even in self_reply mode: a proactive FIRE is a real say() the brain drove, distinct from
        # a reply to a user turn. Tests assert on `said` + `spoke_verbatim`.
        self.said.append(reply)
        self.spoke_verbatim.append(verbatim)

    async def interrupt(self) -> None:
        self.interrupts += 1

    async def sync_state(self, summary: str) -> None:
        self.state_pulses.append(summary)

    # -- context ----------------------------------------------------------
    def set_frame_provider(self, provider: Any) -> None:
        self.frame_provider = provider

    def register_tool(self, name: str, schema: dict, handler: Any) -> None:
        self.tools[name] = (schema, handler)

    # -- test drivers (NOT part of the Protocol) -------------------------
    async def emit_utterance(self, text: str | None = None, *, audio: Any = None, t: float | None = None) -> None:
        ev = UtteranceEvent(text=text, audio=audio, t=t)
        await self._fire(self._on_utterance, ev)

    async def emit_barge_in(self) -> None:
        await self._fire(self._on_barge_in)

    async def emit_event(self, event: dict) -> None:
        await self._fire(self._on_event, event)

    async def emit_user_speaking(self, speaking: bool) -> None:
        await self._fire(self._on_user_speaking, speaking)

    async def emit_bot_speaking(self, speaking: bool) -> None:
        # Simulate the wire's audio-out edge (BotStarted/StoppedSpeakingFrame) — the honest A2V end.
        await self._fire(self._on_bot_speaking, speaking)

    async def emit_transcript(self, text: str) -> None:
        await self._fire(self._on_transcript, text)

    async def call_tool(self, name: str, args: dict) -> Any:
        _schema, handler = self.tools[name]
        res = handler(args)
        return await res if inspect.isawaitable(res) else res

    def current_frame(self) -> Any:
        return self.frame_provider() if self.frame_provider is not None else None

    @staticmethod
    async def _fire(cb: Any, *args: Any) -> None:
        if cb is None:
            return
        res = cb(*args)
        if inspect.isawaitable(res):
            await res


class ScriptedExtractor:
    """Deterministic, offline :class:`~percept.extraction.EvidenceExtractor` double.

    Reads a frame TOKEN (same convention as :class:`FakeVision`) and returns a pre-scripted
    :class:`~percept.extraction.Enrichment` — no OCR, no model, no randomness. Proves the
    extraction seam's WIRING (facts reach ``context=`` and the trace), never extraction
    intelligence (rule 2)."""

    def __init__(self, script: "dict[str, Any] | None" = None) -> None:
        from ..extraction import Enrichment

        self._Enrichment = Enrichment
        self._script = script or {}

    def extract(self, condition: str, frame: Any, roi: Any = None) -> Any:
        token = frame if isinstance(frame, str) else str(frame)
        hit = self._script.get(token)
        if hit is None:
            return self._Enrichment()
        if isinstance(hit, self._Enrichment):
            return hit
        return self._Enrichment(**hit)


__all__ = [
    "FakeVision",
    "FakeSpecialist",
    "FakeConductor",
    "ScriptedConductor",
    "FakeSTT",
    "FakeTTS",
    "FakeVoiceTransport",
    "ScriptedExtractor",
]
