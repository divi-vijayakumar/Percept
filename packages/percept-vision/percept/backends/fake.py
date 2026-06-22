"""Deterministic, offline, key-free backends.

Fakes are load-bearing: they make ``pip install`` -> run work with no keys, power the L1 CI
lane, and isolate *our logic* from *the vendor*. They are **scripted, never random**.

The hard rule (standing rule 2): a fake proves seam wiring and our logic — it must **never**
be used to assert backend *intelligence*. Real-model behaviour is tested in L2/L3 only. So
``FakeVision`` reads a token off the frame and returns a pre-scripted verdict; it does not
"understand" anything.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from ..adapters import Decision, Goal, Observation, STTEvent, Verdict

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
    ) -> Verdict:
        token = str(frame)
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


__all__ = [
    "FakeVision",
    "FakeSpecialist",
    "FakeConductor",
    "FakeSTT",
    "FakeTTS",
]
