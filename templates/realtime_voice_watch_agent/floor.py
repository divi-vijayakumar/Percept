"""Floor — who may speak into the room right now. Application policy, deliberately NOT in the SDK.

Three independent signals (see ``media.py`` for where each comes from):

- ``user_speaking``  — the LOCAL energy VAD on the AEC'd mic. It STAYS ACTIVE during assistant
  playback (that is what makes barge-in detectable) and is never derived from the gated Percept
  world-audio energy (gating that branch during playout would blind barge-in exactly when it matters).
- ``wire_speaking``  — the engine's output edges (audio-out started/stopped). Engine speaking-edges
  give ``wire_speaking`` ONLY — they never detect the user.
- ``fire_playing``   — the renderer's state: a Percept FIRE is currently being rendered as speech.

Priority ladder: **user > trusted interrupt > wire reply > deferred.**

- The user always wins the floor: user speech starting during fire playback is a barge-in — playback
  is cancelled (ported from the session-era barge-in handler; the *detection* lives in the app now).
- A trusted ``preempt="interrupt"`` intent (the policy already clamped untrusted authors to
  ``"queue"`` — see ``percept.incidents.AttentionPolicy``) cuts into whatever the wire is doing and
  even into a user turn (a severe fire is worth the rudeness — ported ``_speak_fire`` semantics).
- ``"queue"`` intents hold behind the current turn/playout and flush when the floor frees
  (``_flush_queued_fires`` port): TTL drop-if-stale — a late nudge is worse than none.
- ``"drop"`` intents speak only into a free floor; a busy floor drops them silently.

Coalescing: when several queued intents flush at once they are rendered as ONE utterance whose
:class:`~engine_adapter.SpeechRequest` carries **every** incident ID — telemetry correlation must
never lose an incident to coalescing. A ``verbatim`` intent is never coalesced (verbatim means
exactly-these-words; gluing two scripts together silently violates both).
"""
from __future__ import annotations

import time
from typing import Any, Callable, List, Tuple


class Floor:
    """Arbitrates the three speaking signals; grants render slots to the :class:`~fire_renderer.FireRenderer`."""

    def __init__(self, engine: Any, renderer: Any, *,
                 clock: Callable[[], float] = time.monotonic, telemetry: Any = None) -> None:
        self._engine = engine
        self._renderer = renderer
        self._clock = clock
        self._telemetry = telemetry
        self.user_speaking = False
        self.wire_speaking = False
        self._queued: List[Tuple[Any, float]] = []   # (intent, queued_at) held behind the turn/playout

    # -- the three signals -------------------------------------------------
    @property
    def fire_playing(self) -> bool:
        """Renderer state — a fire utterance was handed to the engine and audio-out hasn't ended."""
        return bool(getattr(self._renderer, "fire_playing", False))

    def busy(self) -> bool:
        return self.user_speaking or self.wire_speaking or self.fire_playing

    async def set_user_speaking(self, speaking: bool) -> None:
        """The LOCAL VAD edge (and the engine's synthesized turn-end, which only emits ``False``)."""
        speaking = bool(speaking)
        rising = speaking and not self.user_speaking
        self.user_speaking = speaking
        if rising and self.fire_playing:
            # BARGE-IN: the user cut into fire playback — the user outranks everything (top of the
            # ladder). Stop playout; the interrupted fire is NOT re-queued (the moment passed).
            await self._cancel_playback(reason="barge_in")
        if not speaking:
            if self._telemetry is not None:
                # Turn ended → the reactive A2V/V2V clock starts: an S2S wire now renders its reply and
                # the honest latency runs until the engine's output-started edge (audio-out).
                self._telemetry.mark_pending("reactive")
            await self._flush()   # turn ended → deliver anything queued behind it

    async def set_wire_speaking(self, speaking: bool) -> None:
        """The engine's output edges. Wire these to ``engine.on_output_started/stopped``."""
        speaking = bool(speaking)
        was = self.wire_speaking
        self.wire_speaking = speaking
        if was and not speaking:
            await self._flush()   # playout ended → the floor freed → deliver deferred fires

    # -- intent intake -------------------------------------------------------
    async def submit(self, intent: Any) -> None:
        """Route one EFFECTIVE :class:`~percept.AttentionIntent` (already policy-clamped) to speech.

        The floor trusts ``intent.preempt`` as *granted* authority — clamping requested→effective is
        the SDK policy's job (``attention_intent_for``); re-deciding it here would fork the trust
        model across two codebases."""
        preempt = getattr(intent, "preempt", "queue")
        if preempt == "interrupt":
            # Trusted interrupt: cut into wire replies / fire playback, and even into a user turn
            # (ported severe-fire semantics). The user can still barge in on the RESULT.
            if self.wire_speaking or self.fire_playing:
                await self._cancel_playback(reason="preempted")
            await self._speak([intent])
            return
        if preempt == "drop":
            if self.busy():
                if self._telemetry is not None:
                    self._telemetry.dropped(intent, reason="floor_busy")
                return
            await self._speak([intent])
            return
        # "queue" (the default, and what clamped interrupts become): hold behind the current turn.
        if self.busy():
            self._queued.append((intent, self._clock()))
            return
        await self._speak([intent])

    # -- internals -------------------------------------------------------------
    async def _flush(self) -> None:
        if not self._queued or self.busy():
            return
        queued, self._queued = self._queued, []
        now = self._clock()
        fresh: list[Any] = []
        for intent, queued_at in queued:
            ttl = getattr(intent, "ttl_s", None)
            if ttl is not None and (now - queued_at) > float(ttl):
                if self._telemetry is not None:
                    self._telemetry.dropped(intent, reason="stale")
                continue   # went stale while the user talked → drop (a late nudge is worse than none)
            fresh.append(intent)
        if not fresh:
            return
        # Coalesce the flush into as few utterances as possible WITHOUT gluing verbatim scripts:
        # consecutive non-verbatim intents merge into one render (all incident IDs preserved).
        batch: list[Any] = []
        for intent in fresh:
            if getattr(intent, "verbatim", False):
                if batch:
                    await self._speak(batch)
                    batch = []
                await self._speak([intent])
            else:
                batch.append(intent)
        if batch:
            await self._speak(batch)

    async def _speak(self, intents: list) -> None:
        await self._renderer.speak(intents)

    async def _cancel_playback(self, *, reason: str) -> None:
        try:
            await self._engine.cancel_output()
        except Exception:  # noqa: BLE001 - an engine that can't cancel is fine (native barge-in)
            pass
        cancelled = getattr(self._renderer, "playback_cancelled", None)
        if cancelled is not None:
            cancelled()
        if self._telemetry is not None:
            self._telemetry.playback_cancelled(reason=reason)
