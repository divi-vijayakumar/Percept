"""FireRenderer — turn an EFFECTIVE AttentionIntent into an actual delivery on a concrete channel.

Channel selection is the APPLICATION's decision: the SDK's :class:`~percept.AttentionIntent` carries
``preferred_channel`` only as a *hint* — the app folds it with the engine's capabilities and its own
authorization posture:

- ``"speech"`` (the default) → the voice engine renders it (``engine.speak(SpeechRequest)``); WHEN it
  may speak is the :class:`~floor.Floor`'s decision, which is why speech intents route through the
  floor and the other channels do not (a webhook never contends for the room's audio).
- ``"webhook"`` → posted via ``percept.runtime.sinks.webhook`` (SSRF-guarded by the SDK's UrlPolicy).
- ``"log"``     → appended to the app log (and the fallback when a preferred channel isn't available —
  a fire is never silently lost to a missing capability).
- ``"action"``  → **documented out of scope.** A robot/actuator action is not just another notification
  modality: it needs its own authorization contract (consent profile, dry-run, reversibility class)
  before an incident may trigger it. Until that contract exists, ``action`` intents degrade to ``log``
  with an explicit note — never a silent actuation, never a silent drop.
"""
from __future__ import annotations

import sys
import time
from typing import Any, Callable

from engine_adapter import SpeechRequest

_SPEECH = ("speech", "voice", "speak")


class FireRenderer:
    """Renders granted intents; owns the ``fire_playing`` signal the floor reads."""

    def __init__(self, engine: Any, *, webhook_sink: Any = None,
                 log: Callable[[str], None] | None = None,
                 clock: Callable[[], float] = time.monotonic, telemetry: Any = None) -> None:
        self._engine = engine
        self._webhook = webhook_sink       # e.g. percept.runtime.sinks.webhook(url) — None → unavailable
        self._log = log or (lambda line: print(line, file=sys.stderr, flush=True))
        self._clock = clock
        self._telemetry = telemetry
        #: True from the moment a fire utterance is handed to the engine until the engine's next
        #: output-stopped edge (wired in :meth:`bind_engine_edges`). On a held S2S wire the engine
        #: cannot label which audio-out belongs to which request, so this is the honest approximation
        #: the session-era code also lived with.
        self.fire_playing = False

    def bind_engine_edges(self) -> None:
        """Subscribe the output-stopped edge that clears :attr:`fire_playing`."""
        self._engine.on_output_stopped(self._on_output_stopped)

    def _on_output_stopped(self) -> None:
        self.fire_playing = False

    def playback_cancelled(self) -> None:
        """The floor cancelled playout (barge-in / preemption) — clear the signal immediately."""
        self.fire_playing = False

    # -- channel selection ---------------------------------------------------
    def select_channel(self, intent: Any) -> str:
        """Fold the intent's hint with actual capabilities. Never returns an unavailable channel."""
        preferred = (getattr(intent, "preferred_channel", None) or "speech").lower()
        if preferred in _SPEECH:
            return "speech" if self._engine is not None else ("webhook" if self._webhook else "log")
        if preferred == "webhook":
            return "webhook" if self._webhook is not None else "log"
        if preferred == "action":
            # Out of scope by design — see module docstring. Loud degrade, never silent actuation.
            self._log("[fire-renderer] 'action' channel requested but actions need their own "
                      "authorization contract (out of scope) — degrading to log.")
            return "log"
        return "log"

    # -- delivery --------------------------------------------------------------
    async def speak(self, intents: list) -> None:
        """Render one floor-granted batch as a single utterance (coalesced IDs preserved)."""
        intents = [i for i in intents if getattr(i, "message", "")]
        if not intents or self._engine is None:
            return
        text = " ".join(getattr(i, "message", "") for i in intents)
        verbatim = len(intents) == 1 and bool(getattr(intents[0], "verbatim", False))
        ids = tuple(getattr(i, "incident_id", "") for i in intents if getattr(i, "incident_id", ""))
        expires: float | None = None
        ttls = [getattr(i, "ttl_s", None) for i in intents]
        if all(t is not None for t in ttls) and ttls:
            expires = self._clock() + min(float(t) for t in ttls)  # informational; the floor owns TTL
        request = SpeechRequest(text=text, verbatim=verbatim, incident_ids=ids, expires_at=expires)
        self.fire_playing = True
        if self._telemetry is not None:
            # A2V clock for a spoken fire: enqueue → audio-out (the engine's output-started edge).
            self._telemetry.mark_pending("proactive", incident_ids=ids)
        print(f"[fire-renderer] enqueue speech incidents={ids} text={text!r}", file=sys.stderr)
        try:
            await self._engine.speak(request)
        except Exception as exc:  # noqa: BLE001 - a failed delivery must not strand fire_playing
            # If speak() raises, fire_playing would latch True forever (the output-stopped edge that
            # clears it never comes) — the floor would then think a fire is always playing and defer
            # everything after. Clear it and surface the failure loudly.
            self.fire_playing = False
            print(f"[fire-renderer] speak FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)

    async def deliver(self, intent: Any, channel: str) -> None:
        """Deliver on a non-contending channel (no floor involvement)."""
        if channel == "webhook" and self._webhook is not None:
            res = self._webhook(intent)
            if hasattr(res, "__await__"):
                await res
            if self._telemetry is not None:
                self._telemetry.delivered(intent, channel="webhook")
            return
        self._log(f"[fire] {getattr(intent, 'goal_id', '?')}: {getattr(intent, 'message', '')} "
                  f"(incident={getattr(intent, 'incident_id', '?')})")
        if self._telemetry is not None:
            self._telemetry.delivered(intent, channel="log")
