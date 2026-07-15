"""Voice telemetry — honest A2V/V2V latency, correlated by incident ID, on monotonic clocks.

Port of the session-era ``_mark_pending_response``/``_on_bot_speaking`` semantics, template-side:

- **A2V (act-to-voice, proactive):** a fire was handed to the engine (enqueue) → the engine's
  output-STARTED edge (audio actually playing). Measuring to ``speak()``/enqueue instead understates
  true latency by the model's TTS first-token time — the number a realtime demo rests on.
- **V2V (voice-to-voice, reactive):** the user's turn ENDED (local VAD fall / synthesized turn-end)
  → the engine's output-started edge of the reply.

One window at a time, closed ONCE: subsequent output-started edges of the same playout find no pending
stamp and are ignored (the session-era rule). Correlation: a proactive stamp carries the (coalesced)
``incident_ids`` from the :class:`~engine_adapter.SpeechRequest`, so every fired incident gets its own
latency record even when several were rendered as one utterance.

All timestamps come from an injected monotonic ``clock`` — never wall time (NTP steps would corrupt
latency math), and injectable so tests are deterministic.
"""
from __future__ import annotations

import time
from collections import Counter
from typing import Any, Callable, Dict, List, Tuple


class VoiceTelemetry:
    """In-process counters + latency records; ``snapshot()`` feeds a heartbeat/dashboard."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._pending: Tuple[float, str, tuple] | None = None   # (since, kind, incident_ids)
        #: (incident_id or "", kind, latency_s) — one row per correlated incident (or one anonymous
        #: row for a reactive reply, which has no incident).
        self.latencies: List[Tuple[str, str, float]] = []
        self.counters: Counter = Counter()

    # -- window open -----------------------------------------------------------
    def mark_pending(self, kind: str, *, incident_ids: tuple = ()) -> None:
        """Stamp the moment a spoken response was TRIGGERED (a user turn ended → ``"reactive"``, or a
        fire was enqueued → ``"proactive"``). The engine's output-started edge closes the window."""
        self._pending = (self._clock(), kind, tuple(incident_ids))

    # -- window close (wire to engine.on_output_started) ------------------------
    def on_output_started(self) -> None:
        """The engine's voice output actually began — the HONEST A2V/V2V end. Records ONCE per window."""
        if self._pending is None:
            return   # spurious/continuation audio: no pending stamp → nothing recorded
        since, kind, ids = self._pending
        self._pending = None
        dt = max(0.0, self._clock() - since)
        self.counters[f"voice.a2v_latency_s.{kind}.count"] += 1
        for incident_id in (ids or ("",)):
            self.latencies.append((incident_id, kind, dt))

    # -- event counters ----------------------------------------------------------
    def incident(self, intent: Any) -> None:
        self.counters["fires.intents"] += 1

    def delivered(self, intent: Any, *, channel: str) -> None:
        self.counters[f"fires.delivered.{channel}"] += 1

    def dropped(self, intent: Any, *, reason: str) -> None:
        self.counters[f"fires.dropped.{reason}"] += 1

    def playback_cancelled(self, *, reason: str) -> None:
        self.counters[f"playback.cancelled.{reason}"] += 1
        self._pending = None   # the playout this window was waiting on is gone

    # -- read side -----------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        by_kind: Dict[str, List[float]] = {}
        for _id, kind, dt in self.latencies:
            by_kind.setdefault(kind, []).append(dt)
        out: Dict[str, Any] = dict(self.counters)
        for kind, values in by_kind.items():
            out[f"voice.a2v_latency_s.{kind}.mean"] = sum(values) / len(values)
        out["latencies"] = list(self.latencies)
        return out
