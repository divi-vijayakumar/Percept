"""Energy-threshold voice endpointer — the Python twin of ``percept-edge/src/vad.js``.

``run()``'s audio pump detects when a spoken *utterance* STARTS (energy rises above a threshold) and,
critically, when it ENDS (a run of low-energy frames — the speaker stopped). The endpoint is the
voice-escalation seam: a *complete* utterance is what gets handed to ``on_utterance`` (→ the voice
layer's ephemeral reply). Onset detection alone (``AcousticOnset``) can't say when a turn is *done*.

Faithful, pure-stdlib port of ``EnergyVad`` (vad.js:42) — the JS tests are the oracle, so the two
state machines produce identical endpoints on identical energy sequences. No model, no numpy, no clock:
same energy sequence → same endpoints, deterministic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


def frame_energy(samples: Sequence[float]) -> float:
    """Short-time energy (mean of squares) of a frame of PCM samples in ``[-1, 1]`` (→ energy in
    ``[0, 1]``). Mirrors ``frameEnergy`` (vad.js:19)."""
    n = len(samples)
    if n == 0:
        return 0.0
    return sum(s * s for s in samples) / n


@dataclass(frozen=True)
class VadEvent:
    """One frame's endpointer verdict — the twin of the JS ``{speech, started, endpoint, frame}``.

    ``started`` is the rising edge (silence → speech); ``endpoint`` is the frame the utterance is
    considered finished (enough trailing silence after a long-enough speech run)."""

    speech: bool
    started: bool
    endpoint: bool
    frame: int


class EnergyVad:
    """Streaming energy-threshold endpointer (port of ``EnergyVad``, vad.js:42).

    Feed it per-frame energy via :meth:`push` (or raw PCM via :meth:`push_samples`). It tracks whether
    speech is active and fires an ``endpoint`` once ``hangover_frames`` consecutive low-energy frames
    follow a speech run of at least ``min_speech_frames`` frames.

    - ``threshold`` — energy above this counts as speech (default ``0.01``).
    - ``hangover_frames`` — silence frames after speech before an endpoint (default ``8``).
    - ``min_speech_frames`` — minimum speech frames to count as an utterance (default ``3``); a shorter
      blip is dropped silently and the endpointer re-arms.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.01,
        hangover_frames: int = 8,
        min_speech_frames: int = 3,
    ) -> None:
        if (
            not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not math.isfinite(threshold)
            or not 0.0 <= threshold <= 1.0
        ):
            raise ValueError("threshold must be finite and within [0, 1]")
        if (
            not isinstance(hangover_frames, int)
            or isinstance(hangover_frames, bool)
            or hangover_frames <= 0
        ):
            raise ValueError("hangover_frames must be a positive integer")
        if (
            not isinstance(min_speech_frames, int)
            or isinstance(min_speech_frames, bool)
            or min_speech_frames <= 0
        ):
            raise ValueError("min_speech_frames must be a positive integer")
        self.threshold = float(threshold)
        self.hangover_frames = hangover_frames
        self.min_speech_frames = min_speech_frames
        self.reset()

    def reset(self) -> "EnergyVad":
        self._speaking = False
        self._speech_run = 0  # consecutive speech frames in the current utterance
        self._silence_run = 0  # consecutive silence frames since the last speech
        self._frame_index = -1
        return self

    @property
    def speaking(self) -> bool:
        return self._speaking

    def push(self, energy: float) -> VadEvent:
        """Push one frame's energy and return its :class:`VadEvent`. Mirrors ``pushEnergy`` exactly."""
        if (
            not isinstance(energy, (int, float))
            or isinstance(energy, bool)
            or not math.isfinite(energy)
            or not 0.0 <= energy <= 1.0
        ):
            raise ValueError("energy must be finite and within [0, 1]")
        self._frame_index += 1
        is_speech = energy >= self.threshold
        started = False
        endpoint = False

        if is_speech:
            if not self._speaking:
                self._speaking = True
                self._speech_run = 0
                started = True
            self._speech_run += 1
            self._silence_run = 0
        elif self._speaking:
            self._silence_run += 1
            # Endpoint only if we accumulated a real utterance AND have now seen enough trailing silence.
            if self._silence_run >= self.hangover_frames and self._speech_run >= self.min_speech_frames:
                self._speaking = False
                endpoint = True
                self._speech_run = 0
                self._silence_run = 0
            elif self._silence_run >= self.hangover_frames:
                # Too short to be an utterance: drop it silently, re-arm.
                self._speaking = False
                self._speech_run = 0
                self._silence_run = 0

        return VadEvent(speech=is_speech, started=started, endpoint=endpoint, frame=self._frame_index)

    def push_samples(self, samples: Sequence[float]) -> VadEvent:
        """Push one frame of raw PCM samples — computes energy then delegates to :meth:`push`."""
        return self.push(frame_energy(samples))


def endpoint_energy_sequence(energies: Sequence[float], **opts) -> list[int]:
    """Run the endpointer over a whole energy sequence and return the frame indices where an endpoint
    fired (batch API for fixtures/tests). Twin of ``endpointEnergySequence`` (vad.js:131)."""
    vad = EnergyVad(**opts)
    return [ev.frame for ev in (vad.push(e) for e in energies) if ev.endpoint]


__all__ = ["EnergyVad", "VadEvent", "frame_energy", "endpoint_energy_sequence"]
