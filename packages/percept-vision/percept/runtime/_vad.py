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


#: The runtime's documented mic cadence, seconds per chunk — one ``push()`` per chunk. Mirrors
#: ``AudioSourceConfig.chunk_s`` and ``percept.runtime.worker._DEFAULT_CHUNK_DT_S``.
DEFAULT_CHUNK_S = 0.25


def hangover_frames_for(seconds: float, *, chunk_s: float = DEFAULT_CHUNK_S) -> int:
    """Frames of trailing silence equivalent to ``seconds`` at a ``chunk_s`` capture cadence.

    ``hangover_frames`` is a FRAME COUNT, but every question a caller actually has about it is in
    SECONDS ("how long must someone stop talking before I get their turn?"). At the runtime's
    ``chunk_s=0.25`` this is a factor of four, and the shipped ``hangover_frames=8`` is therefore
    **2.00 seconds** of continuous silence — see :class:`EnergyVad` for why that matters.
    """
    if (
        not isinstance(seconds, (int, float))
        or isinstance(seconds, bool)
        or not math.isfinite(seconds)
        or seconds <= 0
    ):
        raise ValueError("seconds must be finite and positive")
    if (
        not isinstance(chunk_s, (int, float))
        or isinstance(chunk_s, bool)
        or not math.isfinite(chunk_s)
        or chunk_s <= 0
    ):
        raise ValueError("chunk_s must be finite and positive")
    return max(1, round(float(seconds) / float(chunk_s)))


class EnergyVad:
    """Streaming energy-threshold endpointer (port of ``EnergyVad``, vad.js:42).

    Feed it per-frame energy via :meth:`push` (or raw PCM via :meth:`push_samples`). It tracks whether
    speech is active and fires an ``endpoint`` once ``hangover_frames`` consecutive low-energy frames
    follow a speech run of at least ``min_speech_frames`` frames.

    - ``threshold`` — MEAN-SQUARE ENERGY above this counts as speech (default ``0.01``). **This is not
      the knob a caller types.** :func:`percept.runtime.run`'s ``utterance_threshold`` is an RMS
      AMPLITUDE and is SQUARED (``_amplitude_threshold_to_energy``) before it lands here: the shipped
      ``utterance_threshold=0.01`` arrives as energy ``1e-4``, two orders of magnitude below what the
      number looks like. Read every comparison against this field in the energy domain.
    - ``hangover_frames`` — silence FRAMES (not seconds) after speech before an endpoint (default
      ``8``). One frame is one capture chunk, so the wall-clock hangover is
      ``hangover_frames * chunk_s``: at the runtime's ``chunk_s=0.25`` the shipped default demands
      **2.00 seconds of continuous silence** before a turn is considered finished. Use
      :func:`hangover_frames_for` to express it in seconds against a real cadence.

      MEASURED CONSEQUENCE (AUDIO-VAD-HANGOVER-UNITS). Continuous narration never contains a 2-second
      silence, so it barely endpoints at all: over a real speech corpus the endpoint count stayed at
      or below 4 across EVERY ``utterance_threshold`` from 0.001 to 0.3, while the speech fraction
      moved from 0.17 to 0.99. The threshold is not the lever — the hangover is, and no amount of
      threshold tuning can compensate for it. The default is left at 8 deliberately: this class is a
      byte-exact port of ``percept-edge/src/vad.js`` whose JS tests are the shared oracle, so the
      number cannot move on one side alone, and no measured hangover sweep exists to justify a
      specific replacement. A deployment that needs conversational endpointing should pass
      ``utterance_hangover_frames=hangover_frames_for(0.6)`` (== 2 frames at 4 chunks/s) today.
    - ``min_speech_frames`` — minimum speech frames to count as an utterance (default ``3``); a shorter
      blip is dropped silently and the endpointer re-arms. Also frames: ``3`` is 0.75 s at
      ``chunk_s=0.25``.
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


__all__ = [
    "EnergyVad",
    "VadEvent",
    "frame_energy",
    "endpoint_energy_sequence",
    "hangover_frames_for",
    "DEFAULT_CHUNK_S",
]
