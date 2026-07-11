"""Voice-escalation seam — `run(..., on_utterance=...)` endpoints a complete spoken utterance.

The ambient-voice primitive: Percept owns the mic (`audio_source`), and when the audio pump endpoints a
finished utterance it hands the utterance's PCM clip to `on_utterance` — the hook the voice layer turns
into an ephemeral reply. No push-to-talk, no fixed record window. Guards: fires ONCE per utterance with
a non-empty clip; a silent stream never fires; and with no `on_utterance` the audio path is byte-for-byte.
"""
from __future__ import annotations

import asyncio
import unittest

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.backends.fake import FakeVision
from percept.contracts import AudioFrame
from percept.edge.tier0 import gray_frame
from percept.runtime import RuntimeStats, run

W, H = 8, 8
STILL = gray_frame(W, H, 0)  # identical frames => zero motion => the visual gate never escalates


def _core() -> Percept:
    vision = FakeVision(script={"quiet": Verdict(satisfied=False, confidence=0.9, evidence="nothing")})
    core = Percept.create(vision=vision, discover_plugins=False)
    core.add_goal(Goal(id="listen", condition="someone is speaking", say="heard you"))
    return core


def _frame(byte: int) -> AudioFrame:
    return AudioFrame(pcm=bytes([byte, byte]), sample_rate=16000, channels=1)


def _audio_from_energies(energies, *, dt=0.1, offset=0.05):
    """(t, AudioFrame, energy) rows: speech frames carry loud PCM, silence carries zeros."""
    rows = []
    for i, e in enumerate(energies):
        rows.append((i * dt + offset, _frame(0x40 if e > 0 else 0x00), e))
    return rows


async def _collect(core, video, **kw):
    return [inc async for inc in run(core, source=video, **kw)]


class VoiceEscalationTests(unittest.TestCase):
    def test_one_utterance_fires_on_utterance_once(self):
        core = _core()
        # 3 silence, 5 speech, 9 silence -> one endpoint (see test_vad).
        audio = _audio_from_energies([0.0] * 3 + [0.5] * 5 + [0.0] * 9)
        video = [(float(i), STILL, "quiet") for i in range(30)]  # longer than audio so the pump drains
        heard = []
        stats = RuntimeStats()
        asyncio.run(_collect(core, video, audio_source=audio, on_utterance=lambda a, t: heard.append((a, t)), stats=stats))

        self.assertEqual(len(heard), 1)
        self.assertEqual(stats.utterances, 1)
        clip, _t = heard[0]
        self.assertIsInstance(clip, AudioFrame)
        self.assertGreater(len(clip.pcm), 0)  # the endpointed clip carries real PCM

    def test_async_on_utterance_awaited(self):
        core = _core()
        audio = _audio_from_energies([0.0] * 2 + [0.5] * 4 + [0.0] * 9)
        video = [(float(i), STILL, "quiet") for i in range(30)]
        heard = []

        async def on_utt(a, t):
            heard.append(a)

        asyncio.run(_collect(core, video, audio_source=audio, on_utterance=on_utt))
        self.assertEqual(len(heard), 1)

    def test_silent_stream_never_fires(self):
        core = _core()
        audio = _audio_from_energies([0.0] * 20)  # all silence
        video = [(float(i), STILL, "quiet") for i in range(25)]
        heard = []
        stats = RuntimeStats()
        asyncio.run(_collect(core, video, audio_source=audio, on_utterance=lambda a, t: heard.append(a), stats=stats))
        self.assertEqual(heard, [])
        self.assertEqual(stats.utterances, 0)

    def test_no_on_utterance_is_byte_for_byte(self):
        # audio_source present but no on_utterance -> the endpointer is never built; audio path unchanged.
        core = _core()
        audio = _audio_from_energies([0.0] * 3 + [0.5] * 5 + [0.0] * 9)
        video = [(float(i), STILL, "quiet") for i in range(30)]
        stats = RuntimeStats()
        asyncio.run(_collect(core, video, audio_source=audio, stats=stats))
        self.assertEqual(stats.utterances, 0)
        self.assertGreater(stats.audio_chunks, 0)  # the pump still filled the AudioWindow


if __name__ == "__main__":
    unittest.main()
