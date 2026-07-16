"""PR 1.2 — audio ingress in the runtime worker.

Regression guard for a stranded capability: `Percept.push_audio` (the only feeder of the
`AudioWindow` that `modality="fused"/"audio"` goals judge over) had ZERO callers, so every fused
judge silently degraded to vision-only. These tests assert the worker now (a) fills the AudioWindow
from an `audio_source`, and (b) lets an off-camera sound onset escalate a still frame the visual
motion gate would skip — while the default (no `audio_source`) stays byte-for-byte video-only.
"""
from __future__ import annotations

import array
import asyncio
import math
import os
import tempfile
import unittest
import wave

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.backends.fake import FakeVision
from percept.contracts import AudioFrame
from percept.edge.tier0 import gray_frame
from percept.runtime import RuntimeStats, run
from percept.runtime.audio import open_audio_source

W, H = 8, 8
STILL = gray_frame(W, H, 0)  # identical frames => zero motion => the visual gate skips


def _quiet_core() -> Percept:
    vision = FakeVision(script={"quiet": Verdict(satisfied=False, confidence=0.9, evidence="nothing")})
    core = Percept.create(vision=vision, discover_plugins=False)
    core.add_goal(Goal(id="cry", condition="the baby is crying", say="Baby is awake"))
    return core


async def _collect(core, source, **kw):
    return [inc async for inc in run(core, source=source, **kw)]


class OpenAudioSourceTests(unittest.TestCase):
    def test_wav_source_yields_audioframe_and_energy(self):
        sr = 16_000
        tone = array.array("h", [int(12000 * math.sin(2 * math.pi * 440 * i / sr)) for i in range(sr)])
        sil = array.array("h", [0] * sr)
        path = tempfile.mktemp(suffix=".wav")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(tone.tobytes())
            w.writeframes(sil.tobytes())

        async def drain():
            return [(t, frame, e) async for (t, frame, e) in open_audio_source(path, chunk_s=0.25)]

        try:
            rows = asyncio.run(drain())
        finally:
            os.unlink(path)

        self.assertEqual(len(rows), 8)  # 2s / 0.25s
        self.assertTrue(all(isinstance(f, AudioFrame) for _, f, _ in rows))
        self.assertGreater(max(e for _, _, e in rows[:4]), 0.2)  # tone is loud
        self.assertLess(max(e for _, _, e in rows[-4:]), 0.01)  # silence is ~0


class NormalizeUriTests(unittest.TestCase):
    def test_mic_uri_mapping(self):
        from percept.runtime.audio import _normalize_uri

        self.assertIsNone(_normalize_uri("mic"))            # bare mic -> DEFAULT input device (not 0)
        self.assertIsNone(_normalize_uri("mic:default"))    # explicit default
        self.assertIsNone(_normalize_uri("mic://default"))
        self.assertEqual(_normalize_uri("mic:0"), 0)        # explicit index preserved
        self.assertEqual(_normalize_uri("mic:2"), 2)
        self.assertEqual(_normalize_uri("mic://1"), 1)
        self.assertEqual(_normalize_uri("clip.wav"), "clip.wav")  # a path passes through


class RuntimeAudioIngressTests(unittest.TestCase):
    def test_audio_window_fills_from_audio_source(self):
        core = _quiet_core()
        video = [(float(i), STILL, "quiet") for i in range(6)]
        audio = [(float(i) + 0.1, AudioFrame(pcm=b"\x00\x00", sample_rate=16000), 0.0) for i in range(4)]
        stats = RuntimeStats()
        asyncio.run(_collect(core, video, audio_source=audio, stats=stats))
        # the pump drove push_audio -> the AudioWindow (previously always empty) now has chunks
        self.assertGreater(stats.audio_chunks, 0)
        self.assertTrue(core._audio_window("camera")._chunks)

    def test_onset_admits_a_still_frame_no_motion(self):
        # Control: static video, NO audio -> the visual gate never escalates.
        ctrl_stats = RuntimeStats()
        ctrl_video = [(float(i), STILL, "quiet") for i in range(6)]
        asyncio.run(_collect(_quiet_core(), ctrl_video, stats=ctrl_stats))
        self.assertEqual(ctrl_stats.escalations, 0)

        # With audio: an energy onset (0.0 arms below `low`, 0.5 crosses `high`) admits a still frame.
        stats = RuntimeStats()
        video = [(float(i), STILL, "quiet") for i in range(6)]
        audio = [
            (0.1, AudioFrame(pcm=b"\x00\x00", sample_rate=16000), 0.0),
            (1.1, AudioFrame(pcm=b"\xff\x7f", sample_rate=16000), 0.5),  # loud -> onset
            (2.1, AudioFrame(pcm=b"\x00\x00", sample_rate=16000), 0.0),
            (3.1, AudioFrame(pcm=b"\x00\x00", sample_rate=16000), 0.0),
        ]
        asyncio.run(_collect(_quiet_core(), video, audio_source=audio, stats=stats))
        self.assertGreaterEqual(stats.escalations, 1)  # audio woke the gate despite zero motion
        self.assertGreaterEqual(stats.audio_wakes, 1)

    def test_video_only_path_unchanged_without_audio_source(self):
        # No audio_source -> audio counters stay 0 (byte-for-byte the pre-PR path).
        stats = RuntimeStats()
        video = [(float(i), STILL, "quiet") for i in range(4)]
        asyncio.run(_collect(_quiet_core(), video, stats=stats))
        self.assertEqual(stats.audio_chunks, 0)
        self.assertEqual(stats.audio_wakes, 0)


if __name__ == "__main__":
    unittest.main()
