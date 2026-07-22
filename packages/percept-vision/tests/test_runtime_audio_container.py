"""MP4 (video-container) audio wired into the existing fused/acoustic channel.

The fused audio channel (``runtime/audio.py``'s ``open_audio_source``, the worker's
``audio_source=``/``audio_gate=`` pump, ``Percept.push_audio`` -> ``AudioWindow``) already existed
end-to-end for mic/wav sources. The one gap: a video container's (mp4/mov/mkv/webm) audio track never
reached ``open_audio_source`` — a bare ``wave.open()`` on an mp4 raises, so a fused goal watching a
video file silently degraded to vision-only. These tests cover the fix:

  (a) ``ContainerDetectionTests`` / ``FFmpegMissingTests`` — URI/dispatch unit tests and the typed
      ``FFmpegNotFoundError`` (never a silent no-audio run), via monkeypatch. No ffmpeg needed.
  (b) ``AVOnsetIntegrationTests.test_audio_onset_admits_and_fires`` — a REAL worker loop: a scripted
      (static) video stream + a real ``.wav``-backed ``audio_source`` carrying a synthetic energy onset
      + a fused-modality goal + ``ScriptedJudge``. Asserts the onset admits a frame the (motionless)
      visual gate would skip (``audio_wakes > 0``), the judge received audio, and the goal fires on the
      rising edge.
  (c) ``AVOnsetIntegrationTests.test_same_scenario_without_audio_source_never_fires`` — the identical
      scenario minus ``audio_source`` — proves audio was the DECIDING signal (no motion, no audio ->
      zero judge calls -> no fire).
  (d) ``AutoWireContainerSourceTests`` — the file-driven run wiring: a container-suffixed ``source=``
      auto-derives ``audio_source`` ONLY when an armed goal's modality is fused/audio (no ffmpeg/demux
      subprocess spawned for a vision-only goal set — the video-only path stays byte-for-byte), and an
      explicit ``audio_source`` always wins. Uses monkeypatched capture entrypoints, so no ffmpeg/cv2
      file I/O is required in CI.

Zero API spend throughout — ``ScriptedJudge``/patched capture functions only, never a real vision
backend.
"""
from __future__ import annotations

import array
import asyncio
import math
import os
import queue
import shutil
import tempfile
import threading
import unittest
import unittest.mock
import wave

from percept import Goal, Percept
from percept.edge.tier0 import gray_frame
from percept.harness.scripted import ScriptedJudge
from percept.runtime import RuntimeStats, run

W, H = 8, 8
STILL = gray_frame(W, H, 0)  # identical every frame -> zero motion -> the visual gate never escalates


def _fused_core(on_exhausted: str = "repeat") -> tuple[Percept, ScriptedJudge]:
    """A single fused-modality goal, judged by a ``ScriptedJudge`` that answers "satisfied" on every
    call it's actually asked — so the ONLY thing that can make it fire is whether the gate ever admits
    a frame to be judged at all."""
    judge = ScriptedJudge(
        script=[{"satisfied": True, "confidence": 0.9, "evidence": "heard it"}],
        on_exhausted=on_exhausted,
    )
    core = Percept.create(vision=judge, discover_plugins=False)
    goal = Goal(id="sound-event", condition="a sound event occurs", say="Heard something")
    goal.state["perception"] = {"modality": "fused", "audio_s": 2.0}
    core.add_goal(goal)
    return core, judge


def _write_onset_wav(path: str, *, sr: int = 16_000) -> None:
    """1s silence -> 1s loud tone (the onset, energy well above AcousticOnset's default high=0.3) ->
    1s silence. Pure stdlib (``wave`` + ``array``) — no ffmpeg, no numpy."""
    silence = array.array("h", [0] * sr)
    tone = array.array(
        "h", [int(20_000 * math.sin(2 * math.pi * 440 * i / sr)) for i in range(sr)]
    )
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(silence.tobytes())
        w.writeframes(tone.tobytes())
        w.writeframes(silence.tobytes())


async def _paced_video(n: int = 60, dt: float = 0.02):
    """A scripted, motionless video stream — real (small) waits between frames so the background
    thread demuxing/streaming ``audio_source`` has real wall-clock time to reach the onset chunk
    before the video source (a plain in-memory generator, delivered near-instantly otherwise) runs
    out. ``judge_frame`` is a constant token — content is irrelevant since ``ScriptedJudge`` here is
    sequence-scripted, not token-routed."""
    for i in range(n):
        yield (float(i) * 0.1, STILL, "quiet")
        await asyncio.sleep(dt)


class ContainerDetectionTests(unittest.TestCase):
    """(a) URI normalization / dispatch — the demux route a container path takes."""

    def test_is_container_recognizes_video_suffixes(self):
        from percept.runtime.audio import _is_container

        for suffix in (".mp4", ".mov", ".mkv", ".webm"):
            self.assertTrue(_is_container(f"clip{suffix}"), suffix)
            self.assertTrue(_is_container(f"CLIP{suffix.upper()}"), suffix)  # case-insensitive
            self.assertTrue(_is_container(f"clip{suffix}?t=5"), suffix)  # query string tolerated

    def test_is_container_false_for_wav_and_mic(self):
        from percept.runtime.audio import _is_container

        self.assertFalse(_is_container("clip.wav"))
        self.assertFalse(_is_container("mic:0"))
        self.assertFalse(_is_container("session.mp3"))

    def test_open_audio_source_dispatches_container_to_capture_container(self):
        import percept.runtime.audio as audio_mod

        seen = {}

        def _fake_capture_container(path, config, q, stop, dropped, *, now=None):
            seen["path"] = path
            q.put(None)  # end sentinel — an empty but well-formed capture

        with unittest.mock.patch.object(audio_mod, "_capture_container", _fake_capture_container), \
             unittest.mock.patch.object(audio_mod, "_capture_wav") as wav_mock:
            async def _drain():
                return [item async for item in audio_mod.open_audio_source("clip.mp4", chunk_s=0.25)]

            rows = asyncio.run(_drain())
        self.assertEqual(rows, [])
        self.assertEqual(seen.get("path"), "clip.mp4")
        wav_mock.assert_not_called()  # a container never takes the wave.open() path


class FFmpegMissingTests(unittest.TestCase):
    """(a) A missing ffmpeg is a typed, surfaced error — never a silent no-audio run."""

    def test_capture_container_raises_typed_error_when_ffmpeg_missing(self):
        import percept.runtime.audio as audio_mod

        q: "queue.Queue" = queue.Queue()
        with unittest.mock.patch.object(shutil, "which", return_value=None):
            audio_mod._capture_container(
                "clip.mp4", audio_mod.AudioSourceConfig(), q, threading.Event(), None
            )
        first = q.get_nowait()
        self.assertIsInstance(first, audio_mod.FFmpegNotFoundError)
        self.assertIn("ffmpeg", str(first).lower())
        self.assertIn("clip.mp4", str(first))
        self.assertIsNone(q.get_nowait())  # end sentinel still follows, like every other capture path

    def test_open_audio_source_surfaces_ffmpeg_missing_to_the_consumer(self):
        import percept.runtime.audio as audio_mod

        async def _drain():
            got = []
            async for item in audio_mod.open_audio_source("clip.mkv", chunk_s=0.25):
                got.append(item)  # pragma: no cover - never reached, the source raises first
            return got

        with unittest.mock.patch.object(shutil, "which", return_value=None):
            with self.assertRaises(audio_mod.FFmpegNotFoundError):
                asyncio.run(_drain())


class AVOnsetIntegrationTests(unittest.TestCase):
    """(b)/(c) — the REAL worker loop over a real wav-backed audio_source."""

    def test_audio_onset_admits_and_fires(self):
        path = tempfile.mktemp(suffix=".wav")
        _write_onset_wav(path)
        try:
            core, judge = _fused_core()
            stats = RuntimeStats()

            async def _go():
                return [
                    inc async for inc in run(
                        core, source=_paced_video(), audio_source=path, stats=stats,
                    )
                ]

            incidents = asyncio.run(_go())
        finally:
            os.unlink(path)

        self.assertGreater(stats.audio_wakes, 0, "the onset never admitted a frame")
        self.assertGreater(len(judge.calls), 0, "the judge was never called")
        self.assertTrue(any(c["has_audio"] for c in judge.calls), "the judge never received audio")
        self.assertGreaterEqual(len(incidents), 1, "the fused goal never fired on the rising edge")

    def test_same_scenario_without_audio_source_never_fires(self):
        # Identical goal/judge/video — no audio_source. No motion + no audio => never admitted =>
        # zero judge calls => no fire. Proves the FIRE above was decided by the audio channel.
        core, judge = _fused_core()
        stats = RuntimeStats()

        async def _go():
            return [inc async for inc in run(core, source=_paced_video(n=20), stats=stats)]

        incidents = asyncio.run(_go())

        self.assertEqual(stats.audio_wakes, 0)
        self.assertEqual(len(judge.calls), 0)
        self.assertEqual(incidents, [])


class AutoWireContainerSourceTests(unittest.TestCase):
    """(d) File-driven run wiring: a container `source=` auto-derives `audio_source` ONLY when an
    armed goal wants fused/audio — and never spawns the audio capture path otherwise. Patches the two
    lazy-imported capture entrypoints so this needs neither cv2 nor ffmpeg in CI."""

    def _patched_video(self):
        async def _fake_open_source(uri, **kwargs):
            for i in range(3):
                yield (float(i), STILL, "quiet")
                await asyncio.sleep(0)

        return unittest.mock.patch("percept.runtime.source.open_source", _fake_open_source)

    def test_fused_goal_over_container_source_auto_derives_audio_source(self):
        calls = []

        async def _fake_open_audio_source(uri, **kwargs):
            calls.append(uri)
            if False:  # pragma: no cover - shape-only: a valid, empty async generator
                yield

        core, _judge = _fused_core()

        async def _go():
            with self._patched_video(), \
                 unittest.mock.patch("percept.runtime.audio.open_audio_source", _fake_open_audio_source):
                return [inc async for inc in run(core, source="clip.mp4")]

        asyncio.run(_go())
        self.assertEqual(calls, ["clip.mp4"])  # the SAME file's audio track, with no extra parameter

    def test_vision_only_goal_over_container_source_spawns_no_audio_capture(self):
        calls = []

        async def _fake_open_audio_source(uri, **kwargs):
            calls.append(uri)
            if False:  # pragma: no cover
                yield

        judge = ScriptedJudge(default_verdict={"satisfied": False, "confidence": 0.9})
        core = Percept.create(vision=judge, discover_plugins=False)
        core.add_goal(Goal(id="plain", condition="anything at all", say="hi"))  # vision-only (default)

        async def _go():
            with self._patched_video(), \
                 unittest.mock.patch("percept.runtime.audio.open_audio_source", _fake_open_audio_source):
                return [inc async for inc in run(core, source="clip.mp4")]

        asyncio.run(_go())
        self.assertEqual(calls, [])  # no ffmpeg/demux subprocess spawned when nothing wants audio

    def test_explicit_audio_source_always_wins_over_auto_derivation(self):
        calls = []

        async def _fake_open_audio_source(uri, **kwargs):
            calls.append(uri)
            if False:  # pragma: no cover
                yield

        core, _judge = _fused_core()

        async def _go():
            with self._patched_video(), \
                 unittest.mock.patch("percept.runtime.audio.open_audio_source", _fake_open_audio_source):
                return [
                    inc async for inc in run(core, source="clip.mp4", audio_source="explicit.wav")
                ]

        asyncio.run(_go())
        self.assertEqual(calls, ["explicit.wav"])  # never overridden by the auto-derivation


if __name__ == "__main__":
    unittest.main()
