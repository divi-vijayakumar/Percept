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
from percept.adapters import AudioFrame, Verdict
from percept.edge.tier0 import gray_frame
from percept.harness.scripted import ScriptedJudge
from percept.runtime import RuntimeStats, run

W, H = 8, 8
STILL = gray_frame(W, H, 0)  # identical every frame -> zero motion -> the visual gate never escalates
MOVED = gray_frame(W, H, 255)  # a still->moved (or moved->still) transition -> the visual gate escalates


class _SpyVision:
    """A minimal ``Vision`` double that records the ``audio`` object handed to every ``judge()`` call,
    keyed by the caller-supplied ``frame`` token. ``ScriptedJudge.calls`` (used elsewhere in this file)
    tracks whether audio was present (``has_audio``) but not the audio OBJECT itself — these regression
    tests need the actual clip content (its raw PCM) to check WHICH portion of the source file a given
    judged frame actually heard, so a dedicated double is simplest."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def judge(self, condition, frame, *, roi=None, exemplar=None, corrections=None,
                    frames=None, audio=None, context=None) -> Verdict:
        self.calls.append({"frame": frame, "audio": audio})
        return Verdict(satisfied=False, confidence=0.9, evidence="spy")

    async def detect(self, frame, goals):
        return []

    async def detect_events(self, frames, goals):
        return []


def _pcm_rms(pcm: bytes) -> float:
    """Normalized RMS ([0, 1]) used only as an intuitive clip-content assertion in these tests."""
    n = len(pcm) - (len(pcm) % 2)
    if n <= 0:
        return 0.0
    samples = array.array("h")
    samples.frombytes(pcm[:n])
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples)) / 32768.0


def _loud_fraction(pcm: bytes, threshold: int = 5000) -> float:
    """Fraction of s16le samples whose magnitude clears ``threshold`` — how much of a clip is the LOUD
    tone versus silence, sample-by-sample (a finer-grained companion to the whole-clip ``_pcm_rms``)."""
    n = len(pcm) - (len(pcm) % 2)
    if n <= 0:
        return 0.0
    samples = array.array("h")
    samples.frombytes(pcm[:n])
    if not samples:
        return 0.0
    loud = sum(1 for s in samples if abs(s) >= threshold)
    return loud / len(samples)


def _write_phase_wav(path: str, *, sr: int = 16_000, silence_lead_s: float, tone_s: float,
                     silence_tail_s: float, freq: float = 440.0, amp: int = 20_000) -> None:
    """silence(lead) -> tone(loud) -> silence(tail), pure stdlib (``wave`` + ``array``). Lets a test place
    the loud phase at an exact, known media-time offset (unlike ``_write_onset_wav``'s fixed 1/1/1s)."""
    def _silence(n: int) -> array.array:
        return array.array("h", [0] * n)

    def _tone(n: int) -> array.array:
        return array.array("h", [int(amp * math.sin(2 * math.pi * freq * i / sr)) for i in range(n)])

    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(_silence(int(sr * silence_lead_s)).tobytes())
        w.writeframes(_tone(int(sr * tone_s)).tobytes())
        w.writeframes(_silence(int(sr * silence_tail_s)).tobytes())


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
    """1s silence -> 1s loud tone (the onset, mean-square energy above the default high=0.09) ->
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


async def _paced_source(entries: list, dt: float = 0.02):
    """Like ``_paced_video`` but driven by explicit ``(t, gate_frame, judge_frame)`` entries — lets a
    test place a VISUAL motion spike (``MOVED``) at an exact, known media time, independent of audio
    content, so admission is decided purely by the tier-0 motion gate."""
    for t, gate_frame, judge_frame in entries:
        yield (t, gate_frame, judge_frame)
        await asyncio.sleep(dt)


def _av_sync_entries(spike_ts: tuple = (0.4, 2.2), *, n: int = 41, step: float = 0.1) -> list:
    """``n`` frames on a 0.0, ``step``, 2*``step``, ... media timeline, all ``STILL`` except a single
    ``MOVED`` frame at each ``t`` in ``spike_ts`` — a still->moved (and the following moved->still)
    transition each admits exactly one judged frame at a KNOWN media time, decoupled from the audio
    onset gate entirely (tests pass ``audio_gate=None``)."""
    spikes = {round(t, 6) for t in spike_ts}
    entries = []
    for i in range(n):
        t = round(i * step, 6)
        frame = MOVED if t in spikes else STILL
        entries.append((t, frame, f"frame@{t:.1f}"))
    return entries


def _fused_core_spy(audio_s: float = 1.0) -> tuple[Percept, _SpyVision]:
    """A single fused-modality goal judged by ``_SpyVision`` — the audio-clip-content counterpart of
    ``_fused_core`` (which uses ``ScriptedJudge`` and doesn't record the audio object itself)."""
    spy = _SpyVision()
    core = Percept.create(vision=spy, discover_plugins=False)
    goal = Goal(id="sound-event", condition="a sound event occurs", say="Heard something")
    goal.state["perception"] = {"modality": "fused", "audio_s": audio_s}
    core.add_goal(goal)
    return core, spy


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


class AVSyncRegressionTests(unittest.TestCase):
    """(e) The AV-desync regression: a FILE-driven run must feed each judged frame the audio that was
    actually sounding AROUND that frame's own media time, never a slice pruned from whatever the whole
    file's ffmpeg/wave pump had already dumped into the AudioWindow by the time the (slower) video loop
    got there. Admission here is driven ONLY by visual motion spikes at two known media times
    (``audio_gate=None``) — decoupling "was this frame judged" from "did the audio pacing work",
    which is the one thing under test."""

    def test_judged_clip_rms_matches_the_frame_it_was_judged_at(self):
        path = tempfile.mktemp(suffix=".wav")
        # silence 0.0-1.5s, LOUD tone 1.5-3.0s (mid-file), silence 3.0-4.0s (tail) -- matches the
        # 0.0-4.0s video timeline below.
        _write_phase_wav(path, silence_lead_s=1.5, tone_s=1.5, silence_tail_s=1.0)
        try:
            core, spy = _fused_core_spy(audio_s=1.0)
            entries = _av_sync_entries((0.4, 2.2))  # 0.4 = early/quiet, 2.2 = mid-file/loud

            async def _go():
                return [
                    inc async for inc in run(
                        core, source=_paced_source(entries), audio_source=path, audio_gate=None,
                    )
                ]

            asyncio.run(_go())
        finally:
            os.unlink(path)

        by_frame = {c["frame"]: c["audio"] for c in spy.calls}
        early = by_frame.get("frame@0.4")
        loud = by_frame.get("frame@2.2")
        self.assertIsNotNone(early, "the early motion spike never reached the judge")
        self.assertIsNotNone(loud, "the mid-file motion spike never reached the judge")
        early_rms = _pcm_rms(early.pcm)
        loud_rms = _pcm_rms(loud.pcm)
        # The regression this guards: WITHOUT media-clock pacing, the whole file streams into the
        # AudioWindow before either judge call, so BOTH clips are sliced from the same fixed pool --
        # here that pool ends in silence (the tail), so the LOUD-phase frame would wrongly hear quiet
        # audio too. Pacing must make the mid-file frame's clip clearly louder than the early frame's.
        self.assertLess(early_rms, 0.05, f"an EARLY frame heard loud audio (rms={early_rms:.3f})")
        self.assertGreater(loud_rms, 0.2, f"the mid-file LOUD-phase frame heard quiet audio (rms={loud_rms:.3f})")


class ConcatAudioFrameTests(unittest.TestCase):
    """(f) ``_concat_audio`` must concatenate typed ``AudioFrame`` chunks (what the runtime pump
    actually fills the AudioWindow with) rather than falling through to its ``chunks[-1]`` opaque
    fallback -- the fix for a fused judge silently receiving only the last (often short/partial) chunk."""

    def test_concatenates_audioframe_chunks_with_true_metadata(self):
        from percept.agent import _concat_audio

        chunks = [
            AudioFrame(pcm=b"\x01\x00" * 4000, sample_rate=22_050, channels=2,
                      sample_format="s16le", t_start=0.0, t_end=0.25),
            AudioFrame(pcm=b"\x02\x00" * 4000, sample_rate=22_050, channels=2,
                      sample_format="s16le", t_start=0.25, t_end=0.5),
            AudioFrame(pcm=b"\x03\x00" * 4000, sample_rate=22_050, channels=2,
                      sample_format="s16le", t_start=0.5, t_end=0.75),
        ]
        clip = _concat_audio(chunks)
        self.assertIsInstance(clip, AudioFrame)
        self.assertEqual(len(clip.pcm), sum(len(c.pcm) for c in chunks))  # summed length, not chunks[-1]
        self.assertEqual(clip.pcm, b"".join(c.pcm for c in chunks))
        # threaded through from the frames themselves -- never hardcoded to AudioFrame's 16000/1 default
        self.assertEqual(clip.sample_rate, 22_050)
        self.assertEqual(clip.channels, 2)
        self.assertEqual(clip.sample_format, "s16le")
        self.assertEqual(clip.t_start, 0.0)
        self.assertEqual(clip.t_end, 0.75)

    def test_single_chunk_window_returns_that_chunks_own_bytes(self):
        from percept.agent import _concat_audio

        chunk = AudioFrame(pcm=b"\xAB\xCD" * 100, sample_rate=16_000, channels=1)
        clip = _concat_audio([chunk])
        self.assertEqual(clip.pcm, chunk.pcm)

    def test_encoded_only_frame_passes_through_unmodified(self):
        from percept.agent import _concat_audio

        encoded = AudioFrame(pcm=b"", encoded=b"RIFF....", mime="audio/wav")
        self.assertIs(_concat_audio([encoded]), encoded)  # nothing to concatenate -> pass-through


class OnsetSettleClipContentTests(unittest.TestCase):
    """(g) Onset+settle end-to-end: a sound starting at media t ~= half the clip wakes the gate, the
    settle delay lets the window fill, and the WOKEN judge's clip must be MAJORITY loud samples -- not
    just "audio was present" (``AVOnsetIntegrationTests`` above) but the RIGHT audio."""

    def test_woken_judge_clip_is_majority_loud(self):
        path = tempfile.mktemp(suffix=".wav")
        # silence 0.0-2.0s, tone 2.0-3.0s (starts at exactly half of the 4.0s file), silence 3.0-4.0s.
        _write_phase_wav(path, silence_lead_s=2.0, tone_s=1.0, silence_tail_s=1.0)
        try:
            core, spy = _fused_core_spy(audio_s=1.0)
            stats = RuntimeStats()

            async def _go():
                return [
                    inc async for inc in run(
                        core, source=_paced_video(n=46, dt=0.02), audio_source=path,
                        audio_settle_s=0.3, stats=stats,
                    )
                ]

            asyncio.run(_go())
        finally:
            os.unlink(path)

        self.assertGreater(stats.audio_wakes, 0, "the onset never woke the gate")
        self.assertGreater(len(spy.calls), 0, "the judge was never called")
        woken = spy.calls[-1]["audio"]
        self.assertIsNotNone(woken, "the woken judge never received audio")
        loud_frac = _loud_fraction(woken.pcm)
        self.assertGreaterEqual(
            loud_frac, 0.5, f"only {loud_frac:.0%} of the woken clip was loud samples"
        )


class LiveMicUnpacedTests(unittest.TestCase):
    """(h) The live-mic path stays byte-for-byte: pacing must engage ONLY for a recorded (wav/mp4/
    container) ``audio_source`` and never for ``mic:`` -- proven here by feeding a fake mic stream whose
    chunk timestamps sit deliberately FAR ahead of a short, near-instant video. If pacing wrongly
    applied to a mic source this would stall (or hang, absent the end-of-source release) waiting for a
    video media time that never arrives; instead every chunk must reach the AudioWindow immediately, and
    the run must complete promptly."""

    def test_mic_source_is_never_paced_against_a_slow_or_short_video(self):
        async def _fake_mic(uri, **kwargs):
            for i in range(5):
                yield (
                    100.0 + i,  # media time FAR past anything the tiny video below ever reaches
                    AudioFrame(pcm=b"\x00\x00" * 100, sample_rate=16_000, channels=1),
                    0.0,
                )
                await asyncio.sleep(0)  # no artificial delay -- exactly like a live mic feeding fast

        core, _judge = _fused_core()
        stats = RuntimeStats()
        video = _paced_video(n=3, dt=0.01)  # media t 0.0, 0.1, 0.2 -- nothing near t=100

        async def _go():
            with unittest.mock.patch("percept.runtime.audio.open_audio_source", _fake_mic):
                return [
                    inc async for inc in run(core, source=video, audio_source="mic:0", stats=stats)
                ]

        incidents = asyncio.run(_go())  # a wrongly-paced mic pump would stall/hang here
        self.assertEqual(incidents, [])  # no assertion on firing -- only that the run completed
        self.assertEqual(stats.audio_chunks, 5)  # every mic chunk reached the AudioWindow, unheld


if __name__ == "__main__":
    unittest.main()
