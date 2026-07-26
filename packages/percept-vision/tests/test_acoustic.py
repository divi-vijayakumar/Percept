"""Acoustic / fused perception (L1) — AudioWindow + the agent assembles an audio clip for a fused spec.

Proves the audio path is wired: push_audio feeds the rolling window, and a fused/audio PerceptionSpec
makes the agent concatenate the last audio_s seconds into one clip passed to judge(audio=...). Wiring,
not intelligence (a fused concern needs vision="gemini" to actually hear; here we just assert the clip flows).
"""

import asyncio
import unittest

from percept.adapters import AudioFrame, Goal, Verdict
from percept.agent import Percept
from percept.backends.fake import FakeVision
from percept.events import AudioWindow


class RecordingVision:
    def __init__(self):
        self.calls: list[dict] = []

    async def judge(self, condition, frame, *, roi=None, exemplar=None, corrections=None,
                    frames=None, audio=None):
        self.calls.append({"frames": frames, "audio": audio})
        return Verdict(satisfied=True, confidence=0.95, evidence="stub")

    async def detect(self, frame, goals):
        return []

    async def detect_events(self, frames, goals):
        return []


class TestAudioWindow(unittest.TestCase):
    def test_time_windowing_evicts_old_chunks(self):
        w = AudioWindow(seconds=2.0)
        for t in range(5):
            w.push(float(t), bytes([t]))
        self.assertEqual(len(w), 3)  # only t=2,3,4 are within the last 2s of t=4

    def test_clip_returns_last_seconds(self):
        w = AudioWindow(seconds=10.0)
        for t in range(5):
            w.push(float(t), bytes([t]))
        self.assertEqual(w.clip(2.0), [bytes([2]), bytes([3]), bytes([4])])  # t >= 4-2

    def test_empty_clip(self):
        self.assertEqual(AudioWindow().clip(2.0), [])


class TestFusedAssembly(unittest.TestCase):
    def _agent(self, goal):
        rec = RecordingVision()
        a = Percept.create(vision=rec, discover_plugins=False)
        a.add_goal(goal)
        return a, rec

    def test_fused_spec_assembles_and_passes_audio(self):
        g = Goal(id="g", condition="the milk is boiling", say="!",
                 state={"type": "watch", "count": 0, "perception": {"modality": "fused", "audio_s": 2.0}})
        a, rec = self._agent(g)

        async def go():
            a.push_audio(("audio/wav", b"\x01"), t=0.0)
            a.push_audio(("audio/wav", b"\x02"), t=0.5)
            await a.perceive_judged("frame", t=1.0)

        asyncio.run(go())
        self.assertEqual(rec.calls[-1]["audio"], ("audio/wav", b"\x01\x02"))  # concatenated clip flowed in

    def test_vision_spec_passes_no_audio(self):
        a, rec = self._agent(Goal(id="g", condition="c", say="s"))  # default vision-only spec

        async def go():
            a.push_audio(("audio/wav", b"\x01"), t=0.0)
            await a.perceive_judged("frame", t=1.0)

        asyncio.run(go())
        self.assertIsNone(rec.calls[-1]["audio"])  # a vision spec never assembles audio (byte-for-byte)


class TestTypedAudioFrame(unittest.TestCase):
    """A typed ``AudioFrame`` flows through ``judge(audio=...)``."""

    def test_audioframe_defaults_and_as_pair(self):
        af = AudioFrame(pcm=b"\x01\x02\x03\x04", sample_rate=16000, channels=1)
        self.assertEqual(af.sample_rate, 16000)
        self.assertEqual(af.channels, 1)
        self.assertEqual(af.sample_format, "s16le")
        self.assertIsNone(af.encoded)
        # raw-PCM render: a self-describing mime tag carrying rate/channels + the raw bytes
        mime, raw = af.as_pair()
        self.assertEqual(raw, b"\x01\x02\x03\x04")
        self.assertEqual(mime, "audio/L16;rate=16000;channels=1")

    def test_as_pair_prefers_encoded_blob(self):
        af = AudioFrame(pcm=b"raw", encoded=b"RIFFfake", mime="audio/wav")
        self.assertEqual(af.as_pair(), ("audio/wav", b"RIFFfake"))

    def test_duration_from_bounds(self):
        self.assertEqual(AudioFrame(t_start=1.0, t_end=3.5).duration_s, 2.5)
        self.assertIsNone(AudioFrame().duration_s)  # unbounded

    def test_frozen(self):
        af = AudioFrame()
        with self.assertRaises(Exception):
            af.sample_rate = 8000  # frozen dataclass

    def test_typed_audioframe_routes_through_fake_judge(self):
        # The fake supports a "{token}|audio" script key — passing a TYPED AudioFrame as audio= must
        # route to it, proving the typed object flows through the (typed-but-flexible) audio param.
        token = "kettle"
        audio_verdict = Verdict(satisfied=True, confidence=0.92, evidence="scripted: hiss heard")
        vision = FakeVision(script={
            token: Verdict(satisfied=False, confidence=0.6, evidence="scripted: silent frame"),
            f"{token}|audio": audio_verdict,
        })

        async def go():
            # no audio -> the vision-only scripted verdict
            v_silent = await vision.judge("the kettle is boiling", token)
            # typed AudioFrame -> routes to the |audio key (the audio modality is consulted)
            v_audio = await vision.judge(
                "the kettle is boiling", token,
                audio=AudioFrame(pcm=b"\x10\x20\x30", sample_rate=16000, channels=1),
            )
            return v_silent, v_audio

        v_silent, v_audio = asyncio.run(go())
        self.assertFalse(v_silent.satisfied)
        self.assertEqual(v_audio, audio_verdict)  # the typed audio object reached judge and changed the read


if __name__ == "__main__":
    unittest.main()
