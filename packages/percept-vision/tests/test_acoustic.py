"""Acoustic / fused perception (L1) — AudioWindow + the agent assembles an audio clip for a fused spec.

Proves the audio path is wired: push_audio feeds the rolling window, and a fused/audio PerceptionSpec
makes the agent concatenate the last audio_s seconds into one clip passed to judge(audio=...). Wiring,
not intelligence (a fused concern needs vision="gemini" to actually hear; here we just assert the clip flows).
"""

import asyncio
import unittest

from percept.adapters import Goal, Verdict
from percept.agent import Percept
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


if __name__ == "__main__":
    unittest.main()
