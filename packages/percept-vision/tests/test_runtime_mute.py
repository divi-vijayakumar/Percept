"""Command-seam: RuntimeSession.detach_audio_source()/attach_audio_source() = MUTED.

MUTED stops FEEDING the mic to perception (the AudioWindow stops filling; energy/onset gate signals
freeze) without ending the session — video keeps watching. Un-mute resumes. `muted=None` (default) is
byte-for-byte.
"""
from __future__ import annotations

import asyncio
import unittest

import percept
from percept.adapters import Goal, Verdict
from percept.contracts import AudioFrame
from percept.edge.tier0 import gray_frame
from percept.runtime.session import start

STILL = gray_frame(8, 8, 0)


class _Vision:
    async def judge(self, c, f, **k):
        return Verdict(satisfied=False, confidence=0.9)

    async def detect(self, f, g):
        return []

    async def detect_events(self, f, g):
        return []


def _core():
    core = percept.Percept.create(vision=_Vision(), discover_plugins=False)
    core.add_goal(Goal(id="z", condition="x", say="hi"))
    return core


async def _paced_audio(n=80, dt=0.004):
    for i in range(n):
        yield (float(i) * 0.25, AudioFrame(pcm=b"\x00\x00", sample_rate=16000), 0.05)
        await asyncio.sleep(dt)


async def _paced_video(n=80, dt=0.004):
    for i in range(n):
        yield (float(i), STILL, "tok")
        await asyncio.sleep(dt)


class MuteTests(unittest.TestCase):
    def test_detach_stops_feeding_audio_attach_resumes(self):
        async def go():
            core = _core()
            session = await start(core, source=_paced_video(), audio_source=_paced_audio())
            session.detach_audio_source()  # MUTED from the start
            await asyncio.sleep(0.10)
            win_muted = len(core._audio_window("camera")._chunks)  # should stay empty while muted
            session.attach_audio_source()  # un-mute
            await asyncio.sleep(0.20)
            win_after = len(core._audio_window("camera")._chunks)
            await session.stop()
            return win_muted, win_after

        win_muted, win_after = asyncio.run(go())
        self.assertEqual(win_muted, 0, "no audio should reach the AudioWindow while muted")
        self.assertGreater(win_after, 0, "audio resumes feeding perception after attach")

    def test_mute_unmute_aliases(self):
        """R-3: mute()/unmute() are the spec's power-ladder verbs, aliasing detach/attach_audio_source."""
        async def go():
            core = _core()
            session = await start(core, source=_paced_video(2), audio_source=_paced_audio(2))
            self.assertFalse(session.muted())
            session.mute()
            self.assertTrue(session.muted())   # mute() == detach_audio_source()
            session.unmute()
            self.assertFalse(session.muted())  # unmute() == attach_audio_source()
            await session.stop()

        asyncio.run(go())


if __name__ == "__main__":
    unittest.main()
