"""Command-seam: multi-consumer frame stream on RuntimeSession.

The seam must let a UI panel AND a voice command adapter both read the watch's frames — the single
`on_frame` callback couldn't. These assert every subscriber receives frames, `latest_frame()` still
works, unsubscribe stops delivery, and a raising subscriber never crashes the watch.
"""
from __future__ import annotations

import asyncio
import unittest

import percept
from percept.adapters import Verdict
from percept.edge.tier0 import gray_frame
from percept.runtime.session import start

STILL = gray_frame(8, 8, 0)
JPEG = ("image/jpeg", b"frame-bytes")
SRC = [(float(i), STILL, JPEG) for i in range(4)]


def _core():
    class V:
        async def judge(self, c, f, **k):
            return Verdict(satisfied=False, confidence=0.9)

        async def detect(self, f, g):
            return []

        async def detect_events(self, f, g):
            return []

    return percept.Percept.create(vision=V(), discover_plugins=False)


class MultiConsumerFrameTests(unittest.TestCase):
    def test_two_subscribers_both_receive_and_latest_frame_works(self):
        async def go():
            a, b = [], []
            session = await start(_core(), source=SRC, on_frame=lambda t, f: a.append(f))  # 1st subscriber
            session.subscribe(lambda t, f: b.append(f))  # 2nd, added after start
            await asyncio.sleep(0.02)
            latest = session.latest_frame()
            await session.stop()
            return a, b, latest

        a, b, latest = asyncio.run(go())
        self.assertTrue(a and b, "both subscribers should receive frames")
        self.assertEqual(a[-1], JPEG)
        self.assertEqual(b[-1], JPEG)
        self.assertEqual(latest, JPEG)  # latest_frame() unchanged

    def test_unsubscribe_stops_delivery(self):
        async def go():
            got = []
            session = await start(_core(), source=[(float(i), STILL, JPEG) for i in range(30)])
            unsub = session.subscribe(lambda t, f: got.append(f))
            await asyncio.sleep(0.01)
            n_before = len(got)
            unsub()
            await asyncio.sleep(0.02)
            n_after = len(got)
            await session.stop()
            return n_before, n_after

        n_before, n_after = asyncio.run(go())
        self.assertGreater(n_before, 0)
        self.assertEqual(n_before, n_after, "no frames delivered after unsubscribe")

    def test_raising_subscriber_does_not_crash_the_watch(self):
        async def go():
            good = []

            def boom(t, f):
                raise RuntimeError("bad consumer")

            session = await start(_core(), source=SRC, on_frame=boom)
            session.subscribe(lambda t, f: good.append(f))  # still delivered despite boom
            await asyncio.sleep(0.02)
            await session.stop()
            return good

        good = asyncio.run(go())
        self.assertTrue(good, "a raising subscriber must not stop delivery to others / crash the watch")


if __name__ == "__main__":
    unittest.main()
