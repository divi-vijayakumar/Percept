"""SOAK: the runtime's resident frame retention is BOUNDED, and the bound is loud.

Two client teams measured RSS 92 -> 342 MB on a 36-second clip and read it as a leak that made every
24/7 deployment impossible. Profiling `percept.runtime.worker.run` offline with a scripted judge
(no key, no spend) named the retaining reference exactly:

    percept/runtime/worker.py  `clips = FrameWindow(size=snippet_size)`   — 48 decoded frames
    percept/agent.py           `FrameWindow(settings.events.window)`      — 12 decoded frames

Both hold the SAME frame objects, so steady-state retention is ``max(48, 12) + ~2 in flight`` = ~50
decoded frames — a **plateau**, not a slope: over 8000 frames the Python heap sat flat at 46.3 MB and
live frames at exactly 50 while RSS plateaued and stopped moving. The "growth" was the allocator's
high-water mark filling that fixed ring, measured on a clip that ended before the plateau.

That plateau is still 50 x frame_bytes, which at 1080p RGB is ~300 MB — the client's number. So the
count bound is not a memory bound, and the fix is a BYTE budget on the pre-roll ring whose evictions
are reported (ADR-001 §9).

These tests are the fence: they fail if per-frame retention starts growing again, if the byte budget
stops bounding, or if a bound-driven drop goes silent.
"""
from __future__ import annotations

import asyncio
import gc
import unittest
import weakref

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.backends.fake import FakeVision
from percept.events import FrameWindow, frame_nbytes
from percept.harness.tier0 import gray_frame
from percept.runtime import RuntimeStats, run

W, H = 8, 8
STILL = gray_frame(W, H, 0)
MOVED = gray_frame(W, H, 255)
PAYLOAD = 64 * 1024  # 64 KB per "decoded frame" — big enough to measure, small enough to be fast


class CensusFrame:
    """A decoded-frame stand-in that COUNTS its own live instances.

    numpy arrays are not gc-tracked, so `gc.get_objects()` cannot see them and a census built on it
    silently reports zero retention (this is how the leak was mis-attributed the first time). A
    weakref-finalized object gives an exact live count instead of an inference.
    """

    live = 0
    __slots__ = ("i", "buf", "__weakref__")

    def __init__(self, i: int) -> None:
        self.i = i
        self.buf = bytearray(PAYLOAD)
        CensusFrame.live += 1
        weakref.finalize(self, CensusFrame._dec)

    @staticmethod
    def _dec() -> None:
        CensusFrame.live -= 1

    @property
    def nbytes(self) -> int:
        return len(self.buf)


class _ScriptedVision(FakeVision):
    """Scripted judge — no API key, no spend, and no ``str(frame)`` blowup on a heavy frame."""

    async def judge(self, condition, frame, **kw):  # noqa: ANN001, D102
        return Verdict(satisfied=True, confidence=0.95, evidence="scripted")


def _core() -> Percept:
    core = Percept.create(vision=_ScriptedVision(), discover_plugins=False)
    core.add_goal(Goal(id="zone", condition="a person is in the restricted zone", say="Alert"))
    return core


def _drive(n_frames: int, *, sample_every: int = 25, **kw):
    """Run `n_frames` through the real worker, sampling the LIVE decoded-frame census as we go."""
    samples: list[tuple[int, int]] = []
    health: list = []

    def source():
        for i in range(n_frames):
            if i % sample_every == 0:
                gc.collect()
                samples.append((i, CensusFrame.live))
            yield (float(i) * 0.25, MOVED if (i % 2) else STILL, CensusFrame(i))

    stats = RuntimeStats()
    core = _core()

    async def go(runtime_core):
        async for _inc in run(runtime_core, source=source(), stats=stats,
                              judge_dispatch="inline", on_health=health.append, **kw):
            pass

    CensusFrame.live = 0
    asyncio.run(go(core))
    del core
    gc.collect()
    return samples, stats, health


class FrameWindowBoundTests(unittest.TestCase):
    """The ring itself — count bound, byte bound, and the never-silent drop."""

    def test_count_bound_is_unchanged_and_byte_bound_is_opt_in(self):
        w = FrameWindow(size=3)
        for i in range(10):
            w.push(float(i), CensusFrame(i))
        self.assertEqual(len(w), 3)
        self.assertEqual(w.nbytes, 0)  # no byte bound configured -> nothing measured, byte-for-byte

    def test_byte_bound_evicts_oldest_first_and_reports_every_drop(self):
        drops: list[tuple[float, int, str]] = []
        cap = 4 * PAYLOAD
        w = FrameWindow(size=64, max_bytes=cap,
                        on_drop=lambda t, f, r: drops.append((t, f.i, r)))
        for i in range(12):
            w.push(float(i), CensusFrame(i))
        self.assertLessEqual(w.nbytes, cap)
        self.assertEqual(len(w), 4)
        # oldest-first eviction, one report per dropped frame, each carrying its reason
        self.assertEqual([d[1] for d in drops], list(range(8)))
        self.assertTrue(all(d[2] == "max_bytes" for d in drops))
        # the ring holds the NEWEST frames — the pre-roll stays adjacent to the incident
        self.assertEqual([f.i for _, f in w.frames()], [8, 9, 10, 11])

    def test_the_newest_frame_is_never_evicted_even_when_it_alone_exceeds_the_budget(self):
        drops = []
        w = FrameWindow(size=8, max_bytes=8, on_drop=lambda t, f, r: drops.append(r))
        w.push(0.0, CensusFrame(0))
        w.push(1.0, CensusFrame(1))
        self.assertEqual(len(w), 1)  # kept: a ring of one always holds the current frame
        self.assertEqual(drops, ["max_bytes"])  # and the drop was reported, not silent

    def test_an_unmeasurable_frame_never_causes_a_drop(self):
        # A frame whose size we cannot measure reports 0 bytes; evicting on a guess would silently
        # shorten every snippet on an unrecognised frame type.
        class Opaque:
            pass

        self.assertEqual(frame_nbytes(Opaque()), 0)
        drops = []
        w = FrameWindow(size=6, max_bytes=1, on_drop=lambda t, f, r: drops.append(r))
        for i in range(6):
            w.push(float(i), Opaque())
        self.assertEqual(len(w), 6)
        self.assertEqual(drops, [])

    def test_frame_nbytes_measures_the_shapes_the_runtime_actually_sees(self):
        self.assertEqual(frame_nbytes(bytearray(1234)), 1234)
        self.assertEqual(frame_nbytes(b"x" * 9), 9)
        self.assertEqual(frame_nbytes(gray_frame(4, 5)), 20)  # harness Frame -> w*h
        self.assertEqual(frame_nbytes(CensusFrame(0)), PAYLOAD)  # nbytes-bearing (numpy-shaped)


class RuntimeSoakTests(unittest.TestCase):
    """The soak fence: retention per frame must not grow with the length of the run."""

    #: Frames the runtime may hold at steady state: the pre-roll ring, the agent's event window (the
    #: same objects), plus a small allowance for the frame in flight / the last incident's snippet.
    #: A regression that retains ONE extra frame per tick blows this out within a few hundred frames.
    RETENTION_SLACK = 6

    def test_steady_state_retention_is_a_plateau_not_a_slope(self):
        n = 400
        samples, stats, _health = _drive(n, sample_every=25, snippet_size=16)
        self.assertEqual(stats.frames, n)
        bound = max(16, 12) + self.RETENTION_SLACK
        worst = max(live for _, live in samples)
        self.assertLessEqual(
            worst, bound,
            f"runtime retained {worst} decoded frames (bound {bound}); samples={samples}",
        )
        # THE SOAK ASSERTION: retention in the last quarter is not above the first quarter's plateau.
        # Any per-frame leak makes the tail strictly larger, however small the per-frame amount.
        warm = [live for i, live in samples if i >= n // 4]
        head, tail = warm[: len(warm) // 2], warm[len(warm) // 2:]
        self.assertTrue(head and tail)
        growth_per_frame = (max(tail) - max(head)) / (n / 2)
        self.assertLessEqual(
            growth_per_frame * PAYLOAD, 1024,
            f"retention grew {growth_per_frame * PAYLOAD:.0f} bytes/frame across the run "
            f"(head={max(head)} tail={max(tail)} frames held); samples={samples}",
        )

    def test_retention_tracks_the_configured_ring_size_so_the_ring_is_named_correctly(self):
        # The diagnosis is only real if turning the named knob moves the measured retention.
        small, _s, _h = _drive(200, sample_every=25, snippet_size=4)
        large, _s2, _h2 = _drive(200, sample_every=25, snippet_size=40)
        self.assertLess(
            max(live_count for _, live_count in small),
            max(live_count for _, live_count in large),
        )
        self.assertLessEqual(
            max(live_count for _, live_count in small),
            12 + self.RETENTION_SLACK,
        )  # window dominates
        self.assertLessEqual(
            max(live_count for _, live_count in large),
            40 + self.RETENTION_SLACK,
        )

    def test_the_byte_budget_bounds_the_run_and_every_drop_is_announced(self):
        cap = 8 * PAYLOAD
        samples, stats, health = _drive(300, sample_every=25, snippet_size=48,
                                        snippet_max_bytes=cap)
        held = max(live for _, live in samples)
        self.assertLessEqual(held, 8 + 12 + self.RETENTION_SLACK)  # capped ring + event window
        self.assertGreater(stats.preroll_frames_dropped, 0)
        capped = [e for e in health if e.kind == "preroll_capped"]
        self.assertEqual(len(capped), stats.preroll_frames_dropped)  # never a silent counter bump
        d = capped[0].data
        self.assertEqual(d["reason"], "max_bytes")
        self.assertEqual(d["max_bytes"], cap)
        self.assertEqual(d["frame_bytes"], PAYLOAD)
        self.assertLessEqual(d["held_bytes"], cap)

    def test_no_byte_budget_is_byte_for_byte_the_count_bounded_ring(self):
        samples, stats, health = _drive(200, sample_every=25, snippet_size=16,
                                        snippet_max_bytes=0)
        self.assertEqual(stats.preroll_frames_dropped, 0)
        self.assertEqual([e for e in health if e.kind == "preroll_capped"], [])
        self.assertLessEqual(
            max(live_count for _, live_count in samples),
            16 + self.RETENTION_SLACK,
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
