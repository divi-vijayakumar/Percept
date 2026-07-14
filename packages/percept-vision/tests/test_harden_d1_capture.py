"""AGENT-D / D1 — a voice-armed brief-action goal must get real Nyquist coverage, not silence.

``_capture_plan`` ran ONCE before the frame loop and ``_build_gate`` recompiled the spec/gate/banks but
NOT the fovea coast. So a brief-action goal (one carrying a ``sense.brief_window_s`` prior) armed
MID-RUN by voice compiled a sensitive above-Nyquist cadence that NOTHING honored — it was sampled at
base rate with no coast, so its sub-second window was silently under-covered.

  * Acceptance 1 — a brief goal armed mid-run gets the same coast-driven judge coverage as one armed
    at t0 (D1a: rebuild the coast in ``_build_gate``).
  * Acceptance 6 — when the OPEN source can't deliver the brief goal's required Nyquist rate, the
    runtime emits ``capture_plan_unmet`` instead of arming silently (D1c: report the unmet guarantee).
"""
from __future__ import annotations

import asyncio
import unittest

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.backends.fake import FakeVision
from percept.edge.tier0 import gray_frame
from percept.runtime import RuntimeStats, run

STILL = gray_frame(8, 8, 0)


def _brief_goal(gid="brief"):
    # A brief-action judge goal: a sub-second window -> a sensitive above-Nyquist cadence (base ~4fps).
    return Goal(id=gid, condition="the cat briefly appears", say="!",
                state={"sense": {"brief_window_s": 0.5}})


def _core():
    vision = FakeVision(script={"tok": Verdict(satisfied=False, confidence=0.9, evidence="n/a")})
    return Percept.create(vision=vision, discover_plugins=False)


async def _drain(core, source, **kw):
    return [inc async for inc in run(core, source=source, regate_on_goal_change=True, **kw)]


class D1CoverageParityTests(unittest.TestCase):
    def test_midrun_brief_goal_gets_same_coast_coverage_as_t0(self):
        # 2s of quiet frames at 8fps. The motion gate never escalates on STILL frames; only the brief
        # goal's fovea coast (base ~4fps) forces judges — so judge_calls measures coverage.
        frames = [(i / 8.0, STILL, "tok") for i in range(17)]  # t = 0.0 .. 2.0

        # Run A: brief goal armed at t0.
        core_a = _core()
        core_a.add_goal(_brief_goal())
        stats_a = RuntimeStats()
        asyncio.run(_drain(core_a, list(frames), stats=stats_a))

        # Run B: brief goal armed MID-RUN (after the first frame), exactly the voice-arm flow.
        core_b = _core()
        stats_b = RuntimeStats()

        async def source_b():
            for i, item in enumerate(frames):
                if i == 1:
                    core_b.add_goal(_brief_goal())
                yield item
                # Explicit yield, matching what _aiter gives run A's sync list. A never-yielding async
                # generator starves the off-loop judge scheduler (judge_dispatch="async" default) —
                # every coast admit would supersede the last and this would measure overload
                # coalescing, not coverage parity.
                await asyncio.sleep(0)

        asyncio.run(_drain(core_b, source_b(), stats=stats_b))

        self.assertGreater(stats_a.judge_calls, 1, "sanity: the t0 brief goal must coast-judge the window")
        self.assertGreater(stats_b.judge_calls, 0,
                           "a brief goal armed mid-run must get a coast (it got NONE before D1a)")
        self.assertLessEqual(abs(stats_a.judge_calls - stats_b.judge_calls), 1,
                             f"mid-run coverage must match t0 within a frame: A={stats_a.judge_calls} "
                             f"B={stats_b.judge_calls}")


class D1UnmetGuaranteeTests(unittest.TestCase):
    def test_source_too_slow_for_the_brief_window_emits_capture_plan_unmet(self):
        # The brief goal needs >= 4fps (Nyquist for a 0.5s window). The source only DELIVERS ~2fps, so
        # the coverage guarantee cannot be met — it must be REPORTED, not silently downgraded.
        frames = [(i / 2.0, STILL, "tok") for i in range(9)]  # t = 0.0 .. 4.0 at 2fps
        core = _core()
        core.add_goal(_brief_goal())
        events: list = []

        async def on_health(ev):
            events.append(ev)

        asyncio.run(_drain(core, list(frames), on_health=on_health))
        kinds = [e.kind for e in events]
        self.assertIn("capture_plan_unmet", kinds,
                      f"an unmeetable coverage guarantee must be reported; got {kinds}")

    def test_met_guarantee_stays_quiet(self):
        # The source delivers 8fps, comfortably above the 4fps Nyquist -> no unmet report.
        frames = [(i / 8.0, STILL, "tok") for i in range(17)]
        core = _core()
        core.add_goal(_brief_goal())
        events: list = []

        async def on_health(ev):
            events.append(ev)

        asyncio.run(_drain(core, list(frames), on_health=on_health))
        self.assertNotIn("capture_plan_unmet", [e.kind for e in events],
                         "a met coverage guarantee must NOT false-report")


if __name__ == "__main__":
    unittest.main()
