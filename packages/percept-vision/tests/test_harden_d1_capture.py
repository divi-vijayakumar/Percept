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
from unittest import mock

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
                           "a brief goal armed mid-run must get a coast (legacy behavior got NONE)")
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


class CapturePlanRefusesAnUnusableCadenceTests(unittest.TestCase):
    """``_capture_plan`` is the ONE place a bad cadence is meant to be dropped, and it is now the only
    place upstream of the fovea that can be: ``SensePolicy`` validates, so an unusable envelope raises
    here instead of quietly misconfiguring the controller. A 24/7 worker must survive that — one bad
    goal drops its own coverage floor, loudly, and every other goal still arms."""

    def _plan(self, *goals, base_fps=4.0):
        from percept.runtime.worker import _capture_plan

        return _capture_plan(list(goals), base_fps)

    def _sense_goal(self, gid, **sense):
        # goal.state is read RAW by sensor_plan_for — coerce_authored_sense is the conductor's
        # firewall, not a guarantee about what reaches the compiler.
        return Goal(id=gid, condition="the cat appears", say="!", state={"sense": sense})

    def test_an_unusable_cadence_drops_that_goal_without_raising(self):
        for sense in ({"base_fps": float("nan")},                 # A-04-02: NaN is TRUTHY
                      {"base_fps": float("inf")},
                      {"base_fps": 0},                            # A-04-01: an authored zero floor
                      {"base_fps": -1},
                      {"base_fps": 20, "max_fps": 2},             # A-03-02: inverted envelope
                      {"base_fps": 4, "escalate_threshold": 5.0}):
            with self.subTest(sense=sense):
                with self.assertLogs("percept.runtime.worker", level="WARNING") as logged:
                    cap, coast, required = self._plan(self._sense_goal("bad", **sense))
                self.assertEqual(cap, 4.0, "a refused cadence must not move the capture rate")
                self.assertIsNone(coast, "a refused cadence must not arm a coast")
                self.assertEqual(required, 0.0,
                                 "a refused cadence must not promise coverage it cannot deliver")
                # Dropped LOUDLY. Silence here is what let a misconfigured concern run a whole
                # session looking like an ordinary quiet scene.
                self.assertTrue(any("capture_plan_goal_dropped" in m for m in logged.output),
                                f"the drop was silent; got {logged.output}")

    def test_a_goal_that_simply_has_no_cadence_floor_is_dropped_in_silence(self):
        # The other half of the contract: an ORDINARY goal compiles no floor and must not warn, or
        # the diagnostic that matters is buried under one line per goal per regate.
        from percept.runtime.worker import _log

        with mock.patch.object(_log, "warning") as warn:
            cap, coast, required = self._plan(Goal(id="plain", condition="someone walks in", say="!"))
        self.assertEqual((cap, coast, required), (4.0, None, 0.0))
        warn.assert_not_called()

    def test_one_bad_goal_never_costs_a_good_goal_its_coverage(self):
        cap, coast, required = self._plan(
            self._sense_goal("bad", base_fps=float("nan")), _brief_goal("good")
        )
        self.assertIsNotNone(coast, "the healthy brief goal must still get its coast")
        self.assertEqual(required, 4.0)  # 2 samples / 0.5 s window
        self.assertGreater(cap, 4.0)

    def test_a_worker_run_survives_a_goal_with_an_unusable_cadence(self):
        # The end-to-end shape of the hardening: with validation added, a previously-silent bad policy
        # now raises — it must be dropped, not take the run down.
        core = _core()
        core.add_goal(self._sense_goal("bad", base_fps=float("nan")))
        core.add_goal(_brief_goal("good"))
        frames = [(i / 8.0, STILL, "tok") for i in range(17)]
        asyncio.run(_drain(core, list(frames)))  # must not raise


if __name__ == "__main__":
    unittest.main()
