"""L1 unit tests for the tier-0 salience gate (spec §6.3, impl plan §8).

Covers, on fakes / fixtures (no keys, deterministic):
  * ``motion_score``        — still -> ~0, motion -> high; symmetry; first-frame; size mismatch.
  * ``compile_watchspec``   — goals -> conditions; frozen contract fields; per-goal bias; type filter.
  * ``MotionGate``          — still scene -> NO escalation; motion -> escalate; empty spec -> never.
  * the wired loop          — Tier0Signal reported up; vision calls only on escalation; **reduces
                              calls vs ungated** on a fixture stream (the proof point).
"""

from __future__ import annotations

import asyncio
import unittest

from percept.adapters import Box, Goal, Tier0Signal, WatchSpec
from percept.agent import Percept
from percept.config import load_settings

from percept_harness.tier0 import (
    Frame,
    GatedWatchLoop,
    MotionGate,
    compile_watchspec,
    gray_frame,
    motion_score,
    run_gated_watch,
)

W, H = 8, 8


def still(fill: int = 0) -> Frame:
    return gray_frame(W, H, fill=fill)


def motion(n_pixels: int = 40, value: int = 255) -> Frame:
    pixels = [0] * (W * H)
    for i in range(min(n_pixels, len(pixels))):
        pixels[i] = value
    return Frame(w=W, h=H, pixels=tuple(pixels))


def watch_goal(gid: str, condition: str, **state) -> Goal:
    g = Goal(id=gid, condition=condition, say=f"saw {gid}")
    g.state.update(state)
    return g


# ---------------------------------------------------------------------------
# motion_score
# ---------------------------------------------------------------------------


class MotionScoreTests(unittest.TestCase):
    def test_identical_frames_score_zero(self):
        f = still()
        self.assertEqual(motion_score(f, f), 0.0)

    def test_no_prev_frame_scores_zero(self):
        # The very first frame has nothing to diff against -> no motion yet.
        self.assertEqual(motion_score(None, still()), 0.0)

    def test_still_scene_scores_near_zero(self):
        # A tiny single-pixel flicker over a still scene is well below any gate threshold.
        a = still()
        b = still()
        b = Frame(w=W, h=H, pixels=(5,) + b.pixels[1:])  # one pixel changes by 5/255
        self.assertLess(motion_score(a, b), 0.01)

    def test_large_change_scores_high(self):
        score = motion_score(still(), motion())
        self.assertGreater(score, 0.1)

    def test_full_flip_scores_one(self):
        # Every pixel 0 -> 255 is the maximum possible diff -> exactly 1.0.
        black = gray_frame(W, H, fill=0)
        white = gray_frame(W, H, fill=255)
        self.assertAlmostEqual(motion_score(black, white), 1.0, places=9)

    def test_symmetric(self):
        a, b = still(), motion()
        self.assertAlmostEqual(motion_score(a, b), motion_score(b, a), places=9)

    def test_accepts_raw_tuple_frame(self):
        a = (W, H, tuple([0] * (W * H)))
        b = (W, H, tuple([255] * (W * H)))
        self.assertAlmostEqual(motion_score(a, b), 1.0, places=9)

    def test_size_mismatch_raises(self):
        with self.assertRaises(ValueError):
            motion_score(gray_frame(4, 4), gray_frame(8, 8))

    def test_more_motion_scores_higher(self):
        small = motion(n_pixels=4)
        big = motion(n_pixels=40)
        self.assertGreater(motion_score(still(), big), motion_score(still(), small))


# ---------------------------------------------------------------------------
# compile_watchspec
# ---------------------------------------------------------------------------


class CompileWatchSpecTests(unittest.TestCase):
    def test_one_condition_per_watch_goal(self):
        goals = [watch_goal("g1", "a cup is up"), watch_goal("g2", "a person entered")]
        spec = compile_watchspec(goals, threshold=0.2)
        self.assertIsInstance(spec, WatchSpec)
        self.assertEqual(len(spec.conditions), 2)
        self.assertEqual({c.goal_id for c in spec.conditions}, {"g1", "g2"})

    def test_contract_fields_correct(self):
        spec = compile_watchspec([watch_goal("g1", "a cup is up")], threshold=0.25)
        c = spec.conditions[0]
        self.assertEqual(c.goal_id, "g1")
        self.assertEqual(c.anchor_text, "a cup is up")  # anchor_text <- goal.condition
        self.assertAlmostEqual(c.threshold, 0.25)
        self.assertIsNone(c.region_hint)
        self.assertEqual(spec.version, 1)

    def test_per_goal_bias_applied_and_clamped(self):
        goals = [
            watch_goal("up", "noisy goal", motion_bias=0.1),
            watch_goal("down", "critical goal", motion_bias=-0.05),
            watch_goal("clamp", "over the top", motion_bias=5.0),
        ]
        spec = compile_watchspec(goals, threshold=0.2)
        by_id = {c.goal_id: c for c in spec.conditions}
        self.assertAlmostEqual(by_id["up"].threshold, 0.3)
        self.assertAlmostEqual(by_id["down"].threshold, 0.15)
        self.assertAlmostEqual(by_id["clamp"].threshold, 1.0)  # clamped to 0..1

    def test_region_hint_passthrough(self):
        g = watch_goal("g1", "cup", region_hint={"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4})
        spec = compile_watchspec([g], threshold=0.2)
        hint = spec.conditions[0].region_hint
        self.assertIsInstance(hint, Box)
        self.assertEqual((hint.x, hint.y, hint.w, hint.h), (0.1, 0.2, 0.3, 0.4))

    def test_non_watch_goals_excluded(self):
        goals = [
            watch_goal("w", "a cup is up"),
            watch_goal("e", "the door opened", type="event"),
            watch_goal("c", "cups today", type="count"),
        ]
        spec = compile_watchspec(goals, threshold=0.2)
        self.assertEqual([c.goal_id for c in spec.conditions], ["w"])

    def test_empty_goals_empty_spec(self):
        self.assertEqual(compile_watchspec([], threshold=0.2).conditions, [])


# ---------------------------------------------------------------------------
# MotionGate
# ---------------------------------------------------------------------------


class MotionGateTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_settings()
        self.spec = compile_watchspec(
            [watch_goal("g1", "a cup is up")], threshold=self.cfg.edge.motion_gate
        )

    def test_still_scene_does_not_escalate(self):
        gate = MotionGate(self.spec, settings=self.cfg)
        d0 = gate.evaluate(still(), ts=0.0)  # first frame: no prev -> motion 0
        d1 = gate.evaluate(still(), ts=1.0)  # still vs still -> motion 0
        self.assertFalse(d0.escalate)
        self.assertFalse(d1.escalate)
        self.assertEqual(d1.signal.motion, 0.0)

    def test_motion_escalates(self):
        gate = MotionGate(self.spec, settings=self.cfg)
        gate.evaluate(still(), ts=0.0)
        d = gate.evaluate(motion(), ts=1.0)
        self.assertTrue(d.escalate)
        self.assertGreater(d.signal.motion, gate.threshold())

    def test_signal_is_tier0signal_motion_only(self):
        gate = MotionGate(self.spec, settings=self.cfg)
        gate.evaluate(still(), ts=0.0)
        d = gate.evaluate(motion(), ts=2.5)
        self.assertIsInstance(d.signal, Tier0Signal)
        self.assertEqual(d.signal.structural, 0.0)
        self.assertEqual(d.signal.semantic, 0.0)
        self.assertEqual(d.signal.ts, 2.5)

    def test_empty_spec_never_escalates(self):
        gate = MotionGate(WatchSpec(conditions=[]), settings=self.cfg)
        gate.evaluate(still(), ts=0.0)
        d = gate.evaluate(motion(), ts=1.0)  # big motion but nothing to watch for
        self.assertFalse(d.escalate)

    def test_threshold_is_min_of_conditions(self):
        spec = compile_watchspec(
            [
                watch_goal("a", "x", motion_bias=0.3),
                watch_goal("b", "y", motion_bias=0.0),
            ],
            threshold=0.2,
        )
        gate = MotionGate(spec, settings=self.cfg)
        self.assertAlmostEqual(gate.threshold(), 0.2)  # most-sensitive goal wins

    def test_env_override_threshold(self):
        # PERCEPT_MOTION_GATE flows through load_settings into the gate default (empty spec).
        import os

        prev = os.environ.get("PERCEPT_MOTION_GATE")
        os.environ["PERCEPT_MOTION_GATE"] = "0.42"
        try:
            cfg = load_settings()
            self.assertAlmostEqual(cfg.edge.motion_gate, 0.42)
            gate = MotionGate(WatchSpec(conditions=[]), settings=cfg)
            self.assertAlmostEqual(gate.threshold(), 0.42)
        finally:
            if prev is None:
                del os.environ["PERCEPT_MOTION_GATE"]
            else:
                os.environ["PERCEPT_MOTION_GATE"] = prev


# ---------------------------------------------------------------------------
# the wired loop — the proof point
# ---------------------------------------------------------------------------


def fixture_stream(total: int = 24, motion_at=(5, 6, 14)) -> list[Frame]:
    """Mostly still, occasional motion bursts (deterministic)."""
    motion_at = set(motion_at)
    return [motion() if i in motion_at else still() for i in range(total)]


class WiredLoopTests(unittest.TestCase):
    def _core(self) -> Percept:
        core = Percept.create()  # offline fakes
        core.add_goal(watch_goal("g1", "a cup is up"))
        return core

    def test_loop_compiles_spec_and_consumes_it(self):
        core = self._core()
        loop = GatedWatchLoop(core, list(core.goals.active()))
        # compile_watchspec's output is actually consumed by the gate (the wired contract).
        self.assertEqual(len(loop.spec.conditions), 1)
        self.assertIs(loop.gate.spec, loop.spec)

    def test_reports_tier0signal_per_frame(self):
        core = self._core()
        stream = fixture_stream(total=10, motion_at=(3,))
        report = asyncio.run(run_gated_watch(core, stream))
        self.assertEqual(len(report.signals), 10)  # one Tier0Signal reported UP per frame
        self.assertTrue(all(isinstance(s, Tier0Signal) for s in report.signals))

    def test_vision_called_only_on_escalation(self):
        core = self._core()
        stream = fixture_stream(total=24, motion_at=(5, 6, 14))
        report = asyncio.run(run_gated_watch(core, stream))
        # 3 motion frames; frame 5 and 14 are still->motion rising edges, frame 6 is motion->motion
        # (also a change since the patch is identical -> diff 0... actually 6 follows 5 identical).
        self.assertEqual(report.vision_calls, report.escalations)
        self.assertGreater(report.escalations, 0)
        self.assertLess(report.vision_calls, report.frames)  # not every frame spent

    def test_reduces_vision_calls_vs_ungated(self):
        core = self._core()
        stream = fixture_stream(total=24, motion_at=(5, 6, 14))
        report = asyncio.run(run_gated_watch(core, stream))
        # The cost claim: the gated loop spends strictly fewer cloud calls than perceive-every-frame.
        self.assertEqual(report.ungated_calls, 24)
        self.assertLess(report.vision_calls, report.ungated_calls)
        self.assertGreater(report.reduction, 0.5)  # a mostly-still scene -> big reduction

    def test_still_scene_spends_no_calls(self):
        core = self._core()
        report = asyncio.run(run_gated_watch(core, fixture_stream(total=12, motion_at=())))
        self.assertEqual(report.vision_calls, 0)  # a still scene never spends a cloud call
        self.assertEqual(report.reduction, 1.0)

    def test_no_active_goals_never_escalates(self):
        core = Percept.create()  # no goals added
        loop = GatedWatchLoop(core)  # falls back to core.goals.active() -> empty
        self.assertEqual(loop.spec.conditions, [])  # empty WatchSpec compiled down
        report = asyncio.run(loop.run(fixture_stream(total=12, motion_at=(3, 7))))
        self.assertEqual(report.vision_calls, 0)  # nothing to watch -> never look

    def test_timestamped_stream_threads_ts_into_signal(self):
        core = self._core()
        stream = [(0.5, still()), (1.5, motion()), (2.5, still())]
        report = asyncio.run(run_gated_watch(core, stream))
        self.assertEqual([s.ts for s in report.signals], [0.5, 1.5, 2.5])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
