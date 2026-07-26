"""L1 unit tests for the tier-0 salience gate and its documented operating-point evidence.

Covers, on fakes / fixtures (no keys, deterministic):
  * ``motion_score``        — still -> ~0, motion -> high; symmetry; first-frame; size mismatch;
                              **region masking** (C1 watch-out zone: only in-rectangle motion counts).
  * ``compile_watchspec``   — goals -> conditions; frozen contract fields; per-goal bias; type filter;
                              **schedule passthrough** (object + dict; absent -> None).
  * ``MotionGate``          — still scene -> NO escalation; motion -> escalate; empty spec -> never;
                              **schedule** (out-of-schedule never escalates; injected clock) +
                              **region_hint** (motion outside the zone doesn't escalate; inside does).
  * the wired loop          — Tier0Signal reported up; vision calls only on escalation; **reduces
                              calls vs ungated** on a fixture stream (the proof point).
"""

from __future__ import annotations

import asyncio
import unittest

from percept.adapters import Box, Goal, Schedule, Tier0Signal, WatchCondition, WatchSpec
from percept.agent import Percept
from percept.config import load_settings
from percept.duration_routes import UnsupportedDurationRouteError
from percept.harness.detectors import WakePolicy

from percept.harness.tier0 import (
    CivilTime,
    EGO_COMPENSATE_KEY,
    Frame,
    GatedWatchLoop,
    MotionGate,
    SALIENCE_STAT_KEY,
    civil_from_epoch,
    compile_watchspec,
    ego_residual_score,
    gray_frame,
    motion_score,
    run_gated_watch,
    schedule_active,
)

W, H = 8, 8


def still(fill: int = 0) -> Frame:
    return gray_frame(W, H, fill=fill)


def motion(n_pixels: int = 40, value: int = 255) -> Frame:
    pixels = [0] * (W * H)
    for i in range(min(n_pixels, len(pixels))):
        pixels[i] = value
    return Frame(w=W, h=H, pixels=tuple(pixels))


def motion_in_rows(row_start: int, row_end: int, value: int = 255) -> Frame:
    """A frame with rows ``[row_start, row_end)`` set to ``value`` (the rest 0) — for region tests."""
    pixels = [0] * (W * H)
    for r in range(row_start, row_end):
        for c in range(W):
            pixels[r * W + c] = value
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

    def test_full_frame_region_equals_no_region(self):
        # A region spanning the whole frame must give exactly the no-region (full-frame) score.
        a, b = still(), motion()
        full = Box(0.0, 0.0, 1.0, 1.0)
        self.assertAlmostEqual(motion_score(a, b, region=full), motion_score(a, b), places=12)

    def test_region_default_none_is_unchanged(self):
        # The default region=None path is the untouched parity path — sanity on the keyword default.
        a, b = still(), motion()
        self.assertEqual(motion_score(a, b, region=None), motion_score(a, b))

    def test_motion_only_inside_region_counts(self):
        # Change confined to the TOP half; a region over the BOTTOM half sees no motion.
        a = still()
        b = motion_in_rows(0, H // 2)  # top half flips to 255
        top = Box(0.0, 0.0, 1.0, 0.5)
        bottom = Box(0.0, 0.5, 1.0, 0.5)
        self.assertGreater(motion_score(a, b, region=top), 0.0)  # the change is in-zone
        self.assertEqual(motion_score(a, b, region=bottom), 0.0)  # masked out -> nothing

    def test_region_score_is_local_density_not_diluted(self):
        # Motion filling the top half scores ~1.0 *within* the top region (not halved by empty rows).
        a = still()
        b = motion_in_rows(0, H // 2, value=255)
        top = Box(0.0, 0.0, 1.0, 0.5)
        self.assertAlmostEqual(motion_score(a, b, region=top), 1.0, places=9)
        # while the same change over the FULL frame is diluted to ~0.5 (only half the pixels moved).
        self.assertAlmostEqual(motion_score(a, b), 0.5, places=9)

    def test_degenerate_region_scores_zero(self):
        a, b = still(), motion()
        empty = Box(0.0, 0.0, 0.0, 0.0)  # zero-area box -> nothing to diff in-zone
        self.assertEqual(motion_score(a, b, region=empty), 0.0)


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

    def test_schedule_defaults_to_none(self):
        # No schedule in goal state -> condition.schedule is None (always armed; default path).
        spec = compile_watchspec([watch_goal("g1", "cup")], threshold=0.2)
        self.assertIsNone(spec.conditions[0].schedule)

    def test_schedule_passthrough_from_object(self):
        sched = Schedule(windows=((540, 1020),), days=(1, 2, 3, 4, 5))
        spec = compile_watchspec([watch_goal("g1", "cup", schedule=sched)], threshold=0.2)
        self.assertEqual(spec.conditions[0].schedule, sched)

    def test_schedule_coerced_from_dict(self):
        g = watch_goal("g1", "cup", schedule={"windows": [[1320, 360]], "days": [6, 7]})
        spec = compile_watchspec([g], threshold=0.2)
        sched = spec.conditions[0].schedule
        self.assertIsInstance(sched, Schedule)
        self.assertEqual(sched.windows, ((1320, 360),))
        self.assertEqual(sched.days, (6, 7))

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

    def test_wake_policy_can_escalate_on_audio_event(self):
        gate = MotionGate(
            one_condition_spec(threshold=0.3),
            settings=self.cfg,
            wake_policy=WakePolicy(weights={"audio_event": 1.0}),
        )
        d = gate.evaluate(still(), ts=0.0, energy=0.5)
        self.assertTrue(d.escalate)

    def test_wake_policy_can_escalate_on_extra_presence_signal(self):
        gate = MotionGate(
            one_condition_spec(threshold=0.4),
            settings=self.cfg,
            wake_policy=WakePolicy(weights={"presence": 0.8, "semantic_relevance": 0.8}),
        )
        d = gate.evaluate(
            still(), ts=0.0, signals={"presence": 0.4, "semantic_relevance": 0.4}
        )
        self.assertTrue(d.escalate)

    # -----------------------------------------------------------------------
    # A4 regression — wake_policy=None must not silently swallow a wake-worthy
    # signal. `fused = wake_score(...) if self._wake_policy is not None else None`
    # (tier0.py) means an explicit `wake_policy=None` degrades escalation to the
    # LEGACY per-condition check (`gate_on`, default "motion") instead of the
    # weighted-fusion path — this is exactly what `percept.runtime.worker.run`
    # wires by default (its own `wake_policy` kwarg defaults to `None`). Proven
    # here end-to-end rather than asserted by reading the branch.
    # -----------------------------------------------------------------------

    def test_wake_policy_none_still_escalates_on_wakeup_worthy_motion(self):
        gate = MotionGate(one_condition_spec(threshold=0.2), settings=self.cfg, wake_policy=None)
        gate.evaluate(still(), ts=0.0)  # prime the prev-frame cursor
        d = gate.evaluate(motion(), ts=1.0)  # real, wake-worthy evidence (motion well above threshold)
        self.assertTrue(d.escalate, "wake_policy=None must not silently drop a wake-worthy signal")
        self.assertGreaterEqual(d.signal.motion, 0.2)

    def test_wake_policy_none_stays_silent_on_a_non_waking_signal(self):
        # The negative half — gives the positive assertion above teeth: wake_policy=None isn't
        # just a gate stuck open; a still scene (no wake-worthy evidence) still stays silent.
        gate = MotionGate(one_condition_spec(threshold=0.2), settings=self.cfg, wake_policy=None)
        d0 = gate.evaluate(still(), ts=0.0)
        d1 = gate.evaluate(still(), ts=1.0)
        self.assertFalse(d0.escalate)
        self.assertFalse(d1.escalate)


# ---------------------------------------------------------------------------
# Schedule enforcement (out-of-schedule frames do not escalate)
# ---------------------------------------------------------------------------


def one_condition_spec(*, threshold=0.2, region_hint=None, schedule=None) -> WatchSpec:
    """A single-condition WatchSpec with optional region_hint and schedule."""
    return WatchSpec(
        conditions=[
            WatchCondition(
                goal_id="g1", anchor_text="a cup is up", threshold=threshold,
                region_hint=region_hint, schedule=schedule,
            )
        ]
    )


class CadenceGateTests(unittest.TestCase):
    """Per-goal temporal cadence: how OFTEN to look, distinct from perception depth."""

    def setUp(self):
        self.cfg = load_settings()

    def _gate(self, **cadence):
        goal = (
            watch_goal("g1", "the screen changed", cadence=cadence)
            if cadence
            else watch_goal("g1", "the screen changed")
        )
        spec = compile_watchspec(
            [goal],
            threshold=self.cfg.edge.motion_gate,
        )
        return MotionGate(spec, settings=self.cfg)

    def test_fixed_cadence_escalates_on_a_still_stream(self):
        # gate_on="fixed" escalates every cadence_s REGARDLESS of motion — the screenshare fix: a
        # content stream with zero pixel motion would never escalate on the motion gate alone.
        gate = self._gate(min_judge_interval=2.0, gate_on="fixed")
        fires = [gate.evaluate(still(), ts=float(t)).escalate for t in range(5)]  # all STILL frames
        self.assertEqual(fires, [True, False, True, False, True])  # ts 0,2,4 — every 2.0s

    def test_motion_cadence_rate_limits(self):
        # gate_on="motion" + cadence_s rate-limits: alternate still/motion -> a big diff every tick, but
        # escalate at most every 3s (without cadence it would escalate on every one).
        gate = self._gate(min_judge_interval=3.0, gate_on="motion")
        frames = [still(), motion(), still(), motion(), still(), motion()]
        gate.evaluate(frames[0], ts=0.0)  # prime prev-frame
        fires = [gate.evaluate(frames[t], ts=float(t)).escalate for t in range(1, 6)]
        self.assertEqual(fires, [True, False, False, True, False])  # ts 1 then 4 — capped to 3s

    def test_no_cadence_is_byte_for_byte_motion_gate(self):
        gate = self._gate()  # empty cadence -> (None, "motion") -> the default path
        gate.evaluate(still(), ts=0.0)
        self.assertFalse(gate.evaluate(still(), ts=1.0).escalate)   # still vs still -> silent
        self.assertTrue(gate.evaluate(motion(), ts=2.0).escalate)   # still -> motion -> escalate

    def test_fixed_without_interval_fails_closed(self):
        # An explicitly fixed cadence without its period is malformed authoring, not a motion fallback.
        with self.assertRaises(UnsupportedDurationRouteError):
            self._gate(gate_on="fixed")


class ScheduleGateTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_settings()
        # Office hours Mon-Fri 09:00..17:00.
        self.office = Schedule(windows=((540, 1020),), days=(1, 2, 3, 4, 5))
        self.in_hours = CivilTime(weekday=1, minute_of_day=600)  # Mon 10:00
        self.out_hours = CivilTime(weekday=1, minute_of_day=60)  # Mon 01:00
        self.weekend = CivilTime(weekday=6, minute_of_day=600)  # Sat 10:00

    def _gate(self):
        return MotionGate(one_condition_spec(schedule=self.office), settings=self.cfg)

    def test_in_schedule_motion_escalates(self):
        gate = self._gate()
        gate.evaluate(still(), ts=0.0, now=self.in_hours)
        d = gate.evaluate(motion(), ts=1.0, now=self.in_hours)
        self.assertTrue(d.escalate)

    def test_out_of_schedule_motion_does_not_escalate(self):
        gate = self._gate()
        gate.evaluate(still(), ts=0.0, now=self.out_hours)
        d = gate.evaluate(motion(), ts=1.0, now=self.out_hours)  # big motion, but out of hours
        self.assertFalse(d.escalate)
        self.assertGreater(d.signal.motion, 0.0)  # signal still reported up (salience is schedule-agnostic)

    def test_out_of_schedule_by_day_does_not_escalate(self):
        gate = self._gate()
        gate.evaluate(still(), ts=0.0, now=self.weekend)
        d = gate.evaluate(motion(), ts=1.0, now=self.weekend)  # right time, wrong day
        self.assertFalse(d.escalate)

    def test_injected_clock_used_when_no_now(self):
        # A gate-level clock supplies civil time when evaluate() is called without an explicit now.
        clock_time = {"v": self.out_hours}
        gate = MotionGate(
            one_condition_spec(schedule=self.office), settings=self.cfg,
            clock=lambda: clock_time["v"],
        )
        gate.evaluate(still(), ts=0.0)
        self.assertFalse(gate.evaluate(motion(), ts=1.0).escalate)  # clock says out-of-hours
        clock_time["v"] = self.in_hours
        gate.evaluate(still(), ts=2.0)
        self.assertTrue(gate.evaluate(motion(), ts=3.0).escalate)  # clock now says in-hours

    def test_no_schedule_ignores_clock(self):
        # A condition with no schedule is always armed even with a clock parked out-of-hours.
        gate = MotionGate(
            one_condition_spec(schedule=None), settings=self.cfg, clock=lambda: self.out_hours
        )
        gate.evaluate(still(), ts=0.0)
        self.assertTrue(gate.evaluate(motion(), ts=1.0).escalate)

    def test_schedule_active_helper_opt_in(self):
        # None schedule OR None clock -> always armed (the byte-for-byte default short-circuits).
        self.assertTrue(schedule_active(None, self.out_hours))
        self.assertTrue(schedule_active(self.office, None))
        self.assertFalse(schedule_active(self.office, self.out_hours))
        self.assertTrue(schedule_active(self.office, self.in_hours))

    def test_civil_from_epoch_utc(self):
        # 1970-01-01 was a Thursday (ISO weekday 4); 01:30 UTC -> minute 90.
        ct = civil_from_epoch(90 * 60)
        self.assertEqual(ct.weekday, 4)
        self.assertEqual(ct.minute_of_day, 90)


# ---------------------------------------------------------------------------
# Watch-out zones (region_hint masks the gate's motion to a rectangle)
# ---------------------------------------------------------------------------


class RegionGateTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_settings()

    def test_motion_inside_zone_escalates(self):
        top = Box(0.0, 0.0, 1.0, 0.5)
        gate = MotionGate(one_condition_spec(threshold=0.2, region_hint=top), settings=self.cfg)
        gate.evaluate(still(), ts=0.0)
        d = gate.evaluate(motion_in_rows(0, H // 2), ts=1.0)  # change in the top zone
        self.assertTrue(d.escalate)

    def test_motion_outside_zone_does_not_escalate(self):
        top = Box(0.0, 0.0, 1.0, 0.5)
        gate = MotionGate(one_condition_spec(threshold=0.2, region_hint=top), settings=self.cfg)
        gate.evaluate(still(), ts=0.0)
        # All the motion is in the BOTTOM half; the watch-out zone is the TOP -> masked out.
        d = gate.evaluate(motion_in_rows(H // 2, H), ts=1.0)
        self.assertFalse(d.escalate)
        self.assertGreater(d.signal.motion, 0.0)  # full-frame salience still reported up

    def test_no_region_matches_full_frame_path(self):
        # Without a region_hint the gate decision is identical to the legacy full-frame gate.
        spec = one_condition_spec(threshold=0.2, region_hint=None)
        gate = MotionGate(spec, settings=self.cfg)
        gate.evaluate(still(), ts=0.0)
        d = gate.evaluate(motion_in_rows(H // 2, H), ts=1.0)  # bottom-half motion
        self.assertTrue(d.escalate)  # full-frame sees it (no masking)


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


# ---------------------------------------------------------------------------
# Standard telemetry funnel over the wired loop (seen→gated→judged→…→fired)
# ---------------------------------------------------------------------------


from percept.observability import funnel_key  # noqa: E402


class FunnelTests(unittest.TestCase):
    """The wired loop emits the funnel into the core's Metrics; the inequality chain holds."""

    def _core(self) -> Percept:
        core = Percept.create()  # offline fakes
        core.add_goal(watch_goal("g1", "a cup is up"))
        return core

    def test_seen_ge_gated_ge_judged_chain(self):
        core = self._core()
        stream = fixture_stream(total=24, motion_at=(5, 6, 14))
        report = asyncio.run(run_gated_watch(core, stream))
        snap = core.metrics.snapshot()
        seen = snap[funnel_key("seen", camera="camera")]
        gated = snap[funnel_key("gated", camera="camera")]
        judged = snap[funnel_key("judged", concern="g1")]
        # the funnel narrows monotonically: every frame seen, fewer gated, fewer-or-equal judged.
        self.assertEqual(seen, report.frames)  # one 'seen' per frame
        self.assertEqual(gated, report.escalations)  # one 'gated' per escalation
        self.assertEqual(judged, report.escalations)  # one judge per escalated frame (single goal)
        self.assertGreaterEqual(seen, gated)
        self.assertGreaterEqual(gated, judged)
        self.assertGreater(gated, 0)  # the motion bursts did escalate

    def test_still_scene_seen_but_never_gated_or_judged(self):
        core = self._core()
        report = asyncio.run(run_gated_watch(core, fixture_stream(total=12, motion_at=())))
        snap = core.metrics.snapshot()
        self.assertEqual(snap[funnel_key("seen", camera="camera")], 12)
        self.assertEqual(report.vision_calls, 0)
        self.assertNotIn(funnel_key("gated", camera="camera"), snap)  # nothing escalated
        self.assertNotIn(funnel_key("judged", concern="g1"), snap)  # so nothing judged

    def test_fired_counted_on_rising_edge(self):
        # Script a confident-YES verdict for the motion frame so an escalation fires (once, rising edge).
        from percept.adapters import Box, Observation, Verdict
        from percept.backends.fake import FakeVision

        mf = motion()
        token = str(mf)
        vision = FakeVision(
            script={token: Verdict(True, 0.95, "cup up")},
            detect_script={token: [Observation("cup", Box(0.4, 0.4, 0.2, 0.2), 0.95)]},
        )
        core = Percept.create(vision=vision)
        core.add_goal(watch_goal("g1", "a cup is up"))
        # still, MOTION (escalates -> judged YES -> FIRE), still (the still vs motion diff also
        # escalates -> judged the still frame -> _UNKNOWN -> refused). Exactly ONE rising-edge fire.
        report = asyncio.run(run_gated_watch(core, [still(), mf, still()]))
        snap = core.metrics.snapshot()
        self.assertEqual(snap[funnel_key("judged", concern="g1")], 2)  # both transitions escalate
        self.assertEqual(snap[funnel_key("fired", concern="g1")], 1)  # but only the rising edge fires
        self.assertEqual(snap[funnel_key("fired", camera="camera")], 1)
        self.assertEqual(snap[funnel_key("refused", concern="g1")], 1)  # the still-frame judge abstained
        self.assertTrue(any(getattr(e, "action", None) == "fire" for e in report.fire_events))


def _localized(val: int = 255, size: int = 2) -> Frame:
    """A small ``size x size`` patch in the top-left corner on an otherwise still 8x8 frame."""
    px = [0] * (W * H)
    for r in range(size):
        for c in range(size):
            px[r * W + c] = val
    return Frame(w=W, h=H, pixels=tuple(px))


class SalienceModeTests(unittest.TestCase):
    """Mode A (localized block statistic) + Mode B (ego compensation) — both parity-mirrored in JS."""

    def test_block_max_recovers_localized_event_that_mean_misses(self):
        prev, cur = still(), _localized()
        mean = motion_score(prev, cur)
        bmax = motion_score(prev, cur, stat="block_max")
        # A corner patch averages away under the whole-frame mean but lights a single block.
        self.assertLess(mean, 0.10)          # below the shipped 0.10 gate -> MISSED by mean
        self.assertGreaterEqual(bmax, 0.10)  # cleared by block_max -> recall recovered
        self.assertGreater(bmax, mean)

    def test_mean_stat_is_byte_for_byte_the_default(self):
        prev, cur = still(), motion(40)
        self.assertEqual(motion_score(prev, cur), motion_score(prev, cur, stat="mean"))

    def test_block_max_dominates_p95_dominates_zero(self):
        prev, cur = still(), _localized(size=3)
        p95 = motion_score(prev, cur, stat="p95")
        bmax = motion_score(prev, cur, stat="block_max")
        self.assertGreaterEqual(bmax, p95)
        self.assertGreaterEqual(p95, 0.0)
        self.assertGreaterEqual(bmax, motion_score(prev, cur) - 1e-12)  # max block >= whole-frame mean

    def test_block_stat_composes_with_region(self):
        # A patch OUTSIDE the region contributes nothing, even to block_max.
        px = [0] * (W * H)
        for r in range(6, 8):
            for c in range(6, 8):
                px[r * W + c] = 255
        cur = Frame(w=W, h=H, pixels=tuple(px))
        roi = Box(x=0.0, y=0.0, w=0.5, h=0.5)  # top-left quadrant; the patch is bottom-right
        self.assertEqual(motion_score(still(), cur, stat="block_max", region=roi), 0.0)

    def test_unknown_stat_raises(self):
        with self.assertRaises(ValueError):
            motion_score(still(), motion(10), stat="median")

    def test_ego_residual_never_exceeds_the_full_frame_mean(self):
        prev, cur = still(), motion(40)  # the (0,0) shift is in the search, so residual <= mean
        self.assertLessEqual(ego_residual_score(prev, cur), motion_score(prev, cur) + 1e-12)

    def test_ego_compensates_a_pure_pan(self):
        w, h = 16, 16
        base = tuple((x * 11 + y * 7) % 256 for y in range(h) for x in range(w))
        panned = tuple(base[y * w + min(x + 2, w - 1)] for y in range(h) for x in range(w))
        prev, cur = (w, h, base), (w, h, panned)
        self.assertGreater(motion_score(prev, cur), 0.02)          # a pan looks like lots of motion
        self.assertLess(ego_residual_score(prev, cur, k=4), 0.01)  # ...but the residual is ~0

    def test_ego_first_frame_is_zero(self):
        self.assertEqual(ego_residual_score(None, motion(10)), 0.0)


class MotionGateModeTests(unittest.TestCase):
    """The gate honors per-condition salience mode; reports energy; fills structural/semantic slots."""

    def setUp(self):
        self.cfg = load_settings()

    def _spec(self, **overrides) -> WatchSpec:
        return WatchSpec(conditions=[WatchCondition(goal_id="g", anchor_text="x", threshold=0.10, **overrides)])

    def test_block_max_condition_escalates_a_localized_event_a_mean_condition_misses(self):
        loc = _localized()
        mean_gate = MotionGate(self._spec(salience_stat="mean"), settings=self.cfg)
        bmax_gate = MotionGate(self._spec(salience_stat="block_max"), settings=self.cfg)
        mean_gate.evaluate(still(), ts=0.0)
        bmax_gate.evaluate(still(), ts=0.0)
        self.assertFalse(mean_gate.evaluate(loc, ts=1.0).escalate)  # missed by the whole-frame mean
        self.assertTrue(bmax_gate.evaluate(loc, ts=1.0).escalate)   # caught by block_max

    def test_ego_condition_suppresses_a_pan_that_a_mean_condition_fires_on(self):
        w = h = 16
        # high column-to-column contrast so a 1px pan looks like heavy whole-frame motion (mean gate fires)
        base = tuple((x * 60 + y * 30) % 256 for y in range(h) for x in range(w))
        panned = tuple(base[y * w + min(x + 1, w - 1)] for y in range(h) for x in range(w))
        prev, cur = (w, h, base), (w, h, panned)
        # precondition: this pan is a false-positive for the mean gate but not the ego gate
        self.assertGreaterEqual(motion_score(prev, cur), 0.10)
        self.assertLess(ego_residual_score(prev, cur), 0.10)
        mean_gate = MotionGate(self._spec(salience_stat="mean"), settings=self.cfg)
        ego_gate = MotionGate(self._spec(ego_compensate=True), settings=self.cfg)
        mean_gate.evaluate(prev, ts=0.0)
        ego_gate.evaluate(prev, ts=0.0)
        self.assertTrue(mean_gate.evaluate(cur, ts=1.0).escalate)   # mean false-escalates on the pan
        self.assertFalse(ego_gate.evaluate(cur, ts=1.0).escalate)   # ego does not

    def test_energy_is_reported_on_the_signal(self):
        d = MotionGate(self._spec(), settings=self.cfg).evaluate(still(), ts=0.0, energy=0.42)
        self.assertEqual(d.signal.energy, 0.42)

    def test_energy_absent_is_none_byte_for_byte(self):
        d = MotionGate(self._spec(), settings=self.cfg).evaluate(still(), ts=0.0)
        self.assertIsNone(d.signal.energy)

    def test_structural_adapter_populates_signal_and_arms_a_structural_goal(self):
        gate = MotionGate(
            self._spec(gate_on="structural"), settings=self.cfg, structural_adapter=lambda f: 0.9
        )
        d = gate.evaluate(still(), ts=0.0)
        self.assertEqual(d.signal.structural, 0.9)
        self.assertTrue(d.escalate)  # 0.9 >= 0.10

    def test_no_adapter_leaves_structural_semantic_zero(self):
        d = MotionGate(self._spec(), settings=self.cfg).evaluate(motion(40), ts=0.0)
        self.assertEqual((d.signal.structural, d.signal.semantic), (0.0, 0.0))

    def test_reported_motion_gauge_is_full_frame_mean_even_for_a_block_max_goal(self):
        # the escalate comparison uses block_max, but the up-channel gauge stays the plain mean
        loc = _localized()
        gate = MotionGate(self._spec(salience_stat="block_max"), settings=self.cfg)
        gate.evaluate(still(), ts=0.0)
        d = gate.evaluate(loc, ts=1.0)
        self.assertAlmostEqual(d.signal.motion, motion_score(still(), loc), places=12)


class CompileWatchSpecModeTests(unittest.TestCase):
    """Selection: deployment default (settings.edge) overlaid with a fail-safe per-goal override."""

    def test_deployment_default_applies_to_all_goals(self):
        spec = compile_watchspec(
            [watch_goal("g", "coffee")], threshold=0.1, salience_stat="block_max", ego_compensate=True
        )
        c = spec.conditions[0]
        self.assertEqual(c.salience_stat, "block_max")
        self.assertTrue(c.ego_compensate)

    def test_per_goal_override_beats_the_default(self):
        g = watch_goal("g", "coffee", **{SALIENCE_STAT_KEY: "p95", EGO_COMPENSATE_KEY: False})
        c = compile_watchspec([g], threshold=0.1, salience_stat="block_max", ego_compensate=True).conditions[0]
        self.assertEqual(c.salience_stat, "p95")  # per-goal wins over the deployment default
        self.assertFalse(c.ego_compensate)

    def test_malformed_override_falls_back_to_the_default(self):
        g = watch_goal("g", "coffee", **{SALIENCE_STAT_KEY: "bogus", EGO_COMPENSATE_KEY: "yes"})
        c = compile_watchspec([g], threshold=0.1, salience_stat="block_max", ego_compensate=True).conditions[0]
        self.assertEqual(c.salience_stat, "block_max")  # invalid vocab -> deployment default
        self.assertTrue(c.ego_compensate)               # non-bool -> deployment default

    def test_default_is_mean_no_ego(self):
        c = compile_watchspec([watch_goal("g", "coffee")], threshold=0.1).conditions[0]
        self.assertEqual(c.salience_stat, "mean")
        self.assertFalse(c.ego_compensate)

    def test_per_stat_threshold_applies_to_the_resolved_stat(self):
        # block_max/p95 goals get their own (higher) base cutoff; mean goals keep the mean threshold.
        st = {"block_max": 0.20, "p95": 0.15}
        gm = watch_goal("m", "mean goal")
        gb = watch_goal("b", "block goal", salience_stat="block_max")
        gp = watch_goal("p", "p95 goal", salience_stat="p95")
        spec = compile_watchspec([gm, gb, gp], threshold=0.10, stat_thresholds=st)
        by = {c.goal_id: c for c in spec.conditions}
        self.assertAlmostEqual(by["m"].threshold, 0.10)  # mean -> base threshold
        self.assertAlmostEqual(by["b"].threshold, 0.20)  # block_max -> its own cutoff
        self.assertAlmostEqual(by["p"].threshold, 0.15)  # p95 -> its own cutoff

    def test_per_stat_threshold_still_adds_motion_bias(self):
        g = watch_goal("b", "block goal", salience_stat="block_max", motion_bias=0.05)
        c = compile_watchspec([g], threshold=0.10, stat_thresholds={"block_max": 0.20}).conditions[0]
        self.assertAlmostEqual(c.threshold, 0.25)  # 0.20 base + 0.05 bias

    def test_no_stat_thresholds_falls_back_to_base(self):
        g = watch_goal("b", "block goal", salience_stat="block_max")
        c = compile_watchspec([g], threshold=0.10).conditions[0]  # no stat_thresholds passed
        self.assertAlmostEqual(c.threshold, 0.10)  # unchanged -> back-compat


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
