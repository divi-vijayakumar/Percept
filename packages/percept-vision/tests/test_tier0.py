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
  * the arithmetic edges    — the block partition's tiling identity and its degenerate grids; the
                              integer nearest-rank percentile; the ego-residual search window; the
                              cadence brake against a hostile clock; wake fusion overriding
                              ``gate_on``; adapter outputs that are not real numbers.
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

    def test_wake_policy_can_escalate_on_an_audio_reading(self):
        # Rewritten: the goal now OPTS IN with gate_on="fused" (a wake policy no longer overrides
        # every channel), and the microphone reading populates `audio` — `audio_event` names an
        # acoustic-event CLASS with no registered detector, so it is no longer aliased onto energy.
        gate = MotionGate(
            one_condition_spec(threshold=0.3, gate_on="fused"),
            settings=self.cfg,
            wake_policy=WakePolicy(weights={"audio": 1.0}),
        )
        d = gate.evaluate(still(), ts=0.0, energy=0.5)
        self.assertTrue(d.escalate)
        self.assertEqual(d.admission_causes, ("fused",))

    def test_wake_policy_can_escalate_on_extra_presence_signal(self):
        # `presence` is an overlay channel the caller measures; `semantic` is the gate's own. They
        # are two independent cues, unlike the `semantic`/`semantic_relevance` alias pair this used
        # to fuse with itself.
        gate = MotionGate(
            one_condition_spec(threshold=0.4, gate_on="fused"),
            settings=self.cfg,
            wake_policy=WakePolicy(weights={"presence": 0.8, "semantic": 0.8}),
            semantic_adapter=lambda f: 0.4,
        )
        d = gate.evaluate(still(), ts=0.0, signals={"presence": 0.4})
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


def one_condition_spec(
    *, threshold=0.2, region_hint=None, schedule=None, gate_on="motion"
) -> WatchSpec:
    """A single-condition WatchSpec with optional region_hint, schedule and gate_on."""
    return WatchSpec(
        conditions=[
            WatchCondition(
                goal_id="g1", anchor_text="a cup is up", threshold=threshold,
                region_hint=region_hint, schedule=schedule, gate_on=gate_on,
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

    def test_admission_maps_are_goal_units_and_escalations_are_frame_units(self):
        """The two denominators reported side by side, pinned so neither can be read as the other.

        ``escalations`` counts FRAMES the eye bought; ``admission_by_goal``/``admission_by_cause``
        count (frame x goal) admissions. On a single-goal run they coincide, which is exactly why
        only a multi-goal run can pin the difference — and why a reader who compares
        ``sum(admission_by_cause.values())`` against ``escalations`` on a 73-goal run sees a 73x
        discrepancy that is arithmetic, not a defect.
        """
        core = Percept.create()  # offline fakes
        core.add_goal(watch_goal("g1", "a cup is up"))
        core.add_goal(watch_goal("g2", "a plate is out"))
        report = asyncio.run(
            run_gated_watch(core, fixture_stream(total=24, motion_at=(5, 6, 14)))
        )
        goals = 2
        self.assertGreater(report.escalations, 0)
        self.assertEqual(sorted(report.admission_by_goal), ["g1", "g2"])
        # Both goals clear the same bar on the same frames, so each map totals goals x frames...
        self.assertEqual(sum(report.admission_by_goal.values()), goals * report.escalations)
        self.assertEqual(
            sum(report.admission_by_cause.values()), sum(report.admission_by_goal.values())
        )
        # ...and the frame-unit terms stay frame-unit.
        self.assertEqual(len(report.signals), report.frames)
        self.assertEqual(report.vision_calls, report.escalations)


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
        with warnings.catch_warnings():
            # 4x4 px under the 8x6 grid clamps (see BlockPartitionBoundaryTests) — expected here,
            # and it does not change the answer: the max over 1-px blocks is a max either way.
            warnings.simplefilter("ignore", DegenerateBlockPartitionWarning)
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


# ---------------------------------------------------------------------------
# Boundary coverage: the localized-salience math and the gate's admission
# arithmetic. Everything below is stdlib-only — no media, no model, no clock.
# ---------------------------------------------------------------------------

import math  # noqa: E402
import warnings  # noqa: E402
from dataclasses import replace  # noqa: E402

from percept.harness.detectors import wake_score  # noqa: E402
from percept.harness.tier0 import (  # noqa: E402
    DEGENERATE_PARTITION_REASON,
    MOTION_BIAS_KEY,
    REGION_HINT_KEY,
    UNSUPPORTED_DETECTOR_HEALTH_KIND,
    UNSUPPORTED_DETECTOR_REASON,
    DegenerateBlockPartitionWarning,
    StreamTimeRewindWarning,
    UnsupportedEdgeCapabilityWarning,
    UnsupportedMotionBiasError,
    _block_scores,
    _coerce_cadence,
    _nearest_rank_p95,
    _partition,
    _region_bounds,
    cheap_signals,
    salience_stats,
    watchspec_kwargs,
)

#: The block partition motion_score / salience_stats default to. It is hardcoded at the call site
#: and has no configuration surface, so a test is the only place its consequences are stated.
GRID = (8, 6)
#: The shipped tier-0 gate raster (percept/runtime/source.py) — every live frame is resized to it
#: before any of this arithmetic runs, so it is the raster these boundaries actually matter at.
GATE_W, GATE_H = 48, 36


class BlockPartitionBoundaryTests(unittest.TestCase):
    """The block partition itself: the tiling identity, degenerate grids, sub-grid active areas."""

    def test_blocks_tile_the_active_area_with_no_gap_and_no_overlap(self):
        # The one-pass tri-statistic derives its mean from the block numerators; that is only the
        # whole-area mean if the blocks partition the area EXACTLY. Recompute the floor-of-product
        # boundaries independently and prove every pixel is claimed exactly once.
        prev, cur = still(), motion(40)
        scores, total, area, degenerate = _block_scores(prev, cur, None, GRID)
        self.assertEqual(len(scores), GRID[0] * GRID[1])
        self.assertEqual(area, W * H)
        self.assertFalse(degenerate)  # 8x8 px carries an 8x6 grid; nothing was clamped
        claimed: set[tuple[int, int]] = set()
        covered = 0
        for j in range(GRID[1]):
            by0, by1 = (j * H) // GRID[1], ((j + 1) * H) // GRID[1]
            for i in range(GRID[0]):
                bx0, bx1 = (i * W) // GRID[0], ((i + 1) * W) // GRID[0]
                covered += (bx1 - bx0) * (by1 - by0)
                for y in range(by0, by1):
                    for x in range(bx0, bx1):
                        self.assertNotIn((x, y), claimed)  # no overlap
                        claimed.add((x, y))
        self.assertEqual(covered, W * H)  # no gap
        self.assertEqual(len(claimed), W * H)
        # ...so summing the integer numerators over the integer area IS motion_score's own mean.
        self.assertEqual(total / (area * 255.0), motion_score(prev, cur))
        self.assertEqual(salience_stats(prev, cur)["mean"], motion_score(prev, cur))

    def test_a_uniform_flip_makes_every_statistic_the_same_number(self):
        # When every block is equally active there is nothing for a localizing statistic to find:
        # block_max and p95 collapse onto the mean exactly, not approximately.
        black, white = gray_frame(W, H, 0), gray_frame(W, H, 255)
        self.assertEqual(salience_stats(black, white), {"mean": 1.0, "block_max": 1.0, "p95": 1.0})
        self.assertEqual(motion_score(black, white, stat="block_max"), motion_score(black, white))
        self.assertEqual(motion_score(black, white, stat="p95"), motion_score(black, white))

    def test_a_zero_or_negative_grid_dimension_is_refused_not_answered(self):
        # REWRITTEN (was test_a_zero_or_negative_grid_dimension_reads_zero_on_a_total_flip, which
        # asserted `([], 0, 0)` and 0.0 everywhere — it encoded the defect as the contract).
        # A non-positive grid describes no partition at all, so there is no salience it could
        # plausibly be; answering 0.0 called a frame that inverted end to end "still". It now
        # raises, exactly as an unknown `stat` name and a non-positive Frame dimension do.
        black, white = gray_frame(W, H, 0), gray_frame(W, H, 255)
        self.assertEqual(motion_score(black, white), 1.0)
        for grid in ((0, 6), (8, 0), (-1, 6), (8, -6), (0, 0)):
            with self.subTest(block_grid=grid):
                for call in (
                    lambda: _block_scores(black, white, None, grid),
                    lambda: motion_score(black, white, stat="block_max", block_grid=grid),
                    lambda: motion_score(black, white, stat="p95", block_grid=grid),
                    lambda: salience_stats(black, white, block_grid=grid),
                ):
                    with self.assertRaises(ValueError) as caught:
                        call()
                    self.assertIn("block_grid dims must be positive", str(caught.exception))
                # ...and the `mean` path never reads block_grid, on either runtime, so it is
                # unaffected by a grid it does not consult.
                self.assertEqual(motion_score(black, white, block_grid=grid), 1.0)

    def test_the_two_guard_rules_are_not_the_same_rule(self):
        # `_partition` owns both, and they must not collapse into each other: an IMPOSSIBLE grid is
        # refused, a grid merely FINER THAN THE AREA is clamped and flagged. A 6x4 region_hint is a
        # valid authored input; a (0,6) grid is not an input at all.
        with self.assertRaises(ValueError):
            _partition(0, 0, 48, 36, 0, 6)
        self.assertEqual(_partition(0, 0, 48, 36, 8, 6), (8, 6, False))  # fits -> untouched
        self.assertEqual(_partition(0, 0, 6, 4, 8, 6), (6, 4, True))     # both dims clamp
        self.assertEqual(_partition(0, 0, 4, 36, 8, 6), (4, 6, True))    # one dim clamps
        self.assertEqual(_partition(0, 0, 8, 6, 8, 6), (8, 6, False))    # exactly on the boundary
        self.assertEqual(_partition(0, 0, 7, 6, 8, 6), (7, 6, True))     # one px under it
        self.assertEqual(_partition(5, 5, 5, 9, 8, 6), (0, 4, True))     # empty span -> no blocks

    def test_an_area_smaller_than_the_grid_clamps_the_grid_instead_of_padding_it(self):
        # REWRITTEN (was test_an_area_smaller_than_the_grid_degenerates_to_single_pixel_blocks,
        # which asserted 48 blocks for a 12-pixel area and p95 == 0.0 — the defect as the contract).
        # 4x3 px under an 8x6 grid now yields a 4x3 grid: 12 blocks, every one holding a real pixel.
        prev = gray_frame(4, 3, 0)
        px = [0] * 12
        px[0] = 255
        cur = Frame(w=4, h=3, pixels=tuple(px))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DegenerateBlockPartitionWarning)
            scores, _total, area, degenerate = _block_scores(prev, cur, None, GRID)
            self.assertEqual((len(scores), area), (12, 12))  # was (48, 12): 36 empty blocks
            self.assertTrue(degenerate)
            self.assertEqual(sum(1 for s in scores if s > 0.0), 1)
            self.assertAlmostEqual(motion_score(prev, cur), 1 / 12, places=12)
            # block_max is unchanged — the padding never touched a max.
            self.assertEqual(motion_score(prev, cur, stat="block_max"), 1.0)
            # p95 MOVES: 0.0 -> 1.0. Over 12 real blocks the nearest rank is 11, the top block; the
            # 36 phantom 0.0 samples used to push the rank down to a block that did not exist.
            self.assertEqual(motion_score(prev, cur, stat="p95"), 1.0)

    def test_a_clamped_partition_is_reported_with_the_adr_reason_code(self):
        # The clamp changes what the statistic MEANS (a p95 over 12 real blocks is not a p95 over 48
        # blocks, 36 of them padding), so it is not allowed to be silent. ADR-002 §2's
        # PLAN_UNOBSERVABLE is the code: a configuration failure, not a runtime one.
        prev, cur = gray_frame(4, 3, 0), gray_frame(4, 3, 255)
        for call in (
            lambda: motion_score(prev, cur, stat="block_max"),
            lambda: motion_score(prev, cur, stat="p95"),
            lambda: salience_stats(prev, cur),
        ):
            with self.subTest(call=call):
                with self.assertWarns(DegenerateBlockPartitionWarning) as caught:
                    call()
                self.assertIn(DEGENERATE_PARTITION_REASON, str(caught.warning))
                self.assertEqual(DEGENERATE_PARTITION_REASON, "PLAN_UNOBSERVABLE")
        # An area the grid fits is silent — the report tracks the clamp, not the block path.
        big_prev, big_cur = gray_frame(GATE_W, GATE_H, 0), gray_frame(GATE_W, GATE_H, 255)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DegenerateBlockPartitionWarning)
            self.assertEqual(motion_score(big_prev, big_cur, stat="p95"), 1.0)

    def test_a_watch_out_zone_smaller_than_the_grid_clamps_the_same_way(self):
        # REWRITTEN (was test_a_watch_out_zone_smaller_than_the_grid_degenerates_the_same_way).
        # The same clamp is reachable from AUTHORED state at the shipped gate raster: a 10%-square
        # region_hint is 6x4 px, narrower than the 8x6 grid in both dimensions.
        zone = Box(0.1, 0.1, 0.1, 0.1)
        x0, y0, x1, y1 = _region_bounds(zone, GATE_W, GATE_H)
        self.assertEqual((x1 - x0, y1 - y0), (6, 4))
        prev = gray_frame(GATE_W, GATE_H, 0)
        px = [0] * (GATE_W * GATE_H)
        px[y0 * GATE_W + x0] = 255  # exactly one pixel inside the zone moves
        cur = Frame(w=GATE_W, h=GATE_H, pixels=tuple(px))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DegenerateBlockPartitionWarning)
            scores, _total, area, degenerate = _block_scores(prev, cur, zone, GRID)
            self.assertEqual((len(scores), area, degenerate), (24, 24, True))  # was 48 blocks
            self.assertEqual(motion_score(prev, cur, region=zone, stat="block_max"), 1.0)
            # p95 still reads 0.0 here, and the empty blocks were never the reason: over the 24 REAL
            # blocks the nearest rank is 22, and 23 of the 24 are 0.0. One active block out of 24 is
            # 4% of the population — under the 5% a 95th percentile is defined to ignore. Clamping
            # fixes the fictional population; it does not make p95 a one-block detector.
            self.assertEqual(motion_score(prev, cur, region=zone, stat="p95"), 0.0)
        # the identical pixel inside a zone the grid CAN tile is averaged over its 3x3-px block
        wide = Box(0.0, 0.0, 0.5, 0.5)
        self.assertEqual(_region_bounds(wide, GATE_W, GATE_H), (0, 0, 24, 18))
        self.assertAlmostEqual(
            motion_score(prev, cur, region=wide, stat="block_max"), 1 / 9, places=12
        )

    def test_clamping_does_not_disarm_the_single_pixel_block_max_hair_trigger(self):
        # PARKED, stated here so it is not mistaken for fixed. block_max over 1-px blocks reads one
        # pixel at full scale, and 51/255 == 0.20 clears the shipped block_max cutoff EXACTLY. The
        # clamp changes how many blocks exist, not how small the smallest one is, so the trigger
        # survives it. Disarming it needs a MINIMUM BLOCK AREA, which moves the calibrated 0.20
        # cutoff onto a different value scale — a re-measurement, not a code change.
        prev = gray_frame(4, 3, 0)
        px = [0] * 12
        px[0] = 51  # one pixel, one fifth of full scale
        cur = Frame(w=4, h=3, pixels=tuple(px))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DegenerateBlockPartitionWarning)
            self.assertEqual(motion_score(prev, cur, stat="block_max"), 0.20)
        self.assertEqual(load_settings().edge.motion_gate_block_max, 0.20)

    def test_block_max_admits_at_exactly_its_threshold(self):
        # 8x8 under an 8x6 grid makes the top-left block exactly 1x1 px, so a 51/255 step is a
        # block_max of exactly 0.20 — the shipped cutoff, hit dead on.
        prev = still()
        px = [0] * (W * H)
        px[0] = 51
        cur = Frame(w=W, h=H, pixels=tuple(px))
        self.assertEqual(motion_score(prev, cur, stat="block_max"), 0.2)
        cfg = load_settings()
        self.assertEqual(cfg.edge.motion_gate_block_max, 0.20)
        gate = MotionGate(
            WatchSpec(conditions=[WatchCondition(
                goal_id="g", anchor_text="x", threshold=0.20, salience_stat="block_max",
            )]),
            settings=cfg,
        )
        gate.evaluate(prev, ts=0.0)
        self.assertTrue(gate.evaluate(cur, ts=1.0).escalate)  # comparison is >=, so equality admits


class NearestRankPercentileTests(unittest.TestCase):
    """The integer nearest-rank percentile, called directly — the shape of its rank is the point."""

    def test_an_empty_score_list_is_zero(self):
        self.assertEqual(_nearest_rank_p95([]), 0.0)

    def test_it_collapses_onto_the_maximum_for_nineteen_or_fewer_blocks(self):
        # ceil(0.95n) == n while 0.05n < 1, i.e. for n <= 19 — at which point the "noise-robust
        # cousin of block_max" IS block_max and contributes no robustness at all.
        for n in range(1, 20):
            with self.subTest(n=n):
                scores = [i / 100 for i in range(n)]
                self.assertEqual(_nearest_rank_p95(scores), max(scores))
        twenty = [i / 100 for i in range(20)]
        self.assertNotEqual(_nearest_rank_p95(twenty), max(twenty))  # 20 is already past the edge
        self.assertEqual(_nearest_rank_p95(twenty), sorted(twenty)[18])

    def test_a_small_block_grid_makes_p95_identical_to_block_max(self):
        # The collapse reached through the public entry point: 4x4 = 16 blocks is under the edge.
        prev, cur = still(), _localized()
        p95 = motion_score(prev, cur, stat="p95", block_grid=(4, 4))
        self.assertGreater(p95, 0.0)
        self.assertEqual(p95, motion_score(prev, cur, stat="block_max", block_grid=(4, 4)))

    def test_the_shipped_forty_eight_block_grid_ranks_the_third_largest_block(self):
        scores = [i / 100 for i in range(48)]
        self.assertEqual(_nearest_rank_p95(scores), sorted(scores)[45])
        self.assertEqual(_nearest_rank_p95(scores), sorted(scores)[-3])

    def test_p95_reports_nothing_until_three_blocks_are_active(self):
        # The consequence of ranking third-from-top over a 48-block grid: a localized event that
        # lights one or two blocks — the exact case block_max exists for — scores exactly 0.0.
        for active in (1, 2):
            with self.subTest(active=active):
                self.assertEqual(_nearest_rank_p95([0.0] * (48 - active) + [0.9] * active), 0.0)
        self.assertEqual(_nearest_rank_p95([0.0] * 45 + [0.9] * 3), 0.9)

    def test_the_index_clamps_are_unreachable_for_any_non_empty_input(self):
        # Both guard branches are dead code for every n >= 1: the rank is already in range.
        for n in range(1, 201):
            idx = (95 * n + 99) // 100 - 1
            self.assertGreaterEqual(idx, 0)
            self.assertLessEqual(idx, n - 1)
            scores = [i / 1000 for i in range(n)]
            self.assertEqual(_nearest_rank_p95(scores), sorted(scores)[idx])

    def test_ties_return_the_tied_value(self):
        self.assertEqual(_nearest_rank_p95([0.4] * 48), 0.4)
        self.assertEqual(_nearest_rank_p95([0.4, 0.4, 0.4, 0.9]), 0.9)  # n=4 -> rank 4 -> the max
        self.assertEqual(_nearest_rank_p95([0.0] * 47 + [0.4]), 0.0)


def _texture(w: int, h: int, seed: int = 20260731) -> list[int]:
    """A deterministic high-entropy luma field: only the RIGHT shift explains a pan of it."""
    px: list[int] = []
    s = seed
    for _ in range(w * h):
        s = (1103515245 * s + 12345) % (2 ** 31)
        px.append(s % 256)
    return px


def _panned(px: list[int], w: int, h: int, dx: int) -> list[int]:
    """``px`` shifted right by ``dx`` columns; the vacated columns are filled with 0."""
    out = [0] * (w * h)
    for y in range(h):
        for x in range(dx, w):
            out[y * w + x] = px[y * w + x - dx]
    return out


class EgoResidualBoundaryTests(unittest.TestCase):
    """The ego-motion residual's search window, its unreachable branch, and what it discards."""

    def setUp(self):
        self.w = self.h = 16
        self.base = _texture(self.w, self.h)
        self.prev = Frame(w=self.w, h=self.h, pixels=tuple(self.base))
        self.pan4 = Frame(w=self.w, h=self.h, pixels=tuple(_panned(self.base, self.w, self.h, 4)))
        self.pan5 = Frame(w=self.w, h=self.h, pixels=tuple(_panned(self.base, self.w, self.h, 5)))

    def test_a_pan_of_exactly_k_pixels_is_compensated_and_k_plus_one_is_not(self):
        # The default k=4 is a hard cliff, not a taper: 4 px of camera motion is explained away to
        # exactly 0.0, and 5 px is indistinguishable from the scene changing.
        self.assertGreater(motion_score(self.prev, self.pan4), 0.10)  # both look like heavy motion
        self.assertGreater(motion_score(self.prev, self.pan5), 0.10)
        self.assertEqual(ego_residual_score(self.prev, self.pan4, k=4), 0.0)
        self.assertGreater(ego_residual_score(self.prev, self.pan5, k=4), 0.30)
        self.assertEqual(ego_residual_score(self.prev, self.pan5, k=5), 0.0)  # widen k, and it is

    def test_k_zero_is_the_plain_mean_and_a_negative_k_clamps_onto_it(self):
        plain = motion_score(self.prev, self.pan4)
        self.assertEqual(ego_residual_score(self.prev, self.pan4, k=0), plain)
        self.assertEqual(ego_residual_score(self.prev, self.pan4, k=-5), plain)
        self.assertLess(ego_residual_score(self.prev, self.pan4, k=4), plain)  # the search pays off

    def test_the_residual_never_exceeds_the_full_frame_mean(self):
        flip = gray_frame(self.w, self.h, 255)
        for name, cur in (("pan4", self.pan4), ("pan5", self.pan5), ("flip", flip)):
            for k in (0, 1, 4, 7):
                with self.subTest(fixture=name, k=k):
                    self.assertLessEqual(
                        ego_residual_score(self.prev, cur, k=k),
                        motion_score(self.prev, cur) + 1e-12,
                    )

    def test_a_frame_narrower_than_the_search_window_still_scores_its_zero_shift(self):
        # A 3x3 frame is smaller than the 2k+1 = 9 px window, so every |shift| >= 3 is skipped —
        # but the (0,0) shift is ALWAYS in the search, so the result is the plain mean. The
        # "no candidate shift" fallback is unreachable: a Frame cannot have a zero dimension.
        prev, cur = gray_frame(3, 3, 0), gray_frame(3, 3, 255)
        self.assertEqual(ego_residual_score(prev, cur, k=4), 1.0)
        self.assertEqual(ego_residual_score(prev, cur, k=4), motion_score(prev, cur))
        one = (gray_frame(1, 1, 0), gray_frame(1, 1, 255))
        self.assertEqual(ego_residual_score(*one, k=4), 1.0)
        with self.assertRaises(ValueError):
            Frame(w=0, h=3, pixels=())

    def _ego_localized_pair(self) -> tuple[Frame, Frame]:
        """A texture background plus a 2x2 patch that fills one whole block at full scale."""
        prev_px, cur_px = list(self.base), list(self.base)
        for y in range(2):
            for x in range(2):
                prev_px[y * self.w + x] = 0
                cur_px[y * self.w + x] = 255
        return (
            Frame(w=self.w, h=self.h, pixels=tuple(prev_px)),
            Frame(w=self.w, h=self.h, pixels=tuple(cur_px)),
        )

    def test_an_ego_goal_keeps_its_selected_residual_statistic_and_threshold_scale(self):
        # The global shift is estimated with mean residual, but the ALIGNED residual is summarized
        # with the condition's stat. That separation is the configuration contract: a condition
        # compiled as block_max@0.20 must never compare a mean-scale number to the block cutoff.
        prev, cur = self._ego_localized_pair()
        self.assertEqual(motion_score(prev, cur, stat="block_max"), 1.0)
        self.assertAlmostEqual(ego_residual_score(prev, cur), 0.015625, places=12)
        self.assertEqual(ego_residual_score(prev, cur, stat="block_max"), 1.0)

        def _gate(**overrides):
            spec = WatchSpec(conditions=[WatchCondition(
                goal_id="g", anchor_text="x", threshold=0.20, **overrides,
            )])
            gate = MotionGate(spec, settings=load_settings())
            gate.evaluate(prev, ts=0.0)
            return gate.evaluate(cur, ts=1.0).escalate

        self.assertTrue(_gate(salience_stat="block_max"))
        self.assertTrue(_gate(salience_stat="block_max", ego_compensate=True))

    def test_ego_block_stats_still_suppress_a_pure_pan(self):
        # Applying a localizing stat happens AFTER global alignment; it must not reintroduce the
        # camera pan that alignment explained away.
        self.assertEqual(ego_residual_score(self.prev, self.pan4, k=4, stat="block_max"), 0.0)
        self.assertEqual(ego_residual_score(self.prev, self.pan4, k=4, stat="p95"), 0.0)

    def _out_of_zone_pair(self) -> tuple[Frame, Frame]:
        """A 6x6 patch in the BOTTOM-RIGHT quadrant — outside a top-left watch-out zone."""
        prev_px, cur_px = list(self.base), list(self.base)
        for y in range(10, 16):
            for x in range(10, 16):
                prev_px[y * self.w + x] = 0
                cur_px[y * self.w + x] = 255
        return (
            Frame(w=self.w, h=self.h, pixels=tuple(prev_px)),
            Frame(w=self.w, h=self.h, pixels=tuple(cur_px)),
        )

    def test_an_ego_goal_still_honours_its_watch_out_zone(self):
        # Was xfail(strict) — an ego goal returned BEFORE the region branch, so `ego_residual_score`
        # had no region parameter at all and the residual was always whole-frame while the condition
        # still carried and still reported its zone. A watch-out zone is a promise that movement
        # elsewhere is ignored; turning on ego compensation must not quietly revoke it.
        prev, cur = self._out_of_zone_pair()
        zone = Box(0.0, 0.0, 0.5, 0.5)  # top-left; every changed pixel is bottom-right
        self.assertEqual(motion_score(prev, cur, region=zone), 0.0)  # nothing happened in-zone
        self.assertGreater(ego_residual_score(prev, cur), 0.0)       # ...but plenty happened
        self.assertEqual(ego_residual_score(prev, cur, region=zone), 0.0)
        spec = WatchSpec(conditions=[WatchCondition(
            goal_id="g", anchor_text="x", threshold=0.10,
            region_hint=zone, ego_compensate=True,
        )])
        gate = MotionGate(spec, settings=load_settings())
        gate.evaluate(prev, ts=0.0)
        self.assertFalse(gate.evaluate(cur, ts=1.0).escalate)

    def test_the_zone_masks_what_is_measured_and_never_what_is_estimated(self):
        # The composition rule, stated as a test. A pan is a whole-frame property: the shift search
        # stays global, so a zone too small or too flat to identify a translation still gets the
        # right one. If the zone picked its own shift, a uniform zone would find a bogus (dx,dy).
        pan = self.pan4
        for zone in (Box(0.0, 0.0, 0.25, 0.25), Box(0.5, 0.5, 0.5, 0.5), Box(0.0, 0.0, 1.0, 1.0)):
            with self.subTest(zone=zone):
                # A pure pan is explained away in every zone, because the global estimate is right.
                self.assertEqual(ego_residual_score(self.prev, pan, k=4, region=zone), 0.0)
        # And a zone still masks: the in-zone residual of a localized subject is 0 outside it.
        prev, cur = self._out_of_zone_pair()
        whole = ego_residual_score(prev, cur)
        self.assertEqual(ego_residual_score(prev, cur, region=Box(0.0, 0.0, 1.0, 1.0)), whole)
        self.assertEqual(ego_residual_score(prev, cur, region=Box(0.0, 0.0, 0.5, 0.5)), 0.0)
        self.assertGreater(ego_residual_score(prev, cur, region=Box(0.5, 0.5, 0.5, 0.5)), whole)

    def test_the_numpy_and_stdlib_region_residuals_are_the_same_double(self):
        # The module claims numpy is "an OPTIONAL fast path ... the stdlib path is the source of
        # truth", and the region mask added a SECOND pair of pixel loops that has to honour it. The
        # numpy branch slices with `my0 - dy`, which silently WRAPS if it ever goes negative, so
        # this walks shifts of both signs against a zone on each edge. `assertEqual` on floats, not
        # almost-equal: an integer sum over an integer denominator has no rounding to forgive.
        import percept.harness.tier0 as t0

        prev, cur = self._out_of_zone_pair()
        zones = [
            Box(0.0, 0.0, 0.5, 0.5), Box(0.5, 0.5, 0.5, 0.5), Box(0.25, 0.25, 0.5, 0.5),
            Box(0.0, 0.0, 1.0, 1.0), Box(0.9, 0.0, 0.2, 1.0), None,
        ]
        with_numpy = [
            [ego_residual_score(prev, cur, k=k, region=z) for z in zones] for k in (0, 1, 4, 9)
        ]
        saved = t0._np
        try:
            t0._np = None
            without = [
                [ego_residual_score(prev, cur, k=k, region=z) for z in zones] for k in (0, 1, 4, 9)
            ]
        finally:
            t0._np = saved
        self.assertIsNotNone(saved, "the fast path was not installed, so this proves nothing")
        self.assertEqual(with_numpy, without)
        self.assertGreater(max(max(row) for row in with_numpy), 0.0)  # not all-zero by accident

    def test_a_zone_the_winning_shift_shifts_off_the_frame_scores_zero(self):
        # region ∩ overlap can be empty: a 1-px-wide zone on the left edge, against a shift that
        # moves the overlap right past it. Nothing in-zone survives the shift, so there is nothing
        # to measure — 0.0, the same answer `_motion_score_region` gives an empty span.
        prev, cur = gray_frame(8, 8, 0), gray_frame(8, 8, 255)
        sliver = Box(0.0, 0.0, 0.05, 1.0)  # x0=0, x1=round(0.4)=0 -> empty span
        self.assertEqual(_region_bounds(sliver, 8, 8), (0, 0, 0, 8))
        self.assertEqual(ego_residual_score(prev, cur, k=2, region=sliver), 0.0)
        # An off-frame zone is the same permanent never-fire it is on the non-ego path.
        off = Box(2.0, 0.0, 0.5, 1.0)
        self.assertEqual(motion_score(prev, cur, region=off), 0.0)
        self.assertEqual(ego_residual_score(prev, cur, k=2, region=off), 0.0)


class GatedLoopSettingsThreadingTests(unittest.TestCase):
    """What the deployment configured, versus what the wired loop actually compiled."""

    def _settings(self):
        cfg = load_settings()
        return replace(cfg, edge=replace(
            cfg.edge, salience_stat="block_max", ego_compensate=False,
            motion_gate=0.10, motion_gate_block_max=0.20,
        ))

    def _core_and_goal(self):
        core = Percept.create()  # offline fakes
        goal = watch_goal("g1", "a cup is up")
        core.add_goal(goal)
        return core, goal

    def test_the_shipped_deployment_default_is_the_localized_statistic(self):
        # The premise of the two expectations below: the product ships block_max at 0.20, not mean.
        edge = load_settings().edge
        self.assertEqual(edge.salience_stat, "block_max")
        self.assertEqual(edge.motion_gate_block_max, 0.20)
        # ...and compile_watchspec honours it when it is actually passed.
        spec = compile_watchspec(
            [watch_goal("g1", "a cup is up")],
            threshold=edge.motion_gate,
            salience_stat=edge.salience_stat,
            stat_thresholds={"mean": edge.motion_gate, "block_max": edge.motion_gate_block_max},
        )
        self.assertEqual(spec.conditions[0].salience_stat, "block_max")
        self.assertAlmostEqual(spec.conditions[0].threshold, 0.20)

    def test_the_loop_compiles_the_deployment_salience_mode_into_its_spec(self):
        core, goal = self._core_and_goal()
        loop = GatedWatchLoop(core, [goal], settings=self._settings())
        condition = loop.spec.conditions[0]
        self.assertEqual(condition.salience_stat, "block_max")
        self.assertAlmostEqual(condition.threshold, 0.20)

    def test_the_loop_and_the_worker_read_one_shared_settings_reader(self):
        # The defect was two call sites building the same four kwargs independently, and the loop
        # building none of them. There is now exactly one function that turns settings into them.
        cfg = self._settings()
        self.assertEqual(
            watchspec_kwargs(cfg),
            {
                "threshold": 0.10,
                "salience_stat": "block_max",
                "ego_compensate": False,
                "stat_thresholds": {
                    "mean": 0.10, "block_max": 0.20, "p95": cfg.edge.motion_gate_p95,
                },
            },
        )
        core, goal = self._core_and_goal()
        loop = GatedWatchLoop(core, [goal], settings=cfg)
        self.assertEqual(loop.spec.conditions[0].salience_stat, cfg.edge.salience_stat)

    def test_an_explicit_threshold_overrides_the_whole_per_stat_table(self):
        # ...and an explicit threshold is still a cutoff, not a mean-scale cutoff: the per-stat
        # table must not shadow what the caller asked for, while the configured STAT is untouched.
        core, goal = self._core_and_goal()
        loop = GatedWatchLoop(core, [goal], threshold=0.55, settings=self._settings())
        condition = loop.spec.conditions[0]
        self.assertAlmostEqual(condition.threshold, 0.55)
        self.assertEqual(condition.salience_stat, "block_max")

    def test_the_loop_admits_the_localized_event_its_deployment_default_is_tuned_for(self):
        # The user-visible half of the same defect: a corner event, on a still scene, at the
        # deployment's own configured mode. block_max scores it 1.0 against a 0.20 cutoff.
        core, goal = self._core_and_goal()
        self.assertEqual(motion_score(still(), _localized(), stat="block_max"), 1.0)
        loop = GatedWatchLoop(core, [goal], settings=self._settings())
        report = asyncio.run(loop.run([still(), _localized(), still(), _localized()]))
        self.assertGreater(report.escalations, 0)

    def test_a_detector_the_edge_cannot_run_is_reported_on_all_three_channels(self):
        # `capabilities=None` and `capabilities=[]` are opposite instructions and only one of them
        # used to be audible: an edge whose handshake advertised nothing compiled a condition that
        # admits frames and buys judge calls but can never emit a boundary, silently.
        core = Percept.create()
        goal = watch_goal("reps", "count squats")
        goal.state["detectors"] = ["motion-periodicity"]
        core.add_goal(goal)
        health: list = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertLogs("percept.harness.tier0", level="WARNING") as logs:
                loop = GatedWatchLoop(
                    core, [goal], settings=self._settings(),
                    edge_capabilities=[], on_health=health.append,
                )
        dropped = [w for w in caught if issubclass(w.category, UnsupportedEdgeCapabilityWarning)]
        self.assertEqual(len(dropped), 1)                                     # 1. the warning
        self.assertTrue(any("motion-periodicity" in line for line in logs.output), logs.output)
        events = [e for e in health if e.kind == UNSUPPORTED_DETECTOR_HEALTH_KIND]  # 3. the health
        self.assertEqual([e.data["goal_id"] for e in events], ["reps"])
        self.assertEqual(events[0].data["dropped_detectors"], ["motion-periodicity"])
        self.assertEqual(events[0].data["reason"], UNSUPPORTED_DETECTOR_REASON)  # ADR-002 §2
        self.assertEqual(events[0].data["driver"], "GatedWatchLoop")
        # ...and the same drop as data, without a new field on the WatchSpec wire contract.
        self.assertEqual(loop.unsupported_detectors, [("reps", ["motion-periodicity"])])
        self.assertEqual(loop.spec.conditions[0].detectors, [])
        self.assertFalse(hasattr(loop.spec, "unsupported"))

    def test_the_in_process_edge_still_arms_every_detector_without_a_word(self):
        core = Percept.create()
        goal = watch_goal("reps", "count squats")
        goal.state["detectors"] = ["motion-periodicity"]
        core.add_goal(goal)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loop = GatedWatchLoop(core, [goal], settings=self._settings())
        self.assertEqual(caught, [])
        self.assertEqual(loop.unsupported_detectors, [])
        self.assertEqual(loop.spec.conditions[0].detectors, ["motion-periodicity"])


class CompileWatchSpecFirewallTests(unittest.TestCase):
    """Untrusted goal state: which hints are firewalled, and which one takes the compile down."""

    def test_every_other_malformed_goal_state_hint_falls_back_to_a_default(self):
        goal = watch_goal(
            "g", "a cup is up",
            **{SALIENCE_STAT_KEY: "bogus", EGO_COMPENSATE_KEY: "yes",
               REGION_HINT_KEY: object(), "schedule": "office hours"},
        )
        condition = compile_watchspec([goal], threshold=0.1).conditions[0]
        self.assertEqual(condition.salience_stat, "mean")
        self.assertFalse(condition.ego_compensate)
        self.assertIsNone(condition.region_hint)  # an unknown hint shape widens to the whole frame
        self.assertIsNone(condition.schedule)     # ...and an unknown schedule arms the goal 24/7
        self.assertEqual(condition.gate_on, "motion")
        self.assertIsNone(condition.cadence_s)

    def test_a_malformed_cadence_is_refused_loudly_before_the_silent_coercer_sees_it(self):
        # The cadence coercer's quiet (None, "motion") fallback is NOT what a caller meets here: a
        # duration-route preflight runs first and names the offending field. Every shape the
        # coercer would have absorbed is a controlled refusal at this boundary.
        for raw in (
            "every now and then", [1, 2],
            {"min_judge_interval": 0}, {"min_judge_interval": -2},
            {"min_judge_interval": "2"}, {"min_judge_interval": float("nan")},
            {"min_judge_interval": float("inf")}, {"min_judge_interval": True},
            {"gate_on": "fixed"}, {"min_judge_interval": 2, "gate_on": "bogus"},
        ):
            with self.subTest(cadence=repr(raw)):
                with self.assertRaises(UnsupportedDurationRouteError):
                    compile_watchspec(
                        [watch_goal("g", "a cup is up", cadence=raw)], threshold=0.1
                    )
        # the well-formed shape still compiles, and an EMPTY mapping reads as "no cadence authored"
        ok = compile_watchspec(
            [watch_goal("g", "a cup is up", cadence={"min_judge_interval": 2.0})], threshold=0.1
        ).conditions[0]
        self.assertEqual((ok.cadence_s, ok.gate_on), (2.0, "motion"))
        empty = compile_watchspec(
            [watch_goal("g", "a cup is up", cadence={})], threshold=0.1
        ).conditions[0]
        self.assertEqual((empty.cadence_s, empty.gate_on), (None, "motion"))

    def test_a_non_numeric_motion_bias_is_refused_by_name_and_takes_the_batch_with_it(self):
        # Rewritten: this used to pin `assertRaises(ValueError)` / `assertRaises(TypeError)` — i.e.
        # whatever bare stdlib error `float()` happened to raise, naming neither the goal nor the
        # field. Aborting the batch is NOT the defect (the module's own precedent,
        # require_duration_routes, refuses an unusable authored value exactly this loudly); the
        # unattributed error was. The bias now has a coercer in the _coerce_* family, and its
        # refusal is typed and says which goal carried what.
        healthy = watch_goal("healthy", "a cup is up")
        noisy = watch_goal("noisy", "a dog is barking", **{MOTION_BIAS_KEY: "high"})
        self.assertEqual(len(compile_watchspec([healthy], threshold=0.1).conditions), 1)
        for order in ([healthy, noisy], [noisy, healthy]):
            with self.subTest(order=[g.id for g in order]):
                with self.assertRaises(UnsupportedMotionBiasError) as caught:
                    compile_watchspec(order, threshold=0.1)
                self.assertEqual(caught.exception.goal_id, "noisy")
                self.assertIsInstance(caught.exception, ValueError)  # still a ValueError
                rendered = str(caught.exception)
                self.assertIn("noisy", rendered)          # WHICH goal
                self.assertIn(MOTION_BIAS_KEY, rendered)  # ...and WHICH field
        for bad in (float("nan"), float("inf"), [0.1]):
            with self.subTest(bias=repr(bad)):
                with self.assertRaises(UnsupportedMotionBiasError):
                    compile_watchspec(
                        [healthy, watch_goal("bad", "x", **{MOTION_BIAS_KEY: bad})], threshold=0.1
                    )

    def test_an_explicitly_null_motion_bias_reads_as_no_bias(self):
        # Rewritten from a `assertRaises(TypeError)` pin. An explicit None is what every other
        # _coerce_* in this module reads as "absent" (no region, no schedule, no cadence), and
        # absent is exactly what an unset bias means. float(None) raising was an accident of the
        # missing coercer, not a decision.
        condition = compile_watchspec(
            [watch_goal("none", "x", **{MOTION_BIAS_KEY: None})], threshold=0.1
        ).conditions[0]
        self.assertAlmostEqual(condition.threshold, 0.1)

    def test_a_boolean_motion_bias_is_refused_instead_of_saturating_the_cutoff(self):
        # Rewritten: this used to assert threshold == 1.0, encoding the defect as the contract.
        # bool is an int, so True passed the old float() and silently biased the cutoff by a full
        # point — clamped to 1.0, a Watch that admits NOTHING, EVER, reported as a successful
        # compile. The loudest possible failure, delivered in total silence.
        for flag in (True, False):
            with self.subTest(bias=flag):
                with self.assertRaises(UnsupportedMotionBiasError) as caught:
                    compile_watchspec(
                        [watch_goal("g", "x", **{MOTION_BIAS_KEY: flag})], threshold=0.1
                    )
                self.assertIn("bool", str(caught.exception))
        # a real number still biases, and still clamps at the ceiling
        self.assertAlmostEqual(
            compile_watchspec(
                [watch_goal("g", "x", **{MOTION_BIAS_KEY: 0.25})], threshold=0.1
            ).conditions[0].threshold,
            0.35,
        )
        self.assertEqual(
            compile_watchspec(
                [watch_goal("g", "x", **{MOTION_BIAS_KEY: 5.0})], threshold=0.1
            ).conditions[0].threshold,
            1.0,
        )


class CadenceClockBoundaryTests(unittest.TestCase):
    """The cadence brake: its coercion table, its exact boundary, and a hostile timestamp."""

    def setUp(self):
        self.cfg = load_settings()

    def _gate(self, cadence_s=2.0, gate_on="motion", threshold=0.10):
        return MotionGate(
            WatchSpec(conditions=[WatchCondition(
                goal_id="g", anchor_text="x", threshold=threshold,
                cadence_s=cadence_s, gate_on=gate_on,
            )]),
            settings=self.cfg,
        )

    def test_malformed_cadence_hints_fall_back_to_no_cap_on_motion(self):
        # The coercer in isolation. Reached by a hand-built WatchCondition or a direct call — NOT
        # by an authored goal, which the duration-route preflight refuses first (see
        # CompileWatchSpecFirewallTests), so these fallbacks are quieter than they look.
        table = [
            (None, (None, "motion")),
            ({}, (None, "motion")),
            ({"min_judge_interval": 0}, (None, "motion")),       # non-positive is dropped
            ({"min_judge_interval": -2.0}, (None, "motion")),
            ({"min_judge_interval": "2"}, (None, "motion")),     # a numeric STRING is not numeric
            ({"min_judge_interval": float("nan")}, (None, "motion")),
            ({"min_judge_interval": float("inf")}, (float("inf"), "motion")),  # inf is admitted
            ({"min_judge_interval": 2, "gate_on": "bogus"}, (2.0, "motion")),
            ({"min_judge_interval": 2, "gate_on": "fixed"}, (2.0, "fixed")),
            ({"gate_on": "fixed"}, (None, "motion")),            # fixed with no period is not fixed
            ([1, 2], (None, "motion")),                          # not a mapping at all
            ("2s", (None, "motion")),
            ({"min_judge_interval": True}, (1.0, "motion")),     # bool is an int -> one second
        ]
        for raw, expected in table:
            with self.subTest(raw=repr(raw)):
                self.assertEqual(_coerce_cadence(raw), expected)

    def test_an_interval_that_has_exactly_elapsed_admits(self):
        gate = self._gate(2.0)
        gate.evaluate(still(), ts=0.0)
        self.assertTrue(gate.evaluate(motion(), ts=1.0).escalate)
        self.assertFalse(gate.evaluate(still(), ts=2.9).escalate)  # 1.9 into a 2.0 s window
        self.assertTrue(gate.evaluate(motion(), ts=3.0).escalate)  # exactly 2.0 -> admits, "<"

    def test_the_first_frame_admits_at_any_timestamp(self):
        # The brake is seeded at -inf, so there is no warm-up: a loop joined mid-stream spends on
        # its first qualifying frame, whatever the wall time says.
        gate = self._gate(2.0)
        gate.evaluate(still(), ts=1.0e9)
        self.assertTrue(gate.evaluate(motion(), ts=1.0e9 + 0.001).escalate)

    def test_a_fixed_goal_spends_twice_inside_its_first_cadence_window(self):
        gate = self._gate(2.0, gate_on="fixed")
        fired = [t for t in (0.0, 0.5, 1.0, 1.5, 2.0) if gate.evaluate(still(), ts=t).escalate]
        self.assertEqual(fired, [0.0, 2.0])  # two calls bought in the closed interval [0, 2]

    def test_a_throttled_goal_is_only_diagnosable_through_the_opt_in_outcome(self):
        gate = self._gate(2.0)
        gate.evaluate(still(), ts=0.0)
        self.assertTrue(gate.evaluate(motion(), ts=1.0).escalate)
        plain = gate.evaluate(still(), ts=1.5)
        self.assertFalse(plain.escalate)
        self.assertEqual(plain.goal_ids, ())    # the default decision cannot say WHY it stayed shut
        self.assertEqual(plain.outcomes, ())
        typed = gate.evaluate(motion(), ts=1.6, include_outcomes=True)
        self.assertEqual(
            [o.non_admission_reason for o in typed.outcomes], ["cadence_limited"]
        )

    def test_a_clock_that_steps_backwards_does_not_mute_the_goal_forever(self):
        gate = self._gate(2.0)
        gate.evaluate(still(), ts=0.0)
        self.assertTrue(gate.evaluate(motion(), ts=100.0).escalate)
        # the source reconnects on a restarted timebase (or the host clock is corrected backwards)
        frames = [still(), motion()] * 6
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fired = [gate.evaluate(f, ts=float(i)).escalate for i, f in enumerate(frames)]
        self.assertTrue(any(fired))
        # ...and the reset is TRACEABLE, not silent: one warning for the one discontinuity, and a
        # counter a caller can read afterwards. A goal that spends again inside what looks like its
        # cadence window must be attributable to this moment.
        self.assertEqual(
            [type(w.message) for w in caught], [StreamTimeRewindWarning]
        )
        self.assertEqual(gate.time_discontinuities, 1)

    def test_a_declared_reset_is_not_reported_as_a_rewind(self):
        # A new replay epoch / new source restarts the timeline deliberately; the caller already
        # said so by calling reset(), so it must not also be warned about.
        gate = self._gate(2.0)
        gate.evaluate(still(), ts=100.0)
        gate.reset()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gate.evaluate(motion(), ts=0.0)
        self.assertEqual(caught, [])
        self.assertEqual(gate.time_discontinuities, 0)

    def test_a_not_a_number_timestamp_does_not_buy_an_admission_the_cadence_forbids(self):
        # Rewritten from an xfail pin that asserted `escalate is False`. The fix is stronger than
        # the pin asked for: a NaN is refused at the boundary rather than quietly scored, because
        # there is no admission decision to make on one. Both halves of the original defect are
        # closed — the cadence is not bypassed, AND _last_escalate is never poisoned with a NaN.
        gate = self._gate(2.0)
        gate.evaluate(still(), ts=0.0)
        self.assertTrue(gate.evaluate(motion(), ts=1.0).escalate)
        for bad in (float("nan"), float("inf"), float("-inf"), None, "1.0", True):
            with self.subTest(ts=repr(bad)):
                with self.assertRaises(ValueError):
                    gate.evaluate(still(), ts=bad)
        # the brake is intact afterwards: 1.5 is still inside the 2.0 s window, 3.0 is not
        self.assertFalse(gate.evaluate(still(), ts=1.5).escalate)
        self.assertTrue(gate.evaluate(motion(), ts=3.0).escalate)

    def test_an_external_wake_is_guarded_by_the_same_timeline_as_the_visual_path(self):
        # admit_external carried the identical unguarded expression, so an audio wake could be
        # muted (or could over-spend) by a route the visual path is protected against.
        gate = self._gate(2.0)
        self.assertEqual(gate.admit_external(["g"], ts=0.0), ("g",))
        self.assertEqual(gate.admit_external(["g"], ts=1.0), ())  # inside the cadence window
        with self.assertRaises(ValueError):
            gate.admit_external(["g"], ts=float("nan"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(gate.admit_external(["g"], ts=0.5), ("g",))  # rewind -> brake cleared
        self.assertEqual([type(w.message) for w in caught], [StreamTimeRewindWarning])

    def test_two_readings_at_one_instant_are_a_duplicate_not_a_rewind(self):
        # evaluate() and admit_external() share one timeline, and a visual frame and an external
        # wake legitimately carry the SAME ts. That must not read as a discontinuity and drop the
        # cadence brake for every goal.
        gate = self._gate(2.0)
        gate.evaluate(still(), ts=9.0)
        self.assertTrue(gate.evaluate(motion(), ts=10.0).escalate)
        self.assertEqual(gate.admit_external(["g"], ts=10.0), ())  # same instant, still throttled
        self.assertEqual(gate.time_discontinuities, 0)


class WakeFusionAndAdapterBoundaryTests(unittest.TestCase):
    """What a wake policy does to gate_on, what one energy reading is worth, and non-numeric channels."""

    def setUp(self):
        self.cfg = load_settings()

    def _spec(self, *, threshold=0.10, gate_on="motion", goal_ids=("g",)) -> WatchSpec:
        return WatchSpec(conditions=[
            WatchCondition(goal_id=gid, anchor_text="x", threshold=threshold, gate_on=gate_on)
            for gid in goal_ids
        ])

    def test_a_semantic_goal_gates_on_its_own_channel_when_no_policy_is_present(self):
        gate = MotionGate(
            self._spec(gate_on="semantic"), settings=self.cfg, semantic_adapter=lambda f: 0.0
        )
        gate.evaluate(still(), ts=0.0)
        self.assertFalse(gate.evaluate(gray_frame(W, H, 255), ts=1.0).escalate)

    def test_a_semantic_goal_keeps_its_channel_when_a_wake_policy_is_present(self):
        # Was a strict xfail (BUG-A-01-04). The `elif fused is not None` branch sat ABOVE the
        # structural and semantic branches, so ANY non-None wake_policy replaced both for every goal
        # in the spec — a semantic goal silently started gating on a noisy-OR over motion.
        gate = MotionGate(
            self._spec(gate_on="semantic"), settings=self.cfg,
            wake_policy=WakePolicy(threshold=0.1), semantic_adapter=lambda f: 0.0,
        )
        gate.evaluate(still(), ts=0.0)
        decision = gate.evaluate(gray_frame(W, H, 255), ts=1.0)  # heavy motion, zero semantics
        self.assertEqual(decision.signal.semantic, 0.0)
        self.assertFalse(decision.escalate)

    def test_a_fused_admission_is_reported_as_the_fused_cause(self):
        # REWRITTEN. This used to pin the reporting half of the same override as the contract: the
        # decision named the SEMANTIC channel while the number actually compared was the fused
        # noisy-OR over motion, so the receipt named a channel whose measured value was 0.0. `fused`
        # is now a first-class channel a goal opts in to, and it reports itself.
        gate = MotionGate(
            self._spec(gate_on="fused"), settings=self.cfg,
            wake_policy=WakePolicy(threshold=0.1), semantic_adapter=lambda f: 0.0,
        )
        gate.evaluate(still(), ts=0.0)
        decision = gate.evaluate(gray_frame(W, H, 255), ts=1.0, include_outcomes=True)
        self.assertTrue(decision.escalate)
        self.assertEqual(decision.admission_causes, ("fused",))
        self.assertEqual(decision.signal.semantic, 0.0)
        # ...and the receipt names the fusion as the producer, not a tier-0 channel that isn't one
        admitted = [o for o in decision.outcomes if o.admitted]
        self.assertEqual([o.cause for o in admitted], ["fused"])
        self.assertEqual([o.detector_id for o in admitted], ["wake_policy.noisy_or"])

    def test_a_fused_goal_with_no_wake_policy_scores_zero_rather_than_falling_back(self):
        # A goal that asked to be fused has nothing to be fused from, so it admits nothing — the same
        # answer gate_on="fixed" gives with no cadence, and never a quiet fallback to motion.
        gate = MotionGate(self._spec(threshold=0.1, gate_on="fused"), settings=self.cfg)
        gate.evaluate(still(), ts=0.0)
        decision = gate.evaluate(gray_frame(W, H, 255), ts=1.0, include_outcomes=True)
        self.assertFalse(decision.escalate)
        self.assertEqual(
            [o.non_admission_reason for o in decision.outcomes], ["signal_below_threshold"]
        )

    def test_a_fixed_cadence_goal_is_unaffected_by_a_wake_policy(self):
        spec = WatchSpec(conditions=[WatchCondition(
            goal_id="g", anchor_text="x", threshold=0.99, cadence_s=2.0, gate_on="fixed",
        )])
        gate = MotionGate(spec, settings=self.cfg, wake_policy=WakePolicy(threshold=0.1))
        self.assertTrue(gate.evaluate(still(), ts=0.0).escalate)  # fixed wins on a still frame

    def test_one_energy_reading_populates_exactly_one_channel(self):
        # REWRITTEN (BUG-A-01-05). evaluate() used to map `energy` into BOTH "audio" and
        # "audio_event", so a policy that weighted them separately fused one microphone reading with
        # itself — 0.51 against 0.30 for the same reading on one channel. The noisy-OR assumes its
        # channels are INDEPENDENT cues, so the duplicate was worth more than the measurement.
        one = WakePolicy(weights={"audio": 0.5})
        both = WakePolicy(weights={"audio": 0.5, "audio_event": 0.5})
        spec = self._spec(threshold=0.51, gate_on="fused")
        single = MotionGate(spec, settings=self.cfg, wake_policy=one)
        doubled = MotionGate(spec, settings=self.cfg, wake_policy=both)
        self.assertFalse(single.evaluate(still(), ts=0.0, energy=0.6).escalate)   # 0.30 fused
        # weighting the retired alias buys nothing: the gate never populates it, and an absent
        # channel casts no vote
        self.assertFalse(doubled.evaluate(still(), ts=0.0, energy=0.6).escalate)  # still 0.30
        self.assertEqual(
            cheap_signals(motion=0.0, structural=0.0, semantic=0.0, energy=0.6),
            {"motion": 0.0, "structural": 0.0, "semantic": 0.0, "audio": 0.6},
        )
        self.assertAlmostEqual(wake_score({"audio": 0.6}, one), 0.30, places=12)
        self.assertAlmostEqual(wake_score({"audio": 0.6}, both), 0.30, places=12)

    def test_the_gate_omits_the_audio_channel_when_no_microphone_is_open(self):
        # "no microphone is open" and "the room is silent" are different facts; only the second is
        # evidence. The channel used to be present as a hard 0.0 either way.
        self.assertNotIn(
            "audio", cheap_signals(motion=0.1, structural=0.0, semantic=0.0, energy=None)
        )
        self.assertEqual(
            cheap_signals(motion=0.1, structural=0.0, semantic=0.0, energy=0.0)["audio"], 0.0
        )

    def test_a_caller_overlay_is_applied_last_and_can_override_a_gate_channel(self):
        self.assertEqual(
            cheap_signals(
                motion=0.1, structural=0.0, semantic=0.0, extra={"presence": 0.7, "motion": 0.9}
            ),
            {"motion": 0.9, "structural": 0.0, "semantic": 0.0, "presence": 0.7},
        )

    def test_an_adapter_returning_not_a_number_reads_as_zero_not_as_deafness(self):
        # Rewritten: this used to assert `math.isnan(decision.signal.structural)` and that not even
        # a 0.0 cutoff could admit — the defect encoded as the contract. A NaN is a broken reading,
        # and a broken reading must be worth what "saw nothing" is worth, never strictly LESS than
        # nothing. It used to be less: every comparison against NaN is False, so the goal could not
        # clear a cutoff that admits every real number, permanently, on one bad frame.
        gate = MotionGate(
            self._spec(gate_on="structural"), settings=self.cfg,
            structural_adapter=lambda f: float("nan"),
        )
        decision = gate.evaluate(motion(40), ts=0.0)
        self.assertFalse(decision.escalate)          # 0.0 still does not clear a 0.10 cutoff
        self.assertEqual(decision.signal.structural, 0.0)
        self.assertFalse(math.isnan(decision.signal.structural))  # and no NaN rides the wire up
        # a zero cutoff admits every real number, and 0.0 is a real number
        wide = MotionGate(
            self._spec(threshold=0.0, gate_on="structural"), settings=self.cfg,
            structural_adapter=lambda f: float("nan"),
        )
        self.assertTrue(wide.evaluate(motion(40), ts=0.0).escalate)
        # ...and the goal RECOVERS: a NaN frame does not poison the ones after it.
        readings = iter([float("nan"), 0.9])
        recovering = MotionGate(
            self._spec(gate_on="structural"), settings=self.cfg,
            structural_adapter=lambda f: next(readings),
        )
        self.assertFalse(recovering.evaluate(still(), ts=0.0).escalate)
        self.assertTrue(recovering.evaluate(motion(40), ts=1.0).escalate)

    def test_the_python_and_javascript_gates_agree_on_a_broken_channel(self):
        # packages/percept-edge/src/skills.js `clamp01` has always been Number.isFinite-guarded, so
        # before the shared helper the two runtimes forked on exactly this input: JS scored a NaN
        # channel 0, Python passed it through into a comparison false against every cutoff. One
        # admission rule, one implementation.
        for reading, expected in (
            (float("nan"), 0.0), (float("inf"), 0.0), (float("-inf"), 0.0),
            (-0.5, 0.0), (1.5, 1.0), (0.25, 0.25),
        ):
            with self.subTest(reading=reading):
                gate = MotionGate(
                    self._spec(gate_on="semantic"), settings=self.cfg,
                    semantic_adapter=lambda f, v=reading: v,
                )
                self.assertEqual(gate.evaluate(still(), ts=0.0).signal.semantic, expected)

    def test_an_adapter_returning_none_aborts_the_frame_for_every_goal(self):
        spec = self._spec(goal_ids=("semantic_goal", "motion_goal"))
        healthy = MotionGate(spec, settings=self.cfg)
        healthy.evaluate(still(), ts=0.0)
        self.assertEqual(
            healthy.evaluate(motion(40), ts=1.0).goal_ids, ("semantic_goal", "motion_goal")
        )
        # One adapter that answers None takes the WHOLE frame down — before any condition is
        # evaluated, so the unrelated motion goal never gets its look either, on this frame or any
        # frame after it.
        broken = MotionGate(spec, settings=self.cfg, semantic_adapter=lambda f: None)
        with self.assertRaises(TypeError):
            broken.evaluate(still(), ts=0.0)
        with self.assertRaises(TypeError):
            broken.evaluate(motion(40), ts=1.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
