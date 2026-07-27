"""The EXTERNAL_RHYTHM route, driven end to end (L1) — ``type="rhythm"`` reaches ``percept.run()``.

Before this, five of the eight authored behavior routes had no driver: ``worker_can_drive`` admitted
only the three FRAME routes, so a ``type="rhythm"`` goal was refused at ``percept.run`` /
``GatedWatchLoop`` construction and ``RhythmTracker`` appeared nowhere in the runtime. The counting
primitive existed and was exercised only through ``core.perceive_rhythm`` called by hand.

What is proven here, offline, with a scripted channel signal and no provider call:

  * a rhythm goal ARMS through ``apply_watch_plan`` and RUNS through both frame drivers;
  * the eye is asked to MEASURE once per frame (not once per admission, not once per judgment) and
    the frame it receives is the real frame off the source;
  * the count comes from the SIGNAL — it is not capped by judgment density, and it spends no judge;
  * what an INCIDENT means for a rhythm goal: a count MILESTONE (an authored ``fire_when``), never
    one per rep. The un-authored default is silent-with-a-count.
"""

from __future__ import annotations

import asyncio
import math
import unittest

from percept import Goal, Percept
from percept.behavior_routes import (
    BehaviorRoute,
    behavior_route,
    drives_rhythm,
    goal_behavior_route,
    needs_watch_condition,
    worker_can_drive,
)
from percept.contracts.watchplan import WatchPlan
from percept.harness.detectors import DetectorManifest, register
from percept.harness.tier0 import Frame, GatedWatchLoop, gray_frame
from percept.rhythm import RhythmDrive, rhythm_emits_incidents
from percept.runtime import run
from percept.watchplan_apply import apply_watch_plan

W, H = 8, 8
PERIOD = 12  # samples per rep cycle
FPS = 6.0

_INSTANCES: list["_ScriptedChannel"] = []


class _ScriptedChannel:
    """Test-only MEASURING eye: one clean sinusoid per ``PERIOD`` frames on ``limb-vertical``.

    Stands in for ``pose-channels`` (whose real provider needs a pose model). It records every frame
    it was handed, so the test can prove the drive calls it once per frame with the real frame.
    """

    signal = "frame"
    name = "test-scripted-channel"

    def __init__(self) -> None:
        self.i = 0
        self.seen: list = []
        _INSTANCES.append(self)

    def reset(self) -> None:
        self.i = 0

    def measure(self, frame) -> dict[str, float]:
        self.seen.append(frame)
        v = math.sin(2 * math.pi * self.i / PERIOD)
        self.i += 1
        return {"limb-vertical": v}

    def push(self, t, frame):  # a measuring eye proposes no boundaries
        return []


class _BoundaryOnly:
    """A detector that proposes occurrences but MEASURES nothing — the wrong primitive for rhythm."""

    signal = "motion"
    name = "test-boundary-only"

    def reset(self) -> None:
        pass

    def push(self, t, v):
        return []


def _register_test_detectors() -> None:
    """(Re)register the fixtures. Called per-test: the detector registry is a process-global that
    other suites legitimately ``clear_detectors()``, and an unregistered name silently falls back to
    the real pose provider instead of failing here."""
    register(
        "test-scripted-channel",
        _ScriptedChannel,
        DetectorManifest(
            "test-scripted-channel", "a scripted oscillating channel", "rhythm route wiring test",
            "per-frame channel measurements",
        ),
    )
    register(
        "test-boundary-only",
        _BoundaryOnly,
        DetectorManifest(
            "test-boundary-only", "nothing", "negative test: a non-measuring detector",
            "no boundaries",
        ),
    )


class _RhythmFixture(unittest.TestCase):
    def setUp(self) -> None:
        _register_test_detectors()
        _INSTANCES.clear()


def _still() -> Frame:
    return gray_frame(W, H, 0)


def _rhythm_goal(goal_id: str = "reps", **state) -> Goal:
    return Goal(
        id=goal_id,
        condition="a completed push-up repetition",
        say="rep",
        state={
            "type": "rhythm",
            "count": 0,
            "detectors": ["test-scripted-channel"],
            **state,
        },
    )


def _core(goal: Goal) -> Percept:
    core = Percept.create(mode="offline", discover_plugins=False)
    core.add_goal(goal)
    return core


def _source(n_cycles: int = 8):
    """(t, gate_frame, judge_frame) — a still gate frame (nothing to admit) and a tagged judge frame."""
    return [(i / FPS, _still(), f"f{i}") for i in range(n_cycles * PERIOD)]


async def _collect(core, source, **kw):
    return [incident async for incident in run(core, source=source, **kw)]


class RhythmRouteContractTests(_RhythmFixture):
    """The route table itself: drivable, but NOT a tier-0 watch condition."""

    def test_rhythm_is_worker_drivable_but_never_compiles_to_a_watch_condition(self):
        route = behavior_route({"type": "rhythm"})
        self.assertIs(route, BehaviorRoute.EXTERNAL_RHYTHM)
        self.assertTrue(worker_can_drive(route))  # the worker HAS a driver for it now
        self.assertTrue(drives_rhythm(route))
        # ...but it is measured, not admitted: no WatchCondition, so no judge is ever spent on it.
        self.assertFalse(needs_watch_condition(route))

    def test_the_other_external_routes_are_still_undrivable(self):
        for behavior_type in ("event", "count", "remind", "scheduler"):
            with self.subTest(type=behavior_type):
                self.assertFalse(worker_can_drive(behavior_route({"type": behavior_type})))

    def test_a_rhythm_plan_arms_through_apply_watch_plan(self):
        core = Percept.create(mode="offline", discover_plugins=False)
        receipt = apply_watch_plan(
            core,
            WatchPlan(
                id="reps",
                condition="a completed push-up repetition",
                explicit={"Goal.state.type": "rhythm", "Goal.state.rhythm.channel": "limb-vertical"},
            ),
        )
        self.assertEqual(receipt.status, "applied")
        armed = core.goals.get("reps")
        self.assertIs(goal_behavior_route(armed), BehaviorRoute.EXTERNAL_RHYTHM)
        self.assertEqual(armed.state["rhythm"]["channel"], "limb-vertical")


class RhythmIncidentPolicyTests(_RhythmFixture):
    """What an Incident MEANS for a rhythm goal: a count milestone, never a rep."""

    def test_only_an_authored_milestone_may_mint_incidents(self):
        self.assertFalse(rhythm_emits_incidents({"type": "rhythm"}))
        self.assertFalse(rhythm_emits_incidents({"type": "rhythm", "fire_when": "  "}))
        self.assertFalse(rhythm_emits_incidents(None))
        self.assertTrue(rhythm_emits_incidents({"type": "rhythm", "fire_when": "n == 20"}))

    def test_a_non_measuring_detector_is_a_loud_shape_error_not_a_silent_no_op(self):
        goal = _rhythm_goal(detectors=["test-boundary-only"])
        with self.assertRaises(ValueError) as caught:
            RhythmDrive(Percept.create(mode="offline", discover_plugins=False), [goal])
        self.assertIn("measure(frame)", str(caught.exception))


class RhythmThroughPerceptRunTests(_RhythmFixture):
    """percept.run(): a rhythm goal arms, runs, and counts — with no judge and no incident per rep."""

    def test_a_rhythm_goal_counts_through_run_and_stays_silent_by_default(self):
        core = _core(_rhythm_goal())
        source = _source(n_cycles=8)
        incidents = asyncio.run(_collect(core, source))

        self.assertAlmostEqual(core.rhythm["reps"].count, 8, delta=1)  # counted from the SIGNAL
        self.assertEqual(core.goals.get("reps").state["count"], core.rhythm["reps"].count)
        # The pathology this route exists to break: a rep is not an Incident.
        self.assertEqual(incidents, [])

    def test_the_eye_measures_every_frame_and_receives_the_real_frame(self):
        core = _core(_rhythm_goal())
        source = _source(n_cycles=4)
        asyncio.run(_collect(core, source))

        self.assertEqual(len(_INSTANCES), 1)  # exactly one eye, built once for the goal
        eye = _INSTANCES[0]
        self.assertEqual(len(eye.seen), len(source))  # once per FRAME, not once per admission
        self.assertEqual(eye.seen, [judge_frame for _, _, judge_frame in source])

    def test_the_count_is_not_capped_by_judgment_density(self):
        # The whole point: the gate never escalated (a still frame, and a rhythm goal compiles to no
        # admission condition at all), so ZERO judgments were available -- yet the count is real.
        core = _core(_rhythm_goal())
        asyncio.run(_collect(core, _source(n_cycles=8)))
        self.assertEqual(core.metrics.snapshot().get("judge.calls", 0), 0)
        self.assertGreaterEqual(core.rhythm["reps"].count, 7)

    def test_an_authored_milestone_mints_exactly_one_incident(self):
        core = _core(_rhythm_goal(fire_when="n == 5"))
        incidents = asyncio.run(_collect(core, _source(n_cycles=8)))

        self.assertEqual(len(incidents), 1)  # ONE incident at the milestone, not one per rep
        self.assertEqual(incidents[0].concern_id, "reps")
        self.assertGreaterEqual(core.rhythm["reps"].count, 7)  # ...and counting continued past it

    def test_a_rhythm_goal_coexists_with_a_judged_watch_on_the_same_source(self):
        from percept.adapters import Verdict
        from percept.backends.fake import FakeVision

        vision = FakeVision(
            script={
                "hot": Verdict(satisfied=True, confidence=0.95, evidence="a person is present"),
                "cold": Verdict(satisfied=False, confidence=0.95, evidence="nobody"),
            }
        )
        core = Percept.create(mode="offline", vision=vision, discover_plugins=False)
        core.add_goal(Goal(id="who", condition="a person is present", say="someone is here"))
        core.add_goal(_rhythm_goal(fire_when="n == 3"))

        moved = gray_frame(W, H, 255)
        source = [
            (i / FPS, _still() if i % 2 == 0 else moved, "hot" if i % 2 else "cold")
            for i in range(8 * PERIOD)
        ]
        incidents = asyncio.run(_collect(core, source))

        self.assertGreaterEqual(core.rhythm["reps"].count, 7)  # the rhythm goal counted
        by_goal = {incident.concern_id for incident in incidents}
        self.assertIn("who", by_goal)  # the judged Watch still fired
        self.assertEqual(sum(1 for i in incidents if i.concern_id == "reps"), 1)  # one milestone


class RhythmThroughGatedWatchLoopTests(_RhythmFixture):
    """The same drive on the harness loop (the cost-demo / bring-your-own-capture seam)."""

    def test_gated_loop_counts_reps_and_reports_the_count(self):
        goal = _rhythm_goal()
        core = _core(goal)
        loop = GatedWatchLoop(core, [goal], threshold=0.05)
        report = asyncio.run(loop.run([_still() for _ in range(8 * PERIOD)], dt=1 / FPS))

        self.assertEqual(list(loop.spec.conditions), [])  # measured, never admitted
        self.assertEqual(report.vision_calls, 0)  # no judge spent
        self.assertAlmostEqual(report.rhythm_counts["reps"], 8, delta=1)
        self.assertEqual(report.fire_events, [])  # no milestone authored -> silence

    def test_gated_loop_emits_the_authored_milestone_once(self):
        goal = _rhythm_goal(fire_when="n == 4")
        core = _core(goal)
        loop = GatedWatchLoop(core, [goal], threshold=0.05)
        report = asyncio.run(loop.run([_still() for _ in range(8 * PERIOD)], dt=1 / FPS))

        self.assertEqual(len(report.fire_events), 1)
        self.assertEqual(report.fire_events[0].goal_id, "reps")


class ConductorToCountTests(_RhythmFixture):
    """The whole chain, offline: plain-language goal -> conductor -> WatchPlan -> arm -> run -> count.

    Part 1 (a drivable route) and Part 2 (something that authors it) are only worth anything joined
    up. Nothing here is hand-armed with a primitive: the ``type="rhythm"`` in the running goal came
    from the conductor deciding it from the goal TEXT.
    """

    def test_a_rep_counting_goal_text_ends_up_counting_reps(self):
        from percept.conductor import PlanConductor

        async def scripted_haiku(system: str, user: str) -> str:
            self.assertIn("RHYTHM", system)  # the real prompt, not a stub
            self.assertIn("push-ups", user)
            return (
                '{"behavior": {"type": "rhythm", "accumulate": {"fire_when": "n == 5"}}, '
                '"rhythm": {"channel": "limb-vertical"}}'
            )

        goal_text = "count my push-ups and tell me when I hit 5"
        primitive = asyncio.run(PlanConductor(call=scripted_haiku).author(goal_text))
        self.assertEqual(primitive.behavior_type, "rhythm")
        self.assertEqual(primitive.explicit["Goal.state.fire_when"], "n == 5")

        core = Percept.create(mode="offline", discover_plugins=False)
        # The eye is a deployment choice, not an authoring one: the conductor picks the PRIMITIVE,
        # the plan/profile picks which detector measures for it.
        plan = primitive.to_watch_plan(id="reps", detectors=("test-scripted-channel",))
        self.assertEqual(apply_watch_plan(core, plan).status, "applied")
        self.assertIs(goal_behavior_route(core.goals.get("reps")), BehaviorRoute.EXTERNAL_RHYTHM)

        incidents = asyncio.run(_collect(core, _source(n_cycles=8)))
        self.assertGreaterEqual(core.rhythm["reps"].count, 7)  # counted from the signal
        self.assertEqual(len(incidents), 1)  # exactly the authored milestone
        self.assertEqual(incidents[0].concern_id, "reps")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
