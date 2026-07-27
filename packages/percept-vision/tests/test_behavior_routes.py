"""Executable behavior-route contract through structured apply and the real runtime worker."""

from __future__ import annotations

import asyncio
import unittest
import warnings
from dataclasses import replace

from percept import Goal, GoalOp, Percept
from percept.adapters import Box, Observation, Verdict
from percept.behavior import coerce_authored_behavior
from percept.behavior_routes import (
    BehaviorRoute,
    UndrivenBehaviorRouteWarning,
    UnsupportedBehaviorRouteError,
    behavior_route,
    goal_behavior_route,
    worker_can_drive,
)
from percept.config import GateSettings, Settings
from percept.contracts.watchplan import WatchPlan
from percept.domain_pack import watch_template
from percept.edge.tier0 import (
    CivilTime,
    GatedWatchLoop,
    compile_watchspec,
    gray_frame,
    run_gated_watch,
    watchspec_unsupported,
)
from percept.incidents import make_incident
from percept.observability import funnel_key
from percept.packs_default import FACILITY_OBSERVATION_V1
from percept.runtime import RuntimeStats, run
from percept.sourceprofile import SourceProfile
from percept.watchplan_apply import apply_watch_plan


STILL = gray_frame(8, 8, 0)
MOVED = gray_frame(8, 8, 255)


async def _collect(core: Percept, source, **kwargs):
    return [incident async for incident in run(core, source=source, **kwargs)]


class _RouteVision:
    """Condition-aware transition judge with observable provider-call accounting."""

    def __init__(self, script: dict[tuple[str, str], Verdict]) -> None:
        self.script = dict(script)
        self.judge_calls: list[tuple[str, str]] = []
        self.detect_calls = 0

    async def judge(self, condition, frame, **_kwargs):
        token = str(frame)
        self.judge_calls.append((condition, token))
        return self.script.get(
            (condition, token),
            Verdict(satisfied=False, confidence=0.95, evidence="not observed"),
        )

    async def detect(self, frame, goals):
        self.detect_calls += 1
        return []

    async def detect_events(self, frames, goals):
        return []


class _EntityRouteVision(_RouteVision):
    def __init__(
        self,
        script: dict[tuple[str, str], Verdict],
        detections: dict[str, list[Observation]],
    ) -> None:
        super().__init__(script)
        self.detections = detections

    async def detect(self, frame, goals):
        self.detect_calls += 1
        return list(self.detections.get(str(frame), ()))


def _transition_plan_core(
    plan: WatchPlan,
    tokens: list[str],
    *,
    settings: Settings | None = None,
):
    before = plan.explicit["Goal.state.from"]
    after = plan.explicit["Goal.state.to"]
    script: dict[tuple[str, str], Verdict] = {}
    for token in tokens:
        script[(before, token)] = Verdict(
            satisfied=token == "from", confidence=0.95, evidence=f"{before}: {token}"
        )
        script[(after, token)] = Verdict(
            satisfied=token == "to", confidence=0.95, evidence=f"{after}: {token}"
        )
    vision = _RouteVision(script)
    core = Percept.create(vision=vision, settings=settings, discover_plugins=False)
    receipt = apply_watch_plan(core, plan, domain_pack=FACILITY_OBSERVATION_V1)
    if receipt.status not in ("applied", "applied_with_warnings"):
        raise AssertionError(receipt)
    return core, vision, receipt.watch_ids[0]


def _transition_core(
    template_name: str,
    tokens: list[str],
    *,
    settings: Settings | None = None,
):
    plan = watch_template(f"facility_observation@1/{template_name}")
    return _transition_plan_core(plan, tokens, settings=settings)


def _motion_source(tokens: list[str]):
    frames = (STILL, MOVED)
    return [(float(index), frames[index % 2], token) for index, token in enumerate(tokens)]


class RouteClassifierTests(unittest.TestCase):
    class _UnhashableStr(str):
        __hash__ = None

    class _HostileStr(str):
        __hash__ = None

        def __eq__(self, _other):
            raise AssertionError("hostile equality must not run")

        def __repr__(self):
            raise AssertionError("hostile repr must not run")

        def strip(self, _chars=None):
            raise AssertionError("hostile strip must not run")

    def test_closed_route_table_and_condition_eligibility_share_one_classifier(self):
        goals = [
            Goal(
                "watch", "watch", "watch",
                state={"type": "watch", "detectors": ["motion-periodicity"]},
            ),
            Goal(
                "transition",
                "transition",
                "transition",
                state={
                    "type": "transition",
                    "from": "a",
                    "to": "b",
                    "detectors": ["motion-periodicity"],
                },
            ),
            Goal(
                "process", "process", "process",
                state={"type": "process", "detectors": ["motion-periodicity"]},
            ),
            Goal(
                "event", "event", "event",
                state={"type": "event", "detectors": ["motion-periodicity"]},
            ),
            Goal(
                "count", "count", "count",
                state={"type": "count", "detectors": ["motion-periodicity"]},
            ),
            Goal(
                "rhythm", "rhythm", "rhythm",
                state={"type": "rhythm", "detectors": ["motion-periodicity"]},
            ),
            Goal(
                "remind", "remind", "remind",
                state={"type": "remind", "detectors": ["motion-periodicity"]},
            ),
            Goal(
                "unknown", "unknown", "unknown",
                state={"type": "made_up", "detectors": ["motion-periodicity"]},
            ),
        ]
        self.assertEqual(
            [goal_behavior_route(goal) for goal in goals],
            [
                BehaviorRoute.FRAME_WATCH,
                BehaviorRoute.FRAME_TRANSITION,
                BehaviorRoute.FRAME_PROCESS,
                BehaviorRoute.EXTERNAL_EVENT_WINDOW,
                BehaviorRoute.EXTERNAL_BOUNDARY,
                BehaviorRoute.EXTERNAL_RHYTHM,
                BehaviorRoute.EXTERNAL_SCHEDULER,
                BehaviorRoute.UNSUPPORTED,
            ],
        )
        self.assertEqual(
            [condition.goal_id for condition in compile_watchspec(goals, threshold=0.1).conditions],
            ["watch", "transition", "process"],
        )
        self.assertEqual(
            [goal_id for goal_id, _dropped in watchspec_unsupported(goals, set())],
            ["watch", "transition", "process"],
        )

    def test_raw_and_coerced_explicit_watch_and_process_keep_the_same_route(self):
        for route_type, expected in (
            ("watch", BehaviorRoute.FRAME_WATCH),
            ("process", BehaviorRoute.FRAME_PROCESS),
        ):
            raw = {"type": "watch", "behavior": {"type": route_type}}
            coerced = {"type": "watch", **coerce_authored_behavior({"type": route_type})}
            self.assertEqual(behavior_route(raw), expected)
            self.assertEqual(behavior_route(coerced), expected)
            self.assertEqual(coerced["type"], route_type)

    def test_unknown_and_blank_transition_stay_unsupported_and_name_authored_type(self):
        self.assertEqual(behavior_route({"type": ["not", "hashable"]}), BehaviorRoute.UNSUPPORTED)
        blank = {"type": "watch", "behavior": {"type": "transition", "from": " ", "to": "ready"}}
        self.assertEqual(behavior_route(blank), BehaviorRoute.UNSUPPORTED)
        self.assertNotIn("type", coerce_authored_behavior(blank["behavior"]))

    def test_flat_and_nested_route_representations_have_closed_conflict_rules(self):
        self.assertEqual(
            behavior_route(
                {
                    "type": "watch",
                    "behavior": {"type": "transition", "from": "closed", "to": "open"},
                }
            ),
            BehaviorRoute.FRAME_TRANSITION,
        )
        self.assertEqual(
            behavior_route({"type": "process", "behavior": {"type": "process"}}),
            BehaviorRoute.FRAME_PROCESS,
        )
        self.assertEqual(
            behavior_route({"type": "watch", "behavior": {"type": "count"}}),
            BehaviorRoute.EXTERNAL_BOUNDARY,
        )
        for flat, nested in (("process", "transition"), ("event", "count"), ("count", "rhythm")):
            with self.subTest(flat=flat, nested=nested):
                self.assertEqual(
                    behavior_route(
                        {
                            "type": flat,
                            "behavior": {"type": nested, "from": "a", "to": "b"},
                        }
                    ),
                    BehaviorRoute.UNSUPPORTED,
                )

    def test_hostile_string_subclasses_fail_closed_without_invoking_user_methods(self):
        for value in (self._UnhashableStr("count"), self._HostileStr("transition")):
            with self.subTest(type=type(value).__name__):
                state = {"type": value}
                self.assertEqual(behavior_route(state), BehaviorRoute.UNSUPPORTED)
                goal = Goal("hostile", "hostile", "hostile", state=state)
                with self.assertRaises(UnsupportedBehaviorRouteError) as caught:
                    GatedWatchLoop(
                        Percept.create(vision=_RouteVision({}), discover_plugins=False),
                        goals=[goal],
                    )
                self.assertIn(f"<{type(value).__name__}>", str(caught.exception))

        hostile_predicate = {
            "type": "transition",
            "from": self._HostileStr("a"),
            "to": "b",
        }
        self.assertEqual(behavior_route(hostile_predicate), BehaviorRoute.UNSUPPORTED)
        self.assertNotIn("type", coerce_authored_behavior(hostile_predicate))

    def test_raw_nested_transition_is_prepare_only_and_every_runtime_seam_fails_closed(self):
        raw_state = {
            "type": "watch",
            "behavior": {"type": "transition", "from": "closed", "to": "open"},
        }
        raw_goal = Goal("raw", "door changes", "changed", state=raw_state)

        # WatchPlan preparation must still be able to validate the raw compiler shape.
        self.assertEqual(behavior_route(raw_state), BehaviorRoute.FRAME_TRANSITION)
        # Runtime consumers accept only the canonical flat state produced by apply-time coercion.
        self.assertEqual(goal_behavior_route(raw_goal), BehaviorRoute.UNSUPPORTED)
        self.assertEqual(
            compile_watchspec([raw_goal], threshold=0.1).conditions,
            [],
        )

        gated_vision = _RouteVision({})
        gated_core = Percept.create(vision=gated_vision, discover_plugins=False)
        with self.assertRaises(UnsupportedBehaviorRouteError):
            GatedWatchLoop(gated_core, goals=[raw_goal])
        self.assertEqual(gated_vision.detect_calls, 0)
        self.assertEqual(gated_vision.judge_calls, [])

        # The ARM is now the earliest seam: this shape used to go ``active`` and only fail once a
        # worker or a direct judge call reached it (and never at all if neither did). It now fails at
        # add_goal, so the later seams below are unreachable via arming rather than merely defended.
        worker_vision = _RouteVision({})
        worker_core = Percept.create(vision=worker_vision, discover_plugins=False)
        with self.assertRaises(UnsupportedBehaviorRouteError):
            worker_core.add_goal(raw_goal)
        self.assertEqual(list(worker_core.goals.active()), [])
        self.assertEqual(worker_core.goals.revision, 0)
        source = WorkerPreflightTests.PullSpy()
        self.assertEqual(asyncio.run(_collect(worker_core, source)), [])
        self.assertEqual(worker_vision.detect_calls, 0)
        self.assertEqual(worker_vision.judge_calls, [])

        direct_vision = _RouteVision({})
        direct_core = Percept.create(vision=direct_vision, discover_plugins=False)
        with self.assertRaises(UnsupportedBehaviorRouteError):
            direct_core.apply_goal_ops([GoalOp(op="create", goal=raw_goal)])
        self.assertEqual(list(direct_core.goals.active()), [])
        self.assertEqual(
            asyncio.run(direct_core.perceive_judged("frame", source="camera", t=1.0)), []
        )
        # No judge is ever spent on it: the goal never armed, so there is nothing to judge.
        self.assertEqual(direct_vision.judge_calls, [])

    def test_direct_perception_still_skips_external_routes_for_dedicated_apis(self):
        vision = _RouteVision({})
        core = Percept.create(vision=vision, discover_plugins=False)
        core.add_goal(Goal("count", "count", "count", state={"type": "count"}))
        self.assertEqual(
            asyncio.run(core.perceive_judged("frame", source="camera", t=1.0)),
            [],
        )
        self.assertEqual(vision.judge_calls, [])


class StructuredApplyRouteTests(unittest.TestCase):
    def test_explicit_watch_and_process_arm_without_route_coercion(self):
        core = Percept.create(mode="offline", discover_plugins=False)
        for route_type in ("watch", "process"):
            plan = WatchPlan(
                id=route_type,
                condition=f"{route_type} condition",
                explicit={"Goal.state.type": route_type},
            )
            receipt = apply_watch_plan(core, plan)
            self.assertEqual(receipt.status, "applied")
            self.assertEqual(core.goals.get(route_type).state["type"], route_type)

    def test_rhythm_arms_because_the_worker_now_drives_the_counting_primitive(self):
        # ``rhythm`` used to sit with the other external primitives here. It no longer does: the
        # frame loop drives it (percept.rhythm.RhythmDrive -> core.perceive_rhythm), so an authored
        # rhythm plan ARMS. It still compiles to no tier-0 WatchCondition — measured, not judged.
        core = Percept.create(mode="offline", discover_plugins=False)
        receipt = apply_watch_plan(
            core,
            WatchPlan(
                id="rhythm",
                condition="a completed repetition",
                explicit={"Goal.state.type": "rhythm"},
            ),
        )
        self.assertEqual(receipt.status, "applied")
        armed = core.goals.get("rhythm")
        self.assertEqual(armed.state["type"], "rhythm")
        self.assertEqual(
            [c.goal_id for c in compile_watchspec([armed], threshold=0.1).conditions], []
        )

    def test_every_undriven_external_route_is_rejected_with_no_goal_mutation(self):
        for route_type in ("event", "count", "remind", "scheduler"):
            core = Percept.create(mode="offline", discover_plugins=False)
            receipt = apply_watch_plan(
                core,
                WatchPlan(
                    id=f"bad-{route_type}",
                    condition="external primitive",
                    explicit={"Goal.state.type": route_type},
                ),
            )
            self.assertEqual(receipt.status, "rejected")
            self.assertEqual(receipt.dropped_fields[0].code, "unsupported_runtime_route")
            self.assertIn(repr(route_type), receipt.dropped_fields[0].message)
            self.assertEqual(list(core.goals.active()), [])

    def test_unknown_route_is_named_without_false_direct_primitive_guidance(self):
        core = Percept.create(mode="offline", discover_plugins=False)
        receipt = apply_watch_plan(
            core,
            WatchPlan(
                id="unknown",
                condition="unknown",
                explicit={"Goal.state.type": "made_up"},
            ),
        )
        self.assertEqual(receipt.status, "rejected")
        diagnostic = receipt.dropped_fields[0]
        self.assertEqual(diagnostic.code, "unsupported_runtime_route")
        self.assertIn("'made_up'", diagnostic.message)
        self.assertNotIn("matching direct", diagnostic.message)
        self.assertEqual(list(core.goals.active()), [])

    def test_mixed_supported_and_unsupported_batch_is_atomic(self):
        core = Percept.create(mode="offline", discover_plugins=False)
        goals_before = core.goals.snapshot()
        revision_before = core.goals.revision
        runtime_bindings_before = dict(core._watch_runtime_bindings)
        runtime_keys_before = dict(core._watch_runtime_keys)
        judge_bindings_before = dict(core._judge_bindings)
        judge_requesting_before = set(core._judge_requesting_goal_ids)
        ledger_before = dict(getattr(core, "_watchplan_applied_hashes", {}))
        receipt = apply_watch_plan(
            core,
            [
                WatchPlan(id="good", condition="a supported Watch"),
                WatchPlan(
                    id="bad",
                    condition="an external count",
                    explicit={"Goal.state.type": "count"},
                ),
            ],
        )
        self.assertEqual(receipt.status, "rejected")
        self.assertEqual(receipt.dropped_fields[0].code, "unsupported_runtime_route")
        self.assertIsNone(core.goals.get("good"))
        self.assertIsNone(core.goals.get("bad"))
        self.assertEqual(core.goals.snapshot(), goals_before)
        self.assertEqual(core.goals.revision, revision_before)
        self.assertEqual(core._watch_runtime_bindings, runtime_bindings_before)
        self.assertEqual(core._watch_runtime_keys, runtime_keys_before)
        self.assertEqual(core._judge_bindings, judge_bindings_before)
        self.assertEqual(core._judge_requesting_goal_ids, judge_requesting_before)
        self.assertEqual(getattr(core, "_watchplan_applied_hashes", {}), ledger_before)


class WorkerPreflightTests(unittest.TestCase):
    class PullSpy:
        def __init__(self) -> None:
            self.pulls = 0

        def __iter__(self):
            self.pulls += 1
            yield (0.0, STILL, "frame")

    def test_raw_unsupported_route_fails_before_source_pull(self):
        # UNSUPPORTED: no driver anywhere and no host feed, so a run carrying one is incoherent and
        # still refuses before a frame is pulled. (Corrupted PAST the arm gate, which refuses it too.)
        vision = _RouteVision({})
        core = Percept.create(vision=vision, discover_plugins=False)
        core.add_goal(Goal("watch", "watch", "watch"))
        core.goals.get("watch").state["type"] = "wobble"
        source = self.PullSpy()
        with self.assertRaises(UnsupportedBehaviorRouteError) as caught:
            asyncio.run(_collect(core, source))
        self.assertEqual(source.pulls, 0)
        self.assertEqual(caught.exception.unsupported[0].authored_type, "wobble")
        self.assertEqual(vision.judge_calls, [])

    def test_a_host_driven_route_is_skipped_by_the_run_not_fatal_to_it(self):
        # A count goal has a complete, shipped driver — the HOST's. The frame worker simply is not
        # it. Aborting the run for that reason took every unrelated Watch on the source down too.
        vision = _RouteVision({})
        core = Percept.create(vision=vision, discover_plugins=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            core.add_goal(Goal("count", "count", "count", state={"type": "count"}))
        source = self.PullSpy()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(asyncio.run(_collect(core, source)), [])

        self.assertEqual(source.pulls, 1)          # the run started instead of dying
        self.assertEqual(vision.judge_calls, [])   # ...and never spent a judge on the skipped goal
        self.assertEqual(vision.detect_calls, 0)
        skipped = [
            w for w in caught if issubclass(w.category, UndrivenBehaviorRouteWarning)
        ]
        self.assertEqual(len(skipped), 1)
        self.assertIn("'count'", str(skipped[0].message))
        self.assertIn("Percept.perceive_boundary(", str(skipped[0].message))
        # The goal is untouched and its documented driver still fires it.
        self.assertIsNotNone(core.perceive_boundary("count", t=1.0))

    def test_foreign_source_unsupported_goal_does_not_block_this_worker(self):
        core = Percept.create(vision=_RouteVision({}), discover_plugins=False)
        core.add_goal(Goal("camera", "camera", "camera", source="camera"))
        core.add_goal(
            Goal("other", "other", "other", source="other", state={"type": "count"})
        )
        source = self.PullSpy()
        self.assertEqual(asyncio.run(_collect(core, source)), [])
        self.assertEqual(source.pulls, 1)

    def test_gated_loop_refuses_a_route_nothing_can_drive_before_a_frame_is_pulled(self):
        """The construction-time refusal, kept — and now scoped to the routes it was ever true of.

        This test used to pin ``type="event"`` (a HOST-driven route) as fatal at construction, which
        made :class:`GatedWatchLoop` the last driver where one such goal still killed every Watch
        beside it. What it was really protecting is asserted here instead, and unchanged: for a
        route nothing can drive, the loop refuses in ``__init__`` — nothing is compiled into the
        gate, the stream is never pulled, and no judge or detector call is spent.
        """
        vision = _RouteVision({})
        core = Percept.create(vision=vision, discover_plugins=False)
        core.add_goal(Goal("watch", "watch", "watch"))
        core.goals.get("watch").state["type"] = "wobble"  # corrupt it PAST the arm-time gate
        pulled: list[int] = []

        def stream():
            pulled.append(1)
            yield (0.0, STILL)

        with self.assertRaises(UnsupportedBehaviorRouteError) as caught:
            asyncio.run(run_gated_watch(core, stream()))

        self.assertEqual(caught.exception.unsupported[0].authored_type, "wobble")
        self.assertIn("GatedWatchLoop cannot drive", str(caught.exception))
        self.assertEqual(pulled, [])  # refused before the stream was ever consumed
        self.assertEqual(vision.judge_calls, [])
        self.assertEqual(vision.detect_calls, 0)

    def test_gated_loop_skips_a_host_driven_route_instead_of_killing_the_watch_beside_it(self):
        """The same composition rule ``percept.run`` follows — one driver, one behaviour.

        A ``count`` goal armed next to a ``watch`` goal on one source is exactly the topology this
        SDK's own advice produces ("drive it with ``Percept.perceive_boundary``"). It has a
        complete, shipped driver; this loop is simply not it. So it is excluded from the compiled
        spec and REPORTED — on the three channels ``percept.run`` reports on, with the same text and
        only the driver name differing — never allowed to decide the watch goal's fate.
        """
        vision = _RouteVision({})
        core = Percept.create(vision=vision, discover_plugins=False)
        core.add_goal(Goal("watch", "watch", "watch"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            core.add_goal(Goal("count", "count", "count", state={"type": "count"}))
        health: list = []

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertLogs("percept.harness.tier0", level="WARNING") as logs:
                loop = GatedWatchLoop(core, on_health=health.append)
                report = asyncio.run(loop.run([(0.0, STILL), (1.0, MOVED)]))

        # 1. The unrelated watch goal was compiled and the loop ran it — the whole point.
        self.assertEqual([c.goal_id for c in loop.spec.conditions], ["watch"])
        self.assertEqual(report.frames, 2)
        # 2. The count goal bought nothing: no condition, no admission, no judge.
        self.assertNotIn("count", report.admission_by_goal)
        self.assertEqual({condition for condition, _ in vision.judge_calls}, {"watch"})
        # 3. ...and it was not skipped silently — warning, log, and structured health, once each.
        skipped = [w for w in caught if issubclass(w.category, UndrivenBehaviorRouteWarning)]
        self.assertEqual(len(skipped), 1)
        message = str(skipped[0].message)
        self.assertIn("GatedWatchLoop is running source 'camera'", message)
        self.assertIn("'count'", message)
        self.assertIn("Percept.perceive_boundary('count', t=...)", message)
        self.assertTrue(
            any("undriven_behavior_route" in line and "'count'" in line for line in logs.output),
            logs.output,
        )
        undriven = [e for e in health if e.kind == "undriven_behavior_route"]
        self.assertEqual([e.data["goal_id"] for e in undriven], ["count"])
        self.assertEqual(undriven[0].data["route"], "external_boundary")
        self.assertEqual(undriven[0].data["driver"], "GatedWatchLoop")
        self.assertIn("Percept.perceive_boundary", undriven[0].data["remedy"])
        self.assertEqual(loop.skipped_routes, [e.data for e in undriven])
        # 4. The goal the loop skipped is still armed, and its documented driver still fires it.
        self.assertIsNotNone(core.perceive_boundary("count", t=9.0))

    def test_gated_loop_lets_a_strict_deployment_keep_the_old_abort(self):
        # The knob is the same one the arm seam and percept.run document, so nobody who WANTED the
        # construction-time abort loses it — they opt in instead of everyone paying for it.
        core = Percept.create(vision=_RouteVision({}), discover_plugins=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            core.add_goal(Goal("watch", "watch", "watch"))
            core.add_goal(Goal("event", "event", "event", state={"type": "event"}))
        with warnings.catch_warnings():
            warnings.simplefilter("error", UndrivenBehaviorRouteWarning)
            with self.assertRaises(UndrivenBehaviorRouteWarning):
                GatedWatchLoop(core)

    def test_gated_loop_filters_source_before_route_preflight_and_compilation(self):
        core = Percept.create(vision=_RouteVision({}), discover_plugins=False)
        core.add_goal(Goal("camera", "camera", "camera", source="camera"))
        core.add_goal(
            Goal("foreign", "foreign", "foreign", source="other", state={"type": "count"})
        )
        loop = GatedWatchLoop(core, source="camera")
        self.assertEqual([condition.goal_id for condition in loop.spec.conditions], ["camera"])

    def test_mid_run_external_route_is_skipped_without_stopping_the_running_watch(self):
        # The regate preflight makes the SAME per-goal decision as the startup one: a host-driven
        # goal armed mid-run (by voice, say) is reported and skipped; the running Watch keeps going.
        vision = _RouteVision({})
        core = Percept.create(vision=vision, discover_plugins=False)
        core.add_goal(Goal("watch", "watch", "watch"))
        pulled: list[int] = []

        def source():
            pulled.append(1)
            yield (0.0, STILL, "first")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                core.add_goal(Goal("count", "count", "count", state={"type": "count"}))
            pulled.append(2)
            yield (1.0, MOVED, "second")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(
                asyncio.run(_collect(core, source(), regate_on_goal_change=True)), []
            )
        self.assertEqual(pulled, [1, 2])
        # The count goal never bought a judge; the watch goal's own judging is unaffected.
        self.assertNotIn("count", [condition for condition, _ in vision.judge_calls])
        skipped = [
            w for w in caught if issubclass(w.category, UndrivenBehaviorRouteWarning)
        ]
        self.assertEqual(len(skipped), 1)  # once per goal per run, not once per frame
        self.assertIn("'count'", str(skipped[0].message))

    def test_mid_run_undrivable_route_still_fails_before_detect_or_judge(self):
        vision = _RouteVision({})
        core = Percept.create(vision=vision, discover_plugins=False)
        core.add_goal(Goal("watch", "watch", "watch"))
        pulled: list[int] = []

        def source():
            pulled.append(1)
            yield (0.0, STILL, "first")
            # add_goal refuses an UNSUPPORTED shape outright, so the only way one exists mid-run is a
            # post-arm mutation; the second (legitimate) arm is what advances the revision to regate.
            core.goals.get("watch").state["type"] = "wobble"
            core.add_goal(Goal("second", "second", "second"))
            pulled.append(2)
            yield (1.0, MOVED, "second")

        with self.assertRaises(UnsupportedBehaviorRouteError):
            asyncio.run(_collect(core, source()))
        self.assertEqual(pulled, [1, 2])
        self.assertEqual(vision.detect_calls, 0)
        self.assertEqual(vision.judge_calls, [])

    def test_stale_source_generation_aborts_before_detect_or_route_dispatch(self):
        vision = _RouteVision({})
        core = Percept.create(vision=vision, discover_plugins=False)
        core.add_goal(
            Goal(
                "transition",
                "transition",
                "transition",
                state={"type": "transition", "from": "a", "to": "b"},
            )
        )
        admitted_generation = core.source_continuity_generation("camera")
        core.advance_source_continuity("camera")
        events = asyncio.run(
            core.perceive_judged(
                "frame",
                source="camera",
                t=1.0,
                expected_source_generation=admitted_generation,
                goal_ids=("transition",),
            )
        )
        self.assertEqual(events, [])
        self.assertEqual(vision.detect_calls, 0)
        self.assertEqual(vision.judge_calls, [])


class FacilityTransitionRuntimeTests(unittest.TestCase):
    def test_each_shipped_transition_template_runs_and_rearms(self):
        for template_name in ("package_appeared", "standing_water"):
            with self.subTest(template=template_name):
                tokens = ["prime", "from", "to", "to", "from", "to"]
                core, vision, goal_id = _transition_core(template_name, tokens)
                stats = RuntimeStats()
                incidents = asyncio.run(
                    _collect(
                        core,
                        _motion_source(tokens),
                        judge_dispatch="inline",
                        stats=stats,
                    )
                )
                self.assertEqual([incident.concern_id for incident in incidents], [goal_id, goal_id])
                admitted_frames = len(tokens) - 1
                self.assertEqual(len(vision.judge_calls), admitted_frames * 2)
                # RuntimeStats.judge_calls is the admission/scheduler-launch invariant. Provider-call
                # accounting is the judged funnel: transitions spend two real calls per admitted frame.
                self.assertEqual(stats.judge_calls, admitted_frames)
                self.assertEqual(
                    core.metrics.get(funnel_key("judged", concern=goal_id)),
                    admitted_frames * 2,
                )
                self.assertEqual(
                    {condition for condition, _token in vision.judge_calls},
                    {
                        core.goals.get(goal_id).state["from"],
                        core.goals.get(goal_id).state["to"],
                    },
                )

    def test_joining_in_to_and_persistent_to_never_fire(self):
        tokens = ["prime", "to", "to", "to"]
        core, vision, _goal_id = _transition_core("package_appeared", tokens)
        incidents = asyncio.run(
            _collect(core, _motion_source(tokens), judge_dispatch="inline")
        )
        self.assertEqual(incidents, [])
        self.assertEqual(len(vision.judge_calls), (len(tokens) - 1) * 2)

    def test_async_dispatch_executes_transition_with_launch_and_provider_accounting(self):
        tokens = ["prime", "from", "to", "to"]
        core, vision, goal_id = _transition_core("package_appeared", tokens)
        stats = RuntimeStats()

        async def paced_source():
            for item in _motion_source(tokens):
                yield item
                await asyncio.sleep(0.01)

        incidents = asyncio.run(
            _collect(core, paced_source(), judge_dispatch="async", stats=stats)
        )
        self.assertEqual([incident.concern_id for incident in incidents], [goal_id])
        admitted_frames = len(tokens) - 1
        self.assertEqual(stats.judge_calls, admitted_frames)
        self.assertEqual(len(vision.judge_calls), admitted_frames * 2)
        self.assertEqual(
            core.metrics.get(funnel_key("judged", concern=goal_id)),
            admitted_frames * 2,
        )

    def test_low_confidence_transition_endpoint_does_not_fire(self):
        tokens = ["prime", "from", "uncertain"]
        core, vision, goal_id = _transition_core("package_appeared", tokens)
        goal = core.goals.get(goal_id)
        vision.script[(goal.state["from"], "uncertain")] = Verdict(
            satisfied=False, confidence=0.95, evidence="not before"
        )
        vision.script[(goal.state["to"], "uncertain")] = Verdict(
            satisfied=True, confidence=0.2, evidence="unclear"
        )
        incidents = asyncio.run(
            _collect(core, _motion_source(tokens), judge_dispatch="inline")
        )
        self.assertEqual(incidents, [])
        self.assertEqual(len(vision.judge_calls), (len(tokens) - 1) * 2)

    def test_transition_schedule_blocks_admission_before_provider_calls(self):
        tokens = ["prime", "from", "to"]
        plan = replace(
            watch_template("facility_observation@1/package_appeared"),
            schedule={"windows": [[0, 60]], "days": [1]},
        )
        core, vision, _goal_id = _transition_plan_core(plan, tokens)
        stats = RuntimeStats()
        incidents = asyncio.run(
            _collect(
                core,
                _motion_source(tokens),
                clock=lambda: CivilTime(weekday=1, minute_of_day=120),
                judge_dispatch="inline",
                stats=stats,
            )
        )
        self.assertEqual(incidents, [])
        self.assertEqual(stats.judge_calls, 0)
        self.assertEqual(vision.judge_calls, [])

    def test_transition_worker_only_judges_goals_for_its_source(self):
        tokens = ["prime", "from", "to"]
        plan = watch_template("facility_observation@1/package_appeared")
        camera_plan = replace(plan, id="camera-transition", source="camera")
        core, vision, camera_id = _transition_plan_core(camera_plan, tokens)
        foreign_plan = replace(plan, id="foreign-transition", source="other")
        foreign_receipt = apply_watch_plan(
            core, foreign_plan, domain_pack=FACILITY_OBSERVATION_V1
        )
        self.assertEqual(foreign_receipt.status, "applied")
        incidents = asyncio.run(
            _collect(
                core,
                _motion_source(tokens),
                gate_source="camera",
                judge_dispatch="inline",
            )
        )
        self.assertEqual([incident.concern_id for incident in incidents], [camera_id])
        self.assertEqual(
            core.metrics.get(funnel_key("judged", concern="foreign-transition")),
            0,
        )
        self.assertEqual(len(vision.judge_calls), (len(tokens) - 1) * 2)

    def test_compiled_transition_retains_bound_threshold(self):
        """The compiled condition keeps the cutoff the BINDING resolved — the one belonging to
        the ACTIVE statistic. ``watchspec_kwargs['threshold']`` is the mean-scale cutoff and is
        deliberately NOT stat-derived (matched-watch arms must be able to differ in
        ``salience_stat`` without ``threshold`` co-varying), so the invariant is
        ``stat_thresholds[salience_stat]``; those two coincide only when the active stat is the
        whole-frame mean."""
        core, _vision, goal_id = _transition_core("package_appeared", ["prime", "from"])
        compiled = core._watch_runtime_bindings[goal_id][0]
        kwargs = dict(compiled.watchspec_kwargs)
        condition = compile_watchspec([core.goals.get(goal_id)], **kwargs).conditions
        self.assertEqual(len(condition), 1)
        bound = kwargs["stat_thresholds"][kwargs["salience_stat"]]
        self.assertEqual(condition[0].threshold, bound)


class FacilityOccupancyRuntimeTests(unittest.TestCase):
    @staticmethod
    def _core(script: dict[str, Verdict], *, settings: Settings | None = None):
        from percept.backends.fake import FakeVision

        core = Percept.create(
            vision=FakeVision(script=script),
            settings=settings,
            discover_plugins=False,
        )
        plan = watch_template("facility_observation@1/zone_occupied")
        receipt = apply_watch_plan(core, plan, domain_pack=FACILITY_OBSERVATION_V1)
        if receipt.status != "applied":
            raise AssertionError(receipt)
        return core, receipt.watch_ids[0]

    def test_one_incident_per_contiguous_occupied_interval(self):
        script = {
            "empty": Verdict(False, 0.95, "the zone is empty"),
            "occupied": Verdict(True, 0.95, "a person is present"),
        }
        core, goal_id = self._core(script)
        tokens = ["prime", "empty", "occupied", "occupied"]
        incidents = asyncio.run(
            _collect(core, _motion_source(tokens), judge_dispatch="inline")
        )
        self.assertEqual([incident.concern_id for incident in incidents], [goal_id])

    def test_empty_rearms_a_second_contiguous_interval(self):
        script = {
            "empty": Verdict(False, 0.95, "the zone is empty"),
            "occupied": Verdict(True, 0.95, "a person is present"),
        }
        core, goal_id = self._core(script)
        tokens = ["prime", "empty", "occupied", "empty", "occupied"]
        incidents = asyncio.run(
            _collect(core, _motion_source(tokens), judge_dispatch="inline")
        )
        self.assertEqual([incident.concern_id for incident in incidents], [goal_id, goal_id])

    def test_unknown_does_not_manufacture_or_rearm_an_interval(self):
        script = {
            "empty": Verdict(False, 0.95, "the zone is empty"),
            "occupied": Verdict(True, 0.95, "a person is present"),
            "unknown": Verdict(True, 0.2, "unclear"),
        }
        core, goal_id = self._core(script)
        tokens = ["prime", "empty", "occupied", "unknown", "occupied"]
        incidents = asyncio.run(
            _collect(core, _motion_source(tokens), judge_dispatch="inline")
        )
        self.assertEqual([incident.concern_id for incident in incidents], [goal_id])

    def test_require_false_first_suppresses_join_then_rearms_after_observed_false(self):
        script = {
            "empty": Verdict(False, 0.95, "the zone is empty"),
            "occupied": Verdict(True, 0.95, "a person is present"),
        }
        settings = Settings(gate=GateSettings(require_false_first=True))
        core, goal_id = self._core(script, settings=settings)
        tokens = ["prime", "occupied", "empty", "occupied"]
        incidents = asyncio.run(
            _collect(core, _motion_source(tokens), judge_dispatch="inline")
        )
        self.assertEqual([incident.concern_id for incident in incidents], [goal_id])


class SceneScopedWatchTests(unittest.TestCase):
    E1 = Observation("person", Box(0.05, 0.05, 0.2, 0.2), 0.95)
    E2 = Observation("person", Box(0.70, 0.70, 0.2, 0.2), 0.95)

    @staticmethod
    def _profile_inheriting_scene() -> SourceProfile:
        return SourceProfile(
            name="scene-default",
            modalities=frozenset({"video"}),
            temporal_unit="frame",
            admission_strategy="tier0_motion_salience",
            evidence_strategy="crop",
            identity_strategy="scene",
        )

    def _direct_focus_turnover(
        self,
        *,
        declared_policy: str | None,
        source_profile: SourceProfile | None = None,
    ):
        condition = "a person is present"
        vision = _EntityRouteVision(
            {
                (condition, "e1"): Verdict(True, 0.95, "person one"),
                (condition, "e2"): Verdict(True, 0.95, "person two"),
            },
            {"e1": [self.E1], "e2": [self.E2]},
        )
        traces = []
        core = Percept.create(
            vision=vision,
            discover_plugins=False,
            on_trace=traces.append,
            source_profile=source_profile,
        )
        state = {"type": "watch"}
        if declared_policy is not None:
            state["identity_policy"] = declared_policy
        core.add_goal(Goal("g", condition, "occupied", state=state))
        first = asyncio.run(core.perceive_judged("e1", t=1.0))
        second = asyncio.run(core.perceive_judged("e2", t=2.0))
        return core, traces, first, second

    def test_declared_scene_uses_one_nonentity_reducer_across_focus_turnover(self):
        core, traces, first, second = self._direct_focus_turnover(
            declared_policy="scene"
        )
        self.assertEqual([event.action for event in first], ["fire"])
        self.assertEqual(second, [])
        fire = first[0]
        self.assertIsNone(fire.entity_id)
        self.assertEqual(fire.judgment.key, ("g", None))
        incident = make_incident(fire, ts=1.0)
        self.assertIsNone(incident.entity_id)
        gate_rows = [row for row in traces if row.stage == "gate.classified"]
        self.assertEqual([row.data["entity_id"] for row in gate_rows], [None, None])
        judgment_row = next(row for row in traces if row.stage == "judgment.event")
        self.assertIsNone(judgment_row.data["entity_id"])
        self.assertEqual(
            incident.judgment_event_id,
            judgment_row.data["judgment_event_id"],
        )
        self.assertIn(("g", None), core.gate._armed)
        self.assertFalse(
            any(key[0] == "g" and key[1] is not None for key in core.gate._armed)
        )

    def test_ordinary_per_entity_and_profile_inherited_scope_are_unchanged(self):
        cases = (
            ("undeclared", None, None),
            ("declared-track", "track_iou_label", None),
            ("profile-inherited-scene", None, self._profile_inheriting_scene()),
        )
        for label, declared, profile in cases:
            with self.subTest(case=label):
                _core, traces, first, second = self._direct_focus_turnover(
                    declared_policy=declared,
                    source_profile=profile,
                )
                self.assertEqual(
                    [event.action for event in first + second],
                    ["fire", "fire"],
                )
                entity_ids = [event.entity_id for event in first + second]
                self.assertTrue(all(entity_id is not None for entity_id in entity_ids))
                self.assertNotEqual(entity_ids[0], entity_ids[1])
                self.assertEqual(
                    [event.judgment.key[-1] for event in first + second],
                    entity_ids,
                )
                gate_rows = [row for row in traces if row.stage == "gate.classified"]
                self.assertEqual(
                    [row.data["entity_id"] for row in gate_rows],
                    entity_ids,
                )

    def test_zone_template_real_worker_fires_once_per_contiguous_scene_interval(self):
        plan = watch_template("facility_observation@1/zone_occupied")
        condition = plan.condition
        tokens = [
            "prime",
            "occupied-e1",
            "occupied-e2",
            "empty-e2",
            "occupied-e2-again",
        ]
        vision = _EntityRouteVision(
            {
                (condition, "occupied-e1"): Verdict(True, 0.95, "person one"),
                (condition, "occupied-e2"): Verdict(True, 0.95, "person two"),
                (condition, "empty-e2"): Verdict(False, 0.95, "zone empty"),
                (condition, "occupied-e2-again"): Verdict(True, 0.95, "person two"),
            },
            {
                "occupied-e1": [self.E1],
                "occupied-e2": [self.E2],
                "empty-e2": [self.E2],
                "occupied-e2-again": [self.E2],
            },
        )
        core = Percept.create(vision=vision, discover_plugins=False)
        receipt = apply_watch_plan(
            core, plan, domain_pack=FACILITY_OBSERVATION_V1
        )
        self.assertEqual(receipt.status, "applied")
        goal_id = receipt.watch_ids[0]
        self.assertEqual(core.goals.get(goal_id).state["identity_policy"], "scene")

        traces = []
        incidents = asyncio.run(
            _collect(
                core,
                _motion_source(tokens),
                judge_dispatch="inline",
                on_trace=traces.append,
            )
        )

        # e1 true opens the scene interval; e2 true is the same interval; false closes/re-arms;
        # e2 true then opens one new interval.
        self.assertEqual(
            [incident.concern_id for incident in incidents],
            [goal_id, goal_id],
        )
        self.assertEqual([incident.entity_id for incident in incidents], [None, None])
        self.assertTrue(
            all(incident.judgment_event_id is not None for incident in incidents)
        )
        gate_rows = [
            row
            for row in traces
            if row.stage == "gate.classified" and row.goal_id == goal_id
        ]
        self.assertEqual(
            [row.data["entity_id"] for row in gate_rows],
            [None, None, None, None],
        )
        judgment_rows = [
            row
            for row in traces
            if row.stage == "judgment.event" and row.goal_id == goal_id
        ]
        self.assertEqual(
            [row.data["entity_id"] for row in judgment_rows],
            [None, None],
        )
        self.assertEqual(
            [row.data["judgment_event_id"] for row in judgment_rows],
            [incident.judgment_event_id for incident in incidents],
        )
        self.assertIn((goal_id, None), core.gate._armed)
        self.assertFalse(
            any(key[0] == goal_id and key[1] is not None for key in core.gate._armed)
        )


class ArmTimeRouteGateTests(unittest.TestCase):
    """The arm boundary must make an undriven route LOUD — the silent no-op is the defect.

    ``percept`` declares eight behavior routes and the frame worker drives four. The other four used
    to arm in total silence: ``add_goal`` returned an ``active`` goal, no frame ever selected it, no
    incident ever landed, and nothing in any log said why. These tests pin the two different honest
    answers to that, per route:

    * a route with a shipped host-facing driver (``count``/``event``/``remind``/``scheduler``) is
      *unwired*, not unsupported — it WARNS and still arms, because every one of those drivers looks
      the goal up by id and refusing the arm would make them unreachable;
    * a route nothing implements is dead on arrival — it RAISES, and arms nothing.
    """

    #: authored type -> (route value, a substring of the exact host API the message must name)
    UNDRIVEN = {
        "count": ("external_boundary", "Percept.perceive_boundary("),
        "event": ("external_event_window", "await Percept.perceive_events("),
        "remind": ("external_scheduler", "Percept.schedule_every("),
        "scheduler": ("external_scheduler", "Percept.schedule_at("),
    }

    @staticmethod
    def _core():
        return Percept.create(vision=_RouteVision({}), discover_plugins=False)

    def test_each_undriven_route_warns_loudly_at_arm_and_names_its_host_driver(self):
        for authored_type, (route_value, api) in self.UNDRIVEN.items():
            with self.subTest(behavior_type=authored_type):
                core = self._core()
                goal = Goal("g", "something happens", "fired", state={"type": authored_type})
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    core.add_goal(goal)
                undriven = [
                    w for w in caught
                    if issubclass(w.category, UndrivenBehaviorRouteWarning)
                ]
                self.assertEqual(len(undriven), 1)
                message = str(undriven[0].message)
                # names the goal, the route, and — the whole point — what to do instead.
                self.assertIn("'g'", message)
                self.assertIn(repr(authored_type), message)
                self.assertIn(repr(route_value), message)
                self.assertIn(api, message)
                self.assertIn("does not drive", message)
                # ARMED anyway: the host driver resolves the goal BY ID, so refusing would delete it.
                self.assertEqual([g.id for g in core.goals.active()], ["g"])

    def test_undriven_route_warning_also_reaches_the_deployment_log_and_metrics(self):
        # A 24/7 deployment has nobody watching stderr; the silent no-op lived exactly there.
        core = self._core()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertLogs("percept.agent", level="WARNING") as logs:
                core.add_goal(Goal("c", "count things", "counted", state={"type": "count"}))
        self.assertTrue(
            any("undriven_behavior_route" in line for line in logs.output), logs.output
        )
        self.assertEqual(core.metrics.get("goal.armed.undriven_route.external_boundary"), 1)

    def test_undriven_route_is_a_deployment_seam_not_a_missing_feature(self):
        # The justification for warn-not-raise, executable: the driver the message names WORKS.
        core = self._core()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            core.add_goal(Goal("c", "count reps", "counted", state={"type": "count"}))
        self.assertIsNotNone(core.perceive_boundary("c", t=1.0))

    def test_a_strict_deployment_can_escalate_the_warning_to_a_hard_arm_failure(self):
        core = self._core()
        with warnings.catch_warnings():
            warnings.simplefilter("error", UndrivenBehaviorRouteWarning)
            with self.assertRaises(UndrivenBehaviorRouteWarning):
                core.add_goal(Goal("c", "count things", "counted", state={"type": "count"}))
        # The check runs in the registry's pre-commit hook, so escalating is fail-CLOSED.
        self.assertEqual(list(core.goals.active()), [])
        self.assertEqual(core.goals.revision, 0)

    def test_a_route_nothing_implements_raises_and_arms_nothing(self):
        for state in (
            {"type": "made_up"},
            {"type": "transition", "from": "closed", "to": "   "},
            {"type": "watch", "behavior": {"type": "count"}},
        ):
            with self.subTest(state=state):
                core = self._core()
                with self.assertRaises(UnsupportedBehaviorRouteError) as caught:
                    core.add_goal(Goal("dead", "dead on arrival", "never", state=state))
                message = str(caught.exception)
                self.assertIn("'dead'", message)
                self.assertIn("route=unsupported", message)
                # No false "use the matching primitive" claim — there is no primitive to match.
                self.assertNotIn("matching direct", message)
                self.assertIn("closed behavior types", message)
                self.assertEqual(list(core.goals.active()), [])
                self.assertEqual(core.goals.revision, 0)

    def test_the_gate_is_one_seam_covering_every_arm_path_including_the_conductor(self):
        # add_goal (structured), apply_goal_ops (the conductor's language path), and a direct
        # GoalRegistry mutation all funnel through the registry's owner hook. One check, not three.
        def count_goal(gid):
            return Goal(gid, "count things", "counted", state={"type": "count"})

        arms = {
            "add_goal": lambda core, gid: core.add_goal(count_goal(gid)),
            "apply_goal_ops": lambda core, gid: core.apply_goal_ops(
                [GoalOp(op="create", goal=count_goal(gid))]
            ),
            "goals.create": lambda core, gid: core.goals.create(count_goal(gid)),
        }
        for name, arm in arms.items():
            with self.subTest(arm_path=name):
                core = self._core()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    arm(core, "g")
                undriven = [
                    w for w in caught
                    if issubclass(w.category, UndrivenBehaviorRouteWarning)
                ]
                # exactly one — the conductor path validates twice but commits (and warns) once.
                self.assertEqual(len(undriven), 1)
                self.assertIn("Percept.perceive_boundary(", str(undriven[0].message))

    def test_every_worker_driven_route_arms_silently_and_unchanged(self):
        drivable = {
            "watch": {"type": "watch"},
            "watch_default": {},
            "transition": {"type": "transition", "from": "closed", "to": "open"},
            "process": {"type": "process"},
            "rhythm": {"type": "rhythm"},
        }
        for name, state in drivable.items():
            with self.subTest(behavior_type=name):
                core = self._core()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    armed = core.add_goal(Goal("g", "a condition", "fired", state=dict(state)))
                self.assertEqual(
                    [w for w in caught if issubclass(w.category, UndrivenBehaviorRouteWarning)],
                    [],
                )
                self.assertEqual([g.id for g in core.goals.active()], ["g"])
                self.assertTrue(worker_can_drive(goal_behavior_route(armed)))
                self.assertEqual(
                    [
                        name
                        for name in core.metrics.snapshot()
                        if name.startswith("goal.armed.undriven_route")
                    ],
                    [],
                )

    def test_apply_watch_plan_rejection_names_the_same_host_driver(self):
        # The apply seam still REJECTS (its contract is "arm a Watch percept.run drives"), but a
        # rejection that says only "unsupported" is not a fix — it must name the same remedy.
        for authored_type, (_route_value, api) in self.UNDRIVEN.items():
            with self.subTest(behavior_type=authored_type):
                core = Percept.create(mode="offline", discover_plugins=False)
                receipt = apply_watch_plan(
                    core,
                    WatchPlan(
                        id="p",
                        condition="external primitive",
                        explicit={"Goal.state.type": authored_type},
                    ),
                )
                self.assertEqual(receipt.status, "rejected")
                self.assertIn(api, receipt.dropped_fields[0].message)
                self.assertEqual(list(core.goals.active()), [])


if __name__ == "__main__":
    unittest.main()
