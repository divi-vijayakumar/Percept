"""Executable behavior-route contract through structured apply and the real runtime worker."""

from __future__ import annotations

import asyncio
import unittest
from dataclasses import replace

from percept import Goal, Percept
from percept.adapters import Box, Observation, Verdict
from percept.behavior import coerce_authored_behavior
from percept.behavior_routes import (
    BehaviorRoute,
    UnsupportedBehaviorRouteError,
    behavior_route,
    goal_behavior_route,
)
from percept.config import GateSettings, Settings
from percept.contracts.watchplan import WatchPlan
from percept.domain_pack import watch_template
from percept.edge.tier0 import (
    CivilTime,
    GatedWatchLoop,
    compile_watchspec,
    gray_frame,
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

        worker_vision = _RouteVision({})
        worker_core = Percept.create(vision=worker_vision, discover_plugins=False)
        worker_core.add_goal(raw_goal)
        source = WorkerPreflightTests.PullSpy()
        with self.assertRaises(UnsupportedBehaviorRouteError):
            asyncio.run(_collect(worker_core, source))
        self.assertEqual(source.pulls, 0)
        self.assertEqual(worker_vision.detect_calls, 0)
        self.assertEqual(worker_vision.judge_calls, [])

        direct_vision = _RouteVision({})
        direct_core = Percept.create(vision=direct_vision, discover_plugins=False)
        direct_core.add_goal(raw_goal)
        with self.assertRaises(UnsupportedBehaviorRouteError):
            asyncio.run(direct_core.perceive_judged("frame", source="camera", t=1.0))
        self.assertEqual(direct_vision.detect_calls, 0)
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

    def test_every_external_route_is_rejected_with_no_goal_mutation(self):
        for route_type in ("event", "count", "rhythm", "remind", "scheduler"):
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
        core = Percept.create(vision=_RouteVision({}), discover_plugins=False)
        core.add_goal(Goal("count", "count", "count", state={"type": "count"}))
        source = self.PullSpy()
        with self.assertRaises(UnsupportedBehaviorRouteError) as caught:
            asyncio.run(_collect(core, source))
        self.assertEqual(source.pulls, 0)
        self.assertEqual(caught.exception.unsupported[0].authored_type, "count")

    def test_foreign_source_unsupported_goal_does_not_block_this_worker(self):
        core = Percept.create(vision=_RouteVision({}), discover_plugins=False)
        core.add_goal(Goal("camera", "camera", "camera", source="camera"))
        core.add_goal(
            Goal("other", "other", "other", source="other", state={"type": "count"})
        )
        source = self.PullSpy()
        self.assertEqual(asyncio.run(_collect(core, source)), [])
        self.assertEqual(source.pulls, 1)

    def test_gated_loop_rejects_external_route_at_construction(self):
        core = Percept.create(vision=_RouteVision({}), discover_plugins=False)
        goal = Goal("rhythm", "rhythm", "rhythm", state={"type": "rhythm"})
        core.add_goal(goal)
        with self.assertRaises(UnsupportedBehaviorRouteError):
            GatedWatchLoop(core)

    def test_gated_loop_filters_source_before_route_preflight_and_compilation(self):
        core = Percept.create(vision=_RouteVision({}), discover_plugins=False)
        core.add_goal(Goal("camera", "camera", "camera", source="camera"))
        core.add_goal(
            Goal("foreign", "foreign", "foreign", source="other", state={"type": "count"})
        )
        loop = GatedWatchLoop(core, source="camera")
        self.assertEqual([condition.goal_id for condition in loop.spec.conditions], ["camera"])

    def test_mid_run_external_route_fails_before_detect_or_judge(self):
        vision = _RouteVision({})
        core = Percept.create(vision=vision, discover_plugins=False)
        core.add_goal(Goal("watch", "watch", "watch"))
        pulled: list[int] = []

        def source():
            pulled.append(1)
            yield (0.0, STILL, "first")
            core.add_goal(Goal("count", "count", "count", state={"type": "count"}))
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
        core, _vision, goal_id = _transition_core("package_appeared", ["prime", "from"])
        compiled = core._watch_runtime_bindings[goal_id][0]
        condition = compile_watchspec(
            [core.goals.get(goal_id)], **dict(compiled.watchspec_kwargs)
        ).conditions
        self.assertEqual(len(condition), 1)
        self.assertEqual(condition[0].threshold, compiled.watchspec_kwargs["threshold"])


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


if __name__ == "__main__":
    unittest.main()
