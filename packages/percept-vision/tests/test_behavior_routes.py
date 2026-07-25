"""Executable behavior-route contract through structured apply and the real runtime worker."""

from __future__ import annotations

import asyncio
import unittest

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.behavior import coerce_authored_behavior
from percept.behavior_routes import (
    BehaviorRoute,
    UnsupportedBehaviorRouteError,
    behavior_route,
    goal_behavior_route,
)
from percept.contracts.watchplan import WatchPlan
from percept.domain_pack import watch_template
from percept.edge.tier0 import GatedWatchLoop, compile_watchspec, gray_frame, watchspec_unsupported
from percept.packs_default import FACILITY_OBSERVATION_V1
from percept.observability import funnel_key
from percept.runtime import RuntimeStats, run
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


def _transition_core(template_name: str, tokens: list[str]):
    plan = watch_template(f"facility_observation@1/{template_name}")
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
    core = Percept.create(vision=vision, discover_plugins=False)
    receipt = apply_watch_plan(core, plan, domain_pack=FACILITY_OBSERVATION_V1)
    if receipt.status not in ("applied", "applied_with_warnings"):
        raise AssertionError(receipt)
    return core, vision, receipt.watch_ids[0]


def _motion_source(tokens: list[str]):
    frames = (STILL, MOVED)
    return [(float(index), frames[index % 2], token) for index, token in enumerate(tokens)]


class RouteClassifierTests(unittest.TestCase):
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
    def _core(script: dict[str, Verdict]):
        from percept.backends.fake import FakeVision

        core = Percept.create(vision=FakeVision(script=script), discover_plugins=False)
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


if __name__ == "__main__":
    unittest.main()
