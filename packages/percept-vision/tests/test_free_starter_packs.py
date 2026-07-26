"""Official free DomainPack mechanics.

These tests prove SDK-owned compilation and state-machine behavior with scripted verdicts. They do
not measure perception accuracy, provider latency, or cost.
"""

from __future__ import annotations

import unittest

from percept.adapters import Verdict
from percept.agent import Percept
from percept.backends.fake import FakeVision
from percept.behavior import program_of
from percept.domain_pack import get_domain_pack, watch_template
from percept.gate import ASK, FIRE
from percept.harness.tier0 import gray_frame
from percept.packs_default import (
    EGO_MOTION_STARTER_V1,
    FACILITY_OBSERVATION_V1,
    PHYSICAL_SCENE_STARTER_V1,
)
from percept.replay import gate_config_of, replay_cognition
from percept.resolver import ConfigResolver
from percept.runtime import RuntimeStats, run
from percept.runtime_compiler import runtime_compile
from percept.tracing import RunTraceManifest, TracePolicy, TraceRecorder
from percept.watchplan_apply import apply_watch_plan


class _ConditionVision:
    """Condition-aware scripted judge for two-call transition mechanics."""

    def __init__(self, script: dict[tuple[str, str], Verdict]) -> None:
        self.script = dict(script)
        self.judge_calls: list[tuple[str, str]] = []

    async def judge(self, condition, frame, **_kwargs):
        key = (str(condition), str(frame))
        self.judge_calls.append(key)
        return self.script.get(
            key,
            Verdict(satisfied=False, confidence=0.95, evidence="scripted: not observed"),
        )

    async def detect(self, frame, goals):
        return []

    async def detect_events(self, frames, goals):
        return []


def _apply(core: Percept, ref: str):
    pack_name = ref.partition("/")[0]
    pack = get_domain_pack(pack_name)
    plan = watch_template(ref)
    receipt = apply_watch_plan(core, plan, domain_pack=pack)
    if receipt.status not in {"applied", "applied_with_warnings"}:
        raise AssertionError(receipt)
    return plan, core.goals.get(receipt.watch_ids[0])


class TestOfficialPackSurface(unittest.TestCase):
    def test_official_pack_names_and_legacy_compatibility_name_resolve(self) -> None:
        self.assertIs(get_domain_pack("physical_scene_starter@1"), PHYSICAL_SCENE_STARTER_V1)
        self.assertIs(get_domain_pack("ego_motion_starter@1"), EGO_MOTION_STARTER_V1)
        self.assertIs(get_domain_pack("facility_observation@1"), FACILITY_OBSERVATION_V1)

    def test_physical_starter_reuses_the_legacy_template_values(self) -> None:
        self.assertEqual(
            tuple(PHYSICAL_SCENE_STARTER_V1.templates),
            ("door_left_open", "package_appeared", "standing_water", "zone_occupied"),
        )
        for name, template in PHYSICAL_SCENE_STARTER_V1.templates.items():
            self.assertIs(template, FACILITY_OBSERVATION_V1.templates[name])

    def test_ego_starter_is_the_approved_mechanics_only_surface(self) -> None:
        self.assertEqual(
            set(EGO_MOTION_STARTER_V1.templates),
            {
                "camera_view_moving",
                "scene_transition",
                "object_focus_gained",
                "task_step_reached",
            },
        )
        self.assertEqual(EGO_MOTION_STARTER_V1.fixtures, ())
        self.assertEqual(dict(EGO_MOTION_STARTER_V1.quality_bars), {})
        self.assertEqual(dict(EGO_MOTION_STARTER_V1.search_space), {})
        self.assertIn("Revisit", EGO_MOTION_STARTER_V1.notes)
        self.assertIn("same-object reacquisition", EGO_MOTION_STARTER_V1.notes)
        self.assertIn("identity", EGO_MOTION_STARTER_V1.notes)
        self.assertIn("tracking", EGO_MOTION_STARTER_V1.notes)
        self.assertIn("task understanding", EGO_MOTION_STARTER_V1.notes)

    def test_both_official_starters_are_explicitly_unqualified(self) -> None:
        for pack in (PHYSICAL_SCENE_STARTER_V1, EGO_MOTION_STARTER_V1):
            with self.subTest(pack=pack.qualified_name):
                self.assertEqual(pack.fixtures, ())
                self.assertEqual(dict(pack.quality_bars), {})
                self.assertIn("Availability is not qualification", pack.notes)
                self.assertIn("no accuracy, latency, or cost claim", pack.notes)
                self.assertEqual(pack.direct_judge_routes, ())

    def test_restricted_claim_tokens_are_not_template_names(self) -> None:
        names = set(EGO_MOTION_STARTER_V1.templates)
        for unsupported in (
            "revisit",
            "reacquisition",
            "identity",
            "tracking",
            "task_understanding",
        ):
            self.assertNotIn(unsupported, names)

    def test_templates_never_auto_arm(self) -> None:
        core = Percept.create(mode="offline", discover_plugins=False)
        before = len(core.goals.all())
        for pack in (PHYSICAL_SCENE_STARTER_V1, EGO_MOTION_STARTER_V1):
            get_domain_pack(pack.qualified_name)
            for name in pack.templates:
                watch_template(f"{pack.qualified_name}/{name}")
        self.assertEqual(len(core.goals.all()), before)


class TestDeterministicCompilation(unittest.TestCase):
    def test_each_official_template_compiles_deterministically(self) -> None:
        resolver = ConfigResolver()
        for pack in (PHYSICAL_SCENE_STARTER_V1, EGO_MOTION_STARTER_V1):
            for name in pack.templates:
                with self.subTest(pack=pack.qualified_name, template=name):
                    plan_a = watch_template(f"{pack.qualified_name}/{name}")
                    plan_b = watch_template(f"{pack.qualified_name}/{name}")
                    self.assertEqual(plan_a.to_dict(), plan_b.to_dict())
                    resolved_a = resolver.resolve(plan_a, domain_pack=pack)
                    resolved_b = resolver.resolve(plan_b, domain_pack=pack)
                    self.assertEqual(resolved_a.plan_hash, resolved_b.plan_hash)
                    self.assertEqual(
                        resolved_a.component_snapshot_hash,
                        resolved_b.component_snapshot_hash,
                    )
                    compiled_a = runtime_compile(resolved_a)
                    compiled_b = runtime_compile(resolved_b)
                    self.assertEqual(compiled_a.goal, compiled_b.goal)
                    self.assertEqual(compiled_a.watchspec_kwargs, compiled_b.watchspec_kwargs)
                    self.assertEqual(compiled_a.sense_policy, compiled_b.sense_policy)

    def test_ego_routes_compile_to_existing_behavior_primitives(self) -> None:
        expected = {
            "camera_view_moving": "judge",
            "scene_transition": "transition",
            "object_focus_gained": "transition",
            "task_step_reached": "transition",
        }
        core = Percept.create(mode="offline", discover_plugins=False)
        for name, sense in expected.items():
            with self.subTest(template=name):
                _plan, goal = _apply(core, f"ego_motion_starter@1/{name}")
                self.assertEqual(program_of(goal).sense, sense)
                self.assertEqual(goal.state["identity_policy"], "scene")

    def test_ego_parameters_remain_typed_through_compilation(self) -> None:
        plan = watch_template(
            "ego_motion_starter@1/scene_transition",
            {
                "from_scene": "the kitchen",
                "to_scene": "the hallway",
                "observation_interval_s": 0.75,
            },
        )
        self.assertEqual(
            plan.explicit["Goal.state.from"], "the kitchen is the current visible scene"
        )
        self.assertEqual(plan.explicit["Goal.state.to"], "the hallway is the current visible scene")
        self.assertEqual(plan.explicit["Goal.state.cadence.min_judge_interval"], 0.75)
        self.assertEqual(plan.explicit["Goal.state.cadence.gate_on"], "fixed")
        self.assertEqual(plan.sense, "vision")


class TestEgoStarterStateMachines(unittest.IsolatedAsyncioTestCase):
    async def test_camera_motion_rising_edge_and_ambiguity(self) -> None:
        vision = FakeVision(
            script={
                "still": Verdict(False, 0.95, "scripted: stable"),
                "unclear": Verdict(True, 0.20, "scripted: ambiguous"),
                "moving": Verdict(True, 0.95, "scripted: moving"),
            }
        )
        core = Percept.create(vision=vision, discover_plugins=False)
        _plan, _goal = _apply(core, "ego_motion_starter@1/camera_view_moving")

        actions: list[str] = []
        for index, token in enumerate(
            ("still", "unclear", "moving", "moving", "still", "moving"),
            start=1,
        ):
            actions.extend(
                event.action for event in await core.perceive_judged(token, t=float(index))
            )

        self.assertEqual(actions, [ASK, FIRE, FIRE])

    async def test_each_transition_arms_fires_once_and_rearms(self) -> None:
        for name in ("scene_transition", "object_focus_gained", "task_step_reached"):
            with self.subTest(template=name):
                plan = watch_template(f"ego_motion_starter@1/{name}")
                before = plan.explicit["Goal.state.from"]
                after = plan.explicit["Goal.state.to"]
                script: dict[tuple[str, str], Verdict] = {}
                for token in ("from", "unclear", "to"):
                    script[(before, token)] = Verdict(
                        token == "from",
                        0.20 if token == "unclear" else 0.95,
                        f"scripted before: {token}",
                    )
                    script[(after, token)] = Verdict(
                        token == "to",
                        0.20 if token == "unclear" else 0.95,
                        f"scripted after: {token}",
                    )
                vision = _ConditionVision(script)
                core = Percept.create(vision=vision, discover_plugins=False)
                _plan, goal = _apply(core, f"ego_motion_starter@1/{name}")

                events = []
                for index, token in enumerate(
                    ("from", "unclear", "to", "to", "from", "to"),
                    start=1,
                ):
                    events.extend(await core.perceive_judged(token, t=float(index)))

                self.assertEqual([event.action for event in events], [FIRE, FIRE])
                self.assertEqual(goal.state["count"], 2)
                self.assertEqual(len(vision.judge_calls), 12)

    async def test_live_ordinary_trace_replays_byte_identically(self) -> None:
        vision = FakeVision(
            script={
                "still": Verdict(False, 0.95, "scripted: stable"),
                "moving": Verdict(True, 0.95, "scripted: moving"),
            }
        )
        core = Percept.create(vision=vision, discover_plugins=False)
        _plan, goal = _apply(core, "ego_motion_starter@1/camera_view_moving")
        recorder = TraceRecorder(
            RunTraceManifest.create(
                source="camera",
                driver="perceive_judged",
                run_id="ego-pack-mechanics",
                goals=[goal],
                policy=TracePolicy(level="full"),
                gate_config=gate_config_of(core.gate),
            )
        )
        core.on_trace = recorder.tracer()

        live_actions: list[str] = []
        for index, token in enumerate(("still", "moving", "moving", "still", "moving"), start=1):
            live_actions.extend(
                event.action for event in await core.perceive_judged(token, t=float(index))
            )

        first = replay_cognition(recorder.manifest, list(recorder.records))
        second = replay_cognition(recorder.manifest, list(recorder.records))
        self.assertEqual(first.digest, second.digest)
        self.assertEqual(
            [decision.to_dict() for decision in first.decisions],
            [decision.to_dict() for decision in second.decisions],
        )
        self.assertEqual(
            [decision.outward_action for decision in first.decisions if decision.outward_action],
            live_actions,
        )
        self.assertEqual(first.divergences, ())

    async def test_scene_transition_runs_through_fixed_cadence_worker(self) -> None:
        plan = watch_template(
            "ego_motion_starter@1/scene_transition",
            {
                "from_scene": "the kitchen",
                "to_scene": "the hallway",
                "observation_interval_s": 1.0,
            },
        )
        before = plan.explicit["Goal.state.from"]
        after = plan.explicit["Goal.state.to"]
        tokens = ("from", "to", "to", "from", "to")
        script: dict[tuple[str, str], Verdict] = {}
        for token in tokens:
            script[(before, token)] = Verdict(token == "from", 0.95, f"scripted before: {token}")
            script[(after, token)] = Verdict(token == "to", 0.95, f"scripted after: {token}")
        vision = _ConditionVision(script)
        core = Percept.create(vision=vision, discover_plugins=False)
        receipt = apply_watch_plan(core, plan, domain_pack=EGO_MOTION_STARTER_V1)
        self.assertEqual(receipt.status, "applied")
        goal_id = receipt.watch_ids[0]
        still = gray_frame(8, 8, 0)
        source = [(float(index), still, token) for index, token in enumerate(tokens)]
        stats = RuntimeStats()

        incidents = [
            incident
            async for incident in run(
                core,
                source=source,
                judge_dispatch="inline",
                stats=stats,
            )
        ]

        self.assertEqual([incident.concern_id for incident in incidents], [goal_id, goal_id])
        self.assertEqual(stats.frames, len(tokens))
        self.assertEqual(stats.goal_admissions, len(tokens))
        self.assertEqual(stats.admission_by_cause, {"fixed": len(tokens)})
        self.assertEqual(stats.judge_calls, len(tokens))
        self.assertEqual(len(vision.judge_calls), len(tokens) * 2)


if __name__ == "__main__":
    unittest.main()
