"""Executable WatchPlan compilation and per-Watch runtime isolation.

These tests cross the full public path:
WatchPlan -> ConfigResolver -> ResolvedWatch -> RuntimeCompiler -> apply -> runtime.run.
All fixtures are deterministic and offline.
"""

from __future__ import annotations

import asyncio
import unittest

from percept import Goal, Percept
from percept.agent import FireEvent
from percept.adapters import Verdict
from percept.backends.anthropic import AnthropicVision
from percept.backends.fake import FakeVision
from percept.backends.gemini import GeminiVision
from percept.backends.openrouter import OpenRouterVision
from percept.behavior import program_of
from percept.contracts import AudioFrame
from percept.contracts.watchplan import WatchPlan
from percept.goals import GoalOp
from percept.harness.tier0 import CivilTime, Frame, gray_frame
from percept.profile_wiring import resolve_watch_profile
from percept.runtime.worker import RuntimeStats, run
from percept.sourceprofile import (
    DEFAULT_SOURCE_REGISTRY,
    SourceProfile,
    register_source_profile,
)
from percept.watchplan_apply import apply_watch_plan
from percept.watchprofile import (
    CadenceProfile,
    GateProfile,
    WatchProfile,
    WatchProfileRegistry,
)


W, H = 48, 36
STILL = gray_frame(W, H, 0)


def _local_motion() -> Frame:
    pixels = [0] * (W * H)
    for y in range(8):
        for x in range(8):
            pixels[y * W + x] = 255
    return Frame(w=W, h=H, pixels=tuple(pixels))


def _profile(
    name: str,
    *,
    stat: str = "mean",
    threshold: float = 0.1,
    coast_interval_s: float | None = None,
    identity: str = "track_continuity",
    timebase: str = "auto",
    coverage_floor: float | None = None,
    modalities: frozenset[str] = frozenset({"video"}),
) -> SourceProfile:
    parameters = {
        "salience_stat": stat,
        "salience_threshold": threshold,
    }
    if coast_interval_s is not None:
        parameters["coast_interval_s"] = coast_interval_s
    if coverage_floor is not None:
        parameters["coverage_floor"] = coverage_floor
    return SourceProfile(
        name=name,
        modalities=modalities,
        temporal_unit="frame",
        admission_strategy="pixel-change",
        evidence_strategy="frame",
        identity_strategy=identity,
        default_timebase=timebase,
        parameters=parameters,
    )


class RecordingVision:
    SUPPORTS_FUSED_AUDIO_INPUT = True

    def __init__(self) -> None:
        self.detected: list[tuple[str, ...]] = []
        self.judged: list[tuple[str, str]] = []
        self.audio_payloads: list[object | None] = []

    async def detect(self, frame, goals):
        self.detected.append(tuple(goal.id for goal in goals))
        return []

    async def judge(self, condition, frame, **kwargs):
        self.judged.append((condition, str(frame)))
        self.audio_payloads.append(kwargs.get("audio"))
        return Verdict(satisfied=False, confidence=0.99, evidence="not present")


class VisionOnlyRecording(RecordingVision):
    SUPPORTS_FUSED_AUDIO_INPUT = False


async def _collect(core: Percept, source, **kwargs):
    return [incident async for incident in run(core, source=source, **kwargs)]


class WatchPlanCompilationTests(unittest.TestCase):
    def test_supported_media_senses_compile_into_the_existing_perception_program(self) -> None:
        core = Percept.create(vision=FakeVision(), discover_plugins=False)
        expected = {
            "video": "vision",
            "vision": "vision",
            "audio": "audio",
            "fused": "fused",
        }
        for index, (sense, modality) in enumerate(expected.items()):
            plan = WatchPlan(
                id=f"g-{index}",
                condition=f"condition {index}",
                sense=sense,
            )
            receipt = apply_watch_plan(core, plan)
            self.assertEqual(receipt.status, "applied")
            self.assertEqual(
                program_of(core.goals.get(f"g-{index}")).perception.modality,
                modality,
            )
            if modality in ("audio", "fused"):
                self.assertEqual(
                    program_of(core.goals.get(f"g-{index}")).perception.audio_s,
                    core.settings.events.audio_window_s,
                )

    def test_unsupported_media_sense_is_rejected_before_arm(self) -> None:
        core = Percept.create(vision=RecordingVision(), discover_plugins=False)
        receipt = apply_watch_plan(
            core,
            WatchPlan(id="lidar", condition="an obstacle is close", sense="lidar"),
        )
        self.assertEqual(receipt.status, "rejected")
        self.assertEqual(receipt.watch_ids, ())
        self.assertEqual(receipt.dropped_fields[0].code, "unsupported_media_sense")
        self.assertEqual(core.goals.all(), [])

    def test_media_sense_must_exist_on_the_resolved_source_profile(self) -> None:
        core = Percept.create(vision=RecordingVision(), discover_plugins=False)
        receipt = apply_watch_plan(
            core,
            WatchPlan(id="audio", condition="someone speaks", sense="audio"),
            source_profile=_profile("video-only"),
        )
        self.assertEqual(receipt.status, "rejected")
        self.assertEqual(
            receipt.dropped_fields[0].code,
            "source_profile_modality_mismatch",
        )
        self.assertEqual(core.goals.all(), [])

    def test_default_vision_modality_must_exist_on_the_source_profile(self) -> None:
        core = Percept.create(vision=RecordingVision(), discover_plugins=False)
        receipt = apply_watch_plan(
            core,
            WatchPlan(id="implicit-video", condition="someone speaks"),
            source_profile=_profile(
                "audio-only",
                modalities=frozenset({"audio"}),
            ),
        )
        self.assertEqual(receipt.status, "rejected")
        self.assertEqual(
            receipt.dropped_fields[0].code,
            "source_profile_modality_mismatch",
        )

    def test_audio_only_compiles_with_an_exact_trusted_backend(self) -> None:
        core = Percept.create(vision=FakeVision(), discover_plugins=False)
        receipt = apply_watch_plan(
            core,
            WatchPlan(id="audio", condition="a sound matters", sense="audio"),
        )
        self.assertEqual(receipt.status, "applied")
        self.assertIn("audio", core._audio_judge_bindings)

    def test_fused_refuses_a_backend_that_discards_audio(self) -> None:
        for backend_type in (AnthropicVision, OpenRouterVision):
            with self.subTest(backend=backend_type.__name__):
                core = Percept.create(
                    vision=backend_type(client=object()),
                    discover_plugins=False,
                )
                receipt = apply_watch_plan(
                    core,
                    WatchPlan(id="fused", condition="a sound matters", sense="fused"),
                )
                self.assertEqual(receipt.status, "rejected")
                self.assertEqual(
                    receipt.dropped_fields[0].code,
                    "audio_input_backend_unsupported",
                )
                self.assertEqual(core.goals.all(), [])

    def test_gemini_and_scripted_accept_fused_before_any_provider_call(self) -> None:
        backends = (
            GeminiVision(client=object()),
            FakeVision(),
        )
        for backend in backends:
            with self.subTest(backend=type(backend).__name__):
                kwargs = {"mode": "offline"} if isinstance(backend, FakeVision) else {}
                core = Percept.create(
                    vision=backend,
                    discover_plugins=False,
                    **kwargs,
                )
                receipt = apply_watch_plan(
                    core,
                    WatchPlan(id="fused", condition="a sound matters", sense="fused"),
                )
                self.assertEqual(receipt.status, "applied")
                self.assertEqual(
                    program_of(core.goals.get("fused")).perception.modality,
                    "fused",
                )

    def test_unverified_custom_backend_is_also_refused_for_fused(self) -> None:
        core = Percept.create(
            vision=VisionOnlyRecording(),
            discover_plugins=False,
        )
        receipt = apply_watch_plan(
            core,
            WatchPlan(id="fused", condition="a sound matters", sense="fused"),
        )
        self.assertEqual(receipt.status, "rejected")
        self.assertEqual(
            receipt.dropped_fields[0].code,
            "audio_input_backend_unsupported",
        )
        self.assertEqual(core.goals.all(), [])

    def test_reserved_but_inert_typed_fields_are_rejected(self) -> None:
        fields = {
            "admission": "custom-gate",
            "evidence": "clip",
            "extractors": ("ocr",),
            "uncertainty": "always-ask",
        }
        for field, value in fields.items():
            with self.subTest(field=field):
                core = Percept.create(vision=RecordingVision(), discover_plugins=False)
                receipt = apply_watch_plan(
                    core,
                    WatchPlan(id=field, condition="condition", **{field: value}),
                )
                self.assertEqual(receipt.status, "rejected")
                self.assertEqual(
                    receipt.dropped_fields[0].code,
                    "unsupported_watchplan_field",
                )
                self.assertEqual(core.goals.all(), [])

    def test_compiled_binding_is_released_with_the_goal(self) -> None:
        core = Percept.create(vision=RecordingVision(), discover_plugins=False)
        receipt = apply_watch_plan(
            core,
            WatchPlan(id="g", condition="a person enters", sense="video"),
        )
        self.assertEqual(receipt.status, "applied")
        self.assertIn("g", core._watch_runtime_bindings)
        self.assertIn("g", core._watch_runtime_keys)
        core.apply_goal_ops([GoalOp(op="cancel", goal_id="g")])
        self.assertNotIn("g", core._watch_runtime_bindings)
        self.assertNotIn("g", core._watch_runtime_keys)

    def test_idempotency_includes_the_resolved_configuration(self) -> None:
        core = Percept.create(vision=RecordingVision(), discover_plugins=False)
        plan = WatchPlan(id="g", condition="a person enters", sense="video")
        strict = _profile("strict", stat="mean", threshold=0.9)
        local = _profile("local", stat="block_max", threshold=0.2)

        first = apply_watch_plan(core, plan, source_profile=strict)
        first_revision = core.goals.revision
        first_binding = core._watch_runtime_bindings["g"]
        duplicate = apply_watch_plan(core, plan, source_profile=strict)
        self.assertEqual(duplicate, first)
        self.assertEqual(core.goals.revision, first_revision)
        self.assertIs(core._watch_runtime_bindings["g"], first_binding)

        reconfigured = apply_watch_plan(core, plan, source_profile=local)
        self.assertEqual(reconfigured.status, "rejected")
        self.assertEqual(
            reconfigured.dropped_fields[0].code,
            "active_goal_id_reconfiguration_unsupported",
        )
        self.assertEqual(core.goals.revision, first_revision)
        self.assertIs(core._watch_runtime_bindings["g"], first_binding)

    def test_changed_same_id_plan_cannot_inherit_live_cognitive_state(self) -> None:
        core = Percept.create(vision=RecordingVision(), discover_plugins=False)
        original = WatchPlan(id="g", condition="a person enters", sense="video")
        self.assertEqual(apply_watch_plan(core, original).status, "applied")
        goal_before = core.goals.get("g")
        binding_before = core._watch_runtime_bindings["g"]
        ctx_before = core.executor.ctx("g")
        ctx_before.process_log = [{"step": "existing episode state"}]
        revision_before = core.goals.revision

        receipt = apply_watch_plan(
            core,
            WatchPlan(id="g", condition="a different person leaves", sense="video"),
        )

        self.assertEqual(receipt.status, "rejected")
        self.assertEqual(
            receipt.dropped_fields[0].code,
            "active_goal_id_reconfiguration_unsupported",
        )
        self.assertIs(core.goals.get("g"), goal_before)
        self.assertIs(core._watch_runtime_bindings["g"], binding_before)
        self.assertIs(core.executor.ctx("g"), ctx_before)
        self.assertEqual(ctx_before.process_log, [{"step": "existing episode state"}])
        self.assertEqual(core.goals.revision, revision_before)

    def test_public_same_id_goal_replacement_and_executable_update_fail_closed(self) -> None:
        core = Percept.create(vision=RecordingVision(), discover_plugins=False)
        original = Goal(id="g", condition="original", say="original")
        core.add_goal(original)
        stored = core.goals.get("g")

        with self.assertRaisesRegex(ValueError, "same-id replacement"):
            core.add_goal(Goal(id="g", condition="replacement", say="replacement"))
        with self.assertRaisesRegex(ValueError, "lifetime-unique"):
            core.apply_goal_ops([
                GoalOp(
                    op="create",
                    goal=Goal(id="g", condition="language replacement", say="replacement"),
                )
            ])
        with self.assertRaisesRegex(ValueError, "only the non-executable 'say'"):
            core.apply_goal_ops([
                GoalOp(op="update", goal_id="g", fields={"condition": "changed"})
            ])

        self.assertIs(core.goals.get("g"), stored)
        self.assertEqual(stored.condition, "original")

    def test_watch_profile_cadence_and_plan_mode_reach_the_armed_watch(self) -> None:
        profile = WatchProfile(
            name="dynamic-fixed",
            environment="fixed_camera",
            goal_type="dynamic",
            state="transition",
            plan_mode="window",
            cadence=CadenceProfile(
                base_fps=2.0,
                max_fps=3.0,
                escalate_threshold=0.4,
            ),
            gate=GateProfile(require_false_first=True),
        )
        registry = WatchProfileRegistry()
        registry.register(profile)
        configured = resolve_watch_profile(profile)
        core = Percept.create(
            vision=RecordingVision(),
            settings=configured.settings,
            sense_policy=configured.sense_policy,
            plan_mode=configured.plan_mode,
            discover_plugins=False,
        )
        receipt = apply_watch_plan(
            core,
            WatchPlan(id="g", condition="a handoff finishes", sense="video"),
            situation=("dynamic", "fixed", "transition"),
            registry=registry,
        )
        self.assertEqual(receipt.status, "applied")
        goal = core.goals.get("g")
        self.assertEqual(core._plan_mode_for_goal(goal), "window")
        self.assertEqual(goal.state["sense"]["base_fps"], 2.0)
        self.assertEqual(goal.state["sense"]["max_fps"], 3.0)

    def test_apply_inherits_the_core_source_profile_and_its_runtime_contract(self) -> None:
        profile = _profile(
            "core-selected",
            stat="block_max",
            threshold=0.23,
            timebase="wall",
            coverage_floor=2.0,
        )
        core = Percept.create(
            vision=RecordingVision(),
            source_profile=profile,
            discover_plugins=False,
        )
        receipt = apply_watch_plan(
            core,
            WatchPlan(id="g", condition="a person enters", sense="video"),
        )
        self.assertEqual(receipt.status, "applied")
        compiled, bound = core._watch_runtime_bindings["g"]
        self.assertIs(bound, profile)
        self.assertEqual(compiled.timebase, "wall")
        self.assertEqual(compiled.coverage_floor, 2.0)
        self.assertEqual(compiled.watchspec_kwargs["salience_stat"], "block_max")

    def test_apply_rejects_a_sense_missing_from_the_core_source_profile(self) -> None:
        core = Percept.create(
            vision=RecordingVision(),
            source_profile=_profile(
                "core-audio-only",
                modalities=frozenset({"audio"}),
            ),
            discover_plugins=False,
        )
        receipt = apply_watch_plan(
            core,
            WatchPlan(id="g", condition="a person enters", sense="video"),
        )
        self.assertEqual(receipt.status, "rejected")
        self.assertEqual(
            receipt.dropped_fields[0].code,
            "source_profile_modality_mismatch",
        )

    def test_profile_name_collision_is_rejected_instead_of_guessing_a_profile_kind(self) -> None:
        name = "p1-profile-kind-collision"
        source_profile = _profile(name)
        watch_profile = WatchProfile(
            name=name,
            environment="fixed",
            goal_type="dynamic",
            state="transition",
        )
        registry = WatchProfileRegistry()
        registry.register(watch_profile)
        register_source_profile(source_profile)
        try:
            core = Percept.create(
                vision=RecordingVision(),
                source_profile=source_profile,
                discover_plugins=False,
            )
            receipt = apply_watch_plan(
                core,
                WatchPlan(id="g", condition="a person enters", profile=name),
                registry=registry,
            )
            self.assertEqual(receipt.status, "rejected")
            self.assertEqual(
                receipt.dropped_fields[0].code,
                "ambiguous_profile_kind",
            )
        finally:
            DEFAULT_SOURCE_REGISTRY.unregister(name)

    def test_explicit_source_profile_disambiguates_a_same_named_watch_profile(self) -> None:
        profile = _profile("p1-explicit-source-kind")
        registry = WatchProfileRegistry()
        registry.register(WatchProfile(
            name=profile.name,
            environment="fixed",
            goal_type="dynamic",
            state="transition",
        ))
        core = Percept.create(vision=RecordingVision(), discover_plugins=False)
        receipt = apply_watch_plan(
            core,
            WatchPlan(id="g", condition="a person enters", profile=profile.name),
            source_profile=profile,
            registry=registry,
        )
        self.assertEqual(receipt.status, "applied")
        self.assertIs(core._watch_runtime_bindings["g"][1], profile)


class PerWatchRuntimeIsolationTests(unittest.TestCase):
    def test_duck_typed_perceive_without_goal_ids_runs_inline_and_async(self) -> None:
        for dispatch in ("inline", "async"):
            with self.subTest(dispatch=dispatch):
                core = Percept.create(vision=RecordingVision(), discover_plugins=False)
                core.add_goal(Goal(id="g", condition="a person enters", say="person"))
                calls = []

                async def legacy_perceive(
                    frame,
                    source="camera",
                    *,
                    t=None,
                    expected_goals_revision=None,
                ):
                    calls.append((frame, source, t, expected_goals_revision))
                    return [
                        FireEvent(
                            "other",
                            "fire",
                            "not admitted",
                            verdict=Verdict(True, 1.0, "legacy result"),
                        )
                    ]

                core.perceive_judged = legacy_perceive
                incidents = asyncio.run(
                    _collect(
                        core,
                        [
                            (0.0, STILL, "prime"),
                            (1.0, gray_frame(W, H, 255), "changed"),
                        ],
                        judge_dispatch=dispatch,
                    )
                )
                self.assertEqual(len(calls), 1)
                self.assertEqual(incidents, [])

    def test_different_source_profiles_admit_only_the_matching_watch(self) -> None:
        for dispatch in ("inline", "async"):
            with self.subTest(dispatch=dispatch):
                vision = RecordingVision()
                core = Percept.create(vision=vision, discover_plugins=False)
                strict = _profile("strict-mean", stat="mean", threshold=0.9)
                local = _profile("local-block", stat="block_max", threshold=0.2)
                self.assertEqual(
                    apply_watch_plan(
                        core,
                        WatchPlan(id="strict", condition="strict condition", sense="video"),
                        source_profile=strict,
                    ).status,
                    "applied",
                )
                self.assertEqual(
                    apply_watch_plan(
                        core,
                        WatchPlan(id="local", condition="local condition", sense="video"),
                        source_profile=local,
                    ).status,
                    "applied",
                )

                source = [(0.0, STILL, "prime"), (1.0, _local_motion(), "changed")]
                incidents = asyncio.run(
                    _collect(core, source, judge_dispatch=dispatch)
                )

                self.assertEqual(incidents, [])
                self.assertEqual(
                    [condition for condition, _frame in vision.judged],
                    ["local condition"],
                )
                self.assertIn(("local",), vision.detected)
                self.assertNotIn(("strict", "local"), vision.detected)

    def test_source_key_filters_gate_and_judge(self) -> None:
        vision = RecordingVision()
        core = Percept.create(vision=vision, discover_plugins=False)
        for goal_id, source in (("a", "camera-a"), ("b", "camera-b")):
            receipt = apply_watch_plan(
                core,
                WatchPlan(
                    id=goal_id,
                    condition=f"condition {goal_id}",
                    source=source,
                    sense="video",
                ),
                source_profile=_profile(f"profile-{goal_id}", threshold=0.1),
            )
            self.assertEqual(receipt.status, "applied")

        asyncio.run(
            _collect(
                core,
                [(0.0, STILL, "prime"), (1.0, gray_frame(W, H, 255), "changed")],
                gate_source="camera-a",
                judge_dispatch="inline",
            )
        )
        self.assertEqual(
            [condition for condition, _frame in vision.judged],
            ["condition a"],
        )
        self.assertTrue(all(ids == ("a",) for ids in vision.detected))

    def test_source_key_also_scopes_boundary_coverage_and_deadline_incidents(self) -> None:
        core = Percept.create(vision=RecordingVision(), discover_plugins=False)
        core.add_goal(Goal(id="a", condition="camera a", say="a", source="camera-a"))
        other = Goal(
            id="b",
            condition="camera b",
            say="b",
            source="camera-b",
            state={
                "type": "process",
                "rearm": "deadline",
                "deadline_at": 1.0,
                "miss_cost": "severe",
            },
        )
        core.add_goal(other)
        core.last_supported["b"] = 0.0
        other_ctx = core.executor.ctx("b")
        other_ctx.process_log = [{"step": "camera b step"}]
        other_ctx.last_step_at = 0.0

        incidents = asyncio.run(
            _collect(
                core,
                [(10.0, STILL, "quiet")],
                gate_source="camera-a",
                coverage_staleness_s=1.0,
                episode_inactivity_s=1.0,
                judge_dispatch="inline",
            )
        )
        self.assertFalse(any(incident.goal_id == "b" for incident in incidents))
        self.assertEqual(other_ctx.process_log, [{"step": "camera b step"}])

    def test_multisource_legacy_callbacks_refuse_before_spend_or_mutation(self) -> None:
        for dispatch in ("inline", "async"):
            with self.subTest(dispatch=dispatch):
                core = Percept.create(vision=RecordingVision(), discover_plugins=False)
                core.add_goal(Goal(
                    id="a", condition="camera a", say="a", source="camera-a"
                ))
                core.add_goal(Goal(
                    id="b", condition="camera b", say="b", source="camera-b"
                ))
                wrong = FireEvent(
                    "b",
                    "fire",
                    "wrong source",
                    verdict=Verdict(True, 1.0, "malicious legacy callback"),
                )
                effects = []

                async def legacy_boundary(*, t=None, inactivity_s=None):
                    effects.append("boundary")
                    return [wrong]

                def legacy_coverage(*, t=None, staleness_s=None):
                    effects.append("coverage")
                    return [wrong]

                def legacy_deadlines(*, t=None):
                    effects.append("deadlines")
                    return [wrong]

                async def legacy_perceive(
                    frame,
                    source="camera",
                    *,
                    t=None,
                    expected_goals_revision=None,
                ):
                    effects.append("perceive")
                    return [wrong]

                core.check_episode_boundary = legacy_boundary
                core.check_observability = legacy_coverage
                core.check_deadlines = legacy_deadlines
                core.perceive_judged = legacy_perceive

                with self.assertRaisesRegex(ValueError, "spanning multiple source keys"):
                    asyncio.run(
                        _collect(
                            core,
                            [
                                (0.0, STILL, "prime"),
                                (1.0, gray_frame(W, H, 255), "changed"),
                            ],
                            gate_source="camera-a",
                            judge_dispatch=dispatch,
                        )
                    )
                self.assertEqual(effects, [])

    def test_regate_rejects_new_multisource_legacy_topology_before_callbacks(self) -> None:
        core = Percept.create(vision=RecordingVision(), discover_plugins=False)
        core.add_goal(Goal(id="a", condition="camera a", say="a", source="camera-a"))
        effects = []

        async def legacy_perceive(
            frame,
            source="camera",
            *,
            t=None,
            expected_goals_revision=None,
        ):
            effects.append("perceive")
            return []

        async def legacy_boundary(*, t=None, inactivity_s=None):
            effects.append("boundary")
            return []

        core.perceive_judged = legacy_perceive
        core.check_episode_boundary = legacy_boundary

        async def changing_source():
            yield (0.0, STILL, "prime")
            effects.clear()
            core.goals.create(
                Goal(id="b", condition="camera b", say="b", source="camera-b")
            )
            yield (1.0, gray_frame(W, H, 255), "changed")

        with self.assertRaisesRegex(ValueError, "spanning multiple source keys"):
            asyncio.run(
                _collect(
                    core,
                    changing_source(),
                    gate_source="camera-a",
                    judge_dispatch="inline",
                )
            )
        self.assertEqual(effects, [])

    def test_per_watch_coast_does_not_wake_a_sparser_watch(self) -> None:
        vision = RecordingVision()
        core = Percept.create(vision=vision, discover_plugins=False)
        profiles = (
            ("dense", _profile("dense-profile", threshold=0.99, coast_interval_s=0.5)),
            ("sparse", _profile("sparse-profile", threshold=0.99, coast_interval_s=2.0)),
        )
        for goal_id, profile in profiles:
            receipt = apply_watch_plan(
                core,
                WatchPlan(id=goal_id, condition=goal_id, sense="video"),
                source_profile=profile,
            )
            self.assertEqual(receipt.status, "applied")

        source = [(i * 0.5, STILL, f"f{i}") for i in range(5)]
        asyncio.run(_collect(core, source, judge_dispatch="inline"))
        calls = [condition for condition, _frame in vision.judged]
        self.assertEqual(calls.count("dense"), 5)
        self.assertEqual(calls.count("sparse"), 2)

    def test_compiled_fused_watch_delivers_audio_to_the_judge(self) -> None:
        vision = FakeVision(script={"changed|audio": Verdict(satisfied=True, confidence=0.99)})
        core = Percept.create(vision=vision, discover_plugins=False)
        receipt = apply_watch_plan(
            core,
            WatchPlan(id="fused", condition="speaker points", sense="fused"),
            source_profile=_profile(
                "av",
                threshold=0.1,
                modalities=frozenset({"video", "audio"}),
            ),
        )
        self.assertEqual(receipt.status, "applied")
        audio = [
            (
                0.5,
                AudioFrame(pcm=b"\x01\x00" * 100, sample_rate=16_000),
                0.5,
            )
        ]
        incidents = asyncio.run(
            _collect(
                core,
                [(0.0, STILL, "prime"), (1.0, gray_frame(W, H, 255), "changed")],
                audio_source=audio,
                audio_gate=None,
                judge_dispatch="inline",
            )
        )
        # This script has no plain "changed" verdict; a fire proves the exact audio-bearing route
        # selected the "changed|audio" token without adding an always-on call log to FakeVision.
        self.assertEqual(len(incidents), 1)
        self.assertEqual(core.metrics.get("judge.latency.count"), 1)

    def test_compiled_fused_requires_an_audio_source(self) -> None:
        core = Percept.create(vision=FakeVision(), discover_plugins=False)
        self.assertEqual(
            apply_watch_plan(
                core,
                WatchPlan(id="fused", condition="a sound matters", sense="fused"),
            ).status,
            "applied",
        )
        with self.assertRaisesRegex(ValueError, "require audio_source"):
            asyncio.run(_collect(core, [], judge_dispatch="inline"))

    def test_compiled_fused_watch_skips_and_reports_an_empty_audio_window(self) -> None:
        vision = FakeVision()
        core = Percept.create(vision=vision, discover_plugins=False)
        self.assertEqual(
            apply_watch_plan(
                core,
                WatchPlan(id="fused", condition="speaker points", sense="fused"),
            ).status,
            "applied",
        )
        health = []
        asyncio.run(
            _collect(
                core,
                [(0.0, STILL, "prime"), (1.0, gray_frame(W, H, 255), "changed")],
                audio_source=[],
                audio_gate=None,
                on_health=health.append,
                judge_dispatch="inline",
            )
        )
        self.assertEqual(core.metrics.get("judge.latency.count"), 0)
        unavailable = [event for event in health if event.kind == "audio_evidence_unavailable"]
        self.assertEqual(len(unavailable), 1)
        self.assertEqual(unavailable[0].data["goal_id"], "fused")

    def test_wrong_day_audio_onset_cannot_admit_a_fused_watch(self) -> None:
        vision = FakeVision()
        core = Percept.create(vision=vision, discover_plugins=False)
        receipt = apply_watch_plan(
            core,
            WatchPlan(
                id="fused",
                condition="someone calls for help",
                sense="fused",
                schedule={"days": [1]},
            ),
            source_profile=_profile(
                "scheduled-av",
                threshold=0.99,
                modalities=frozenset({"video", "audio"}),
            ),
        )
        self.assertEqual(receipt.status, "applied")
        audio = [
            (0.1, AudioFrame(pcm=b"\x00\x00", sample_rate=16_000), 0.0),
            (1.1, AudioFrame(pcm=b"\xff\x7f", sample_rate=16_000), 0.5),
        ]
        stats = RuntimeStats()
        asyncio.run(
            _collect(
                core,
                [(float(i), STILL, f"f{i}") for i in range(5)],
                audio_source=audio,
                audio_settle_s=0.0,
                clock=lambda: CivilTime(weekday=2, minute_of_day=600),
                stats=stats,
                judge_dispatch="inline",
            )
        )
        self.assertEqual(stats.escalations, 0)
        self.assertEqual(core.metrics.get("judge.latency.count"), 0)

    def test_audio_only_watch_arms_only_with_an_audio_source_profile(self) -> None:
        core = Percept.create(vision=FakeVision(), discover_plugins=False)
        receipt = apply_watch_plan(
            core,
            WatchPlan(id="audio", condition="someone speaks", sense="audio"),
            source_profile=_profile(
                "audio",
                threshold=0.1,
                modalities=frozenset({"audio"}),
            ),
        )
        self.assertEqual(receipt.status, "applied")
        self.assertEqual(
            program_of(core.goals.get("audio")).perception.modality,
            "audio",
        )

    def test_profile_identity_defaults_remain_bound_to_their_own_watch(self) -> None:
        core = Percept.create(vision=FakeVision(), discover_plugins=False)
        content = _profile("content-profile", identity="content_key")
        track = _profile("track-profile", identity="track_continuity")
        apply_watch_plan(
            core,
            WatchPlan(id="content", condition="a document is visible"),
            source_profile=content,
        )
        apply_watch_plan(
            core,
            WatchPlan(id="track", condition="a person is visible"),
            source_profile=track,
        )
        self.assertIsNone(core.source_profile)
        self.assertEqual(
            core._resolved_identity_policy(core.goals.get("content")),
            "content_key",
        )
        self.assertEqual(
            core._resolved_identity_policy(core.goals.get("track")),
            "track_continuity",
        )

    def test_compiled_source_timebase_is_used_and_conflicts_fail_closed(self) -> None:
        core = Percept.create(vision=FakeVision(), discover_plugins=False)
        receipt = apply_watch_plan(
            core,
            WatchPlan(id="wall", condition="a person enters", sense="video"),
            source_profile=_profile("wall-profile", timebase="wall"),
        )
        self.assertEqual(receipt.status, "applied")
        asyncio.run(_collect(core, [], judge_dispatch="inline"))
        # Deterministic iterable sources always use their supplied media timestamps, but the
        # effective record still proves the compiled profile supplied the requested policy.
        self.assertEqual(core.effective_config.video_timebase.requested, "wall")
        self.assertEqual(core.effective_config.video_timebase.resolved, "media")

        conflict = Percept.create(vision=RecordingVision(), discover_plugins=False)
        for goal_id, timebase in (("wall", "wall"), ("media", "media")):
            receipt = apply_watch_plan(
                conflict,
                WatchPlan(id=goal_id, condition=goal_id, sense="video"),
                source_profile=_profile(f"{goal_id}-profile", timebase=timebase),
            )
            self.assertEqual(receipt.status, "applied")
        with self.assertRaisesRegex(ValueError, "incompatible compiled timebases"):
            asyncio.run(_collect(conflict, [], judge_dispatch="inline"))

    def test_different_fused_schedules_fail_closed_on_one_shared_audio_worker(self) -> None:
        core = Percept.create(vision=FakeVision(), discover_plugins=False)
        apply_watch_plan(
            core,
            WatchPlan(
                id="fused-a",
                condition="someone speaks",
                sense="fused",
                schedule={"days": [1]},
            ),
            source_profile=_profile(
                "fused-a-profile",
                threshold=0.1,
                modalities=frozenset({"video", "audio"}),
            ),
        )
        apply_watch_plan(
            core,
            WatchPlan(
                id="fused-b",
                condition="a speaker points",
                sense="fused",
                schedule={"days": [2]},
            ),
            source_profile=_profile(
                "fused-b-profile",
                threshold=0.1,
                modalities=frozenset({"video", "audio"}),
            ),
        )
        with self.assertRaisesRegex(ValueError, "shared by this worker"):
            asyncio.run(_collect(core, [], audio_source=[]))
