"""Product-path contract for temporal, occurrence-aware ``rearm=each_event`` Watches.

An event Watch is not a level poll.  The judge sees an ordered clip and reports the latest
event phase.  The cognition layer owns the occurrence lifecycle:

    pre -> during -> post == one occurrence
    ... -> during          == a later occurrence

Repeated observations inside one ``during``/``post`` episode must not refire, while a later
phase transition must re-arm without depending on a noisy one-frame false verdict.
"""

from __future__ import annotations

import asyncio
import unittest
from dataclasses import replace

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.behavior import program_of
from percept.contracts.watchplan import WatchPlan
from percept.fused_sync import _FusedSyncPolicy, _FusedSynchronizer
from percept.watchplan_apply import apply_watch_plan


def _verdict(phase: str, *, confidence: float = 0.95) -> Verdict:
    return Verdict(
        satisfied=phase in {"during", "post"},
        confidence=confidence,
        evidence=f"phase={phase}",
        phase=phase,
        action_present=phase == "during",
    )


class _PhaseVision:
    def __init__(self, script: dict[str, Verdict]) -> None:
        self.script = script
        self.calls: list[dict] = []

    async def detect(self, frame, goals):
        return []

    async def detect_events(self, frames, goals):
        return []

    async def judge(self, condition, frame, **kwargs):
        self.calls.append(
            {
                "condition": condition,
                "frame": frame,
                "frames": kwargs.get("frames"),
                "context": kwargs.get("context"),
            }
        )
        return self.script[str(frame)]


def _goal() -> Goal:
    return Goal(
        id="action",
        condition="someone completes one discrete putting action",
        say="putting action",
        state={"type": "watch", "count": 0, "rearm": "each_event"},
    )


async def _run(phases: list[str], *, confidence: dict[int, float] | None = None):
    confidence = confidence or {}
    script = {
        f"f{i}": _verdict(phase, confidence=confidence.get(i, 0.95))
        for i, phase in enumerate(phases)
    }
    vision = _PhaseVision(script)
    core = Percept.create(vision=vision, discover_plugins=False)
    core.add_goal(_goal())
    emitted = []
    for i in range(len(phases)):
        emitted.extend(await core.perceive_judged(f"f{i}", t=float(i)))
    return core, vision, emitted


class EachEventProgramTests(unittest.TestCase):
    def test_each_event_compiles_to_ordered_temporal_evidence(self):
        program = program_of(_goal())
        self.assertTrue(program.key.each_event)
        self.assertEqual(program.perception.temporal, "strip")
        self.assertGreaterEqual(program.perception.n_frames, 3)

    def test_authored_temporal_shape_still_wins(self):
        goal = _goal()
        goal.state["perception"] = {
            "temporal": "window",
            "window_s": 3.0,
            "sampling": "uniform",
        }
        program = program_of(goal)
        self.assertEqual(program.perception.temporal, "window")
        self.assertEqual(program.perception.window_s, 3.0)

    def test_explicit_content_or_claim_episode_policy_is_rejected_at_arm(self):
        for identity_policy in ("content_key", "claim_slot"):
            with self.subTest(identity_policy=identity_policy):
                core = Percept.create(
                    vision=_PhaseVision({}),
                    discover_plugins=False,
                )
                goal = _goal()
                goal.state["identity_policy"] = identity_policy
                with self.assertRaisesRegex(
                    ValueError,
                    "owns occurrence episode identity",
                ):
                    core.add_goal(goal)

    def test_source_profile_default_does_not_add_a_second_episode_fold(self):
        from percept.source_profiles_default import SCREEN_DOCUMENT

        core = Percept.create(
            vision=_PhaseVision({}),
            discover_plugins=False,
            source_profile=SCREEN_DOCUMENT,
        )
        goal = core.add_goal(_goal())
        self.assertEqual(core._resolved_identity_policy(goal), "content_key")
        self.assertIsNone(core._document_episode_fold(goal, _verdict("during"), 1.0))

    def test_restore_loudly_announces_session_local_occurrence_reset(self):
        source = Percept.create(
            vision=_PhaseVision({}),
            discover_plugins=False,
        )
        source.add_goal(_goal())
        restored = Percept.create(
            vision=_PhaseVision({}),
            discover_plugins=False,
        ).restore(source.snapshot())
        self.assertEqual(len(restored._restore_health_log), 1)
        kind, data = restored._restore_health_log[0]
        self.assertEqual(kind, "episode_memory_reset")
        self.assertEqual(data["occurrence_goals"], 1)
        self.assertEqual(data["goal_ids"], ["action"])


class EachEventProductPathTests(unittest.TestCase):
    def test_one_continuous_occurrence_sampled_fifty_times_fires_once(self):
        phases = ["pre"] + ["during"] * 50 + ["post"] * 4
        _core, _vision, emitted = asyncio.run(_run(phases))
        self.assertEqual([event.action for event in emitted], ["fire"])

    def test_two_phase_separated_occurrences_fire_twice(self):
        phases = [
            "pre",
            "during",
            "during",
            "post",
            "post",
            "pre",
            "pre",
            "pre",
            "during",
            "during",
            "post",
        ]
        _core, _vision, emitted = asyncio.run(_run(phases))
        self.assertEqual([event.action for event in emitted], ["fire", "fire"])
        self.assertEqual(len({event.revision_id for event in emitted}), 2)
        self.assertTrue(all(event.revision_id for event in emitted))

    def test_post_only_evidence_can_recover_a_missed_short_occurrence_once(self):
        _core, _vision, emitted = asyncio.run(_run(["pre", "post", "post", "pre"]))
        self.assertEqual([event.action for event in emitted], ["fire"])

    def test_overlapping_window_post_to_during_regression_does_not_refire(self):
        _core, _vision, emitted = asyncio.run(
            _run(["pre", "post", "during", "post", "during"])
        )
        self.assertEqual([event.action for event in emitted], ["fire"])

    def test_brief_pre_gap_inside_evidence_horizon_does_not_refire(self):
        _core, _vision, emitted = asyncio.run(
            _run(["pre", "during", "pre", "during", "post"])
        )
        self.assertEqual([event.action for event in emitted], ["fire"])

    def test_low_confidence_during_does_not_consume_the_occurrence(self):
        _core, _vision, emitted = asyncio.run(
            _run(["pre", "during", "during", "post"], confidence={1: 0.2})
        )
        self.assertEqual([event.action for event in emitted], ["ask", "fire"])

    def test_each_event_judge_receives_ordered_frames_and_control_context(self):
        _core, vision, _emitted = asyncio.run(_run(["pre", "during", "post"]))
        self.assertIsNone(vision.calls[0]["frames"])
        self.assertEqual(vision.calls[1]["frames"], ["f0"])
        self.assertEqual(vision.calls[2]["frames"], ["f0", "f1"])
        self.assertTrue(vision.calls[-1]["context"]["event_chain"])

    def test_missing_event_metadata_refuses_to_guess(self):
        vision = _PhaseVision(
            {"f0": Verdict(satisfied=True, confidence=0.99, evidence="single-frame level guess")}
        )
        core = Percept.create(vision=vision, discover_plugins=False)
        core.add_goal(_goal())
        emitted = asyncio.run(core.perceive_judged("f0", t=0.0))
        self.assertEqual([event.action for event in emitted], ["ask"])
        self.assertEqual(core.last_satisfied, {})

    def test_inconsistent_event_tuple_refuses_to_guess(self):
        vision = _PhaseVision(
            {
                "f0": Verdict(
                    satisfied=True,
                    confidence=0.99,
                    evidence="contradictory structured reply",
                    phase="during",
                    action_present=False,
                )
            }
        )
        core = Percept.create(vision=vision, discover_plugins=False)
        core.add_goal(_goal())
        emitted = asyncio.run(core.perceive_judged("f0", t=0.0))
        self.assertEqual([event.action for event in emitted], ["ask"])
        self.assertEqual(core.last_satisfied, {})

    def test_trace_preserves_raw_provider_tuple_and_occurrence_episode(self):
        traces = []
        vision = _PhaseVision({"f0": _verdict("during")})
        core = Percept.create(
            vision=vision,
            discover_plugins=False,
            on_trace=traces.append,
        )
        core.add_goal(_goal())
        emitted = asyncio.run(core.perceive_judged("f0", t=0.0))
        self.assertEqual([event.action for event in emitted], ["fire"])

        finished = next(event for event in traces if event.stage == "judge.finished")
        self.assertEqual(finished.data["phase"], "during")
        self.assertIs(finished.data["action_present"], True)
        classified = next(event for event in traces if event.stage == "gate.classified")
        self.assertEqual(classified.data["event_phase"], "during")
        self.assertIs(classified.data["event_action_present"], True)
        self.assertEqual(classified.data["event_occurrence_sequence"], 1)
        self.assertTrue(classified.data["event_episode_id"].startswith("action:event:"))
        self.assertIs(classified.data["event_episode_emitted"], True)
        self.assertIsNone(classified.data["event_pre_since"])
        self.assertEqual(classified.data["event_close_quiet_s"], 2.0)
        self.assertEqual(emitted[0].revision_id, classified.data["event_episode_id"] + "#r1")

    def test_trace_exposes_provider_claim_before_fail_closed_normalization(self):
        traces = []
        vision = _PhaseVision(
            {"f0": Verdict(satisfied=True, confidence=0.99, evidence="level-only answer")}
        )
        core = Percept.create(
            vision=vision,
            discover_plugins=False,
            on_trace=traces.append,
        )
        core.add_goal(_goal())
        emitted = asyncio.run(core.perceive_judged("f0", t=0.0))
        self.assertEqual([event.action for event in emitted], ["ask"])

        finished = next(event for event in traces if event.stage == "judge.finished")
        self.assertIs(finished.data["satisfied"], True)
        self.assertIsNone(finished.data["phase"])
        classified = next(event for event in traces if event.stage == "gate.classified")
        self.assertIs(classified.data["satisfied"], False)
        self.assertEqual(classified.data["confidence"], 0.0)
        self.assertEqual(classified.data["truth"], "unknown")

    def test_split_phase_owner_path_preserves_occurrence_semantics(self):
        phases = [
            "pre",
            "during",
            "during",
            "post",
            "pre",
            "pre",
            "pre",
            "during",
        ]
        vision = _PhaseVision(
            {f"f{i}": _verdict(phase) for i, phase in enumerate(phases)}
        )
        core = Percept.create(vision=vision, discover_plugins=False)
        core.add_goal(_goal())

        async def go():
            emitted = []
            for i in range(len(phases)):
                candidate = core.prepare_observation_candidate(
                    f"f{i}",
                    "camera",
                    t=float(i),
                    expected_goals_revision=core.goals.revision,
                    expected_source_generation=core.source_continuity_generation("camera"),
                    goal_ids=("action",),
                )
                self.assertIsNotNone(candidate)
                self.assertTrue(core.launch_observation_candidate(candidate))
                observation = await core.dispatch_observation_candidate(candidate)
                reduction = core.reduce_observation_completion(observation)
                job = core.prepare_frame_judgment(reduction, goal_id="action")
                self.assertIsNotNone(job)
                emitted.extend(
                    core.reduce_judgment_completion(await core.dispatch_judgment(job))
                )
            return emitted

        emitted = asyncio.run(go())
        self.assertEqual([event.action for event in emitted], ["fire", "fire"])
        self.assertTrue(all(call["context"]["event_chain"] for call in vision.calls))

    def test_default_watch_remains_level_rising_edge(self):
        vision = _PhaseVision(
            {
                "f0": Verdict(satisfied=True, confidence=0.95),
                "f1": Verdict(satisfied=True, confidence=0.95),
                "f2": Verdict(satisfied=False, confidence=0.95),
                "f3": Verdict(satisfied=True, confidence=0.95),
            }
        )
        core = Percept.create(vision=vision, discover_plugins=False)
        core.add_goal(Goal(id="level", condition="door open", say="open"))

        async def go():
            out = []
            for i in range(4):
                out.extend(await core.perceive_judged(f"f{i}", t=float(i)))
            return out

        emitted = asyncio.run(go())
        self.assertEqual([event.action for event in emitted], ["fire", "fire"])


class EachEventFusedProductPathTests(unittest.TestCase):
    def test_fused_reducer_uses_the_same_occurrence_lifecycle(self):
        from tests.test_fused_sync import _audio, _video
        from tests.test_watchplan_runtime_binding import _profile

        phases = [
            "pre",
            "during",
            "during",
            "post",
            "pre",
            "pre",
            "pre",
            "during",
        ]

        from percept.backends.fake import FakeVision

        vision = FakeVision(
            script={
                f"f{index}|audio": _verdict(phase)
                for index, phase in enumerate(phases)
            }
        )
        contexts = []

        # Keep the exact SDK-owned FakeVision class required by the trusted fused-input binding
        # while scripting the phase sequence locally. This proves wiring, never model acuity.
        core = Percept.create(
            vision=vision,
            discover_plugins=False,
            on_trace=lambda _event: None,
        )
        receipt = apply_watch_plan(
            core,
            WatchPlan(
                id="action",
                condition="someone performs one discrete occurrence of pointing",
                sense="fused",
                rearm="each_event",
            ),
            source_profile=_profile(
                "event-chain-av",
                threshold=0.1,
                modalities=frozenset({"video", "audio"}),
            ),
        )
        self.assertEqual(receipt.status, "applied")
        binding = core._audio_judge_bindings["action"]
        revision = core.goals.revision
        sync = _FusedSynchronizer(
            _FusedSyncPolicy(window_s=0.5, max_skew_s=0.5),
            watch_application=binding.application_key,
            watch_revision=revision,
        )

        emitted = []
        for index in range(len(phases)):
            at = float(index + 1)
            anchor = replace(
                _video(index, at, watermark=at + 1.0),
                payload=f"f{index}",
                byte_size=None,
                watch_application=binding.application_key,
                watch_revision=revision,
            )
            audio = replace(
                _audio(index, at - 0.5, at, watermark=at + 1.0),
                watch_application=binding.application_key,
                watch_revision=revision,
            )
            self.assertTrue(sync.accept(anchor))
            self.assertTrue(sync.accept(audio))
            job = core.prepare_fused_judgment(
                sync.bundle_for(anchor),
                goal_id="action",
                source="camera",
                source_t=at,
                goals_revision=revision,
                source_generation=core.source_continuity_generation("camera"),
                fused_generation=sync.generation,
            )
            self.assertIsNotNone(job)
            contexts.append(dict(job.context))
            completion = asyncio.run(core.dispatch_fused_judgment(job))
            emitted.extend(
                core.reduce_fused_completion(
                    completion,
                    live_fused_generation=sync.generation,
                )
            )

        self.assertEqual([event.action for event in emitted], ["fire", "fire"])
        self.assertEqual(len({event.revision_id for event in emitted}), 2)
        self.assertTrue(all(context["event_chain"] for context in contexts))
        records = core.drain_fused_trace_records()
        classified = [data for _trace, _t, stage, data in records if stage == "gate.classified"]
        self.assertEqual(
            [row["event_occurrence_sequence"] for row in classified],
            [0, 1, 1, 1, 1, 1, 1, 2],
        )


if __name__ == "__main__":
    unittest.main()
