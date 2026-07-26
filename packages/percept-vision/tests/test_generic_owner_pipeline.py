from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from percept import Goal, Percept
from percept.adapters import Box, Observation, Verdict
from percept.goals import GoalOp
from percept.harness.tier0 import gray_frame


class BlockingObservationVision:
    def __init__(self, observation: Observation) -> None:
        self.observation = observation
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def detect(self, frame, goals):
        self.started.set()
        await self.release.wait()
        return [self.observation]

    async def judge(self, condition, frame, **kwargs):
        return Verdict(False, 0.99, "not used")


def _core(vision) -> Percept:
    core = Percept.create(vision=vision, discover_plugins=False)
    core.add_goal(
        Goal(
            id="ordinary",
            condition="a person is present",
            say="seen",
            state={
                "type": "watch",
                "count": 0,
                "nested": {"labels": ["person"]},
            },
        )
    )
    return core


def test_observation_provider_is_state_free_and_inputs_are_deeply_read_only():
    async def drive():
        provider_observation = Observation(
            "person",
            Box(0.1, 0.1, 0.5, 0.5),
            0.99,
            attrs={"role": "original"},
        )
        vision = BlockingObservationVision(provider_observation)
        core = _core(vision)
        live_goal = core.goals.get("ordinary")
        candidate = core.prepare_observation_candidate(
            gray_frame(8, 8, 255),
            "camera",
            t=1.0,
            expected_goals_revision=core.goals.revision,
            expected_source_generation=core.source_continuity_generation("camera"),
            goal_ids=("ordinary",),
        )
        assert candidate is not None
        live_goal.state["nested"]["labels"].append("mutated-after-admission")
        assert candidate.goals[0].state["nested"]["labels"] == ("person",)
        with pytest.raises(TypeError):
            candidate.goals[0].state["new"] = "forbidden"

        assert core.launch_observation_candidate(candidate)
        assert not core.launch_observation_candidate(candidate)
        task = asyncio.create_task(core.dispatch_observation_candidate(candidate))
        await vision.started.wait()
        assert len(core._window("camera").frames()) == 1
        assert core.graph.in_focus() is None
        assert core.action_log.entries() == []
        assert core.last_supported == {}
        assert core.last_satisfied == {}

        vision.release.set()
        completion = await task
        provider_observation.attrs["role"] = "mutated-after-return"
        assert completion.observations[0].to_observation().attrs == {
            "role": "original"
        }
        assert len(core._window("camera").frames()) == 1

        reduction = core.reduce_observation_completion(completion)
        assert reduction.stale_reason is None
        assert reduction.failure is None
        assert reduction.focus is not None
        assert len(core._window("camera").frames()) == 1

    asyncio.run(drive())


def test_stale_observation_completion_is_semantically_inert():
    async def drive():
        observation = Observation(
            "person", Box(0.1, 0.1, 0.5, 0.5), 0.99
        )
        vision = BlockingObservationVision(observation)
        core = _core(vision)
        candidate = core.prepare_observation_candidate(
            gray_frame(8, 8, 255),
            "camera",
            t=1.0,
            expected_goals_revision=core.goals.revision,
            expected_source_generation=core.source_continuity_generation("camera"),
            goal_ids=("ordinary",),
        )
        assert core.launch_observation_candidate(candidate)
        task = asyncio.create_task(core.dispatch_observation_candidate(candidate))
        await vision.started.wait()
        vision.release.set()
        completion = await task
        core.advance_source_continuity("camera")
        reduction = core.reduce_observation_completion(completion)
        assert reduction.stale_reason == "source_generation"
        assert core._window("camera").frames() == []
        assert core.graph.in_focus() is None
        assert core.action_log.entries() == []

    asyncio.run(drive())


def test_observation_provider_failure_is_redacted_and_semantically_inert():
    secret = "SECRET-IN-DETECTOR-REQUEST"

    class FailingVision:
        async def detect(self, frame, goals):
            raise RuntimeError(f"https://user:{secret}@provider.invalid")

        async def judge(self, condition, frame, **kwargs):
            return Verdict(False, 0.99)

    async def drive():
        core = _core(FailingVision())
        candidate = core.prepare_observation_candidate(
            gray_frame(8, 8, 255),
            "camera",
            t=1.0,
            expected_goals_revision=core.goals.revision,
            expected_source_generation=core.source_continuity_generation("camera"),
            goal_ids=("ordinary",),
        )
        assert core.launch_observation_candidate(candidate)
        completion = await core.dispatch_observation_candidate(candidate)
        reduction = core.reduce_observation_completion(completion)
        assert reduction.failure is not None
        assert reduction.failure.kind == "observation_provider_failure"
        assert secret not in repr(completion)
        assert len(core._window("camera").frames()) == 1
        assert core.graph.in_focus() is None

    asyncio.run(drive())


def test_unlaunched_or_pending_observation_candidate_never_pushes_window():
    observation = Observation("person", Box(0.1, 0.1, 0.5, 0.5), 0.99)
    core = _core(BlockingObservationVision(observation))
    revision = core.goals.revision
    generation = core.source_continuity_generation("camera")
    active = core.prepare_observation_candidate(
        gray_frame(8, 8, 1),
        "camera",
        t=1.0,
        expected_goals_revision=revision,
        expected_source_generation=generation,
        goal_ids=("ordinary",),
    )
    pending = core.prepare_observation_candidate(
        gray_frame(8, 8, 2),
        "camera",
        t=2.0,
        expected_goals_revision=revision,
        expected_source_generation=generation,
        goal_ids=("ordinary",),
    )
    assert core._window("camera").frames() == []
    assert core.launch_observation_candidate(active)
    assert len(core._window("camera").frames()) == 1
    assert not core.launch_observation_candidate(pending)
    assert len(core._window("camera").frames()) == 1


def test_detector_frame_alias_cannot_rewrite_owner_window_evidence():
    class MutatingVision:
        async def detect(self, frame, goals):
            frame["nested"]["pixels"].append("provider-mutation")
            return []

        async def judge(self, condition, frame, **kwargs):
            return Verdict(False, 0.99)

    async def drive():
        core = _core(MutatingVision())
        caller_frame = {"nested": {"pixels": ["original"]}}
        candidate = core.prepare_observation_candidate(
            caller_frame,
            "camera",
            t=1.0,
            expected_goals_revision=core.goals.revision,
            expected_source_generation=core.source_continuity_generation("camera"),
            goal_ids=("ordinary",),
        )
        caller_frame["nested"]["pixels"].append("caller-mutation")
        assert core.launch_observation_candidate(candidate)
        completion = await core.dispatch_observation_candidate(candidate)
        reduction = core.reduce_observation_completion(completion)
        assert reduction.stale_reason is None
        assert core._window("camera").frames() == [
            (1.0, {"nested": {"pixels": ["original"]}})
        ]

    asyncio.run(drive())


def test_ordinary_frame_watch_provider_is_state_free_until_owner_reduction():
    class BlockingJudgeVision:
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def detect(self, frame, goals):
            return []

        async def judge(self, condition, frame, **kwargs):
            self.started.set()
            await self.release.wait()
            return Verdict(True, 0.99, "present")

    async def drive():
        vision = BlockingJudgeVision()
        core = _core(vision)
        revision = core.goals.revision
        generation = core.source_continuity_generation("camera")
        candidate = core.prepare_observation_candidate(
            "present",
            "camera",
            t=1.0,
            expected_goals_revision=revision,
            expected_source_generation=generation,
            goal_ids=("ordinary",),
        )
        assert core.launch_observation_candidate(candidate)
        observation = await core.dispatch_observation_candidate(candidate)
        reduction = core.reduce_observation_completion(observation)
        job = core.prepare_frame_judgment(reduction, goal_id="ordinary")
        assert job is not None
        task = asyncio.create_task(core.dispatch_judgment(job))
        await vision.started.wait()
        assert core.action_log.entries() == []
        assert core.last_supported == {}
        assert core.last_satisfied == {}
        assert core.gate.snapshot()["armed"] == []

        vision.release.set()
        completion = await task
        assert core.action_log.entries() == []
        events = core.reduce_judgment_completion(completion)
        assert [event.action for event in events] == ["fire"]
        assert core.last_supported == {"ordinary": 1.0}
        assert core.last_satisfied == {"ordinary": 1.0}
        assert len(core.action_log.entries()) == 1
        return events[0], core.action_log.entries()[0]

    split_event, split_record = asyncio.run(drive())

    from percept.backends.fake import FakeVision

    baseline = _core(
        FakeVision(
            script={"present": Verdict(True, 0.99, "present")}
        )
    )
    baseline_events = asyncio.run(
        baseline.perceive_judged(
            "present",
            "camera",
            t=1.0,
            goal_ids=("ordinary",),
        )
    )
    assert (
        split_event.goal_id,
        split_event.action,
        split_event.text,
        split_event.verdict.satisfied,
        split_event.verdict.confidence,
    ) == (
        baseline_events[0].goal_id,
        baseline_events[0].action,
        baseline_events[0].text,
        baseline_events[0].verdict.satisfied,
        baseline_events[0].verdict.confidence,
    )
    assert split_record == baseline.action_log.entries()[0]


def test_fusion_disabled_preserves_trace_before_action_log_order():
    from percept.backends.fake import FakeVision

    async def drive():
        observations = []
        core = Percept.create(
            vision=FakeVision(
                script={
                    "present": Verdict(
                        satisfied=True,
                        confidence=0.99,
                        evidence="present",
                    )
                }
            ),
            discover_plugins=False,
        )
        core.add_goal(
            Goal(id="ordinary", condition="present", say="seen", state={"type": "watch"})
        )

        async def trace(event):
            observations.append((event.stage, len(core.action_log.entries())))

        core.on_trace = trace
        events = await core.perceive_judged("present", t=1.0)
        return core, events, observations

    core, events, observations = asyncio.run(drive())
    assert [event.action for event in events] == ["fire"]
    assert len(core.action_log.entries()) == 1
    assert [stage for stage, _ in observations] == [
        "source.frame",
        "judge.started",
        "judge.finished",
        "gate.classified",
        "judgment.event",
        "shadow.fold",
    ]
    assert all(log_size == 0 for _, log_size in observations)


async def _reduce_observation_for(core, frame, goal_id, *, t):
    candidate = core.prepare_observation_candidate(
        frame,
        "camera",
        t=t,
        expected_goals_revision=core.goals.revision,
        expected_source_generation=core.source_continuity_generation("camera"),
        goal_ids=(goal_id,),
    )
    assert candidate is not None
    assert core.launch_observation_candidate(candidate)
    return core.reduce_observation_completion(
        await core.dispatch_observation_candidate(candidate)
    )


def _transition_core(vision) -> Percept:
    core = Percept.create(vision=vision, discover_plugins=False)
    core.add_goal(
        Goal(
            id="transition",
            condition="the user woke up",
            say="good morning",
            state={
                "type": "transition",
                "from": "the user is asleep",
                "to": "the user is awake",
                "count": 0,
            },
        )
    )
    return core


def _process_core(vision) -> Percept:
    core = Percept.create(vision=vision, discover_plugins=False)
    core.add_goal(
        Goal(
            id="process",
            condition="what steps are happening",
            say="answer ready",
            state={"type": "process"},
        )
    )
    return core


def test_transition_pair_is_one_state_free_completion_until_owner_reduction():
    class BlockingTransitionVision:
        def __init__(self):
            self.before_started = asyncio.Event()
            self.before_release = asyncio.Event()
            self.after_started = asyncio.Event()
            self.after_release = asyncio.Event()

        async def detect(self, frame, goals):
            return []

        async def judge(self, condition, frame, **kwargs):
            if condition == "the user is asleep":
                self.before_started.set()
                await self.before_release.wait()
                return Verdict(True, 0.99, "asleep")
            self.after_started.set()
            await self.after_release.wait()
            return Verdict(False, 0.99, "not awake")

    async def drive():
        vision = BlockingTransitionVision()
        core = _transition_core(vision)
        reduction = await _reduce_observation_for(
            core, "sleeping", "transition", t=1.0
        )
        job = core.prepare_transition_judgment(
            reduction, goal_id="transition"
        )
        assert job is not None
        task = asyncio.create_task(core.dispatch_judgment(job))
        await vision.before_started.wait()
        assert core.executor.ctx("transition").transitions == {}
        assert core.goals.get("transition").state["count"] == 0
        assert core.action_log.entries() == []

        vision.before_release.set()
        await vision.after_started.wait()
        assert core.executor.ctx("transition").transitions == {}
        assert core.action_log.entries() == []

        vision.after_release.set()
        completion = await task
        assert len(completion.calls) == 2
        assert completion.job.token.admission_id == job.token.admission_id
        assert completion.completion_ordinal is None
        assert core.executor.ctx("transition").transitions == {}
        assert core.action_log.entries() == []

        assert core.reduce_judgment_completion(completion) == []
        assert core.executor.ctx("transition").transitions["scene"] == {
            "seen_from": True,
            "fired": False,
        }
        assert core.action_log.entries() == []

    asyncio.run(drive())


def test_unlaunched_superseded_job_spends_nothing_and_launched_pair_counts_once():
    class CountingTransitionVision:
        def __init__(self):
            self.calls = []

        async def detect(self, frame, goals):
            return []

        async def judge(self, condition, frame, **kwargs):
            self.calls.append((condition, frame))
            return Verdict(
                condition == "the user is asleep",
                0.99,
                condition,
            )

    async def drive():
        vision = CountingTransitionVision()
        core = _transition_core(vision)
        first_reduction = await _reduce_observation_for(
            core, "first", "transition", t=1.0
        )
        never_launched = core.prepare_transition_judgment(
            first_reduction, goal_id="transition"
        )
        assert never_launched is not None
        assert core.metrics.get("funnel.judged") == 0
        assert vision.calls == []

        stale_reduction = await _reduce_observation_for(
            core, "second", "transition", t=2.0
        )
        stale_launched = core.prepare_transition_judgment(
            stale_reduction, goal_id="transition"
        )
        assert stale_launched is not None
        stale_completion = await core.dispatch_judgment(stale_launched)
        assert core.metrics.get("funnel.judged") == 0

        live_reduction = await _reduce_observation_for(
            core, "third", "transition", t=3.0
        )
        live_launched = core.prepare_transition_judgment(
            live_reduction, goal_id="transition"
        )
        assert live_launched is not None
        # Projection alone cannot supersede accepted work. The latest
        # admission changes only when the lane launches this job.
        assert (
            core._judgment_latest_admission[("transition", "scene")]
            == stale_launched.token.admission_id
            != never_launched.token.admission_id
        )
        assert core.metrics.get("funnel.judged") == 0
        live_completion = await core.dispatch_judgment(live_launched)
        assert (
            core._judgment_latest_admission[("transition", "scene")]
            == live_launched.token.admission_id
        )
        # A finished provider task is still only a receipt; spend is published
        # once, atomically with owner reduction.
        assert core.metrics.get("funnel.judged") == 0
        core.reduce_judgment_completion(live_completion)
        assert core.metrics.get("funnel.judged") == 2
        assert len(vision.calls) == 4

        # The stale completion is semantically inert but its two real calls
        # still count once; replay cannot double-count them. The first job
        # remains a never-dispatched projection and contributes no spend.
        assert core.reduce_judgment_completion(stale_completion) == []
        assert core.metrics.get("funnel.judged") == 4
        assert core.reduce_judgment_completion(stale_completion) == []
        assert core.metrics.get("funnel.judged") == 4
        assert len(vision.calls) == 4

    asyncio.run(drive())


@pytest.mark.parametrize("stale_kind", ["goal_revision", "source_generation", "judge_binding"])
def test_transition_stale_completion_is_inert(stale_kind):
    class TransitionVision:
        async def detect(self, frame, goals):
            return []

        async def judge(self, condition, frame, **kwargs):
            return Verdict(
                condition == "the user is asleep",
                0.99,
                condition,
            )

    async def drive():
        vision = TransitionVision()
        core = _transition_core(vision)
        reduction = await _reduce_observation_for(
            core, "sleeping", "transition", t=1.0
        )
        job = core.prepare_transition_judgment(
            reduction, goal_id="transition"
        )
        completion = await core.dispatch_judgment(job)
        if stale_kind == "goal_revision":
            core.goals.apply(
                [
                    GoalOp(
                        op="update",
                        goal_id="transition",
                        fields={"say": "changed presentation"},
                    )
                ]
            )
        elif stale_kind == "source_generation":
            core.advance_source_continuity("camera")
        else:
            core.vision = TransitionVision()
        assert core.reduce_judgment_completion(completion) == []
        assert core.executor.ctx("transition").transitions == {}
        assert core.action_log.entries() == []
        assert core.goals.get("transition").state["count"] == 0
        assert core.metrics.get("funnel.judged") == 2
        assert core.metrics.get("transition.fire") == 0
        assert core.metrics.get("judge.degraded") == 0

    asyncio.run(drive())


@pytest.mark.parametrize("route", ["ordinary", "transition"])
@pytest.mark.parametrize("stale_kind", ["watch_application", "watch_revision"])
def test_routed_token_validates_exact_watch_application_and_revision(
    route, stale_kind
):
    class RoutedVision:
        async def detect(self, frame, goals):
            return []

        async def judge(self, condition, frame, **kwargs):
            return Verdict(condition != "the user is awake", 0.99, condition)

    async def drive():
        core = (
            _core(RoutedVision())
            if route == "ordinary"
            else _transition_core(RoutedVision())
        )
        runtime = SimpleNamespace(
            settings=core.settings,
            plan_mode=core.plan_mode,
        )
        core._register_watch_runtime(
            route,
            runtime,
            application_key="application-a",
        )
        reduction = await _reduce_observation_for(
            core, "frame", route, t=1.0
        )
        job = (
            core.prepare_frame_judgment(reduction, goal_id=route)
            if route == "ordinary"
            else core.prepare_transition_judgment(reduction, goal_id=route)
        )
        assert job.token.watch_application == "application-a"
        assert job.token.watch_revision == core.goals.revision
        completion = await core.dispatch_judgment(job)
        if stale_kind == "watch_revision":
            completion = replace(
                completion,
                job=replace(
                    completion.job,
                    token=replace(
                        completion.job.token,
                        watch_revision=(
                            completion.job.token.watch_revision + 1
                        ),
                    ),
                ),
            )
        if stale_kind == "watch_application":
            core._watch_runtime_keys[route] = "application-b"
        assert core.reduce_judgment_completion(completion) == []
        assert core.action_log.entries() == []
        assert core.gate.snapshot()["armed"] == []
        if route == "transition":
            assert core.executor.ctx(route).transitions == {}
        else:
            assert core.last_supported == {}
            assert core.last_satisfied == {}

    asyncio.run(drive())


def test_transition_split_pipeline_matches_legacy_event_action_and_trace_order():
    class ScriptTransitionVision:
        def __init__(self):
            self.script = {
                ("the user is asleep", "sleeping"): Verdict(True, 0.99, "asleep"),
                ("the user is awake", "sleeping"): Verdict(False, 0.99, "not awake"),
                ("the user is asleep", "awake"): Verdict(False, 0.99, "not asleep"),
                ("the user is awake", "awake"): Verdict(True, 0.99, "awake"),
            }

        async def detect(self, frame, goals):
            return []

        async def judge(self, condition, frame, **kwargs):
            return self.script[(condition, frame)]

    async def split_drive():
        core = _transition_core(ScriptTransitionVision())

        async def enabled_trace(_event):
            raise AssertionError("split reducer must not invoke user callbacks")

        core.on_trace = enabled_trace
        all_events = []
        all_rows = []
        for t, frame in ((1.0, "sleeping"), (2.0, "awake")):
            reduction = await _reduce_observation_for(
                core, frame, "transition", t=t
            )
            job = core.prepare_transition_judgment(
                reduction, goal_id="transition"
            )
            completion = await core.dispatch_judgment(job)
            events = core.reduce_judgment_completion(completion)
            rows = core.drain_judgment_trace_rows()
            # The owner synchronously materializes exactly one ActionRecord
            # before a later trace/subscriber delivery can run.
            assert len(core.action_log.entries()) == len(all_events) + len(events)
            all_events.extend(events)
            all_rows.extend(rows)
        return core, all_events, all_rows

    async def legacy_drive():
        core = _transition_core(ScriptTransitionVision())
        rows = []

        async def trace(event):
            rows.append((event.stage, dict(event.data)))

        core.on_trace = trace
        events = []
        for t, frame in ((1.0, "sleeping"), (2.0, "awake")):
            events.extend(await core.perceive_judged(frame, t=t))
        return core, events, rows

    split, split_events, split_rows = asyncio.run(split_drive())
    legacy, legacy_events, legacy_rows = asyncio.run(legacy_drive())
    assert len(split_events) == len(legacy_events) == 1
    split_event, legacy_event = split_events[0], legacy_events[0]
    assert (
        split_event.goal_id,
        split_event.action,
        split_event.text,
        split_event.entity_id,
        split_event.verdict,
        split_event.judgment.action,
        split_event.judgment.evidence_origin,
    ) == (
        legacy_event.goal_id,
        legacy_event.action,
        legacy_event.text,
        legacy_event.entity_id,
        legacy_event.verdict,
        legacy_event.judgment.action,
        legacy_event.judgment.evidence_origin,
    )
    assert split.goals.get("transition").state["count"] == 1
    assert split.action_log.entries() == legacy.action_log.entries()
    assert [row[2] for row in split_rows] == [
        "transition.decision",
        "transition.decision",
        "judgment.event",
    ]
    assert [
        stage
        for stage, _ in legacy_rows
        if stage in ("transition.decision", "judgment.event")
    ] == [row[2] for row in split_rows]


def test_public_transition_materializes_one_action_before_decision_callbacks():
    class ScriptTransitionVision:
        async def detect(self, frame, goals):
            return []

        async def judge(self, condition, frame, **kwargs):
            answers = {
                ("the user is asleep", "sleeping"): True,
                ("the user is awake", "sleeping"): False,
                ("the user is asleep", "awake"): False,
                ("the user is awake", "awake"): True,
            }
            return Verdict(answers[(condition, frame)], 0.99, condition)

    async def drive():
        core = _transition_core(ScriptTransitionVision())
        callback_log_sizes = []

        async def trace(event):
            if (
                event.t == 2.0
                and event.stage in ("transition.decision", "judgment.event")
            ):
                callback_log_sizes.append(
                    (event.stage, len(core.action_log.entries("fire")))
                )

        core.on_trace = trace
        assert await core.perceive_judged("sleeping", t=1.0) == []
        events = await core.perceive_judged("awake", t=2.0)
        return core, events, callback_log_sizes

    core, events, callback_log_sizes = asyncio.run(drive())
    assert [event.action for event in events] == ["fire"]
    assert len(core.action_log.entries("fire")) == 1
    assert callback_log_sizes == [
        ("transition.decision", 1),
        ("judgment.event", 1),
    ]


def test_process_completion_is_log_only_and_matches_legacy():
    from percept.backends.fake import FakeVision

    script = {
        "step|step": Verdict(False, 0.99, "open the package"),
        "empty|step": Verdict(False, 0.99, "not enough evidence"),
    }

    async def split_drive():
        core = _process_core(FakeVision(script=script))
        gate_before = core.gate.snapshot()
        for t, frame in ((1.0, "step"), (2.0, "empty")):
            reduction = await _reduce_observation_for(
                core, frame, "process", t=t
            )
            job = core.prepare_process_judgment(reduction, goal_id="process")
            completion = await core.dispatch_judgment(job)
            assert core.executor.ctx("process").process_log == (
                [] if t == 1.0 else ["open the package"]
            )
            assert core.reduce_judgment_completion(completion) == []
        return core, gate_before

    async def legacy_drive():
        core = _process_core(FakeVision(script=script))
        for t, frame in ((1.0, "step"), (2.0, "empty")):
            assert await core.perceive_judged(frame, t=t) == []
        return core

    split, gate_before = asyncio.run(split_drive())
    legacy = asyncio.run(legacy_drive())
    split_ctx = split.executor.ctx("process")
    legacy_ctx = legacy.executor.ctx("process")
    assert split_ctx.process_log == legacy_ctx.process_log == ["open the package"]
    assert split_ctx.last_step_at == legacy_ctx.last_step_at == 1.0
    assert split.gate.snapshot() == gate_before
    assert split.action_log.entries() == legacy.action_log.entries() == []


def test_process_provider_is_state_free_while_blocked_and_until_reduction():
    class BlockingProcessVision:
        def __init__(self):
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def detect(self, frame, goals):
            return []

        async def judge(self, condition, frame, **kwargs):
            self.started.set()
            await self.release.wait()
            return Verdict(False, 0.99, "inspect the label")

    async def drive():
        vision = BlockingProcessVision()
        core = _process_core(vision)
        gate_before = core.gate.snapshot()
        reduction = await _reduce_observation_for(
            core, "frame", "process", t=1.0
        )
        job = core.prepare_process_judgment(reduction, goal_id="process")
        task = asyncio.create_task(core.dispatch_judgment(job))
        await vision.started.wait()
        ctx = core.executor.ctx("process")
        assert ctx.process_log == []
        assert ctx.last_step_at == float("-inf")
        assert core.action_log.entries() == []
        assert core.gate.snapshot() == gate_before

        vision.release.set()
        completion = await task
        assert ctx.process_log == []
        assert ctx.last_step_at == float("-inf")
        assert core.reduce_judgment_completion(completion) == []
        assert ctx.process_log == ["inspect the label"]
        assert ctx.last_step_at == 1.0
        assert core.action_log.entries() == []
        assert core.gate.snapshot() == gate_before

    asyncio.run(drive())


@pytest.mark.parametrize("stale_kind", ["goal_revision", "source_generation", "judge_binding"])
def test_process_stale_completion_is_log_and_gate_inert(stale_kind):
    class ProcessVision:
        async def detect(self, frame, goals):
            return []

        async def judge(self, condition, frame, **kwargs):
            return Verdict(False, 0.99, "inspect the label")

    async def drive():
        core = _process_core(ProcessVision())
        gate_before = core.gate.snapshot()
        reduction = await _reduce_observation_for(
            core, "frame", "process", t=1.0
        )
        job = core.prepare_process_judgment(reduction, goal_id="process")
        completion = await core.dispatch_judgment(job)
        if stale_kind == "goal_revision":
            core.goals.apply(
                [
                    GoalOp(
                        op="update",
                        goal_id="process",
                        fields={"say": "changed presentation"},
                    )
                ]
            )
        elif stale_kind == "source_generation":
            core.advance_source_continuity("camera")
        else:
            core.vision = ProcessVision()
        assert core.reduce_judgment_completion(completion) == []
        ctx = core.executor.ctx("process")
        assert ctx.process_log == []
        assert ctx.last_step_at == float("-inf")
        assert core.gate.snapshot() == gate_before
        assert core.action_log.entries() == []
        assert core.metrics.get("funnel.judged") == 1

    asyncio.run(drive())


@pytest.mark.parametrize("route", ["transition", "process"])
def test_routed_provider_failure_is_redacted_and_semantically_inert(route):
    secret = "SECRET-PROVIDER-CREDENTIAL"

    class FailingVision:
        def __init__(self):
            self.calls = []

        async def detect(self, frame, goals):
            return []

        async def judge(self, condition, frame, **kwargs):
            self.calls.append(condition)
            if route == "transition" and condition == "the user is asleep":
                return Verdict(True, 0.99, "asleep")
            raise RuntimeError(f"https://user:{secret}@provider.invalid")

    async def drive():
        vision = FailingVision()
        core = (
            _transition_core(vision)
            if route == "transition"
            else _process_core(vision)
        )
        reduction = await _reduce_observation_for(
            core, "frame", route, t=1.0
        )
        job = (
            core.prepare_transition_judgment(reduction, goal_id=route)
            if route == "transition"
            else core.prepare_process_judgment(reduction, goal_id=route)
        )
        completion = await core.dispatch_judgment(job)
        assert completion.failure is not None
        assert secret not in repr(completion)
        assert core.reduce_judgment_completion(completion) == []
        assert core.action_log.entries() == []
        assert core.gate.snapshot()["armed"] == []
        if route == "transition":
            assert vision.calls == ["the user is asleep", "the user is awake"]
            assert core.executor.ctx(route).transitions == {}
        else:
            assert vision.calls == ["what steps are happening"]
            assert core.executor.ctx(route).process_log == []

    asyncio.run(drive())
