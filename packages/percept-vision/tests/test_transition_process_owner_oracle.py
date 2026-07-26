"""Independent transition/process oracle for the unified cognition owner.

The generic carriers are not sufficient by themselves.  This module proves that route-specific
preparation has the exact legacy provider contract, that rejected offers cannot supersede accepted
work, that stale work remains semantically inert without erasing factual spend, and that a FIRE is
fully owner-materialized before any asynchronous trace or subscriber callback.
"""

from __future__ import annotations

import asyncio
import inspect
from copy import deepcopy

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.goals import GoalOp
from percept.observability import funnel_key


FROM = "the form is pending"
TO = "the form is approved"


class _ScriptVision:
    def __init__(self, answers=None) -> None:
        self.answers = dict(answers or {})
        self.calls: list[dict] = []

    async def detect(self, _frame, _goals):
        return []

    async def judge(
        self,
        condition,
        frame,
        *,
        roi=None,
        exemplar=None,
        corrections=None,
        frames=None,
        audio=None,
        context=None,
    ):
        self.calls.append(
            {
                "condition": condition,
                "frame": deepcopy(frame),
                "roi": deepcopy(roi),
                "exemplar": deepcopy(exemplar),
                "corrections": deepcopy(corrections),
                "frames": deepcopy(frames),
                "audio": deepcopy(audio),
                "context": dict(context) if context is not None else None,
            }
        )
        token = frame.get("token") if isinstance(frame, dict) else frame
        answer = self.answers.get(
            (condition, token),
            self.answers.get(condition, Verdict(False, 0.99, "")),
        )
        if isinstance(answer, BaseException):
            raise answer
        return answer


class _StrictLegacyTransitionVision:
    """The exact keyword surface historically accepted by ``_perceive_transition``."""

    def __init__(self) -> None:
        self.calls = []

    async def detect(self, _frame, _goals):
        return []

    async def judge(
        self,
        condition,
        frame,
        *,
        roi=None,
        corrections=None,
        frames=None,
        context=None,
    ):
        self.calls.append(
            {
                "condition": condition,
                "roi": roi,
                "corrections": corrections,
                "frames": frames,
                "context": context,
            }
        )
        return Verdict(condition == FROM, 0.99, condition)


class _MalformedVision:
    def __init__(self) -> None:
        self.calls = 0

    async def detect(self, _frame, _goals):
        return []

    async def judge(self, *_args, **_kwargs):
        self.calls += 1
        return {"satisfied": True}


def _core_with_goal(
    vision,
    goal: Goal,
    *,
    on_trace=None,
    extractor=None,
) -> Percept:
    core = Percept.create(
        vision=vision,
        discover_plugins=False,
        on_trace=on_trace,
        extractor=extractor,
    )
    core.add_goal(goal)
    return core


def _transition_goal(*, exemplar=None) -> Goal:
    state = {
        "type": "transition",
        "from": FROM,
        "to": TO,
        "count": 0,
    }
    if exemplar is not None:
        state["exemplar"] = exemplar
    return Goal(
        id="transition",
        condition="the form became approved",
        say="approved",
        source="camera",
        state=state,
    )


def _process_goal() -> Goal:
    return Goal(
        id="process",
        condition="what step is happening",
        say="step",
        source="camera",
        state={"type": "process"},
    )


async def _observation_reduction(
    core: Percept,
    *,
    frame,
    t: float,
    goal_id: str,
):
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
    completion = await core.dispatch_observation_candidate(candidate)
    reduction = core.reduce_observation_completion(completion)
    assert reduction.stale_reason is None
    assert reduction.failure is None
    return reduction


async def _prepare_process(core: Percept, *, frame, t: float):
    reduction = await _observation_reduction(
        core,
        frame=frame,
        t=t,
        goal_id="process",
    )
    job = core.prepare_process_judgment(reduction, goal_id="process")
    assert job is not None
    return job


async def _prepare_transition(core: Percept, *, frame, t: float):
    reduction = await _observation_reduction(
        core,
        frame=frame,
        t=t,
        goal_id="transition",
    )
    job = core.prepare_transition_judgment(reduction, goal_id="transition")
    assert job is not None
    return job


def _metric(core: Percept, key: str) -> int | float:
    return core.metrics.snapshot().get(key, 0)


def test_process_split_path_matches_legacy_provider_request_and_state():
    answer = Verdict(False, 0.99, "slice the onion")
    legacy_vision = _ScriptVision({"what step is happening": answer})
    split_vision = _ScriptVision({"what step is happening": answer})
    legacy = _core_with_goal(legacy_vision, _process_goal())
    split = _core_with_goal(split_vision, _process_goal())

    async def drive():
        await legacy._observe_and_log(
            legacy.goals.get("process"),
            {"token": "frame"},
            "camera",
            1.0,
        )
        job = await _prepare_process(
            split,
            frame={"token": "frame"},
            t=1.0,
        )
        completion = await split.dispatch_judgment(job)
        assert split.reduce_judgment_completion(completion) == []

    asyncio.run(drive())
    assert split.executor.ctx("process").process_log == ["slice the onion"]
    assert split.executor.ctx("process").process_log == legacy.executor.ctx(
        "process"
    ).process_log
    assert split.executor.ctx("process").last_step_at == 1.0
    assert split_vision.calls == legacy_vision.calls
    assert _metric(split, funnel_key("judged", concern="process")) == 1
    assert _metric(split, "judge.latency.count") == 1


def test_transition_split_path_preserves_exact_legacy_keyword_surface():
    """Adding even a benign kwarg breaks existing strict adapters and is not parity."""
    legacy_vision = _StrictLegacyTransitionVision()
    split_vision = _StrictLegacyTransitionVision()
    legacy = _core_with_goal(
        legacy_vision,
        _transition_goal(exemplar={"reference": ["legacy-must-not-send"]}),
    )
    split = _core_with_goal(
        split_vision,
        _transition_goal(exemplar={"reference": ["legacy-must-not-send"]}),
    )

    async def drive():
        legacy_event = await legacy._perceive_transition(
            legacy.goals.get("transition"),
            {"token": "frame"},
            None,
            now=1.0,
            t=1.0,
            ingest_observed_at=1.0,
            trace_id="legacy",
            tracing=False,
            source="camera",
        )
        job = await _prepare_transition(
            split,
            frame={"token": "frame"},
            t=1.0,
        )
        completion = await split.dispatch_judgment(job)
        return legacy_event, completion

    legacy_event, completion = asyncio.run(drive())
    assert legacy_event is None
    assert len(legacy_vision.calls) == 2
    assert completion.failure is None, (
        "split transition passed a keyword (currently exemplar) absent from the "
        "legacy transition provider contract"
    )
    assert len(split_vision.calls) == 2


def test_rejected_new_process_offer_does_not_supersede_accepted_completion():
    """Preparation cannot publish ``latest_admitted`` before a lane accepts the slot."""
    vision = _ScriptVision(
        {
            ("what step is happening", "accepted"): Verdict(
                False, 0.99, "accepted step"
            ),
            ("what step is happening", "rejected"): Verdict(
                False, 0.99, "must never launch"
            ),
        }
    )
    core = _core_with_goal(vision, _process_goal())

    async def drive():
        active_job = await _prepare_process(
            core,
            frame={"token": "accepted"},
            t=1.0,
        )
        active_completion = await core.dispatch_judgment(active_job)

        # This represents an owner checking/offering to a closing or full lane.  Merely projecting
        # the candidate must not publish a superseding admission until the lane accepts it.
        rejected_job = await _prepare_process(
            core,
            frame={"token": "rejected"},
            t=2.0,
        )
        assert rejected_job is not None
        assert len(vision.calls) == 1  # the rejected offer occupied no provider call

        assert core.reduce_judgment_completion(active_completion) == []

    asyncio.run(drive())
    assert core.executor.ctx("process").process_log == ["accepted step"], (
        "a prepared-but-rejected offer superseded the only accepted completion"
    )


def test_stale_process_completion_keeps_factual_spend_but_no_semantic_state():
    answer = Verdict(
        False,
        0.99,
        "obsolete step",
        usage={"input_tokens": 7, "cost_usd": 0.007},
    )
    vision = _ScriptVision({"what step is happening": answer})
    core = _core_with_goal(vision, _process_goal())

    async def drive():
        job = await _prepare_process(
            core,
            frame={"token": "stale"},
            t=1.0,
        )
        completion = await core.dispatch_judgment(job)
        core.apply_goal_ops(
            [
                GoalOp(
                    op="update",
                    goal_id="process",
                    fields={"say": "new presentation"},
                )
            ]
        )
        assert core.reduce_judgment_completion(completion) == []
        first_accounting = deepcopy(core.metrics.snapshot())
        assert core.reduce_judgment_completion(completion) == []
        assert core.metrics.snapshot() == first_accounting

    asyncio.run(drive())
    assert core.executor.ctx("process").process_log == []
    assert _metric(core, funnel_key("judged", concern="process")) == 1
    assert _metric(core, "judge.latency.count") == 1
    assert _metric(core, "judge.tokens.input") == 7
    assert _metric(core, "judge.cost.usd") == 0.007


def test_malformed_process_response_retains_exact_legacy_metering():
    legacy = _core_with_goal(_MalformedVision(), _process_goal())
    split = _core_with_goal(_MalformedVision(), _process_goal())

    async def drive():
        await legacy._observe_and_log(
            legacy.goals.get("process"),
            {"token": "malformed"},
            "camera",
            1.0,
        )
        job = await _prepare_process(
            split,
            frame={"token": "malformed"},
            t=1.0,
        )
        completion = await split.dispatch_judgment(job)
        assert completion.failure is not None
        assert split.reduce_judgment_completion(completion) == []

    asyncio.run(drive())
    keys = (
        funnel_key("judged", concern="process"),
        "judge.latency.count",
        "judge.degraded",
    )
    assert {
        key: _metric(split, key) for key in keys
    } == {
        key: _metric(legacy, key) for key in keys
    }


def test_transition_fire_action_and_trace_are_materialized_before_callback():
    answers = {
        (FROM, "arm"): Verdict(True, 0.99, "pending"),
        (TO, "arm"): Verdict(False, 0.99, "not approved"),
        (FROM, "fire"): Verdict(False, 0.99, "not pending"),
        (TO, "fire"): Verdict(True, 0.99, "approved"),
    }
    callback_rows = []

    async def trace_callback(row):
        callback_rows.append(row)

    core = _core_with_goal(
        _ScriptVision(answers),
        _transition_goal(),
        on_trace=trace_callback,
    )

    async def drive():
        arm_job = await _prepare_transition(
            core,
            frame={"token": "arm"},
            t=1.0,
        )
        arm_completion = await core.dispatch_judgment(arm_job)
        assert core.reduce_judgment_completion(arm_completion) == []
        core.drain_judgment_trace_rows()

        fire_job = await _prepare_transition(
            core,
            frame={"token": "fire"},
            t=2.0,
        )
        fire_completion = await core.dispatch_judgment(fire_job)
        events = core.reduce_judgment_completion(fire_completion)
        return fire_job, events, core.drain_judgment_trace_rows()

    job, events, trace_rows = asyncio.run(drive())
    assert [event.action for event in events] == ["fire"]
    assert callback_rows == []  # reducer invoked no async/user callback
    # Legacy perceive_judged records a transition at the route boundary and again in its aggregate
    # event log.  Promotion may later repair that historical duplication only as an explicit
    # behavior change; this parity slice must retain both, synchronously, before callbacks.
    assert [row.goal_id for row in core.action_log.entries("fire")] == [
        "transition",
        "transition",
    ], "transition FIRE returned before synchronous legacy action-log materialization"
    assert trace_rows
    assert all(row[0] == job.token.judgment_trace_id for row in trace_rows)


def test_transition_extractor_trace_has_exact_legacy_stage_before_decision():
    from percept.extraction import Enrichment

    class Extractor:
        def extract(self, _condition, _frame, _roi=None):
            return Enrichment(facts={"status": "approved"})

    callback_rows = []

    async def trace_callback(row):
        callback_rows.append(row)

    core = _core_with_goal(
        _ScriptVision(
            {
                FROM: Verdict(True, 0.99, "pending"),
                TO: Verdict(False, 0.99, "not approved"),
            }
        ),
        _transition_goal(),
        on_trace=trace_callback,
        extractor=Extractor(),
    )

    async def drive():
        job = await _prepare_transition(
            core,
            frame={"token": "frame"},
            t=1.0,
        )
        completion = await core.dispatch_judgment(job)
        assert core.reduce_judgment_completion(completion) == []
        return core.drain_judgment_trace_rows()

    rows = asyncio.run(drive())
    assert callback_rows == []
    assert [stage for _trace_id, _t, stage, _data in rows] == [
        "extractor.enriched",
        "transition.decision",
    ]


def test_transition_provider_cannot_mutate_frozen_job_or_second_call_input():
    class MutatingVision:
        def __init__(self) -> None:
            self.seen = []

        async def detect(self, _frame, _goals):
            return []

        async def judge(self, condition, frame, **_kwargs):
            self.seen.append(deepcopy(frame))
            frame["nested"]["provider_mutations"].append(condition)
            return Verdict(condition == FROM, 0.99, condition)

    vision = MutatingVision()
    core = _core_with_goal(vision, _transition_goal())

    async def drive():
        job = await _prepare_transition(
            core,
            frame={
                "token": "frame",
                "nested": {"provider_mutations": []},
            },
            t=1.0,
        )
        completion = await core.dispatch_judgment(job)
        return job, completion

    job, completion = asyncio.run(drive())
    assert completion.failure is None
    expected = {
        "token": "frame",
        "nested": {"provider_mutations": []},
    }
    assert vision.seen == [expected, expected], (
        "the before provider call rewrote the after call's frozen input"
    )
    assert job.provider.before.frame == expected
    assert job.provider.after.frame == expected


def test_transition_process_preparers_are_live_worker_protocol_not_dead_seams():
    import percept.runtime.worker as worker

    source = inspect.getsource(worker)
    for seam in (
        "prepare_observation_candidate",
        "dispatch_observation_candidate",
        "reduce_observation_completion",
        "prepare_transition_judgment",
        "prepare_process_judgment",
        "dispatch_judgment",
        "reduce_judgment_completion",
    ):
        assert seam in source, (
            f"missing runtime integration: {seam} exists on Percept but the worker "
            "never calls it"
        )
    assert "core.perceive_judged(" not in source


def test_route_validator_checks_compiled_application_identity():
    source = inspect.getsource(Percept._judgment_stale_reason)
    assert "watch_application" in source, (
        "route validator ignores the admitted Watch application token"
    )
