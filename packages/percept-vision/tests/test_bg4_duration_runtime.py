"""Adversarial BG4 duration/runtime contract tests.

This file intentionally lives apart from the implementation's own acceptance tests.  It exercises
the public seams as a client would: direct capture, the runtime worker, tier-0 admission, template
application, and remote-topology refusal.
"""

from __future__ import annotations

import asyncio
import copy
import math
from dataclasses import replace

import pytest

from percept import Goal, Percept
from percept.adapters import Verdict, WatchCondition, WatchSpec
from percept.backends.fake import FakeVision
from percept.contracts.watchplan import WatchPlan
from percept.domain_pack import watch_template
from percept.duration_routes import (
    UNSUPPORTED_DURATION_ROUTE,
    UnsupportedDurationRouteError,
)
from percept.edge.tier0 import (
    GateDecision,
    MotionGate,
    RemoteEdgeReceiver,
    compile_watchspec,
    gray_frame,
)
from percept.gate import FIRE, SILENT, Gate
from percept.goals import GoalOp
from percept.packs_default import FACILITY_OBSERVATION_V1
from percept.runtime import RuntimeStats, run
from percept.watchplan_apply import apply_watch_plan


YES = Verdict(satisfied=True, confidence=0.99, evidence="scripted: door open")
NO = Verdict(satisfied=False, confidence=0.99, evidence="scripted: door closed")
STILL = gray_frame(8, 8, 0)
MOVED = gray_frame(8, 8, 255)
FIXED_5 = {"min_judge_interval": 5.0, "gate_on": "fixed"}
FIXED_10 = {"min_judge_interval": 10.0, "gate_on": "fixed"}


class RecordingFakeVision(FakeVision):
    """The shipped deterministic fake plus call counters for fail-before-provider assertions."""

    def __init__(self, script=None):
        super().__init__(script=script or {"open": YES, "closed": NO})
        self.judge_calls: list[tuple[str, object]] = []
        self.detect_calls = 0

    async def judge(self, condition, frame, **kwargs):
        self.judge_calls.append((condition, frame))
        return await super().judge(condition, frame, **kwargs)

    async def detect(self, frame, goals):
        self.detect_calls += 1
        return await super().detect(frame, goals)


def duration_goal(
    goal_id: str = "door",
    *,
    source: str = "camera",
    hold_s: object = 10.0,
    cadence: object = FIXED_5,
    state: dict | None = None,
) -> Goal:
    goal_state = {
        "type": "watch",
        "hold_s": hold_s,
        "identity_policy": "scene",
    }
    if cadence is not None:
        goal_state["cadence"] = copy.deepcopy(cadence)
    if state:
        goal_state.update(state)
    return Goal(
        id=goal_id,
        condition="the door is open",
        say="door left open",
        source=source,
        state=goal_state,
    )


async def collect(core, source, **kwargs):
    return [incident async for incident in run(core, source=source, **kwargs)]


@pytest.mark.parametrize(
    ("state", "path"),
    [
        ({"type": "watch", "hold_s": True, "cadence": FIXED_5}, "Goal.state.hold_s"),
        (
            {"type": "transition", "from": "closed", "to": "open", "hold_s": 5.0, "cadence": FIXED_5},
            "Goal.state.hold_s",
        ),
        (
            {"type": "watch", "hold_s": 5.0, "cadence": FIXED_5, "schedule": {"windows": []}},
            "Goal.state.schedule",
        ),
        (
            {
                "type": "watch",
                "hold_s": 5.0,
                "cadence": {"min_judge_interval": 1.0, "gate_on": "motion"},
            },
            "Goal.state.cadence.gate_on",
        ),
        (
            {"type": "watch", "cadence": {"min_judge_interval": math.nan, "gate_on": "fixed"}},
            "Goal.state.cadence.min_judge_interval",
        ),
    ],
)
def test_direct_route_matrix_rejects_before_detection_or_judgment(state, path):
    """Direct mode may omit cadence, but a cadence that is present must still be valid."""

    vision = RecordingFakeVision()
    core = Percept.create(vision=vision, discover_plugins=False)
    core.add_goal(
        Goal(
            id="bad",
            condition="the door is open",
            say="bad",
            state=copy.deepcopy(state),
        )
    )

    with pytest.raises(UnsupportedDurationRouteError) as caught:
        asyncio.run(core.perceive_judged("open", t=0.0))

    assert caught.value.issues[0].path == path
    assert caught.value.issues[0].code == UNSUPPORTED_DURATION_ROUTE
    assert vision.detect_calls == 0
    assert vision.judge_calls == []


def test_direct_mode_allows_missing_cadence_but_worker_refuses_before_source_pull():
    vision = RecordingFakeVision()
    core = Percept.create(vision=vision, discover_plugins=False)
    core.add_goal(duration_goal(cadence=None, hold_s=10.0))

    assert asyncio.run(core.perceive_judged("open", t=0.0)) == []
    direct = asyncio.run(core.perceive_judged("open", t=10.0))
    assert [event.action for event in direct] == [FIRE]

    pulled = False

    class PoisonSource:
        def __iter__(self):
            nonlocal pulled
            pulled = True
            raise AssertionError("worker pulled an unsupported source")

    with pytest.raises(UnsupportedDurationRouteError) as caught:
        asyncio.run(collect(core, PoisonSource(), judge_dispatch="inline"))
    assert caught.value.issues[0].path == "Goal.state.cadence.min_judge_interval"
    assert caught.value.issues[0].code == UNSUPPORTED_DURATION_ROUTE
    assert pulled is False


def test_dynamic_duration_regate_rejects_generic_core_without_continuity_hook():
    vision = RecordingFakeVision()
    base = Percept.create(vision=vision, discover_plugins=False)

    class CoreWithoutContinuityHook:
        def __getattr__(self, name):
            if name == "clear_pending_dwell":
                raise AttributeError(name)
            return getattr(base, name)

    pulled: list[float] = []

    async def source():
        pulled.append(0.0)
        yield (0.0, STILL, "open")
        base.add_goal(duration_goal())
        pulled.append(5.0)
        yield (5.0, STILL, "open")
        pulled.append(10.0)
        yield (10.0, STILL, "open")

    with pytest.raises(UnsupportedDurationRouteError) as caught:
        asyncio.run(
            collect(
                CoreWithoutContinuityHook(),
                source(),
                judge_dispatch="inline",
                regate_on_goal_change=True,
            )
        )

    assert caught.value.issues[0].path == "Runtime.clear_pending_dwell"
    assert pulled == [0.0, 5.0]
    assert vision.detect_calls == 0
    assert vision.judge_calls == []


def test_gate_support_gap_boundary_and_pending_clear_do_not_rearm():
    key = ("door", "scene")
    gate = Gate()

    assert gate.evaluate(key, YES, hold_s=15.0, max_support_gap_s=10.0, now=0.0) == SILENT
    # Equality is supported continuity, not a gap.
    assert gate.evaluate(key, YES, hold_s=15.0, max_support_gap_s=10.0, now=10.0) == SILENT
    assert gate.evaluate(key, YES, hold_s=15.0, max_support_gap_s=10.0, now=15.0) == FIRE

    gate.clear_pending_dwell(key)
    assert gate.evaluate(key, YES, hold_s=15.0, max_support_gap_s=10.0, now=16.0) == SILENT
    assert gate._armed[key] is False

    reset = Gate()
    assert reset.evaluate(key, YES, hold_s=15.0, max_support_gap_s=10.0, now=0.0) == SILENT
    # Strictly greater than 2*C restarts the proof at this observation.
    assert reset.evaluate(key, YES, hold_s=15.0, max_support_gap_s=10.0, now=11.0) == SILENT
    assert reset._dwell_since[key] == 11.0
    assert reset.evaluate(key, YES, hold_s=15.0, max_support_gap_s=10.0, now=21.0) == SILENT
    assert reset.evaluate(key, YES, hold_s=15.0, max_support_gap_s=10.0, now=26.0) == FIRE


def test_pending_clear_is_source_scoped_and_includes_inactive_goals():
    core = Percept.create(mode="offline", discover_plugins=False)
    camera = duration_goal("camera-door", source="camera")
    rear = duration_goal("rear-door", source="rear")
    core.add_goal(camera)
    core.add_goal(rear)

    fired_key = (camera.id, "fired")
    camera_pending = (camera.id, "pending")
    rear_pending = (rear.id, "pending")
    assert core.gate.evaluate(fired_key, YES, hold_s=1.0, now=0.0) == SILENT
    assert core.gate.evaluate(fired_key, YES, hold_s=1.0, now=1.0) == FIRE
    core.gate.evaluate(camera_pending, YES, hold_s=10.0, now=2.0)
    core.gate.evaluate(rear_pending, YES, hold_s=10.0, now=2.0)
    core.goals.apply([GoalOp(op="cancel", goal_id=camera.id)])

    core.clear_pending_dwell("camera")

    assert camera_pending not in core.gate._dwell_since
    assert camera_pending not in core.gate._dwell_last_supported
    assert rear_pending in core.gate._dwell_since
    assert core.gate._armed[fired_key] is False
    assert core.gate.evaluate(fired_key, YES, hold_s=1.0, now=3.0) == SILENT


def test_run_start_pause_resume_and_source_generation_break_pending_dwell():
    generation_core = Percept.create(
        vision=FakeVision(script={"open": YES}),
        discover_plugins=False,
    )
    generation_core.add_goal(duration_goal(cadence=None, hold_s=10.0))

    assert asyncio.run(generation_core.perceive_judged("open", t=0.0)) == []
    assert generation_core.advance_source_continuity("camera") == 1
    assert generation_core.source_continuity_generation("camera") == 1
    assert asyncio.run(generation_core.perceive_judged("open", t=10.0)) == []
    events = asyncio.run(generation_core.perceive_judged("open", t=20.0))
    assert [event.action for event in events] == [FIRE]

    generation_core.advance_source_continuity("camera")
    assert asyncio.run(generation_core.perceive_judged("open", t=40.0)) == []

    start_core = Percept.create(
        vision=FakeVision(script={"open": YES}),
        discover_plugins=False,
    )
    start_core.add_goal(duration_goal(hold_s=10.0, cadence=FIXED_10))
    assert asyncio.run(start_core.perceive_judged("open", t=0.0)) == []

    stats = RuntimeStats()
    incidents = asyncio.run(
        collect(
            start_core,
            [(10.0, STILL, "open"), (20.0, STILL, "open")],
            judge_dispatch="inline",
            stats=stats,
        )
    )

    assert [incident.ts for incident in incidents] == [20.0]
    assert stats.fixed_opportunities == 2
    assert stats.perceive_dispatches == 2

    pause_core = Percept.create(
        vision=FakeVision(script={"open": YES}),
        discover_plugins=False,
    )
    pause_core.add_goal(duration_goal(hold_s=10.0, cadence=FIXED_10))
    paused_state = [False]

    async def source():
        yield (0.0, STILL, "open")
        paused_state[0] = True
        yield (5.0, STILL, "open")
        # Seed a new pending proof while the worker remains paused.  Without an independent
        # resume-boundary clear, the worker's next read at t=16 would mature this t=6 proof.
        assert await pause_core.perceive_judged("open", t=6.0) == []
        paused_state[0] = False
        yield (16.0, STILL, "open")
        yield (26.0, STILL, "open")

    incidents = asyncio.run(
        collect(
            pause_core,
            source(),
            paused=lambda: paused_state[0],
            judge_dispatch="inline",
        )
    )
    assert [incident.ts for incident in incidents] == [26.0]


def test_fixed_admission_cause_is_aligned_ordered_and_closed():
    spec = WatchSpec(
        conditions=[
            WatchCondition("fixed", "fixed", threshold=0.2, cadence_s=10.0, gate_on="fixed"),
            WatchCondition("motion", "motion", threshold=0.2, gate_on="motion"),
        ]
    )
    gate = MotionGate(spec)
    first = gate.evaluate(STILL, ts=0.0)
    assert first.goal_ids == ("fixed",)
    assert first.admission_causes == ("fixed",)

    both = gate.evaluate(MOVED, ts=10.0)
    assert both.goal_ids == ("fixed", "motion")
    assert both.admission_causes == ("fixed", "motion")

    with pytest.raises(ValueError, match="aligned"):
        GateDecision(both.signal, True, both.threshold, ("a", "b"), ("fixed",))
    with pytest.raises(ValueError, match="exactly once"):
        GateDecision(both.signal, True, both.threshold, ("a", "a"), ("fixed", "motion"))
    with pytest.raises(ValueError, match="closed vocabulary"):
        GateDecision(both.signal, True, both.threshold, ("a",), ("audio",))


@pytest.mark.parametrize("dispatch", ["inline", "async"])
def test_static_door_template_apply_and_run_has_exact_fixed_accounting(dispatch):
    core = Percept.create(
        vision=FakeVision(script={"open": YES}),
        discover_plugins=False,
    )
    plan = replace(
        watch_template(
            "facility_observation@1/door_left_open",
            {"deadline_after_s": 10.0, "observation_interval_s": 5.0},
        ),
        id=f"door-{dispatch}",
    )
    receipt = apply_watch_plan(core, plan, domain_pack=FACILITY_OBSERVATION_V1)
    assert receipt.status == "applied"
    goal = core.goals.get(f"door-{dispatch}")
    assert goal.state["hold_s"] == 10.0
    assert goal.state["cadence"] == {
        "min_judge_interval": 5.0,
        "gate_on": "fixed",
    }

    frames = [(0.0, STILL, "open"), (5.0, STILL, "open"), (10.0, STILL, "open")]

    async def paced_frames():
        for frame in frames:
            yield frame
            # Let the async single-flight scheduler finish each deterministic fake call.
            for _ in range(3):
                await asyncio.sleep(0)

    stats = RuntimeStats()
    source = frames if dispatch == "inline" else paced_frames()
    incidents = asyncio.run(
        collect(core, source, judge_dispatch=dispatch, stats=stats)
    )

    assert [(incident.concern_id, incident.ts) for incident in incidents] == [
        (f"door-{dispatch}", 10.0)
    ]
    assert stats.frames == 3
    assert stats.escalations == 3
    assert stats.fixed_opportunities == 3
    assert stats.goal_admissions == 3
    assert stats.admission_by_goal == {f"door-{dispatch}": 3}
    assert stats.admission_by_cause == {"fixed": 3}
    assert stats.fixed_admits == 3
    assert stats.signal_admits == 0
    assert stats.perceive_dispatches == 3
    assert stats.judge_calls == 3
    assert stats.fires == 1
    assert stats.incidents == 1


def test_fixed_runtime_health_and_trace_retain_exact_admission_and_dwell_reasons():
    core = Percept.create(
        vision=FakeVision(script={"open": YES}),
        discover_plugins=False,
    )
    core.add_goal(duration_goal(hold_s=10.0, cadence=FIXED_5))
    health = []
    traces = []

    incidents = asyncio.run(
        collect(
            core,
            [(0.0, STILL, "open"), (5.0, STILL, "open"), (10.0, STILL, "open")],
            judge_dispatch="inline",
            on_health=health.append,
            on_trace=traces.append,
        )
    )

    assert len(incidents) == 1
    assert incidents[0].reason == "dwell_satisfied"

    ended = [event for event in health if event.kind == "runtime_ended"]
    assert len(ended) == 1
    assert {
        key: ended[0].data[key]
        for key in (
            "fixed_admits",
            "fixed_opportunities",
            "goal_admissions",
            "admission_by_goal",
            "admission_by_cause",
            "perceive_dispatches",
        )
    } == {
        "fixed_admits": 3,
        "fixed_opportunities": 3,
        "goal_admissions": 3,
        "admission_by_goal": {"door": 3},
        "admission_by_cause": {"fixed": 3},
        "perceive_dispatches": 3,
    }

    edge = [event for event in traces if event.stage == "edge.decision"]
    assert [event.data["goal_ids"] for event in edge] == [("door",)] * 3
    assert [event.data["admission_causes"] for event in edge] == [("fixed",)] * 3

    gate = [event for event in traces if event.stage == "gate.classified"]
    assert [event.data["reason"] for event in gate] == [
        "silent: dwell_pending",
        "silent: dwell_pending",
        "fire: dwell_satisfied",
    ]
    assert [event.data["gate_reason"] for event in gate] == [
        "dwell_pending",
        "dwell_pending",
        "dwell_satisfied",
    ]


def test_remote_topologies_reject_duration_with_structured_issue():
    goal = duration_goal()

    with pytest.raises(UnsupportedDurationRouteError) as compile_error:
        compile_watchspec([goal], threshold=0.2, edge_capabilities=())
    issue = compile_error.value.issues[0]
    assert issue.path == "Goal.state.hold_s"
    assert issue.code == UNSUPPORTED_DURATION_ROUTE

    core = Percept.create(mode="offline", discover_plugins=False)
    core.add_goal(goal)
    receiver = RemoteEdgeReceiver(core)
    with pytest.raises(UnsupportedDurationRouteError) as receiver_error:
        receiver.watchspec(threshold=0.2)
    issue = receiver_error.value.issues[0]
    assert issue.path == "Goal.state.hold_s"
    assert issue.code == UNSUPPORTED_DURATION_ROUTE


def test_apply_rejects_unsafe_route_without_arming():
    core = Percept.create(mode="offline", discover_plugins=False)
    plan = WatchPlan(
        id="unsafe",
        condition="the door is open",
        explicit={
            "Goal.state.hold_s": 5.0,
            "Goal.state.cadence.min_judge_interval": 1.0,
            "Goal.state.cadence.gate_on": "motion",
        },
    )

    receipt = apply_watch_plan(core, plan)

    assert receipt.status == "rejected"
    assert receipt.watch_ids == ()
    assert core.goals.get("unsafe") is None
    assert receipt.dropped_fields[0].path == "Goal.state.cadence.gate_on"
    assert receipt.dropped_fields[0].code == UNSUPPORTED_DURATION_ROUTE


def test_same_id_legacy_to_corrected_apply_preserves_live_runtime_state():
    corrected = replace(
        watch_template("facility_observation@1/door_left_open"),
        id="same-door",
    )
    legacy = WatchPlan(
        id="same-door",
        condition=corrected.condition,
        target="door",
        consequence={"miss_cost": "mild", "false_fire_cost": "trivial"},
        explicit={
            "Goal.state.rearm": "deadline",
            "Goal.state.deadline_after_s": 300.0,
            "Goal.state.episode.policy": "door_interval",
        },
    )
    core = Percept.create(mode="offline", discover_plugins=False)
    first = apply_watch_plan(core, legacy, domain_pack=FACILITY_OBSERVATION_V1)
    assert first.status == "applied"
    before_state = copy.deepcopy(core.goals.get("same-door").state)
    before_key = core._watch_runtime_keys["same-door"]

    second = apply_watch_plan(core, corrected, domain_pack=FACILITY_OBSERVATION_V1)

    assert second.status == "rejected"
    assert second.dropped_fields[0].path == "WatchPlan.id"
    assert second.dropped_fields[0].code == "active_goal_id_reconfiguration_unsupported"
    assert core.goals.get("same-door").state == before_state
    assert core._watch_runtime_keys["same-door"] == before_key
    assert "hold_s" not in core.goals.get("same-door").state
