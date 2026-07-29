"""BG4 acceptance: executable sustained-duration Watches on the in-process Python runtime."""

from __future__ import annotations

import asyncio
import copy
import math
from dataclasses import replace
from pathlib import Path

import pytest

from percept import Goal, Percept
from percept.adapters import Box, Observation, Verdict, WatchCondition, WatchSpec
from percept.config import CacheSettings, Settings
from percept.contracts.watchplan import WatchPlan
from percept.domain_pack import DomainPackError, GoalTemplate, watch_template
from percept.duration_routes import (
    UNSUPPORTED_DURATION_ROUTE,
    UnsupportedDurationRouteError,
    duration_route_issues,
)
from percept.edge.tier0 import (
    GateDecision,
    GatedWatchLoop,
    MotionGate,
    RemoteEdgeReceiver,
    compile_watchspec,
    gray_frame,
)
from percept.gate import ASK, FIRE, SILENT, Gate
from percept.packs_default import FACILITY_OBSERVATION_V1
from percept.resolver import ConfigResolver
from percept.runtime import RuntimeStats, run
from percept.watchplan_apply import _application_key, apply_watch_plan


STILL = gray_frame(8, 8, 0)
MOVED = gray_frame(8, 8, 255)
YES = Verdict(satisfied=True, confidence=0.99, evidence="door open")
NO = Verdict(satisfied=False, confidence=0.99, evidence="door closed")
UNSURE = Verdict(satisfied=True, confidence=0.1, evidence="uncertain")
FIXED = {"min_judge_interval": 10.0, "gate_on": "fixed"}


class ScriptVision:
    def __init__(self, script=None, detections=None):
        self.script = dict(script or {})
        self.detections = dict(detections or {})
        self.judge_calls = []
        self.detect_calls = 0

    async def judge(self, condition, frame, **_kwargs):
        self.judge_calls.append((condition, frame))
        value = self.script.get(frame, YES)
        if isinstance(value, BaseException):
            raise value
        return value

    async def detect(self, frame, goals):
        self.detect_calls += 1
        return list(self.detections.get(frame, ()))

    async def detect_events(self, frames, goals):
        return []


def duration_goal(
    goal_id="door",
    *,
    source="camera",
    hold_s=30.0,
    cadence=FIXED,
    extra=None,
    identity_policy="scene",
):
    """The shipped ``door_left_open`` shape: sustained hold + a fixed cadence, scene-scoped.

    ``identity_policy=None`` arms the same Watch with NO declared subject scope. That is not a
    stylistic choice: a scene-scoped concern maintains an open occurrence, and an open occurrence is
    injected into the next judge call as context, which by design bypasses the single-frame verdict
    cache. A test about the CACHE therefore has to arm a concern that holds no occurrence open.
    """
    state = {
        "type": "watch",
        "count": 0,
        "hold_s": hold_s,
    }
    if identity_policy is not None:
        state["identity_policy"] = identity_policy
    if cadence is not None:
        state["cadence"] = dict(cadence)
    if extra:
        state.update(extra)
    return Goal(goal_id, "the door is open", "door left open", source=source, state=state)


async def collect(core, source, **kwargs):
    return [incident async for incident in run(core, source=source, **kwargs)]


def test_template_receipt_and_serialization_expose_exact_policy():
    template = FACILITY_OBSERVATION_V1.template("door_left_open")
    plan = template.render(deadline_after_s=45.0, observation_interval_s=7.5)
    expected = {
        "Goal.state.hold_s": 45.0,
        "Goal.state.cadence.min_judge_interval": 7.5,
        "Goal.state.cadence.gate_on": "fixed",
        "Goal.state.identity_policy": "scene",
    }
    for path, value in expected.items():
        assert plan.explicit[path] == value
    assert "Goal.state.deadline_after_s" not in plan.explicit
    assert "Goal.state.rearm" not in plan.explicit
    assert not any(path.startswith("Goal.state.episode.") for path in plan.explicit)
    serialized = template.to_dict()
    assert serialized["hold_s"] == "{deadline_after_s}"
    assert serialized["cadence"] == {
        "min_judge_interval": "{observation_interval_s}",
        "gate_on": "fixed",
    }

    core = Percept.create(mode="offline", discover_plugins=False)
    receipt = apply_watch_plan(core, plan, domain_pack=FACILITY_OBSERVATION_V1)
    assert receipt.status == "applied"
    assert set(expected).issubset(receipt.accepted_fields)
    goal = core.goals.get(receipt.watch_ids[0])
    assert goal.state["hold_s"] == 45.0
    assert goal.state["cadence"] == {
        "min_judge_interval": 7.5,
        "gate_on": "fixed",
    }


def test_goal_template_append_only_positional_compatibility_and_unset_fields():
    # The prior final positional field was identity_policy. New fields append after it.
    template = GoalTemplate(
        "legacy", "condition", {}, None, None, (), (), {}, None, None,
        "camera", "notes", "scene",
    )
    assert template.hold_s is None
    assert template.cadence is None
    assert template.to_dict()["hold_s"] is None
    assert template.to_dict()["cadence"] is None
    assert not any(path.startswith("Goal.state.cadence.") for path in template.render().explicit)


@pytest.mark.parametrize("bad", [True, 0, -1, math.nan, math.inf, -math.inf, "15"])
def test_template_rejects_invalid_duration_defaults_and_overrides(bad):
    with pytest.raises(DomainPackError, match=UNSUPPORTED_DURATION_ROUTE):
        GoalTemplate(
            name="bad",
            condition="bad",
            parameters={"hold": bad, "interval": 5.0},
            hold_s="{hold}",
            cadence={"min_judge_interval": "{interval}", "gate_on": "fixed"},
        )
    template = GoalTemplate(
        name="override",
        condition="bad",
        parameters={"hold": 10.0, "interval": 5.0},
        hold_s="{hold}",
        cadence={"min_judge_interval": "{interval}", "gate_on": "fixed"},
    )
    with pytest.raises(DomainPackError, match=UNSUPPORTED_DURATION_ROUTE):
        template.render(hold=bad)


@pytest.mark.parametrize("bad", [True, 0, -1, math.nan, math.inf, -math.inf, "15"])
def test_template_rejects_invalid_cadence_even_without_hold(bad):
    with pytest.raises(DomainPackError, match=UNSUPPORTED_DURATION_ROUTE):
        GoalTemplate(
            name="bad-cadence",
            condition="bad",
            parameters={"interval": bad},
            cadence={"min_judge_interval": "{interval}", "gate_on": "motion"},
        )


def test_shared_validator_rejects_routes_schedule_and_nonfixed_worker_shape():
    cases = [
        ({"type": "transition", "from": "a", "to": "b", "hold_s": 5, "cadence": FIXED},
         "Goal.state.hold_s"),
        ({"type": "process", "hold_s": 5, "cadence": FIXED}, "Goal.state.hold_s"),
        ({"type": "event", "hold_s": 5, "cadence": FIXED}, "Goal.state.hold_s"),
        ({"type": "count", "hold_s": 5, "cadence": FIXED}, "Goal.state.hold_s"),
        ({"type": "rhythm", "hold_s": 5, "cadence": FIXED}, "Goal.state.hold_s"),
        ({"type": "scheduler", "hold_s": 5, "cadence": FIXED}, "Goal.state.hold_s"),
        ({"type": "watch", "hold_s": 5, "cadence": FIXED, "schedule": {"windows": [[0, 1]]}},
         "Goal.state.schedule"),
        ({"type": "watch", "hold_s": 5, "cadence": {"min_judge_interval": 1, "gate_on": "motion"}},
         "Goal.state.cadence.gate_on"),
        ({"type": "watch", "hold_s": 5}, "Goal.state.cadence.min_judge_interval"),
    ]
    for state, path in cases:
        issues = duration_route_issues(
            state,
            require_worker_cadence=True,
            canonical_goal=False,
        )
        assert issues
        assert any(issue.path == path and issue.code == UNSUPPORTED_DURATION_ROUTE for issue in issues)


def test_structured_apply_rejects_unsafe_duration_with_exact_diagnostic_and_arms_nothing():
    core = Percept.create(mode="offline", discover_plugins=False)
    plan = WatchPlan(
        condition="unsafe",
        explicit={
            "Goal.state.hold_s": 5.0,
            "Goal.state.cadence.min_judge_interval": 1.0,
            "Goal.state.cadence.gate_on": "motion",
        },
    )
    receipt = apply_watch_plan(core, plan)
    assert receipt.status == "rejected"
    assert receipt.watch_ids == ()
    assert core.goals.all() == []
    assert receipt.dropped_fields[0].path == "Goal.state.cadence.gate_on"
    assert receipt.dropped_fields[0].code == UNSUPPORTED_DURATION_ROUTE


def test_manual_worker_seams_fail_before_source_detect_or_judge():
    unsafe = duration_goal(cadence=None)
    vision = ScriptVision()
    core = Percept.create(vision=vision, discover_plugins=False)
    core.add_goal(unsafe)

    with pytest.raises(UnsupportedDurationRouteError):
        compile_watchspec([unsafe], threshold=0.1)
    with pytest.raises(UnsupportedDurationRouteError):
        GatedWatchLoop(core)
    # Direct bring-your-own-capture owns timing and accepts a valid hold without cadence.
    asyncio.run(core.perceive_judged("frame", t=0.0))
    detect_before_worker = vision.detect_calls
    judge_before_worker = list(vision.judge_calls)
    invalid_direct = Percept.create(vision=ScriptVision(), discover_plugins=False)
    invalid_direct.add_goal(duration_goal("bad-direct", hold_s=True, cadence=None))
    with pytest.raises(UnsupportedDurationRouteError):
        asyncio.run(invalid_direct.perceive_judged("frame", t=0.0))

    pulls = 0

    def source():
        nonlocal pulls
        pulls += 1
        yield (0.0, STILL, "frame")

    with pytest.raises(UnsupportedDurationRouteError):
        asyncio.run(collect(core, source(), judge_dispatch="inline"))
    assert pulls == 0
    assert vision.detect_calls == detect_before_worker
    assert vision.judge_calls == judge_before_worker


def test_worker_rejects_duration_core_without_continuity_hook_before_source_pull():
    base = Percept.create(vision=ScriptVision(), discover_plugins=False)
    base.add_goal(duration_goal())

    class CoreWithoutContinuityHook:
        def __getattr__(self, name):
            if name == "clear_pending_dwell":
                raise AttributeError(name)
            return getattr(base, name)

    pulls = 0

    def poison_source():
        nonlocal pulls
        pulls += 1
        raise AssertionError("duration topology must fail before source pull")
        yield  # pragma: no cover

    with pytest.raises(
        UnsupportedDurationRouteError,
        match=r"Runtime\.clear_pending_dwell",
    ):
        asyncio.run(
            collect(
                CoreWithoutContinuityHook(),
                poison_source(),
                judge_dispatch="inline",
            )
        )
    assert pulls == 0


def test_direct_capture_allows_valid_hold_without_cadence_and_breaks_unknown():
    vision = ScriptVision({"yes": YES, "ask": UNSURE})
    core = Percept.create(vision=vision, discover_plugins=False)
    core.add_goal(duration_goal(cadence=None, hold_s=10.0))
    assert asyncio.run(core.perceive_judged("yes", t=0.0)) == []
    asks = asyncio.run(core.perceive_judged("ask", t=3.0))
    assert [event.action for event in asks] == [ASK]
    assert asyncio.run(core.perceive_judged("yes", t=10.0)) == []
    fires = asyncio.run(core.perceive_judged("yes", t=20.0))
    assert [event.action for event in fires] == [FIRE]


def test_direct_capture_rejects_malformed_cadence_only_before_provider_work():
    vision = ScriptVision()
    core = Percept.create(vision=vision, discover_plugins=False)
    core.add_goal(
        Goal(
            "bad-cadence",
            "the door is open",
            "door open",
            state={
                "type": "watch",
                "cadence": {
                    "min_judge_interval": math.nan,
                    "gate_on": "fixed",
                },
            },
        )
    )
    with pytest.raises(
        UnsupportedDurationRouteError,
        match=r"Goal\.state\.cadence\.min_judge_interval",
    ):
        asyncio.run(core.perceive_judged("frame", t=0.0))
    assert vision.detect_calls == 0
    assert vision.judge_calls == []


def test_gate_support_gap_and_pending_only_clear_do_not_rearm_fired_state():
    gate = Gate()
    key = ("door", "scene")
    assert gate.evaluate(key, YES, hold_s=10, max_support_gap_s=4, now=0) == SILENT
    assert gate.evaluate(key, YES, hold_s=10, max_support_gap_s=4, now=4) == SILENT
    assert gate.evaluate(key, YES, hold_s=10, max_support_gap_s=4, now=8) == SILENT
    assert gate.evaluate(key, YES, hold_s=10, max_support_gap_s=4, now=10) == FIRE
    gate.clear_pending_dwell(key)
    assert gate.evaluate(key, YES, hold_s=10, max_support_gap_s=4, now=30) == SILENT
    assert gate.evaluate(key, NO, hold_s=10, max_support_gap_s=4, now=31) == SILENT
    assert gate.evaluate(key, YES, hold_s=10, max_support_gap_s=4, now=32) == SILENT
    assert gate.evaluate(key, YES, hold_s=10, max_support_gap_s=4, now=36.0001) == SILENT
    assert gate.evaluate(key, YES, hold_s=10, max_support_gap_s=4, now=40) == SILENT
    assert gate.evaluate(key, YES, hold_s=10, max_support_gap_s=4, now=44) == SILENT
    assert gate.evaluate(key, YES, hold_s=10, max_support_gap_s=4, now=46.0001) == FIRE


def test_source_clear_includes_inactive_matching_goals_and_preserves_other_source():
    core = Percept.create(mode="offline", discover_plugins=False)
    camera = duration_goal("camera-door", source="camera")
    other = duration_goal("other-door", source="camera:other")
    core.add_goal(camera)
    core.add_goal(other)
    camera_key = (camera.id, "scene")
    other_key = (other.id, "scene")
    # Simulate an inactive retained Watch; the registry has complete/cancel but no pause state.
    core.goals._status[camera.id] = "complete"
    core.gate.evaluate(camera_key, YES, hold_s=30, max_support_gap_s=20, now=0)
    core.gate.evaluate(other_key, YES, hold_s=30, max_support_gap_s=20, now=0)
    core.advance_source_continuity("camera")
    assert camera_key not in core.gate._dwell_since
    assert other_key in core.gate._dwell_since


def test_provider_timeout_and_malformed_reply_break_pending_dwell():
    for broken in (TimeoutError("provider timeout"), object()):
        vision = ScriptVision({"yes": YES, "broken": broken})
        core = Percept.create(vision=vision, discover_plugins=False)
        core.add_goal(duration_goal(cadence=None, hold_s=10.0))
        assert asyncio.run(core.perceive_judged("yes", t=0)) == []
        assert asyncio.run(core.perceive_judged("broken", t=5)) == []
        assert asyncio.run(core.perceive_judged("yes", t=10)) == []
        fires = asyncio.run(core.perceive_judged("yes", t=20))
        assert [event.action for event in fires] == [FIRE]


def test_audited_cache_hit_degradation_breaks_prior_dwell():
    settings = replace(Settings(), cache=CacheSettings(enabled=True, threshold=0.0, audit_rate=1.0))

    class AuditVision(ScriptVision):
        async def judge(self, condition, frame, **_kwargs):
            self.judge_calls.append((condition, frame))
            if len(self.judge_calls) > 1:
                raise TimeoutError("audit failed")
            return YES

    vision = AuditVision()
    core = Percept.create(
        vision=vision,
        settings=settings,
        audit=lambda _key: True,
        discover_plugins=False,
    )
    # No declared subject scope: this test is about an AUDITED CACHE HIT, and a concern holding an
    # occurrence open sends that occurrence as judge context, which never reads the frame cache.
    core.add_goal(duration_goal(cadence=None, hold_s=10.0, identity_policy=None))
    frame = tuple(tuple(0 for _ in range(9)) for _ in range(8))
    assert asyncio.run(core.perceive_judged(frame, t=0)) == []
    assert asyncio.run(core.perceive_judged(frame, t=5)) == []
    assert core.metrics.get("cache.hit") == 1
    assert core.metrics.get("judge.degraded") == 1
    # The cached read may start new support at t=5, but the pre-audit t=0 timer was discarded.
    assert asyncio.run(core.perceive_judged(frame, t=10)) == []


def test_gate_decision_cause_contract_and_legacy_normalization():
    signal = MotionGate(WatchSpec(conditions=[])).evaluate(STILL, ts=0).signal
    legacy = GateDecision(signal, True, 0.1, ("g",))
    assert legacy.admission_causes == ("motion",)
    with pytest.raises(ValueError, match="aligned"):
        GateDecision(signal, True, 0.1, ("a", "b"), ("fixed",))
    with pytest.raises(ValueError, match="exactly once"):
        GateDecision(signal, True, 0.1, ("a", "a"), ("fixed", "motion"))
    with pytest.raises(ValueError, match="closed vocabulary"):
        GateDecision(signal, True, 0.1, ("a",), ("coast",))  # type: ignore[arg-type]


def test_mixed_fixed_motion_attribution_is_watchspec_ordered_and_regate_keeps_throttle():
    spec = WatchSpec(
        conditions=[
            WatchCondition("fixed", "fixed", threshold=0.1, cadence_s=10, gate_on="fixed"),
            WatchCondition("motion", "motion", threshold=0.1, gate_on="motion"),
        ]
    )
    first = MotionGate(spec)
    d0 = first.evaluate(STILL, ts=0)
    assert d0.goal_ids == ("fixed",)
    assert d0.admission_causes == ("fixed",)
    second = MotionGate(spec)
    second.adopt_state(first)
    d5 = second.evaluate(STILL, ts=5)
    assert d5.goal_ids == ()
    d10 = second.evaluate(MOVED, ts=10)
    assert d10.goal_ids == ("fixed", "motion")
    assert d10.admission_causes == ("fixed", "motion")


def test_remote_edge_topology_rejects_duration_watch_explicitly():
    goal = duration_goal()
    with pytest.raises(UnsupportedDurationRouteError, match="remote percept-edge"):
        compile_watchspec([goal], threshold=0.1, edge_capabilities=())
    core = Percept.create(mode="offline", discover_plugins=False)
    core.add_goal(goal)
    receiver = RemoteEdgeReceiver(core)
    with pytest.raises(UnsupportedDurationRouteError, match="in-process Python"):
        receiver.watchspec(threshold=0.1)


def test_static_fixed_runtime_fires_once_with_exact_accounting_and_focus_turnover():
    observations = {
        "a": [Observation("person", Box(0.0, 0.0, 0.2, 0.2), 0.99)],
        "b": [Observation("person", Box(0.8, 0.8, 0.2, 0.2), 0.99)],
    }
    vision = ScriptVision({"a": YES, "b": YES}, detections=observations)
    core = Percept.create(vision=vision, discover_plugins=False)
    plan = watch_template(
        "facility_observation@1/door_left_open",
        {"deadline_after_s": 30.0, "observation_interval_s": 10.0},
    )
    receipt = apply_watch_plan(core, plan, domain_pack=FACILITY_OBSERVATION_V1)
    stats = RuntimeStats()
    source = [(0.0, STILL, "a"), (10.0, STILL, "b"), (20.0, STILL, "a"),
              (30.0, STILL, "b"), (40.0, STILL, "a")]
    incidents = asyncio.run(collect(core, source, judge_dispatch="inline", stats=stats))
    assert len(incidents) == 1
    assert incidents[0].concern_id == receipt.watch_ids[0]
    assert stats.frames == 5
    assert stats.fixed_opportunities == 5
    assert stats.goal_admissions == 5
    assert stats.admission_by_goal == {receipt.watch_ids[0]: 5}
    assert stats.admission_by_cause == {"fixed": 5}
    assert stats.perceive_dispatches == 5
    assert stats.judge_calls == 5
    assert stats.fixed_admits == 5
    assert stats.signal_admits == 0
    assert stats.fires == 1
    assert stats.incidents == 1
    assert core.metrics.get("judge.latency.count") == 5
    assert core.metrics.get("cache.hit") == 0
    assert core.metrics.get("direct_judge.hit") == 0


def test_plan_hash_application_key_change_and_same_id_reconfiguration_preserves_legacy_state():
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
    resolver = ConfigResolver()
    legacy_resolved = resolver.resolve(legacy, domain_pack=FACILITY_OBSERVATION_V1)
    corrected_resolved = resolver.resolve(corrected, domain_pack=FACILITY_OBSERVATION_V1)
    assert legacy_resolved.plan_hash != corrected_resolved.plan_hash
    assert legacy_resolved.component_snapshot_hash == corrected_resolved.component_snapshot_hash
    assert _application_key(legacy_resolved) != _application_key(corrected_resolved)

    core = Percept.create(mode="offline", discover_plugins=False)
    first = apply_watch_plan(core, legacy, domain_pack=FACILITY_OBSERVATION_V1)
    assert first.status == "applied"
    before = copy.deepcopy(core.goals.get("same-door").state)
    second = apply_watch_plan(core, corrected, domain_pack=FACILITY_OBSERVATION_V1)
    assert second.status == "rejected"
    assert second.dropped_fields[0].code == "active_goal_id_reconfiguration_unsupported"
    assert core.goals.get("same-door").state == before


def test_public_duration_contract_states_topology_migration_restart_and_qualification():
    repo = Path(__file__).resolve().parents[3]
    contract = (repo / "docs/watches/sustained-duration-watches.md").read_text()
    edge = (repo / "docs/admission/edge-parity.md").read_text()
    scope = (repo / "docs/incidents/scope-and-persistence.md").read_text()

    assert "in-process Python" in contract
    assert "new ID" in contract
    assert "CompiledWatchSnapshotUnsupportedError" in contract
    assert "not quality-qualified" in contract
    assert "remote JavaScript edge" in edge
    # The old `docs/tutorials/choose-your-path.md` chooser page carried the fourth guard
    # ("sustained-duration Watches" named as out of scope for the JS edge). That page was removed
    # by the object-first docs restructure; `edge` above now carries the same claim.
    assert "Sustained-duration Watches" in edge
    assert "pending `hold_s` dwell starts again" in scope
