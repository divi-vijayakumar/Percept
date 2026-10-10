"""Named scenarios for the decision reducer, then properties over random input sequences.

Each scenario is a short story with explicit wall-clock times. Small numbers keep the
stories readable; the reducer only ever compares and adds them.

Every test runs against each implementation in ``IMPLEMENTATIONS``. Today that is the
reference model; the real reducer joins the list when it exists, and must pass unchanged.
"""

from __future__ import annotations

import json
import random
from dataclasses import replace

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import reference_model
from percept.contracts import Condition, Observation, Usage

IMPLEMENTATIONS = [reference_model]

pytestmark = pytest.mark.parametrize("impl", IMPLEMENTATIONS, ids=lambda m: m.__name__)


# --------------------------------------------------------------------------- harness


class Stream:
    """One stream's reducer state, driven the way the runtime will drive it."""

    def __init__(self, impl, *conditions: Condition, stream_id: str = "cam-1", max_keys: int = 64,
                 created_at: float = 0.0):
        self.impl = impl
        self.stream_id = stream_id
        self.conditions = {c.id: c for c in conditions}
        self.state = impl.new_state(stream_id, created_at=created_at, max_keys=max_keys)
        self.epoch = 0
        self.results = []
        self._requests = 0
        for c in conditions:
            self.feed(impl.ConditionChanged(c.id), at=0.0)

    # driving

    def feed(self, inp, *, at: float):
        result = self.impl.step(self.state, inp, at, self.conditions)
        self.state = result.state
        self.results.append(result)
        if isinstance(inp, self.impl.EpochStarted) and inp.stream_epoch > self.epoch:
            self.epoch = inp.stream_epoch
        return result

    def dispatch(self, condition_id: str, subject: str = "scene", purpose: str = "detect") -> dict:
        """Tag a judge request the way the runtime does at dispatch time."""
        self._requests += 1
        return dict(condition_id=condition_id, subject=subject, request_id=f"r-{self._requests}",
                    stream_epoch=self.epoch, condition_version=self.conditions[condition_id].version,
                    occurrence_gen=self.impl.gen_for(self.state, condition_id, subject), purpose=purpose)

    def reply(self, request: dict, answer: str, confidence: float = 0.9, *, at: float,
              evidence: tuple[float, float] | None = None, failure: str | None = None, usage: Usage = Usage()):
        if failure is not None:
            answer, confidence = "unknown", 0.0
        o = Observation(answer=answer, confidence=confidence, reason=f"model says {answer}",
                        stream_id=self.stream_id, evidence_t=evidence or (at - 1.0, at - 0.5),
                        judge="fake", failure=failure, **request)
        return self.feed(self.impl.Observed(o, usage), at=at)

    def observe(self, condition_id: str, answer: str, confidence: float = 0.9, *, at: float,
                subject: str = "scene", evidence: tuple[float, float] | None = None, failure: str | None = None):
        """A detect request dispatched and answered at once."""
        return self.reply(self.dispatch(condition_id, subject), answer, confidence, at=at,
                          evidence=evidence, failure=failure)

    def verify_reply(self, request, answer: str, confidence: float = 0.9, *, at: float,
                     failure: str | None = None):
        """Answer a RequestVerify effect with an observation judged on its own evidence."""
        tags = dict(condition_id=request.condition_id, subject=request.subject, request_id=f"v-{at}",
                    stream_epoch=request.stream_epoch, condition_version=request.condition_version,
                    occurrence_gen=request.occurrence_gen, purpose="verify")
        evidence = request.interval or (at - 0.5, at)
        return self.reply(tags, answer, confidence, at=at, evidence=evidence, failure=failure)

    def fire(self, condition_id: str, kind: str, *, at: float, subject: str = "scene"):
        return self.feed(self.impl.TimerFired(condition_id, subject, kind), at=at)

    def update(self, condition: Condition, *, at: float):
        self.conditions = {**self.conditions, condition.id: condition}
        return self.feed(self.impl.ConditionChanged(condition.id), at=at)

    # reading

    def phase(self, condition_id: str, subject: str = "scene") -> str:
        return self.impl.phase_of(self.state, condition_id, subject)

    def snapshot(self) -> str:
        return json.dumps(self.impl.to_dict(self.state), sort_keys=True)

    def restore(self, snapshot: str) -> None:
        self.state = self.impl.from_dict(json.loads(snapshot))


def types(result) -> list[str]:
    return [d.type for d in result.decisions]


def effects(result, cls) -> list:
    return [e for e in result.effects if isinstance(e, cls)]


def _traces(result) -> list:
    # by name, so the helper works for every implementation's own Trace class
    return [e for e in result.effects if type(e).__name__ == "Trace"]


def fates(result) -> list[str]:
    return [e.fate for e in _traces(result) if e.fate is not None]


def trace_events(result) -> list[str]:
    return [e.event for e in _traces(result)]


def door(**kw) -> Condition:
    return Condition("door", "The door is open", kind="state", **kw)


def fall(**kw) -> Condition:
    return Condition("fall", "A person falls to the floor", kind="event", **kw)


# --------------------------------------------------------------------------- the first yes


def test_state_first_confident_yes_with_confirm_1_is_met_in_the_same_step(impl):
    s = Stream(impl, door())
    r = s.observe("door", "yes", 0.8, at=10.0)
    assert types(r) == ["met"]
    assert s.phase("door") == "holding"
    assert effects(r, impl.ScheduleTimer) == [impl.ScheduleTimer("door", "scene", "recheck", 70.0)]


def test_event_first_confident_yes_requests_verification_of_that_interval_in_the_same_step(impl):
    s = Stream(impl, fall())
    r = s.observe("fall", "yes", 0.78, at=10.0, evidence=(8.0, 9.5))
    assert types(r) == []
    assert s.phase("fall") == "verifying"
    [verify] = effects(r, impl.RequestVerify)
    assert verify.mode == "original" and verify.interval == (8.0, 9.5)
    assert verify.deadline == 40.0 and verify.occurrence_gen == impl.gen_for(s.state, "fall", "scene")


def test_confirm_2_within_the_window_is_met(impl):
    s = Stream(impl, door(confirm=2, confirm_window_s=20.0))
    assert types(s.observe("door", "yes", at=10.0)) == []
    assert s.phase("door") == "confirming"
    assert types(s.observe("door", "yes", at=25.0)) == ["met"]


def test_confirm_2_second_yes_after_the_window_restarts_the_window(impl):
    s = Stream(impl, door(confirm=2, confirm_window_s=20.0))
    first = s.observe("door", "yes", at=10.0)
    late = s.observe("door", "yes", at=31.0)          # the window ended at 30
    assert types(late) == []
    assert "flicker" in trace_events(late)
    assert s.phase("door") == "confirming"            # the late yes opened a new window
    assert late.decisions == () and first.decisions == ()
    assert types(s.observe("door", "yes", at=45.0)) == ["met"]


def test_a_confident_no_while_confirming_returns_a_state_to_idle(impl):
    s = Stream(impl, door(confirm=2))
    s.observe("door", "yes", at=10.0)
    s.observe("door", "no", at=12.0)
    assert s.phase("door") == "idle"
    assert types(s.observe("door", "yes", at=14.0)) == []   # starts counting again


# --------------------------------------------------------------------------- event verification


def test_fall_then_the_person_lies_still_verification_of_the_original_interval_wins(impl):
    s = Stream(impl, fall())
    [verify] = effects(s.observe("fall", "yes", 0.78, at=1.2, evidence=(0.0, 1.0)), impl.RequestVerify)

    # While verification is in flight, a detect on later frames sees a person lying still.
    still = s.observe("fall", "no", 0.95, at=1.8, evidence=(1.5, 1.8))
    assert types(still) == [] and s.phase("fall") == "verifying"

    met = s.verify_reply(verify, "yes", 0.88, at=2.3)
    assert types(met) == ["met"]
    assert met.decisions[0].t_observed == 1.0         # when the fall was seen, not when it was verified
    assert s.phase("fall") == "reported"

    # And a later "no" for the same subject does not reject the reported event.
    later = s.observe("fall", "no", 0.95, at=5.0)
    assert types(later) == [] and s.phase("fall") == "reported"


def test_an_event_with_confirm_2_verifies_the_span_of_both_yeses_and_ignores_a_no_between(impl):
    s = Stream(impl, fall(confirm=2))
    s.observe("fall", "yes", at=10.0, evidence=(8.0, 9.0))
    assert s.phase("fall") == "candidate"
    s.observe("fall", "no", 0.95, at=11.0)             # the moment may have passed; not evidence against
    assert s.phase("fall") == "candidate"
    [verify] = effects(s.observe("fall", "yes", at=12.0, evidence=(10.5, 11.5)), impl.RequestVerify)
    assert verify.interval == (8.0, 11.5)
    assert types(s.verify_reply(verify, "yes", at=13.0)) == ["met"]


def test_a_confident_no_on_the_original_interval_rejects_the_candidate(impl):
    s = Stream(impl, fall())
    [verify] = effects(s.observe("fall", "yes", at=10.0), impl.RequestVerify)
    r = s.verify_reply(verify, "no", 0.9, at=11.0)
    assert types(r) == [] and "verify_rejected" in trace_events(r)
    assert s.phase("fall") == "idle"


def test_two_verification_timeouts_record_no_negative_then_the_deadline_is_uncertain(impl):
    s = Stream(impl, fall(candidate_deadline_s=30.0))
    [verify] = effects(s.observe("fall", "yes", 0.8, at=10.0), impl.RequestVerify)
    for t in (18.0, 26.0):
        r = s.verify_reply(verify, "unknown", at=t, failure="timeout")
        assert types(r) == [] and fates(r) == ["failed"]
        assert s.phase("fall") == "verifying"
    r = s.fire("fall", "verify", at=40.0)
    assert types(r) == ["uncertain"]
    assert r.decisions[0].reason == "unverified"
    assert s.phase("fall") == "idle"


def test_unverified_with_on_unverified_health_emits_only_a_health_effect(impl):
    s = Stream(impl, fall(candidate_deadline_s=30.0, on_unverified="health"))
    s.observe("fall", "yes", at=10.0)
    r = s.fire("fall", "verify", at=40.0)
    assert r.decisions == ()
    [health] = effects(r, impl.EmitHealth)
    assert health.health.condition_id == "fall" and "unverified" in health.health.detail


def test_the_deadline_applies_even_if_its_timer_never_fires(impl):
    s = Stream(impl, fall(candidate_deadline_s=30.0))
    [verify] = effects(s.observe("fall", "yes", at=10.0), impl.RequestVerify)
    r = s.verify_reply(verify, "yes", at=41.0)        # too late: the deadline passed at 40
    assert types(r) == ["uncertain"]
    assert fates(r) == ["stale"]                      # the late verification is about a closed candidate


def test_the_verifier_answering_unknown_resolves_the_candidate_as_unverified(impl):
    s = Stream(impl, fall())
    [verify] = effects(s.observe("fall", "yes", at=10.0), impl.RequestVerify)
    r = s.verify_reply(verify, "unknown", 0.9, at=11.0)
    assert types(r) == ["uncertain"] and s.phase("fall") == "idle"


def test_a_weak_verification_answer_is_no_information(impl):
    s = Stream(impl, fall())
    [verify] = effects(s.observe("fall", "yes", at=10.0), impl.RequestVerify)
    r = s.verify_reply(verify, "no", 0.3, at=11.0)
    assert types(r) == [] and fates(r) == ["judged_weak"] and s.phase("fall") == "verifying"


def test_fast_path_skips_verification_only_when_enabled_and_confident_enough(impl):
    s = Stream(impl, fall(fast_path_confidence=0.9))
    assert types(s.observe("fall", "yes", 0.85, at=10.0)) == []
    assert s.phase("fall") == "verifying"

    s = Stream(impl, fall(fast_path_confidence=0.9))
    r = s.observe("fall", "yes", 0.95, at=10.0)
    assert types(r) == ["met"] and effects(r, impl.RequestVerify) == []


def test_a_state_with_fresh_verification_holds_only_after_the_fresh_frame_agrees(impl):
    s = Stream(impl, door(verify="fresh"))
    [verify] = effects(s.observe("door", "yes", at=10.0), impl.RequestVerify)
    assert verify.mode == "fresh" and verify.interval is None
    assert s.phase("door") == "verifying"
    assert types(s.verify_reply(verify, "yes", at=12.0)) == ["met"]
    assert s.phase("door") == "holding"


# --------------------------------------------------------------------------- confidence


def test_a_weak_no_while_holding_changes_nothing(impl):
    s = Stream(impl, door(clear_after_no=2, emit_cleared=True))
    s.observe("door", "yes", at=10.0)
    s.observe("door", "no", 0.9, at=20.0)             # 1 of 2
    before = impl.decision_view(s.state)
    weak = s.observe("door", "no", 0.4, at=30.0)      # below min_confidence: no information
    assert types(weak) == [] and fates(weak) == ["judged_weak"]
    assert impl.decision_view(s.state) == before
    assert s.phase("door") == "holding"
    assert types(s.observe("door", "no", 0.9, at=40.0)) == ["cleared"]   # 2 of 2


def test_a_weak_yes_in_idle_does_nothing(impl):
    for condition in (door(), fall()):
        s = Stream(impl, condition)
        before = impl.decision_view(s.state)
        r = s.observe(condition.id, "yes", 0.5, at=10.0)
        assert r.decisions == () and fates(r) == ["judged_weak"]
        assert effects(r, impl.RequestVerify) == []
        assert impl.decision_view(s.state) == before


def test_model_unknown_in_idle_is_uncertain_once_per_run_of_unknowns(impl):
    s = Stream(impl, door())
    assert types(s.observe("door", "unknown", at=10.0)) == ["uncertain"]
    assert types(s.observe("door", "unknown", at=60.0)) == []        # same run
    s.observe("door", "no", at=70.0)                                  # a confident answer ends it
    r = s.observe("door", "unknown", at=80.0)
    assert types(r) == ["uncertain"] and r.decisions[0].reason == "model_unknown"


# --------------------------------------------------------------------------- staleness


def test_resolve_while_a_request_is_in_flight_rejects_the_late_yes_as_stale(impl):
    s = Stream(impl, door())
    occurrence = s.observe("door", "yes", at=10.0).decisions[0].occurrence_id
    in_flight = s.dispatch("door")                    # a recheck leaves at 20
    s.feed(impl.Resolve(occurrence), at=21.0)
    assert s.phase("door") == "idle"
    late = s.reply(in_flight, "yes", 0.95, at=25.0)
    assert late.decisions == () and fates(late) == ["stale"]
    assert s.phase("door") == "idle"                  # nothing resurrected


def test_a_condition_update_makes_observations_for_the_old_version_stale(impl):
    s = Stream(impl, door(emit_cleared=True))
    s.observe("door", "yes", at=10.0)
    in_flight = s.dispatch("door")
    r = s.update(replace(s.conditions["door"], text="The front door is open", version=2), at=20.0)
    assert types(r) == ["cleared"] and r.decisions[0].reason == "condition_changed"
    assert s.phase("door") == "idle"
    late = s.reply(in_flight, "yes", at=21.0)
    assert late.decisions == () and fates(late) == ["stale"]
    assert types(s.observe("door", "yes", at=22.0)) == ["met"]     # judged against version 2


def test_a_verify_answer_when_nothing_is_being_verified_is_stale(impl):
    s = Stream(impl, door())
    r = s.reply(s.dispatch("door", purpose="verify"), "yes", at=10.0)
    assert r.decisions == () and fates(r) == ["stale"] and s.phase("door") == "idle"


# --------------------------------------------------------------------------- restart


def test_restart_with_a_held_state_keeps_it_held_and_rechecks_without_a_duplicate_met(impl):
    s = Stream(impl, door())
    met = s.observe("door", "yes", at=10.0)
    in_flight = s.dispatch("door")                    # epoch 0, lost in the crash
    snapshot = s.snapshot()

    s.restore(snapshot)                               # a new process
    r = s.feed(impl.EpochStarted(1, restart=True), at=500.0)
    assert r.decisions == ()
    assert s.phase("door") == "holding"
    [recheck] = effects(r, impl.RequestRecheck)
    assert recheck.occurrence_id == met.decisions[0].occurrence_id
    assert impl.ScheduleTimer("door", "scene", "recheck", 560.0) in effects(r, impl.ScheduleTimer)

    old = s.reply(in_flight, "yes", at=501.0)
    assert old.decisions == () and fates(old) == ["stale"]
    assert s.reply(s.dispatch("door"), "yes", at=502.0).decisions == ()   # still the same occurrence


def test_restart_resolves_an_event_being_verified_as_unverified(impl):
    s = Stream(impl, fall())
    s.observe("fall", "yes", at=10.0)
    s.restore(s.snapshot())
    r = s.feed(impl.EpochStarted(1, restart=True), at=12.0)
    assert types(r) == ["uncertain"] and s.phase("fall") == "idle"


def test_a_reconnect_keeps_verifying_the_original_interval(impl):
    s = Stream(impl, fall())
    [verify] = effects(s.observe("fall", "yes", at=10.0), impl.RequestVerify)
    s.feed(impl.EpochStarted(1, restart=False), at=11.0)
    assert s.phase("fall") == "verifying"
    assert types(s.verify_reply(verify, "yes", at=12.0)) == ["met"]   # its evidence is from epoch 0


# --------------------------------------------------------------------------- holding and clearing


def test_max_open_s_emits_expired_not_cleared_and_never_a_repeat_met(impl):
    s = Stream(impl, door(max_open_s=100.0, recheck_every_s=30.0, emit_cleared=True))
    occurrence = s.observe("door", "yes", at=10.0).decisions[0].occurrence_id
    recheck = s.fire("door", "recheck", at=40.0)
    assert len(effects(recheck, impl.RequestRecheck)) == 1
    assert s.observe("door", "yes", at=41.0).decisions == ()
    r = s.fire("door", "max_open", at=110.0)
    assert types(r) == ["expired"] and r.decisions[0].occurrence_id == occurrence
    assert s.phase("door") == "holding"
    assert s.observe("door", "yes", at=120.0).decisions == ()
    assert s.fire("door", "max_open", at=300.0).decisions == ()         # once only


@pytest.mark.parametrize("emit_cleared", [True, False])
def test_a_state_clears_after_clear_after_no_confident_nos(impl, emit_cleared):
    s = Stream(impl, door(clear_after_no=3, emit_cleared=emit_cleared))
    s.observe("door", "yes", at=10.0)
    s.observe("door", "no", at=20.0)
    s.observe("door", "no", at=30.0)
    s.observe("door", "yes", at=40.0)                 # a confident yes resets the count
    s.observe("door", "no", at=50.0)
    s.observe("door", "no", at=60.0)
    assert s.phase("door") == "holding"
    r = s.observe("door", "no", at=70.0)
    assert s.phase("door") == "idle"
    assert types(r) == (["cleared"] if emit_cleared else [])
    if emit_cleared:
        assert r.decisions[0].reason == "no"


@pytest.mark.parametrize("emit_cleared", [True, False])
def test_false_alarm_closes_the_occurrence_and_stores_a_counter_example(impl, emit_cleared):
    s = Stream(impl, door(emit_cleared=emit_cleared))
    occurrence = s.observe("door", "yes", at=10.0).decisions[0].occurrence_id
    in_flight = s.dispatch("door")
    r = s.feed(impl.FalseAlarm(occurrence, note="a shadow, not the door"), at=20.0)
    assert effects(r, impl.StoreCounterExample) == [
        impl.StoreCounterExample("door", occurrence, "a shadow, not the door")]
    assert types(r) == (["cleared"] if emit_cleared else [])
    if emit_cleared:
        assert r.decisions[0].reason == "false_alarm" and r.decisions[0].occurrence_id == occurrence
    assert s.phase("door") == "idle"
    assert s.reply(in_flight, "yes", at=21.0).decisions == ()          # stale


def test_false_alarm_on_an_event_that_already_rearmed_still_stores_the_example(impl):
    s = Stream(impl, fall(verify="off", rearm_s=60.0))
    occurrence = s.observe("fall", "yes", at=10.0).decisions[0].occurrence_id
    s.fire("fall", "rearm", at=70.0)
    r = s.feed(impl.FalseAlarm(occurrence), at=100.0)
    assert len(effects(r, impl.StoreCounterExample)) == 1 and r.decisions == ()


def test_scene_return_clears_a_held_state(impl):
    s = Stream(impl, door(emit_cleared=True))
    s.observe("door", "yes", at=10.0)
    r = s.feed(impl.SceneReturn("door"), at=20.0)
    assert types(r) == ["cleared"] and r.decisions[0].reason == "scene_return"
    assert s.phase("door") == "idle"


# --------------------------------------------------------------------------- events rearm


def test_event_rearm_absorbs_a_repeat_then_a_new_occurrence_fires_after_rearm_s(impl):
    s = Stream(impl, fall(verify="off", rearm_s=120.0))
    first = s.observe("fall", "yes", at=10.0).decisions[0]
    repeat = s.observe("fall", "yes", at=60.0)
    assert repeat.decisions == () and "repeat_absorbed" in trace_events(repeat)
    s.fire("fall", "rearm", at=130.0)
    assert s.phase("fall") == "idle"
    second = s.observe("fall", "yes", at=140.0).decisions[0]
    assert second.type == "met" and second.occurrence_id != first.occurrence_id


def test_rearm_is_applied_even_when_its_timer_is_late(impl):
    s = Stream(impl, fall(verify="off", rearm_s=120.0))
    s.observe("fall", "yes", at=10.0)
    assert types(s.observe("fall", "yes", at=135.0)) == ["met"]


def test_another_subject_is_not_absorbed_by_a_reported_event(impl):
    s = Stream(impl, fall(verify="off"))
    s.observe("fall", "yes", at=10.0, subject="person-1")
    assert types(s.observe("fall", "yes", at=11.0, subject="person-2")) == ["met"]


# --------------------------------------------------------------------------- emission control


def test_cooldown_suppresses_a_second_met_and_records_a_trace(impl):
    s = Stream(impl, door(clear_after_no=1, cooldown_s=300.0))
    assert types(s.observe("door", "yes", at=10.0)) == ["met"]
    s.observe("door", "no", at=20.0)
    r = s.observe("door", "yes", at=30.0)
    assert r.decisions == () and s.phase("door") == "holding"
    assert any(e.event == "suppressed" and "cooldown" in e.detail for e in effects(r, impl.Trace))
    s.observe("door", "no", at=40.0)
    assert types(s.observe("door", "yes", at=320.0)) == ["met"]


def test_snooze_suppresses_new_occurrences_until_it_ends(impl):
    s = Stream(impl, door(clear_after_no=1))
    s.feed(impl.Snooze("door", until=100.0), at=5.0)
    assert s.observe("door", "yes", at=10.0).decisions == ()
    s.observe("door", "no", at=20.0)
    assert types(s.observe("door", "yes", at=110.0)) == ["met"]


# --------------------------------------------------------------------------- capacity


def test_capacity_never_evicts_an_active_occurrence_and_refuses_a_new_subject(impl):
    s = Stream(impl, door(clear_after_no=1), max_keys=2)
    s.observe("door", "yes", at=10.0, subject="p1")
    s.observe("door", "yes", at=11.0, subject="p2")
    r = s.observe("door", "yes", at=12.0, subject="p3")
    assert r.decisions == ()
    [health] = effects(r, impl.EmitHealth)
    assert health.health.kind == "capacity" and health.health.subject == "p3"
    assert (s.phase("door", "p1"), s.phase("door", "p2"), s.phase("door", "p3")) == ("holding", "holding", "idle")

    s.observe("door", "no", at=20.0, subject="p1")    # p1 is idle now, so it can be evicted
    r = s.observe("door", "yes", at=21.0, subject="p3")
    assert types(r) == ["met"] and "evicted" in trace_events(r)
    assert s.phase("door", "p2") == "holding"


# --------------------------------------------------------------------------- identity


def _story(impl) -> list:
    s = Stream(impl, door(clear_after_no=1, emit_cleared=True, max_open_s=50.0), fall(verify="off"))
    s.observe("door", "yes", at=10.0)
    s.fire("door", "max_open", at=60.0)
    s.observe("door", "no", at=70.0)
    s.observe("fall", "yes", at=71.0)
    s.observe("door", "yes", at=80.0)
    s.observe("door", "unknown", at=81.0, subject="p9")
    return [d for r in s.results for d in r.decisions]


def test_decision_ids_are_stable_and_unique(impl):
    first, again = _story(impl), _story(impl)
    assert [d.decision_id for d in first] == [d.decision_id for d in again]
    ids = [d.decision_id for d in first]
    assert len(ids) == len(set(ids)) == 6
    assert all(d.decision_id.startswith(d.occurrence_id + "/") for d in first)


def test_occurrence_ids_are_never_reused_after_a_snapshot_round_trip(impl):
    s = Stream(impl, door(clear_after_no=1))
    seen = []
    for t in (10.0, 30.0):
        seen.append(s.observe("door", "yes", at=t).decisions[0].occurrence_id)
        s.observe("door", "no", at=t + 5)
    s.restore(s.snapshot())
    s.feed(impl.EpochStarted(1, restart=True), at=100.0)
    for t in (110.0, 130.0):
        seen.append(s.observe("door", "yes", at=t).decisions[0].occurrence_id)
        s.observe("door", "no", at=t + 5)
    assert len(set(seen)) == 4

    lost = Stream(impl, door(), created_at=99.0)      # a lost snapshot starts a new incarnation
    assert lost.observe("door", "yes", at=10.0).decisions[0].occurrence_id not in seen


# --------------------------------------------------------------------------- failures, in every phase


def _drive_to(impl, phase: str) -> tuple[Stream, Condition]:
    kind, name = phase.split(":")
    setups = {
        "state:idle": (door(), 0),
        "state:confirming": (door(confirm=2), 1),
        "state:verifying": (door(verify="fresh"), 1),
        "state:holding": (door(), 1),
        "event:idle": (fall(), 0),
        "event:candidate": (fall(confirm=2), 1),
        "event:verifying": (fall(), 1),
        "event:reported": (fall(verify="off"), 1),
    }
    condition, yeses = setups[phase]
    s = Stream(impl, condition)
    for _ in range(yeses):
        s.observe(condition.id, "yes", at=10.0)
    assert s.phase(condition.id) == name
    return s, condition


ALL_PHASES = ["state:idle", "state:confirming", "state:verifying", "state:holding",
              "event:idle", "event:candidate", "event:verifying", "event:reported"]


@pytest.mark.parametrize("phase", ALL_PHASES)
@pytest.mark.parametrize("failure", ["timeout", "unavailable", "malformed", "missing", "refused", "budget"])
def test_a_failed_observation_never_changes_state(impl, phase, failure):
    s, condition = _drive_to(impl, phase)
    purpose = "verify" if phase.endswith("verifying") else "detect"
    before = impl.decision_view(s.state)
    r = s.reply(s.dispatch(condition.id, purpose=purpose), "unknown", at=11.0, failure=failure)
    assert r.decisions == ()
    assert fates(r) == ["failed"]
    assert impl.decision_view(s.state) == before
    assert [e for e in r.effects if not isinstance(e, impl.Trace)] == []


@pytest.mark.parametrize("phase", ALL_PHASES)
def test_every_phase_has_a_defined_outcome_for_every_answer(impl, phase):
    """No undefined cells: every answer in every phase steps without error and leaves a fate."""
    for answer, confidence in (("yes", 0.9), ("no", 0.9), ("yes", 0.3), ("no", 0.3), ("unknown", 0.9)):
        for purpose in ("detect", "verify"):
            s, condition = _drive_to(impl, phase)
            r = s.reply(s.dispatch(condition.id, purpose=purpose), answer, confidence, at=11.0)
            assert len(fates(r)) == 1


# --------------------------------------------------------------------------- properties
#
# One generator of random runs, driven two ways: by Hypothesis (which shrinks a failure to a
# minimal story) and by a seeded uniform random walk (which reaches deep paths, such as
# verify -> report -> rearm, far more often than Hypothesis's biased draws do).


SUBJECTS = ("scene", "p1")
GAPS = (0.0, 0.5, 3.0, 12.0, 25.0, 45.0, 130.0)
VERIFY_GAPS = (0.0, 0.5, 3.0)
# Mostly judge answers, so occurrences live long enough to be verified, held and cleared.
INPUT_MIX = (["detect"] * 14 + ["verify"] * 6 + ["timer"] * 3
             + ["resolve", "false_alarm", "scene_return", "epoch", "update", "snooze"])
LATER = 1e7   # after every deadline any generated run can set


def _random_condition(choose) -> Condition:
    kind = choose(["state", "event"])
    return Condition(
        "c", "something", kind=kind,
        verify=choose(["auto", "off", "fresh" if kind == "state" else "original"]),
        confirm=choose([1, 2, 3]), confirm_window_s=20.0,
        fast_path_confidence=choose([None, 0.95]),
        candidate_deadline_s=30.0, on_unverified=choose(["uncertain", "health"]),
        rearm_s=60.0, recheck_every_s=40.0, clear_after_no=choose([1, 2, 3]),
        max_open_s=choose([None, 90.0]), cooldown_s=choose([0.0, 100.0]),
        emit_cleared=choose([True, False]),
    )


def _random_run(impl, choose, *, steps=None):
    """Drive a stream with random inputs. Returns the stream and [(input, now, conditions)]."""
    s = Stream(impl, _random_condition(choose))
    script = []
    now = 1.0
    seen_ids: list[str] = []
    for _ in range(steps if steps is not None else choose(range(1, 41))):
        what = choose(INPUT_MIX)
        # verification replies come back quickly; everything else may arrive at any gap
        now += choose(VERIFY_GAPS if what == "verify" else GAPS)
        cond = s.conditions["c"]
        subject = choose(SUBJECTS)
        if what in ("detect", "verify"):
            verifying = [x for x in SUBJECTS if s.phase("c", x) == "verifying"]
            if what == "verify" and verifying:
                subject = choose(verifying)
            tags = s.dispatch("c", subject, purpose=what)
            staleness = choose(["fresh"] * 6 + ["gen", "epoch"])
            if staleness == "gen":
                tags["occurrence_gen"] = max(0, tags["occurrence_gen"] - 1)
            elif staleness == "epoch" and s.epoch > 0:
                tags["stream_epoch"] = s.epoch - 1
            answer = choose(["yes", "yes", "no", "no", "unknown"])
            confidence = choose([0.3, 0.7, 0.97])
            failure = choose([None, None, None, "timeout"])
            if failure:
                answer, confidence = "unknown", 0.0
            o = Observation(answer=answer, confidence=confidence, reason="r", stream_id=s.stream_id,
                            evidence_t=(now - 1.0, now), judge="fake", failure=failure, **tags)
            inp = impl.Observed(o, Usage(attempts=1))
        elif what == "timer":
            inp = impl.TimerFired("c", subject, choose(impl.TIMER_KINDS))
        elif what in ("resolve", "false_alarm"):
            if not seen_ids:
                continue
            occurrence = choose(seen_ids)
            inp = impl.Resolve(occurrence) if what == "resolve" else impl.FalseAlarm(occurrence)
        elif what == "scene_return":
            inp = impl.SceneReturn("c", choose([None, subject]))
        elif what == "epoch":
            inp = impl.EpochStarted(s.epoch + 1, restart=choose([True, False]))
        elif what == "update":
            s.conditions = {"c": replace(cond, version=cond.version + 1)}
            inp = impl.ConditionChanged("c")
        else:
            inp = impl.Snooze("c", until=now + choose([5.0, 50.0]))
        script.append((inp, now, s.conditions))
        result = s.feed(inp, at=now)
        seen_ids += [d.occurrence_id for d in result.decisions]
    return s, script


def _replay(impl, script, *, round_trip_at=None):
    state = impl.new_state("cam-1", created_at=0.0)
    outputs = []
    for i, (inp, now, conds) in enumerate(script):
        if i == round_trip_at:
            state = impl.from_dict(json.loads(json.dumps(impl.to_dict(state))))
        r = impl.step(state, inp, now, conds)
        state = r.state
        outputs.append((r.decisions, r.effects))
    return state, outputs


# The invariants, each checked on one random run.


def check_no_occurrence_is_met_twice(impl, s, script):
    mets = [d.occurrence_id for r in s.results for d in r.decisions if d.type == "met"]
    assert len(mets) == len(set(mets))
    ids = [d.decision_id for r in s.results for d in r.decisions]
    assert len(ids) == len(set(ids))


def check_every_observation_has_exactly_one_fate(impl, s, script):
    for (inp, _, _), result in zip(script, s.results[len(s.results) - len(script):]):
        assert len(fates(result)) == (1 if isinstance(inp, impl.Observed) else 0)


def check_the_input_state_is_never_mutated(impl, s, script):
    state = impl.new_state("cam-1", created_at=0.0)
    for inp, now, conds in script:
        before = impl.to_dict(state)
        after = impl.step(state, inp, now, conds).state
        assert impl.to_dict(state) == before
        state = after


def check_a_snapshot_round_trip_changes_nothing(impl, s, script, where):
    final, outputs = _replay(impl, script)
    final_rt, outputs_rt = _replay(impl, script, round_trip_at=where)
    assert outputs_rt == outputs
    assert impl.to_dict(final_rt) == impl.to_dict(final)


def check_failures_and_weak_answers_never_change_state(impl, s, script):
    # Let every deadline pass first, so the only thing that could change state is the answer.
    for subject in SUBJECTS:
        for kind in impl.TIMER_KINDS:
            s.fire("c", kind, at=LATER, subject=subject)
    for subject in SUBJECTS:
        for answer, confidence, failure in (("unknown", 0.0, "timeout"), ("unknown", 0.0, "refused"),
                                            ("yes", 0.1, None), ("no", 0.1, None)):
            for purpose in ("detect", "verify"):
                before = impl.decision_view(s.state)
                r = s.reply(s.dispatch("c", subject, purpose=purpose), answer, confidence, at=LATER,
                            failure=failure)
                assert r.decisions == ()
                assert impl.decision_view(s.state) == before


PROPERTY_SETTINGS = settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])


def _hypothesis_choose(data):
    return lambda options: data.draw(st.sampled_from(list(options)))


@PROPERTY_SETTINGS
@given(data=st.data())
def test_property_no_occurrence_is_met_twice(impl, data):
    check_no_occurrence_is_met_twice(impl, *_random_run(impl, _hypothesis_choose(data)))


@PROPERTY_SETTINGS
@given(data=st.data())
def test_property_failures_and_weak_answers_never_change_state(impl, data):
    check_failures_and_weak_answers_never_change_state(impl, *_random_run(impl, _hypothesis_choose(data)))


@PROPERTY_SETTINGS
@given(data=st.data())
def test_property_every_observation_has_exactly_one_fate(impl, data):
    check_every_observation_has_exactly_one_fate(impl, *_random_run(impl, _hypothesis_choose(data)))


@PROPERTY_SETTINGS
@given(data=st.data(), where=st.floats(0, 1))
def test_property_a_snapshot_round_trip_at_any_point_changes_nothing(impl, data, where):
    s, script = _random_run(impl, _hypothesis_choose(data))
    check_a_snapshot_round_trip_changes_nothing(impl, s, script, int(where * len(script)))


@PROPERTY_SETTINGS
@given(data=st.data())
def test_property_the_input_state_is_never_mutated(impl, data):
    check_the_input_state_is_never_mutated(impl, *_random_run(impl, _hypothesis_choose(data)))


@pytest.mark.parametrize("seed", range(40))
def test_random_walks_keep_every_invariant(impl, seed):
    rng = random.Random(seed)
    for _ in range(10):
        s, script = _random_run(impl, rng.choice, steps=60)
        check_no_occurrence_is_met_twice(impl, s, script)
        check_every_observation_has_exactly_one_fate(impl, s, script)
        check_the_input_state_is_never_mutated(impl, s, script)
        check_a_snapshot_round_trip_changes_nothing(impl, s, script, rng.randrange(len(script) + 1))
        check_failures_and_weak_answers_never_change_state(impl, s, script)
