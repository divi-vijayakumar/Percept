"""Named scenarios for the decision reducer, then properties over random input sequences.

Each scenario is a short story with explicit wall-clock times. Small numbers keep the
stories readable; the reducer only ever compares and adds them.

Every test runs against each implementation in ``IMPLEMENTATIONS``: the reference model and
the real reducer, ``percept.decide``, which must pass unchanged.
On top of each scenario's own assertions, every stream a scenario drives is replayed through
the reference model at teardown and must produce exactly the same decisions and effects
(``check_matches_reference``).
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field, fields, replace

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import reference_model
from percept import decide
from percept.contracts import Condition, Observation, Usage

IMPLEMENTATIONS = [reference_model, decide]

pytestmark = pytest.mark.parametrize("impl", IMPLEMENTATIONS, ids=lambda m: m.__name__)

# One observation's share of a request's usage, in every scenario unless a test says otherwise.
SHARE = Usage(attempts=1, tokens_in=100, tokens_out=10, cost_usd=0.25, latency_s=0.5)


# --------------------------------------------------------------------------- scripts and the differential check


@dataclass
class Script:
    """An implementation-neutral record of one stream: how it was created and every input."""

    stream_id: str
    created_at: float
    max_keys: int
    steps: list = field(default_factory=list)   # (input class name, fields, now, conditions)


def _neutral(inp) -> tuple[str, dict]:
    return type(inp).__name__, {f.name: getattr(inp, f.name) for f in fields(inp)}


def _rebuild(impl, name: str, kwargs: dict):
    return getattr(impl, name)(**kwargs)


def _outputs(result) -> tuple:
    return result.decisions, [(type(e).__name__, asdict(e)) for e in result.effects]


def run_script(impl, script: Script, *, round_trip_at=None):
    """Replay a script through ``impl``. Returns (final state, outputs per step)."""
    state = impl.new_state(script.stream_id, created_at=script.created_at, max_keys=script.max_keys)
    outputs = []
    for i, (name, kwargs, now, conditions) in enumerate(script.steps):
        if i == round_trip_at:
            state = impl.from_dict(json.loads(json.dumps(impl.to_dict(state))))
        result = impl.step(state, _rebuild(impl, name, kwargs), now, conditions)
        state = result.state
        outputs.append(_outputs(result))
    return state, outputs


def check_matches_reference(impl, script: Script, live_outputs=None) -> None:
    """The implementation must produce exactly the reference model's decisions and effects,
    and end in the same snapshot. With ``live_outputs``, the replay must also reproduce what
    the live run produced."""
    final_expected, expected = run_script(reference_model, script)
    final_actual, actual = run_script(impl, script)
    assert actual == expected
    assert impl.to_dict(final_actual) == reference_model.to_dict(final_expected)
    if live_outputs is not None:
        assert live_outputs == expected


_STREAMS: list = []


@pytest.fixture(autouse=True)
def _differential_check():
    _STREAMS.clear()
    yield
    streams, _STREAMS[:] = list(_STREAMS), []
    for s in streams:
        check_matches_reference(s.impl, s.script, [_outputs(r) for r in s.results])


# --------------------------------------------------------------------------- harness


class Stream:
    """One stream's reducer state, driven the way the runtime will drive it."""

    def __init__(self, impl, *conditions: Condition, stream_id: str = "cam-1", max_keys: int = 64,
                 created_at: float = 0.0, register: bool = True):
        self.impl = impl
        self.stream_id = stream_id
        self.conditions = {c.id: c for c in conditions}
        self.retired: dict[str, Condition] = {}     # removed conditions, for tagging late requests
        self.state = impl.new_state(stream_id, created_at=created_at, max_keys=max_keys)
        self.script = Script(stream_id, created_at, max_keys)
        self.epoch = 0
        self.results = []
        self._requests = 0
        self._verifies = 0
        if register:
            _STREAMS.append(self)
        for c in conditions:
            self.feed(impl.ConditionChanged(c.id), at=0.0)

    # driving

    def feed(self, inp, *, at: float):
        result = self.impl.step(self.state, inp, at, self.conditions)
        self.script.steps.append((*_neutral(inp), at, self.conditions))
        self.state = result.state
        self.results.append(result)
        if isinstance(inp, self.impl.EpochStarted) and inp.stream_epoch > self.epoch:
            self.epoch = inp.stream_epoch
        return result

    def dispatch(self, condition_id: str, subject: str = "scene", purpose: str = "detect") -> dict:
        """Tag a judge request the way the runtime does at dispatch time."""
        self._requests += 1
        return dict(condition_id=condition_id, subject=subject, request_id=f"r-{self._requests}",
                    stream_epoch=self.epoch, condition_version=self._condition(condition_id).version,
                    occurrence_gen=self.impl.gen_for(self.state, condition_id, subject), purpose=purpose)

    def reply(self, request: dict, answer: str, confidence: float = 0.9, *, at: float,
              evidence: tuple[float, float] | None = None, failure: str | None = None, usage: Usage = SHARE):
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
        """Answer a RequestVerify effect. The observation carries the request's own tags."""
        self._verifies += 1
        tags = dict(condition_id=request.condition_id, subject=request.subject, request_id=f"v-{self._verifies}",
                    stream_epoch=request.stream_epoch, condition_version=request.condition_version,
                    occurrence_gen=request.occurrence_gen, purpose="verify")
        evidence = request.interval or (at - 0.5, at)
        return self.reply(tags, answer, confidence, at=at, evidence=evidence, failure=failure)

    def _condition(self, condition_id: str) -> Condition:
        return self.conditions.get(condition_id) or self.retired[condition_id]

    def fire(self, condition_id: str, kind: str, *, at: float, subject: str = "scene"):
        return self.feed(self.impl.TimerFired(condition_id, subject, kind), at=at)

    def update(self, condition: Condition, *, at: float):
        self.conditions = {**self.conditions, condition.id: condition}
        return self.feed(self.impl.ConditionChanged(condition.id), at=at)

    def remove(self, condition_id: str, *, at: float | None = None):
        """Remove a condition. With ``at``, announce it; without, the reducer finds out by
        version sync on the next input that names it (or at the next epoch start)."""
        self.retired[condition_id] = self.conditions[condition_id]
        self.conditions = {k: c for k, c in self.conditions.items() if k != condition_id}
        return None if at is None else self.feed(self.impl.ConditionChanged(condition_id), at=at)

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


def stale_details(result) -> list[str]:
    return [e.detail for e in _traces(result) if e.fate == "stale"]


def opened(result) -> str:
    [occurrence] = [e.occurrence_id for e in _traces(result) if e.event == "opened"]
    return occurrence


def door(**kw) -> Condition:
    return Condition("door", "The door is open", kind="state", **kw)


def fall(**kw) -> Condition:
    return Condition("fall", "A person falls to the floor", kind="event", **kw)


# --------------------------------------------------------------------------- the first yes


def test_state_first_confident_yes_with_confirm_1_is_met_in_the_same_step(impl):
    s = Stream(impl, door())
    r = s.observe("door", "yes", 0.8, at=10.0)
    assert types(r) == ["met"]
    assert r.decisions[0].cause is None and r.decisions[0].reason == "model says yes"
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


# --------------------------------------------------------------------------- confirm windows (evidence time)


def test_confirm_2_within_the_window_is_met(impl):
    s = Stream(impl, door(confirm=2, confirm_window_s=20.0))
    assert types(s.observe("door", "yes", at=11.0, evidence=(9.0, 10.0))) == []
    assert s.phase("door") == "confirming"
    assert types(s.observe("door", "yes", at=26.0, evidence=(24.0, 25.0))) == ["met"]


def test_the_window_is_measured_on_evidence_time_not_arrival(impl):
    s = Stream(impl, door(confirm=2, confirm_window_s=20.0))
    s.observe("door", "yes", at=11.0, evidence=(9.0, 10.0))
    # Judged on frames from inside the window, delivered well after it.
    assert types(s.observe("door", "yes", at=40.0, evidence=(24.0, 25.0))) == ["met"]


def test_confirm_2_second_yes_after_the_window_restarts_the_window(impl):
    s = Stream(impl, door(confirm=2, confirm_window_s=20.0))
    s.observe("door", "yes", at=11.0, evidence=(9.0, 10.0))
    late = s.observe("door", "yes", at=32.0, evidence=(30.0, 31.0))    # the window ended at 30
    assert types(late) == [] and "window_restarted" in trace_events(late)
    assert s.phase("door") == "confirming"
    assert types(s.observe("door", "yes", at=46.0, evidence=(44.0, 45.0))) == ["met"]


def test_an_unfinished_window_gives_up_once_no_yes_inside_it_can_arrive(impl):
    s = Stream(impl, door(confirm=2, confirm_window_s=20.0, candidate_deadline_s=30.0))
    r = s.observe("door", "yes", at=11.0, evidence=(9.0, 10.0))
    assert impl.ScheduleTimer("door", "scene", "window", 60.0) in effects(r, impl.ScheduleTimer)
    s.fire("door", "window", at=60.0)
    assert s.phase("door") == "idle"


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


def test_a_met_carries_the_cost_and_request_ids_of_everything_that_led_to_it(impl):
    s = Stream(impl, fall())
    [verify] = effects(s.observe("fall", "yes", at=10.0), impl.RequestVerify)
    s.observe("fall", "no", at=10.5)                  # absorbed, but it was paid for
    [met] = s.verify_reply(verify, "yes", at=11.0).decisions
    assert set(met.request_ids) == {"r-1", "r-2", "v-1"}
    assert met.cost == SHARE + SHARE + SHARE


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
    assert effects(r, impl.CancelVerify) == []         # the verification answered; nothing to cancel
    assert s.phase("fall") == "idle"


def test_two_verification_timeouts_record_no_negative_then_the_deadline_is_uncertain(impl):
    s = Stream(impl, fall(candidate_deadline_s=30.0))
    first = s.observe("fall", "yes", 0.8, at=10.0)
    [verify] = effects(first, impl.RequestVerify)
    for t in (18.0, 26.0):
        r = s.verify_reply(verify, "unknown", at=t, failure="timeout")
        assert types(r) == [] and fates(r) == ["failed"]
        assert s.phase("fall") == "verifying"
    r = s.fire("fall", "verify", at=40.0)
    [uncertain] = r.decisions
    assert uncertain.type == "uncertain" and uncertain.cause == "timeout"
    assert uncertain.occurrence_id == opened(first) and uncertain.confidence == 0.8
    assert uncertain.request_ids == ("r-1", "v-1", "v-2")   # failed attempts were paid for too
    assert len(effects(r, impl.CancelVerify)) == 1
    assert s.phase("fall") == "idle"


def test_unverified_with_on_unverified_health_emits_only_an_unverified_health_effect(impl):
    s = Stream(impl, fall(candidate_deadline_s=30.0, on_unverified="health"))
    s.observe("fall", "yes", at=10.0)
    r = s.fire("fall", "verify", at=40.0)
    assert r.decisions == ()
    [health] = effects(r, impl.EmitHealth)
    assert health.health.kind == "unverified" and health.health.condition_id == "fall"


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
    assert types(r) == ["uncertain"] and r.decisions[0].cause == "unverified"
    assert s.phase("fall") == "idle"


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


@pytest.mark.parametrize("how", ["resolve", "false_alarm", "condition_update"])
def test_closing_a_verifying_occurrence_cancels_its_verification(impl, how):
    s = Stream(impl, fall())
    first = s.observe("fall", "yes", at=10.0)
    [verify] = effects(first, impl.RequestVerify)
    if how == "resolve":
        r = s.feed(impl.Resolve(opened(first)), at=11.0)
    elif how == "false_alarm":
        r = s.feed(impl.FalseAlarm(opened(first)), at=11.0)
    else:
        r = s.update(replace(s.conditions["fall"], version=2), at=11.0)
    assert effects(r, impl.CancelVerify) == [
        impl.CancelVerify("fall", "scene", verify.occurrence_id, verify.occurrence_gen)]
    late = s.verify_reply(verify, "yes", at=12.0)
    assert late.decisions == () and fates(late) == ["stale"]


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


def test_model_unknown_in_idle_changes_nothing(impl):
    for condition in (door(), fall()):
        s = Stream(impl, condition)
        before = impl.decision_view(s.state)
        r = s.observe(condition.id, "unknown", 0.95, at=10.0)
        assert r.decisions == () and fates(r) == ["judged_weak"]
        assert impl.decision_view(s.state) == before


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
    assert types(r) == ["cleared"] and r.decisions[0].cause == "condition_changed"
    assert s.phase("door") == "idle"
    late = s.reply(in_flight, "yes", at=21.0)
    assert late.decisions == () and stale_details(late) == ["version"]
    assert types(s.observe("door", "yes", at=22.0)) == ["met"]     # judged against version 2


def test_a_verify_answer_when_nothing_is_being_verified_is_stale(impl):
    s = Stream(impl, door())
    r = s.reply(s.dispatch("door", purpose="verify"), "yes", at=10.0)
    assert r.decisions == () and stale_details(r) == ["purpose"] and s.phase("door") == "idle"


# --------------------------------------------------------------------------- the evidence floor


def test_a_late_recheck_yes_after_a_clear_does_not_start_a_new_occurrence(impl):
    s = Stream(impl, door(clear_after_no=2))
    s.observe("door", "yes", at=10.0, evidence=(9.0, 9.5))
    recheck = s.dispatch("door")                      # leaves at 40 on frames up to 40
    s.observe("door", "no", at=41.5, evidence=(40.5, 41.0))
    s.observe("door", "no", at=42.5, evidence=(41.5, 42.0))      # cleared: over as of 42
    assert s.phase("door") == "idle"
    late = s.reply(recheck, "yes", at=50.0, evidence=(39.5, 40.0))
    assert late.decisions == () and stale_details(late) == ["evidence"]
    assert s.phase("door") == "idle"
    assert types(s.observe("door", "yes", at=56.0, evidence=(55.0, 55.5))) == ["met"]


def test_a_late_detect_yes_after_rearm_ends_does_not_start_a_new_occurrence(impl):
    s = Stream(impl, fall(verify="off", rearm_s=120.0))
    s.observe("fall", "yes", at=10.0)                 # rearm until 130
    in_flight = s.dispatch("fall")
    late = s.reply(in_flight, "yes", at=131.0, evidence=(124.0, 125.0))
    assert late.decisions == () and stale_details(late) == ["evidence"]
    assert s.phase("fall") == "idle"
    assert types(s.observe("fall", "yes", at=136.0, evidence=(134.0, 135.0))) == ["met"]


def test_a_late_detect_yes_after_verify_rejected_does_not_start_a_new_candidate(impl):
    s = Stream(impl, fall())
    [verify] = effects(s.observe("fall", "yes", at=10.0, evidence=(8.0, 10.0)), impl.RequestVerify)
    in_flight = s.dispatch("fall")                    # judged on frames up to 9.8
    s.verify_reply(verify, "no", at=11.0)
    late = s.reply(in_flight, "yes", at=12.0, evidence=(9.0, 9.8))
    assert stale_details(late) == ["evidence"] and s.phase("fall") == "idle"
    assert effects(late, impl.RequestVerify) == []
    assert s.phase("fall") == "idle"
    s.observe("fall", "yes", at=14.0, evidence=(12.0, 13.0))
    assert s.phase("fall") == "verifying"             # newer evidence opens a new candidate


def test_a_window_end_or_unverified_close_sets_no_floor(impl):
    s = Stream(impl, fall(candidate_deadline_s=30.0))
    s.observe("fall", "yes", at=10.0, evidence=(8.0, 9.0))
    s.fire("fall", "verify", at=40.0)                 # unverified
    s.observe("fall", "yes", at=41.0, evidence=(8.5, 9.5))   # older evidence is still accepted
    assert s.phase("fall") == "verifying"


# --------------------------------------------------------------------------- an entry absorbed while a candidate verifies


def _absorbing_candidate(s, *, absorbed=(11.0, 11.5)):
    """A candidate verifying on evidence (8, 9) that absorbs a later detect yes (request r-2) about newer evidence."""
    [verify] = effects(s.observe("fall", "yes", at=10.0, evidence=(8.0, 9.0)), s.impl.RequestVerify)
    r = s.observe("fall", "yes", at=12.0, evidence=absorbed)
    assert r.decisions == () and "awaiting_verification" in trace_events(r) and fates(r) == ["judged"]
    assert effects(r, s.impl.RequestVerify) == []        # nothing is decided or requested until the candidate closes
    return verify


def _successor(result, impl, interval=(11.0, 11.5)):
    """The one verification of the successor promoted from the absorbed detect, on exactly its evidence."""
    [verify] = effects(result, impl.RequestVerify)
    assert verify.absorbed and verify.mode == "original" and verify.interval == interval and verify.stream_epoch == 0
    assert "promoted" in trace_events(result) and "opened" not in trace_events(result)
    assert fates(result) in ([], ["judged"], ["judged_weak"])      # the absorbed yes is not counted twice
    return verify


def test_an_absorbed_entry_is_promoted_to_a_successor_when_the_candidate_is_rejected(impl):
    s = Stream(impl, fall(rearm_s=30.0))
    verify = _absorbing_candidate(s)
    r = s.verify_reply(verify, "no", at=13.0)
    assert r.decisions == () and s.phase("fall") == "verifying" and "verify_rejected" in trace_events(r)
    assert effects(r, impl.CancelVerify) == []                        # the answered verification needs no cancelling
    successor = _successor(r, impl)
    assert successor.occurrence_id != verify.occurrence_id and successor.deadline == 13.0 + 30.0
    r = s.verify_reply(successor, "yes", at=14.0)
    [met] = r.decisions
    assert met.type == "met" and met.cause is None and met.occurrence_id == successor.occurrence_id
    assert met.t_observed == 11.5 and met.decision_id.endswith("/1")
    assert met.cost == SHARE + SHARE and met.request_ids == ("r-2", "v-2")    # the absorbed detect, charged once


def test_an_absorbed_entry_is_promoted_to_a_successor_when_the_candidate_is_uncertain(impl):
    s = Stream(impl, fall(rearm_s=30.0))
    verify = _absorbing_candidate(s)
    r = s.verify_reply(verify, "unknown", 0.0, at=13.0)
    [uncertain] = r.decisions
    assert uncertain.type == "uncertain" and uncertain.cause == "unverified" and uncertain.t_observed == 9.0
    assert uncertain.cost == SHARE + SHARE and "r-2" not in uncertain.request_ids    # detect 1 and verify 1 only
    successor = _successor(r, impl)
    [met] = s.verify_reply(successor, "yes", at=14.0).decisions
    assert met.type == "met" and met.occurrence_id != uncertain.occurrence_id and met.t_observed == 11.5
    assert uncertain.cost + met.cost == SHARE + SHARE + SHARE + SHARE          # four requests, each charged to exactly one decision


def test_an_absorbed_entry_is_promoted_to_a_successor_when_verification_times_out(impl):
    s = Stream(impl, fall(candidate_deadline_s=30.0))
    verify = _absorbing_candidate(s)
    r = s.fire("fall", "verify", at=40.0)
    assert types(r) == ["uncertain"] and r.decisions[0].cause == "timeout"
    assert effects(r, impl.CancelVerify) == [impl.CancelVerify("fall", "scene", verify.occurrence_id, verify.occurrence_gen)]
    successor = _successor(r, impl)
    assert successor.deadline == 70.0                                  # a full deadline of its own
    assert impl.ScheduleTimer("fall", "scene", "verify", 70.0) in effects(r, impl.ScheduleTimer)
    s.fire("fall", "verify", at=70.0)                                  # and it ends the same way: no loop
    assert s.phase("fall") == "idle"


def test_a_verified_candidate_discards_the_absorbed_entry_no_duplicate_alert(impl):
    s = Stream(impl, fall(rearm_s=30.0))
    verify = _absorbing_candidate(s)
    r = s.verify_reply(verify, "yes", at=13.0)
    [met] = r.decisions
    assert met.type == "met" and effects(r, impl.RequestVerify) == [] and "promoted" not in trace_events(r)
    assert met.cost == SHARE + SHARE + SHARE and set(met.request_ids) == {"r-1", "r-2", "v-1"}      # charged once, here
    assert s.phase("fall") == "reported"
    later = s.observe("fall", "yes", at=14.0, evidence=(13.0, 13.5))
    assert later.decisions == () and "repeat_absorbed" in trace_events(later)
    rearmed = s.fire("fall", "rearm", at=43.0)
    assert rearmed.decisions == () and effects(rearmed, impl.RequestVerify) == [] and s.phase("fall") == "idle"


def test_nothing_is_promoted_when_nothing_was_absorbed(impl):
    s = Stream(impl, fall())
    [verify] = effects(s.observe("fall", "yes", at=10.0, evidence=(8.0, 9.0)), impl.RequestVerify)
    plain_no = s.observe("fall", "no", at=11.0, evidence=(10.0, 10.5))     # a no is not an absorbed entry
    assert "awaiting_verification" in trace_events(plain_no)
    r = s.verify_reply(verify, "no", at=13.0)
    assert effects(r, impl.RequestVerify) == [] and s.phase("fall") == "idle"


def test_nothing_is_promoted_for_a_yes_about_the_candidates_own_evidence(impl):
    s = Stream(impl, fall())
    [verify] = effects(s.observe("fall", "yes", at=10.0, evidence=(8.0, 9.0)), impl.RequestVerify)
    s.observe("fall", "yes", at=11.0, evidence=(8.5, 9.0))                 # nothing newer than the interval
    assert effects(s.verify_reply(verify, "no", at=13.0), impl.RequestVerify) == []


def test_at_most_one_successor_per_closed_candidate_and_the_first_absorbed_yes_is_the_one(impl):
    s = Stream(impl, fall())
    [verify] = effects(s.observe("fall", "yes", at=10.0, evidence=(8.0, 9.0)), impl.RequestVerify)
    s.observe("fall", "yes", at=11.0, evidence=(10.0, 10.5))               # r-2: kept
    s.observe("fall", "yes", at=12.0, evidence=(11.0, 11.5))               # r-3: charged to the candidate, not kept
    r = s.verify_reply(verify, "no", at=13.0)
    successor = _successor(r, impl, interval=(10.0, 10.5))
    assert len(effects(r, impl.RequestVerify)) == 1


def test_no_loop_a_successor_that_absorbed_nothing_closes_without_another_successor(impl):
    s = Stream(impl, fall())
    verify = _absorbing_candidate(s)
    successor = _successor(s.verify_reply(verify, "no", at=13.0), impl)
    r = s.verify_reply(successor, "no", at=14.0)                           # the same observation is not used twice
    assert effects(r, impl.RequestVerify) == [] and s.phase("fall") == "idle" and r.decisions == ()


def test_a_successor_that_absorbs_a_new_detect_promotes_that_one_not_the_old_one(impl):
    s = Stream(impl, fall())
    verify = _absorbing_candidate(s)
    successor = _successor(s.verify_reply(verify, "no", at=13.0), impl)
    s.observe("fall", "yes", at=15.0, evidence=(14.0, 14.5))               # new evidence, a new absorbed detect
    r = s.verify_reply(successor, "no", at=16.0)
    _successor(r, impl, interval=(14.0, 14.5))


def test_a_state_condition_promotes_nothing(impl):
    s = Stream(impl, door(verify="fresh", candidate_deadline_s=30.0))
    [verify] = effects(s.observe("door", "yes", at=10.0, evidence=(8.0, 9.0)), impl.RequestVerify)
    s.observe("door", "yes", at=12.0, evidence=(11.0, 11.5))
    r = s.verify_reply(verify, "no", at=13.0)
    assert effects(r, impl.RequestVerify) == [] and s.phase("door") == "idle"


def test_closing_a_verifying_candidate_by_a_control_promotes_nothing(impl):
    s = Stream(impl, fall())
    verify = _absorbing_candidate(s)
    r = s.feed(impl.FalseAlarm(verify.occurrence_id), at=13.0)
    assert effects(r, impl.RequestVerify) == [] and s.phase("fall") == "idle"


def test_a_restart_during_the_verification_promotes_nothing_and_charges_the_absorbed_detect_to_the_uncertain(impl):
    # The absorbed detection's frames died with the process: the candidate ends uncertain (cause "restart")
    # and the new epoch's triggers carry on. The absorbed detect is charged to that decision, once.
    s = Stream(impl, fall())
    _absorbing_candidate(s)
    r = s.feed(impl.EpochStarted(1, restart=True), at=20.0)
    [uncertain] = r.decisions
    assert (uncertain.type, uncertain.cause) == ("uncertain", "restart") and uncertain.cost == SHARE + SHARE
    assert "r-2" in uncertain.request_ids
    assert effects(r, impl.RequestVerify) == [] and s.phase("fall") == "idle"
    assert effects(s.observe("fall", "yes", at=21.0, evidence=(20.0, 20.5)), impl.RequestVerify)


def test_an_absorbed_detect_from_another_epoch_is_reported_as_a_gap_and_nothing_is_verified(impl):
    s = Stream(impl, fall())
    verify = _absorbing_candidate(s)
    s.feed(impl.EpochStarted(1, restart=False), at=14.0)                   # a reconnect: the candidate stays
    assert s.phase("fall") == "verifying"
    r = s.verify_reply(verify, "no", at=15.0)
    assert effects(r, impl.RequestVerify) == [] and r.decisions == () and s.phase("fall") == "idle"
    assert effects(r, impl.EvidenceGap) == [impl.EvidenceGap("fall", "scene", (11.0, 11.5), "epoch")]
    assert "promotion_refused" in trace_events(r)


def test_the_absorbed_detect_is_in_the_snapshot_only_while_a_candidate_holds_it_and_survives_a_round_trip(impl):
    s = Stream(impl, fall())
    [verify] = effects(s.observe("fall", "yes", at=10.0, evidence=(8.0, 9.0)), impl.RequestVerify)
    assert "absorbed" not in json.loads(s.snapshot())["keys"][0]            # a 1.0.0 reader still loads it
    s.observe("fall", "yes", at=12.0, evidence=(11.0, 11.5))
    kept = json.loads(s.snapshot())["keys"][0]
    assert kept["absorbed"]["request_id"] == "r-2" and kept["absorbed"]["evidence_t"] == [11.0, 11.5]
    assert kept["absorbed"]["stream_epoch"] == 0
    s.restore(s.snapshot())                                                 # a persisted verifying candidate
    r = s.verify_reply(verify, "no", at=13.0)
    _successor(r, impl)                                                     # still promoted after the round trip
    assert "absorbed" not in json.loads(s.snapshot())["keys"][0]


def test_a_successor_of_a_confirm_2_condition_is_a_candidate_until_a_second_yes_arrives(impl):
    s = Stream(impl, fall(confirm=2))
    s.observe("fall", "yes", at=10.0, evidence=(8.0, 9.0))
    [verify] = effects(s.observe("fall", "yes", at=10.5, evidence=(9.5, 10.0)), impl.RequestVerify)
    s.observe("fall", "yes", at=12.0, evidence=(11.0, 11.5))               # absorbed
    r = s.verify_reply(verify, "no", at=13.0)
    assert effects(r, impl.RequestVerify) == [] and s.phase("fall") == "candidate" and "promoted" in trace_events(r)
    [again] = effects(s.observe("fall", "yes", at=14.0, evidence=(13.0, 13.5)), impl.RequestVerify)
    assert again.interval == (11.0, 13.5)


# --------------------------------------------------------------------------- restart and reconnect


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
    assert old.decisions == () and stale_details(old) == ["epoch"]
    assert s.reply(s.dispatch("door"), "yes", at=502.0).decisions == ()   # still the same occurrence


def test_restart_resolves_an_event_being_verified_as_uncertain_with_the_same_occurrence(impl):
    s = Stream(impl, fall())
    first = s.observe("fall", "yes", at=10.0)
    s.restore(s.snapshot())
    r = s.feed(impl.EpochStarted(1, restart=True), at=12.0)
    [uncertain] = r.decisions
    assert uncertain.type == "uncertain" and uncertain.cause == "restart"
    assert uncertain.occurrence_id == opened(first) and uncertain.decision_id == opened(first) + "/1"
    assert s.phase("fall") == "idle"


def test_restart_resolves_an_event_candidate_supported_by_a_yes(impl):
    s = Stream(impl, fall(confirm=2))
    first = s.observe("fall", "yes", at=10.0)
    assert s.phase("fall") == "candidate"
    s.restore(s.snapshot())
    r = s.feed(impl.EpochStarted(1, restart=True), at=12.0)
    assert types(r) == ["uncertain"] and r.decisions[0].cause == "restart"
    assert r.decisions[0].occurrence_id == opened(first)
    assert s.phase("fall") == "idle"


def test_a_reconnect_keeps_verifying_the_original_interval(impl):
    s = Stream(impl, fall())
    [verify] = effects(s.observe("fall", "yes", at=10.0), impl.RequestVerify)
    s.feed(impl.EpochStarted(1, restart=False), at=11.0)
    assert s.phase("fall") == "verifying"
    assert types(s.verify_reply(verify, "yes", at=12.0)) == ["met"]   # its evidence is from epoch 0


def test_an_epoch_start_reschedules_every_live_timer(impl):
    s = Stream(impl, door(max_open_s=500.0), Condition("kettle", "The kettle boils", confirm=2),
               fall(), Condition("wave", "Someone waves", kind="event", verify="off"))
    s.observe("door", "yes", at=10.0)                 # holding: recheck, max_open
    s.observe("kettle", "yes", at=10.0)               # confirming: window
    s.observe("fall", "yes", at=10.0)                 # verifying: verify
    s.observe("wave", "yes", at=10.0)                 # reported: rearm
    r = s.feed(impl.EpochStarted(1, restart=False), at=11.0)
    scheduled = {(e.condition_id, e.kind) for e in effects(r, impl.ScheduleTimer)}
    assert scheduled == {("door", "recheck"), ("door", "max_open"), ("kettle", "window"),
                         ("fall", "verify"), ("wave", "rearm")}


# --------------------------------------------------------------------------- holding and clearing


def test_max_open_s_emits_expired_not_cleared_and_never_a_repeat_met(impl):
    s = Stream(impl, door(max_open_s=100.0, recheck_every_s=30.0, emit_cleared=True))
    occurrence = s.observe("door", "yes", at=10.0).decisions[0].occurrence_id
    recheck = s.fire("door", "recheck", at=40.0)
    assert len(effects(recheck, impl.RequestRecheck)) == 1
    assert s.observe("door", "yes", at=41.0).decisions == ()
    r = s.fire("door", "max_open", at=110.0)
    assert types(r) == ["expired"] and r.decisions[0].occurrence_id == occurrence
    assert r.decisions[0].cause == "max_open"
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
        cleared = r.decisions[0]
        assert cleared.cause == "no" and cleared.reason == "model says no"
        assert cleared.request_ids == ("r-2", "r-3", "r-4", "r-5", "r-6", "r-7")
        assert cleared.cost == SHARE + SHARE + SHARE + SHARE + SHARE + SHARE


def test_request_ids_kept_for_one_decision_are_capped(impl):
    s = Stream(impl, door(clear_after_no=1, emit_cleared=True))
    s.observe("door", "yes", at=1.0)
    for t in range(2, 32):
        s.observe("door", "yes", at=float(t))
    [cleared] = s.observe("door", "no", at=40.0).decisions
    assert len(cleared.request_ids) == impl.MAX_REQUEST_IDS
    assert cleared.request_ids[-1] == "r-32"


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
        assert r.decisions[0].cause == "false_alarm" and r.decisions[0].occurrence_id == occurrence
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
    assert types(r) == ["cleared"] and r.decisions[0].cause == "scene_return"
    assert s.phase("door") == "idle"


def test_resolve_clears_a_held_state_with_cause_resolve(impl):
    s = Stream(impl, door(emit_cleared=True))
    occurrence = s.observe("door", "yes", at=10.0).decisions[0].occurrence_id
    r = s.feed(impl.Resolve(occurrence), at=20.0)
    assert types(r) == ["cleared"] and r.decisions[0].cause == "resolve"


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


# --------------------------------------------------------------------------- cooldown, snooze, deferred met


def test_cooldown_suppresses_a_second_met_and_records_a_trace(impl):
    s = Stream(impl, door(clear_after_no=1, cooldown_s=300.0))
    assert types(s.observe("door", "yes", at=10.0)) == ["met"]
    s.observe("door", "no", at=20.0)
    r = s.observe("door", "yes", at=30.0)
    assert r.decisions == () and s.phase("door") == "holding"
    assert any(e.event == "suppressed" and "cooldown" in e.detail for e in _traces(r))


def test_cooldown_ends_exactly_at_cooldown_s(impl):
    s = Stream(impl, door(clear_after_no=1, cooldown_s=100.0))
    s.observe("door", "yes", at=10.0)
    s.observe("door", "no", at=20.0)
    assert types(s.observe("door", "yes", at=110.0)) == ["met"]      # 100 s elapsed: not cooling down


def test_a_met_suppressed_by_cooldown_is_delivered_at_the_first_yes_after_it_lifts(impl):
    s = Stream(impl, door(clear_after_no=1, cooldown_s=300.0))
    s.observe("door", "yes", at=10.0)
    s.observe("door", "no", at=20.0)
    second = s.observe("door", "yes", at=30.0, evidence=(29.0, 29.5))   # held, met suppressed
    assert s.observe("door", "yes", at=100.0).decisions == ()           # still cooling down
    [met] = s.observe("door", "yes", 0.95, at=320.0).decisions
    assert met.type == "met" and met.cause == "deferred"
    assert met.occurrence_id == opened(second) and met.t_observed == 29.5 and met.confidence == 0.95
    assert s.observe("door", "yes", at=400.0).decisions == ()           # delivered once


def test_a_met_suppressed_by_snooze_is_delivered_after_the_snooze_ends(impl):
    s = Stream(impl, door())
    s.feed(impl.Snooze("door", until=100.0), at=5.0)
    assert s.observe("door", "yes", at=10.0).decisions == ()
    assert s.observe("door", "yes", at=50.0).decisions == ()
    r = s.observe("door", "yes", at=110.0)
    assert types(r) == ["met"] and r.decisions[0].cause == "deferred"


def test_a_suppressed_met_is_never_delivered_for_an_occurrence_that_cleared(impl):
    s = Stream(impl, door(clear_after_no=1, emit_cleared=True))
    s.feed(impl.Snooze("door", until=100.0), at=5.0)
    s.observe("door", "yes", at=10.0)
    r = s.observe("door", "no", at=20.0)
    assert r.decisions == ()                           # the agent never heard of it
    [met] = s.observe("door", "yes", at=110.0).decisions
    assert met.cause is None and met.decision_id.endswith("/1")      # a new occurrence


def test_a_suppressed_met_is_never_delivered_after_expiry(impl):
    s = Stream(impl, door(max_open_s=50.0))
    s.feed(impl.Snooze("door", until=100.0), at=5.0)
    s.observe("door", "yes", at=10.0)
    assert s.fire("door", "max_open", at=60.0).decisions == ()
    assert s.observe("door", "yes", at=110.0).decisions == ()


def test_a_suppressed_event_met_stays_silent(impl):
    s = Stream(impl, fall(verify="off"))
    s.feed(impl.Snooze("fall", until=100.0), at=5.0)
    assert s.observe("fall", "yes", at=10.0).decisions == ()
    assert s.observe("fall", "yes", at=110.0).decisions == ()        # absorbed, never deferred


def test_snooze_replaces_and_a_past_until_cancels(impl):
    s = Stream(impl, door(clear_after_no=1))
    s.feed(impl.Snooze("door", until=100.0), at=1.0)
    s.feed(impl.Snooze("door", until=20.0), at=2.0)   # replaces: ends at 20
    assert types(s.observe("door", "yes", at=30.0)) == ["met"]
    s.observe("door", "no", at=31.0)
    s.feed(impl.Snooze("door", until=500.0), at=32.0)
    s.feed(impl.Snooze("door", until=10.0), at=33.0)  # in the past: cancels
    assert types(s.observe("door", "yes", at=34.0)) == ["met"]


# These close with Resolve, which sets no evidence floor, so only the cooldown is exercised.


def test_a_backwards_clock_step_never_suppresses_with_zero_cooldown(impl):
    s = Stream(impl, door())
    occurrence = s.observe("door", "yes", at=100.0).decisions[0].occurrence_id
    s.feed(impl.Resolve(occurrence), at=101.0)
    assert types(s.observe("door", "yes", at=50.0, evidence=(49.0, 49.5))) == ["met"]


def test_a_backwards_clock_step_counts_as_no_time_elapsed_for_cooldown(impl):
    s = Stream(impl, door(cooldown_s=10.0))
    occurrence = s.observe("door", "yes", at=100.0).decisions[0].occurrence_id
    s.feed(impl.Resolve(occurrence), at=101.0)
    r = s.observe("door", "yes", at=50.0, evidence=(49.0, 49.5))
    assert r.decisions == () and s.phase("door") == "holding"
    assert any(e.event == "suppressed" and "cooldown" in e.detail for e in _traces(r))


def test_cooldown_also_applies_to_unverified_but_never_holds_back_a_met(impl):
    s = Stream(impl, fall(cooldown_s=100.0, candidate_deadline_s=30.0))
    s.observe("fall", "yes", at=10.0)
    assert types(s.fire("fall", "verify", at=40.0)) == ["uncertain"]
    s.observe("fall", "yes", at=50.0)
    assert s.fire("fall", "verify", at=80.0).decisions == ()           # within the cooldown
    [verify] = effects(s.observe("fall", "yes", at=90.0), impl.RequestVerify)
    assert types(s.verify_reply(verify, "yes", at=91.0)) == ["met"]


# --------------------------------------------------------------------------- capacity


def test_capacity_never_evicts_an_active_occurrence_and_refuses_a_new_subject(impl):
    s = Stream(impl, door(clear_after_no=1), max_keys=2)
    s.observe("door", "yes", at=10.0, subject="p1")
    s.observe("door", "yes", at=11.0, subject="p2")
    r = s.observe("door", "yes", at=12.0, subject="p3")
    assert r.decisions == () and fates(r) == ["dropped_capacity"]
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
    return [d for r in s.results for d in r.decisions]


def test_decision_ids_are_stable_and_unique(impl):
    first, again = _story(impl), _story(impl)
    assert [d.decision_id for d in first] == [d.decision_id for d in again]
    ids = [d.decision_id for d in first]
    assert len(ids) == len(set(ids)) == 5
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
    assert s.phase(condition.id) == phase.split(":")[1]
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
    assert [e for e in r.effects if type(e).__name__ != "Trace"] == []


@pytest.mark.parametrize("phase", ALL_PHASES)
def test_every_phase_has_a_defined_outcome_for_every_answer(impl, phase):
    """No undefined cells: every answer in every phase steps without error and leaves a fate."""
    for answer, confidence in (("yes", 0.9), ("no", 0.9), ("yes", 0.3), ("no", 0.3), ("unknown", 0.9)):
        for purpose in ("detect", "verify"):
            s, condition = _drive_to(impl, phase)
            r = s.reply(s.dispatch(condition.id, purpose=purpose), answer, confidence, at=11.0)
            assert len(fates(r)) == 1


# --------------------------------------------------------------------------- expiry after a deferred met


def test_expired_after_a_deferred_met_carries_the_delivered_mets_confidence(impl):
    s = Stream(impl, door(clear_after_no=1, cooldown_s=100.0, max_open_s=500.0))
    assert types(s.observe("door", "yes", 0.9, at=10.0)) == ["met"]
    s.observe("door", "no", at=20.0)
    assert s.observe("door", "yes", 0.7, at=30.0).decisions == ()      # held; met held back by cooldown
    [met] = s.observe("door", "yes", 0.95, at=120.0).decisions
    assert met.cause == "deferred" and met.confidence == 0.95
    [expired] = s.fire("door", "max_open", at=600.0).decisions
    assert expired.type == "expired" and expired.confidence == 0.95    # the met the agent received


# --------------------------------------------------------------------------- condition removal


def test_removing_a_condition_deletes_its_keys_without_a_decision(impl):
    s = Stream(impl, door(emit_cleared=True), fall(), Condition("kettle", "The kettle boils", confirm=2))
    s.observe("door", "yes", at=10.0)                 # holding
    [verify] = effects(s.observe("fall", "yes", at=10.0), impl.RequestVerify)
    s.observe("kettle", "yes", at=10.0)               # confirming
    in_flight = s.dispatch("door")
    s.feed(impl.Snooze("door", until=1_000.0), at=11.0)

    r = s.remove("door", at=12.0)
    assert r.decisions == ()                          # a removed condition says nothing more
    assert effects(r, impl.CancelTimer) == [impl.CancelTimer("door", "scene", "recheck")]
    assert "condition_removed" in trace_events(r)
    snapshot = impl.to_dict(s.state)
    assert "door" not in snapshot["versions"] and snapshot["snoozes"] == []
    assert [k["condition_id"] for k in snapshot["keys"]] == ["fall", "kettle"]
    assert impl.gen_for(s.state, "door") == 0

    r = s.remove("fall", at=13.0)
    assert r.decisions == ()
    assert effects(r, impl.CancelVerify) == [
        impl.CancelVerify("fall", "scene", verify.occurrence_id, verify.occurrence_gen)]

    late = s.reply(in_flight, "yes", at=14.0)
    assert late.decisions == () and stale_details(late) == ["unknown_condition"]

    s.remove("kettle")                                # unannounced: found by version sync
    r = s.feed(impl.SceneReturn("kettle"), at=15.0)
    assert "condition_removed" in trace_events(r) and effects(r, impl.CancelTimer)
    assert impl.to_dict(s.state)["keys"] == [] and impl.to_dict(s.state)["versions"] == {}


def test_conditions_that_vanished_unannounced_are_purged_at_epoch_start(impl):
    conditions = [Condition(f"c{i}", f"thing {i}") for i in range(300)]
    s = Stream(impl, *conditions)
    assert len(impl.to_dict(s.state)["versions"]) == 300
    for c in conditions:
        s.remove(c.id)                                # never announced
    s.feed(impl.EpochStarted(1, restart=False), at=10.0)
    assert impl.to_dict(s.state)["versions"] == {}


def test_a_snooze_for_an_unknown_condition_is_not_recorded(impl):
    s = Stream(impl, door())
    r = s.feed(impl.Snooze("ghost", until=100.0), at=1.0)
    assert impl.to_dict(s.state)["snoozes"] == []
    assert "control_ignored" in trace_events(r)


# --------------------------------------------------------------------------- snapshots


def test_a_snapshot_carries_its_format(impl):
    assert impl.to_dict(impl.new_state("cam-1", created_at=0.0))["format"] == 1


def _valid_snapshot(impl) -> dict:
    s = Stream(impl, door(), register=False)
    s.observe("door", "yes", at=10.0)
    s.feed(impl.Snooze("door", until=100.0), at=11.0)
    return json.loads(s.snapshot())


@pytest.mark.parametrize("corrupt", [
    "empty", "unknown_key_field", "bad_snoozes", "keys_none", "bogus_phase", "future_format", "no_format",
])
def test_a_malformed_snapshot_raises_value_error(impl, corrupt):
    data = _valid_snapshot(impl)
    impl.from_dict(data)                              # the uncorrupted snapshot loads
    if corrupt == "empty":
        data = {}
    elif corrupt == "unknown_key_field":
        data["keys"][0]["surprise"] = 1
    elif corrupt == "bad_snoozes":
        data["snoozes"] = [["door", 100.0]]
    elif corrupt == "keys_none":
        data["keys"] = None
    elif corrupt == "bogus_phase":
        data["keys"][0]["phase"] = "bogus"
    elif corrupt == "future_format":
        data["format"] = 2
    else:
        del data["format"]
    with pytest.raises(ValueError):
        impl.from_dict(data)


# --------------------------------------------------------------------------- restart cost


def test_an_epoch_start_with_thousands_of_changed_conditions_is_fast(impl):
    """Keys are grouped by condition once: O(K), not O(conditions x keys)."""
    if impl is reference_model:
        pytest.skip("the reference model optimises for reading, not speed")
    n = 4_000
    s = Stream(impl, door(), register=False)
    s.observe("door", "yes", at=10.0)
    data = impl.to_dict(s.state)                      # one held key, copied to every condition
    [held] = data["keys"]
    data["max_keys"] = n
    data["versions"] = {f"c{i:04d}": 1 for i in range(n)}
    data["keys"] = [{**held, "condition_id": f"c{i:04d}", "occurrence_id": f"cam-1/c{i:04d}/0-{i}"}
                    for i in range(n)]
    state = impl.from_dict(data)
    conditions = {f"c{i:04d}": Condition(f"c{i:04d}", "x", version=2) for i in range(n)}  # all changed
    started = time.perf_counter()
    result = impl.step(state, impl.EpochStarted(1, restart=True), 20.0, conditions)
    elapsed = time.perf_counter() - started
    assert len(result.decisions) == 0 and all(k["phase"] == "idle" for k in impl.to_dict(result.state)["keys"])
    assert elapsed < 1.0, f"epoch start took {elapsed:.2f}s for {n} conditions and keys"


# --------------------------------------------------------------------------- properties
#
# One generator of random runs, driven two ways: by Hypothesis (which shrinks a failure to a
# minimal story) and by a seeded uniform random walk (which reaches deep paths, such as
# verify -> report -> rearm, far more often than Hypothesis's biased draws do).


GAPS = (0.0, 0.5, 3.0, 12.0, 25.0, 45.0, 130.0)
VERIFY_GAPS = (0.0, 0.5, 3.0)
LAGS = (0.0, 0.5, 3.0, 8.0)        # judge latency: evidence ends this long before arrival
# Mostly judge answers, so occurrences live long enough to be verified, held and cleared.
INPUT_MIX = (["detect"] * 14 + ["verify"] * 6 + ["timer"] * 3
             + ["resolve", "false_alarm", "scene_return", "epoch", "update", "snooze", "remove"])
FLAVOURS = {"roomy": (64, ("scene", "p1")), "crowded": (1, ("scene", "p1", "p2"))}
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
    """Drive a stream with random inputs. Returns the (unregistered) stream."""
    max_keys, subjects = FLAVOURS[choose(sorted(FLAVOURS))]
    s = Stream(impl, _random_condition(choose), max_keys=max_keys, register=False)
    s.subjects = subjects
    now = 1.0
    seen_ids: list[str] = []
    for _ in range(steps if steps is not None else choose(range(1, 41))):
        what = choose(INPUT_MIX)
        if "c" not in s.conditions and choose([True, False, False]):
            what = "update"    # a removed condition comes back soon, so lifecycles still run deep
        # verification replies come back quickly; everything else may arrive at any gap
        now += choose(VERIFY_GAPS if what == "verify" else GAPS)
        cond = s._condition("c")
        subject = choose(subjects)
        if what in ("detect", "verify"):
            verifying = [x for x in subjects if s.phase("c", x) == "verifying"]
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
            lag = choose(LAGS)
            if failure:
                answer, confidence = "unknown", 0.0
            o = Observation(answer=answer, confidence=confidence, reason="r", stream_id=s.stream_id,
                            evidence_t=(now - lag - 1.0, now - lag), judge="fake", failure=failure, **tags)
            inp = impl.Observed(o, SHARE)
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
            # an update, or a re-add after a removal
            s.conditions = {"c": replace(cond, version=cond.version + 1)}
            s.retired.pop("c", None)
            inp = impl.ConditionChanged("c")
        elif what == "remove":
            if "c" not in s.conditions:
                continue
            announce = choose([True, False])
            s.remove("c")
            if not announce:
                continue      # the next input naming it (or an epoch start) syncs the removal
            inp = impl.ConditionChanged("c")
        else:
            inp = impl.Snooze("c", until=now + choose([-1.0, 5.0, 50.0]))
        result = s.feed(inp, at=now)
        seen_ids += [d.occurrence_id for d in result.decisions]
    return s


# The invariants, each checked on one random run.


def check_no_occurrence_is_met_twice(impl, s):
    mets = [d.occurrence_id for r in s.results for d in r.decisions if d.type == "met"]
    assert len(mets) == len(set(mets))
    ids = [d.decision_id for r in s.results for d in r.decisions]
    assert len(ids) == len(set(ids))


def check_every_observation_has_exactly_one_fate(impl, s):
    for (name, _, _, _), result in zip(s.script.steps, s.results):
        assert len(fates(result)) == (1 if name == "Observed" else 0)


def check_the_input_state_is_never_mutated(impl, s):
    state = impl.new_state(s.stream_id, created_at=0.0, max_keys=s.script.max_keys)
    for name, kwargs, now, conds in s.script.steps:
        before = impl.to_dict(state)
        after = impl.step(state, _rebuild(impl, name, kwargs), now, conds).state
        assert impl.to_dict(state) == before
        state = after


def check_a_snapshot_round_trip_changes_nothing(impl, s, where):
    final, outputs = run_script(impl, s.script)
    final_rt, outputs_rt = run_script(impl, s.script, round_trip_at=where)
    assert outputs_rt == outputs
    assert impl.to_dict(final_rt) == impl.to_dict(final)


def check_failures_and_weak_answers_never_change_state(impl, s):
    # Let every deadline pass first, so the only thing that could change state is the answer.
    for subject in s.subjects:
        for kind in impl.TIMER_KINDS:
            s.fire("c", kind, at=LATER, subject=subject)
    for subject in s.subjects:
        for answer, confidence, failure in (("unknown", 0.0, "timeout"), ("unknown", 0.0, "refused"),
                                            ("yes", 0.1, None), ("no", 0.1, None), ("unknown", 0.9, None)):
            for purpose in ("detect", "verify"):
                before = impl.decision_view(s.state)
                r = s.reply(s.dispatch("c", subject, purpose=purpose), answer, confidence, at=LATER,
                            failure=failure)
                assert r.decisions == ()
                assert impl.decision_view(s.state) == before


def check_the_live_run_matches_the_reference(impl, s):
    check_matches_reference(impl, s.script, [_outputs(r) for r in s.results])


PROPERTY_SETTINGS = settings(max_examples=300, deadline=None,
                             suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])


def _hypothesis_choose(data):
    return lambda options: data.draw(st.sampled_from(list(options)))


@PROPERTY_SETTINGS
@given(data=st.data())
def test_property_no_occurrence_is_met_twice(impl, data):
    check_no_occurrence_is_met_twice(impl, _random_run(impl, _hypothesis_choose(data)))


@PROPERTY_SETTINGS
@given(data=st.data())
def test_property_failures_and_weak_answers_never_change_state(impl, data):
    check_failures_and_weak_answers_never_change_state(impl, _random_run(impl, _hypothesis_choose(data)))


@PROPERTY_SETTINGS
@given(data=st.data())
def test_property_every_observation_has_exactly_one_fate(impl, data):
    check_every_observation_has_exactly_one_fate(impl, _random_run(impl, _hypothesis_choose(data)))


@PROPERTY_SETTINGS
@given(data=st.data(), where=st.floats(0, 1))
def test_property_a_snapshot_round_trip_at_any_point_changes_nothing(impl, data, where):
    s = _random_run(impl, _hypothesis_choose(data))
    check_a_snapshot_round_trip_changes_nothing(impl, s, int(where * len(s.script.steps)))


@PROPERTY_SETTINGS
@given(data=st.data())
def test_property_the_input_state_is_never_mutated(impl, data):
    check_the_input_state_is_never_mutated(impl, _random_run(impl, _hypothesis_choose(data)))


@PROPERTY_SETTINGS
@given(data=st.data())
def test_property_matches_the_reference_model(impl, data):
    check_the_live_run_matches_the_reference(impl, _random_run(impl, _hypothesis_choose(data)))


DIFFERENTIAL_SEEDS = range(1_000, 1_400)    # disjoint from the invariant walks' seeds
DIFFERENTIAL_RUNS_PER_SEED = 3
DIFFERENTIAL_STEPS_PER_RUN = 100
DIFFERENTIAL_MIN_STEPS = 100_000


def test_generated_runs_match_the_reference_model(impl):
    """The done-criterion for an implementation: identical decisions and effects on every
    generated run, well over 100,000 generated inputs in total."""
    if impl is reference_model:
        pytest.skip("the reference model is the oracle")
    total = 0
    for seed in DIFFERENTIAL_SEEDS:
        rng = random.Random(seed)
        for run in range(DIFFERENTIAL_RUNS_PER_SEED):
            s = _random_run(impl, rng.choice, steps=DIFFERENTIAL_STEPS_PER_RUN)
            try:
                check_the_live_run_matches_the_reference(impl, s)
            except AssertionError as exc:
                raise AssertionError(f"differs from the reference model: seed {seed}, run {run}") from exc
            total += len(s.script.steps)
    assert total >= DIFFERENTIAL_MIN_STEPS


@pytest.mark.parametrize("seed", range(40))
def test_random_walks_keep_every_invariant(impl, seed):
    rng = random.Random(seed)
    for _ in range(10):
        s = _random_run(impl, rng.choice, steps=60)
        check_no_occurrence_is_met_twice(impl, s)
        check_every_observation_has_exactly_one_fate(impl, s)
        check_the_input_state_is_never_mutated(impl, s)
        check_a_snapshot_round_trip_changes_nothing(impl, s, rng.randrange(len(s.script.steps) + 1))
        check_the_live_run_matches_the_reference(impl, s)
        check_failures_and_weak_answers_never_change_state(impl, s)
