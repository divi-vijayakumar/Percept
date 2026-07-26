"""— the task-level capability suite: the deterministic UNIT layer for cognition.

Nine capabilities, one assertion each, driven off the synthetic corpora
(``percept.harness.synthetic.corpora``) through :class:`~percept.harness.scripted.ScriptedJudge` — no
API key, no spend, no statistical bar. ``ScriptedJudge`` stands in for a PERFECT classifier (the
corpus's own labelled ground truth, replayed in order); what is under test is the REAL, deterministic
runtime that consumes its output: :class:`~percept.episodes.EpisodeLog`, :mod:`percept.claims`,
:func:`~percept.episodes.interval_coverage`, :class:`~percept.gate.Gate` and
:class:`~percept.behavior.BehaviorExecutor`. None of these are test doubles — they are the shipped
mechanisms Y1–Y15 built; this module is their acceptance gate (the implementation wires it into CI next).

Where a capability genuinely fails against the real runtime, the test is
``@unittest.expectedFailure`` with the exact defect in its docstring — never a weakened assertion.

+------+---------------------------------------------------------+---------------------------+--------+
| Cap. | What it proves                                           | Corpus / case             | Status |
+======+===========================================================+===========================+========+
| 1    | Two same-region same-label sightings -> TWO subjects     | subject_identity          | PASS   |
| 2    | Unreadable key -> binding='unknown', never a fabricated  | subject_identity          | PASS   |
|      | key                                                       |                            |        |
| 3    | Same subject inside the gap continues; past it, a new    | continuation               | PASS   |
|      | episode opens                                             |                            |        |
| 4    | A, B, A produces no spurious third fire                  | continuation               | PASS   |
| 5    | B failing its bar does not let A refire                  | continuation               | PASS*  |
| 6    | Only an explicit replace mutates; a question/attribution/ | mutation                  | PASS   |
|      | negation does not                                         |                            |        |
| 7    | The cup problem: 3 occurrences count as 3, not 50 ticks   | episode_counting           | PASS   |
| 8    | t=1 and t=20 is NOT a 19-second interval (no forward-fill)| intervals                  | PASS   |
| 9    | An unchanged confirmation emits no second incident        | idempotency                | PASS   |
+------+---------------------------------------------------------+---------------------------+--------+

  * Cap. 5's own criterion (A never refires) passes. A DIFFERENT assumption baked into the same
    fixture's full per-step trace — that a SECOND unresolved low-confidence read of an already-ASKed
    subject should go SILENT rather than ASK again — does not hold against the real gate/fold and is
    filed separately as an ``expectedFailure`` (see ``CapabilityFiveFullTraceTests`` below) so it does
    not mask the capability itself.
"""

from __future__ import annotations

import asyncio
import unittest

from percept.adapters import Goal, Verdict
from percept.behavior import BehaviorExecutor, BehaviorProgram
from percept.claims import fold_verdicts
from percept.contracts import SubjectRef
from percept.episodes import EpisodeLog, EpisodePolicy, Observation, interval_coverage
from percept.gate import Gate
from percept.harness.scripted import ScriptedJudge
from percept.harness.synthetic.corpora import (
    continuation as continuation_corpus,
    episode_counting as episode_counting_corpus,
    idempotency as idempotency_corpus,
    intervals as intervals_corpus,
    mutation as mutation_corpus,
    subject_identity as subject_identity_corpus,
)


def _run(coro):
    return asyncio.run(coro)


def _goal() -> Goal:
    return Goal(id="g", condition="c", say="fired", state={"type": "watch"})


def _case(built: tuple[dict, dict], case_id: str) -> tuple[list[dict], dict]:
    """Pull one case's ``steps`` (from the stream doc) and ``expect`` (from the labels doc)."""
    stream, labels = built
    steps = next(c["steps"] for c in stream["cases"] if c["id"] == case_id)
    expect = next(c["expect"] for c in labels["cases"] if c["id"] == case_id)
    return steps, expect


async def _scripted_verdicts(steps: list[dict]) -> list[Verdict]:
    """Replay a frame-kind case's ``steps`` through a REAL :class:`ScriptedJudge`, one ``judge()``
    call per step, in order — the seam the card asks for: each step's labelled ``verdict`` block is
    what a perfect classifier would answer; the judge is what actually hands it to the runtime."""
    judge = ScriptedJudge.from_steps(steps)
    return [await judge.judge("scripted condition", s["frame"]) for s in steps]


def _subject_ref(v: Verdict) -> SubjectRef:
    """Build a REAL :class:`SubjectRef` from a subject_identity case's scripted extras. ``binding`` and
    ``subject_key`` are additive fields the fixtures label ahead of a real judge emitting
    ``verdict.subject`` directly (see ``scripted.py``) — this is the translation a real backend will
    one day do itself; today the test does it so the DOWNSTREAM identity/episode wiring is exercised
    for real."""
    if v.binding == "resolved":
        return SubjectRef("content", "doc", v.subject_key, "bound", "durable", "judge")
    return SubjectRef("content", "doc", None, "unknown", "session", "judge")


# ====================================================================================================
# Capability 1 — two same-region, same-label sightings resolve to TWO subjects.
# ====================================================================================================


class Capability01DistinctSubjects(unittest.TestCase):
    """subject_identity/same_region_same_label_different_keys — the IoU+label collapse this corpus
    pins: identical region, identical label, different content key. The real seam that must express
    TWO subjects is ``Verdict.subject`` (a bound :class:`SubjectRef`) feeding
    ``EpisodeLog._subject_match`` — not IoU/label continuity, which cannot see the difference."""

    def test_two_distinct_content_keys_open_two_episodes(self):
        steps, expect = _case(subject_identity_corpus.build(), "same_region_same_label_different_keys")
        scripted = _run(_scripted_verdicts(steps))

        log = EpisodeLog("g")
        for s, v in zip(steps, scripted):
            real = Verdict(v.satisfied, v.confidence, subject=_subject_ref(v))
            log.observe_decision(s["t"], real)

        self.assertEqual(log.count(), expect["distinct_subject_count"])
        self.assertEqual(log.count(), 2)
        keys = {e.subject_key for e in log._episodes}
        self.assertEqual(keys, {s["key"] for s in expect["subjects"]})


# ====================================================================================================
# Capability 2 — an unreadable key binds 'unknown', never a fabricated key.
# ====================================================================================================


class Capability02UnknownBindingNeverFabricates(unittest.TestCase):
    """subject_identity/unreadable_key_binds_unknown — the honest THIRD answer. Neither an invented
    key (e.g. a positional ordinal) nor a silently dropped sighting is acceptable; the real
    ``SubjectRef(resolution='unknown')`` -> ``EpisodeLog`` path must express exactly this."""

    def test_unreadable_key_binds_unknown_not_fabricated(self):
        steps, expect = _case(subject_identity_corpus.build(), "unreadable_key_binds_unknown")
        scripted = _run(_scripted_verdicts(steps))
        self.assertEqual(scripted[0].binding, "unknown")

        log = EpisodeLog("g")
        real = Verdict(scripted[0].satisfied, scripted[0].confidence, subject=_subject_ref(scripted[0]))
        decision = log.observe_decision(steps[0]["t"], real)

        # Not fabricated: the episode carries a PROVISIONAL key derived from its own id
        # (UN_<episode_id>), never an invented content key.
        self.assertEqual(decision.episode.subject_key, f"UN_{decision.episode.id}")
        self.assertTrue(decision.episode.subject_key.startswith("UN_"))
        # Not dropped: the sighting still opened an episode (a visible abstention, not a disappearance).
        self.assertEqual(log.count(), 1)
        self.assertEqual(log.count(), expect["distinct_subject_count"])


# ====================================================================================================
# Capability 3 — same subject inside the continuity gap continues; past it, a new episode opens.
# ====================================================================================================


class Capability03ContinuityGap(unittest.TestCase):
    """continuation/{same_subject_inside_gap_is_one_episode, same_subject_after_gap_is_two_episodes}
    — ``EpisodeLog.observe_decision`` against the concern-level ``EpisodePolicy.continuity_gap_s``."""

    def _episode_count(self, case_id: str) -> tuple[int, int]:
        steps, expect = _case(continuation_corpus.build(), case_id)
        scripted = _run(_scripted_verdicts(steps))
        log = EpisodeLog("g", policy=EpisodePolicy(continuity_gap_s=continuation_corpus.CONTINUITY_GAP_S))
        for s, v in zip(steps, scripted):
            log.observe_decision(s["t"], s["frame"], subject_key=v.subject_key)
        return log.count(), expect["episode_count"]

    def test_inside_gap_is_one_episode(self):
        got, want = self._episode_count("same_subject_inside_gap_is_one_episode")
        self.assertEqual(got, want)
        self.assertEqual(got, 1)

    def test_after_gap_is_two_episodes(self):
        got, want = self._episode_count("same_subject_after_gap_is_two_episodes")
        self.assertEqual(got, want)
        self.assertEqual(got, 2)


# ====================================================================================================
# Shared drive for the fire/ask/silent capabilities (4, 5, 9) — EpisodeLog + fold + Gate, the same
# wiring ``percept.behavior.BehaviorExecutor.fire_judged`` uses in the live path.
# ====================================================================================================


def _drive_actions(steps: list[dict], scripted: list[Verdict], *, continuity_gap_s: float) -> tuple[list[str], int]:
    gate = Gate()
    ex = BehaviorExecutor(gate)
    goal, prog = _goal(), BehaviorProgram(sense="judge")
    log = EpisodeLog("g", policy=EpisodePolicy(continuity_gap_s=continuity_gap_s))
    key = (goal.id, "scene")
    actions: list[str] = []
    fire_count = 0
    for s, v in zip(steps, scripted):
        decision = log.observe_decision(s["t"], s["frame"], subject_key=v.subject_key)
        fold = ex.fold_episode(decision, mutation="confirmed")
        fired = ex.fire_judged(goal, Verdict(v.satisfied, v.confidence), key, prog, t=s["t"], fold=fold)
        if fired is None:
            actions.append("silent")
        else:
            actions.append(fired.action)
            if fired.action == "fire":
                fire_count += 1
    return actions, fire_count


# ====================================================================================================
# Capability 4 — A, B, A produces no spurious third fire.
# ====================================================================================================


class Capability04NoSpuriousThirdFire(unittest.TestCase):
    """continuation/interleaved_a_b_a_continues_a — episode state is per-SUBJECT: B interleaving
    between two A sightings must not manufacture a THIRD fire on A's return."""

    def test_a_b_a_fires_exactly_twice(self):
        steps, expect = _case(continuation_corpus.build(), "interleaved_a_b_a_continues_a")
        scripted = _run(_scripted_verdicts(steps))
        actions, incidents = _drive_actions(
            steps, scripted, continuity_gap_s=continuation_corpus.CONTINUITY_GAP_S
        )
        self.assertEqual(actions, expect["actions"])
        self.assertEqual(incidents, expect["fire_count"])
        self.assertEqual(incidents, 2)
        self.assertTrue(expect["spurious_third_fire_forbidden"])


# ====================================================================================================
# Capability 5 — B failing its bar (stuck in ASK) must not let A refire.
# ====================================================================================================


class Capability05BFailsDwellDoesNotLetARefire(unittest.TestCase):
    """continuation/b_stuck_in_ask_must_not_refire_a — one subject's pending-ASK state must not govern
    another subject's duplicate suppression. The card's OWN criterion: A fires once, not twice."""

    def test_a_does_not_refire_when_b_is_stuck_in_ask(self):
        steps, expect = _case(continuation_corpus.build(), "b_stuck_in_ask_must_not_refire_a")
        scripted = _run(_scripted_verdicts(steps))
        actions, incidents = _drive_actions(
            steps, scripted, continuity_gap_s=continuation_corpus.CONTINUITY_GAP_S
        )
        # A is steps 0 and 3: fire once, never a second time.
        a_actions = [actions[0], actions[3]]
        self.assertEqual(a_actions, ["fire", "silent"])
        self.assertEqual(incidents, expect["fire_count"])
        self.assertEqual(incidents, 1)
        self.assertTrue(expect["subject_states_independent"])


class Capability05FullTraceDiscrepancy(unittest.TestCase):
    """DISCOVERED DEFECT (filed, not required by the card's own capability-5 wording): the fixture's
    full per-step ``actions`` additionally expects B's SECOND unresolved low-confidence read (step 2)
    to be SILENT, not ASK again. The real gate/fold does not do that — ``episode_fold``'s own
    docstring says a not-yet-emitted continuation "keeps evaluating" the gate on every tick, so a
    still-unresolved subject is asked again each time new evidence arrives, by design. That may well
    be the RIGHT behaviour (repeated evidence should keep being weighed) rather than a bug, but it does
    contradict this fixture's literal ``expect.actions``, so it is filed here rather than silently
    folded into capability 5's own (passing) assertion above."""

    @unittest.expectedFailure
    def test_full_action_trace_matches_the_fixture_literally(self):
        steps, expect = _case(continuation_corpus.build(), "b_stuck_in_ask_must_not_refire_a")
        scripted = _run(_scripted_verdicts(steps))
        actions, _ = _drive_actions(steps, scripted, continuity_gap_s=continuation_corpus.CONTINUITY_GAP_S)
        # got ['fire', 'ask', 'ask', 'silent'] vs fixture's ['fire', 'ask', 'silent', 'silent'].
        self.assertEqual(actions, expect["actions"])


# ====================================================================================================
# Capability 6 — only an explicit replace mutates a claim's value.
# ====================================================================================================


def _operation_of(act: str) -> str:
    """Map a mutation-corpus speech-act label to a real ``Verdict.operation``. ``negation`` has no
    real-vocabulary analogue (see ``percept.contracts.OPERATIONS``) — it is a value referenced and
    denied, not asserted, so it folds to ``mention`` (non-mutating), matching the convention already
    established in ``tests/test_claims.py::_op_of``."""
    return "mention" if act == "negation" else act


class Capability06OnlyExplicitReplaceMutates(unittest.TestCase):
    """mutation/only_one_of_four_25k_utterances_replaces — four of five buyer turns contain '$25K';
    only 'Actually, the approved budget is $25K' (a real ``replace``) changes the value. The judge's
    text->speech-act classification is explicitly out of scope for this fixture (a statistical
    question); ``ScriptedJudge`` is scripted straight from the corpus's OWN labelled acts, and what is
    under test is ``percept.claims`` — the state machine downstream of that classification."""

    def test_only_the_replace_turn_changes_the_value(self):
        _, expect = _case(mutation_corpus.build(), "only_one_of_four_25k_utterances_replaces")
        acts, values = expect["acts"], expect["value_after"]

        script = [
            {
                "satisfied": True,
                "confidence": 0.9,
                "operation": _operation_of(act),
                "value": val if _operation_of(act) in ("assert", "replace") else None,
                "asserted_by": "buyer",
            }
            for act, val in zip(acts, values)
        ]
        judge = ScriptedJudge(script)

        async def _sequential():
            # Sequential, ordered `await`s: the corpus turns are a stream, and ScriptedJudge's cursor
            # consumes the script in call order — the same guarantee `judge.judge(...)` per turn gets
            # in the live path.
            return [await judge.judge("scripted condition", str(i)) for i in range(len(script))]

        verdicts = _run(_sequential())

        state, outcomes = fold_verdicts(verdicts, claim_id="budget")
        self.assertEqual(state.value, expect["final_value"])
        self.assertEqual(state.value, "$25K")
        self.assertEqual([o.value_after for o in outcomes], values)
        # Exactly the turns in utterances_mutating_value actually changed an ESTABLISHED value; turn 0
        # (assert into an empty slot) "applies" but is not a mutation of an existing value.
        changed_indices = [i for i in range(1, len(outcomes)) if outcomes[i].value_after != outcomes[i - 1].value_after]
        self.assertEqual(changed_indices, expect["utterances_mutating_value"])
        self.assertEqual(changed_indices, [4])


# ====================================================================================================
# Capability 7 — the cup problem: 3 separated occurrences count as 3; 50 samples of ONE occurrence
# count as 1, never 50.
# ====================================================================================================


class Capability07CupProblem(unittest.TestCase):
    """episode_counting/{three_separated_occurrences, one_continuous_occurrence_sampled_fifty_times} —
    counting must be over EPISODES (time-gapped occurrences), never over satisfied ticks."""

    def _count(self, case_id: str) -> tuple[int, int]:
        steps, expect = _case(episode_counting_corpus.build(), case_id)
        scripted = _run(_scripted_verdicts(steps))
        log = EpisodeLog("g", policy=EpisodePolicy(continuity_gap_s=episode_counting_corpus.BOUNDARY_GAP_S))
        for s, v in zip(steps, scripted):
            log.observe_decision(s["t"], s["frame"], subject_key=v.subject_key)
        return log.count(), expect["count"]

    def test_three_separated_occurrences_count_as_three(self):
        got, want = self._count("three_separated_occurrences")
        self.assertEqual(got, want)
        self.assertEqual(got, 3)

    def test_fifty_samples_of_one_occurrence_count_as_one_not_fifty(self):
        got, want = self._count("one_continuous_occurrence_sampled_fifty_times")
        self.assertEqual(got, want)
        self.assertEqual(got, 1)
        self.assertNotEqual(got, 50)  # the forbidden per-tick answer


class Capability07BoundaryOffByOne(unittest.TestCase):
    """— fixed. The fixture's documented rule is inclusive: "separation >= boundary_gap_s
    starts a NEW episode". ``EpisodeLog.continuation_reason`` now compares with ``>=`` (formerly a
    strict ``>``) in the ``gap_exceeded`` branch: ``(t - episode.last_seen) >= policy.continuity_gap_s``.
    At a separation of EXACTLY the gap, a new episode opens (matching the corpus), not a continuation."""

    def test_exact_boundary_separation_opens_a_new_episode(self):
        steps, expect = _case(episode_counting_corpus.build(), "boundary_separation_exactly_gap")
        scripted = _run(_scripted_verdicts(steps))
        log = EpisodeLog("g", policy=EpisodePolicy(continuity_gap_s=episode_counting_corpus.BOUNDARY_GAP_S))
        for s, v in zip(steps, scripted):
            log.observe_decision(s["t"], s["frame"], subject_key=v.subject_key)
        self.assertEqual(log.count(), expect["count"])  # got 1, fixture wants 2


# ====================================================================================================
# Capability 8 — t=1 and t=20 is NOT a 19-second interval (no forward-fill).
# ====================================================================================================


class Capability08NoForwardFill(unittest.TestCase):
    """intervals/two_sparse_observations_are_not_a_span — coverage comes from OBSERVATION, never from
    interpolating between two sparse sightings."""

    def test_two_sparse_observations_yield_a_gap_not_a_span(self):
        steps, expect = _case(intervals_corpus.build(), "two_sparse_observations_are_not_a_span")
        scripted = _run(_scripted_verdicts(steps))
        observations = [
            Observation(t=s["t"], observed=bool(v.scripted_extras.get("observed", True)),
                        satisfied=v.satisfied, degraded=bool(v.scripted_extras.get("degraded", False)))
            for s, v in zip(steps, scripted)
        ]
        cov = interval_coverage(observations)

        self.assertEqual(cov.observed_total_s, expect["observed_total_s"])
        self.assertEqual(cov.observed_total_s, 0.0)
        self.assertNotEqual(cov.observed_total_s, 19.0)  # the forbidden forward-filled answer
        self.assertEqual(
            [(g.start, g.end, g.reason) for g in cov.gaps],
            [(g["start"], g["end"], g["reason"]) for g in expect["gaps"]],
        )


# ====================================================================================================
# Capability 9 — an unchanged confirmation emits no second incident.
# ====================================================================================================


class Capability09NoSecondIncidentOnConfirmation(unittest.TestCase):
    """idempotency/unchanged_confirmation_does_not_reincident — re-confirming an already-reported
    episode, unchanged, must produce exactly ONE user-visible incident, never one per tick."""

    def test_unchanged_confirmation_does_not_reincident(self):
        steps, expect = _case(idempotency_corpus.build(), "unchanged_confirmation_does_not_reincident")
        scripted = _run(_scripted_verdicts(steps))
        actions, incidents = _drive_actions(steps, scripted, continuity_gap_s=1000.0)

        self.assertEqual(actions, expect["actions"])
        self.assertEqual(incidents, expect["incident_count"])
        self.assertEqual(incidents, 1)


if __name__ == "__main__":
    unittest.main()
