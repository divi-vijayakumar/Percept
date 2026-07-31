"""— the claim-mode / speech-act channel: a value DIFFERENCE is not a REPLACEMENT.

Two layers:

* **unit** — the `Verdict` additive fields (back-compat), `ConcernAuthority`, the `percept.claims`
  state machine, and the `operation → EpisodeMutation` fold wiring, each asserted directly.
* **fixture-driven** — the synthetic `mutation` corpus (`eval/synthetic/mutation/`) is THE SPEC:
  every labelled case is replayed through the state machine and its `value_after`/`final_value`/
  authority ground truth is asserted. The corpus's own text-→value classification is NOT under test
  (the fixture says so); the acts come from the labels and the value from the utterance, and the test
  proves only the state machine DOWNSTREAM of that classification.
"""

from __future__ import annotations

import logging
import re
import tempfile
import unittest
from pathlib import Path

from percept.behavior import BehaviorExecutor, episode_mutation_of
from percept.claims import ClaimState, apply_operation, fold_verdicts
from percept.contracts import (
    MUTATING_OPERATIONS,
    OPERATIONS,
    Concern,
    ConcernAuthority,
    Subject,
    Verdict,
)
from percept.gate import Gate
from percept.harness.synthetic import generate_all, load_corpus

_VALUE_RE = re.compile(r"\$\d+K")


def _value_in(text: str) -> "str | None":
    """The value token an utterance CARRIES (containing it is not asserting it — that is the point)."""
    m = _VALUE_RE.search(text)
    return m.group(0) if m else None


def _op_of(act: str) -> str:
    """Map a fixture speech-act label to a `Verdict.operation`. `negation` is a `mention` (the value is
    referenced but denied, not asserted); every other label is already an operation."""
    return "mention" if act == "negation" else act


class VerdictAdditiveFieldsTests(unittest.TestCase):
    def test_defaults_reproduce_todays_verdict(self):
        v = Verdict(True, 0.9)
        self.assertEqual(v.operation, "assert")
        self.assertIsNone(v.value)
        self.assertIsNone(v.normalized_value)
        self.assertIsNone(v.target_revision_id)
        self.assertIsNone(v.asserted_by)

    def test_positional_construction_is_unchanged(self):
        # The three original positional fields still mean the same thing — the new fields are appended.
        v = Verdict(True, 0.8, "eyes open")
        self.assertTrue(v.satisfied)
        self.assertEqual(v.confidence, 0.8)
        self.assertEqual(v.evidence, "eyes open")
        self.assertEqual(v.operation, "assert")

    def test_claim_fields_are_settable(self):
        v = Verdict(True, 1.0, operation="replace", value="$25K",
                    normalized_value=25000, target_revision_id="c#r1", asserted_by="buyer")
        self.assertEqual(v.operation, "replace")
        self.assertEqual(v.value, "$25K")
        self.assertEqual(v.normalized_value, 25000)
        self.assertEqual(v.asserted_by, "buyer")


class ConcernAuthorityTests(unittest.TestCase):
    def test_mutating_op_from_unauthorized_asserter_is_denied(self):
        a = ConcernAuthority(frozenset({"buyer"}))
        for op in MUTATING_OPERATIONS:
            self.assertFalse(a.permits("seller", op), op)
            self.assertTrue(a.permits("buyer", op), op)

    def test_non_mutating_ops_are_always_permitted(self):
        a = ConcernAuthority(frozenset({"buyer"}))
        for op in ("assert", "confirm", "mention"):
            self.assertTrue(a.permits("seller", op), op)
            self.assertTrue(a.permits(None, op), op)

    def test_empty_authority_is_unrestricted(self):
        a = ConcernAuthority()  # empty set == no restriction, never a total lockout by surprise
        self.assertTrue(a.permits("anyone", "replace"))
        self.assertTrue(a.permits(None, "retract"))

    def test_concern_default_authority_is_none(self):
        c = Concern(id="c", identity=Subject(kind="entity", ref="x"), condition="x", say="y")
        self.assertIsNone(c.authority)


class ClaimStateTests(unittest.TestCase):
    def setUp(self) -> None:
        logging.disable(logging.CRITICAL)  # silence the authority-downgrade trace during unit tests

    def tearDown(self) -> None:
        logging.disable(logging.NOTSET)

    def test_assert_establishes_an_empty_slot(self):
        s = ClaimState(claim_id="c")
        out = apply_operation(s, "assert", value="$40K")
        self.assertEqual(s.value, "$40K")
        self.assertTrue(out.applied)
        self.assertEqual(out.revision_id, "c#r1")

    def test_assert_does_not_overwrite_an_established_value(self):
        s = ClaimState(claim_id="c", value="$40K", revisions=1, revision_id="c#r1")
        out = apply_operation(s, "assert", value="$25K")
        self.assertEqual(s.value, "$40K")  # a value difference is NOT a replacement
        self.assertFalse(out.applied)

    def test_mention_and_confirm_leave_the_value_untouched(self):
        for op in ("mention", "confirm"):
            s = ClaimState(claim_id="c", value="$40K", revisions=1, revision_id="c#r1")
            out = apply_operation(s, op, value="$25K")
            self.assertEqual(s.value, "$40K", op)
            self.assertFalse(out.applied, op)

    def test_replace_supersedes_and_names_the_prior_revision(self):
        s = ClaimState(claim_id="c")
        apply_operation(s, "assert", value="$40K")  # c#r1
        out = apply_operation(s, "replace", value="$25K")
        self.assertEqual(s.value, "$25K")
        self.assertTrue(out.applied)
        self.assertEqual(out.supersedes, "c#r1")
        self.assertEqual(out.revision_id, "c#r2")

    def test_retract_is_valueless_and_clears_to_none(self):
        s = ClaimState(claim_id="c")
        apply_operation(s, "assert", value="$40K")
        out = apply_operation(s, "retract")
        self.assertIsNone(s.value)  # None, never "" — absence has one spelling
        self.assertTrue(out.applied)
        self.assertEqual(out.supersedes, "c#r1")

    def test_close_clears_the_value(self):
        s = ClaimState(claim_id="c")
        apply_operation(s, "assert", value="$40K")
        apply_operation(s, "close")
        self.assertIsNone(s.value)

    def test_normalized_value_is_folded_in_parallel(self):
        s = ClaimState(claim_id="c")
        apply_operation(s, "assert", value="$40K", normalized_value=40000)
        self.assertEqual(s.normalized_value, 40000)
        apply_operation(s, "replace", value="$25K", normalized_value=25000)
        self.assertEqual(s.normalized_value, 25000)


class IdenticalValueReplaceTests(unittest.TestCase):
    """— re-asserting the SAME value: `applied` is a value DIFF, so a genuine speech act vanishes.

    `apply_operation` decides "did this write?" by diffing the slot. A speaker who says "$25K" again
    — a real, deliberate re-assertion of the agreed number — moves no bytes, so the fold reports
    `applied=False`, mints no revision, and the utterance leaves NO trace on the timeline. That is
    documented behavior and these tests pin it; the sharp edge worth naming is that the very same
    outcome still fills in `supersedes` with the head revision it did not supersede.

    The slot being diffed is the PAIR `(value, normalized_value)` (BUG-B-05-01), so "no bytes moved"
    now means no bytes moved in EITHER field — an omitted normalization is not a cleared one, and a
    normalized-only correction is a real write.
    """

    def setUp(self) -> None:
        logging.disable(logging.CRITICAL)

    def tearDown(self) -> None:
        logging.disable(logging.NOTSET)

    @staticmethod
    def _established() -> ClaimState:
        """A slot established at $25K by an `assert` — one revision, `c1#r1` at the head."""
        s = ClaimState(claim_id="c1")
        out = apply_operation(s, "assert", value="$25K")
        # Guard the premise: the contrast below is only meaningful against a REAL first write.
        assert (out.applied, out.revision_id, out.supersedes) == (True, "c1#r1", None)
        assert (s.revisions, s.revision_id) == (1, "c1#r1")
        return s

    def test_replace_with_an_identical_value_mints_no_revision(self):
        """The row: re-stating the established value is not a write, so the ledger does not move."""
        s = self._established()
        out = apply_operation(s, "replace", value="$25K")

        self.assertFalse(out.applied)  # applied is `state.value != before`, and it did not differ
        self.assertIsNone(out.revision_id)  # nothing minted — the re-assertion is off the timeline
        self.assertEqual(s.value, "$25K")  # the value itself is intact
        self.assertEqual(s.revisions, 1)  # still ONE revision after TWO assertions of $25K
        self.assertEqual(s.revision_id, "c1#r1")  # the head never advanced
        self.assertEqual(out.effective, "replace")  # not downgraded — it was permitted and ran

    def test_an_unapplied_replace_still_names_a_predecessor_it_never_superseded(self):
        """The asymmetry. `supersedes` is read off the head BEFORE the value fold and is returned
        regardless of whether anything was written, so the outcome says "I superseded c1#r1" while
        `c1#r1` is still the live head and no successor exists to have replaced it."""
        s = self._established()
        out = apply_operation(s, "replace", value="$25K")

        self.assertEqual(out.supersedes, "c1#r1")  # a predecessor IS named…
        self.assertEqual(s.revision_id, "c1#r1")  # …and it is also still the head
        self.assertIsNone(out.revision_id)  # …with nothing minted to have superseded it
        # Stated as the invariant it breaks: a named `supersedes` normally implies a successor.
        self.assertEqual((out.supersedes is not None, out.revision_id is not None), (True, False))

    def test_three_assertions_of_one_value_leave_a_one_row_timeline(self):
        """The user-visible consequence: repetition is compressed away entirely. Say "$25K" three
        times and the revision chain records a single write, so a reader of the ledger cannot tell a
        value that was stated once from one that was re-affirmed twice."""
        s = self._established()
        second = apply_operation(s, "replace", value="$25K")
        third = apply_operation(s, "replace", value="$25K")

        self.assertEqual([second.applied, third.applied], [False, False])
        self.assertEqual([second.revision_id, third.revision_id], [None, None])
        self.assertEqual(s.revisions, 1)

    def test_replace_with_a_different_value_does_mint_the_next_revision(self):
        """The control, so the assertions above are not passing for a trivial reason: the SAME call
        shape with a genuinely different value writes, mints `c1#r2`, and supersedes `c1#r1`."""
        s = self._established()
        out = apply_operation(s, "replace", value="$30K")

        self.assertTrue(out.applied)
        self.assertEqual(s.value, "$30K")
        self.assertEqual(s.revisions, 2)
        self.assertEqual(out.revision_id, "c1#r2")
        self.assertEqual(s.revision_id, "c1#r2")
        self.assertEqual(out.supersedes, "c1#r1")

    def test_a_no_op_replace_does_not_burn_a_revision_number(self):
        """And the no-op leaves no hole: the next REAL correction is still `c1#r2` superseding
        `c1#r1`, so an identical re-assertion cannot desynchronize the chain."""
        s = self._established()
        apply_operation(s, "replace", value="$25K")  # the no-op
        out = apply_operation(s, "replace", value="$30K")  # the real correction, right after

        self.assertEqual(out.revision_id, "c1#r2")  # no gap, no skipped number
        self.assertEqual(out.supersedes, "c1#r1")
        self.assertEqual(s.revisions, 2)

    def test_an_unapplied_replace_leaves_the_whole_state_untouched(self):
        """`applied=False` means "nothing happened" — over the WHOLE slot (BUG-B-05-01).

        The fold used to write `normalized_value` unconditionally while `applied` was computed from
        `value` alone, so a `replace` restating $25K without repeating the normalization — exactly
        what `Verdict.normalized_value` defaulting to `None` produces — kept the display value and
        dropped the machine-readable 25000, with no revision and no `applied` flag to notice it by.

        Fixed on the omission side: `None` means NOT STATED, so the canonical form survives.
        """
        s = ClaimState(claim_id="c1")
        apply_operation(s, "assert", value="$25K", normalized_value=25000)
        before = (s.value, s.normalized_value, s.revisions, s.revision_id)

        out = apply_operation(s, "replace", value="$25K")  # same value, normalization not restated

        self.assertFalse(out.applied)
        self.assertEqual(
            (s.value, s.normalized_value, s.revisions, s.revision_id),
            before,  # ("$25K", 25000, 1, "c1#r1")
        )
        self.assertEqual(out.normalized_after, 25000)  # and the outcome reports the live form

    def test_an_omitted_normalization_on_assert_also_never_clears_one(self):
        """The same asymmetry lived on the `assert` branch, which the bug report missed. `assert`
        only establishes an EMPTY value slot, but it wrote `normalized_value` unconditionally when
        it did, so an establish that omits the canonical form could wipe one already present."""
        s = ClaimState(claim_id="c1", normalized_value=25000)  # a canonical form with no display text
        out = apply_operation(s, "assert", value="$25K")  # establishes; normalization not restated

        self.assertEqual(s.value, "$25K")
        self.assertEqual(s.normalized_value, 25000)  # kept — None was an omission, not a clear
        self.assertTrue(out.applied)

    def test_a_normalized_only_correction_is_a_real_write_that_mints_a_revision(self):
        """The behavior change `applied` over the pair buys: a correction that moves ONLY the
        canonical form (the display text was already right, the parse was wrong) is a genuine write.
        It used to change state invisibly — no `applied`, no revision, nothing on the timeline."""
        s = ClaimState(claim_id="c1")
        apply_operation(s, "assert", value="$25K", normalized_value=2500)  # a bad parse

        out = apply_operation(s, "replace", value="$25K", normalized_value=25000)

        self.assertTrue(out.applied)
        self.assertEqual(s.normalized_value, 25000)
        self.assertEqual(out.revision_id, "c1#r2")
        self.assertEqual(out.supersedes, "c1#r1")
        self.assertEqual(s.revisions, 2)

    def test_a_retraction_still_clears_the_canonical_form_unconditionally(self):
        """The one place an unconditional clear is correct: a retraction is an EXPLICIT clear, not
        an omission, so both fields go to None even though `retract` carries no values at all."""
        s = ClaimState(claim_id="c1")
        apply_operation(s, "assert", value="$25K", normalized_value=25000)

        out = apply_operation(s, "retract")

        self.assertIsNone(s.value)
        self.assertIsNone(s.normalized_value)  # NOT preserved — a retraction means "there is none"
        self.assertTrue(out.applied)
        self.assertIsNone(out.normalized_after)


class AuthorityDowngradeTests(unittest.TestCase):
    def test_unauthorized_replace_is_downgraded_not_applied(self):
        logging.disable(logging.CRITICAL)
        try:
            s = ClaimState(claim_id="budget", value="$40K", revisions=1, revision_id="budget#r1")
            auth = ConcernAuthority(frozenset({"buyer"}))
            out = apply_operation(s, "replace", value="$25K", asserted_by="seller", authority=auth)
            self.assertTrue(out.downgraded)
            self.assertEqual(out.effective, "mention")
            self.assertFalse(out.applied)
            self.assertEqual(s.value, "$40K")  # the buyer's value stands
        finally:
            logging.disable(logging.NOTSET)

    def test_downgrade_is_traced_never_silent(self):
        auth = ConcernAuthority(frozenset({"buyer"}))
        s = ClaimState(claim_id="budget", value="$40K", revisions=1, revision_id="budget#r1")
        with self.assertLogs("percept.claims", level="INFO") as cm:
            apply_operation(s, "replace", value="$25K", asserted_by="seller", authority=auth)
        self.assertTrue(any("claim.authority_downgrade" in line for line in cm.output))

    def test_authorized_replace_is_applied(self):
        auth = ConcernAuthority(frozenset({"buyer"}))
        s = ClaimState(claim_id="budget", value="$40K", revisions=1, revision_id="budget#r1")
        out = apply_operation(s, "replace", value="$25K", asserted_by="buyer", authority=auth)
        self.assertFalse(out.downgraded)
        self.assertTrue(out.applied)
        self.assertEqual(s.value, "$25K")


class EpisodeMutationWiringTests(unittest.TestCase):
    def test_operation_maps_to_episode_mutation(self):
        self.assertEqual(episode_mutation_of("assert"), "confirmed")
        self.assertEqual(episode_mutation_of("confirm"), "confirmed")
        self.assertEqual(episode_mutation_of("mention"), "confirmed")
        self.assertEqual(episode_mutation_of("replace"), "updated")
        self.assertEqual(episode_mutation_of("retract"), "retracted")
        self.assertEqual(episode_mutation_of("close"), "closed")

    def test_every_operation_is_mapped(self):
        for op in OPERATIONS:
            self.assertIn(episode_mutation_of(op), ("confirmed", "updated", "retracted", "closed", "none"))

    def test_unknown_operation_folds_to_confirmed(self):
        self.assertEqual(episode_mutation_of("something-else"), "confirmed")

    def test_fold_episode_operation_matches_explicit_mutation(self):
        ex = BehaviorExecutor(Gate())
        # With no decision (unbound) the fold is gate-only regardless — but the derived mutation must
        # equal the explicit one, proving the operation= path is wired to episode_mutation_of.
        by_op = ex.fold_episode(None, operation="replace")
        by_mut = ex.fold_episode(None, mutation="updated")
        self.assertEqual(by_op.mutation, by_mut.mutation)


class _MutationCorpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        generate_all(cls.root)
        cls.corpus = load_corpus("mutation", cls.root)
        logging.disable(logging.CRITICAL)  # the seller case emits a downgrade trace

    @classmethod
    def tearDownClass(cls) -> None:
        logging.disable(logging.NOTSET)
        cls._tmp.cleanup()

    def _verdicts_for(self, case):
        """Turn a labelled case into the classified (operation, value, asserted_by) stream the state
        machine consumes. Acts come from the labels; value from the utterance; speaker == asserted_by."""
        acts = case.expect["acts"]
        verdicts = []
        for step, act in zip(case.steps, acts):
            verdicts.append(
                Verdict(True, 1.0, operation=_op_of(act), value=_value_in(step["text"]),
                        asserted_by=step["speaker"])
            )
        return verdicts

    def _authority_for(self, case):
        req = case.expect.get("authority_required")
        return ConcernAuthority(frozenset({req})) if req else None


class MutationFixtureTests(_MutationCorpus):
    """FIXTURES ARE THE SPEC — every labelled mutation case, replayed through the state machine."""

    def test_every_case_reproduces_its_value_after_and_final_value(self):
        for case in self.corpus:
            with self.subTest(case=case.id):
                verdicts = self._verdicts_for(case)
                state, outcomes = fold_verdicts(
                    verdicts, claim_id=case.id, authority=self._authority_for(case)
                )
                got = [o.value_after for o in outcomes]
                self.assertEqual(got, case.expect["value_after"], f"{case.id}: value_after trace")
                self.assertEqual(state.value, case.expect["final_value"], f"{case.id}: final_value")

    def test_non_mutating_acts_never_change_the_value(self):
        """The decisive claim: a question / attribution / negation CONTAINS the value but never sets it."""
        case = self.corpus.case("only_one_of_four_25k_utterances_replaces")
        verdicts = self._verdicts_for(case)
        _state, outcomes = fold_verdicts(verdicts, claim_id=case.id)
        # Exactly the last utterance (a replace) is the only applied write past the opening assert.
        applied_after_open = [i for i, o in enumerate(outcomes) if o.applied and i > 0]
        self.assertEqual(applied_after_open, [4])
        # ...and turn 4 supersedes turn 0's revision (the $40K it replaced).
        self.assertEqual(outcomes[4].supersedes, outcomes[0].revision_id)
        self.assertEqual(outcomes[0].value_after, case.expect["supersedes"]["superseded_value"])

    def test_valueless_retraction_clears_the_value(self):
        case = self.corpus.case("valueless_retraction_clears_the_value")
        verdicts = self._verdicts_for(case)
        # The retraction turn carries NO value — proving a turn with no number can still clear one.
        self.assertIsNone(verdicts[-1].value)
        state, outcomes = fold_verdicts(verdicts, claim_id=case.id)
        self.assertIsNone(state.value)
        self.assertTrue(outcomes[-1].applied)

    def test_seller_replace_is_blocked_by_authority(self):
        case = self.corpus.case("seller_lacks_authority_to_replace_buyer_value")
        verdicts = self._verdicts_for(case)
        authority = self._authority_for(case)
        self.assertIsNotNone(authority)
        state, outcomes = fold_verdicts(verdicts, claim_id=case.id, authority=authority)
        self.assertEqual(state.value, "$40K")  # the buyer's value stands
        # The per-turn downgrade aligns with the fixture's blocked_by_authority ground truth.
        self.assertEqual(
            [o.downgraded for o in outcomes], case.expect["blocked_by_authority"]
        )


if __name__ == "__main__":
    unittest.main()
