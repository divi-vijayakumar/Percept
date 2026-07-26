"""— CLAIM MODE rollout: structured mutation, authority policy, supersession go LIVE for
claim-slot concerns.

Y12 flipped episode grouping + idempotency to GOVERNING for concerns that SELECT the content-key
resolver (``goal.state['identity_policy'] == 'content_key'``) via ``Percept._document_episode_fold``;
a claim-slot / conversational concern still got ``fold=None`` (shadow-only), pinned by
``test_y12_document_governing.py::TestClaimSlotStaysShadow``. Y13 flips the SAME seam for concerns that
SELECT the claim-slot resolver (``identity_policy == 'claim_slot'``) via the new twin method
``Percept._claim_episode_fold``, wired into ``perceive_judged`` ALONGSIDE (never replacing)
``_document_episode_fold`` — a goal is never both types, so the two are mutually exclusive.

Three landed pieces go live for claim-slot concerns:

1. **The mutation channel (``percept.claims``)** — only an explicit ``replace`` speech act
   supersedes; a question/attribution/negation (all classified ``mention``) never moves the value. The
   frozen ``mutation`` synthetic corpus (``percept.harness.synthetic.corpora.mutation``) pins this.
2. **Per-concern authority at fold time (``ConcernAuthority``)** — a mutating op from an asserter this
   concern's authority (``goal.state['authority']``, :func:`percept.claims.concern_authority_of`) does
   not permit is DOWNGRADED to a non-mutating ``mention`` before it ever reaches the episode fold.
3. **Supersession + provisional-incident delivery (landed)** — an applied correction re-arms
   and fires a superseding revision (``fold.supersedes`` / ``<metric>.superseding_revision``), never a
   silent overwrite; the ORIGINAL and the CORRECTION both land, linked, exactly like a content-key
   concern's ``updated``/``retracted`` continuation.

The acceptance properties below:

* Probe 02's 3 mutation-corpus cases resolve LIVE exactly as the fixture says, and
  ``episode_current`` surfaces the corrected value WITH its revision chain.
* ZERO fabricated retractions/updates on a 32s digression control (unrelated turns must never move
  the value or append a spurious revision).
* Y12's 12 document-governing tests pass UNTOUCHED (verified by running that module unmodified).
* The gate is mechanically unchanged — governing is entirely a pre-gate ``fold=`` decision, exactly
  the Y3/Y10 contract document mode already exercises.
"""

from __future__ import annotations

import asyncio
import unittest

from percept.adapters import Goal, Verdict
from percept.agent import Percept
from percept.backends.fake import FakeVision
from percept.claims import ClaimState, concern_authority_of
from percept.config import GateSettings, Settings
from percept.contracts import ConcernAuthority
from percept.eval.shadow_report import run_probe_02_mutation
from percept.harness.synthetic.corpora import mutation as _mutation


def _build(*, identity_policy: "str | None" = "claim_slot", authority=None, threshold: float = 0.7) -> Percept:
    settings = Settings(gate=GateSettings(threshold=threshold, validity_window_s=10_000.0))
    a = Percept.create(vision=FakeVision(), settings=settings, discover_plugins=False)
    state = {"type": "watch", "count": 0}
    if identity_policy is not None:
        state["identity_policy"] = identity_policy
    if authority is not None:
        state["authority"] = authority
    a.add_goal(Goal(
        id="g1", condition="a BUYER-side speaker states the approved budget", say="budget claimed",
        state=state,
    ))
    return a


CONDITION_TEXT = "a BUYER-side speaker states the approved budget"


# --------------------------------------------------------------------------------------------------
# 1 — claim-slot concerns now GOVERN (the Y12 seam, flipped for the claim-slot policy).
# --------------------------------------------------------------------------------------------------


class TestClaimSlotGoesGoverning(unittest.TestCase):
    def test_claim_slot_gets_a_real_fold_not_none(self):
        a = _build()
        goal = a.goals.get("g1")
        v = Verdict(True, 0.9, "The budget is $40K", operation="assert", value="$40K", asserted_by="buyer")
        fold = a._claim_episode_fold(goal, v, 0.0)
        self.assertIsNotNone(fold, "a claim-slot concern must now get a governing fold (Y13)")
        self.assertEqual(fold.match, "new")
        self.assertIn("g1", a.episodes)

    def test_document_mode_fold_is_still_none_for_claim_slot(self):
        # The scope fence: _document_episode_fold (Y12) is UNCHANGED — claim-slot never governs
        # through THAT seam. Y13 adds a second, parallel one instead of widening Y12's.
        a = _build()
        goal = a.goals.get("g1")
        v = Verdict(True, 0.9, "The budget is $40K", operation="assert", value="$40K", asserted_by="buyer")
        self.assertIsNone(a._document_episode_fold(goal, v, 0.0))

    def test_content_key_concern_never_gets_a_claim_fold(self):
        # The mirror-image scope fence: a content-key concern must not pick up governing through the
        # NEW claim seam either.
        a = _build(identity_policy="content_key")
        goal = a.goals.get("g1")
        v = Verdict(True, 0.9, "PO-88", operation="assert", value="PO-88", asserted_by="buyer")
        self.assertIsNone(a._claim_episode_fold(goal, v, 0.0))

    def test_undeclared_identity_policy_never_gets_a_claim_fold(self):
        a = _build(identity_policy=None)
        goal = a.goals.get("g1")
        v = Verdict(True, 0.9, "The budget is $40K", operation="assert", value="$40K", asserted_by="buyer")
        self.assertIsNone(a._claim_episode_fold(goal, v, 0.0))


# --------------------------------------------------------------------------------------------------
# 2 — the mutation corpus, driven LIVE through _claim_episode_fold: only replace mutates; a question/
#     attribution/negation never move the value; a valueless retract clears it; an unauthorized replace
#     is downgraded. Mirrors probe_02_mutation's own fold, but through the real agent seam.
# --------------------------------------------------------------------------------------------------


def _drive_corpus_case(a: Percept, goal, stream_case: dict, expect: dict):
    """Fold every turn of one mutation-corpus case through the LIVE claim seam; return the final
    episode (or None if nothing ever governed) plus the per-turn folds."""
    import re

    money = re.compile(r"\$\d+(?:\.\d+)?[KkMm]?")
    acts = expect.get("acts", [])
    folds = []
    for i, step in enumerate(stream_case["steps"]):
        act = acts[i] if i < len(acts) else "assert"
        op = "mention" if act == "negation" else act
        m = money.search(step.get("text", ""))
        value = m.group(0) if m else None
        v = Verdict(True, 0.9, step.get("text", ""), operation=op, value=value, asserted_by=step.get("speaker"))
        folds.append(a._claim_episode_fold(goal, v, float(step["t"])))
    return a.episode_current(goal.id), folds


class TestMutationCorpusResolvesLiveExactlyAsFixtured(unittest.TestCase):
    def _case(self, case_id: str) -> tuple[dict, dict]:
        stream_doc, labels_doc = _mutation.build()
        stream_case = next(c for c in stream_doc["cases"] if c["id"] == case_id)
        expect = next(c["expect"] for c in labels_doc["cases"] if c["id"] == case_id)
        return stream_case, expect

    def test_only_the_explicit_replace_moves_the_value(self):
        stream_case, expect = self._case("only_one_of_four_25k_utterances_replaces")
        a = _build()
        goal = a.goals.get("g1")
        episode, folds = _drive_corpus_case(a, goal, stream_case, expect)
        self.assertEqual(episode.value, expect["final_value"])
        # Turn 0 opens the episode (a birth -> mutation='none'); only the LAST turn (the replace) is a
        # real state change — the three value-CONTAINING mentions/negation between never move anything.
        self.assertEqual([f.mutation for f in folds], ["none", "confirmed", "confirmed", "confirmed", "updated"])
        kinds = [r.kind for r in episode.revisions]
        self.assertEqual(kinds, ["opened", "updated"], "exactly one extra row for the one real mutation")
        self.assertEqual(episode.revisions[-1].value, "$25K")

    def test_valueless_retraction_clears_the_value_with_a_chain_entry(self):
        stream_case, expect = self._case("valueless_retraction_clears_the_value")
        a = _build()
        goal = a.goals.get("g1")
        episode, folds = _drive_corpus_case(a, goal, stream_case, expect)
        self.assertIsNone(episode.value, "a retraction clears to None, never ''")
        self.assertEqual(episode.value, expect["final_value"])
        self.assertEqual([f.mutation for f in folds], ["none", "retracted"])
        self.assertEqual([r.kind for r in episode.revisions], ["opened", "retracted"])

    def test_seller_lacks_authority_the_value_stands(self):
        stream_case, expect = self._case("seller_lacks_authority_to_replace_buyer_value")
        a = _build(authority=[expect["authority_required"]])  # ["buyer"]
        goal = a.goals.get("g1")
        episode, folds = _drive_corpus_case(a, goal, stream_case, expect)
        self.assertEqual(episode.value, expect["final_value"])  # "$40K" — unauthorized replace refused
        # Downgraded to a mention -> folds as a plain confirmed re-statement, never an 'updated' row.
        self.assertEqual([f.mutation for f in folds], ["none", "confirmed"])
        self.assertEqual([r.kind for r in episode.revisions], ["opened"])

    def test_probe_02_shadow_prediction_is_clean_and_matches_the_live_run(self):
        # Y11's shadow report predicted this corpus would fold cleanly; confirm that prediction is
        # STILL accurate now that claim-slot concerns govern live (Y13 changed nothing about the pure
        # claims.fold_verdicts machinery the shadow probe itself exercises).
        report = run_probe_02_mutation()
        self.assertEqual(report.disagreements, ())
        self.assertEqual(report.match_rate, 1.0)
        self.assertTrue(report.acceptance.accepted)


# --------------------------------------------------------------------------------------------------
# 3 — authority is enforced AT FOLD TIME, before the episode/gate ever sees a mutating op.
# --------------------------------------------------------------------------------------------------


class TestAuthorityEnforcedAtFoldTime(unittest.TestCase):
    def test_unauthorized_replace_never_rearms_or_supersedes(self):
        a = _build(authority=["buyer"])
        goal = a.goals.get("g1")
        v0 = Verdict(True, 0.9, "buyer states 40k", operation="assert", value="$40K", asserted_by="buyer")
        fold0 = a._claim_episode_fold(goal, v0, 0.0)
        self.assertTrue(fold0.should_rearm)  # a genuine new episode
        v1 = Verdict(True, 0.9, "seller replace", operation="replace", value="$25K", asserted_by="seller")
        fold1 = a._claim_episode_fold(goal, v1, 20.0)
        self.assertEqual(fold1.mutation, "confirmed", "downgraded replace folds as a non-mutating mention")
        self.assertFalse(fold1.should_rearm)
        self.assertIsNone(fold1.supersedes)

    def test_authorized_asserter_may_replace(self):
        a = _build(authority=["buyer"])
        goal = a.goals.get("g1")
        v0 = Verdict(True, 0.9, "buyer states 40k", operation="assert", value="$40K", asserted_by="buyer")
        a._claim_episode_fold(goal, v0, 0.0)
        v1 = Verdict(True, 0.9, "buyer corrects", operation="replace", value="$25K", asserted_by="buyer")
        fold1 = a._claim_episode_fold(goal, v1, 20.0)
        self.assertEqual(fold1.mutation, "updated")
        self.assertEqual(a.episode_current("g1").value, "$25K")

    def test_no_authority_declared_is_unrestricted_byte_for_byte(self):
        a = _build(authority=None)
        goal = a.goals.get("g1")
        self.assertIsNone(concern_authority_of(goal))
        v0 = Verdict(True, 0.9, "buyer states 40k", operation="assert", value="$40K", asserted_by="buyer")
        a._claim_episode_fold(goal, v0, 0.0)
        v1 = Verdict(True, 0.9, "seller replace, no authority declared", operation="replace",
                      value="$25K", asserted_by="seller")
        fold1 = a._claim_episode_fold(goal, v1, 20.0)
        self.assertEqual(fold1.mutation, "updated", "an undeclared authority policy is unrestricted")
        self.assertEqual(a.episode_current("g1").value, "$25K")

    def test_authority_downgrade_is_traced_not_silent(self):
        # claims.apply_operation logs claim.authority_downgrade; confirm the underlying ClaimState
        # this fold delegates to reports the downgrade explicitly (not merely "nothing changed").
        state = ClaimState(claim_id="g1")
        outcome = state.apply(
            "replace", value="$25K", asserted_by="seller",
            authority=ConcernAuthority(frozenset({"buyer"})), trace=False,
        )
        self.assertTrue(outcome.downgraded)
        self.assertEqual(outcome.effective, "mention")
        self.assertFalse(outcome.applied)


# --------------------------------------------------------------------------------------------------
# 4 — supersession + provisional-incident delivery (Y10) reaches the claim-slot fire path, and
#     episode_current surfaces the corrected value WITH its revision chain.
# --------------------------------------------------------------------------------------------------


class TestSupersessionAndRevisionChainGoLive(unittest.TestCase):
    def test_provisional_fire_then_a_linked_superseding_correction(self):
        a = _build()
        a.vision = FakeVision(script={
            "f0": Verdict(True, 0.9, "The budget is $40K", operation="assert", value="$40K", asserted_by="buyer"),
            # A 25s-later correction — inside the 21-34s window Y10's own docstring calls out.
            "f1": Verdict(True, 0.9, "Actually the approved budget is $25K", operation="replace",
                          value="$25K", asserted_by="buyer"),
        })
        e0 = asyncio.run(a.perceive_judged("f0", t=0.0))
        e1 = asyncio.run(a.perceive_judged("f1", t=25.0))
        self.assertEqual([e.action for e in e0], ["fire"])
        self.assertEqual([e.action for e in e1], ["fire"])
        self.assertEqual(a.metrics.get("gate.superseding_revision"), 1, "the correction is never silent")
        episode = a.episode_current("g1")
        self.assertEqual(episode.value, "$25K")
        self.assertEqual([r.kind for r in episode.revisions], ["opened", "updated"])

    def test_retraction_also_fires_a_linked_superseding_revision(self):
        a = _build()
        a.vision = FakeVision(script={
            "f0": Verdict(True, 0.9, "The budget is $40K", operation="assert", value="$40K", asserted_by="buyer"),
            "f1": Verdict(True, 0.9, "Scratch that", operation="retract", asserted_by="buyer"),
        })
        asyncio.run(a.perceive_judged("f0", t=0.0))
        asyncio.run(a.perceive_judged("f1", t=25.0))
        self.assertEqual(a.metrics.get("gate.superseding_revision"), 1)
        episode = a.episode_current("g1")
        self.assertIsNone(episode.value)
        self.assertEqual([r.kind for r in episode.revisions], ["opened", "retracted"])

    def test_episode_current_is_none_before_anything_is_observed(self):
        a = _build()
        self.assertIsNone(a.episode_current("g1"))
        self.assertIsNone(a.episode_current("does-not-exist"))

    def test_unchanged_confirmation_is_not_a_superseding_revision(self):
        a = _build()
        a.vision = FakeVision(script={
            "f0": Verdict(True, 0.9, "The budget is $40K", operation="assert", value="$40K", asserted_by="buyer"),
            "f1": Verdict(True, 0.9, "Yes, $40K as discussed", operation="confirm", value="$40K", asserted_by="buyer"),
        })
        e0 = asyncio.run(a.perceive_judged("f0", t=0.0))
        e1 = asyncio.run(a.perceive_judged("f1", t=10.0))
        self.assertEqual([e.action for e in e0], ["fire"])
        self.assertEqual(e1, [], "an unchanged confirmation is idempotently suppressed, not a re-fire")
        self.assertEqual(a.metrics.get("gate.superseding_revision"), 0)
        self.assertEqual([r.kind for r in a.episode_current("g1").revisions], ["opened"])


# --------------------------------------------------------------------------------------------------
# 5 — ZERO fabricated retractions on a 32s digression control: unrelated turns must never move the
#     value or append a spurious revision, however many of them stream through.
# --------------------------------------------------------------------------------------------------


class TestDigressionControlNeverFabricatesARetraction(unittest.TestCase):
    def test_32s_of_unrelated_turns_leaves_the_claim_untouched(self):
        a = _build()
        goal = a.goals.get("g1")
        v0 = Verdict(True, 0.9, "The budget is $40K", operation="assert", value="$40K", asserted_by="buyer")
        fold0 = a._claim_episode_fold(goal, v0, 0.0)
        self.assertEqual(fold0.match, "new")

        # 8 unrelated turns over the next 32 seconds — small talk, scheduling, a different topic
        # entirely. None of them assert/replace/retract the budget; the judge would emit these as
        # non-mutating (mention/confirm) precisely because nothing about the claim was said.
        digression_texts = [
            "Let's push the follow-up to Thursday.",
            "Can you loop in the ops team?",
            "The weather's been rough this week.",
            "I'll send the deck by end of day.",
            "Who's presenting the roadmap slide?",
            "Let's take a five minute break.",
            "Reminder: badge access expires Friday.",
            "Sounds good, talk soon.",
        ]
        folds = []
        for i, text in enumerate(digression_texts):
            t = 4.0 * (i + 1)  # spaced across ~32s
            v = Verdict(True, 0.5, text, operation="mention", asserted_by="buyer")
            folds.append(a._claim_episode_fold(goal, v, t))
        self.assertTrue(all(f.mutation == "confirmed" for f in folds))
        self.assertTrue(all(not f.should_rearm for f in folds))
        self.assertTrue(all(f.supersedes is None for f in folds))

        episode = a.episode_current("g1")
        self.assertEqual(episode.value, "$40K", "a digression must never move the claimed value")
        self.assertEqual(
            [r.kind for r in episode.revisions], ["opened"],
            "zero fabricated retractions/updates: the chain gains no extra row for a digression",
        )

    def test_a_default_operation_digression_also_never_retracts(self):
        # Even a turn that authors NO operation at all (Verdict's own default, 'assert') must not
        # fabricate a retraction: assert only ESTABLISHES an empty slot, never overwrites/clears one.
        a = _build()
        goal = a.goals.get("g1")
        a._claim_episode_fold(
            goal, Verdict(True, 0.9, "The budget is $40K", operation="assert", value="$40K", asserted_by="buyer"),
            0.0,
        )
        for i in range(6):
            v = Verdict(True, 0.4, "unrelated small talk", asserted_by="buyer")  # operation defaults to 'assert'
            a._claim_episode_fold(goal, v, 4.0 * (i + 1))
        episode = a.episode_current("g1")
        self.assertEqual(episode.value, "$40K")
        self.assertEqual([r.kind for r in episode.revisions], ["opened"])


if __name__ == "__main__":
    unittest.main()
