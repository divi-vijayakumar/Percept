"""— DOCUMENT MODE rollout: episode grouping + EpisodeFoldResult-owned idempotency go
GOVERNING, scoped to screen_document / content-key concerns ONLY.

Y11 (shadow mode, landed) proved the mechanism — subject resolution + the episode fold — would have
been RIGHT on synthetic ground truth, but changed nothing: ``_shadow_observe`` always folds
``decision=None`` (unbound) and the answer only lands in the ``shadow.fold`` trace stage. This suite flips
the switch for exactly one scope: a concern that SELECTS the content-key resolver
(``goal.state['identity_policy'] == 'content_key'``) now gets a REAL per-goal
:class:`~percept.episodes.EpisodeLog` threaded into the firing path via ``_fire_concern(fold=...)``
(``Percept._document_episode_fold``, ``percept/agent.py``). Every other concern — including a
claim-slot / conversational (mutation) goal — is untouched: it still gets ``fold=None`` and stays
shadow-only.

The four acceptance properties below:

1. Probe 01's clean walkthrough (the ``subject_identity`` + ``continuation`` synthetic corpora,
   ``percept.eval.shadow_report``) produces ZERO fabricated divergences — including from an
   out-of-order arrival, which the live governing path must refuse to let corrupt an episode's
   ``last_seen`` (``TestOutOfOrderGuard``).
2. Every fire carries a stable content key or an explicit ``unknown`` — never a fabricated one
   (``TestNeverFabricatesAContentKey``).
3. The shadow measurement PREDICTED this outcome: the shadow report agrees with the live governing
   run on the same fixture (``TestShadowAgreesWithGoverning``).
4. Claim-slot concerns stay in shadow until claim governance is enabled — verified still advisory
   (``TestClaimSlotStaysShadow``).
"""

from __future__ import annotations

import asyncio
import unittest

from percept.adapters import Goal, Verdict
from percept.agent import Percept
from percept.backends.fake import FakeVision
from percept.behavior import ShadowObservation
from percept.config import GateSettings, Settings
from percept.contracts import SubjectRef
from percept.eval.shadow_report import run_probe_01_identity_continuation
from percept.gate import Gate
from percept.replay import gate_config_of
from percept.tracing import STAGE_SHADOW_FOLD, RunTraceManifest, TracePolicy, TraceRecorder


def _content_ref(key: "str | None", *, resolution: str = "bound", confidence: float = 0.92) -> SubjectRef:
    return SubjectRef(
        scheme="content", namespace="workflow", key=key, resolution=resolution,
        scope="durable", strategy="test", confidence=confidence,
    )


def _build(*, identity_policy: "str | None" = "content_key", threshold: float = 0.7) -> Percept:
    settings = Settings(gate=GateSettings(threshold=threshold, validity_window_s=1000.0))
    a = Percept.create(vision=FakeVision(), settings=settings, discover_plugins=False)
    state = {"type": "watch", "count": 0}
    if identity_policy is not None:
        state["identity_policy"] = identity_policy
    a.add_goal(Goal(id="g1", condition="a purchase order is on screen", say="PO seen", state=state))
    return a


def _recorder() -> TraceRecorder:
    manifest = RunTraceManifest.create(
        run_id="run-y12", source="camera", policy=TracePolicy(level="full"),
        gate_config=gate_config_of(Gate()),
    )
    return TraceRecorder(manifest)


# --------------------------------------------------------------------------------------------------
# 1 + 3 — the live GOVERNING run reproduces exactly what the shadow report predicted, on the SAME
# subject shape the ``continuation`` corpus's ``interleaved_a_b_a_continues_a`` case pins: a subject
# (A) fires, a DIFFERENT subject (B) fires next, and A's return is a SILENT continuation — never a
# spurious third fire. A screen/document concern's watch tick has no per-entity focus, so every
# observation shares ONE gate key ((goal_id, None)): this is exactly the shared-key collision that
# episode-owned idempotency fixes, and the control test below shows the legacy (fold=None)
# path actually losing B's fire to it.
# --------------------------------------------------------------------------------------------------


class TestGoverningFixesTheSharedKeyCollision(unittest.TestCase):
    def _script(self):
        return {
            "a0": Verdict(True, 0.92, "PO-88 visible", subject=_content_ref("PO-88")),
            "b5": Verdict(True, 0.92, "PO-91 visible", subject=_content_ref("PO-91")),
            "a10": Verdict(True, 0.92, "PO-88 visible again", subject=_content_ref("PO-88")),
        }

    def test_content_key_concern_fires_for_every_distinct_subject(self):
        a = _build(identity_policy="content_key")
        a.vision = FakeVision(script=self._script())
        e0 = asyncio.run(a.perceive_judged("a0", t=0.0))
        e1 = asyncio.run(a.perceive_judged("b5", t=5.0))
        e2 = asyncio.run(a.perceive_judged("a10", t=10.0))
        self.assertEqual([len(e0), len(e1), len(e2)], [1, 1, 0], "expected fire, fire, silent")
        self.assertEqual(e0[0].action, "fire")
        self.assertEqual(e1[0].action, "fire")

    def test_non_content_key_concern_loses_the_second_subjects_fire(self):
        # THE CONTROL: without a declared content-key policy, _document_episode_fold returns None
        # (the byte-for-byte legacy path) and both subjects share the SAME disarmed gate key -- B's fire is
        # silently lost. This is what content-key governance fixes and leaves UNCHANGED for
        # every other concern (the scope fence is real, not just documented).
        a = _build(identity_policy=None)
        a.vision = FakeVision(script=self._script())
        e0 = asyncio.run(a.perceive_judged("a0", t=0.0))
        e1 = asyncio.run(a.perceive_judged("b5", t=5.0))
        self.assertEqual(len(e0), 1)
        self.assertEqual(len(e1), 0, "legacy gate-only path collides both subjects onto one key")


# --------------------------------------------------------------------------------------------------
# 2 — never a fabricated content key.
# --------------------------------------------------------------------------------------------------


class TestNeverFabricatesAContentKey(unittest.TestCase):
    def test_unknown_subject_folds_unknown_never_a_synthesized_key(self):
        a = _build(identity_policy="content_key")
        verdict = Verdict(True, 0.4, "a document, key unreadable", subject=_content_ref(None, resolution="unknown"))
        fold = a._document_episode_fold(a.goals.get("g1"), verdict, 0.0)
        self.assertIsNotNone(fold)
        self.assertEqual(fold.match, "unknown")
        self.assertFalse(fold.should_rearm, "a visible abstention must never rearm the gate")
        # The episode the log opened for this sighting carries a provisional key derived
        # from its own id (UN_<episode_id>) -- never an invented content key.
        log = a.episodes["g1"]
        self.assertEqual(log.count(), 1)
        ep = log._episodes[0]
        self.assertEqual(ep.subject_key, f"UN_{ep.id}")

    def test_bound_subject_key_is_exactly_the_judge_emitted_key(self):
        a = _build(identity_policy="content_key")
        verdict = Verdict(True, 0.9, "an invoice", subject=_content_ref("INV-2291"))
        fold = a._document_episode_fold(a.goals.get("g1"), verdict, 0.0)
        self.assertEqual(fold.match, "new")
        self.assertEqual(a.episodes["g1"]._episodes[0].subject_key, "INV-2291")

    def test_no_subject_at_all_is_never_grouped(self):
        # Absence of a subject is NOT a fabricated 'unknown' -- it degrades to "nothing to fold".
        a = _build(identity_policy="content_key")
        verdict = Verdict(True, 0.9, "something on screen")  # subject defaults to None
        fold = a._document_episode_fold(a.goals.get("g1"), verdict, 0.0)
        self.assertIsNone(fold)
        self.assertNotIn("g1", a.episodes)


# --------------------------------------------------------------------------------------------------
# 1 (out-of-order half) — a stale/out-of-order observation must never corrupt the live log.
# --------------------------------------------------------------------------------------------------


class TestOutOfOrderGuard(unittest.TestCase):
    def test_stale_observation_is_skipped_never_rewinds_last_seen(self):
        a = _build(identity_policy="content_key")
        goal = a.goals.get("g1")
        v_a = Verdict(True, 0.9, "PO-88", subject=_content_ref("PO-88"))
        fold0 = a._document_episode_fold(goal, v_a, 10.0)
        self.assertIsNotNone(fold0)
        episode = a.episodes["g1"]._episodes[0]
        self.assertEqual(episode.last_seen, 10.0)

        # A late-resolving, out-of-order completion for an EARLIER t arrives after the t=10 one.
        stale = a._document_episode_fold(goal, v_a, 2.0)
        self.assertIsNone(stale, "an out-of-order observation must degrade to gate-only, not govern")
        self.assertEqual(
            episode.last_seen, 10.0,
            "a stale observation must never rewind an episode's last_seen (fabricates later divergences)",
        )

        # A genuinely later, in-order observation still folds correctly against the UNCORRUPTED state.
        fold2 = a._document_episode_fold(goal, v_a, 15.0)
        self.assertEqual(fold2.match, "continuation")
        self.assertEqual(episode.last_seen, 15.0)

    def test_out_of_order_is_counted_not_silently_swallowed(self):
        a = _build(identity_policy="content_key")
        goal = a.goals.get("g1")
        v_a = Verdict(True, 0.9, "PO-88", subject=_content_ref("PO-88"))
        a._document_episode_fold(goal, v_a, 10.0)
        before = a.metrics.get("episode.document.out_of_order_skipped")
        a._document_episode_fold(goal, v_a, 1.0)
        after = a.metrics.get("episode.document.out_of_order_skipped")
        self.assertEqual(after, before + 1)


# --------------------------------------------------------------------------------------------------
# 3 — the shadow measurement predicted a clean run: assert the shadow report AGREES with the
# governing behavior exercised above, on the same fixture shape (probe 01's corpora).
# --------------------------------------------------------------------------------------------------


class TestShadowAgreesWithGoverning(unittest.TestCase):
    def test_probe_01_clean_walkthrough_has_zero_disagreements(self):
        report = run_probe_01_identity_continuation()
        self.assertEqual(
            len(report.disagreements), 0,
            f"probe 01 predicted a clean walkthrough; got disagreements: {report.disagreements}",
        )
        self.assertEqual(report.metrics.false_bound_rate, 0.0)
        self.assertTrue(report.acceptance.accepted)
        self.assertEqual(report.match_rate, 1.0)

    def test_interleaved_case_shadow_matches_the_live_governing_fire_sequence(self):
        # The corpus case this mirrors: continuation/interleaved_a_b_a_continues_a expects
        # actions=["fire", "fire", "silent"], fire_count=2, spurious_third_fire_forbidden=True.
        # TestGoverningFixesTheSharedKeyCollision.test_content_key_concern_fires_for_every_distinct_subject
        # reproduces exactly that sequence through the LIVE watch path; here we confirm the shadow
        # report never flagged this shape as a disagreement (it predicted zero, and the live run above
        # delivers zero).
        report = run_probe_01_identity_continuation()
        case_disagreements = [d for d in report.disagreements if d.case_id == "interleaved_a_b_a_continues_a"]
        self.assertEqual(case_disagreements, [])


# --------------------------------------------------------------------------------------------------
# 4 — claim-slot concerns stay in shadow until claim governance is enabled.
# --------------------------------------------------------------------------------------------------


class TestClaimSlotStaysShadow(unittest.TestCase):
    def test_claim_slot_identity_policy_never_governs(self):
        a = _build(identity_policy="claim_slot")  # a claim-slot / conversational selection, NOT content_key
        goal = a.goals.get("g1")
        verdict = Verdict(True, 0.9, "asserted", subject=_content_ref("slot-1"))
        fold = a._document_episode_fold(goal, verdict, 0.0)
        self.assertIsNone(fold, "a claim-slot concern must never get a document-governing fold")
        self.assertNotIn("g1", a.episodes)

    def test_undeclared_identity_policy_never_governs(self):
        a = _build(identity_policy=None)
        goal = a.goals.get("g1")
        verdict = Verdict(True, 0.9, "asserted", subject=_content_ref("slot-1"))
        fold = a._document_episode_fold(goal, verdict, 0.0)
        self.assertIsNone(fold)

    def test_claim_slot_concern_still_emits_the_advisory_shadow_trace(self):
        # Governance is off for this concern, but the shadow.fold advisory must still be observing
        # (untouched) -- proving Y12 narrowed WHO governs without narrowing WHO is shadow-observed.
        a = _build(identity_policy="claim_slot")
        a.vision = FakeVision(script={
            "f1": Verdict(True, 0.9, "asserted", subject=_content_ref("slot-1")),
        })
        rec = _recorder()
        a.on_trace = rec.tracer()
        events = asyncio.run(a.perceive_judged("f1", t=1.0))
        self.assertTrue(events, "the tick should still fire — the gate is mechanically unchanged")
        shadow_recs = [r for r in rec.records if r.stage == STAGE_SHADOW_FOLD]
        self.assertEqual(len(shadow_recs), 1)
        back = ShadowObservation.from_advisory(shadow_recs[0].data)
        self.assertEqual(back.subject.key, "slot-1")
        # Shadow's OWN fold is still the "unbound" advisory (decision=None) -- it does not know
        # about the real governing log at all, exactly as designed.
        self.assertEqual(back.fold.match, "unbound")


if __name__ == "__main__":
    unittest.main()
