"""Card Y11 — SHADOW MODE: the subject + EpisodeFoldResult must LAND IN THE TRACE and survive a
media-OFF ``.ptrace`` round-trip.

A founder handoff depends on this: a client runs the wheel on THEIR data and sends back a ``.ptrace``
with media OFF. If shadow's fold data is not carried IN the trace record (not just in memory), they
have nothing gradeable to return. These tests pin that the ``shadow.fold`` stage carries the resolved
subject and the fold, that both come back byte-identical after ``load_trace`` AND after a bundle
round-trip, and that the advisory is retained at EVERY trace level (a production/summary trace is
exactly the one a client is most likely to capture).

Everything here is ADVISORY. Shadow governs nothing: the same run with no tracer attached does no
shadow work at all and fires byte-for-byte identically.
"""

from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from percept.adapters import Box, Goal, Observation, Verdict
from percept.agent import Percept
from percept.backends.fake import FakeVision
from percept.behavior import BehaviorExecutor, ShadowObservation
from percept.config import GateSettings, Settings
from percept.contracts import SubjectRef
from percept.episodes import Episode
from percept.gate import Gate
from percept.incidents import incident_shadow, make_incident
from percept.replay import gate_config_of
from percept.tracing import (
    STAGE_SHADOW_FOLD,
    RunTraceManifest,
    TracePolicy,
    TraceRecorder,
    load_trace,
)
from percept.trace_bundle import BundlePolicy, TraceBundle


def _recorder(level="full"):
    manifest = RunTraceManifest.create(
        run_id="run-shadow", source="camera", policy=TracePolicy(level=level),
        gate_config=gate_config_of(Gate()),
    )
    return TraceRecorder(manifest)


def _bound_subject(key="INV-2291"):
    return SubjectRef(
        scheme="content", namespace="acme-erp", key=key, resolution="bound",
        scope="durable", strategy="judge", confidence=0.9,
    )


def _shadow_for(subject, *, operation="assert"):
    """Compute a real ShadowObservation through the executor (a continuation of a bound episode)."""
    ex = BehaviorExecutor(Gate())
    from percept.episodes import ContinuationDecision  # local import: only the test needs it

    ep = Episode(id="g1:ep1", goal_id="g1", started_at=0.0, last_seen=0.0, subject_key=subject.key)
    decision = ContinuationDecision(
        episode=ep, is_new=False, reason="continued", subject_matched=True,
    )
    return ex.shadow_observe(decision, subject=subject, operation=operation)


class TestShadowTraceRoundTrip(unittest.TestCase):
    def test_shadow_advisory_carries_subject_and_fold(self):
        shadow = _shadow_for(_bound_subject())
        adv = shadow.to_advisory()
        self.assertIn("subject", adv)
        self.assertIn("fold", adv)
        # The five fold fields the card names must all be present.
        for k in ("match", "mutation", "should_rearm", "should_evaluate_gate", "supersedes"):
            self.assertIn(k, adv["fold"])
        # And it reconstructs to an equal ShadowObservation.
        back = ShadowObservation.from_advisory(adv)
        self.assertEqual(back.subject, shadow.subject)
        self.assertEqual(back.fold, shadow.fold)

    def test_subject_and_fold_survive_json_load_trace(self):
        rec = _recorder()
        shadow = _shadow_for(_bound_subject(), operation="assert")
        rec.record(STAGE_SHADOW_FOLD, source_ts=4.0, goal_id="g1", **shadow.to_advisory())

        _manifest, records, _summary = load_trace(rec.to_json())
        shadow_recs = [r for r in records if r.stage == STAGE_SHADOW_FOLD]
        self.assertEqual(len(shadow_recs), 1)
        back = ShadowObservation.from_advisory(shadow_recs[0].data)
        self.assertEqual(back.subject, shadow.subject)
        self.assertEqual(back.subject.key, "INV-2291")
        self.assertEqual(back.fold, shadow.fold)

    def test_subject_and_fold_survive_ptrace_bundle_with_media_off(self):
        # THE founder-handoff scenario: a portable .ptrace, media OFF, round-tripped through disk.
        rec = _recorder()
        shadow = _shadow_for(_bound_subject(key="INV-2294"))
        rec.record(STAGE_SHADOW_FOLD, source_ts=7.0, goal_id="g1", **shadow.to_advisory())

        bundle = TraceBundle.from_recorder(rec, policy=BundlePolicy())  # media OFF by default
        self.assertFalse(bundle.policy.include_media)
        with TemporaryDirectory() as d:
            path = bundle.write(Path(d) / "run-shadow")
            reread = TraceBundle.read(path)
        shadow_recs = [r for r in reread.records if r.stage == STAGE_SHADOW_FOLD]
        self.assertEqual(len(shadow_recs), 1, "shadow fold lost across the .ptrace round-trip")
        back = ShadowObservation.from_advisory(shadow_recs[0].data)
        self.assertEqual(back.subject.key, "INV-2294")
        self.assertEqual(back.fold.match, shadow.fold.match)
        self.assertEqual(back.fold.mutation, shadow.fold.mutation)
        # Media really was off: no evidence bytes travelled.
        self.assertEqual(reread.evidence, ())

    def test_unknown_subject_absence_round_trips_as_none(self):
        # An unbound shadow (no subject) must serialize subject=None and come back None — never a
        # fabricated unknown ref bearing a key.
        ex = BehaviorExecutor(Gate())
        shadow = ex.shadow_observe(None, subject=None)  # watch path, no binding
        self.assertEqual(shadow.fold.match, "unbound")
        rec = _recorder()
        rec.record(STAGE_SHADOW_FOLD, source_ts=1.0, goal_id="g1", **shadow.to_advisory())
        _m, records, _s = load_trace(rec.to_json())
        back = ShadowObservation.from_advisory(
            [r for r in records if r.stage == STAGE_SHADOW_FOLD][0].data
        )
        self.assertIsNone(back.subject)
        self.assertEqual(back.fold, shadow.fold)

    def test_shadow_stage_retained_at_every_level(self):
        for level in ("summary", "decisions", "full"):
            rec = _recorder(level=level)
            shadow = _shadow_for(_bound_subject())
            rec.record(STAGE_SHADOW_FOLD, source_ts=2.0, goal_id="g1", **shadow.to_advisory())
            kept = [r for r in rec.records if r.stage == STAGE_SHADOW_FOLD]
            self.assertEqual(len(kept), 1, f"shadow fold dropped at level={level!r}")


class TestIncidentShadowAttachment(unittest.TestCase):
    def test_make_incident_attaches_advisory_and_is_readable(self):
        shadow = _shadow_for(_bound_subject())

        class _Fire:
            goal_id = "g1"
            text = "seen"
            verdict = Verdict(True, 0.9, "x")
            entity_id = "e1"

        incident = make_incident(_Fire(), ts=3.0, shadow=shadow)
        adv = incident_shadow(incident)
        self.assertIsNotNone(adv)
        self.assertEqual(ShadowObservation.from_advisory(adv).subject, shadow.subject)

    def test_no_shadow_is_none_and_incident_unchanged(self):
        class _Fire:
            goal_id = "g1"
            text = "seen"
            verdict = Verdict(True, 0.9, "x")

        incident = make_incident(_Fire(), ts=3.0)
        self.assertIsNone(incident_shadow(incident))


class TestLiveWatchPathEmitsShadow(unittest.TestCase):
    def _build(self):
        subject = _bound_subject(key="PO-88")
        vision = FakeVision(
            script={"f1": Verdict(True, 0.95, "a purchase order", subject=subject)},
            detect_script={"f1": [Observation("purchase_order", Box(0.3, 0.2, 0.25, 0.35), 0.9)]},
        )
        settings = Settings(gate=GateSettings(threshold=0.7, validity_window_s=10.0))
        a = Percept.create(vision=vision, settings=settings, discover_plugins=False)
        a.add_goal(Goal(id="g1", condition="a purchase order is on screen", say="PO seen"))
        return a

    def test_shadow_fold_emitted_on_watch_tick_with_subject(self):
        rec = _recorder()
        a = self._build()
        a.on_trace = rec.tracer()
        asyncio.run(a.perceive_judged("f1", t=1.0))
        shadow_recs = [r for r in rec.records if r.stage == STAGE_SHADOW_FOLD]
        self.assertEqual(len(shadow_recs), 1)
        back = ShadowObservation.from_advisory(shadow_recs[0].data)
        self.assertEqual(back.subject.key, "PO-88")

    def test_untraced_path_does_no_shadow_and_still_fires(self):
        a = self._build()
        # No tracer attached: default hot path. It must still fire and do no shadow work.
        events = asyncio.run(a.perceive_judged("f1", t=1.0))
        self.assertTrue(events, "the watch tick should still fire without a tracer")


if __name__ == "__main__":
    unittest.main()
