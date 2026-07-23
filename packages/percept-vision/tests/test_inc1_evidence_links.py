"""Evidence-link correlation (contract 1.9): every delivered Incident resolves to EXACTLY ONE
judgment record, including the A->B transition path that historically emitted no judgment at all.

The nine checks the client audit gate needs:
  1. judgment_event_id injectivity (incl. '#'/'%' in components + None entity)
  2. THE CLIENT P0 REPRO — two entities, one goal, one trace, both fire, distinct ids, each Incident
     resolves to exactly one judgment.event row
  3. transition fixture e2e — non-null evidence_seq, evidence_origin="transition", recorder-stamped
     data.evidence_seq == source.frame.seq (same trace_id), real timing stamps present
  4. B-without-A never fires and transition.decision reports idle/armed truthfully
  5. judgment.build_failed on a transition still delivers the fire (links None — a broken stamp never
     changes what fires)
  6. a pre-1.9 trace replays with a byte-identical decision digest (the 1.9 stages are inert to replay)
  7. verify_evidence_links: zero-match, multi-match, legacy counting
  8. JS golden-vector parity (the shared vectors the vitest twin also asserts)
  9. existing fire_transition positional-call sites are unchanged (see test_program_equivalence.py;
     re-asserted here for locality)
"""
from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path

from percept.adapters import Box, Goal, Observation, Verdict
from percept.agent import Percept
from percept.behavior import BehaviorExecutor
from percept.config import GateSettings, Settings
from percept.contracts import JudgmentTimingBuilder, judgment_event_id
from percept.gate import FIRE, Gate
from percept.incidents import make_incident
from percept.replay import (
    EvidenceLinkReport,
    gate_config_of,
    replay_cognition,
    verify_evidence_links,
)
from percept.tracing import (
    STAGE_INCIDENT_CREATED,
    STAGE_JUDGMENT_EVENT,
    STAGE_SOURCE_FRAME,
    STAGE_TRANSITION_DECISION,
    RunTraceManifest,
    TracePolicy,
    TraceRecord,
    TraceRecorder,
)

YES = Verdict(satisfied=True, confidence=0.9, evidence="yes")
NO = Verdict(satisfied=False, confidence=0.9, evidence="no")

# The shared golden-vector fixture — Python is source of truth; the vitest twin asserts the same file.
_GOLDEN = (
    Path(__file__).resolve().parents[2]
    / "percept-edge" / "test" / "fixtures" / "judgment_event_id.golden.json"
)


class _CondVision:
    """A fake that routes on the CONDITION text (not just the frame token), so a transition goal's
    ``from``/``to`` conditions can resolve to DIFFERENT verdicts on the same frame. Scripted wiring,
    never intelligence."""

    def __init__(self, script: dict, detect: dict | None = None) -> None:
        self._script = script            # frame_token -> {condition_text: Verdict}
        self._detect = detect or {}      # frame_token -> [Observation]

    async def judge(self, condition, frame, **kw):
        return self._script[str(frame)][condition]

    async def detect(self, frame, goals):
        return list(self._detect.get(str(frame), []))


def _recorder(level="full", source="camera"):
    manifest = RunTraceManifest.create(
        run_id="run-inc1", source=source, policy=TracePolicy(level=level),
        gate_config=gate_config_of(Gate()),
    )
    return TraceRecorder(manifest)


def _emit_incident_created(rec, fire, ts, *, source="camera"):
    """Replicate runtime/worker.py's incident.created emission for a fire (make_incident + the trace
    row carrying the 1.9 evidence link). Returns the built Incident."""
    inc = make_incident(fire, ts=ts)
    rec.record(
        STAGE_INCIDENT_CREATED, trace_id=f"{source}@{ts}", source_ts=ts,
        goal_id=inc.concern_id, incident_id=inc.id, entity_id=inc.entity_id,
        judgment_event_id=inc.judgment_event_id, evidence_origin=inc.evidence_origin,
    )
    return inc


# ---------------------------------------------------------------------------
# 1. injectivity
# ---------------------------------------------------------------------------
class TestJudgmentEventIdInjective(unittest.TestCase):
    def test_none_entity_equals_empty(self):
        self.assertEqual(judgment_event_id("g", None, 3), judgment_event_id("g", "", 3))
        self.assertEqual(judgment_event_id("g", None, 3), "g##3")

    def test_hash_and_percent_in_components_stay_injective(self):
        # A '#' or '%' inside a component must never let two DISTINCT triples collide.
        triples = [
            ("a#b", "c", 1), ("a", "#b", 1), ("a#", "b", 1), ("a", "b#", 1),
            ("a%b", "c", 1), ("pct%25", "hash%23", 5), ("100%", "#1", 0),
            ("g#e", "", 9), ("g", "e", 9), ("%", "#", 1), ("", "", 0),
        ]
        ids = [judgment_event_id(*t) for t in triples]
        self.assertEqual(len(ids), len(set(ids)), f"collision among {ids}")

    def test_opaque_stable_form(self):
        self.assertEqual(judgment_event_id("a#b", "c", 1), "a%23b#c#1")
        self.assertEqual(judgment_event_id("a", "#b", 1), "a#%23b#1")
        self.assertEqual(judgment_event_id("a%b", "c%d", 2), "a%25b#c%25d#2")


# ---------------------------------------------------------------------------
# 2. THE CLIENT P0 REPRO
# ---------------------------------------------------------------------------
class TestP0TwoEntitiesResolve(unittest.TestCase):
    def test_two_entities_each_resolve_to_exactly_one_judgment(self):
        async def run():
            CUP_A = Box(0.40, 0.40, 0.20, 0.20)
            CUP_B = Box(0.05, 0.05, 0.20, 0.20)  # disjoint -> a distinct entity
            vision = _CondVision(
                script={
                    "a": {"a fresh cup": YES},
                    "b": {"a fresh cup": YES},
                },
                detect={"a": [Observation("cup", CUP_A, 0.9)], "b": [Observation("cup", CUP_B, 0.9)]},
            )
            settings = Settings(gate=GateSettings(threshold=0.7, validity_window_s=10.0))
            a = Percept.create(vision=vision, settings=settings, discover_plugins=False)
            a.add_goal(Goal(id="cups", condition="a fresh cup", say="a fresh cup"))
            rec = _recorder()
            a.on_trace = rec.tracer()

            evA = await a.perceive_judged("a", t=1.0)
            evB = await a.perceive_judged("b", t=2.0)
            self.assertEqual([e.action for e in evA], [FIRE])
            self.assertEqual([e.action for e in evB], [FIRE])
            fireA, fireB = evA[0], evB[0]
            # distinct entities -> distinct gate keys -> distinct judgment_event_ids
            self.assertIsNotNone(fireA.entity_id)
            self.assertIsNotNone(fireB.entity_id)
            self.assertNotEqual(fireA.entity_id, fireB.entity_id)

            incA = _emit_incident_created(rec, fireA, 1.0)
            incB = _emit_incident_created(rec, fireB, 2.0)
            self.assertNotEqual(incA.judgment_event_id, incB.judgment_event_id)

            report = verify_evidence_links(rec.records)
            self.assertTrue(report.ok, report.to_dict())
            self.assertEqual(report.checked, 2)
            self.assertEqual(report.resolved, 2)
            self.assertEqual(report.legacy_unverifiable, 0)
            self.assertEqual(report.violations, ())

        asyncio.run(run())


# ---------------------------------------------------------------------------
# 3. transition fixture e2e
# ---------------------------------------------------------------------------
class TestTransitionE2E(unittest.TestCase):
    def _build(self):
        vision = _CondVision(script={
            "closed": {"the door is closed": YES, "the door is open": NO},
            "open": {"the door is closed": NO, "the door is open": YES},
        })
        settings = Settings(gate=GateSettings(threshold=0.7, validity_window_s=10.0))
        a = Percept.create(vision=vision, settings=settings, discover_plugins=False)
        a.add_goal(Goal(
            id="door", condition="the door opened", say="the door opened",
            state={"type": "transition", "from": "the door is closed", "to": "the door is open"},
        ))
        return a

    def test_transition_fire_carries_a_resolvable_judgment(self):
        async def run():
            a = self._build()
            rec = _recorder()
            a.on_trace = rec.tracer()

            ev1 = await a.perceive_judged("closed", t=1.0)  # arm (see 'from')
            self.assertEqual(ev1, [])
            ev2 = await a.perceive_judged("open", t=2.0)     # fire (see 'to' after 'from')
            self.assertEqual([e.action for e in ev2], [FIRE])
            fire = ev2[0]

            # The FireEvent carries a REAL JudgmentEvent marked transition-origin, with real stamps.
            self.assertIsNotNone(fire.judgment)
            self.assertEqual(fire.judgment.evidence_origin, "transition")
            self.assertIsNotNone(fire.judgment.evidence_seq)
            tm = fire.judgment.timing
            self.assertTrue(
                tm.ingest_observed_ts <= tm.admitted_ts <= tm.judge_started_ts
                <= tm.judge_completed_ts <= tm.decision_emitted_ts
            )
            self.assertEqual(tm.source_media_ts, 2.0)  # the source timeline rode through, untouched

            je = [r for r in rec.records if r.stage == STAGE_JUDGMENT_EVENT]
            self.assertEqual(len(je), 1)
            self.assertEqual(je[0].data["evidence_origin"], "transition")
            self.assertIsNotNone(je[0].data["dispatch_seq"])
            # The recorder stamped data.evidence_seq = this trace's source.frame seq.
            sf = [r for r in rec.records
                  if r.stage == STAGE_SOURCE_FRAME and r.trace_id == je[0].trace_id]
            self.assertEqual(len(sf), 1)
            self.assertEqual(je[0].data["evidence_seq"], sf[0].seq)

            # The delivered Incident carries the same origin + a non-null link that resolves.
            inc = _emit_incident_created(rec, fire, 2.0)
            self.assertEqual(inc.evidence_origin, "transition")
            self.assertIsNotNone(inc.judgment_event_id)
            self.assertIsNotNone(inc.evidence_seq)
            report = verify_evidence_links(rec.records)
            self.assertTrue(report.ok, report.to_dict())
            self.assertEqual(report.resolved, 1)

        asyncio.run(run())


# ---------------------------------------------------------------------------
# 4. B-without-A never fires; transition.decision is truthful
# ---------------------------------------------------------------------------
class TestBWithoutANeverFires(unittest.TestCase):
    def test_idle_and_armed_are_reported_truthfully(self):
        async def run():
            vision = _CondVision(script={
                "open": {"the door is closed": NO, "the door is open": YES},   # in B, never saw A
                "closed": {"the door is closed": YES, "the door is open": NO},
            })
            settings = Settings(gate=GateSettings(threshold=0.7, validity_window_s=10.0))
            a = Percept.create(vision=vision, settings=settings, discover_plugins=False)
            a.add_goal(Goal(
                id="door", condition="the door opened", say="opened",
                state={"type": "transition", "from": "the door is closed", "to": "the door is open"},
            ))
            rec = _recorder()
            a.on_trace = rec.tracer()

            # Join already in B (to==yes, from never seen) -> NEVER fires, decision is idle.
            self.assertEqual(await a.perceive_judged("open", t=1.0), [])
            self.assertEqual(await a.perceive_judged("open", t=2.0), [])
            decisions = [r for r in rec.records if r.stage == STAGE_TRANSITION_DECISION]
            self.assertEqual(len(decisions), 2)
            for d in decisions:
                self.assertEqual(d.data["outcome"], "idle")
                self.assertFalse(d.data["armed"])
                self.assertFalse(d.data["fired_latch"])

            # Now witness A -> the NEXT tick reports armed, and still no fire on that tick.
            self.assertEqual(await a.perceive_judged("closed", t=3.0), [])
            armed = [r for r in rec.records if r.stage == STAGE_TRANSITION_DECISION][-1]
            self.assertEqual(armed.data["outcome"], "armed")
            self.assertTrue(armed.data["armed"])
            # No judgment.event / incident-resolvable link was produced by a non-firing run.
            self.assertEqual([r for r in rec.records if r.stage == STAGE_JUDGMENT_EVENT], [])

        asyncio.run(run())


# ---------------------------------------------------------------------------
# 5. a broken judgment stamp still delivers the fire (links None)
# ---------------------------------------------------------------------------
class TestBuildFailedStillFires(unittest.TestCase):
    def test_unfreezable_timing_delivers_fire_without_a_link(self):
        class Metrics:
            def __init__(self):
                self.seen = []

            def incr(self, key):
                self.seen.append(key)

        metrics = Metrics()
        ex = BehaviorExecutor(Gate(), metrics)
        goal = Goal(id="door", condition="opened", say="opened",
                    state={"type": "transition", "from": "A", "to": "B"})
        # A timing builder missing the judge stamps -> freeze() raises -> _judgment swallows -> None.
        broken = JudgmentTimingBuilder(clock=lambda: 0.0)
        broken.mark_ingest_observed(at=0.0)  # deliberately incomplete
        ex.transition_decide(goal, YES, NO, focus_id="scene")           # arm
        tick = ex.transition_decide(goal, NO, YES, focus_id="scene", timing=broken, evidence_seq=7)
        self.assertIsNotNone(tick.fired)                                # the FIRE is still delivered
        self.assertEqual(tick.fired.action, FIRE)
        self.assertIsNone(tick.fired.judgment)                          # ...but carries no judgment
        self.assertIn("judgment.build_failed", metrics.seen)
        inc = make_incident(tick.fired, ts=1.0)
        self.assertIsNone(inc.judgment_event_id)                        # links are absent, not fabricated
        self.assertIsNone(inc.evidence_origin)
        self.assertIsNone(inc.evidence_seq)


# ---------------------------------------------------------------------------
# 6. a pre-1.9 trace replays with a byte-identical decision digest
# ---------------------------------------------------------------------------
class TestPre19ReplayDigestUnchanged(unittest.TestCase):
    def test_the_evidence_link_stages_are_inert_to_replay(self):
        async def run():
            CUP = Box(0.4, 0.4, 0.2, 0.2)
            vision = _CondVision(
                script={"a": {"a fresh cup": YES}},
                detect={"a": [Observation("cup", CUP, 0.9)]},
            )
            settings = Settings(gate=GateSettings(threshold=0.7, validity_window_s=10.0))
            a = Percept.create(vision=vision, settings=settings, discover_plugins=False)
            a.add_goal(Goal(id="cups", condition="a fresh cup", say="cup"))
            rec = _recorder()
            a.on_trace = rec.tracer()
            await a.perceive_judged("a", t=1.0)

            records = list(rec.records)
            manifest = rec.manifest
            full = replay_cognition(manifest, records).digest
            # A pre-1.9 stream had no judgment.event stage at all — dropping it must not move the digest,
            # because replay reads ONLY gate.classified inputs. This is the byte-identical guarantee.
            pre19 = [r for r in records if r.stage != STAGE_JUDGMENT_EVENT]
            self.assertNotEqual(len(pre19), len(records))  # there WAS a 1.9 stage to drop
            self.assertEqual(replay_cognition(manifest, pre19).digest, full)
            # And a re-replay is deterministic.
            self.assertEqual(replay_cognition(manifest, records).digest, full)

        asyncio.run(run())


# ---------------------------------------------------------------------------
# 7. verify_evidence_links: zero-match, multi-match, legacy counting
# ---------------------------------------------------------------------------
class TestVerifyEvidenceLinks(unittest.TestCase):
    def _rec(self, seq, stage, trace_id, **data):
        return TraceRecord(seq=seq, trace_id=trace_id, stage=stage, source_ts=float(seq), data=data)

    def test_clean_multi_zero_and_legacy(self):
        jid = judgment_event_id("g", "e1", 1)
        other = judgment_event_id("g", "e2", 1)
        records = [
            # a clean pair: source.frame(seq0) -> judgment.event stamps evidence_seq=0 -> incident resolves
            self._rec(0, STAGE_SOURCE_FRAME, "camera@1"),
            self._rec(1, STAGE_JUDGMENT_EVENT, "camera@1", judgment_event_id=jid, evidence_seq=0),
            self._rec(2, STAGE_INCIDENT_CREATED, "camera@1", judgment_event_id=jid, incident_id="inc-1"),
            # ZERO-match: an incident referencing a handle no judgment.event carries
            self._rec(3, STAGE_INCIDENT_CREATED, "camera@2", judgment_event_id="ghost", incident_id="inc-2"),
            # MULTI-match: two judgment.event rows share one handle -> ambiguous
            self._rec(4, STAGE_SOURCE_FRAME, "camera@3"),
            self._rec(5, STAGE_JUDGMENT_EVENT, "camera@3", judgment_event_id=other, evidence_seq=4),
            self._rec(6, STAGE_JUDGMENT_EVENT, "camera@3", judgment_event_id=other, evidence_seq=4),
            self._rec(7, STAGE_INCIDENT_CREATED, "camera@3", judgment_event_id=other, incident_id="inc-3"),
            # LEGACY: a pre-1.9 incident with no link -> counted, never a violation
            self._rec(8, STAGE_INCIDENT_CREATED, "camera@4", incident_id="inc-legacy"),
        ]
        report = verify_evidence_links(records)
        self.assertIsInstance(report, EvidenceLinkReport)
        self.assertFalse(report.ok)
        self.assertEqual(report.checked, 3)        # inc-1, inc-2, inc-3 (inc-legacy is not checked)
        self.assertEqual(report.resolved, 1)       # only inc-1 resolves cleanly
        self.assertEqual(report.legacy_unverifiable, 1)
        kinds = sorted(v.kind for v in report.violations)
        self.assertEqual(kinds, ["ambiguous_evidence_ref", "unresolved_evidence_ref"])

    def test_missing_source_frame(self):
        jid = judgment_event_id("g", "e", 1)
        records = [
            # judgment.event's stamped evidence_seq (99) names no source.frame in this trace
            self._rec(0, STAGE_JUDGMENT_EVENT, "camera@1", judgment_event_id=jid, evidence_seq=99),
            self._rec(1, STAGE_INCIDENT_CREATED, "camera@1", judgment_event_id=jid, incident_id="inc-1"),
        ]
        report = verify_evidence_links(records)
        self.assertFalse(report.ok)
        self.assertEqual([v.kind for v in report.violations], ["missing_source_frame"])

    def test_all_legacy_stream_is_ok(self):
        records = [
            self._rec(0, STAGE_INCIDENT_CREATED, "camera@1", incident_id="a"),
            self._rec(1, STAGE_INCIDENT_CREATED, "camera@2", incident_id="b"),
        ]
        report = verify_evidence_links(records)
        self.assertTrue(report.ok)
        self.assertEqual(report.checked, 0)
        self.assertEqual(report.legacy_unverifiable, 2)


# ---------------------------------------------------------------------------
# 8. JS golden-vector parity (Python side asserts the shared fixture)
# ---------------------------------------------------------------------------
class TestGoldenVectorParity(unittest.TestCase):
    def test_python_matches_the_shared_vectors(self):
        payload = json.loads(_GOLDEN.read_text())
        vectors = payload["vectors"]
        self.assertGreaterEqual(len(vectors), 10)
        for v in vectors:
            self.assertEqual(
                judgment_event_id(v["goal"], v["entity"], v["seq"]), v["id"],
                f"python drifted from the shared golden vector {v!r}",
            )


# ---------------------------------------------------------------------------
# 9. existing positional fire_transition call sites unchanged
# ---------------------------------------------------------------------------
class TestFireTransitionPositionalUnchanged(unittest.TestCase):
    def test_positional_call_still_returns_fired_or_none(self):
        ex = BehaviorExecutor(Gate())
        goal = Goal(id="wake", condition="awake", say="good morning",
                    state={"type": "transition", "from": "A", "to": "B"})
        # join in B (no A) -> None; arm on A -> None; fire on B after A -> Fired; no re-fire -> None
        script = [(NO, YES), (YES, NO), (NO, YES), (NO, YES)]
        out = [ex.fire_transition(goal, fr, to) for fr, to in script]
        self.assertEqual([f.action if f else None for f in out], [None, None, FIRE, None])
        self.assertEqual(out[2].text, "good morning")
        self.assertIsNone(out[2].judgment)  # no timing threaded -> no fabricated judgment


if __name__ == "__main__":
    unittest.main()
