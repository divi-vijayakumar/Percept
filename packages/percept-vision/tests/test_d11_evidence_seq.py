"""— Incident.evidence_seq + the recorder-side ``judgment.event`` trace linkage.

Two bars:

1. ROUND-TRIP: a real fire's ``judgment.event`` trace record resolves (through
   ``data["evidence_seq"]``) to a REAL ``source.frame`` record's ``seq`` — the SAME trace_id, the SAME
   ``.ptrace`` — and that link survives a JSON write/reload cycle.
2. BACKWARD-COMPAT: a pre-1.7 ``Incident`` construction (positional, no ``evidence_seq``) and a fire
   with no ``.judgment`` at all both still parse/construct fine, with ``evidence_seq=None`` — never a
   fabricated number. A pre-D11 trace record dict (no ``judgment.event`` stage, no ``evidence_seq`` key
   anywhere) still loads and derives cleanly.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from percept.adapters import Box, Goal, Observation, Verdict
from percept.agent import Percept
from percept.backends.fake import FakeVision
from percept.config import GateSettings, Settings
from percept.contracts import Incident
from percept.incidents import make_incident
from percept.tracing import (
    STAGE_JUDGMENT_EVENT,
    STAGE_SOURCE_FRAME,
    RunTraceManifest,
    TracePolicy,
    TraceRecord,
    TraceRecorder,
    load_trace,
)


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


CUP = Box(0.4, 0.4, 0.2, 0.2)


def _core() -> Percept:
    vision = FakeVision(
        script={"cup": Verdict(True, 0.95, "a cup of coffee")},
        detect_script={"cup": [Observation("cup", CUP, 0.9)]},
    )
    core = Percept.create(
        vision=vision,
        settings=Settings(gate=GateSettings(threshold=0.7, validity_window_s=10.0)),
        discover_plugins=False,
    )
    core.add_goal(Goal(id="g1", condition="the user is drinking coffee", say="stay off caffeine"))
    return core


class TestJudgmentEvidenceSeqRoundTrip(unittest.TestCase):
    """Bar 1 — the judgment.event record's evidence_seq resolves to a real source.frame seq."""

    def _drive_one_fire(self):
        core = _core()
        rec = TraceRecorder(
            RunTraceManifest.create(
                source="camera", driver="perceive_judged", goals=list(core.goals.active()),
                policy=TracePolicy(level="full"),
            ),
            wall_clock=FakeClock(),
        )
        core.on_trace = rec.tracer()

        async def drive():
            return await core.perceive_judged("cup", t=1.0)

        fires = asyncio.run(drive())
        rec.close()
        return core, rec, fires

    def test_fire_carries_a_judgment_with_a_positive_dispatch_seq(self):
        _, _, fires = self._drive_one_fire()
        fired = [f for f in fires if getattr(f, "action", None) == "fire"]
        self.assertEqual(len(fired), 1)
        judgment = fired[0].judgment
        self.assertIsNotNone(judgment)
        self.assertGreaterEqual(judgment.evidence_seq, 0)

    def test_judgment_event_trace_record_resolves_to_the_same_trace_source_frame(self):
        _, rec, fires = self._drive_one_fire()
        frame_recs = [r for r in rec.records if r.stage == STAGE_SOURCE_FRAME]
        judgment_recs = [r for r in rec.records if r.stage == STAGE_JUDGMENT_EVENT]
        self.assertEqual(len(frame_recs), 1)  # one frame admitted
        self.assertGreaterEqual(len(judgment_recs), 1)  # this goal judged and fired

        frame = frame_recs[0]
        jrec = judgment_recs[0]
        # SAME trace_id — one unit of evidence, correlated stages.
        self.assertEqual(jrec.trace_id, frame.trace_id)
        # The recorder stamped evidence_seq — not the caller — as the source.frame's OWN seq.
        self.assertEqual(jrec.data["evidence_seq"], frame.seq)
        # dispatch_seq carries the JudgmentEvent's OWN ADR-001 §8 sequence — a distinct number,
        # never conflated with the recorder-stamped evidence_seq.
        self.assertIn("dispatch_seq", jrec.data)
        fired = [f for f in fires if getattr(f, "action", None) == "fire"][0]
        self.assertEqual(jrec.data["dispatch_seq"], fired.judgment.evidence_seq)

    def test_link_survives_a_json_write_reload_cycle(self):
        _, rec, _ = self._drive_one_fire()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "run.ptrace"
            rec.write_json(path)
            manifest, records, _summary = load_trace(path.read_text())

        frame_recs = [r for r in records if r.stage == STAGE_SOURCE_FRAME]
        judgment_recs = [r for r in records if r.stage == STAGE_JUDGMENT_EVENT]
        self.assertEqual(len(frame_recs), 1)
        self.assertGreaterEqual(len(judgment_recs), 1)
        self.assertEqual(judgment_recs[0].trace_id, frame_recs[0].trace_id)
        self.assertEqual(judgment_recs[0].data["evidence_seq"], frame_recs[0].seq)
        self.assertEqual(manifest.contract_version and manifest.contract_version.split(".")[0], "1")

    def test_make_incident_carries_the_same_dispatch_seq_off_the_fire(self):
        _, _, fires = self._drive_one_fire()
        fired = [f for f in fires if getattr(f, "action", None) == "fire"][0]
        incident = make_incident(fired, ts=1.0)
        self.assertEqual(incident.evidence_seq, fired.judgment.evidence_seq)


class _NoJudgmentFire:
    """Shape-compatible stand-in for a fire from a path that builds no JudgmentEvent at all
    (episode/rhythm/boundary/transition/absence) — pre-1.7 shape, no ``.judgment`` attribute."""

    def __init__(self):
        self.goal_id = "g1"
        self.action = "fire"
        self.text = "hi"
        self.entity_id = None
        self.verdict = None


class TestBackwardCompat(unittest.TestCase):
    """Bar 2 — nothing pre-D11 breaks, and nothing pre-D11 is silently upgraded into a fabricated
    number. ``None`` stays ``None``."""

    def test_incident_positional_construction_without_evidence_seq_still_works(self):
        # The exact call shape every pre-1.7 caller already uses — no evidence_seq argument at all.
        incident = Incident("id-1", "g1", None, 1.0)
        self.assertIsNone(incident.evidence_seq)

    def test_make_incident_on_a_fire_with_no_judgment_attribute_yields_none(self):
        incident = make_incident(_NoJudgmentFire(), ts=2.0)
        self.assertIsNone(incident.evidence_seq)

    def test_pre_d11_trace_record_dict_parses_with_no_evidence_seq(self):
        # A gate.classified record shaped exactly like a trace captured before this card — no
        # judgment.event stage exists at all in this stream, and no data key mentions evidence_seq.
        payload = {
            "seq": 3, "trace_id": "camera@1.0", "stage": "gate.classified",
            "source_ts": 1.0, "observed_wall_ts": 1000.0, "emitted_wall_ts": 1000.0,
            "goal_id": "g1", "source": "camera",
            "data": {"truth": "yes", "action": "fire", "reason": "rising_edge"},
        }
        rec = TraceRecord.from_dict(payload)
        self.assertNotIn("evidence_seq", rec.data)
        self.assertIsNone(rec.data.get("evidence_seq"))

    def test_pre_d11_full_artifact_round_trips_and_derives_cleanly(self):
        manifest = RunTraceManifest.create(
            source="camera", driver="run", run_id="pre-d11-run",
            policy=TracePolicy(level="full"), wall_clock=lambda: 1000.0,
        )
        old_payload = {
            "schema_version": 1,
            "manifest": manifest.to_dict(),
            "summary": None,
            "records": [
                {"seq": 0, "trace_id": "camera@1.0", "stage": "source.frame", "source_ts": 1.0,
                 "observed_wall_ts": 1000.0, "emitted_wall_ts": 1000.0, "goal_id": None,
                 "source": "camera", "data": {}},
                {"seq": 1, "trace_id": "camera@1.0", "stage": "gate.classified", "source_ts": 1.0,
                 "observed_wall_ts": 1000.0, "emitted_wall_ts": 1000.0, "goal_id": "g1",
                 "source": "camera", "data": {"truth": "yes", "action": "fire", "reason": "rising_edge"}},
            ],
        }
        loaded_manifest, records, summary = load_trace(old_payload)
        self.assertEqual(loaded_manifest.run_id, "pre-d11-run")
        self.assertEqual(len(records), 2)
        self.assertEqual(summary.gate_outcomes["fire"], 1)
        # No judgment.event stage in this pre-D11 stream at all — never fabricated on reload.
        self.assertFalse(any(r.stage == STAGE_JUDGMENT_EVENT for r in records))


if __name__ == "__main__":
    unittest.main()
