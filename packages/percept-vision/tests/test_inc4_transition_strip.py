"""The A->B transition path judges from/to over the SAME temporal strip the watch path uses.

The motivating client repro (keyboard-submitted approval): the status flip "lands between 4fps
samples or renders a frame late", so NO single sampled frame cleanly shows the completed 'to' state
— the change is only recognizable ACROSS frames (a pre-flip frame + the latest post-flip frame in
one clip). The legacy transition path judged BARE single frames (``vision.judge(to, frame)`` with
no ``frames=``), so it missed the flip and the armed goal never fired.

This module pins the regression as a strip-vs-single A/B on the SAME frames / SAME judge:

* ``plan_mode='single'`` -> ``strip is None`` -> the judge sees only the anchor frame -> MISS (0 fires)
* ``plan_mode='strip'``  -> the judge sees the 3-frame clip (before + ... + after) -> FIRE exactly once

Plus: 10/10 negatives stay quiet under the new strip path, the fire still satisfies the evidence-link
rules (``verify_evidence_links`` resolves it, ``evidence_seq`` / ``judgment_event_id`` present), and
the extractor seam threads facts into BOTH judge calls exactly like ``perceive_judged``. The five
byte-for-byte single-frame variants live in ``tests/test_transition.py`` (unchanged).
"""
from __future__ import annotations

import asyncio
import unittest

from percept.adapters import Goal, Verdict
from percept.agent import Percept
from percept.config import GateSettings, Settings
from percept.extraction import Enrichment
from percept.gate import FIRE, Gate
from percept.incidents import make_incident
from percept.replay import gate_config_of, verify_evidence_links
from percept.tracing import (
    STAGE_INCIDENT_CREATED,
    STAGE_JUDGMENT_EVENT,
    STAGE_SOURCE_FRAME,
    STAGE_TRANSITION_DECISION,
    RunTraceManifest,
    TracePolicy,
    TraceRecorder,
)

FROM = "the form is pending approval"
TO = "the form shows approved"


class ClipVision:
    """A ``Vision`` double that judges from/to over the WHOLE clip (strip earlier frames + anchor).

    'from' is a plain single-frame state (any ``pending*`` frame in the clip). 'to' models the
    keyboard-submit flip that is ONLY recognizable across frames: the clip must bracket BOTH the
    ``before`` (pre-flip) and ``after`` (post-flip) frame. No single isolated frame satisfies 'to' —
    so a single-frame judge of the latest frame alone never sees the flip (that IS the client repro).
    Records every call so a test can assert on the strip width / context the runtime assembled.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def judge(self, condition, frame, *, roi=None, exemplar=None,
                    corrections=None, frames=None, context=None):
        clip = [str(f) for f in (frames or [])] + [str(frame)]
        self.calls.append({
            "condition": condition, "anchor": str(frame),
            "strip_len": len(frames or []), "clip": clip, "context": context,
        })
        if condition == FROM:
            sat = any(tok.startswith("pending") for tok in clip)
        elif condition == TO:
            sat = ("before" in clip) and ("after" in clip)  # emergent across frames — never a lone frame
        else:
            sat = False
        return Verdict(satisfied=sat, confidence=0.95)

    async def detect(self, frame, goals):
        return []

    async def detect_events(self, frames, goals):
        return []


def _approval_goal() -> Goal:
    return Goal(
        id="approval",
        condition="the form was approved",
        say="approved",
        state={"type": "transition", "from": FROM, "to": TO, "count": 0},
    )


def _settings() -> Settings:
    return Settings(gate=GateSettings(threshold=0.7, validity_window_s=10.0))


def _run(frames: list[str], *, plan_mode: str, vision: ClipVision | None = None,
         extractor=None, on_trace=None):
    """Push ``frames`` one per tick through perceive_judged; return (fires, agent, vision)."""
    vision = vision or ClipVision()
    a = Percept.create(
        vision=vision, settings=_settings(), discover_plugins=False,
        plan_mode=plan_mode, extractor=extractor,
    )
    if on_trace is not None:
        a.on_trace = on_trace
    a.add_goal(_approval_goal())
    fires: list = []

    async def go():
        for i, tok in enumerate(frames):
            evs = await a.perceive_judged(tok, t=float(i + 1))
            fires.extend(e for e in evs if e.action == FIRE)

    asyncio.run(go())
    return fires, a, vision


# The keyboard-submit sequence: arm on 'pending', a gap, then the flip brackets 'before'->'after'.
# Under a 3-frame strip the fire tick's clip is ["gap","before","after"] — the 'to' state (the
# completed flip) is confirmed by the LATEST frame ("after") but ONLY given the earlier "before" in
# the strip; no single frame carries it.
_KEYBOARD_SUBMIT = ["pending", "gap", "before", "after"]


class TestKeyboardSubmitRegression(unittest.TestCase):
    def test_single_frame_judging_misses_the_flip(self):
        # PRE-INC4 BEHAVIOUR, pinned: single-frame judging (strip is None) sees only the anchor frame,
        # never before+after together, so the armed transition never fires. This is the missed incident.
        fires, a, vision = _run(_KEYBOARD_SUBMIT, plan_mode="single")
        self.assertEqual(len(fires), 0)
        self.assertEqual(a.goals.get("approval").state["count"], 0)
        # ...and it truly judged BARE single frames — no strip was ever assembled.
        self.assertTrue(all(c["strip_len"] == 0 for c in vision.calls))

    def test_strip_judging_fires_exactly_once(self):
        fires, a, vision = _run(_KEYBOARD_SUBMIT, plan_mode="strip")
        self.assertEqual(len(fires), 1)
        self.assertEqual(a.goals.get("approval").state["count"], 1)
        # The fire tick judged the 3-frame clip (2 earlier strip frames + the anchor).
        to_calls = [c for c in vision.calls if c["condition"] == TO]
        fired_call = next(c for c in to_calls if "before" in c["clip"] and "after" in c["clip"])
        self.assertEqual(fired_call["strip_len"], 2)
        self.assertEqual(fired_call["clip"][-1], "after")  # the 'to' is confirmed by the LATEST frame

    def test_strip_decision_trace_carries_strip_len_and_enriched(self):
        rec = _recorder()
        _run(_KEYBOARD_SUBMIT, plan_mode="strip", on_trace=rec.tracer())
        decisions = [r for r in rec.records if r.stage == STAGE_TRANSITION_DECISION]
        self.assertEqual(len(decisions), len(_KEYBOARD_SUBMIT))  # one per tick, fire or not
        fire_dec = next(d for d in decisions if d.data["outcome"] == "fire")
        self.assertEqual(fire_dec.data["strip_len"], 2)   # single-vs-strip is diagnosable from the trace
        self.assertFalse(fire_dec.data["enriched"])       # no extractor here
        # The very first tick armed off a bare single frame (window had no earlier frame yet).
        self.assertEqual(decisions[0].data["strip_len"], 0)


# ---------------------------------------------------------------------------
# 10/10 negatives stay quiet under the new strip path
# ---------------------------------------------------------------------------
_NEGATIVES = [
    ["gap"],                               # 1. nothing at all
    ["pending"],                           # 2. armed only, no flip
    ["pending", "pending"],                # 3. armed only, no flip
    ["before", "after"],                   # 4. flip but 'from' never witnessed -> never armed
    ["gap", "before", "after"],            # 5. flip (with gap) but never armed
    ["pending", "before"],                 # 6. armed, only pre-flip half of the flip
    ["pending", "after"],                  # 7. armed, only post-flip half of the flip
    ["pending", "gap", "before"],          # 8. armed, 'after' never arrives
    ["before", "after", "gap", "gap2"],    # 9. flip early, never armed
    ["gap", "gap2", "gap3"],               # 10. pure noise
]


class TestNegativesQuiet(unittest.TestCase):
    def test_ten_negatives_never_fire_under_strip(self):
        quiet = 0
        for seq in _NEGATIVES:
            fires, a, _ = _run(seq, plan_mode="strip")
            self.assertEqual(len(fires), 0, f"negative should stay quiet: {seq}")
            self.assertEqual(a.goals.get("approval").state["count"], 0, f"{seq}")
            quiet += 1
        self.assertEqual(quiet, 10)  # 10/10 quiet


# ---------------------------------------------------------------------------
# The strip fire still satisfies the evidence-link rules
# ---------------------------------------------------------------------------
def _recorder(level="full", source="camera"):
    manifest = RunTraceManifest.create(
        run_id="run-inc4", source=source, policy=TracePolicy(level=level),
        gate_config=gate_config_of(Gate()),
    )
    return TraceRecorder(manifest)


class TestStripFireEvidenceLinks(unittest.TestCase):
    def test_strip_fire_resolves_to_exactly_one_judgment(self):
        async def run():
            vision = ClipVision()
            a = Percept.create(
                vision=vision, settings=_settings(), discover_plugins=False, plan_mode="strip",
            )
            a.add_goal(_approval_goal())
            rec = _recorder()
            a.on_trace = rec.tracer()

            fire = None
            for i, tok in enumerate(_KEYBOARD_SUBMIT):
                for e in await a.perceive_judged(tok, t=float(i + 1)):
                    if e.action == FIRE:
                        fire = e
            self.assertIsNotNone(fire)

            # A REAL JudgmentEvent, transition-origin, with a non-null evidence_seq.
            self.assertIsNotNone(fire.judgment)
            self.assertEqual(fire.judgment.evidence_origin, "transition")
            self.assertIsNotNone(fire.judgment.evidence_seq)

            # judgment.event is projected into the trace and stamped against THIS trace's source.frame.
            je = [r for r in rec.records if r.stage == STAGE_JUDGMENT_EVENT]
            self.assertEqual(len(je), 1)
            sf = [r for r in rec.records
                  if r.stage == STAGE_SOURCE_FRAME and r.trace_id == je[0].trace_id]
            self.assertEqual(len(sf), 1)
            self.assertEqual(je[0].data["evidence_seq"], sf[0].seq)

            # The delivered Incident carries a link that resolves — reuse the client-audit verifier.
            inc = make_incident(fire, ts=99.0)
            rec.record(
                STAGE_INCIDENT_CREATED, trace_id=je[0].trace_id, source_ts=99.0,
                goal_id=inc.concern_id, incident_id=inc.id, entity_id=inc.entity_id,
                judgment_event_id=inc.judgment_event_id, evidence_origin=inc.evidence_origin,
            )
            self.assertIsNotNone(inc.judgment_event_id)
            self.assertIsNotNone(inc.evidence_seq)
            report = verify_evidence_links(rec.records)
            self.assertTrue(report.ok, report.to_dict())
            self.assertEqual(report.resolved, 1)
            self.assertEqual(report.violations, ())

        asyncio.run(run())


# ---------------------------------------------------------------------------
# The extractor seam threads facts into BOTH judge calls, exactly like perceive_judged
# ---------------------------------------------------------------------------
class _StubExtractor:
    """Returns the same fact on every frame, and records each call's (condition, frame, roi)."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def extract(self, condition, frame, roi=None):
        self.calls.append((condition, str(frame), roi))
        return Enrichment(facts={"invoice_number": "INV-2291"})


class TestExtractorThreadedIntoBothJudgeCalls(unittest.TestCase):
    def test_extractor_facts_reach_from_and_to_and_trace(self):
        async def run():
            vision = ClipVision()
            extractor = _StubExtractor()
            a = Percept.create(
                vision=vision, settings=_settings(), discover_plugins=False,
                plan_mode="strip", extractor=extractor,
            )
            a.add_goal(_approval_goal())
            rec = _recorder()
            a.on_trace = rec.tracer()

            for i, tok in enumerate(_KEYBOARD_SUBMIT):
                await a.perceive_judged(tok, t=float(i + 1))

            # Extracted ONCE per tick, off the concern's condition anchor (not from/to).
            self.assertEqual(len(extractor.calls), len(_KEYBOARD_SUBMIT))
            self.assertTrue(all(c[0] == "the form was approved" for c in extractor.calls))

            # BOTH judge calls on every tick carried the reserved extractor sub-dict.
            self.assertTrue(vision.calls)
            for c in vision.calls:
                self.assertIsNotNone(c["context"], c)
                self.assertIn("extractor", c["context"])
                self.assertEqual(c["context"]["extractor"]["facts"]["invoice_number"], "INV-2291")

            # The extractor.enriched trace stage fired, and transition.decision reports enriched=True.
            enriched_stages = [r for r in rec.records if r.stage == "extractor.enriched"]
            self.assertEqual(len(enriched_stages), len(_KEYBOARD_SUBMIT))
            decisions = [r for r in rec.records if r.stage == STAGE_TRANSITION_DECISION]
            self.assertTrue(all(d.data["enriched"] for d in decisions))

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
