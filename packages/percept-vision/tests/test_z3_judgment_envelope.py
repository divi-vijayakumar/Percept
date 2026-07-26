"""The judgment envelope is enriched and survives to the delivered Incident.

Previously, only about 5 of the reference envelope's 11 fields reached a delivered ``Incident``.
The gate reason (why a fire was taken
or withheld), the plan the judgment ran under, and the region/modality/provenance of the evidence all
existed internally yet none reached the durable record an operator inspects. The enriched contract
carries them additively.

The invariants these tests pin:

  1. the executor stamps ``reason`` (the gate's own O2 code), ``modality`` (how the eye looked) and
     ``provenance='observed'`` (a judged read is a direct observation) on both the frozen
     ``JudgmentEvent`` and the ``Fired``/``FireEvent`` — ``roi`` stays ``None`` (the judge does not
     report the region it fired on; echoing the input hint would misreport an input as an output);
  2. ``make_incident`` carries the whole envelope THROUGH the emission boundary onto the ``Incident``,
     and stamps a runtime-supplied ``watchplan_id``;
  3. BYTE-FOR-BYTE back-compat: a fire that carries no envelope (a bare duck-typed fire) yields an
     Incident whose five new fields are all ``None`` — a consumer that ignores them sees today's record;
  4. the JSON schema exposes every new field as OPTIONAL (never ``required``).
"""

from __future__ import annotations

import asyncio
import unittest
from dataclasses import replace

from percept.adapters import Goal, Verdict
from percept.agent import FireEvent, Percept
from percept.behavior import BehaviorExecutor, Fired, PerceptionSpec, program_of
from percept.contracts import Box, Incident, JudgmentEvent, JudgmentTimingBuilder
from percept.gate import ASK, FIRE, Gate
from percept.incidents import make_incident


class StepClock:
    def __init__(self, step: float = 1.0) -> None:
        self._t = 0.0
        self._step = step

    def __call__(self) -> float:
        self._t += self._step
        return self._t


class ScriptedVision:
    def __init__(self, verdicts):
        self._verdicts = list(verdicts)
        self.calls = 0

    async def judge(self, condition, frame, *, roi=None, exemplar=None, corrections=None,
                    frames=None, audio=None, context=None):
        v = self._verdicts[min(self.calls, len(self._verdicts) - 1)]
        self.calls += 1
        return v

    async def detect(self, frame, goals):
        return []

    async def detect_events(self, frames, goals):
        return []


YES = Verdict(satisfied=True, confidence=0.95, evidence="yes")
UNSURE = Verdict(satisfied=True, confidence=0.2, evidence="unsure")


def _goal(gid: str = "g") -> Goal:
    return Goal(id=gid, condition="the door is open", say="door open", state={"type": "watch"})


def _run_watch(verdicts, *, ts=(0.0,)):
    vision = ScriptedVision(verdicts)
    agent = Percept.create(vision=vision, discover_plugins=False, clock=StepClock())
    agent.add_goal(_goal())
    out: list[FireEvent] = []

    async def go():
        for t in ts:
            out.extend(await agent.perceive_judged("frame", t=t))

    asyncio.run(go())
    return agent, out


class _Fire:
    """Duck-typed fire with an optional Z3 envelope (mirrors FireEvent's attribute surface)."""

    def __init__(self, goal_id="g", text="hi", verdict=None, **env):
        self.goal_id = goal_id
        self.text = text
        self.verdict = verdict
        for k, v in env.items():
            setattr(self, k, v)


# --------------------------------------------------------------------------------------------------
# (1) The executor stamps the envelope on the live judged path.
# --------------------------------------------------------------------------------------------------


class LivePathStampsTheEnvelope(unittest.TestCase):
    def test_fire_carries_reason_modality_provenance_on_both_carriers(self):
        _, fires = _run_watch([YES])
        self.assertEqual([f.action for f in fires], [FIRE])
        fe = fires[0]
        # on the FireEvent (the carrier make_incident reads)
        self.assertEqual(fe.reason, "rising_edge")
        self.assertEqual(fe.modality, "vision")
        self.assertEqual(fe.provenance, "observed")
        self.assertIsNone(fe.roi)  # honest: the judge did not report a region it fired on
        # ...and mirrored on the frozen JudgmentEvent (the enriched envelope proper)
        j: JudgmentEvent = fe.judgment
        self.assertEqual((j.reason, j.modality, j.provenance, j.roi),
                         ("rising_edge", "vision", "observed", None))
        # watchplan_id is a PLAN fact the per-goal executor does not know -> None here (runtime stamps it)
        self.assertIsNone(j.watchplan_id)

    def test_ask_carries_the_ask_reason(self):
        _, fires = _run_watch([UNSURE])
        self.assertEqual([f.action for f in fires], [ASK])
        fe = fires[0]
        self.assertEqual(fe.reason, "low_confidence")  # the O2 split of "unknown": not stale, low conf
        self.assertEqual(fe.judgment.reason, "low_confidence")
        self.assertEqual(fe.judgment.action, "ask")

    def test_modality_reflects_the_concerns_perception_spec(self):
        # A concern that authors a non-default modality carries it onto the envelope (executor reads
        # program.perception.modality, independent of the actual judged input assembly).
        gate = Gate()
        ex = BehaviorExecutor(gate)
        goal = _goal()
        prog = replace(program_of(goal), perception=PerceptionSpec(modality="fused"))
        builder = JudgmentTimingBuilder(clock=StepClock())
        builder.mark_ingest_observed()
        builder.mark_admitted()
        builder.mark_judge_started()
        builder.mark_judge_completed()
        fired = ex.fire_judged(goal, YES, ("g", None), prog, t=0.0, timing=builder)
        self.assertIsInstance(fired, Fired)
        self.assertEqual(fired.modality, "fused")
        self.assertEqual(fired.judgment.modality, "fused")


# --------------------------------------------------------------------------------------------------
# (2) make_incident carries the envelope through the emission boundary + stamps watchplan_id.
# --------------------------------------------------------------------------------------------------


class MakeIncidentCarriesTheEnvelope(unittest.TestCase):
    def test_envelope_copies_off_the_fire(self):
        roi = Box(0.1, 0.2, 0.3, 0.4)
        fire = _Fire(goal_id="door", text="door open", verdict=YES,
                     reason="rising_edge", modality="vision", provenance="observed", roi=roi)
        inc = make_incident(fire, ts=5.0)
        self.assertEqual(inc.reason, "rising_edge")
        self.assertEqual(inc.modality, "vision")
        self.assertEqual(inc.provenance, "observed")
        self.assertIs(inc.roi, roi)

    def test_watchplan_id_param_stamps_the_incident(self):
        fire = _Fire(goal_id="door", reason="rising_edge")
        inc = make_incident(fire, ts=5.0, watchplan_id="plan-abc123")
        self.assertEqual(inc.watchplan_id, "plan-abc123")

    def test_watchplan_id_falls_back_to_the_fire(self):
        fire = _Fire(goal_id="door", watchplan_id="from-fire")
        self.assertEqual(make_incident(fire, ts=1.0).watchplan_id, "from-fire")
        # an explicit param wins over the fire's own value
        self.assertEqual(make_incident(fire, ts=1.0, watchplan_id="override").watchplan_id, "override")

    def test_end_to_end_live_fire_becomes_an_enriched_incident(self):
        _, fires = _run_watch([YES])
        inc = make_incident(fires[0], ts=9.0, watchplan_id="plan-xyz")
        self.assertEqual(inc.reason, "rising_edge")
        self.assertEqual(inc.modality, "vision")
        self.assertEqual(inc.provenance, "observed")
        self.assertIsNone(inc.roi)
        self.assertEqual(inc.watchplan_id, "plan-xyz")


# --------------------------------------------------------------------------------------------------
# (3) Byte-for-byte back-compat: a fire with NO envelope yields an all-None-enriched Incident.
# --------------------------------------------------------------------------------------------------


class BackCompatIsByteForByte(unittest.TestCase):
    def test_bare_fire_has_no_envelope(self):
        inc = make_incident(_Fire(goal_id="g", text="hi", verdict=YES), ts=1.0)
        self.assertEqual(
            (inc.reason, inc.watchplan_id, inc.roi, inc.modality, inc.provenance),
            (None, None, None, None, None),
        )

    def test_default_incident_is_unchanged(self):
        # A directly-constructed Incident with none of the new fields defaults them all to None.
        inc = Incident(id="i", concern_id="g", verdict=YES, ts=0.0)
        self.assertEqual(
            (inc.reason, inc.watchplan_id, inc.roi, inc.modality, inc.provenance),
            (None, None, None, None, None),
        )


# --------------------------------------------------------------------------------------------------
# (4) The JSON schema exposes every new field, all OPTIONAL.
# --------------------------------------------------------------------------------------------------


class SchemaExposesTheEnvelope(unittest.TestCase):
    def test_incident_schema_has_optional_envelope_fields(self):
        from percept.contracts import jsonschema as js

        s = js.generate_all()["$defs"]["Incident"]
        for name in ("reason", "watchplan_id", "roi", "modality", "provenance"):
            self.assertIn(name, s["properties"])
            self.assertNotIn(name, s.get("required", []))
        # roi is Optional[Box] -> anyOf carrying the $ref + null
        self.assertIn({"$ref": "#/$defs/Box"}, s["properties"]["roi"]["anyOf"])
        self.assertIn({"type": "null"}, s["properties"]["roi"]["anyOf"])
        # modality/provenance are closed vocab -> enum arm + null arm
        self.assertIn({"enum": ["vision", "audio", "fused"]}, s["properties"]["modality"]["anyOf"])
        self.assertIn({"enum": ["observed", "inferred"]}, s["properties"]["provenance"]["anyOf"])


# --------------------------------------------------------------------------------------------------
# (5) The real worker.run() path stamps the whole envelope — including a computed watchplan_id.
# --------------------------------------------------------------------------------------------------


class WorkerRunStampsTheEnvelope(unittest.TestCase):
    def _run(self):
        from percept import Goal as _Goal, Percept as _Percept
        from percept.backends.fake import FakeVision
        from percept.edge.tier0 import gray_frame
        from percept.runtime import run

        still, moved = gray_frame(8, 8, 0), gray_frame(8, 8, 255)
        source = [
            (0.0, still, "clear"),
            (1.0, moved, "intruder"),   # rising edge -> FIRE -> incident
        ]
        vision = FakeVision(script={
            "intruder": Verdict(satisfied=True, confidence=0.95, evidence="a person is in the zone"),
            "clear": Verdict(satisfied=False, confidence=0.95, evidence="the zone is empty"),
        })
        core = _Percept.create(vision=vision, discover_plugins=False)
        core.add_goal(_Goal(id="zone", condition="a person is in the restricted zone", say="Alert"))

        async def go():
            return [inc async for inc in run(core, source=source)]

        return core, asyncio.run(go())

    def test_delivered_incident_carries_reason_modality_provenance_and_a_watchplan_id(self):
        core, incidents = self._run()
        self.assertEqual(len(incidents), 1)
        inc = incidents[0]
        self.assertEqual(inc.reason, "rising_edge")
        self.assertEqual(inc.modality, "vision")
        self.assertEqual(inc.provenance, "observed")
        self.assertIsNone(inc.roi)
        # the watchplan_id is a real, stable plan_hash — and it MATCHES the plan the run watched under
        from percept.tracing import PlanIdentity
        expected = PlanIdentity.from_goals(
            core.goals.active(), revision=getattr(core.goals, "revision", None)).plan_hash
        self.assertIsNotNone(inc.watchplan_id)
        self.assertEqual(inc.watchplan_id, expected)


if __name__ == "__main__":
    unittest.main()
