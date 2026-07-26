"""— the provisional/superseding link reaches LIVE delivery, not just the executor.

The pre-existing gap: ``behavior.Fired`` carries ``revision_id``/``supersedes`` (exercised at the
executor level, see ``test_y10_supersession.py``), but ``agent.FireEvent`` and ``_fire_concern``'s
``Fired`` -> ``FireEvent`` adaptation dropped both — so ``incidents.make_incident`` in the LIVE
``percept.runtime.worker.run()`` delivery path read ``None`` for both on EVERY fire, and a webhook
consumer never saw a supersession link.

This module drives the round trip through the REAL delivery path (``run()``, not a bare
``Percept.perceive_judged`` call): a claim-mode correction fires a superseding revision that must
reach a webhook-sink test double with ``revision_id``/``supersedes`` populated, on the WIRE
(``incident_payload()``, the JSON a real ``WebhookSink`` POSTs) as well as on the ``Incident`` object;
a plain (non-claim) fire must carry ``None``/``None``; and the pre-1.8 positional ``Incident(...)``
construction must stay byte-for-byte.
"""
from __future__ import annotations

import asyncio
import unittest

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.backends.fake import FakeVision
from percept.config import GateSettings, Settings
from percept.contracts import Incident, Verdict as ContractVerdict
from percept.harness.tier0 import gray_frame
from percept.runtime import incident_payload, run

W, H = 8, 8
STILL = gray_frame(W, H, 0)
MOVED = gray_frame(W, H, 255)


def _collect(core, source, **kw):
    async def _go():
        return [inc async for inc in run(core, source=source, judge_dispatch="inline", **kw)]
    return asyncio.run(_go())


# --------------------------------------------------------------------------------------------------
# A claim-mode correction, driven through the REAL run() delivery path.
# --------------------------------------------------------------------------------------------------


class TestClaimSupersessionReachesLiveDelivery(unittest.TestCase):
    def _core(self):
        vision = FakeVision(script={
            "assert_budget": Verdict(
                True, 0.9, "The budget is $40K", operation="assert", value="$40K", asserted_by="buyer",
            ),
            "replace_budget": Verdict(
                True, 0.9, "Actually the approved budget is $25K", operation="replace",
                value="$25K", asserted_by="buyer",
            ),
        })
        settings = Settings(gate=GateSettings(threshold=0.7, validity_window_s=10_000.0))
        core = Percept.create(vision=vision, settings=settings, discover_plugins=False)
        core.add_goal(Goal(
            id="g1", condition="a BUYER-side speaker states the approved budget", say="budget claimed",
            state={"type": "watch", "count": 0, "identity_policy": "claim_slot"},
        ))
        return core

    # (t, gate_frame, judge_frame): STILL->MOVED transitions are the rising edges the motion gate
    # escalates on (mirrors tests/test_runtime.py's SOURCE shape) — one for the original claim, one
    # (after a reset back to STILL) for the later correction.
    SOURCE = [
        (0.0, STILL, "assert_budget"),   # prime: no motion yet, gate does not escalate
        (1.0, MOVED, "assert_budget"),   # still->moved: escalate -> assert $40K -> FIRE (provisional)
        (2.0, STILL, "assert_budget"),   # moved->still: re-arm the motion edge
        (25.0, MOVED, "replace_budget"), # still->moved: escalate -> replace $25K -> FIRE (superseding)
    ]

    def test_webhook_double_sees_revision_id_and_supersedes(self):
        delivered = []
        incidents = _collect(self._core(), self.SOURCE, on_incident=delivered.append)
        self.assertEqual(len(incidents), 2, "both the original claim and the correction fire")
        self.assertIs(delivered, delivered)
        self.assertEqual([i.id for i in delivered], [i.id for i in incidents])

        original, correction = incidents

        # The ORIGINAL (provisional) incident carries its own revision id and no supersedes link.
        self.assertIsNotNone(original.revision_id)
        self.assertIsNone(original.supersedes)

        # The CORRECTION carries a DIFFERENT revision id and supersedes the original's.
        self.assertIsNotNone(correction.revision_id)
        self.assertNotEqual(correction.revision_id, original.revision_id)
        self.assertEqual(correction.supersedes, original.revision_id)

        # And it is on the WIRE — the exact JSON a real WebhookSink POSTs (percept.runtime.sinks.
        # incident_payload / WebhookSink.__call__ both build this dict), not just a Python attribute.
        payload = incident_payload(correction)
        self.assertEqual(payload["revision_id"], correction.revision_id)
        self.assertEqual(payload["supersedes"], original.revision_id)
        original_payload = incident_payload(original)
        self.assertEqual(original_payload["revision_id"], original.revision_id)
        self.assertIsNone(original_payload["supersedes"])


# --------------------------------------------------------------------------------------------------
# A plain (non-claim) fire — no episode fold at all — carries None/None, unchanged.
# --------------------------------------------------------------------------------------------------


class TestPlainFireCarriesNoLink(unittest.TestCase):
    def _core(self):
        vision = FakeVision(script={
            "intruder": Verdict(satisfied=True, confidence=0.95, evidence="a person is in the zone"),
            "clear": Verdict(satisfied=False, confidence=0.95, evidence="the zone is empty"),
        })
        core = Percept.create(vision=vision, discover_plugins=False)
        core.add_goal(Goal(id="zone", condition="a person is in the restricted zone", say="Alert: person in zone"))
        return core

    SOURCE = [
        (0.0, STILL, "clear"),
        (1.0, MOVED, "intruder"),  # still->moved: escalate -> FIRE, no episode fold at all
    ]

    def test_plain_fire_delivers_none_none_on_the_wire(self):
        delivered = []
        incidents = _collect(self._core(), self.SOURCE, on_incident=delivered.append)
        self.assertEqual(len(incidents), 1)
        inc = incidents[0]
        self.assertIsNone(inc.revision_id)
        self.assertIsNone(inc.supersedes)
        payload = incident_payload(inc)
        self.assertIsNone(payload["revision_id"])
        self.assertIsNone(payload["supersedes"])
        self.assertIs(delivered[0], inc)


# --------------------------------------------------------------------------------------------------
# The pre-1.8 positional Incident(...) construction stays byte-for-byte.
# --------------------------------------------------------------------------------------------------


class TestOldIncidentConstructionByteForByte(unittest.TestCase):
    def test_positional_construction_defaults_new_fields_to_none(self):
        v = ContractVerdict(satisfied=True, confidence=0.9, evidence="baby eyes open")
        inc = Incident("inc-1", "c-baby-awake", v, 123.0)  # the exact pre-1.8 positional shape
        self.assertEqual(inc.id, "inc-1")
        self.assertEqual(inc.concern_id, "c-baby-awake")
        self.assertIs(inc.verdict, v)
        self.assertEqual(inc.ts, 123.0)
        self.assertIsNone(inc.label)
        self.assertIsNone(inc.snippet)
        self.assertEqual(inc.message, "")
        self.assertIsNone(inc.label_url)
        self.assertIsNone(inc.entity_id)
        self.assertIsNone(inc.reason)
        self.assertIsNone(inc.watchplan_id)
        self.assertIsNone(inc.roi)
        self.assertIsNone(inc.modality)
        self.assertIsNone(inc.provenance)
        self.assertIsNone(inc.evidence_seq)
        # the new 1.8 fields — additive, appended last, defaulted None
        self.assertIsNone(inc.revision_id)
        self.assertIsNone(inc.supersedes)

    def test_keyword_construction_with_the_new_fields_round_trips_on_the_wire(self):
        v = ContractVerdict(satisfied=True, confidence=0.9, evidence="e")
        inc = Incident(
            id="inc-2", concern_id="c", verdict=v, ts=1.0,
            revision_id="g:ep1#r2", supersedes="g:ep1#r1",
        )
        payload = incident_payload(inc)
        self.assertEqual(payload["revision_id"], "g:ep1#r2")
        self.assertEqual(payload["supersedes"], "g:ep1#r1")


if __name__ == "__main__":
    unittest.main()
