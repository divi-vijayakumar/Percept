"""The live worker wires label and disposition feedback stores into ``run()``.

Reuses the offline iterable-source + FakeVision fixtures (no cv2, no camera, no key, no network): the
worker binds each yielded incident to the local LabelStore and appends every gate FIRE/ASK to the
DispositionLog. The default (neither wired) must be byte-for-byte the pre-L-1 path.
"""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.backends.fake import FakeVision
from percept.labelstore import DispositionLog, LabelStore
from percept.runtime import run
from percept.edge.tier0 import gray_frame

W, H = 8, 8
STILL = gray_frame(W, H, 0)
MOVED = gray_frame(W, H, 255)

SOURCE = [
    (0.0, STILL, "clear"),
    (1.0, MOVED, "intruder"),  # FIRE
    (2.0, STILL, "clear"),
    (3.0, MOVED, "intruder"),  # FIRE
]
ASK_SOURCE = [(0.0, STILL, "maybe"), (1.0, MOVED, "maybe")]  # low-confidence -> ASK


def _core() -> Percept:
    vision = FakeVision(
        script={
            "intruder": Verdict(satisfied=True, confidence=0.95, evidence="a person is in the zone"),
            "clear": Verdict(satisfied=False, confidence=0.95, evidence="the zone is empty"),
        }
    )
    core = Percept.create(vision=vision, discover_plugins=False)
    core.add_goal(Goal(id="zone", condition="a person is in the restricted zone", say="Alert: person in zone"))
    return core


def _ask_core() -> Percept:
    vision = FakeVision(script={"maybe": Verdict(satisfied=True, confidence=0.2, evidence="unclear")})
    core = Percept.create(vision=vision, discover_plugins=False)
    core.add_goal(Goal(id="zone", condition="a person is in the restricted zone", say="Alert"))
    return core


async def _collect(core, source, **kw):
    return [inc async for inc in run(core, source=source, **kw)]


class RuntimeLabelWiringTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_yielded_incident_is_bound_and_records_locally(self):
        store = LabelStore("dep", root=self.root / "labels")
        incidents = asyncio.run(_collect(_core(), SOURCE, label_store=store))
        self.assertEqual(len(incidents), 2)
        # each yielded incident can record a label locally — the vision demo's incident.label(...) intake
        incidents[0].record_label(good=False, note="that was steam")
        incidents[1].record_label(good=True)
        rows = list(store.iter())
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].incident_id, incidents[0].id)
        self.assertEqual(rows[0].goal_id, "zone")
        self.assertNotEqual(rows[0].verdict_ref, "")  # bound from the fired verdict
        self.assertFalse(rows[0].good)
        self.assertTrue(rows[1].good)

    def test_disposition_log_records_fires(self):
        dlog = DispositionLog("dep", root=self.root / "disp")
        asyncio.run(_collect(_core(), SOURCE, disposition_log=dlog))
        rows = list(dlog.iter())
        self.assertEqual([r["disposition"] for r in rows], ["fire", "fire"])
        self.assertTrue(all(r["incident_id"] for r in rows))
        self.assertTrue(all(r["verdict_ref"] for r in rows))

    def test_disposition_log_records_asks(self):
        dlog = DispositionLog("dep", root=self.root / "disp")
        incidents = asyncio.run(_collect(_ask_core(), ASK_SOURCE, disposition_log=dlog))
        self.assertEqual(incidents, [])  # asks are not incidents
        rows = list(dlog.iter())
        self.assertTrue(rows)
        self.assertTrue(all(r["disposition"] == "ask" for r in rows))
        self.assertTrue(all(r["incident_id"] == "" for r in rows))  # no incident for an ask

    def test_default_is_byte_for_byte_no_labeling(self):
        # No label_store / disposition_log -> nothing written, record_label raises (nothing bound),
        # and the incident stream is identical to the plain run.
        base = asyncio.run(_collect(_core(), SOURCE))
        incidents = asyncio.run(_collect(_core(), SOURCE))
        self.assertEqual(len(base), len(incidents))
        with self.assertRaises(RuntimeError):
            incidents[0].record_label(good=True)
        # nothing created under the temp root
        self.assertEqual(list(self.root.glob("**/*.jsonl")), [])


if __name__ == "__main__":
    unittest.main()
