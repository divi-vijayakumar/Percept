"""L-1: the LabelStore / DispositionLog rail — local, keyless, zero-network.

Every write here is plain stdlib file IO under a temp dir; no key, no socket, no HTTP is touched.
Covers: the append-only label JSONL (append/iter/stats), the disposition-stream JSONL, the
``verdict_ref`` join-key, the ``incident.record_label`` zero-network write, and
``RuntimeSession.label`` out-of-band triage.
"""

import json
import tempfile
import unittest
from pathlib import Path

from percept.contracts import Incident, LabelRecord, Verdict
from percept.labelstore import DispositionLog, LabelStore, verdict_ref


class TestLabelStore(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _store(self, deployment_id="dep-a"):
        return LabelStore(deployment_id, root=self.root)

    def _rec(self, **over):
        base = dict(incident_id="i", verdict_ref="v", goal_id="stove", ts=1.0, good=True)
        base.update(over)
        return LabelRecord(**base)

    def test_path_layout(self):
        s = self._store("cam-7")
        self.assertEqual(s.path, self.root / "cam-7.jsonl")

    def test_append_creates_file_and_grows(self):
        s = self._store()
        self.assertFalse(s.path.exists())
        s.append(self._rec(incident_id="a"))
        s.append(self._rec(incident_id="b"))
        lines = s.path.read_text().strip().splitlines()
        self.assertEqual(len(lines), 2)  # append-only, one JSONL row each
        self.assertEqual(json.loads(lines[0])["incident_id"], "a")

    def test_append_persists_frame_ref_locally(self):
        # the store is LOCAL — it keeps frame_ref (the whole point); only egress drops it.
        s = self._store()
        s.append(self._rec(frame_ref="/local/frames/x.png"))
        self.assertEqual(json.loads(s.path.read_text())["frame_ref"], "/local/frames/x.png")

    def test_iter_round_trips_records(self):
        s = self._store()
        s.append(self._rec(incident_id="a", tags=["x"], note="n", source="ask_answer"))
        got = list(s.iter())
        self.assertEqual(len(got), 1)
        self.assertIsInstance(got[0], LabelRecord)
        self.assertEqual(got[0].tags, ["x"])
        self.assertEqual(got[0].source, "ask_answer")

    def test_iter_filters_by_goal(self):
        s = self._store()
        s.append(self._rec(goal_id="stove"))
        s.append(self._rec(goal_id="door"))
        self.assertEqual([r.goal_id for r in s.iter(goal_id="door")], ["door"])

    def test_iter_missing_file_is_empty(self):
        self.assertEqual(list(self._store("nope").iter()), [])

    def test_iter_skips_malformed_line(self):
        s = self._store()
        s.append(self._rec(incident_id="ok"))
        with s.path.open("a") as fh:
            fh.write("{ not json\n")
        self.assertEqual([r.incident_id for r in s.iter()], ["ok"])

    def test_stats(self):
        s = self._store()
        s.append(self._rec(good=True, goal_id="stove", source="api"))
        s.append(self._rec(good=False, goal_id="stove", source="api"))
        s.append(self._rec(good=False, goal_id="door", source="ask_answer"))
        st = s.stats()
        self.assertEqual(st["total"], 3)
        self.assertEqual(st["good"], 1)
        self.assertEqual(st["bad"], 2)
        self.assertEqual(st["by_goal"], {"stove": 2, "door": 1})
        self.assertEqual(st["by_source"], {"api": 2, "ask_answer": 1})


class TestVerdictRef(unittest.TestCase):
    def test_deterministic_and_content_bound(self):
        v = Verdict(satisfied=True, confidence=0.9, evidence="baby awake")
        self.assertEqual(verdict_ref(v), verdict_ref(v))  # stable
        other = Verdict(satisfied=False, confidence=0.9, evidence="baby awake")
        self.assertNotEqual(verdict_ref(v), verdict_ref(other))  # sensitive to the decision
        self.assertEqual(len(verdict_ref(v)), 16)

    def test_never_raises_on_missing_attrs(self):
        self.assertIsInstance(verdict_ref(object()), str)


class TestDispositionLog(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_append_and_iter(self):
        log = DispositionLog("dep-a", root=self.root)
        log.append(incident_id="inc-1", verdict_ref="v1", disposition="fire", ts=10.0)
        log.append(incident_id="", verdict_ref="v2", disposition="ask", ts=11.0)
        rows = list(log.iter())
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], {"incident_id": "inc-1", "verdict_ref": "v1", "disposition": "fire", "ts": 10.0})
        self.assertEqual(rows[1]["disposition"], "ask")
        self.assertEqual(rows[1]["incident_id"], "")  # asks are not incidents


class TestIncidentRecordLabelZeroNetwork(unittest.TestCase):
    """CI gate 1: incident.record_label() writes locally with NO key present — a zero-network test."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_record_label_writes_local_jsonl(self):
        store = LabelStore("dep-a", root=self.root)
        inc = Incident(id="inc-5", concern_id="stove", verdict=Verdict(True, 0.95), ts=7.0)
        inc.bind_labeler(store.append, verdict_ref="vref", goal_id="stove")
        rec = inc.record_label(good=False, note="that was steam", tags=("steam",))
        # written to the local file, nothing else
        rows = list(store.iter())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].incident_id, "inc-5")
        self.assertFalse(rows[0].good)
        self.assertEqual(rows[0].note, "that was steam")
        self.assertEqual(rows[0].source, "api")
        self.assertEqual(rec, rows[0])


class TestRuntimeSessionLabel(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _session(self):
        from percept.runtime.session import RuntimeSession

        return RuntimeSession(object(), label_store=LabelStore("dep-a", root=self.root))

    def test_label_known_incident_delegates_to_record_label(self):
        sess = self._session()
        inc = Incident(id="inc-1", concern_id="stove", verdict=Verdict(True, 0.9), ts=1.0)
        inc.bind_labeler(sess._label_store.append, verdict_ref="vref", goal_id="stove")
        sess.incidents.append(inc)
        sess.label("inc-1", good=False, note="steam")
        rows = list(sess._label_store.iter())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].incident_id, "inc-1")
        self.assertEqual(rows[0].verdict_ref, "vref")
        self.assertEqual(rows[0].goal_id, "stove")
        self.assertFalse(rows[0].good)

    def test_label_unknown_incident_records_minimal(self):
        sess = self._session()
        sess.label("ghost-42", good=True, source="ask_answer")
        rows = list(sess._label_store.iter())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].incident_id, "ghost-42")
        self.assertEqual(rows[0].goal_id, "")  # unknown — recorded what we could
        self.assertTrue(rows[0].good)

    def test_label_without_store_raises(self):
        from percept.runtime.session import RuntimeSession

        sess = RuntimeSession(object())
        with self.assertRaises(RuntimeError):
            sess.label("inc-1", good=True)


if __name__ == "__main__":
    unittest.main()
