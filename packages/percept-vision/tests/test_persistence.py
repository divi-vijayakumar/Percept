"""L1: persistence — kv round-trip, survives a 'restart', and full snapshot/restore of state."""

import os
import tempfile
import unittest

from percept.adapters import Box, Goal, Observation, Verdict
from percept.agent import Percept
from percept.consent import ActionRecord
from percept.persistence import InMemoryPersistence, SqlitePersistence


class TestBackends(unittest.TestCase):
    def test_inmemory_round_trip(self):
        p = InMemoryPersistence()
        p.save("k", {"a": 1, "b": [1, 2, 3]})
        self.assertEqual(p.load("k"), {"a": 1, "b": [1, 2, 3]})
        self.assertIsNone(p.load("missing"))

    def test_sqlite_round_trip(self):
        p = SqlitePersistence(":memory:")
        p.save("k", {"x": "y"})
        self.assertEqual(p.load("k"), {"x": "y"})

    def test_sqlite_survives_restart(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "state.db")
            p1 = SqlitePersistence(path)
            p1.save("percept", {"goals": ["g1"]})
            p1.close()
            p2 = SqlitePersistence(path)  # a fresh connection == a process restart
            self.assertEqual(p2.load("percept"), {"goals": ["g1"]})
            p2.close()


def _populated_agent():
    a = Percept.create(discover_plugins=False)
    a.add_goal(Goal(id="g1", condition="user drinks coffee", say="careful"))
    a.graph.ingest([Observation("cup", Box(0.4, 0.4, 0.2, 0.2), 0.9, {"state": "full"})], t=1.0)
    a.graph.assert_attribute("e1", "contents", "tea", t=1.0)
    a.action_log.record(ActionRecord(kind="fire", goal_id="g1", detail={"text": "hi"}, t=1.0))
    return a


class TestSnapshotRestore(unittest.TestCase):
    def test_snapshot_restore_reproduces_state(self):
        a = _populated_agent()
        snap = a.snapshot()

        b = Percept.create(discover_plugins=False)
        b.restore(snap)
        self.assertEqual(b.goals.get("g1").condition, "user drinks coffee")
        e = b.graph.get("e1")
        self.assertEqual(e.attributes["contents"].value, "tea")
        self.assertEqual(e.attributes["contents"].source, "asserted")
        self.assertEqual(len(b.action_log), 1)

    def test_restored_graph_continues_id_sequence(self):
        a = _populated_agent()
        b = Percept.create(discover_plugins=False)
        b.restore(a.snapshot())
        # a new detection must not collide with the restored e1
        [e2] = b.graph.ingest([Observation("mug", Box(0.0, 0.0, 0.1, 0.1), 0.9)], t=2.0)
        self.assertEqual(e2.id, "e2")

    def test_save_then_load_through_sqlite_restart(self):
        a = _populated_agent()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "state.db")
            p1 = SqlitePersistence(path)
            a.save_state(p1)
            p1.close()

            # "restart": a fresh agent + a fresh connection to the same file
            b = Percept.create(discover_plugins=False)
            p2 = SqlitePersistence(path)
            self.assertTrue(b.load_state(p2))
            p2.close()
            self.assertIsNotNone(b.goals.get("g1"))
            self.assertEqual(b.graph.get("e1").attributes["contents"].value, "tea")

    def test_load_state_false_when_empty(self):
        b = Percept.create(discover_plugins=False)
        self.assertFalse(b.load_state(InMemoryPersistence()))

    def test_corrupt_v2_deadline_rolls_back_every_populated_store(self):
        destination = _populated_agent()
        destination.gate.evaluate(("g1", "e1"), Verdict(True, 0.99, "seen"))
        binding = object()
        destination._watch_runtime_bindings["g1"] = (binding, None)
        destination._watch_runtime_keys["g1"] = "application"
        destination._judge_bindings["g1"] = binding
        destination._judge_binding_tokens["g1"] = id(binding)
        destination._judge_requesting_goal_ids.add("g1")
        destination._audio_judge_bindings["g1"] = binding
        destination._audio_judge_binding_tokens["g1"] = binding
        destination._watchplan_applied_hashes = {"application": ("g1", binding)}
        destination._restore_health_log.append(("existing", {"preserve": True}))
        goals_before = destination.goals.snapshot()
        entities_before = destination.graph.snapshot()
        gate_before = destination.gate.snapshot()
        action_log_before = destination.action_log
        action_records_before = destination.action_log.entries()
        watch_runtime_before = dict(destination._watch_runtime_bindings)
        watch_keys_before = dict(destination._watch_runtime_keys)
        judge_bindings_before = dict(destination._judge_bindings)
        judge_tokens_before = dict(destination._judge_binding_tokens)
        judge_requesting_before = set(destination._judge_requesting_goal_ids)
        audio_bindings_before = dict(destination._audio_judge_bindings)
        audio_tokens_before = dict(destination._audio_judge_binding_tokens)
        ledger_before = dict(destination._watchplan_applied_hashes)
        restore_health_before = list(destination._restore_health_log)

        source = Percept.create(discover_plugins=False, clock=lambda: 100.0)
        deadline = Goal(
            id="deadline",
            condition="the package has arrived",
            say="still waiting",
            state={"rearm": "deadline", "deadline_after_s": 60.0},
        )
        source.add_goal(deadline)
        corrupt = source.snapshot()
        corrupt["goals"][0]["goal"]["state"]["deadline_at_epoch"] = "not-an-epoch"

        with self.assertRaises(TypeError):
            destination.restore(corrupt)

        self.assertEqual(destination.goals.snapshot(), goals_before)
        self.assertEqual(destination.graph.snapshot(), entities_before)
        self.assertEqual(destination.gate.snapshot(), gate_before)
        self.assertIs(destination.action_log, action_log_before)
        self.assertEqual(destination.action_log.entries(), action_records_before)
        self.assertEqual(destination._watch_runtime_bindings, watch_runtime_before)
        self.assertEqual(destination._watch_runtime_keys, watch_keys_before)
        self.assertEqual(destination._judge_bindings, judge_bindings_before)
        self.assertEqual(destination._judge_binding_tokens, judge_tokens_before)
        self.assertEqual(destination._judge_requesting_goal_ids, judge_requesting_before)
        self.assertEqual(destination._audio_judge_bindings, audio_bindings_before)
        self.assertEqual(destination._audio_judge_binding_tokens, audio_tokens_before)
        self.assertEqual(destination._watchplan_applied_hashes, ledger_before)
        self.assertEqual(destination._restore_health_log, restore_health_before)
        self.assertEqual(
            destination.metrics.snapshot().get("snapshot.restore_rolled_back"), 1
        )


if __name__ == "__main__":
    unittest.main()
