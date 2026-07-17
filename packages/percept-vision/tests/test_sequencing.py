"""Card A8: evidence_seq + serialized decision reducer + result_policy (ADR-001 §8).

Proves the mechanism a future worker pool needs, per ADR-001 §8: ``evidence_seq < last_applied``
alone is insufficient (an older in-flight job can still complete after a newer one is admitted). The
reducer must gate on ``latest_admitted_sequence`` and drop a superseded completion as a TRACED,
non-mutating stale drop — never applying it to the gate.
"""

import asyncio
import unittest

from percept.adapters import Verdict
from percept.gate import FIRE, SILENT, Gate
from percept.sequencing import (
    EvidenceSequencer,
    GateDecision,
    ResultPolicy,
    SerializedDecisionReducer,
)


def yes(conf: float = 0.9) -> Verdict:
    return Verdict(satisfied=True, confidence=conf)


def no(conf: float = 0.9) -> Verdict:
    return Verdict(satisfied=False, confidence=conf)


def run(coro):
    return asyncio.run(coro)


class TestEvidenceSequencer(unittest.TestCase):
    """Requirement 1: a monotonic per-key evidence_seq."""

    def test_admit_is_monotonic_per_key(self):
        seq = EvidenceSequencer()
        self.assertEqual(seq.admit("k1"), 1)
        self.assertEqual(seq.admit("k1"), 2)
        self.assertEqual(seq.admit("k1"), 3)

    def test_each_key_has_its_own_counter(self):
        seq = EvidenceSequencer()
        self.assertEqual(seq.admit("k1"), 1)
        self.assertEqual(seq.admit("k2"), 1)  # independent counter, not global
        self.assertEqual(seq.admit("k1"), 2)

    def test_latest_admitted_tracks_dispatch_not_completion(self):
        seq = EvidenceSequencer()
        seq.admit("k1")  # seq 1 dispatched
        seq.admit("k1")  # seq 2 dispatched (1 may still be in flight)
        self.assertEqual(seq.latest_admitted("k1"), 2)

    def test_forget_clears_the_counter(self):
        seq = EvidenceSequencer()
        seq.admit("k1")
        seq.admit("k1")
        seq.forget("k1")
        self.assertEqual(seq.latest_admitted("k1"), 0)
        self.assertEqual(seq.admit("k1"), 1)  # counter restarts clean


class TestOutOfOrderCompletion(unittest.TestCase):
    """(a) An older in-flight job completing AFTER a newer admitted job is dropped as traced stale;
    the gate is never mutated by the stale result."""

    def setUp(self):
        self.gate = Gate(threshold=0.7, fresh_min=0.5)
        self.reducer = SerializedDecisionReducer(self.gate)
        self.key = ("g1", "e1")

    def test_stale_completion_is_dropped_not_applied(self):
        seq_old = self.reducer.admit(self.key)  # job A dispatched first -> seq 1
        seq_new = self.reducer.admit(self.key)  # job B dispatched second -> seq 2 (supersedes A)

        # B (the newer job) completes first and fires the rising edge.
        decision_new = run(self.reducer.submit(self.key, seq_new, yes()))
        self.assertEqual(decision_new.action, FIRE)
        self.assertFalse(decision_new.dropped)

        # A (the older job) now completes, carrying stale evidence (e.g. a confident "no", which
        # would otherwise re-arm the gate and corrupt the edge B just set).
        decision_old = run(self.reducer.submit(self.key, seq_old, no()))
        self.assertTrue(decision_old.dropped)
        self.assertEqual(decision_old.action, SILENT)  # never FIRE/ASK for a dropped result

        # The gate must be UNTOUCHED by the stale drop: it is still disarmed (B's fire stands), so a
        # repeat "yes" stays SILENT, not a fresh FIRE.
        self.assertEqual(self.gate.evaluate(self.key, yes()), SILENT)

    def test_gates_on_latest_admitted_not_last_applied(self):
        """ADR-001 §8's core claim: `seq < last_applied` is NOT enough. Admit 1, 2, 3 — all three
        in flight, NOTHING applied yet, so `last_applied` is still 0. A naive `seq < last_applied`
        check would see `2 < 0 == False` and treat seq 2 as fresh, applying it. The correct gate
        compares against `latest_admitted` (3), correctly recognizing seq 2 as already stale even
        though it would be the FIRST result ever applied for this key."""
        s1 = self.reducer.admit(self.key)
        s2 = self.reducer.admit(self.key)
        s3 = self.reducer.admit(self.key)  # latest_admitted is now 3, none of 1/2/3 applied yet
        self.assertEqual(self.reducer.last_applied(self.key), 0)

        d2 = run(self.reducer.submit(self.key, s2, no()))
        self.assertTrue(d2.dropped)
        self.assertEqual(d2.drop_reason, "superseded_by_newer_admission")

        # Only the TRULY latest admission (s3) may still apply.
        d3 = run(self.reducer.submit(self.key, s3, yes()))
        self.assertFalse(d3.dropped)
        self.assertEqual(d3.action, FIRE)

        # And s1, older still, is dropped too -- once s3 is admitted nothing behind it ever applies,
        # regardless of arrival order.
        d1 = run(self.reducer.submit(self.key, s1, no()))
        self.assertTrue(d1.dropped)

    def test_drop_carries_a_reason(self):
        """(c) Every drop names why."""
        s1 = self.reducer.admit(self.key)
        s2 = self.reducer.admit(self.key)
        run(self.reducer.submit(self.key, s2, yes()))
        decision = run(self.reducer.submit(self.key, s1, no()))
        self.assertTrue(decision.dropped)
        self.assertIsInstance(decision.drop_reason, str)
        self.assertTrue(decision.drop_reason)

    def test_in_order_completion_is_never_dropped(self):
        """Sanity: today's single-flight order (admit -> complete -> admit -> complete) never drops."""
        s1 = self.reducer.admit(self.key)
        d1 = run(self.reducer.submit(self.key, s1, yes()))
        self.assertFalse(d1.dropped)
        s2 = self.reducer.admit(self.key)
        d2 = run(self.reducer.submit(self.key, s2, no()))
        self.assertFalse(d2.dropped)


class TestResultPolicyOrdering(unittest.TestCase):
    """(b) latest_only vs episode_ordered behavior."""

    def setUp(self):
        self.gate = Gate(threshold=0.7, fresh_min=0.5)
        self.key = ("g1", "e1")

    def test_latest_only_drops_regardless_of_episode(self):
        reducer = SerializedDecisionReducer(self.gate, policy=ResultPolicy(ordering="latest_only"))
        s1 = reducer.admit(self.key, episode_id="ep-1")
        s2 = reducer.admit(self.key, episode_id="ep-2")  # a NEW episode, but still supersedes globally
        run(reducer.submit(self.key, s2, yes(), episode_id="ep-2"))
        # s1 belongs to a DIFFERENT (earlier) episode, but latest_only only cares about the key's
        # all-time latest admission, so it is still dropped.
        decision = run(reducer.submit(self.key, s1, no(), episode_id="ep-1"))
        self.assertTrue(decision.dropped)
        self.assertEqual(decision.drop_reason, "superseded_by_newer_admission")

    def test_episode_ordered_does_not_penalize_a_fresh_episode(self):
        reducer = SerializedDecisionReducer(self.gate, policy=ResultPolicy(ordering="episode_ordered"))
        # ep-1 runs two jobs (seq 1, seq 2 in its own admission history).
        ep1_s1 = reducer.admit(self.key, episode_id="ep-1")
        reducer.admit(self.key, episode_id="ep-1")  # ep-1's second job, admitted but not yet resolved
        # A brand-new episode starts; its first job gets a fresh per-key seq (3) but that's the FIRST
        # admission within ep-2, so it must not be treated as behind ep-1's admissions.
        ep2_s1 = reducer.admit(self.key, episode_id="ep-2")
        decision = run(reducer.submit(self.key, ep2_s1, yes(), episode_id="ep-2"))
        self.assertFalse(decision.dropped)  # ep-2's own first job is never "superseded" within ep-2

        # Meanwhile the ep-1 straggler (seq 1) IS superseded -- but only by ep-1's own later admission,
        # not by ep-2's arrival (episode_ordered compares within-episode, not globally-latest).
        stale = run(reducer.submit(self.key, ep1_s1, no(), episode_id="ep-1"))
        self.assertTrue(stale.dropped)
        self.assertEqual(stale.drop_reason, "superseded_within_episode")

    def test_episode_ordered_allows_a_late_but_still_latest_in_episode_result(self):
        reducer = SerializedDecisionReducer(self.gate, policy=ResultPolicy(ordering="episode_ordered"))
        s1 = reducer.admit(self.key, episode_id="ep-1")
        # Nothing newer admitted in ep-1 -> s1 is still ep-1's latest, so it applies even though a
        # later (different) episode has since started elsewhere for the same key.
        reducer.admit(self.key, episode_id="ep-2")
        decision = run(reducer.submit(self.key, s1, yes(), episode_id="ep-1"))
        self.assertFalse(decision.dropped)
        self.assertEqual(decision.action, FIRE)


class TestStaleAfterMsAndSupersessionOverride(unittest.TestCase):
    def setUp(self):
        self.gate = Gate(threshold=0.7, fresh_min=0.5)
        self.key = ("g1", "e1")

    def test_stale_after_ms_drops_a_slow_result_even_without_a_newer_admission(self):
        clock = {"t": 0.0}
        reducer = SerializedDecisionReducer(
            self.gate, policy=ResultPolicy(stale_after_ms=10.0), clock=lambda: clock["t"],
        )
        s1 = reducer.admit(self.key)  # admitted at t=0
        clock["t"] = 0.5  # 500ms later, no newer job admitted
        decision = run(reducer.submit(self.key, s1, yes()))
        self.assertTrue(decision.dropped)
        self.assertEqual(decision.drop_reason, "stale_timeout")

    def test_apply_after_supersession_overrides_the_drop(self):
        reducer = SerializedDecisionReducer(
            self.gate, policy=ResultPolicy(apply_after_supersession=True),
        )
        s1 = reducer.admit(self.key)
        s2 = reducer.admit(self.key)
        run(reducer.submit(self.key, s2, yes()))
        decision = run(reducer.submit(self.key, s1, no()))
        self.assertFalse(decision.dropped)  # applied despite being superseded
        self.assertTrue(decision.applied_after_supersession)
        self.assertEqual(decision.drop_reason, "superseded_by_newer_admission")  # reason still surfaced


class TestSerializationPerKey(unittest.TestCase):
    """Requirement 2: the reducer is serialized per key — concurrent submits for the same key never
    interleave their gate mutations, even if the underlying gate call itself yields control."""

    def test_submit_blocks_on_a_held_per_key_lock(self):
        """Directly exercises the serialization primitive: while the per-key lock is held (as a
        second in-flight `submit()` for the SAME key would hold it across its gate mutation), a new
        `submit()` for that key cannot proceed. This is the structural guarantee requirement 2 asks
        for -- it holds regardless of whether today's synchronous `Gate.evaluate` happens to yield."""
        gate = Gate(threshold=0.7, fresh_min=0.5)
        reducer = SerializedDecisionReducer(gate)
        key = ("g1", "e1")
        s1 = reducer.admit(key)

        async def go():
            lock = reducer._lock_for(key)
            await lock.acquire()
            try:
                task = asyncio.ensure_future(reducer.submit(key, s1, yes()))
                await asyncio.sleep(0.01)
                self.assertFalse(task.done())  # blocked behind the held lock
            finally:
                lock.release()
            decision = await task
            self.assertFalse(decision.dropped)
            self.assertEqual(decision.action, FIRE)

        run(go())

    def test_concurrent_submits_for_same_key_apply_one_at_a_time(self):
        gate = Gate(threshold=0.7, fresh_min=0.5)
        reducer = SerializedDecisionReducer(
            gate, policy=ResultPolicy(apply_after_supersession=True),
        )
        key = ("g1", "e1")
        trace: list[str] = []

        real_evaluate = gate.evaluate

        def traced_evaluate(*args, **kwargs):
            trace.append("enter")
            result = real_evaluate(*args, **kwargs)
            trace.append("exit")
            return result

        gate.evaluate = traced_evaluate  # type: ignore[assignment]

        s1 = reducer.admit(key)
        s2 = reducer.admit(key)

        async def go():
            await asyncio.gather(
                reducer.submit(key, s1, yes()),
                reducer.submit(key, s2, no()),
            )

        run(go())
        # apply_after_supersession=True means BOTH submits reach gate.evaluate (neither is silently
        # skipped); each call fully completes (enter, exit) before the next starts -- proof the
        # per-key lock serialized the two concurrent submits rather than interleaving them.
        self.assertEqual(trace, ["enter", "exit", "enter", "exit"])

    def test_different_keys_do_not_share_a_lock(self):
        gate = Gate(threshold=0.7, fresh_min=0.5)
        reducer = SerializedDecisionReducer(gate)
        s_a = reducer.admit(("g1", "eA"))
        s_b = reducer.admit(("g1", "eB"))

        async def go():
            return await asyncio.gather(
                reducer.submit(("g1", "eA"), s_a, yes()),
                reducer.submit(("g1", "eB"), s_b, yes()),
            )

        results = run(go())
        self.assertTrue(all(isinstance(r, GateDecision) and not r.dropped for r in results))
        self.assertEqual(results[0].action, FIRE)
        self.assertEqual(results[1].action, FIRE)


class TestForget(unittest.TestCase):
    def test_forget_clears_sequencing_and_lock_state(self):
        gate = Gate(threshold=0.7, fresh_min=0.5)
        reducer = SerializedDecisionReducer(gate)
        key = ("g1", "e1")
        s1 = reducer.admit(key)
        run(reducer.submit(key, s1, yes()))
        reducer.forget(key)
        self.assertEqual(reducer.sequencer.latest_admitted(key), 0)
        self.assertEqual(reducer.last_applied(key), 0)
        # a fresh admit after forget restarts the counter clean
        self.assertEqual(reducer.admit(key), 1)


if __name__ == "__main__":
    unittest.main()
