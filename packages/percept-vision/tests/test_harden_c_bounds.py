"""AGENT-C hardening — 24/7 growth bounds on the gate + incident history (R1/R2/R2b/R3).

Every test here targets a leak that "runs, looks correct, and silently does nothing" to memory: an
unbounded dict/list on a path that fires forever. Acceptance items map 1:1 to the task brief.
"""
from __future__ import annotations

import random
import unittest
from types import SimpleNamespace

from percept.adapters import Goal, Verdict
from percept.behavior import BehaviorExecutor, program_of
from percept.entity import EntityGraph
from percept.gate import FIRE, SILENT, Gate
from percept.adapters import Box, Observation
from percept.runtime.session import RuntimeSession

YES = Verdict(True, 0.95, "yes")
NO = Verdict(False, 0.95, "no")


def _interval_goal(cadence_s: float = 1.0) -> Goal:
    g = Goal(id="g", condition="c", say="s")
    g.state.update({"rearm": "interval", "cadence_s": cadence_s})
    return g


# ---- Acceptance 1: interval goal for 10,000s -> gate keys O(goals), not O(time) ------------------
class TestIntervalKeyBound(unittest.TestCase):
    def test_interval_10000s_gate_stays_bounded(self):
        gate = Gate()
        ex = BehaviorExecutor(gate)
        g = _interval_goal(cadence_s=1.0)
        prog = program_of(g)
        for t in range(10_000):
            ex.fire_judged(g, YES, ("g", "e"), prog, freshness=1.0, t=float(t))
        # One goal, one entity -> at most a handful of keys, NEVER one-per-time-bucket.
        self.assertLessEqual(len(gate._armed), 4, f"interval leaked {len(gate._armed)} keys")

    def test_interval_firing_trace_byte_for_byte(self):
        """Acceptance 3 (the one that matters): NEW single-key interval fires identically to the OLD
        one-key-per-bucket synthesis, over a randomized yes/no/unsure stream."""
        rng = random.Random(1234)
        verdicts = []
        for _ in range(2000):
            r = rng.random()
            verdicts.append(YES if r < 0.6 else NO if r < 0.85 else Verdict(True, 0.4, "unsure"))

        cadence = 7.0
        # OLD behavior: reconstruct the pre-fix path — a fresh gate key per (entity, time-bucket).
        old_gate = Gate()
        old_trace = []
        for i, v in enumerate(verdicts):
            key = ("g", "e", int(i // cadence))  # the leaky synthesized key
            old_trace.append(old_gate.evaluate(key, v, freshness=1.0))

        # NEW behavior: the executor's bounded single-key + rearm-on-rollover.
        new_gate = Gate()
        ex = BehaviorExecutor(new_gate)
        g = _interval_goal(cadence_s=cadence)
        prog = program_of(g)
        new_trace = []
        for i, v in enumerate(verdicts):
            f = ex.fire_judged(g, v, ("g", "e"), prog, freshness=1.0, t=float(i))
            new_trace.append(f.action if f is not None else SILENT)

        self.assertEqual(new_trace, old_trace, "interval firing trace diverged from legacy synthesis")
        self.assertLessEqual(len(new_gate._armed), 4)


# ---- Acceptance 2: 10,000 distinct entities ingested+evicted -> gate stays bounded ---------------
class TestEntityDeathEviction(unittest.TestCase):
    def test_evicted_entity_drops_its_gate_keys(self):
        gate = Gate()
        graph = EntityGraph(max_entities=64)
        graph.on_evict = gate.reset_entity  # the R3 wiring Percept installs
        for i in range(10_000):
            label = f"thing{i}"  # a unique label -> a brand-new entity every time
            ents = graph.ingest([Observation(label=label, box=Box(0.1, 0.1, 0.2, 0.2), confidence=0.9)])
            eid = ents[0].id
            gate.evaluate(("g", eid), YES, freshness=1.0)  # a fire on that entity -> a gate key
        # The graph caps at 64; every eviction must have swept the entity's gate key with it.
        self.assertLessEqual(len(graph), 64)
        self.assertLessEqual(len(gate._armed), 64, f"gate leaked {len(gate._armed)} dead-entity keys")

    def test_reset_entity_leaves_other_entities(self):
        gate = Gate()
        gate.evaluate(("g", "e1"), YES, freshness=1.0)
        gate.evaluate(("g", "e2"), YES, freshness=1.0)
        gate.evaluate(("g", "__resolved__"), YES, freshness=1.0)  # a non-entity singleton must survive
        gate.reset_entity("e1")
        self.assertNotIn(("g", "e1"), gate._armed)
        self.assertIn(("g", "e2"), gate._armed)
        self.assertIn(("g", "__resolved__"), gate._armed)


# ---- Acceptance 4 + 5: TTL evicts ordinary keys (re-arm) but EXEMPTS never-goals ----------------
class TestKeyTTL(unittest.TestCase):
    def test_ordinary_key_evicted_by_ttl_rearms(self):
        gate = Gate(key_ttl_s=100.0)
        # fire once at t=0
        self.assertEqual(gate.evaluate(("g", "e"), YES, freshness=1.0, now=0.0), FIRE)
        self.assertEqual(gate.evaluate(("g", "e"), YES, freshness=1.0, now=1.0), SILENT)  # armed, refractory
        # a long-untouched key is swept; the NEXT confident-yes re-arms and fires (INTENDED for ordinary goals)
        self.assertEqual(gate.evaluate(("g", "other"), YES, freshness=1.0, now=500.0), FIRE)  # advances clock+sweeps
        self.assertEqual(gate.evaluate(("g", "e"), YES, freshness=1.0, now=501.0), FIRE)

    def test_never_goal_past_ttl_does_not_refire(self):
        """THE TRAP: a never-goal fires exactly once, ever. A TTL drop that re-arms it silently converts
        a never-goal into an interval goal. It must be exempt."""
        gate = Gate(key_ttl_s=100.0)
        ex = BehaviorExecutor(gate)
        g = Goal(id="g", condition="c", say="s")
        g.state["rearm"] = "never"
        prog = program_of(g)
        actions = []
        # fire at t=0, then keep evaluating far past the TTL; it must never fire twice.
        for t in (0.0, 1.0, 200.0, 5000.0, 100000.0):
            f = ex.fire_judged(g, YES, ("g", "e"), prog, freshness=1.0, t=t)
            actions.append(f.action if f is not None else SILENT)
        self.assertEqual(actions.count(FIRE), 1, f"never-goal re-fired: {actions}")

    def test_max_keys_lru_backstop(self):
        gate = Gate(max_keys=10)
        for i in range(1000):
            gate.evaluate(("g", f"e{i}"), YES, freshness=1.0, now=float(i))
        self.assertLessEqual(len(gate._armed), 10)


# ---- Gate snapshot/restore round-trips the new eviction state ----------------------------------
class TestGateSnapshotWithEviction(unittest.TestCase):
    def test_snapshot_restore_carries_touched_and_no_evict(self):
        gate = Gate(key_ttl_s=100.0)
        ex = BehaviorExecutor(gate)
        g = Goal(id="g", condition="c", say="s")
        g.state["rearm"] = "never"
        ex.fire_judged(g, YES, ("g", "e"), program_of(g), freshness=1.0, t=0.0)
        snap = gate.snapshot()
        restored = Gate(key_ttl_s=100.0)
        restored.restore(snap)
        self.assertEqual(restored._armed, gate._armed)
        self.assertIn(("g", "e"), restored._no_evict)  # the never-key stays exempt across a restart

    def test_restore_of_legacy_snapshot_is_noop_safe(self):
        gate = Gate()
        gate.restore({"armed": [[["g", "e"], False]]})  # a pre-eviction snapshot (no touched/no_evict)
        self.assertEqual(gate._armed, {("g", "e"): False})


# ---- Acceptance 6: RuntimeSession.incidents bounded under sustained fire ------------------------
class TestIncidentsBound(unittest.TestCase):
    def test_incidents_stay_bounded(self):
        sess = RuntimeSession(core=SimpleNamespace())
        for i in range(10_000):
            sess.incidents.append(SimpleNamespace(id=f"inc{i}", message=f"m{i}"))
        cap = sess.max_incidents
        self.assertLessEqual(len(sess.incidents), cap)
        # the most-recent history is what label()/pulse need — it must be retained
        ids = [getattr(i, "id", None) for i in sess.incidents]
        self.assertEqual(ids[-1], "inc9999")

    def test_max_incidents_configurable(self):
        sess = RuntimeSession(core=SimpleNamespace(), max_incidents=5)
        for i in range(50):
            sess.incidents.append(SimpleNamespace(id=f"inc{i}"))
        self.assertEqual(len(sess.incidents), 5)
        self.assertEqual([i.id for i in sess.incidents], [f"inc{i}" for i in range(45, 50)])


# ---- R2b: episode + resolved-answer dedup keys bounded, firing preserved -----------------------
class TestEpisodeKeyBound(unittest.TestCase):
    def test_every_new_episode_fires_once_with_one_key(self):
        gate = Gate()
        ex = BehaviorExecutor(gate)
        g = Goal(id="g", condition="c", say="s")
        g.state["type"] = "count"
        prog = program_of(g)
        fires = 0
        for n in range(1, 5001):  # 5000 distinct reps
            f = ex.fire_episode(g, prog, n=n, is_new=True)
            if f is not None and f.action == FIRE:
                fires += 1
        self.assertEqual(fires, 5000)  # every new episode still fires exactly once
        self.assertLessEqual(len(gate._armed), 2, "episode leaked a key per rep")


def _resolved_trace(seq, *, bounded):
    """Drive the resolved-answer dedup two ways: bounded=False replays the OLD answer-in-key path;
    bounded=True the NEW single-key + rearm-on-change path. Returns the FIRE/SILENT trace."""
    gate = Gate()
    out, last = [], None
    for ans, ready in seq:
        v = Verdict(satisfied=ready, confidence=0.95, evidence=ans)
        if bounded:
            key = ("g", "__resolved__")
            if last is not None and ans != last:
                gate.rearm(key)
            last = ans
            out.append(gate.evaluate(key, v, no_evict=True))
        else:
            out.append(gate.evaluate(("g", "__resolved__", ans), v))
    return out


class TestResolvedAnswerKeyBound(unittest.TestCase):
    def test_monotonic_answers_are_byte_for_byte(self):
        # The realistic path: the running answer only grows/rewrites forward (never re-emits an OLD exact
        # string), so distinct-answer resolution is byte-for-byte with the legacy answer-in-key path.
        seq = [("", False), ("a", False), ("ab", True), ("ab", True), ("abc", True), ("abcd", True)]
        self.assertEqual(_resolved_trace(seq, bounded=True), _resolved_trace(seq, bounded=False))

    def test_confident_no_rearm_same_answer_byte_for_byte(self):
        seq = [("ab", True), ("ab", False), ("ab", True)]
        self.assertEqual(_resolved_trace(seq, bounded=True), _resolved_trace(seq, bounded=False))

    def test_ABA_oscillation_is_the_ONE_documented_divergence(self):
        # KNOWN + INTENTIONAL: bounded memory cannot preserve a per-answer refractory across an
        # intervening different answer. Legacy silents on the return-to-A (A already fired, never re-armed);
        # the bounded path treats a genuinely new A<-B<-A transition as a fresh resolution and re-fires. This
        # is unreachable under a monotonically-accumulating running answer; pinned here so it is never silent.
        seq = [("a", True), ("b", True), ("a", True)]
        self.assertEqual(_resolved_trace(seq, bounded=False), [FIRE, FIRE, SILENT])
        self.assertEqual(_resolved_trace(seq, bounded=True), [FIRE, FIRE, FIRE])


if __name__ == "__main__":
    unittest.main()
