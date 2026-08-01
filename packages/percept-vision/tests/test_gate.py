"""L1: the three-state gate matrix — deterministic, no backends."""

import typing
import unittest

import percept.gate as gate_mod
from percept.adapters import Verdict
from percept.gate import ASK, FIRE, SILENT, Gate, GateAction


def yes(conf=0.9):
    return Verdict(satisfied=True, confidence=conf)


def no(conf=0.9):
    return Verdict(satisfied=False, confidence=conf)


class TestGate(unittest.TestCase):
    def setUp(self):
        self.gate = Gate(threshold=0.7, fresh_min=0.5)
        self.k = ("g1", "e1")

    def test_fires_on_rising_edge(self):
        self.assertEqual(self.gate.evaluate(self.k, yes()), FIRE)

    def test_silent_on_repeat_while_true(self):
        self.gate.evaluate(self.k, yes())
        self.assertEqual(self.gate.evaluate(self.k, yes()), SILENT)  # already fired, still true

    def test_confident_no_is_silent_and_rearms(self):
        self.assertEqual(self.gate.evaluate(self.k, yes()), FIRE)
        self.assertEqual(self.gate.evaluate(self.k, no()), SILENT)  # re-arms
        self.assertEqual(self.gate.evaluate(self.k, yes()), FIRE)  # fires again after re-arm

    def test_low_confidence_asks_and_never_fires(self):
        self.assertEqual(self.gate.evaluate(self.k, yes(conf=0.4)), ASK)
        # asking did not disarm — a later confident yes still fires
        self.assertEqual(self.gate.evaluate(self.k, yes()), FIRE)

    def test_stale_supporting_fact_asks(self):
        self.assertEqual(self.gate.evaluate(self.k, yes(), freshness=0.0), ASK)

    def test_refractory_single_fire_per_key(self):
        self.assertEqual(self.gate.evaluate(self.k, yes()), FIRE)
        self.assertEqual(self.gate.evaluate(self.k, yes()), SILENT)
        self.assertEqual(self.gate.evaluate(self.k, yes()), SILENT)

    def test_new_entity_key_gets_fresh_edge(self):
        self.gate.evaluate(("g1", "e1"), yes())  # e1 fires
        self.assertEqual(self.gate.evaluate(("g1", "e2"), yes()), FIRE)  # e2 is fresh

    def test_reset_goal_clears_edge(self):
        self.gate.evaluate(self.k, yes())  # fired, disarmed
        self.gate.reset_goal("g1")
        self.assertEqual(self.gate.evaluate(self.k, yes()), FIRE)  # edge reopened

    def test_ask_at_threshold_boundary(self):
        # exactly at threshold is confident (>=), just below asks
        self.assertEqual(self.gate.evaluate(("g", "a"), yes(conf=0.7)), FIRE)
        self.assertEqual(self.gate.evaluate(("g", "b"), yes(conf=0.69)), ASK)


class TestGateActionMembers(unittest.TestCase):
    """PIN reality: the gate is a THREE-verdict system today, not the five-verdict design.

    OBSERVE and LOG do NOT exist. This test fails the moment ``GateAction`` diverges from
    {fire, ask, silent}, forcing any future five-verdict build (or any external doc claiming one) to
    reconcile with the source. Do NOT relax this to "make it pass" — extend the gate first, then the
    pin. Whether to build the five-verdict gate at all is an escalated human decision, not a code fix.
    """

    def test_gate_action_is_exactly_fire_ask_silent(self):
        self.assertEqual(set(typing.get_args(GateAction)), {"fire", "ask", "silent"})

    def test_constants_match_the_literal(self):
        # the module constants and the Literal members are the same three strings, no more
        self.assertEqual({FIRE, ASK, SILENT}, set(typing.get_args(GateAction)))
        self.assertEqual((FIRE, ASK, SILENT), ("fire", "ask", "silent"))

    def test_no_observe_or_log_verdicts(self):
        # explicit guard against the design leaking into the system as bare module attributes
        self.assertFalse(hasattr(gate_mod, "OBSERVE"))
        self.assertFalse(hasattr(gate_mod, "LOG"))


# ---- key growth bound: the `_bound` LRU table (gate.py:441-470) ----------------------------------
class TestBoundLruOrdering(unittest.TestCase):
    """A direct table for the LRU half of the growth bound.

    `test_harden_c_bounds.py` proves only that a default gate stays bounded *somehow*. Nothing pins
    the three decisions inside `_bound` that decide WHICH key dies: the ``pending`` slot reservation
    (`:433`), the ``_touched``-order sort with an unstamped key at ``-inf`` (`:438-444`), and the
    ``k != current`` self-eviction guard (`:436`). Each of those is a route by which a live edge is
    dropped and a continuing condition re-notified.
    """

    def setUp(self):
        self.evicted: list[tuple] = []

    def _gate(self, max_keys):
        # key_ttl_s=None isolates the LRU branch; no clock, so only an explicit `now` stamps a key.
        return Gate(
            max_keys=max_keys,
            key_ttl_s=None,
            on_evict=lambda key, reason: self.evicted.append((key, reason)),
        )

    def test_ceiling_settles_at_exactly_max_keys_never_one_key_high(self):
        """The `pending` reservation at `:433` is what makes the ceiling *max_keys*, not max_keys+1.

        Without it the current key is armed after the sweep and the population parks one key above
        the configured bound forever — a bound that is quietly off by one on every 24/7 watch.
        """
        gate = self._gate(3)
        sizes = []
        for i in range(8):
            gate.evaluate(("g", f"e{i}"), yes(), now=float(i))
            sizes.append(len(gate._armed))
            self.assertLessEqual(len(gate._armed), 3, f"ceiling breached at step {i}: {sizes}")
        self.assertEqual(sizes, [1, 2, 3, 3, 3, 3, 3, 3])
        # the ceiling is reached and held, never overshot by the current key's own slot
        self.assertEqual(len(gate._armed), 3)
        self.assertEqual([reason for _, reason in self.evicted], ["max_keys"] * 5)

    def test_re_evaluating_a_key_already_at_the_ceiling_reserves_nothing_and_evicts_nothing(self):
        """`pending = 0` when the current key is already armed — a steady-state watch at the ceiling
        must not shed a neighbour on every tick just because it is full."""
        gate = self._gate(3)
        for i in range(3):
            gate.evaluate(("g", f"e{i}"), yes(), now=float(i))
        self.assertEqual(self.evicted, [])
        for tick in range(5):
            gate.evaluate(("g", "e1"), yes(), now=10.0 + tick)
        self.assertEqual(self.evicted, [], "a re-evaluation at the ceiling evicted a live key")
        self.assertEqual(len(gate._armed), 3)

    def test_least_recently_touched_sheds_first_not_least_recently_inserted(self):
        """The sort key is `_touched`, not insertion order. A key that is still being evaluated every
        tick must outlive one that went quiet, or the bound preferentially kills LIVE edges."""
        gate = self._gate(3)
        for i in range(3):
            gate.evaluate(("g", f"e{i}"), yes(), now=float(i))
        # e0 was inserted first but is touched most recently -> e1 is now the coldest key
        self.assertEqual(gate.evaluate(("g", "e0"), yes(), now=3.0), SILENT)  # already fired
        gate.evaluate(("g", "e3"), yes(), now=4.0)
        self.assertEqual(self.evicted, [(("g", "e1"), "max_keys")])
        self.assertEqual(set(gate._armed), {("g", "e0"), ("g", "e2"), ("g", "e3")})

    def test_an_unstamped_key_sorts_oldest_and_sheds_before_any_stamped_key(self):
        """The unstamped-sorts-oldest fallback at `:438-444`. A caller that passes neither ``now`` nor a
        clock never stamps its keys, so those keys are always the first victims — the documented
        consequence of driving the gate off-clock, pinned here rather than discovered in production."""
        gate = self._gate(3)
        gate.evaluate(("g", "stamped_early"), yes(), now=1.0)
        gate.evaluate(("g", "unstamped"), yes())  # no now, no clock -> never enters `_touched`
        gate.evaluate(("g", "stamped_late"), yes(), now=2.0)
        self.assertNotIn(("g", "unstamped"), gate._touched)
        self.assertEqual(self.evicted, [])

        gate.evaluate(("g", "newcomer"), yes(), now=3.0)
        # inserted SECOND, but unstamped -> sorts below stamped_early's 1.0 and dies first
        self.assertEqual(self.evicted, [(("g", "unstamped"), "max_keys")])
        self.assertEqual(
            set(gate._armed),
            {("g", "stamped_early"), ("g", "stamped_late"), ("g", "newcomer")},
        )

    def test_the_key_being_evaluated_is_never_its_own_victim(self):
        """`k != current` at `:436`. The current key is stamped before the sweep, so it is normally the
        freshest — except off-clock, where it stays unstamped and would sort into the victim list
        first. A gate that evicted the key it is deciding would re-arm mid-occurrence and re-notify."""
        gate = self._gate(2)
        current, stamped, other_unstamped = ("g", "cur"), ("g", "a"), ("g", "b")
        gate.evaluate(current, yes())              # unstamped, armed first
        gate.evaluate(stamped, yes(), now=10.0)
        gate.rearm(other_unstamped)                # `rearm` writes `_armed` without running `_bound`
        self.assertEqual(len(gate._armed), 3)      # deliberately over the ceiling, so a sweep is due

        # Re-evaluate the current key off-clock: it is unstamped (-inf) AND the oldest insertion, so
        # without the guard it would be first in the sorted victim list and evict itself.
        self.assertEqual(gate.evaluate(current, yes()), SILENT)  # edge still spent, not re-armed
        self.assertIn(current, gate._armed)
        self.assertEqual(self.evicted, [(other_unstamped, "max_keys")])
        self.assertEqual(set(gate._armed), {current, stamped})


class TestBoundTtlCutoff(unittest.TestCase):
    """The TTL comparison at `:429` is strict ``<`` against ``cutoff = ref - key_ttl_s``.

    Which side of that boundary is inclusive decides whether a key exactly one TTL old survives. An
    off-by-epsilon here drops a live edge and re-notifies a condition that never stopped being true.
    """

    def setUp(self):
        self.evicted: list[tuple] = []

    def _gate(self):
        # max_keys=None isolates the TTL branch from the LRU backstop.
        return Gate(
            max_keys=None,
            key_ttl_s=100.0,
            on_evict=lambda key, reason: self.evicted.append((key, reason)),
        )

    def test_a_stamp_exactly_at_the_cutoff_survives(self):
        gate = self._gate()
        self.assertEqual(gate.evaluate(("g", "K"), yes(), now=0.0), FIRE)
        # sweeping at now=100.0 puts cutoff at exactly 0.0; `0.0 < 0.0` is False -> K lives
        gate.evaluate(("g", "other"), yes(), now=100.0)
        self.assertEqual(self.evicted, [])
        self.assertIn(("g", "K"), gate._armed)
        # ...and the surviving edge is still spent: no re-notification of a continuing condition
        self.assertEqual(gate.evaluate(("g", "K"), yes(), now=100.0), SILENT)

    def test_a_stamp_one_epsilon_older_than_the_cutoff_is_evicted_with_reason_ttl(self):
        gate = self._gate()
        self.assertEqual(gate.evaluate(("g", "K"), yes(), now=0.0), FIRE)
        gate.evaluate(("g", "other"), yes(), now=100.001)  # cutoff 0.001; 0.0 < 0.001 -> swept
        self.assertEqual(self.evicted, [(("g", "K"), "ttl")])
        self.assertNotIn(("g", "K"), gate._armed)
        # the user-visible cost of crossing the boundary: the spent edge is gone, so the same
        # continuing condition fires a second time
        self.assertEqual(gate.evaluate(("g", "K"), yes(), now=100.002), FIRE)

    def test_the_sweep_needs_a_time_reference_at_all(self):
        """No ``now`` and no clock means no `ref`, so the TTL branch never runs — a clockless caller
        is protected by `max_keys` alone, exactly as the module docstring claims."""
        gate = self._gate()  # max_keys is None here, so this gate has NO effective bound off-clock
        for i in range(200):
            gate.evaluate(("g", f"e{i}"), yes())
        self.assertEqual(self.evicted, [])
        self.assertEqual(len(gate._armed), 200)


class TestNoEvictExemption(unittest.TestCase):
    """`no_evict` keys (never-goals, the per-goal absence/resolved singletons) are exempt from BOTH
    bounds. `test_harden_c_bounds.py::test_pinned_no_evict_keys_survive_the_default_bound` pins the
    LRU half only — it passes no ``now``, so the TTL sweep never runs there. The failure this guards:
    a dropped never-key re-arms, and a fire-once-ever goal silently becomes an interval goal.
    """

    NEVER = ("g", "__never__")

    def setUp(self):
        self.evicted: list[tuple] = []

    def _gate(self):
        return Gate(
            max_keys=2,
            key_ttl_s=100.0,
            on_evict=lambda key, reason: self.evicted.append((key, reason)),
        )

    def test_a_no_evict_key_survives_a_flood_past_max_keys(self):
        gate = self._gate()
        self.assertEqual(gate.evaluate(self.NEVER, yes(), now=0.0, no_evict=True), FIRE)
        for i in range(6):
            gate.evaluate(("g", f"e{i}"), yes(), now=1.0 + i)
        self.assertIn(self.NEVER, gate._armed)
        self.assertIn(self.NEVER, gate._no_evict)
        self.assertNotIn(self.NEVER, [key for key, _ in self.evicted])
        self.assertEqual([reason for _, reason in self.evicted], ["max_keys"] * 5)
        # fire-once-ever still holds
        self.assertEqual(gate.evaluate(self.NEVER, yes(), now=100.0, no_evict=True), SILENT)

    def test_a_no_evict_key_survives_a_ttl_sweep_far_past_its_stamp(self):
        gate = self._gate()
        self.assertEqual(gate.evaluate(self.NEVER, yes(), now=0.0, no_evict=True), FIRE)
        # one thousand TTLs later: any ordinary key stamped at 0.0 would be long gone
        gate.evaluate(("g", "much_later"), yes(), now=100_000.0)
        self.assertIn(self.NEVER, gate._armed)
        self.assertIn(self.NEVER, gate._no_evict)
        self.assertNotIn(self.NEVER, [key for key, _ in self.evicted])
        self.assertEqual(gate.evaluate(self.NEVER, yes(), now=100_001.0, no_evict=True), SILENT)

    def test_an_ordinary_key_with_the_same_stamp_is_swept_by_that_same_sweep(self):
        """The control: the exemption is doing the work, not a quiet TTL branch."""
        gate = self._gate()
        gate.evaluate(("g", "ordinary"), yes(), now=0.0)
        gate.evaluate(("g", "much_later"), yes(), now=100_000.0)
        self.assertIn((("g", "ordinary"), "ttl"), self.evicted)
        self.assertNotIn(("g", "ordinary"), gate._armed)


class TestEvictionIsNeverSilent(unittest.TestCase):
    """With NO `on_evict` wired the default path must still log the drop and its reason
    (`gate.py:439`). A silent eviction is a re-notification whose cause is unrecoverable from the
    trace."""

    def test_a_max_keys_drop_is_logged_with_its_reason(self):
        gate = Gate(max_keys=1, key_ttl_s=None)  # no on_evict
        with self.assertLogs("percept.gate", level="INFO") as caught:
            gate.evaluate(("g", "a"), yes(), now=0.0)
            gate.evaluate(("g", "b"), yes(), now=1.0)
        self.assertTrue(
            any("gate.evict" in line for line in caught.output), caught.output
        )
        reasons = [
            record.fields.get("reason")
            for record in caught.records
            if record.getMessage() == "gate.evict"
        ]
        self.assertEqual(reasons, ["max_keys"])

    def test_a_ttl_drop_is_logged_with_its_reason(self):
        gate = Gate(max_keys=None, key_ttl_s=10.0)  # no on_evict
        with self.assertLogs("percept.gate", level="INFO") as caught:
            gate.evaluate(("g", "a"), yes(), now=0.0)
            gate.evaluate(("g", "b"), yes(), now=100.0)
        self.assertTrue(
            any("gate.evict" in line for line in caught.output), caught.output
        )
        evicts = [record for record in caught.records if record.getMessage() == "gate.evict"]
        self.assertEqual([record.fields.get("reason") for record in evicts], ["ttl"])
        # the key itself travels with the reason, or the line names no victim
        self.assertEqual(evicts[0].fields.get("key"), repr(("g", "a")))


class TestResetEntitySweepsCompanionState(unittest.TestCase):
    """`reset_entity` (`gate.py:774`) is wired to EntityGraph eviction: when a subject dies, its gate
    state must die with it. A dwell-pending key is the case that exposes the seam — the dwell
    branches (`:607-632`) write `dwell_since`/`dwell_last_supported`/`touched` but NEVER `armed`,
    and `reset_entity` iterates `_armed` alone.
    """

    KEY = ("g1", "e1")
    COMPANION_MAPS = ("_dwell_since", "_dwell_last_supported", "_touched")

    def _gate_with_a_dwell_pending_key(self):
        gate = Gate()  # shipped defaults; hold_s makes this a duration goal
        self.assertEqual(gate.evaluate(self.KEY, yes(), hold_s=5.0, now=0.0), SILENT)
        self.assertNotIn(self.KEY, gate._armed, "precondition: the dwell branch writes no `_armed`")
        for name in self.COMPANION_MAPS:
            self.assertIn(self.KEY, getattr(gate, name), f"precondition: {name} holds the key")
        return gate

    def _assert_fully_swept(self, gate, after):
        for name in self.COMPANION_MAPS:
            self.assertNotIn(
                self.KEY, getattr(gate, name), f"{after} left an orphan entry in {name}"
            )

    def test_reset_goal_sweeps_every_companion_map(self):
        """The passing control. `reset_goal` (`:666-674`) sweeps the whole keyspace, not just the armed keys,
        so a re-minted key starts its duration clock from scratch."""
        gate = self._gate_with_a_dwell_pending_key()
        gate.reset_goal("g1")
        self._assert_fully_swept(gate, "reset_goal")
        # 10s later a re-minted subject must still serve its own 5s of proven continuity
        self.assertEqual(gate.evaluate(self.KEY, yes(), hold_s=5.0, now=10.0), SILENT)

    def test_reset_entity_sweeps_every_companion_map(self):
        """BUG-D-01-01, closed. `reset_entity` used to comprehend over `_armed` ALONE, so a
        dwell-pending key — which the dwell branches populate without ever writing `_armed` — matched
        nothing and orphaned its state. One record per key makes `_drop` total by construction, so
        this and its `reset_goal` sibling now filter the same keyspace and cannot disagree."""
        gate = self._gate_with_a_dwell_pending_key()
        gate.reset_entity("e1")
        self._assert_fully_swept(gate, "reset_entity")
        # The consequence of the orphan: the dead subject's dwell clock is inherited, so a re-minted
        # id fires on its FIRST observation — a "held for 5s" claim backed by one sample.
        self.assertEqual(gate.evaluate(self.KEY, yes(), hold_s=5.0, now=10.0), SILENT)


if __name__ == "__main__":
    unittest.main()
