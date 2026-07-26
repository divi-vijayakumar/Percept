"""— consequence gate GA: FIRE/ASK/SILENT + rising-edge/refractory as a STABLE surface.

The decision model already worked; this suite protects its public surface. Three things are asserted here, and they
are the three things "GA" has to mean for a decision surface an integrator builds on:

1. **The vocabulary is closed, typed and self-consistent.** :data:`~percept.gate.GATE_REASONS` is the
   complete WHY set; each reason implies exactly one action (:data:`~percept.gate.REASON_ACTION`), and
   a :class:`~percept.gate.GateOutcome` that disagrees with that mapping cannot be constructed.
2. **Suppression is observable.** A fire the BASELINE rising-edge gate would have delivered, withheld
   by an opt-in precision guard (``require_false_first`` -> ``start_anchored``; ``refractory_s`` ->
   ``refractory``), reaches the runtime's health channel as
   ``RuntimeHealthEvent("gate_suppressed")`` instead of being a silent non-event.
3. **SILENT stays internal-only, and nothing changes by default.** ``SILENT`` is a value of the
   decision model, never an action on a delivered event; and a default-configured gate/core suppresses
   nothing, queues nothing and emits no new health event.
"""
from __future__ import annotations

import asyncio
import unittest

from percept.adapters import Verdict
from percept.agent import _GATE_HEALTH_LOG_MAX, Percept
from percept.behavior import BehaviorExecutor, program_of
from percept.config import GateSettings, Settings
from percept.gate import (
    ASK,
    DELIVERED_ACTIONS,
    FIRE,
    GATE_ACTIONS,
    GATE_REASONS,
    REASON_ACTION,
    SILENT,
    SUPPRESSION_REASONS,
    Gate,
    GateOutcome,
)
from percept.goals import Goal
from percept.runtime.worker import RuntimeHealthEvent, _drain_gate_health

YES = Verdict(satisfied=True, confidence=0.95, evidence="present")
NO = Verdict(satisfied=False, confidence=0.95, evidence="absent")
UNSURE = Verdict(satisfied=True, confidence=0.2, evidence="maybe")


def _run(coro):
    return asyncio.run(coro)


class VocabularyTests(unittest.TestCase):
    """The typed surface itself — closed, consistent, and documented in one place."""

    def test_reason_action_covers_exactly_the_reason_vocabulary(self):
        self.assertEqual(set(REASON_ACTION), set(GATE_REASONS))
        self.assertEqual(len(GATE_REASONS), len(set(GATE_REASONS)), "reason codes must be unique")

    def test_every_reason_implies_a_real_action(self):
        for reason in GATE_REASONS:
            self.assertIn(REASON_ACTION[reason], GATE_ACTIONS, reason)

    def test_every_action_is_reachable_from_some_reason(self):
        """No orphan action: the reason vocabulary is finer than the action vocabulary, not narrower."""
        self.assertEqual(set(REASON_ACTION.values()), set(GATE_ACTIONS))

    def test_suppression_reasons_are_silent_and_in_the_vocabulary(self):
        for reason in SUPPRESSION_REASONS:
            self.assertIn(reason, GATE_REASONS)
            self.assertEqual(REASON_ACTION[reason], SILENT)

    def test_suppression_is_the_two_opt_in_precision_guards_only(self):
        """The set is deliberately NOT "every silent": already_fired is the base re-fire contract (one
        per tick of any continuing true state) and dwell_pending is a deferral, not a withheld fire."""
        self.assertEqual(set(SUPPRESSION_REASONS), {"start_anchored", "refractory"})
        for not_a_suppression in ("already_fired", "dwell_pending", "not_satisfied"):
            self.assertNotIn(not_a_suppression, SUPPRESSION_REASONS)

    def test_silent_is_not_a_delivered_action(self):
        """The whole point of the card's 'SILENT stays internal-only' clause, stated as data."""
        self.assertNotIn(SILENT, DELIVERED_ACTIONS)
        self.assertEqual(set(DELIVERED_ACTIONS), {FIRE, ASK})
        self.assertIn(SILENT, GATE_ACTIONS)

    def test_outcome_rejects_an_action_its_reason_does_not_imply(self):
        with self.assertRaises(ValueError):
            GateOutcome(action=FIRE, reason="refractory")
        with self.assertRaises(ValueError):
            GateOutcome(action=SILENT, reason="rising_edge")

    def test_outcome_rejects_an_unknown_reason(self):
        with self.assertRaises(ValueError):
            GateOutcome(action=SILENT, reason="made_up_code")  # type: ignore[arg-type]

    def test_outcome_flags(self):
        self.assertTrue(GateOutcome(action=SILENT, reason="refractory").suppressed)
        self.assertFalse(GateOutcome(action=SILENT, reason="already_fired").suppressed)
        self.assertTrue(GateOutcome(action=FIRE, reason="rising_edge").delivered)
        self.assertTrue(GateOutcome(action=ASK, reason="stale").delivered)
        self.assertFalse(GateOutcome(action=SILENT, reason="not_satisfied").delivered)

    def test_public_namespace_exports_the_vocabulary(self):
        import percept

        for name in ("GateOutcome", "GATE_REASONS", "REASON_ACTION", "SUPPRESSION_REASONS",
                     "GATE_ACTIONS", "DELIVERED_ACTIONS"):
            self.assertIn(name, percept.__all__, name)
            self.assertTrue(hasattr(percept, name), name)


class ReasonCodeTests(unittest.TestCase):
    """Every reason code is actually produced by the real decision path — no dead vocabulary."""

    def test_rising_edge(self):
        self.assertEqual(Gate().decide("k", YES, now=0.0).reason, "rising_edge")

    def test_not_satisfied(self):
        self.assertEqual(Gate().decide("k", NO, now=0.0).reason, "not_satisfied")

    def test_already_fired(self):
        g = Gate()
        g.decide("k", YES, now=0.0)
        self.assertEqual(g.decide("k", YES, now=1.0).reason, "already_fired")

    def test_low_confidence(self):
        d = Gate(threshold=0.7).decide("k", UNSURE, now=0.0)
        self.assertEqual((d.action, d.reason), (ASK, "low_confidence"))

    def test_stale(self):
        """A CONFIDENT verdict resting on stale support is the OTHER arm of invariant 2 — reported
        distinctly, so 'we refused' and 'we refused because our facts aged out' are separable."""
        d = Gate(threshold=0.7, fresh_min=0.5).decide("k", YES, freshness=0.0, now=0.0)
        self.assertEqual((d.action, d.reason), (ASK, "stale"))

    def test_start_anchored(self):
        d = Gate(require_false_first=True).decide("k", YES, now=0.0)
        self.assertEqual((d.action, d.reason), (SILENT, "start_anchored"))

    def test_refractory(self):
        g = Gate(refractory_s=5.0)
        g.decide("k", YES, now=0.0)
        g.decide("k", NO, now=1.0)  # spurious blip re-arms
        d = g.decide("k", YES, now=2.0)
        self.assertEqual((d.action, d.reason), (SILENT, "refractory"))

    def test_dwell_pending_then_dwell_satisfied(self):
        g = Gate()
        self.assertEqual(g.decide("k", YES, hold_s=3.0, now=0.0).reason, "dwell_pending")
        self.assertEqual(g.decide("k", YES, hold_s=3.0, now=1.0).reason, "dwell_pending")
        d = g.decide("k", YES, hold_s=3.0, now=4.0)
        self.assertEqual((d.action, d.reason), (FIRE, "dwell_satisfied"))

    def test_every_reason_code_is_reachable_from_the_real_decision_path(self):
        """No dead vocabulary: drive the gate through scenarios that between them produce EVERY code,
        and assert the observed set is exactly GATE_REASONS. Guards against the vocabulary growing a
        code no decision path can actually emit."""
        seen: set[str] = set()

        plain = Gate(threshold=0.7, fresh_min=0.5)
        seen.add(plain.decide("k", YES, now=0.0).reason)            # rising_edge
        seen.add(plain.decide("k", YES, now=1.0).reason)            # already_fired
        seen.add(plain.decide("k", NO, now=2.0).reason)             # not_satisfied
        seen.add(plain.decide("k", UNSURE, now=3.0).reason)         # low_confidence
        seen.add(plain.decide("k", YES, freshness=0.0, now=4.0).reason)  # stale

        dwell = Gate()
        seen.add(dwell.decide("d", YES, hold_s=3.0, now=0.0).reason)   # dwell_pending
        seen.add(dwell.decide("d", YES, hold_s=3.0, now=4.0).reason)   # dwell_satisfied

        seen.add(Gate(require_false_first=True).decide("s", YES, now=0.0).reason)  # start_anchored

        refr = Gate(refractory_s=5.0)
        refr.decide("r", YES, now=0.0)
        refr.decide("r", NO, now=1.0)
        seen.add(refr.decide("r", YES, now=2.0).reason)             # refractory

        self.assertEqual(seen, set(GATE_REASONS))

    def test_outcome_carries_key_and_time(self):
        d = Gate().decide(("g", "e"), YES, now=12.5)
        self.assertEqual(d.key, ("g", "e"))
        self.assertEqual(d.at, 12.5)


class EvaluateParityTests(unittest.TestCase):
    """``evaluate`` is exactly ``decide(...).action`` — the historical entry point is untouched."""

    SEQ = [
        (YES, 0.0, None), (YES, 1.0, None), (NO, 2.0, None), (YES, 3.0, None),
        (UNSURE, 4.0, None), (YES, 5.0, 2.0), (YES, 6.0, 2.0), (YES, 9.0, 2.0),
    ]

    def _drive(self, gate, use_decide):
        out = []
        for verdict, now, hold in self.SEQ:
            if use_decide:
                out.append(gate.decide("k", verdict, now=now, hold_s=hold).action)
            else:
                out.append(gate.evaluate("k", verdict, now=now, hold_s=hold))
        return out

    def test_same_actions_for_a_plain_gate(self):
        self.assertEqual(
            self._drive(Gate(), False), self._drive(Gate(), True),
        )

    def test_same_actions_with_both_precision_guards_on(self):
        def g():
            return Gate(require_false_first=True, refractory_s=4.0)

        self.assertEqual(self._drive(g(), False), self._drive(g(), True))

    def test_evaluate_applies_the_edge_exactly_once(self):
        """``evaluate`` delegating to ``decide`` must not double-apply edge state — a second identical
        YES stays SILENT (spent edge), it does not re-fire."""
        g = Gate()
        self.assertEqual(g.evaluate("k", YES, now=0.0), FIRE)
        self.assertEqual(g.evaluate("k", YES, now=1.0), SILENT)
        self.assertEqual(g._armed["k"], False)


class SuppressionHookTests(unittest.TestCase):
    """``on_suppress`` fires for suppressions and NOTHING else."""

    def setUp(self):
        self.seen: list = []

    def test_refractory_suppression_is_announced(self):
        g = Gate(refractory_s=5.0, on_suppress=self.seen.append)
        g.evaluate("k", YES, now=0.0)
        g.evaluate("k", NO, now=1.0)
        g.evaluate("k", YES, now=2.0)
        self.assertEqual([(d.reason, d.key, d.at) for d in self.seen], [("refractory", "k", 2.0)])

    def test_start_anchored_suppression_is_announced(self):
        g = Gate(require_false_first=True, on_suppress=self.seen.append)
        g.evaluate("k", YES, now=0.0)
        self.assertEqual([d.reason for d in self.seen], ["start_anchored"])

    def test_default_gate_never_suppresses(self):
        """The no-default-behavior-change bar: with neither guard on, no code path suppresses, so a
        default deployment sees zero suppression events no matter what the judge says."""
        g = Gate(on_suppress=self.seen.append)
        for now, verdict in enumerate([YES, YES, NO, YES, UNSURE, NO, YES]):
            g.evaluate("k", verdict, now=float(now))
        self.assertEqual(self.seen, [])

    def test_ordinary_silents_are_not_suppressions(self):
        """already_fired / dwell_pending / not_satisfied must never reach the hook — they are the base
        contract, and reporting them would be per-frame noise on every continuing true state."""
        g = Gate(on_suppress=self.seen.append)
        g.evaluate("k", YES, now=0.0)            # rising_edge
        g.evaluate("k", YES, now=1.0)            # already_fired
        g.evaluate("k", NO, now=2.0)             # not_satisfied
        g.evaluate("k", YES, now=3.0, hold_s=9)  # dwell_pending
        self.assertEqual(self.seen, [])

    def test_a_raising_hook_never_breaks_the_firing_path(self):
        def boom(_decision):
            raise RuntimeError("telemetry sink is down")

        g = Gate(refractory_s=5.0, on_suppress=boom)
        self.assertEqual(g.evaluate("k", YES, now=0.0), FIRE)
        self.assertEqual(g.evaluate("k", NO, now=1.0), SILENT)
        self.assertEqual(g.evaluate("k", YES, now=2.0), SILENT)  # suppressed, hook raised, gate fine
        self.assertEqual(g.evaluate("k", YES, now=9.0), SILENT)  # edge still correctly spent

    def test_suppression_is_logged_even_with_no_hook_wired(self):
        """ADR-001 §9 — a bound-driven withholding is never silent, hook or no hook."""
        g = Gate(refractory_s=5.0)
        with self.assertLogs("percept.gate", level="INFO") as caught:
            g.evaluate("k", YES, now=0.0)
            g.evaluate("k", NO, now=1.0)
            g.evaluate("k", YES, now=2.0)
        self.assertTrue(any("gate.suppressed" in line for line in caught.output), caught.output)


class CoreWiringTests(unittest.TestCase):
    """``Percept.create`` wires the gate's suppression hook into the health QUEUE (never on_health)."""

    def _core(self, **gate_kwargs):
        settings = Settings(gate=GateSettings(**gate_kwargs))
        return Percept.create(mode="offline", settings=settings)

    def test_create_wires_the_hook(self):
        core = self._core()
        self.assertEqual(core.gate.on_suppress, core._note_gate_suppressed)

    def test_suppression_queues_onto_the_health_log(self):
        core = self._core(refractory_s=5.0)
        core.gate.evaluate(("g", "e"), YES, now=0.0)
        core.gate.evaluate(("g", "e"), NO, now=1.0)
        core.gate.evaluate(("g", "e"), YES, now=2.0)
        self.assertEqual(core._gate_health_log, [(("g", "e"), "refractory", 2.0)])

    def test_default_core_queues_nothing(self):
        core = self._core()
        for now, verdict in enumerate([YES, YES, NO, YES, NO, YES]):
            core.gate.evaluate(("g", "e"), verdict, now=float(now))
        self.assertEqual(core._gate_health_log, [])

    def test_queue_is_bounded_when_nobody_drains(self):
        """A core driven WITHOUT the runtime loop never drains — the queue must shed, not leak."""
        core = self._core(require_false_first=True)
        for i in range(_GATE_HEALTH_LOG_MAX * 2 + 7):
            core.gate.evaluate(("g", i), YES, now=float(i))  # each new key -> start_anchored
        self.assertEqual(len(core._gate_health_log), _GATE_HEALTH_LOG_MAX)
        # oldest shed first: the newest suppression is still present
        self.assertEqual(core._gate_health_log[-1][0], ("g", _GATE_HEALTH_LOG_MAX * 2 + 6))


class _Core:
    """Minimal stand-in for the drain contract: just the queue the worker reads."""

    def __init__(self, entries):
        self._gate_health_log = list(entries)


class DrainTests(unittest.TestCase):
    """``_drain_gate_health`` turns queued suppressions into ``RuntimeHealthEvent``s."""

    def test_emits_one_event_per_suppression(self):
        core = _Core([(("g", "e"), "refractory", 2.0), (("g", "f"), "start_anchored", 3.0)])
        got: list[RuntimeHealthEvent] = []
        _run(_drain_gate_health(core, got.append, 9.0))
        self.assertEqual([e.kind for e in got], ["gate_suppressed", "gate_suppressed"])
        self.assertEqual([e.data["reason"] for e in got], ["refractory", "start_anchored"])
        self.assertEqual(got[0].data["key"], repr(("g", "e")))
        self.assertEqual(got[0].data["at"], 2.0)
        self.assertEqual(got[0].ts, 9.0)  # stamped on the DRAINING tick

    def test_drains_the_queue(self):
        core = _Core([(("g", "e"), "refractory", 2.0)])
        _run(_drain_gate_health(core, [].append, 9.0))
        self.assertEqual(core._gate_health_log, [])

    def test_drains_even_without_a_health_sink(self):
        """Drain-regardless: with ``on_health=None`` the queue must still empty, or a run with no
        health subscriber accumulates one tick's worth forever."""
        core = _Core([(("g", "e"), "refractory", 2.0)])
        _run(_drain_gate_health(core, None, 9.0))
        self.assertEqual(core._gate_health_log, [])

    def test_empty_queue_emits_nothing(self):
        got: list = []
        _run(_drain_gate_health(_Core([]), got.append, 9.0))
        self.assertEqual(got, [])

    def test_a_core_without_the_queue_is_a_noop(self):
        _run(_drain_gate_health(object(), [].append, 9.0))  # must not raise

    def test_worker_drains_unconditionally_each_tick(self):
        """The drain call sits OUTSIDE the dispatch-mode branch and BEFORE the per-frame escalation
        `continue`, so it runs on every tick in both inline and async judge dispatch."""
        import inspect

        from percept.runtime import worker

        src = inspect.getsource(worker.run)
        self.assertIn("_drain_gate_health(core, on_health, t)", src)


class SilentStaysInternalTests(unittest.TestCase):
    """SILENT is a value of the decision model, never an action on a delivered event."""

    def test_executor_returns_nothing_on_a_silent_decision(self):
        core = Percept.create(mode="offline")
        ex = BehaviorExecutor(core.gate)
        goal = Goal(id="g", condition="a cup is on the desk", say="cup!")
        program = program_of(goal)
        first = ex.fire_judged(goal, YES, ("g", "e"), program, t=0.0)
        self.assertIsNotNone(first)
        self.assertEqual(first.action, FIRE)
        # the SAME true state one tick later is SILENT internally -> nothing is delivered at all
        self.assertIsNone(ex.fire_judged(goal, YES, ("g", "e"), program, t=1.0))
        # ...and a confident-false is SILENT too -> still nothing delivered
        self.assertIsNone(ex.fire_judged(goal, NO, ("g", "e"), program, t=2.0))

    def test_every_delivered_action_is_in_delivered_actions(self):
        core = Percept.create(mode="offline")
        ex = BehaviorExecutor(core.gate)
        goal = Goal(id="g", condition="a cup is on the desk", say="cup!")
        program = program_of(goal)
        delivered = []
        for i, verdict in enumerate([YES, YES, NO, UNSURE, YES, NO, YES, UNSURE]):
            fired = ex.fire_judged(goal, verdict, ("g", "e"), program, t=float(i))
            if fired is not None:
                delivered.append(fired.action)
        self.assertTrue(delivered, "the drive sequence must actually deliver something")
        for action in delivered:
            self.assertIn(action, DELIVERED_ACTIONS)
            self.assertNotEqual(action, SILENT)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
