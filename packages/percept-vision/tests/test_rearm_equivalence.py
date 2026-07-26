"""S2 gating spike — re-arm boundary equivalence (L1, no backends, no API keys).

GOAL: prove that the Concern (C, R) firing model's per-concern *re-arm boundaries* ``R`` can each
be produced WITHOUT touching ``Gate.evaluate`` / ``classify`` — i.e. purely by

  (a) KEY-SYNTHESIS: folding an epoch / event-id / time-bucket into the gate ``key`` so a "new key"
      lands on the gate's default-armed edge (``self._armed.get(key, True)``), and/or
  (b) calling the EXISTING ``gate.rearm(key)`` / ``gate.reset_goal(goal_id)`` at the right moment.

The linchpin invariant for the rearm refactor: gate.py stays LITERALLY unchanged. These tests are the
proof. Each test scripts a Verdict sequence, drives the real ``Gate``, and asserts the
key-synthesis/rearm action trace EQUALS the intended FIRE/ASK/SILENT trace for that ``R``.

Captured current gate behavior under test (the R15 shim pins these defaults):
  * Gate.__init__: threshold=0.7, fresh_min=0.5
  * classify(): confidence < threshold -> "unknown"(ASK); freshness < fresh_min -> "unknown"(ASK);
                else satisfied -> "yes"/"no"
  * evaluate(): unknown -> ASK (edge untouched); no -> SILENT + re-arm; yes on armed -> FIRE+disarm;
                yes on disarmed -> SILENT.
  * _armed semantics: new key defaults to True (armed); FIRE disarms; confident-FALSE re-arms;
                      ASK never changes the edge.
  * The TODAY default re-arm rule == ``R = on_false`` (re-arm only on a confident-false verdict).

Re-arm boundaries enumerated (from the (C, R) Concern model / TrackingBehavior):
  on_false    today's default. R re-arms on a confident-false verdict. FIRE -> stay silent until a
              confident NO is seen, then a later YES fires again.
  never       one-shot. R never re-arms. Fire exactly once per concern, ever.
  each_event  re-arm on each distinct event / episode (a new event-id is a fresh fire).
  interval    re-arm on a timer cadence (every N seconds the concern may fire again).
  deadline    overdue / absence: fire if NOT satisfied by time T (an inverted, time-triggered edge).
"""

import unittest

from percept.adapters import Verdict
from percept.gate import ASK, FIRE, SILENT, Gate


def yes(conf: float = 0.9) -> Verdict:
    return Verdict(satisfied=True, confidence=conf)


def no(conf: float = 0.9) -> Verdict:
    return Verdict(satisfied=False, confidence=conf)


# --- captured current defaults (pin these into the R15 shim) -------------------------------------
CURRENT_THRESHOLD = 0.7
CURRENT_FRESH_MIN = 0.5


class TestCapturedDefaults(unittest.TestCase):
    """Lock today's EXACT gate defaults + the today re-arm rule (== R = on_false)."""

    def test_default_threshold_and_fresh_min(self):
        g = Gate()
        self.assertEqual(g.threshold, CURRENT_THRESHOLD)
        self.assertEqual(g.fresh_min, CURRENT_FRESH_MIN)

    def test_today_rule_is_on_false(self):
        # The hardcoded re-arm at gate.py: confident-false re-arms, nothing else does.
        g = Gate()
        k = ("g", "e")
        self.assertEqual(g.evaluate(k, yes()), FIRE)  # rising edge
        self.assertEqual(g.evaluate(k, yes()), SILENT)  # still true -> no re-fire
        self.assertEqual(g.evaluate(k, no()), SILENT)  # confident-false RE-ARMS
        self.assertEqual(g.evaluate(k, yes()), FIRE)  # re-armed -> fires again


# --- R = on_false : the native edge, no synthesis needed -----------------------------------------
class TestRearmOnFalse(unittest.TestCase):
    """R = on_false is the gate's native behavior: same fixed key, re-arm falls out of evaluate()."""

    def test_on_false_matches_intended(self):
        g = Gate()
        k = ("goal", "entity")
        # intended: FIRE on rising edge; SILENT while still true; a confident NO re-arms; YES fires.
        verdicts = [yes(), yes(), no(), yes(), yes()]
        intended = [FIRE, SILENT, SILENT, FIRE, SILENT]
        got = [g.evaluate(k, v) for v in verdicts]
        self.assertEqual(got, intended)


# --- R = never : one-shot via NO re-arm ----------------------------------------------------------
class TestRearmNever(unittest.TestCase):
    """R = never (one-shot): fire once ever, never re-arm.

    Approach: hold ONE fixed key AND never call rearm/reset. The only thing that could re-arm the
    native edge is a confident-FALSE verdict (gate.py: ``truth == 'no'`` -> ``_armed[key]=True``).
    So one-shot is expressible ONLY when the concern never feeds a confident-false through the gate.
    The (C, R=never) adapter must therefore SUPPRESS confident-false from reaching evaluate() (feed
    it as the existing ASK/no-op, or simply not evaluate on false). Given that upstream contract,
    the gate yields a perfect one-shot with zero gate changes.
    """

    def test_never_one_shot_when_false_suppressed(self):
        g = Gate()
        k = ("goal", "oneshot")
        # Adapter contract for R=never: never push a confident-false into the gate. Sequence of
        # YES verdicts (the only edge-moving inputs) -> exactly one FIRE, forever silent after.
        verdicts = [yes(), yes(), yes(), yes()]
        intended = [FIRE, SILENT, SILENT, SILENT]
        got = [g.evaluate(k, v) for v in verdicts]
        self.assertEqual(got, intended)

    def test_never_with_low_conf_still_one_shot(self):
        # ASK never moves the edge, so interleaved low-confidence inputs don't break one-shot.
        g = Gate()
        k = ("goal", "oneshot")
        verdicts = [yes(conf=0.4), yes(), yes(conf=0.4), yes()]
        intended = [ASK, FIRE, ASK, SILENT]
        got = [g.evaluate(k, v) for v in verdicts]
        self.assertEqual(got, intended)


# --- R = each_event : re-arm per distinct event via key-synthesis --------------------------------
class TestRearmEachEvent(unittest.TestCase):
    """R = each_event: a new distinct event/episode re-arms.

    Approach: fold the event/episode id into the key — ``(goal_id, event_id)``. Each NEW event id is
    an unseen key -> ``_armed.get(key, True)`` defaults armed -> FIRE. This is EXACTLY the pattern
    the agent already uses for the event path (agent.py:228, ``(goal.id, distinct.id)``) and the
    watch path (``(goal.id, focus.id)``). Zero gate changes.
    """

    def test_each_distinct_event_fires_once(self):
        g = Gate()
        gid = "goal"
        # Three distinct events; each fires once. A repeat of an event id is SILENT (refractory).
        seq = ["ev1", "ev1", "ev2", "ev3", "ev2"]
        intended = [FIRE, SILENT, FIRE, FIRE, SILENT]
        got = [g.evaluate((gid, ev), yes()) for ev in seq]
        self.assertEqual(got, intended)

    def test_each_event_equivalence_vs_explicit_rearm(self):
        # Key-synthesis must match the "rearm the same key on each new event" interpretation.
        # Path A: synthesize a fresh key per event.
        ga = Gate()
        events = ["a", "b", "c"]
        synth = [ga.evaluate(("g", ev), yes()) for ev in events]
        # Path B: one key, explicit rearm() before each new event (the equivalent semantics).
        gb = Gate()
        explicit = []
        for i, _ev in enumerate(events):
            if i > 0:
                gb.rearm(("g", "fixed"))  # boundary crossed -> re-open the edge
            explicit.append(gb.evaluate(("g", "fixed"), yes()))
        self.assertEqual(synth, [FIRE, FIRE, FIRE])
        self.assertEqual(synth, explicit)


# --- R = interval : re-arm on a timer cadence via time-bucket key-synthesis ----------------------
class TestRearmInterval(unittest.TestCase):
    """R = interval: re-arm every N seconds.

    Approach: fold a time-bucket ``floor(t / interval)`` into the key — ``(goal_id, bucket)``. A new
    bucket is a new key -> default armed -> FIRE; repeated YES within the same bucket is SILENT. The
    cadence lives entirely in how the upstream adapter computes the bucket; the gate is untouched.
    """

    @staticmethod
    def bucket(t: float, interval: float) -> int:
        return int(t // interval)

    def test_interval_fires_once_per_bucket(self):
        g = Gate()
        gid = "goal"
        interval = 10.0
        # times within/across 10s buckets; YES held continuously true.
        times = [0.0, 3.0, 9.9, 10.0, 12.0, 25.0]
        intended = [FIRE, SILENT, SILENT, FIRE, SILENT, FIRE]
        got = [g.evaluate((gid, self.bucket(t, interval)), yes()) for t in times]
        self.assertEqual(got, intended)

    def test_interval_equivalence_vs_explicit_rearm(self):
        # Synthesis (time-bucket key) must match "rearm the fixed key when the bucket advances".
        g_synth = Gate()
        g_rearm = Gate()
        interval = 10.0
        times = [0.0, 4.0, 11.0, 21.0]
        synth, rearmed = [], []
        last_bucket = None
        for t in times:
            b = self.bucket(t, interval)
            synth.append(g_synth.evaluate(("g", b), yes()))
            if last_bucket is not None and b != last_bucket:
                g_rearm.rearm(("g", "fixed"))
            rearmed.append(g_rearm.evaluate(("g", "fixed"), yes()))
            last_bucket = b
        self.assertEqual(synth, [FIRE, SILENT, FIRE, FIRE])
        self.assertEqual(synth, rearmed)


# --- R = deadline : overdue / absence -- NOT expressible by key/rearm alone -----------------------
class TestRearmDeadline(unittest.TestCase):
    # R=deadline is an INVERTED, time-TRIGGERED edge: "fire if NOT satisfied by time T". The gate
    # only fires on a confident-TRUE rising edge fed to evaluate(); absence/overdue is the absence
    # of a YES, which the gate never observes and a scheduler timer must synthesize. Two parts are
    # needed and NEITHER is a gate change: (1) a SCHEDULER timer at T (already exists:
    # Agent.schedule_at / tick, agent.py:242) and (2) an upstream adapter that, at T, checks "was the
    # concern satisfied?" and only then pushes a synthetic YES into the gate. So deadline is NOT
    # expressible by key-synthesis/rearm ALONE, but it ALSO does not require an evaluate()/classify()
    # change — it is composed OUTSIDE the gate. expectedFailure documents that boundary: the gate
    # stays unchanged; deadline lives in the scheduler+adapter layer.
    @unittest.expectedFailure
    def test_deadline_via_gate_alone(self):
        # There is NO sequence of (key, verdict) into the unchanged gate that, on its own, fires on
        # the ABSENCE of a satisfying verdict by a deadline. The gate has no clock and no inversion.
        # We assert the (impossible) expectation so the xfail records WHY: the gate never fires
        # without a confident-true verdict, so "fire because nothing was true by T" can't originate
        # in evaluate(). This MUST be composed by scheduler + adapter (outside the gate).
        g = Gate()
        k = ("goal", "deadline")
        # Concern was never satisfied before the deadline -> intended action at T is FIRE (overdue).
        # But feeding only "not satisfied" confident-false verdicts yields SILENT (and re-arms).
        deadline_passed_unsatisfied = [no(), no(), no()]
        got = [g.evaluate(k, v) for v in deadline_passed_unsatisfied]
        # Intended deadline semantics want a FIRE here; the bare gate cannot produce it.
        self.assertEqual(got, [FIRE, SILENT, SILENT])  # deliberately false -> documents the gap


if __name__ == "__main__":
    unittest.main()
