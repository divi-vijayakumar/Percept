"""Card X7 — the live path CONSTRUCTS the contract's centre.

``percept.contracts.JudgmentEvent`` is ADR-001 §1's frozen output of one pass through the judge
funnel, and until X7 nothing on the hot path ever built one: the runtime went
``Verdict -> GateOutcome -> Fired -> FireEvent`` and the contract governed an object it never made.
These tests pin the fix and, just as importantly, its ADDITIVE-ONLY constraint:

  1. the judged watch path emits a real ``JudgmentEvent`` on ``FireEvent.judgment`` — for FIRE and
     for ASK — keyed by the same gate key the decision was made on;
  2. its ``JudgmentTiming`` is honest: five monotonic-domain stamps in funnel order, all from THIS
     core's injected clock, with ``source_media_ts`` carrying the caller's stream timeline ONLY when
     one was supplied (never back-filled from the monotonic clock);
  3. the latency helpers compute real numbers off it (the judge call's duration is visible);
  4. ``evidence_seq`` is the per-key monotonic sequence reserved at judge DISPATCH (ADR-001 §8), and
     its map is bounded so a long multi-entity watch cannot leak through it;
  5. NOTHING ELSE CHANGED: the firing trace with judgment stamping is byte-for-byte the trace the
     executor produces without it, paths that take no stamps get ``judgment=None`` (never an invented
     event), and a timing record that cannot be frozen degrades to ``judgment=None`` — it never
     suppresses, delays or alters a fire.
"""

from __future__ import annotations

import asyncio
import unittest

from percept.adapters import Goal, Verdict
from percept.agent import FireEvent, Percept
from percept.behavior import BehaviorExecutor, Fired, program_of
from percept.contracts import JudgmentEvent, JudgmentTiming, JudgmentTimingBuilder
from percept.gate import ASK, FIRE, Gate
from percept.sequencing import EvidenceSequencer


class StepClock:
    """A monotonic clock that advances one tick per read — every stamp is distinct and ordered."""

    def __init__(self, step: float = 1.0) -> None:
        self._t = 0.0
        self._step = step

    def __call__(self) -> float:
        self._t += self._step
        return self._t


class ScriptedVision:
    """No model, no keys: a fixed sequence of verdicts, one per judge call."""

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
NO = Verdict(satisfied=False, confidence=0.95, evidence="no")
UNSURE = Verdict(satisfied=True, confidence=0.2, evidence="unsure")


def _goal(gid: str = "g") -> Goal:
    return Goal(id=gid, condition="the door is open", say="door open", state={"type": "watch"})


def _run_watch(verdicts, *, ts=(0.0,), clock=None, goals=("g",)):
    """Drive ``perceive_judged`` over ``ts`` and return (agent, [FireEvent...])."""
    vision = ScriptedVision(verdicts)
    agent = Percept.create(vision=vision, discover_plugins=False, clock=clock or StepClock())
    for gid in goals:
        agent.add_goal(_goal(gid))
    out: list[FireEvent] = []

    async def go():
        for t in ts:
            out.extend(await agent.perceive_judged("frame", t=t))

    asyncio.run(go())
    return agent, out


class LivePathConstructsAJudgmentEventTests(unittest.TestCase):
    """(1) The runtime now makes the object ADR-001 governs."""

    def test_fire_carries_a_real_judgment_event(self):
        _, fires = _run_watch([YES])
        self.assertEqual([f.action for f in fires], [FIRE])
        judgment = fires[0].judgment
        self.assertIsInstance(judgment, JudgmentEvent)
        self.assertEqual(judgment.action, "fire")
        self.assertEqual(judgment.key, ("g", None))  # the SAME key the gate decided on
        self.assertIsInstance(judgment.timing, JudgmentTiming)

    def test_ask_carries_a_judgment_event_too(self):
        _, fires = _run_watch([UNSURE])
        self.assertEqual([f.action for f in fires], [ASK])
        self.assertIsNotNone(fires[0].judgment)
        self.assertEqual(fires[0].judgment.action, "ask")

    def test_judgment_event_is_immutable(self):
        _, fires = _run_watch([YES])
        with self.assertRaises(Exception):
            fires[0].judgment.action = "silent"  # frozen dataclass — the §1 immutability invariant

    def test_a_silent_tick_emits_no_judgment_event(self):
        # Nothing fired -> no FireEvent -> no JudgmentEvent (a suppression is not a judgment record).
        _, fires = _run_watch([NO])
        self.assertEqual(fires, [])


class TimingIsHonestTests(unittest.TestCase):
    """(2)+(3) Five stamps, one domain, real latencies — and nothing invented."""

    def test_stamps_are_in_funnel_order_and_distinct(self):
        _, fires = _run_watch([YES], clock=StepClock())
        timing = fires[0].judgment.timing
        stamps = (timing.ingest_observed_ts, timing.admitted_ts, timing.judge_started_ts,
                  timing.judge_completed_ts, timing.decision_emitted_ts)
        self.assertEqual(list(stamps), sorted(stamps))
        self.assertEqual(len(set(stamps)), 5)  # a step clock => every funnel stage is distinguishable

    def test_latency_helpers_measure_the_real_funnel(self):
        _, fires = _run_watch([YES], clock=StepClock(step=0.5))
        timing = fires[0].judgment.timing
        self.assertGreater(timing.ingest_to_decision_s(), 0.0)
        self.assertGreater(timing.judge_duration_s(), 0.0)
        self.assertGreater(timing.decision_overhead_s(), 0.0)
        self.assertAlmostEqual(
            timing.ingest_to_decision_s(),
            timing.judge_wait_s() + timing.judge_duration_s() + timing.decision_overhead_s()
            + (timing.admitted_ts - timing.ingest_observed_ts),
        )

    def test_source_media_ts_carries_the_stream_timeline_when_supplied(self):
        _, fires = _run_watch([YES], ts=(26.2,))
        self.assertEqual(fires[0].judgment.timing.source_media_ts, 26.2)

    def test_source_media_ts_is_left_unset_when_the_caller_has_no_timeline(self):
        """t=None means the runtime has no source timeline — it must NOT fabricate one from its clock."""
        vision = ScriptedVision([YES])
        agent = Percept.create(vision=vision, discover_plugins=False, clock=StepClock())
        agent.add_goal(_goal())
        fires = asyncio.run(agent.perceive_judged("frame"))
        self.assertIsNone(fires[0].judgment.timing.source_media_ts)

    def test_stamps_never_mix_the_media_domain_into_the_monotonic_one(self):
        # A stream timeline far in the future must not leak into the monotonic stamps (which would
        # silently corrupt every latency number computed off them).
        _, fires = _run_watch([YES], ts=(10_000.0,), clock=StepClock())
        timing = fires[0].judgment.timing
        self.assertLess(timing.decision_emitted_ts, 100.0)
        self.assertEqual(timing.source_media_ts, 10_000.0)


class EvidenceSequenceTests(unittest.TestCase):
    """(4) evidence_seq is the per-key dispatch sequence, and its bookkeeping is bounded."""

    def test_sequence_is_monotonic_per_key(self):
        # fire / re-arm on a confident false / fire again -> two fires on the same key.
        _, fires = _run_watch([YES, NO, YES], ts=(0.0, 1.0, 2.0))
        seqs = [f.judgment.evidence_seq for f in fires]
        self.assertEqual(len(seqs), 2)
        self.assertLess(seqs[0], seqs[1])
        self.assertEqual(seqs[0], 1)  # reserved at the FIRST dispatch for this key

    def test_distinct_keys_have_independent_sequences(self):
        _, fires = _run_watch([YES], goals=("g", "h"))
        by_goal = {f.goal_id: f.judgment.evidence_seq for f in fires}
        self.assertEqual(by_goal, {"g": 1, "h": 1})

    def test_sequencer_map_is_lru_bounded(self):
        vision = ScriptedVision([YES])
        agent = Percept.create(vision=vision, discover_plugins=False)
        import percept.agent as agent_mod

        for i in range(agent_mod._SEQ_MAX_KEYS + 25):
            agent._admit_evidence(("g", f"entity-{i}"))
        self.assertLessEqual(len(agent._sequencer.tracked_keys()), agent_mod._SEQ_MAX_KEYS)

    def test_tracked_keys_is_first_admission_ordered(self):
        seq = EvidenceSequencer()
        seq.admit("a")
        seq.admit("b")
        seq.admit("a")
        self.assertEqual(seq.tracked_keys(), ("a", "b"))
        seq.forget("a")
        self.assertEqual(seq.tracked_keys(), ("b",))


class AdditiveOnlyTests(unittest.TestCase):
    """(5) A consumer that ignores the judgment sees today's behaviour, byte for byte."""

    def _trace(self, *, with_timing: bool):
        """The firing trace of one executor over a fixed verdict sequence."""
        ex = BehaviorExecutor(Gate())
        goal = _goal()
        program = program_of(goal)
        out = []
        for i, v in enumerate([YES, YES, NO, YES, UNSURE, NO, YES]):
            timing = None
            if with_timing:
                timing = JudgmentTimingBuilder(clock=StepClock())
                timing.mark_ingest_observed()
                timing.mark_admitted()
                timing.mark_judge_started()
                timing.mark_judge_completed()
            fired = ex.fire_judged(goal, v, ("g", "e"), program, t=float(i),
                                   timing=timing, evidence_seq=i + 1)
            out.append(None if fired is None else (fired.goal_id, fired.action, fired.text,
                                                   fired.entity_id, fired.verdict))
        return out

    def test_firing_trace_is_byte_for_byte_with_and_without_stamping(self):
        self.assertEqual(self._trace(with_timing=False), self._trace(with_timing=True))

    def test_without_timing_the_judgment_is_none_not_invented(self):
        ex = BehaviorExecutor(Gate())
        goal = _goal()
        fired = ex.fire_judged(goal, YES, ("g", "e"), program_of(goal))
        self.assertIsInstance(fired, Fired)
        self.assertIsNone(fired.judgment)

    def test_non_judged_paths_carry_no_judgment(self):
        """Episode / transition / absence fires take no judge-funnel stamps — absent, not fabricated."""
        ex = BehaviorExecutor(Gate())
        goal = _goal()
        ep = ex.fire_episode(goal, program_of(goal), n=1, is_new=True)
        self.assertIsNotNone(ep)
        self.assertIsNone(ep.judgment)
        tr = ex.fire_transition(goal, YES, YES)
        self.assertIsNotNone(tr)
        self.assertIsNone(tr.judgment)

    def test_fire_event_default_keeps_existing_positional_construction(self):
        ev = FireEvent("g", FIRE, "text", "e", YES)  # every pre-X7 call site
        self.assertIsNone(ev.judgment)

    def test_an_unfreezable_timing_still_fires(self):
        """The fire is the product; the record is observability. A broken clock must never eat a fire."""

        class Metrics:
            def __init__(self):
                self.seen = []

            def incr(self, key):
                self.seen.append(key)

        metrics = Metrics()
        ex = BehaviorExecutor(Gate(), metrics)
        goal = _goal()
        broken = JudgmentTimingBuilder(clock=StepClock())
        broken.mark_ingest_observed()  # the judge stamps were never taken -> freeze() raises
        fired = ex.fire_judged(goal, YES, ("g", "e"), program_of(goal), timing=broken)
        self.assertIsNotNone(fired)
        self.assertEqual(fired.action, FIRE)
        self.assertIsNone(fired.judgment)
        self.assertIn("judgment.build_failed", metrics.seen)

    def test_a_degraded_judge_tick_emits_no_judgment(self):
        """A fail-closed judge (A30) never produced a decision, so it never freezes one."""

        class Exploding(ScriptedVision):
            async def judge(self, *a, **kw):
                raise TimeoutError("provider down")

        agent = Percept.create(vision=Exploding([YES]), discover_plugins=False, clock=StepClock())
        agent.add_goal(_goal())
        fires = asyncio.run(agent.perceive_judged("frame", t=0.0))
        self.assertEqual(fires, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
