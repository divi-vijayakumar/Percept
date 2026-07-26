"""P-ALLOC-1 (the release criteria; ADR-001 §8) — SCAFFOLD PROBE.

Property under test: allocation + stale-completion safety under an N-WORKER POOL. With multiple
concurrent judge jobs in flight for the SAME key, out-of-order completion must NOT corrupt gate
state:

  * every job carries a monotonic per-key ``evidence_seq`` (``percept.sequencing.EvidenceSequencer``) reserved at DISPATCH time, in dispatch order — never completion order;
  * the decision reducer (``percept.sequencing.SerializedDecisionReducer``) serializes application
    per ``(source_id, plan_id, episode_or_entity_key)`` key via a per-key ``asyncio.Lock``;
  * a result whose sequence is behind ``latest_admitted_sequence`` is dropped as a TRACED stale
    completion (``dropped=True`` + a named ``drop_reason``) and NEVER reaches ``Gate.evaluate`` — the
    gate is provably untouched by a stale drop, not merely "the effect happened to not show up";
  * no watch starves: every admitted job resolves to either an applied decision or a traced drop —
    never a silent hang — even when many keys share the same worker pool concurrently.

SCAFFOLD-ONLY (Viability phase, gated on bars-locked before any for-record run): this file exercises
existing, already-landed primitives under a REAL concurrent asyncio race (multiple in-flight
``FakeJudge.judge()`` coroutines with different per-call latency, driven through ``asyncio.gather`` so
completion order is decided by actual awaited time, not by program call order) — not a simulated
timeline. This is a standalone scaffold and edits no product source.

Reuses (imported, never edited):
  * ``percept.sequencing.{EvidenceSequencer, GateDecision, ResultPolicy, SerializedDecisionReducer}`` — the sequencing + serialized-reducer mechanism under test.
  * ``percept.backends.fake_judge.{FakeJudge, LatencyProfile}`` — a deterministic, seeded
    judge double; two instances with a large (15x) fixed-latency gap give the OLD/slow job a real
    per-call delay so it completes AFTER the NEW/fast job under genuine asyncio concurrency. Its own
    scripted verdict CONTENT is irrelevant here — each job's ``submit()`` verdict is supplied
    explicitly by the test so the gate-mutation outcome is fully controlled and asserted on.
  * ``percept.harness.metrics.MetricsCollector`` — ``record_gate_decision`` /
    ``drop_counts`` / ``total_drops`` for the stale-completion drop-count assertions.
  * ``percept.clock.FakeClock`` — the collector's injected time source (no wall-clock reads).
  * ``percept.gate.{Gate, FIRE, SILENT}`` — the underlying three-state gate the reducer wraps.

Key shape used throughout: ``(source_id, plan_id, episode_or_entity_key)`` — a 3-tuple, matching the
the allocator's public key contract; the reducer/sequencer treat it as an opaque ``Hashable``, so nothing
downstream needs to know the tuple's internal shape.
"""

from __future__ import annotations

import asyncio
import unittest
from typing import Hashable

from percept.adapters import Verdict
from percept.backends.fake_judge import FakeJudge, LatencyProfile
from percept.clock import FakeClock
from percept.gate import FIRE, SILENT, Gate
from percept.harness.metrics import MetricsCollector
from percept.sequencing import GateDecision, ResultPolicy, SerializedDecisionReducer


def yes(conf: float = 0.9) -> Verdict:
    return Verdict(satisfied=True, confidence=conf)


def no(conf: float = 0.9) -> Verdict:
    return Verdict(satisfied=False, confidence=conf)


def _run(coro):
    return asyncio.run(coro)


async def _gather(*coros):
    """``asyncio.gather`` must be called INSIDE a running loop (constructing its Tasks eagerly
    wraps each coroutine via ``ensure_future``); this tiny wrapper is what lets every race below
    read as ``_run(_gather(job_a, job_b))`` while still only ever calling ``gather`` from within
    ``asyncio.run``'s loop."""
    return await asyncio.gather(*coros)


def _slow_judge(seed: int = 1) -> FakeJudge:
    """The OLD job's provider: a real (small but non-trivial) per-call delay, ~15x the fast
    judge's, so it genuinely completes LAST under real asyncio scheduling — not a simulated
    timeline (see module docstring)."""
    return FakeJudge(seed=seed, latency=LatencyProfile(kind="fixed", value_s=0.15), simulate_sleep=True)


def _fast_judge(seed: int = 2) -> FakeJudge:
    """The NEW job's provider: completes quickly, well before ``_slow_judge``'s delay elapses."""
    return FakeJudge(seed=seed, latency=LatencyProfile(kind="fixed", value_s=0.01), simulate_sleep=True)


async def _run_job(
    reducer: SerializedDecisionReducer,
    judge: FakeJudge,
    key: Hashable,
    seq: int,
    verdict: Verdict,
    *,
    episode_id: Hashable | None = None,
    policy: ResultPolicy | None = None,
) -> GateDecision:
    """One worker-pool job: await the judge's REAL per-call latency (the only role ``FakeJudge``
    plays in this probe — see module docstring), then submit an EXPLICIT verdict through the
    reducer. Completion order across jobs gathered together is decided by actual asyncio
    scheduling, never by this function's own call order."""
    await judge.judge("probe-condition", "probe-frame")
    return await reducer.submit(key, seq, verdict, episode_id=episode_id, policy=policy)


class TestClassicOutOfOrderRace(unittest.TestCase):
    """Admit seq=1 (OLD, slow provider) then seq=2 (NEW, fast
    provider) for the SAME key under the SAME plan revision; drive completions out of order (2
    finishes before 1) via a REAL asyncio race, and prove the gate ends up reflecting seq=2 ONLY."""

    KEY: Hashable = ("src-1", "plan-rev-7", "ep-A")

    def setUp(self) -> None:
        self.gate = Gate(threshold=0.7, fresh_min=0.5)
        self.reducer = SerializedDecisionReducer(self.gate)
        self.metrics = MetricsCollector(FakeClock())
        self.evaluate_calls: list[Hashable] = []
        real_evaluate = self.gate.evaluate

        def traced_evaluate(key, verdict, **kwargs):
            self.evaluate_calls.append(key)
            return real_evaluate(key, verdict, **kwargs)

        self.gate.evaluate = traced_evaluate  # type: ignore[assignment]

    def test_out_of_order_completion_does_not_corrupt_gate_state(self) -> None:
        # Dispatch order (program order) fixes evidence_seq -- OLD first, NEW second -- REGARDLESS
        # of which one is about to finish first.
        seq_old = self.reducer.admit(self.KEY)  # 1: OLD evidence, slow provider
        seq_new = self.reducer.admit(self.KEY)  # 2: NEW evidence, fast provider

        slow = _slow_judge()
        fast = _fast_judge()

        # A REAL out-of-order race: both jobs are scheduled concurrently; the slow one's larger
        # per-call latency means the fast one's `submit()` genuinely lands at the reducer FIRST,
        # despite having been admitted SECOND. `asyncio.gather` still returns results in ARGUMENT
        # order, so `decision_old`/`decision_new` line up with `seq_old`/`seq_new` regardless of
        # which finished first.
        decision_old, decision_new = _run(_gather(
            _run_job(self.reducer, slow, self.KEY, seq_old, no()),   # stale "no" -- would
                                                                       # re-arm/disarm if wrongly applied
            _run_job(self.reducer, fast, self.KEY, seq_new, yes()),  # the real, current evidence
        ))

        for d in (decision_old, decision_new):
            self.metrics.record_gate_decision(d)

        # The NEW/fast job applied cleanly and fired.
        self.assertFalse(decision_new.dropped)
        self.assertEqual(decision_new.action, FIRE)

        # The OLD/slow job lost the race -- dropped as a TRACED stale completion, never FIRE/ASK.
        self.assertTrue(decision_old.dropped)
        self.assertEqual(decision_old.drop_reason, "superseded_by_newer_admission")
        self.assertEqual(decision_old.action, SILENT)

        # The strongest form of "never mutating the gate": Gate.evaluate was invoked exactly ONCE
        # for this key -- the dropped completion never even reached the gate; it did not merely
        # produce a no-op once there.
        self.assertEqual(self.evaluate_calls, [self.KEY])

        # Gate state reflects seq=2 ONLY: a repeat "yes" stays SILENT (already armed by seq=2's
        # fire) -- not a fresh FIRE, which a corrupted/re-armed edge would incorrectly produce.
        self.assertEqual(self.gate.evaluate(self.KEY, yes()), SILENT)
        self.assertEqual(self.reducer.last_applied(self.KEY), seq_new)

        # MetricsCollector sees exactly the one stale drop, correctly reasoned.
        self.assertEqual(self.metrics.total_drops(), 1)
        self.assertEqual(self.metrics.drop_counts(), {"superseded_by_newer_admission": 1})


class TestResultPolicyUnderConcurrency(unittest.TestCase):
    """latest_only vs episode_ordered (``ResultPolicy``) exercised through the SAME real
    async worker-pool race this probe is built around -- not just direct sequential calls."""

    KEY: Hashable = ("src-2", "plan-rev-3", "entity-42")

    def setUp(self) -> None:
        self.gate = Gate(threshold=0.7, fresh_min=0.5)

    def test_latest_only_drops_a_cross_episode_straggler(self) -> None:
        reducer = SerializedDecisionReducer(self.gate, policy=ResultPolicy(ordering="latest_only"))
        seq_ep1 = reducer.admit(self.KEY, episode_id="ep-1")  # OLD episode, admitted first
        seq_ep2 = reducer.admit(self.KEY, episode_id="ep-2")  # a NEW episode, admitted second

        slow = _slow_judge(seed=11)
        fast = _fast_judge(seed=12)
        decision_ep1, decision_ep2 = _run(_gather(
            _run_job(reducer, slow, self.KEY, seq_ep1, no(), episode_id="ep-1"),
            _run_job(reducer, fast, self.KEY, seq_ep2, yes(), episode_id="ep-2"),
        ))

        self.assertFalse(decision_ep2.dropped)
        self.assertEqual(decision_ep2.action, FIRE)
        # latest_only: dropped GLOBALLY, even though it belongs to a DIFFERENT episode.
        self.assertTrue(decision_ep1.dropped)
        self.assertEqual(decision_ep1.drop_reason, "superseded_by_newer_admission")

    def test_episode_ordered_does_not_penalize_a_fresh_episode_under_the_race(self) -> None:
        reducer = SerializedDecisionReducer(self.gate, policy=ResultPolicy(ordering="episode_ordered"))
        seq_ep1 = reducer.admit(self.KEY, episode_id="ep-1")  # OLD episode's only job
        seq_ep2 = reducer.admit(self.KEY, episode_id="ep-2")  # a fresh episode's first job

        slow = _slow_judge(seed=21)
        fast = _fast_judge(seed=22)
        decision_ep1, decision_ep2 = _run(_gather(
            _run_job(reducer, slow, self.KEY, seq_ep1, no(), episode_id="ep-1"),
            _run_job(reducer, fast, self.KEY, seq_ep2, yes(), episode_id="ep-2"),
        ))

        # ep-2's own first job is never "superseded" WITHIN ep-2, even though a key-global seq was
        # admitted for a different episode.
        self.assertFalse(decision_ep2.dropped)
        self.assertEqual(decision_ep2.action, FIRE)

        # ep-1's straggler is compared only against ep-1's OWN admissions (it was the only one), so
        # under episode_ordered it is NOT dropped either -- it is still correctly SERIALIZED (never
        # interleaved with ep-2's own gate mutation) by the per-key lock, just not treated as stale.
        self.assertFalse(decision_ep1.dropped)


class TestNoStarvationAcrossManyKeys(unittest.TestCase):
    """No watch starves: many keys' jobs share the SAME worker pool concurrently; every admitted
    job resolves to either an applied decision or a traced drop -- never a silent hang -- and the
    whole pool finishes in bounded time (a stall would time out the ``asyncio.wait_for`` below,
    turning a hang into a loud test failure instead of a silent CI wedge)."""

    NUM_KEYS = 25

    def test_every_key_resolves_and_nothing_stalls(self) -> None:
        gate = Gate(threshold=0.7, fresh_min=0.5)
        reducer = SerializedDecisionReducer(gate)
        metrics = MetricsCollector(FakeClock())

        keys: list[Hashable] = [("src-n", "plan-rev-1", f"ep-{i}") for i in range(self.NUM_KEYS)]
        jobs = []
        for i, key in enumerate(keys):
            seq_old = reducer.admit(key)
            seq_new = reducer.admit(key)
            slow = _slow_judge(seed=100 + i)
            fast = _fast_judge(seed=200 + i)
            jobs.append(_run_job(reducer, slow, key, seq_old, no()))
            jobs.append(_run_job(reducer, fast, key, seq_new, yes()))

        async def _drive():
            return await asyncio.wait_for(asyncio.gather(*jobs), timeout=5.0)

        try:
            decisions = _run(_drive())
        except asyncio.TimeoutError:
            self.fail(
                "worker pool STALLED -- a stale completion must degrade by drop-with-reason, "
                "never a hang"
            )

        for d in decisions:
            metrics.record_gate_decision(d)

        # Every key resolved to EXACTLY one applied decision (the new/fast one) and one traced
        # drop (the old/slow one) -- no key's decision pair went missing (starved).
        self.assertEqual(len(decisions), 2 * self.NUM_KEYS)
        for key in keys:
            self.assertEqual(reducer.last_applied(key), 2, f"{key}: newest evidence never applied")

        applied = [d for d in decisions if not d.dropped]
        dropped = [d for d in decisions if d.dropped]
        self.assertEqual(len(applied), self.NUM_KEYS)
        self.assertEqual(len(dropped), self.NUM_KEYS)
        self.assertTrue(all(d.drop_reason == "superseded_by_newer_admission" for d in dropped))

        self.assertEqual(metrics.total_drops(), self.NUM_KEYS)
        self.assertEqual(metrics.drop_counts(), {"superseded_by_newer_admission": self.NUM_KEYS})


if __name__ == "__main__":
    unittest.main()
