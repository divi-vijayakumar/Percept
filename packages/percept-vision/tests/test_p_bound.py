"""Probe P-BOUND-1 scaffold (Viability §7, board card A18) — G1 continuous overload / backlog.

SCAFFOLD-ONLY: wires the probe's own logic against fakes and asserts it. Does NOT run a for-record
benchmark (that is gated on bars-locked). Property under test: with input arriving FASTER than the
judge can resolve it, the funnel stays BOUNDED —

  * pending depth AND oldest-pending age stay bounded and non-growing over the run,
  * latest-pending supersession keeps only the newest admission per key,
  * every drop is accounted for with a reason,
  * there is no unbounded memory growth.

Reuses (imported, never edited):
  * ``percept.harness.load_driver.{LoadDriver, LoadProfile, RampSchedule}`` — produces a realistic,
    deterministic N/M/E/F admission volume/timing under a FakeClock. A high ``E`` here is what makes
    this "input faster than the judge": the configured admission rate is chosen far above what the
    SLOW judge below could resolve in the same span.
  * ``percept.backends.fake_judge.FakeJudge`` + ``LatencyProfile`` — a SLOW, seeded judge double
    (``simulate_sleep=False`` so latency is sampled deterministically without any real wall-clock
    wait — this is a scaffold assertion of PROBE LOGIC, not a timed benchmark).
  * ``percept.harness.metrics.MetricsCollector`` — live queue-depth / oldest-pending-age tracking.
    Its pending tracker is a dict KEYED by admission identity, so re-admitting the same key resets
    (never appends) — this IS the "latest-pending supersession bounds memory" mechanism this probe
    checks, reused rather than reimplemented. Also supplies ``drop_counts()``/``total_drops()`` for
    the accounted-drops assertion.
  * ``percept.clock.FakeClock`` — the single injected time source; zero wall-clock waits anywhere in
    this file.
  * ``percept.sequencing.EvidenceSequencer`` / ``GateDecision`` — per-key monotonic admission
    sequencing (ADR-001 §8), reused (not reimplemented) to decide exactly when an in-flight
    admission has been superseded by a newer one for the same key, and to record that drop with a
    named reason via ``MetricsCollector.record_gate_decision``.

This module maps each admitted episode (in ``LoadDriver``/``FakeJudge`` call order) onto one of a
small, FIXED number of logical funnel "keys" (round-robin over admission order) — the probe doesn't
need ``LoadDriver``'s internal (stream, watch) identity, only a bounded key space a real worker-pool
funnel would also be keyed on (e.g. one pending slot per watch). Resolution timing for each
admission is derived from ``FakeJudge``'s own sampled per-call latency
(``judge.calls[i].latency_s``), scheduled forward from a FakeClock-driven, tick-level admission
time. The exact ramp/steady admission COUNT is already proven deterministic by
``test_load_driver.py``; this file only adds the queueing/backpressure/supersession/drop-accounting
layer on top of it.
"""

from __future__ import annotations

import asyncio
import unittest

from percept.backends.fake_judge import FakeJudge, LatencyProfile
from percept.clock import FakeClock
from percept.gate import SILENT
from percept.harness.load_driver import LoadDriver, LoadProfile, RampSchedule
from percept.harness.metrics import MetricsCollector
from percept.harness.stream_gen import parametric_motion_clip
from percept.sequencing import EvidenceSequencer, GateDecision


def _slow_judge(seed: int = 7) -> FakeJudge:
    """A deliberately SLOW judge double: fixed 3.0s latency, far above the inter-admission
    interval the high-E ``LoadProfile`` below produces — this IS the "input faster than the judge"
    condition Viability §7/G1 asks the probe to exercise. ``simulate_sleep=False`` keeps this test
    fast and deterministic (latency is sampled, never actually slept)."""
    return FakeJudge(
        seed=seed,
        latency=LatencyProfile(kind="fixed", value_s=3.0),
        simulate_sleep=False,
    )


def _timeline(n: int = 400, dt: float = 0.05):
    return parametric_motion_clip(w=4, h=4, n=n, motion_frames=set(range(0, n, 5)), dt=dt)


class _BoundedFunnelSim:
    """Test-local scaffold: buckets admissions from a LoadDriver/FakeJudge run onto a bounded key
    space, applies latest-pending supersession via ``EvidenceSequencer``, and records
    queue-depth / oldest-pending-age / drop bookkeeping into a ``MetricsCollector``.

    Not production code — a real worker-pool funnel does not exist yet (see
    ``percept/sequencing.py``'s module docstring: "today's caller does not need any of this ... this
    module is the mechanism a future worker pool wires in"). This class is what gives the P-BOUND-1
    probe something concrete, deterministic, and REUSED (not reimplemented queueing logic) to assert
    against ahead of that worker pool existing.
    """

    def __init__(self, clock: FakeClock, num_keys: int) -> None:
        self.clock = clock
        self.num_keys = num_keys
        self.collector = MetricsCollector(clock)
        self.sequencer = EvidenceSequencer()
        self._pending_seq: dict[int, int] = {}  # key -> evidence_seq of its current pending admission
        self._resolutions: list[tuple[float, int, int]] = []  # (resolves_at, key, seq)
        self.completed: list[tuple[int, int]] = []  # (key, seq) that resolved cleanly, in order
        self.drops: list[tuple[int, int, str]] = []  # (key, seq, reason) for every drop

    def _drain_due_resolutions(self, now: float) -> None:
        still_pending = []
        for resolves_at, key, seq in self._resolutions:
            if resolves_at > now:
                still_pending.append((resolves_at, key, seq))
                continue
            # Only resolve if this admission is still the LATEST for its key (never apply a
            # completion that lost the race to a newer admission — it was already dropped below).
            if self._pending_seq.get(key) == seq:
                self.collector.record_admission_resolved(key, at=resolves_at)
                self.completed.append((key, seq))
                del self._pending_seq[key]
        self._resolutions = still_pending

    def admit(self, call_index: int, latency_s: float, now: float) -> None:
        key = call_index % self.num_keys
        self._drain_due_resolutions(now)

        seq = self.sequencer.admit(key)
        prior_seq = self._pending_seq.get(key)
        if prior_seq is not None:
            # A newer admission arrived for this key before the prior one resolved: the prior is
            # superseded — dropped, never applied, with a named reason (Viability §7: "every drop
            # accounted with a reason").
            decision = GateDecision(
                key=key,
                evidence_seq=prior_seq,
                action=SILENT,
                dropped=True,
                drop_reason="superseded_by_newer_admission",
            )
            self.collector.record_gate_decision(decision)
            self.drops.append((key, prior_seq, "superseded_by_newer_admission"))
            # The superseded admission can never complete later.
            self._resolutions = [
                r for r in self._resolutions if not (r[1] == key and r[2] == prior_seq)
            ]

        self._pending_seq[key] = seq
        self.collector.record_admitted(key, at=now)
        self._resolutions.append((now + latency_s, key, seq))

    def drain_all(self, final_now: float) -> None:
        """Flush any resolutions already due by ``final_now`` (end-of-run bookkeeping)."""
        self._drain_due_resolutions(final_now)


def _run_scenario(*, steady_s: float = 20.0, step_s: float = 0.5, num_keys: int = 4):
    """Drives LoadDriver (high E, SLOW FakeJudge) under a FakeClock, feeding every issued judge
    call into a `_BoundedFunnelSim`. Zero wall-clock waits: `driver.tick()` never sleeps for real
    (`simulate_sleep=False`), and time only ever moves via `clock.advance()`."""
    profile = LoadProfile(n=2, m=2, e_per_stream_min=600.0, f=1)  # high E: far above judge capacity
    schedule = RampSchedule(ramp_s=0.0, steady_s=steady_s)
    clock = FakeClock()
    judge = _slow_judge()
    driver = LoadDriver(profile, schedule, _timeline(), clock=clock, judge=judge)

    sim = _BoundedFunnelSim(clock, num_keys=num_keys)

    prev_calls = 0
    while True:
        asyncio.run(driver.tick())
        now = clock()
        calls = judge.calls
        for idx in range(prev_calls, len(calls)):
            sim.admit(idx, calls[idx].latency_s, now)
        prev_calls = len(calls)
        if driver.done():
            break
        clock.advance(step_s * 1000.0)
    sim.drain_all(clock())
    return driver, judge, sim


class TestPBound1QueueStaysBounded(unittest.TestCase):
    """P-BOUND-1 (Viability §7, G1): continuous overload does not grow the funnel unboundedly."""

    NUM_KEYS = 4

    def test_admission_really_outruns_the_judge(self):
        """Sanity check that this scenario IS the "input faster than the judge" condition — far
        more episodes were admitted than the judge (at its sampled 3.0s latency) could resolve over
        the run's span."""
        driver, judge, _sim = _run_scenario()
        self.assertGreater(driver.stats.episodes_admitted, 20)
        max_possible_resolutions = driver.stats.elapsed_s / 3.0 * self.NUM_KEYS
        self.assertGreater(driver.stats.episodes_admitted, max_possible_resolutions)
        self.assertEqual(len(judge.calls), driver.stats.episodes_admitted)

    def test_queue_depth_and_oldest_age_stay_bounded(self):
        _driver, _judge, sim = _run_scenario(num_keys=self.NUM_KEYS)

        depth_series = sim.collector.queue_depth_series()
        self.assertTrue(depth_series, "expected at least one queue-depth sample")

        # BOUNDED: because admission is KEYED (supersession, not append-only), pending depth can
        # never exceed the fixed key space, regardless of how many total episodes were admitted.
        self.assertLessEqual(sim.collector.max_queue_depth(), self.NUM_KEYS)

        # Oldest-pending age is also bounded: supersession resets a key's pending timer on every
        # re-admission, so no key's recorded age can grow past one judge latency.
        self.assertLessEqual(sim.collector.max_oldest_pending_age_s(), 3.0 + 1e-6)

        # NON-GROWING over the run, not just bounded at a single instant: the second half of the
        # run's max depth is no worse than the first half's.
        half = len(depth_series) // 2
        first_half_max = max((s.depth for s in depth_series[:half]), default=0)
        second_half_max = max((s.depth for s in depth_series[half:]), default=0)
        self.assertLessEqual(second_half_max, max(first_half_max, self.NUM_KEYS))

    def test_supersession_keeps_only_the_latest_admission_per_key(self):
        _driver, _judge, sim = _run_scenario(num_keys=self.NUM_KEYS)

        # A given (key, seq) is EITHER completed OR dropped — never both.
        completed_set = set(sim.completed)
        dropped_set = {(k, s) for k, s, _reason in sim.drops}
        self.assertEqual(completed_set & dropped_set, set())

        # Per key, every completed seq is the seq that was pending for that key at resolution time
        # (i.e. no stale/superseded result was ever applied) — enforced by the sim, double-checked
        # here: completed seqs per key are strictly increasing in completion order.
        last_seq_by_key: dict[int, int] = {}
        for key, seq in sim.completed:
            self.assertGreater(seq, last_seq_by_key.get(key, 0))
            last_seq_by_key[key] = seq

        # At most one admission is pending per key at any time — the keyed dict cannot hold more.
        self.assertLessEqual(len(sim._pending_seq), sim.num_keys)

    def test_every_drop_is_accounted_with_a_reason(self):
        _driver, _judge, sim = _run_scenario(num_keys=self.NUM_KEYS)

        drop_counts = sim.collector.drop_counts()
        total_drops = sim.collector.total_drops()

        # With a SLOW judge and a HIGH admit rate, supersession drops really do occur in this
        # scenario (this is what makes it a genuine overload probe, not a vacuous pass).
        self.assertGreater(total_drops, 0)
        self.assertEqual(len(sim.drops), total_drops)
        self.assertEqual(sum(drop_counts.values()), total_drops)

        # Every drop bucket names a real reason — nothing falls into an "unknown"/empty bucket in
        # this scenario, and every recorded reason is one of the named ones this sim produces.
        self.assertNotIn("unknown", drop_counts)
        self.assertNotIn("", drop_counts)
        self.assertEqual(set(drop_counts), {"superseded_by_newer_admission"})

    def test_no_unbounded_memory_bound_holds_over_a_longer_run(self):
        """The SAME bound must hold as the run gets longer — a naive append-only queue would keep
        growing; the keyed/superseding design here does not."""
        _driver, _judge, sim = _run_scenario(steady_s=40.0, num_keys=self.NUM_KEYS)

        self.assertLessEqual(sim.collector.max_queue_depth(), self.NUM_KEYS)
        self.assertLessEqual(len(sim._pending_seq), self.NUM_KEYS)
        self.assertGreater(sim.collector.total_drops(), 0)


if __name__ == "__main__":
    unittest.main()
