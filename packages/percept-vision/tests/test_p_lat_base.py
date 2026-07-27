"""Probe P-LAT-BASE-1 — single-call baseline latency (the release criteria).

**What this measures.** Each judge's raw, isolated single-frame call latency: N independent
``FakeJudge.judge()`` calls, no stream, no queueing, no concurrent load, no admission funnel. This
is a DIAGNOSTIC FLOOR ONLY — the best case a judge call could ever look like, stripped of every
runtime overhead (queue wait, fan-out, decision assembly). It is NEVER the product number: a real
deployment's admitted-to-decision latency (queueing + fan-out + decision overhead layered on top)
will always be >= this floor, often by a wide margin under load. See P-BOUND-1/P-LOAD probes for
the *product* latency number.

**Why this probe exists (regression bar).** The diagnostic floor is where a judge round-trip
regression shows up first and most cleanly. If the isolated single-call p95 has drifted past the
bar in the *best possible* case, every product number downstream (queueing + fan-out + decision
overhead layered on top) has already drifted further. See
``test_diagnostic_floor_judge_round_trip_p95_bar`` below, which encodes that as an executable
assertion against :data:`percept.slo.JUDGE_ROUND_TRIP_P95_CEILING_S`.

**What this probe deliberately no longer claims.** It used to encode a *feasibility verdict*: a
single-call p95 >= 2s meant a "Reflex" tier was IMPOSSIBLE. That framing has been retired. Reflex is
not a latency tier and never was enforceable as one — it is the cheap, terse, bounded-evidence path
(bounded frames in, two-field verdict out), and the 2s figure it was judged against was a deadline
the pipeline was never measured to clear. Holding that deadline is what produced the defect this
bar replaces: it sat below the median provider call time, so the majority of judgments were
cancelled rather than answered, and the resulting latency "measurement" was just the deadline
reflected back. The bar below is therefore a REGRESSION bar derived from measured, uncensored data
(``percept.slo`` §5), not a verdict on whether a responsiveness tier is achievable.

**Reuse only — no source edits.** This file imports, unmodified:
  * ``percept.backends.fake_judge.FakeJudge`` / ``LatencyProfile`` — seeded, key-free judge double
    with configurable per-call latency (the latency measurement contract/§6).
  * ``percept.harness.metrics.MetricsCollector`` / ``PercentileSummary`` — nearest-rank p50/p95/p99
    aggregation (the benchmark metrics contract).
  * ``percept.clock.FakeClock`` — deterministic clock (unused for latency math itself here, since
    ``JudgmentTiming`` stamps are built directly from ``FakeJudge``'s recorded per-call latencies,
    but threaded through ``MetricsCollector`` per its constructor contract).
  * ``percept.contracts.JudgmentTiming`` — the production timing record; each isolated call is
    represented as a "degenerate" funnel where admitted == judge_started and judge_completed ==
    decision_emitted (zero queue wait, zero decision overhead) so ``judge_duration_s()`` is EXACTLY
    the sampled single-call latency, and ``MetricsCollector.latency_summary()`` is exercised via its
    real, unmodified aggregation path rather than a bespoke percentile helper.

No new source files and no ``__init__``/``conftest`` additions: this scaffold is self-contained.
"""

from __future__ import annotations

import asyncio
import unittest

from percept.backends.fake_judge import FakeJudge, LatencyProfile
from percept.clock import FakeClock
from percept.contracts import JudgmentTiming
from percept.harness.metrics import MetricsCollector, PercentileSummary, percentile
from percept.slo import JUDGE_ROUND_TRIP_P95_CEILING_S

N_CALLS = 1000


def _run_isolated_calls(judge: FakeJudge, n: int) -> list[float]:
    """Drive ``n`` independent, isolated ``judge()`` calls (no stream, no concurrency, no load) and
    return the per-call sampled latency in call order. ``simulate_sleep=False`` on the judge keeps
    the test fast; the sampled latency is recorded into ``judge.calls`` regardless (see
    ``FakeJudge.judge`` — latency is sampled + recorded before the sleep-or-not branch), so this
    does not change what is being measured, only whether wall-clock time is actually spent waiting.
    """

    async def _drive() -> None:
        for i in range(n):
            await judge.judge("isolated single-call baseline probe", f"frame-{i}")

    asyncio.run(_drive())
    return list(judge.latencies)


def _isolated_call_timings(latencies: list[float]) -> list[JudgmentTiming]:
    """Build one degenerate :class:`JudgmentTiming` per isolated call: admitted == judge_started
    (no queue wait — nothing was queued) and judge_completed == decision_emitted (no decision
    overhead — nothing beyond the raw call). ``judge_duration_s()`` is therefore exactly the sampled
    single-call latency, letting this probe feed ``MetricsCollector``'s real, unmodified aggregation
    path instead of reimplementing percentile math.

    Each record starts from ``started=0.0`` rather than a running offset across calls: these are
    ISOLATED calls (no stream, no shared timeline), ``JudgmentTiming``'s monotonicity invariant is
    checked only *within* one record's own five stamps (never across records), and a running
    float-accumulated offset would reintroduce floating-point drift into
    ``judge_completed_ts - judge_started_ts`` that the raw samples never had — exactly what this
    probe must NOT do, since it asserts exact (not approximately-equal) percentile recovery.
    """
    timings = []
    for lat in latencies:
        timings.append(
            JudgmentTiming(
                ingest_observed_ts=0.0,
                admitted_ts=0.0,
                judge_started_ts=0.0,
                judge_completed_ts=lat,
                decision_emitted_ts=lat,
            )
        )
    return timings


class TestPLatBase1SingleCallBaseline(unittest.TestCase):
    """P-LAT-BASE-1: the harness must recover an injected latency distribution's percentiles from
    isolated single-call samples, deterministically under a fixed seed."""

    def _collect(self, latency: LatencyProfile, *, seed: int) -> tuple[list[float], PercentileSummary]:
        judge = FakeJudge(seed=seed, latency=latency, simulate_sleep=False)
        latencies = _run_isolated_calls(judge, N_CALLS)
        self.assertEqual(len(latencies), N_CALLS)

        collector = MetricsCollector(clock=FakeClock(0.0))
        for timing in _isolated_call_timings(latencies):
            collector.record_judgment(timing)

        recovered = collector.latency_summary().judge_duration
        return latencies, recovered

    def test_uniform_profile_percentiles_recovered_exactly(self):
        latency = LatencyProfile(kind="uniform", low_s=0.05, high_s=0.15)
        raw, recovered = self._collect(latency, seed=42)

        # Ground truth: percentiles computed directly (same nearest-rank definition,
        # percept.harness.metrics.percentile) over FakeJudge's own recorded samples — the injected
        # distribution's realized percentiles for this exact seed + call sequence.
        expected = PercentileSummary.of(raw)

        self.assertEqual(recovered, expected)
        self.assertEqual(recovered.n, N_CALLS)
        # Sanity: percentiles land inside the profile's support and are non-decreasing.
        self.assertLessEqual(recovered.p50, recovered.p95)
        self.assertLessEqual(recovered.p95, recovered.p99)
        self.assertGreaterEqual(recovered.p50, latency.low_s)
        self.assertLessEqual(recovered.p99, latency.high_s)

    def test_lognormal_profile_percentiles_recovered_exactly(self):
        # mu/sigma chosen so the bulk of mass sits well under 1s (a plausible real single-call
        # judge latency shape: a long tail of slow calls) while still exercising the tail.
        latency = LatencyProfile(kind="lognormal", mu=-2.5, sigma=0.6)
        raw, recovered = self._collect(latency, seed=7)

        expected = PercentileSummary.of(raw)

        self.assertEqual(recovered, expected)
        self.assertEqual(recovered.n, N_CALLS)
        self.assertLessEqual(recovered.p50, recovered.p95)
        self.assertLessEqual(recovered.p95, recovered.p99)

    def test_deterministic_under_seed(self):
        """Same seed + same isolated call sequence -> byte-identical recovered percentiles, run
        through the full FakeJudge -> JudgmentTiming -> MetricsCollector path twice independently."""
        latency = LatencyProfile(kind="uniform", low_s=0.02, high_s=0.08)

        raw1, recovered1 = self._collect(latency, seed=99)
        raw2, recovered2 = self._collect(latency, seed=99)

        self.assertEqual(raw1, raw2)
        self.assertEqual(recovered1, recovered2)

    def test_diagnostic_floor_judge_round_trip_p95_bar(self):
        """Executable form of the regression-bar comment above.

        This is a SCAFFOLD-ONLY assertion over a FAKE, injected distribution — it proves the
        harness's floor-vs-bar arithmetic, not any real provider's number (no real provider is
        called anywhere in this repo per standing rule; NO real keys/providers here). The real
        check happens later, for-record, against an actual vendor's measured p95.

        THE BAR IS MEASURED, NOT ASPIRATIONAL. It is imported, not restated, from
        :data:`percept.slo.JUDGE_ROUND_TRIP_P95_CEILING_S` — the highest measured UNCENSORED p95
        (deliberate, 2.2834s) plus the tree's own p95 drift tolerance (0.2s, the
        ``latency_drift_within_tolerance`` default in ``test_p_lat_load.py``), rounded up: 2.5s.
        That is LOOSER than the 2.0s it replaces, and deliberately so — 2.0s was a figure the
        pipeline was never measured to clear, kept alive by a deadline that discarded the
        contradicting samples. A bar above the measurement still catches a regression; a bar below
        it only guarantees a red build. It tightens again when the reflex-uncensored run lands.
        """
        # A profile deliberately shaped so its p95 crosses the bar, to prove the condition
        # actually fires (not just that it could).
        slow_latency = LatencyProfile(kind="uniform", low_s=1.5, high_s=3.0)
        _, slow_summary = self._collect(slow_latency, seed=11)
        self.assertGreaterEqual(slow_summary.p95, JUDGE_ROUND_TRIP_P95_CEILING_S)
        within_bar = slow_summary.p95 < JUDGE_ROUND_TRIP_P95_CEILING_S
        self.assertFalse(
            within_bar,
            f"diagnostic floor p95 >= {JUDGE_ROUND_TRIP_P95_CEILING_S:g}s => judge round-trip "
            "has regressed past the measured bar",
        )

        # And a fast profile, to prove the bar does NOT fire when the floor is well under it.
        fast_latency = LatencyProfile(kind="uniform", low_s=0.05, high_s=0.2)
        _, fast_summary = self._collect(fast_latency, seed=12)
        self.assertLess(fast_summary.p95, JUDGE_ROUND_TRIP_P95_CEILING_S)

    def test_percentile_helper_agrees_with_collector(self):
        """Cross-check: the bare ``percentile()`` function (used directly, without building
        JudgmentTiming/MetricsCollector plumbing) agrees with the collector's aggregation, on the
        same raw samples."""
        latency = LatencyProfile(kind="fixed", value_s=0.33)
        raw, recovered = self._collect(latency, seed=0)

        self.assertEqual(percentile(raw, 50), recovered.p50)
        self.assertEqual(percentile(raw, 95), recovered.p95)
        self.assertEqual(percentile(raw, 99), recovered.p99)
        # Fixed profile: every sample is identical, so every percentile is exactly value_s.
        self.assertEqual(recovered.p50, 0.33)
        self.assertEqual(recovered.p99, 0.33)


if __name__ == "__main__":
    unittest.main()
