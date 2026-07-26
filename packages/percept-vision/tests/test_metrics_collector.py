"""the implementation: probe-harness ``metrics_collector`` (the benchmark metrics contract metrics catalog).

Proves, deterministically (a :class:`~percept.clock.FakeClock` throughout, no wall-clock, no
randomness), that :class:`~percept.harness.metrics.MetricsCollector` correctly aggregates:

  * the five §6 derived latencies at EXACT p50/p95/p99 (nearest-rank) from ingested
    :class:`~percept.contracts.JudgmentTiming` records.
  * queue depth + oldest-pending age, live-tracked from admit/resolve events.
  * stale-completion drop counts, from ingested :class:`~percept.sequencing.GateDecision` records.
  * judge fan-out per admission (F), from ingested
    :class:`~percept.harness.load_driver.LoadDriverStats` snapshots.
  * judgments-per-hour + $/stream-hour, from injected (never estimated) cost inputs.
  * RSS + clock skew series over time.
"""

from __future__ import annotations

import unittest

from percept.clock import FakeClock
from percept.contracts import JudgmentTiming
from percept.harness.load_driver import LoadDriverStats
from percept.harness.metrics import (
    CostSample,
    MetricsCollector,
    PercentileSummary,
    percentile,
)
from percept.sequencing import GateDecision
from percept.gate import FIRE, SILENT


def _timing(ingest: float, admitted: float, started: float, completed: float, decided: float) -> JudgmentTiming:
    return JudgmentTiming(
        ingest_observed_ts=ingest,
        admitted_ts=admitted,
        judge_started_ts=started,
        judge_completed_ts=completed,
        decision_emitted_ts=decided,
    )


class TestPercentile(unittest.TestCase):
    """Nearest-rank percentile — exact, deterministic, no interpolation."""

    def test_known_distribution_exact_p50_p95_p99(self):
        # 1..100 — nearest-rank: rank = ceil(p/100 * n), 1-indexed.
        values = list(range(1, 101))
        self.assertEqual(percentile(values, 50), 50)  # ceil(0.50*100)=50 -> values[49]=50
        self.assertEqual(percentile(values, 95), 95)  # ceil(0.95*100)=95 -> values[94]=95
        self.assertEqual(percentile(values, 99), 99)  # ceil(0.99*100)=99 -> values[98]=99

    def test_single_value(self):
        self.assertEqual(percentile([42.0], 50), 42.0)
        self.assertEqual(percentile([42.0], 99), 42.0)
        self.assertEqual(percentile([42.0], 0), 42.0)

    def test_unsorted_input_order_independent(self):
        values = [5, 1, 4, 2, 3]
        self.assertEqual(percentile(values, 100), 5)
        self.assertEqual(percentile(values, 0), 1)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            percentile([], 50)

    def test_out_of_range_p_raises(self):
        with self.assertRaises(ValueError):
            percentile([1.0, 2.0], 101)
        with self.assertRaises(ValueError):
            percentile([1.0, 2.0], -1)

    def test_percentile_summary_of(self):
        values = list(range(1, 101))
        summary = PercentileSummary.of(values)
        self.assertEqual(summary, PercentileSummary(p50=50, p95=95, p99=99, n=100))


class TestLatencySummary(unittest.TestCase):
    """The five §6 derived latencies, at exact p50/p95/p99, over ingested JudgmentTiming."""

    def test_exact_percentiles_over_known_timings(self):
        clock = FakeClock()
        collector = MetricsCollector(clock)
        # 10 judgments, i-th one (i = 1..10): ingest=-i, admitted=0, judge_started=i,
        # judge_completed=2i, decision=3i. Monotonic (ingest<=admitted<=started<=completed<=decided)
        # for every i>=1, and each of the five derived latencies is then a SIMPLE, distinct
        # arithmetic progression over i so the nearest-rank percentiles are hand-verifiable:
        #   judge_wait_s        = started  - admitted = i        -> {1..10}
        #   judge_duration_s    = completed - started = i        -> {1..10}
        #   decision_overhead_s = decided  - completed = i       -> {1..10}
        #   admitted_to_decision_s = decided - admitted = 3i     -> {3,6,...,30}
        #   ingest_to_decision_s   = decided - ingest   = 4i     -> {4,8,...,40}
        for i in range(1, 11):
            collector.record_judgment(_timing(float(-i), 0.0, float(i), float(2 * i), float(3 * i)))

        report = collector.latency_summary()

        # n=10 nearest-rank: p50 -> rank=ceil(0.5*10)=5 -> 5th-smallest; p95/p99 -> rank=10 -> max.
        self.assertEqual(report.judge_wait, PercentileSummary(p50=5.0, p95=10.0, p99=10.0, n=10))
        self.assertEqual(report.judge_duration, PercentileSummary(p50=5.0, p95=10.0, p99=10.0, n=10))
        self.assertEqual(report.decision_overhead, PercentileSummary(p50=5.0, p95=10.0, p99=10.0, n=10))
        self.assertEqual(
            report.admitted_to_decision, PercentileSummary(p50=15.0, p95=30.0, p99=30.0, n=10)
        )
        self.assertEqual(
            report.ingest_to_decision, PercentileSummary(p50=20.0, p95=40.0, p99=40.0, n=10)
        )

    def test_empty_raises(self):
        collector = MetricsCollector(FakeClock())
        with self.assertRaises(ValueError):
            collector.latency_summary()


class TestQueueDepthAndOldestPendingAge(unittest.TestCase):
    def test_depth_and_oldest_age_tracked_across_admit_resolve(self):
        clock = FakeClock()
        collector = MetricsCollector(clock)

        s1 = collector.record_admitted("k1", at=0.0)
        self.assertEqual(s1.depth, 1)
        self.assertEqual(s1.oldest_pending_age_s, 0.0)

        s2 = collector.record_admitted("k2", at=2.0)
        self.assertEqual(s2.depth, 2)
        # k1 admitted at t=0, sample taken at t=2 -> oldest pending age = 2.0
        self.assertEqual(s2.oldest_pending_age_s, 2.0)

        s3 = collector.record_admitted("k3", at=5.0)
        self.assertEqual(s3.depth, 3)
        self.assertEqual(s3.oldest_pending_age_s, 5.0)

        s4 = collector.record_admission_resolved("k1", at=6.0)
        self.assertEqual(s4.depth, 2)
        # oldest pending is now k2 (admitted t=2), sample at t=6 -> age 4.0
        self.assertEqual(s4.oldest_pending_age_s, 4.0)

        self.assertEqual(collector.max_queue_depth(), 3)
        self.assertEqual(collector.max_oldest_pending_age_s(), 5.0)
        self.assertEqual(collector.current_pending_count(), 2)
        self.assertEqual(len(collector.queue_depth_series()), 4)

        collector.record_admission_resolved("k2", at=7.0)
        s6 = collector.record_admission_resolved("k3", at=8.0)
        self.assertEqual(s6.depth, 0)
        self.assertEqual(s6.oldest_pending_age_s, 0.0)
        self.assertEqual(collector.current_pending_count(), 0)

    def test_uses_injected_clock_when_at_omitted(self):
        clock = FakeClock(start_s=100.0)
        collector = MetricsCollector(clock)
        sample = collector.record_admitted("k1")
        self.assertEqual(sample.t, 100.0)
        clock.advance(3000.0)  # +3s
        sample2 = collector.record_admitted("k2")
        self.assertEqual(sample2.t, 103.0)
        self.assertEqual(sample2.oldest_pending_age_s, 3.0)


class TestStaleCompletionDrops(unittest.TestCase):
    def test_counts_drops_by_reason_and_ignores_applied(self):
        collector = MetricsCollector(FakeClock())
        collector.record_gate_decision(
            GateDecision(key="k1", evidence_seq=1, action=SILENT, dropped=True,
                         drop_reason="superseded_by_newer_admission")
        )
        collector.record_gate_decision(
            GateDecision(key="k1", evidence_seq=2, action=FIRE, dropped=False)
        )
        collector.record_gate_decision(
            GateDecision(key="k2", evidence_seq=1, action=SILENT, dropped=True,
                         drop_reason="stale_timeout")
        )
        collector.record_gate_decision(
            GateDecision(key="k2", evidence_seq=2, action=SILENT, dropped=True,
                         drop_reason="superseded_by_newer_admission")
        )

        self.assertEqual(collector.total_drops(), 3)
        self.assertEqual(
            collector.drop_counts(),
            {"superseded_by_newer_admission": 2, "stale_timeout": 1},
        )

    def test_no_drops_is_empty(self):
        collector = MetricsCollector(FakeClock())
        collector.record_gate_decision(GateDecision(key="k1", evidence_seq=1, action=FIRE))
        self.assertEqual(collector.total_drops(), 0)
        self.assertEqual(collector.drop_counts(), {})


class TestFanoutSummary(unittest.TestCase):
    def test_fanout_per_admission_from_load_driver_stats(self):
        collector = MetricsCollector(FakeClock())
        stats1 = LoadDriverStats(episodes_admitted=10, judge_calls_issued=30)  # F=3
        stats2 = LoadDriverStats(episodes_admitted=5, judge_calls_issued=10)  # F=2
        collector.record_load_driver_stats(stats1)
        collector.record_load_driver_stats(stats2)

        summary = collector.fanout_summary()
        self.assertEqual(summary.observed_f_values, (3.0, 2.0))
        self.assertEqual(summary.total_episodes_admitted, 15)
        self.assertEqual(summary.total_judge_calls_issued, 40)
        # ratio-of-totals, not mean-of-ratios: 40/15, not (3+2)/2=2.5
        self.assertAlmostEqual(summary.mean_observed_f, 40 / 15)

    def test_empty_is_zero(self):
        collector = MetricsCollector(FakeClock())
        summary = collector.fanout_summary()
        self.assertEqual(summary.observed_f_values, ())
        self.assertEqual(summary.mean_observed_f, 0.0)


class TestCostSummary(unittest.TestCase):
    def test_dollars_per_stream_hour_and_judgments_per_hour(self):
        collector = MetricsCollector(FakeClock())
        # window 1: 3600s elapsed, 2 streams live the whole window -> 2 stream-hours, 100 judgments, $1.00
        collector.record_cost_sample(
            CostSample(judgments=100, cost_usd=1.00, elapsed_s=3600.0, stream_hours=2.0)
        )
        # window 2: another 3600s, 100 judgments, $1.00, 2 stream-hours
        collector.record_cost_sample(
            CostSample(judgments=100, cost_usd=1.00, elapsed_s=3600.0, stream_hours=2.0)
        )

        summary = collector.cost_summary()
        self.assertEqual(summary.total_judgments, 200)
        self.assertAlmostEqual(summary.total_cost_usd, 2.00)
        self.assertAlmostEqual(summary.total_elapsed_s, 7200.0)
        self.assertAlmostEqual(summary.total_stream_hours, 4.0)
        # judgments/hour = 200 judgments / (7200s / 3600) = 200 / 2 = 100/hr
        self.assertAlmostEqual(summary.judgments_per_hour, 100.0)
        # $/stream-hour = $2.00 / 4 stream-hours = $0.50/stream-hour
        self.assertAlmostEqual(summary.dollars_per_stream_hour, 0.50)

    def test_no_samples_is_zero_not_division_error(self):
        collector = MetricsCollector(FakeClock())
        summary = collector.cost_summary()
        self.assertEqual(summary.judgments_per_hour, 0.0)
        self.assertEqual(summary.dollars_per_stream_hour, 0.0)

    def test_rejects_negative_inputs(self):
        with self.assertRaises(ValueError):
            CostSample(judgments=-1, cost_usd=1.0, elapsed_s=1.0, stream_hours=1.0)
        with self.assertRaises(ValueError):
            CostSample(judgments=1, cost_usd=-1.0, elapsed_s=1.0, stream_hours=1.0)


class TestRSSAndClockSkewSeries(unittest.TestCase):
    def test_series_recorded_over_time_via_injected_clock(self):
        clock = FakeClock()
        collector = MetricsCollector(clock)

        collector.record_rss(100_000_000)
        clock.advance(1000.0)
        collector.record_rss(105_000_000)
        clock.advance(1000.0)
        collector.record_rss(110_000_000, at=999.0)  # explicit override

        series = collector.rss_series()
        self.assertEqual([s.rss_bytes for s in series], [100_000_000, 105_000_000, 110_000_000])
        self.assertEqual([s.t for s in series], [0.0, 1.0, 999.0])

        collector.record_clock_skew(0.001, at=0.0)
        collector.record_clock_skew(-0.002, at=1.0)
        skew_series = collector.clock_skew_series()
        self.assertEqual([s.skew_s for s in skew_series], [0.001, -0.002])
        self.assertEqual([s.t for s in skew_series], [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
