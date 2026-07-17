"""PROBE-HARNESS SCAFFOLDING — ``metrics_collector`` (Viability §6, board card A13).

**Standalone aggregation module — not runtime instrumentation.** This ingests already-produced
records/samples (from probes, fixtures, or later the real pipeline via card A7) and computes the
Viability §6 metrics catalog. It does NOT touch the runtime pipeline itself (``worker.py``,
``agent.py``, ``gate.py``, ``observability.py`` are untouched — that wiring is card A7's job).

Reuses, rather than reimplements, three already-landed pieces:

* :class:`percept.contracts.JudgmentTiming` — the frozen six-timestamp production contract and its
  latency helpers (``ingest_to_decision_s`` / ``admitted_to_decision_s`` / ``judge_wait_s`` /
  ``judge_duration_s`` / ``decision_overhead_s``). This module reads those helpers; it never
  recomputes a latency from raw timestamps itself.
* :class:`percept.harness.load_driver.LoadDriverStats` — ``observed_f`` (judge fan-out per
  admission), read back into :meth:`MetricsCollector.fanout_summary`.
* :class:`percept.sequencing.GateDecision` — ``dropped``/``drop_reason`` (stale-completion drops
  from evidence_seq supersession, card A8), read back into :meth:`MetricsCollector.drop_counts`.

**Determinism.** Every "now" this module can produce comes from exactly one injected
:data:`~percept.clock.Clock` (``percept/clock.py``) passed to :class:`MetricsCollector` at
construction — no wall-clock calls (``time.time`` / ``time.monotonic``), no ``random``/``Math.random``
anywhere in this file. Every ingestion method also accepts an explicit ``at=`` override so a probe
replaying a fixed timeline never needs the collector's own clock to be ticked in lockstep.

**Percentiles.** :func:`percentile` uses the nearest-rank method on a sorted copy of the input
(``rank = ceil(p/100 * n)``, 1-indexed, clamped to ``[1, n]``) — a fixed, well-known, dependency-free
definition (no numpy) so the same input always produces the same output bit-for-bit, and so the
p50/p95/p99 asserted in tests are exact rather than approximate/interpolated.

Downstream: card A7 emits into the shapes defined here (:class:`QueueDepthSample` /
:class:`CostSample` / :class:`RSSSample` / :class:`ClockSkewSample`, plus feeding
:class:`~percept.contracts.JudgmentTiming` / :class:`~percept.sequencing.GateDecision` /
:class:`~percept.harness.load_driver.LoadDriverStats` instances straight through); cards
A18/A21/A25/A26 consume :class:`MetricsCollector`'s aggregate reports.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Sequence

from ..clock import Clock
from ..contracts import JudgmentTiming
from ..harness.load_driver import LoadDriverStats
from ..sequencing import GateDecision

# ---------------------------------------------------------------------------
# Percentiles — deterministic nearest-rank, no numpy, no interpolation.
# ---------------------------------------------------------------------------


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile of ``values`` at ``p`` (0..100).

    ``rank = ceil(p / 100 * n)``, clamped to ``[1, n]``, 1-indexed into the SORTED values — i.e.
    ``percentile(values, 100)`` is always the max and ``percentile(values, 0)`` (with n>=1) is
    always the min. Deterministic and dependency-free: two calls with the same input always return
    the identical value, never an interpolated one.
    """
    if not values:
        raise ValueError("percentile() requires at least one value")
    if not (0.0 <= p <= 100.0):
        raise ValueError(f"p must be in [0, 100], got {p!r}")
    ordered = sorted(values)
    n = len(ordered)
    rank = max(1, min(n, math.ceil(p / 100.0 * n)))
    return ordered[rank - 1]


@dataclass(frozen=True)
class PercentileSummary:
    """p50/p95/p99 (nearest-rank, see :func:`percentile`) + the sample count they were drawn from."""

    p50: float
    p95: float
    p99: float
    n: int

    @classmethod
    def of(cls, values: Sequence[float]) -> "PercentileSummary":
        if not values:
            raise ValueError("PercentileSummary.of() requires at least one value")
        return cls(
            p50=percentile(values, 50),
            p95=percentile(values, 95),
            p99=percentile(values, 99),
            n=len(values),
        )


# ---------------------------------------------------------------------------
# Latency — five derived latencies over ingested JudgmentTiming samples.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatencyReport:
    """p50/p95/p99 for the five §6 derived latencies, each read via a :class:`JudgmentTiming`
    helper (never recomputed from raw timestamps here)."""

    ingest_to_decision: PercentileSummary
    admitted_to_decision: PercentileSummary  # the primary G3-A number
    judge_wait: PercentileSummary  # admitted -> judge_started (queue wait)
    judge_duration: PercentileSummary  # judge_started -> judge_completed (provider latency)
    decision_overhead: PercentileSummary  # judge_completed -> decision_emitted


# ---------------------------------------------------------------------------
# Queue depth + oldest-pending age — live-tracked from admit/resolve events.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueueDepthSample:
    """One point-in-time observation of the judge queue: how many admissions are pending, and how
    long (seconds) the OLDEST of them has been waiting. ``oldest_pending_age_s`` is ``0.0`` when the
    queue is empty."""

    t: float
    depth: int
    oldest_pending_age_s: float


# ---------------------------------------------------------------------------
# Cost inputs — judgments/hour + $/stream-hour, from injected (never estimated) cost figures.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostSample:
    """One windowed cost observation, injected by the caller (never estimated here).

    ``judgments`` / ``cost_usd`` are the totals observed over a window spanning ``elapsed_s`` wall
    seconds across ``stream_hours`` cumulative stream-hours (e.g. ``N`` streams live for
    ``elapsed_s`` each contribute ``elapsed_s/3600`` stream-hours)."""

    judgments: int
    cost_usd: float
    elapsed_s: float
    stream_hours: float

    def __post_init__(self) -> None:
        if self.judgments < 0:
            raise ValueError(f"CostSample.judgments must be >= 0, got {self.judgments!r}")
        if self.cost_usd < 0:
            raise ValueError(f"CostSample.cost_usd must be >= 0, got {self.cost_usd!r}")
        if self.elapsed_s < 0:
            raise ValueError(f"CostSample.elapsed_s must be >= 0, got {self.elapsed_s!r}")
        if self.stream_hours < 0:
            raise ValueError(f"CostSample.stream_hours must be >= 0, got {self.stream_hours!r}")


@dataclass(frozen=True)
class CostSummary:
    """Aggregated cost figures across every ingested :class:`CostSample`."""

    total_judgments: int
    total_cost_usd: float
    total_elapsed_s: float
    total_stream_hours: float

    @property
    def judgments_per_hour(self) -> float:
        if self.total_elapsed_s <= 0:
            return 0.0
        return self.total_judgments / (self.total_elapsed_s / 3600.0)

    @property
    def dollars_per_stream_hour(self) -> float:
        if self.total_stream_hours <= 0:
            return 0.0
        return self.total_cost_usd / self.total_stream_hours


# ---------------------------------------------------------------------------
# RSS + clock skew over time — plain timestamped samples.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RSSSample:
    t: float
    rss_bytes: int


@dataclass(frozen=True)
class ClockSkewSample:
    t: float
    skew_s: float


# ---------------------------------------------------------------------------
# Fan-out — judge calls issued per admission (F), read back from LoadDriverStats.observed_f.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FanoutSummary:
    """Judge fan-out per admission (F), aggregated across every ingested
    :class:`~percept.harness.load_driver.LoadDriverStats` snapshot."""

    observed_f_values: tuple[float, ...]
    mean_observed_f: float
    total_episodes_admitted: int
    total_judge_calls_issued: int


# ---------------------------------------------------------------------------
# The collector
# ---------------------------------------------------------------------------


class MetricsCollector:
    """Ingests probe/pipeline records and aggregates the Viability §6 metrics catalog.

    Every method that stamps a "now" accepts an explicit ``at=`` override; when omitted it reads
    the injected ``clock`` exactly once — never a wall clock. Pure aggregation: nothing here mutates
    or reaches into the runtime pipeline.
    """

    def __init__(self, clock: Clock) -> None:
        self._clock = clock

        # -- latency ---------------------------------------------------------------------------
        self._latency_samples: list[JudgmentTiming] = []

        # -- queue depth / oldest-pending age (live-tracked from admit/resolve events) ----------
        self._pending_since: dict[Hashable, float] = {}
        self._queue_samples: list[QueueDepthSample] = []

        # -- stale-completion drops --------------------------------------------------------------
        self._drop_counts: Counter[str] = Counter()
        self._drop_total = 0

        # -- judge fan-out per admission ----------------------------------------------------------
        self._fanout_stats: list[LoadDriverStats] = []

        # -- cost -----------------------------------------------------------------------------------
        self._cost_samples: list[CostSample] = []

        # -- RSS / clock skew -------------------------------------------------------------------------
        self._rss_samples: list[RSSSample] = []
        self._clock_skew_samples: list[ClockSkewSample] = []

    def _now(self, at: float | None) -> float:
        return self._clock() if at is None else at

    # -- latency: ingest + summarize ------------------------------------------------------------

    def record_judgment(self, timing: JudgmentTiming) -> None:
        """Ingest one judgment's frozen :class:`JudgmentTiming` for latency aggregation."""
        self._latency_samples.append(timing)

    def latency_summary(self) -> LatencyReport:
        """p50/p95/p99 for all five §6 derived latencies, over every ingested judgment.

        Raises :class:`ValueError` if no judgments have been recorded — matches
        :meth:`PercentileSummary.of`'s "at least one value" contract rather than silently
        returning zeros a caller could mistake for real data.
        """
        if not self._latency_samples:
            raise ValueError("latency_summary() requires at least one recorded judgment")
        return LatencyReport(
            ingest_to_decision=PercentileSummary.of(
                [t.ingest_to_decision_s() for t in self._latency_samples]
            ),
            admitted_to_decision=PercentileSummary.of(
                [t.admitted_to_decision_s() for t in self._latency_samples]
            ),
            judge_wait=PercentileSummary.of([t.judge_wait_s() for t in self._latency_samples]),
            judge_duration=PercentileSummary.of(
                [t.judge_duration_s() for t in self._latency_samples]
            ),
            decision_overhead=PercentileSummary.of(
                [t.decision_overhead_s() for t in self._latency_samples]
            ),
        )

    # -- queue depth + oldest-pending age ---------------------------------------------------------

    def record_admitted(self, key: Hashable, *, at: float | None = None) -> QueueDepthSample:
        """Mark ``key`` as newly pending (admitted into the judge funnel, not yet resolved).

        Re-admitting an already-pending key resets its wait clock (a straggler-safe overwrite, not
        a duplicate-count bug: a key is pending exactly once at a time from this tracker's view).
        """
        ts = self._now(at)
        self._pending_since[key] = ts
        return self._sample_queue(ts)

    def record_admission_resolved(self, key: Hashable, *, at: float | None = None) -> QueueDepthSample:
        """Mark ``key`` resolved (judge completed + decision emitted) — no longer pending."""
        ts = self._now(at)
        self._pending_since.pop(key, None)
        return self._sample_queue(ts)

    def _sample_queue(self, ts: float) -> QueueDepthSample:
        depth = len(self._pending_since)
        oldest_age = (ts - min(self._pending_since.values())) if self._pending_since else 0.0
        sample = QueueDepthSample(t=ts, depth=depth, oldest_pending_age_s=oldest_age)
        self._queue_samples.append(sample)
        return sample

    def queue_depth_series(self) -> tuple[QueueDepthSample, ...]:
        """Every queue-state sample taken so far, in ingestion order."""
        return tuple(self._queue_samples)

    def max_queue_depth(self) -> int:
        return max((s.depth for s in self._queue_samples), default=0)

    def max_oldest_pending_age_s(self) -> float:
        return max((s.oldest_pending_age_s for s in self._queue_samples), default=0.0)

    def current_pending_count(self) -> int:
        """Currently-pending admissions per this tracker's live view (not a historical sample)."""
        return len(self._pending_since)

    # -- stale-completion drops (evidence_seq supersession, card A8) -------------------------------

    def record_gate_decision(self, decision: GateDecision) -> None:
        """Ingest one :class:`~percept.sequencing.GateDecision`; counts it iff ``dropped``."""
        if decision.dropped:
            self._drop_total += 1
            self._drop_counts[decision.drop_reason or "unknown"] += 1

    def drop_counts(self) -> dict[str, int]:
        """Stale-completion drop counts, keyed by ``drop_reason`` (e.g.
        ``"superseded_by_newer_admission"`` / ``"superseded_within_episode"`` / ``"stale_timeout"``)."""
        return dict(self._drop_counts)

    def total_drops(self) -> int:
        return self._drop_total

    # -- judge fan-out per admission (F) ------------------------------------------------------------

    def record_load_driver_stats(self, stats: LoadDriverStats) -> None:
        """Ingest one :class:`~percept.harness.load_driver.LoadDriverStats` snapshot."""
        self._fanout_stats.append(stats)

    def fanout_summary(self) -> FanoutSummary:
        """Observed F (judge calls issued per admitted episode) across every ingested snapshot.

        ``mean_observed_f`` is the RATIO of totals (``sum(calls) / sum(episodes)``), not the mean of
        each snapshot's own ratio — correct even when snapshots cover unequal episode counts."""
        values = tuple(s.observed_f for s in self._fanout_stats)
        total_episodes = sum(s.episodes_admitted for s in self._fanout_stats)
        total_calls = sum(s.judge_calls_issued for s in self._fanout_stats)
        mean = (total_calls / total_episodes) if total_episodes else 0.0
        return FanoutSummary(
            observed_f_values=values,
            mean_observed_f=mean,
            total_episodes_admitted=total_episodes,
            total_judge_calls_issued=total_calls,
        )

    # -- judgments/hour + $/stream-hour ------------------------------------------------------------

    def record_cost_sample(self, sample: CostSample) -> None:
        """Ingest one INJECTED :class:`CostSample` window (never estimated by this module)."""
        self._cost_samples.append(sample)

    def cost_summary(self) -> CostSummary:
        return CostSummary(
            total_judgments=sum(s.judgments for s in self._cost_samples),
            total_cost_usd=sum(s.cost_usd for s in self._cost_samples),
            total_elapsed_s=sum(s.elapsed_s for s in self._cost_samples),
            total_stream_hours=sum(s.stream_hours for s in self._cost_samples),
        )

    # -- RSS + clock skew over time -------------------------------------------------------------------

    def record_rss(self, rss_bytes: int, *, at: float | None = None) -> None:
        self._rss_samples.append(RSSSample(t=self._now(at), rss_bytes=rss_bytes))

    def rss_series(self) -> tuple[RSSSample, ...]:
        return tuple(self._rss_samples)

    def record_clock_skew(self, skew_s: float, *, at: float | None = None) -> None:
        self._clock_skew_samples.append(ClockSkewSample(t=self._now(at), skew_s=skew_s))

    def clock_skew_series(self) -> tuple[ClockSkewSample, ...]:
        return tuple(self._clock_skew_samples)


__all__ = [
    "percentile",
    "PercentileSummary",
    "LatencyReport",
    "QueueDepthSample",
    "CostSample",
    "CostSummary",
    "RSSSample",
    "ClockSkewSample",
    "FanoutSummary",
    "MetricsCollector",
]
