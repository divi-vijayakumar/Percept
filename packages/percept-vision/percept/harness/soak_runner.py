"""PROBE-HARNESS SCAFFOLDING — ``soak_runner``.

**Wiring-only, not a production driver.** A long-run soak: steady load held for many simulated
*hours*, with an hourly :class:`~percept.harness.metrics.MetricsCollector` snapshot and a
hour-``N``-vs-hour-1 comparison of the three drift signals this soak tracks — queue-age slope,
memory slope, latency drift — plus support for an injected provider-failure window with an
assertable clean recovery. Standing rule 2 (see :mod:`percept.backends.fake`) still applies: this
proves wiring/architecture, never model intelligence.

Reuses, rather than reimplements, four already-landed pieces (imported, never edited):

* :class:`percept.harness.load_driver.LoadDriver` — the steady-state ``N/M/E/F`` load shape. One
  fresh driver is built per simulated hour (``RampSchedule(ramp_s=0.0, steady_s=hour_s)`` — no
  ramp, matching a soak's "already at steady state" premise), driven by the SAME clock across
  hour boundaries so the run is one continuous simulated timeline. The outer per-tick loop
  (:meth:`SoakRunner.run`) mirrors :meth:`LoadDriver.run_to_completion`'s own
  tick-then-``clock.advance()`` shape — that method's docstring explicitly sanctions this: "drive
  ``tick()`` manually against other clocks" — done here only so this module can hook each tick's
  newly-issued judge calls into the metrics collector.
* :class:`percept.backends.fake_judge.FakeJudge` — a fresh instance built per hour so injected
  provider failures (:class:`~percept.backends.fake_judge.ErrorInjectionConfig`) can be scoped to
  exactly the hours in :attr:`SoakConfig.failure_hours`, and recovered from cleanly outside them.
* :class:`percept.harness.metrics.MetricsCollector` — one instance for the whole run. Queue-age
  (:meth:`~percept.harness.metrics.MetricsCollector.record_admitted` /
  ``record_admission_resolved``) and RSS (``record_rss``) are both fed via their explicit ``at=``
  override — exactly the usage their docstrings anticipate ("a probe replaying a fixed timeline
  never needs the collector's own clock to be ticked in lockstep") — using ANALYTIC timestamps
  (admit time + the judge call's own sampled latency), never a real sleep.
* :class:`percept.clock.FakeClock` — the sole time source. ``advance()`` fast-forwards simulated
  hours; nothing in this module calls ``time.sleep``/``asyncio.sleep`` with a real duration
  (``FakeJudge`` is always built with ``simulate_sleep=False`` here).

**Queue-age model.** Real concurrent-queue depth isn't observable through ``LoadDriver`` (its
``F`` fan-out calls are awaited via ``asyncio.gather`` with ``simulate_sleep=False``, so nothing
elapses on the clock during a tick). Instead, every tick's newly-issued judge calls are treated as
admitted simultaneously at that tick's elapsed time, then resolved in ascending order of their own
sampled latency (``admit_t + latency_s``, arithmetic, never a sleep) — a deterministic, analytic
stand-in for "many calls admitted together, staggered completions" that is reproducible bit-for-bit
given the same seed and config, and whose oldest-pending-age reading rises when the configured
latency distribution rises (the leak/drift signal this soak is designed to detect).

**Memory model.** This harness never touches a real process's RSS (there is nothing "real" running
under a soak driven purely by ``FakeClock``). :attr:`SoakConfig.rss_fn` is an INJECTED
hour-indexed sampler (never estimated here) — mirrors :class:`~percept.harness.metrics.CostSample`'s
"injected, never estimated" convention in :mod:`percept.harness.metrics`.

Not wired into ``percept/registry.py``. Downstream: the full gated, bars-locked stability soak is
follow-up work — this module only lands the deterministic driver + its own self-test.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable, Sequence

from ..backends.fake_judge import ErrorInjectionConfig, FakeJudge, LatencyProfile
from ..clock import FakeClock
from ..runtime.worker import RuntimeHealthEvent
from .load_driver import LoadDriver, LoadProfile, RampSchedule
from .metrics import MetricsCollector, PercentileSummary
from .stream_gen import Timeline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _flat_rss(_hour: int) -> int:
    return 0


def _flat_latency(_hour: int) -> LatencyProfile:
    return LatencyProfile(kind="fixed", value_s=0.0)


def _no_errors(_hour: int) -> ErrorInjectionConfig:
    return ErrorInjectionConfig(rate=0.0)


@dataclass(frozen=True)
class SoakConfig:
    """A soak run's shape: ``hours`` simulated hours of steady ``profile`` load, each hour built
    from a fresh, hour-indexed ``latency_fn``/``errors_fn`` (so latency drift and a provider-
    failure window can both be modeled), plus an injected ``rss_fn`` memory sampler.

    All ``*_fn`` callables receive the 1-indexed hour number (``1..hours``) and must be pure/
    deterministic (no wall-clock, no unseeded randomness) for the run to be reproducible.
    """

    hours: int
    profile: LoadProfile
    timeline: Timeline
    hour_s: float = 3600.0
    tick_s: float = 300.0
    seed: int = 0
    latency_fn: Callable[[int], LatencyProfile] = field(default_factory=lambda: _flat_latency)
    errors_fn: Callable[[int], ErrorInjectionConfig] = field(default_factory=lambda: _no_errors)
    rss_fn: Callable[[int], int] = field(default_factory=lambda: _flat_rss)

    def __post_init__(self) -> None:
        if self.hours < 1:
            raise ValueError(f"SoakConfig.hours must be >= 1, got {self.hours!r}")
        if self.hour_s <= 0:
            raise ValueError(f"SoakConfig.hour_s must be > 0, got {self.hour_s!r}")
        if self.tick_s <= 0:
            raise ValueError(f"SoakConfig.tick_s must be > 0, got {self.tick_s!r}")


# ---------------------------------------------------------------------------
# Per-hour snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HourSnapshot:
    """One hourly :class:`~percept.harness.metrics.MetricsCollector` reading — hour ``N``'s slice
    of the drift signals: queue-age (``max_oldest_pending_age_s``), memory (``rss_bytes``),
    latency (``latency_p50_s`` / ``p95`` / ``p99``) — plus the load-driver telemetry for that hour."""

    hour: int  # 1-indexed
    t_s: float  # absolute simulated seconds (soak-relative) at the end of this hour
    episodes_admitted: int
    judge_calls_issued: int
    judge_calls_ok: int
    judge_calls_failed: int
    max_queue_depth: int
    max_oldest_pending_age_s: float
    rss_bytes: int
    latency_p50_s: float
    latency_p95_s: float
    latency_p99_s: float


def _latency_percentiles(latencies: Sequence[float]) -> tuple[float, float, float]:
    if not latencies:
        return (0.0, 0.0, 0.0)
    summary = PercentileSummary.of(latencies)
    return (summary.p50, summary.p95, summary.p99)


# ---------------------------------------------------------------------------
# Hour-N vs hour-1 trend
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SoakTrend:
    """Hour-``N``-vs-hour-1 comparison — a simple two-point slope/delta between the
    first and last supplied snapshots, not a full regression: exactly the comparison the name
    implies ("hour-N-vs-hour-1"), no more machinery than that requires."""

    hour_first: int
    hour_last: int
    queue_age_slope_s_per_hour: float
    memory_slope_bytes_per_hour: float
    latency_p95_drift_s: float
    latency_p95_first_s: float
    latency_p95_last_s: float


def compute_trend(snapshots: Sequence[HourSnapshot]) -> SoakTrend:
    """Hour-``N`` (last snapshot) vs hour-1 (first snapshot) slope for queue-age + memory, and the
    raw p95 latency delta. Requires at least 2 snapshots (a slope needs two points)."""
    if len(snapshots) < 2:
        raise ValueError("compute_trend() requires at least 2 hourly snapshots")
    first, last = snapshots[0], snapshots[-1]
    dh = last.hour - first.hour
    if dh <= 0:
        raise ValueError(
            f"compute_trend() requires snapshots[-1].hour > snapshots[0].hour, "
            f"got {first.hour!r} -> {last.hour!r}"
        )
    return SoakTrend(
        hour_first=first.hour,
        hour_last=last.hour,
        queue_age_slope_s_per_hour=(last.max_oldest_pending_age_s - first.max_oldest_pending_age_s) / dh,
        memory_slope_bytes_per_hour=(last.rss_bytes - first.rss_bytes) / dh,
        latency_p95_drift_s=last.latency_p95_s - first.latency_p95_s,
        latency_p95_first_s=first.latency_p95_s,
        latency_p95_last_s=last.latency_p95_s,
    )


def assert_clean_recovery(snapshots: Sequence[HourSnapshot], *, failure_hours: frozenset[int]) -> bool:
    """True iff injected provider failures are confined EXACTLY to ``failure_hours``: every hour
    in the window shows at least one failed judge call, and every hour outside it shows zero —
    i.e. the run recovers cleanly the very next hour, with no residual failures bleeding forward."""
    for snap in snapshots:
        in_window = snap.hour in failure_hours
        if in_window and snap.judge_calls_failed == 0:
            return False
        if not in_window and snap.judge_calls_failed != 0:
            return False
    return True


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


class SoakRunner:
    """Drives :class:`SoakConfig` over an injected :class:`~percept.clock.FakeClock` — hours
    advance only via ``clock.advance()``, never a real wall-clock wait. See the module docstring
    for the queue-age/memory/latency models.
    """

    def __init__(
        self, config: SoakConfig, *, clock: FakeClock, on_health: Callable[[RuntimeHealthEvent], None] | None = None
    ) -> None:
        self.config = config
        self._clock = clock
        self.collector = MetricsCollector(clock)
        self._soak_start_t = clock()
        # Surface judge-call degradation through the SAME RuntimeHealthEvent shape
        # runtime/worker.py uses — this harness bypasses the live capture loop (LoadDriver +
        # FakeJudge, not run()/_JudgeScheduler), so it has no scheduler of its own to be loud from;
        # what it CAN do is reuse the one already-loud channel rather than a parallel one. Best-
        # effort and purely additive: default None keeps every existing caller byte-for-byte.
        self._on_health = on_health
        self._prev_hour_had_failure = False

    def _soak_t(self) -> float:
        return self._clock() - self._soak_start_t

    def _emit_health(self, kind: str, *, ts: float, data: dict) -> None:
        if self._on_health is None:
            return
        self._on_health(RuntimeHealthEvent(kind, ts=ts, data=data))

    async def run(self) -> list[HourSnapshot]:
        """Run all :attr:`SoakConfig.hours` simulated hours; returns one :class:`HourSnapshot`
        per hour, in order. Safe to call at most once per :class:`SoakRunner` instance (the
        collector's queue/RSS series accumulate across the whole run)."""
        snapshots: list[HourSnapshot] = []
        cfg = self.config

        for hour in range(1, cfg.hours + 1):
            hour_start_t = self._soak_t()
            judge = FakeJudge(
                seed=cfg.seed + hour,
                latency=cfg.latency_fn(hour),
                errors=cfg.errors_fn(hour),
                simulate_sleep=False,
            )
            schedule = RampSchedule(ramp_s=0.0, steady_s=cfg.hour_s)
            driver = LoadDriver(cfg.profile, schedule, cfg.timeline, clock=self._clock, judge=judge)

            prev_issued = 0
            tick_no = 0
            while True:
                await driver.tick()
                new_calls = judge.calls[prev_issued:]
                if new_calls:
                    self._admit_and_resolve_tick(hour, tick_no, new_calls)
                prev_issued = len(judge.calls)
                tick_no += 1
                if driver.done():
                    break
                self._clock.advance(cfg.tick_s * 1000.0)

            hour_end_t = self._soak_t()
            self.collector.record_rss(cfg.rss_fn(hour), at=hour_end_t)

            queue_samples = [
                s for s in self.collector.queue_depth_series() if hour_start_t <= s.t <= hour_end_t
            ]
            max_qd = max((s.depth for s in queue_samples), default=0)
            max_age = max((s.oldest_pending_age_s for s in queue_samples), default=0.0)
            p50, p95, p99 = _latency_percentiles(judge.latencies)

            snapshots.append(
                HourSnapshot(
                    hour=hour,
                    t_s=hour_end_t,
                    episodes_admitted=driver.stats.episodes_admitted,
                    judge_calls_issued=driver.stats.judge_calls_issued,
                    judge_calls_ok=driver.stats.judge_calls_ok,
                    judge_calls_failed=driver.stats.judge_calls_failed,
                    max_queue_depth=max_qd,
                    max_oldest_pending_age_s=max_age,
                    rss_bytes=cfg.rss_fn(hour),
                    latency_p50_s=p50,
                    latency_p95_s=p95,
                    latency_p99_s=p99,
                )
            )

            # Surface this hour's degradation loud, hourly-granular (this harness's own
            # resolution — see __init__). Every call failed this hour -> the persistent-failure
            # escalation (mirrors agent.py's judge_persistent_failure: a dead-but-running judge must
            # be loud, not just a counter). A clean hour immediately after a failing one -> recovery,
            # reported exactly once, never latched red past the transient.
            hour_had_failure = driver.stats.judge_calls_failed > 0
            if driver.stats.judge_calls_issued > 0 and driver.stats.judge_calls_failed == driver.stats.judge_calls_issued:
                self._emit_health(
                    "judge_persistent_failure", ts=hour_end_t,
                    data={"hour": hour, "judge_calls_failed": driver.stats.judge_calls_failed},
                )
            elif self._prev_hour_had_failure and not hour_had_failure:
                self._emit_health("judge_recovered", ts=hour_end_t, data={"hour": hour})
            self._prev_hour_had_failure = hour_had_failure

        return snapshots

    def _admit_and_resolve_tick(self, hour: int, tick_no: int, new_calls: Sequence) -> None:
        """Admit every judge call newly issued THIS tick simultaneously (at the tick's current
        simulated time), then resolve them in ascending order of their own sampled latency —
        analytic ``admit_t + latency_s`` completion times, never a real sleep. See the module
        docstring's "Queue-age model" section."""
        admit_t = self._soak_t()
        keys = [(hour, tick_no, i) for i in range(len(new_calls))]
        for key in keys:
            self.collector.record_admitted(key, at=admit_t)
        order = sorted(range(len(new_calls)), key=lambda i: new_calls[i].latency_s)
        for i in order:
            resolve_t = admit_t + new_calls[i].latency_s
            self.collector.record_admission_resolved(keys[i], at=resolve_t)
            if new_calls[i].injected is not None:  # surface each injected failure, loud
                self._emit_health(
                    "judge_degraded", ts=resolve_t,
                    data={"hour": hour, "tick": tick_no, "mode": new_calls[i].injected},
                )


__all__ = [
    "SoakConfig",
    "HourSnapshot",
    "SoakTrend",
    "compute_trend",
    "assert_clean_recovery",
    "SoakRunner",
]
