"""PROBE-HARNESS SCAFFOLDING — ``load_driver`` (Viability §2/§6, board card A11).

**Wiring-only, not a production connector.** The probe harness's load driver: it ramps and then
holds the **full** ``N/M/E/F`` load profile Viability §2 asks for —

* ``N`` — concurrent streams
* ``M`` — watches per stream
* ``E`` — admitted episodes per stream-minute
* ``F`` — fan-out judge calls issued per admitted episode

not just the ``N x M`` concurrency shape a naive driver would stop at. It reuses, rather than
reimplements, the two already-landed harness pieces:

* :func:`percept.harness.stream_gen.fan_out` builds the ``N`` independent, clock-driven frame
  streams (board card A10).
* :class:`percept.backends.fake_judge.FakeJudge` is the queueing/backpressure/latency double every
  admitted episode's ``F`` fan-out calls are routed into (board card A12).

Standing rule 2 (see :mod:`percept.backends.fake`) still applies: this proves wiring and load
*shape*, never model intelligence — verdict content is FakeJudge's concern, not this module's.

**Determinism.** Time enters through exactly one injected :data:`~percept.clock.Clock` (v3 §9's
run-time contract, reused here for the probe harness itself, same as :mod:`stream_gen`). The core
loop (:meth:`LoadDriver.tick`) never sleeps and never reads a wall clock directly; episode
admission uses a closed-form integral of the ramp/steady target rate over ``[last_tick, now]``, so
the admitted-episode count for a given clock trajectory is exact and reproducible regardless of how
finely ``tick()`` is called — one big jump to the end of the run yields the identical count as many
small ticks.

**Ramp / steady semantics.** All ``N`` streams (and their ``M`` watches each) are live from t=0 —
there is no staggered stream activation, so the driver genuinely exercises the full ``N`` x ``M``
concurrency shape throughout. What *ramps* is the per-watch episode-admission RATE: it grows
linearly from 0 at ``t=0`` to the full target rate at ``t=ramp_s``, then holds flat for
``steady_s``. Each of a stream's ``M`` watches is phase-offset (``watch_index / M`` head-start
credit) so admissions spread across the cycle instead of bursting in lockstep. Every admitted
episode immediately issues exactly ``F`` concurrent :meth:`FakeJudge.judge` calls (awaited via
``asyncio.gather``), so the ``F`` fan-out is exact and multiplicative, never approximated.

Not wired into ``percept/registry.py``. Downstream: Phase-A probes ``P-BOUND-1`` (queueing/
backpressure under load) and ``P-LAT-LOAD-1`` (latency distribution under the full N/M/E/F profile,
board card A28) drive sessions through this module instead of hand-rolling load generation.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from ..backends.fake_judge import FakeJudge
from ..clock import Clock
from .stream_gen import Timeline, fan_out


# ---------------------------------------------------------------------------
# Configuration — the full N/M/E/F load profile + ramp/steady schedule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadProfile:
    """The full Viability §2 load shape: ``N`` concurrent streams x ``M`` watches per stream,
    each watch admitting episodes at ``e_per_stream_min`` (aggregate per stream, i.e. divided
    evenly across that stream's ``M`` watches) with ``f`` fan-out judge calls per admission."""

    n: int  # concurrent streams
    m: int  # watches per stream
    e_per_stream_min: float  # admitted episodes per stream-minute (aggregate over the stream's M watches)
    f: int  # fan-out judge calls per admitted episode

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError(f"LoadProfile.n must be >= 1, got {self.n!r}")
        if self.m < 1:
            raise ValueError(f"LoadProfile.m must be >= 1, got {self.m!r}")
        if self.e_per_stream_min <= 0:
            raise ValueError(f"LoadProfile.e_per_stream_min must be > 0, got {self.e_per_stream_min!r}")
        if self.f < 1:
            raise ValueError(f"LoadProfile.f must be >= 1, got {self.f!r}")

    @property
    def full_rate_per_watch(self) -> float:
        """Target steady-state admission rate for ONE watch, in episodes/second."""
        return self.e_per_stream_min / 60.0 / self.m


@dataclass(frozen=True)
class RampSchedule:
    """Ramp phase (``[0, ramp_s)``, rate grows linearly 0 -> full) followed by a steady phase
    (``[ramp_s, ramp_s + steady_s]``, rate held at full)."""

    ramp_s: float = 0.0
    steady_s: float = 0.0

    def __post_init__(self) -> None:
        if self.ramp_s < 0:
            raise ValueError(f"RampSchedule.ramp_s must be >= 0, got {self.ramp_s!r}")
        if self.steady_s <= 0:
            raise ValueError(f"RampSchedule.steady_s must be > 0, got {self.steady_s!r}")

    @property
    def total_s(self) -> float:
        return self.ramp_s + self.steady_s

    def phase_at(self, t: float) -> str:
        """``"ramp"`` for ``t <= ramp_s``, else ``"steady"`` (boundary tick counts as ramp)."""
        return "ramp" if t <= self.ramp_s else "steady"


# ---------------------------------------------------------------------------
# Observed telemetry — what a probe asserts the intended N/M/E/F actually produced
# ---------------------------------------------------------------------------


@dataclass
class LoadDriverStats:
    """Telemetry a probe reads back to confirm the configured ``N/M/E/F`` was actually produced."""

    streams_active: int = 0
    frames_emitted: int = 0
    episodes_admitted: int = 0
    episodes_by_phase: dict[str, int] = field(default_factory=lambda: {"ramp": 0, "steady": 0})
    judge_calls_issued: int = 0
    judge_calls_ok: int = 0
    judge_calls_failed: int = 0
    elapsed_s: float = 0.0

    @property
    def observed_f(self) -> float:
        """Judge calls issued per admitted episode — should equal the configured ``F`` exactly
        (every admission issues exactly ``F`` calls, no approximation)."""
        if self.episodes_admitted == 0:
            return 0.0
        return self.judge_calls_issued / self.episodes_admitted

    def observed_e_per_stream_min(self, n: int) -> float:
        """Admitted episodes per stream-minute, averaged over the whole run so far (ramp included
        — so this is expected to read BELOW the configured target ``E`` unless ``ramp_s == 0``)."""
        if n <= 0 or self.elapsed_s <= 0:
            return 0.0
        return self.episodes_admitted / n / (self.elapsed_s / 60.0)


# ---------------------------------------------------------------------------
# LoadDriver — drives N fan_out streams + the M/E/F episode/judge-call load through a clock
# ---------------------------------------------------------------------------


class LoadDriver:
    """Ramps and then holds the full ``N/M/E/F`` load profile, driven entirely by an injected
    :data:`~percept.clock.Clock` — no wall-clock sleeps anywhere in :meth:`tick`.

    Parameters
    ----------
    profile:
        The ``N/M/E/F`` load shape.
    schedule:
        Ramp + steady durations.
    timeline:
        The frame :data:`~percept.harness.stream_gen.Timeline` fanned out to all ``N`` streams via
        :func:`~percept.harness.stream_gen.fan_out` (one shared clock — see that function's
        docstring for the shared- vs per-stream-clock trade-off; ``load_driver`` always uses the
        shared-clock form since all N streams are meant to be paced by the same simulated
        wall-clock).
    clock:
        The injected :data:`~percept.clock.Clock`. In tests this is a
        :class:`~percept.clock.FakeClock`; :meth:`run_to_completion` drives it via ``.advance()``.
    judge:
        The :class:`~percept.backends.fake_judge.FakeJudge` every admitted episode's ``F`` fan-out
        calls are routed into.
    """

    def __init__(
        self,
        profile: LoadProfile,
        schedule: RampSchedule,
        timeline: Timeline,
        *,
        clock: Clock,
        judge: FakeJudge,
    ) -> None:
        self.profile = profile
        self.schedule = schedule
        self._clock = clock
        self._judge = judge
        self._start_t = clock()
        self._last_t = 0.0

        # A10 dependency: the N concurrent, clock-driven frame streams. Built once, up front — all
        # N streams (and thus all N*M watches) are live for the entire run; what ramps is the
        # admission RATE (see module docstring), not stream/watch existence.
        self._streams = fan_out(timeline, profile.n, clock=clock)

        # Per (stream, watch) fractional admission credit. Watch j starts with a j/M head-start so
        # a stream's M watches admit staggered across the cycle instead of bursting in lockstep.
        self._credit: list[list[float]] = [
            [w / profile.m for w in range(profile.m)] for _ in range(profile.n)
        ]

        self.stats = LoadDriverStats(streams_active=profile.n)

    # -- rate model ------------------------------------------------------------------------------

    def _rate_integral(self, t_from: float, t_to: float) -> float:
        """Exact closed-form episodes/watch admitted over ``[t_from, t_to]`` under the linear
        ramp-then-hold rate curve, clamped to the schedule's total duration."""
        total_s = self.schedule.total_s
        t_from = min(max(t_from, 0.0), total_s)
        t_to = min(max(t_to, 0.0), total_s)
        if t_to <= t_from:
            return 0.0

        full = self.profile.full_rate_per_watch
        ramp_s = self.schedule.ramp_s
        if ramp_s <= 0:
            return full * (t_to - t_from)

        result = 0.0
        if t_from < ramp_s:
            seg_to = min(t_to, ramp_s)
            # integral of (full * t / ramp_s) dt from t_from to seg_to
            result += full / ramp_s * (seg_to**2 - t_from**2) / 2.0
            t_from = seg_to
        if t_to > t_from:
            result += full * (t_to - t_from)
        return result

    # -- core loop ---------------------------------------------------------------------------------

    async def tick(self) -> None:
        """One non-blocking step: drain due frames from all N streams, credit the elapsed-time
        admission rate to every (stream, watch), admit whatever crosses 1.0 credit, and fire
        exactly ``F`` :meth:`FakeJudge.judge` calls per admission (awaited via ``asyncio.gather``).

        Reads ``clock()`` exactly once. Safe to call with an arbitrarily large or small elapsed gap
        since the last call — the admitted-episode count over that gap is computed via the exact
        closed-form rate integral, not by simulating sub-steps.
        """
        now = min(self._clock() - self._start_t, self.schedule.total_s)

        for stream in self._streams:
            self.stats.frames_emitted += len(stream.poll())

        # Split the elapsed interval at the ramp/steady boundary so phase attribution of each
        # admission is exact regardless of how coarsely tick() is called — a single tick spanning
        # both phases still attributes each admission to the phase its threshold-crossing actually
        # falls in, matching what many small ticks across the same interval would produce (credit
        # is monotonically non-decreasing, so processing the ramp-time delta fully before the
        # steady-time delta preserves the true temporal order of crossings).
        ramp_s = self.schedule.ramp_s
        delta_ramp = self._rate_integral(self._last_t, min(now, ramp_s))
        delta_steady = self._rate_integral(max(self._last_t, ramp_s), now)

        judge_coros = []
        for si in range(self.profile.n):
            for wi in range(self.profile.m):
                credit = self._credit[si][wi]
                for phase, delta in (("ramp", delta_ramp), ("steady", delta_steady)):
                    if delta <= 0.0:
                        continue
                    credit += delta
                    while credit >= 1.0 - 1e-9:
                        credit -= 1.0
                        self.stats.episodes_admitted += 1
                        self.stats.episodes_by_phase[phase] += 1
                        for _ in range(self.profile.f):
                            judge_coros.append(self._issue_judge_call())
                self._credit[si][wi] = credit

        self._last_t = now
        self.stats.elapsed_s = now
        if judge_coros:
            await asyncio.gather(*judge_coros)

    async def _issue_judge_call(self) -> None:
        self.stats.judge_calls_issued += 1
        try:
            await self._judge.judge("load-probe: synthetic admitted episode", "load-driver-frame")
            self.stats.judge_calls_ok += 1
        except Exception:
            self.stats.judge_calls_failed += 1

    def done(self) -> bool:
        """True once the clock has reached (or passed) the end of the steady phase."""
        return (self._clock() - self._start_t) >= self.schedule.total_s

    async def run_to_completion(self, *, step_s: float) -> LoadDriverStats:
        """Convenience driver loop for a :class:`~percept.clock.FakeClock`: repeatedly ``tick()``
        then ``clock.advance(step_s * 1000)`` until :meth:`done`. Requires a clock exposing
        ``.advance()`` (the ``FakeClock`` contract); drive :meth:`tick` manually against a
        real/other clock instead.
        """
        if step_s <= 0:
            raise ValueError(f"run_to_completion step_s must be > 0, got {step_s!r}")
        if not hasattr(self._clock, "advance"):
            raise TypeError(
                "run_to_completion requires a clock with .advance() (e.g. FakeClock); "
                "drive tick() manually against other clocks"
            )
        while True:
            await self.tick()
            if self.done():
                break
            self._clock.advance(step_s * 1000.0)
        return self.stats


__all__ = [
    "LoadProfile",
    "RampSchedule",
    "LoadDriverStats",
    "LoadDriver",
]
