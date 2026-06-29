"""The canonical measurement layer — Build 1 of the rate-invariant individuation blueprint.

Everything that wants to *individuate events* over time (count reps, list steps, detect onsets, dedup
VLM observations) first needs a representation of the underlying signal that does **not** depend on how
densely the stream was sampled. That is this module.

The one rule: **everything is a timestamped ``(t_seconds, value)`` pair — never a frame index.** A frame
index secretly encodes the sampling rate (sample 30 is 1 s at 30 fps but 5 s at 6 fps), so any threshold,
window, or period expressed in *samples* silently changes meaning when fps changes. That is precisely the
"denser-is-worse" pathology the investigation traced to a frame-rate-dependent retrieval layer. Expressing
every quantity in **seconds + amplitude** makes the layer rate-invariant *by construction*.

This is the foundation the rate-invariant counter (Build 2) and channel selector (Build 3) sit on:

  * :func:`to_time_grid` — resample/aggregate an irregularly-sampled stream onto a uniform time grid at a
    chosen canonical rate, low-passing first so downsampling does not alias. Channel scoring runs on this
    canonical representation, so "fps is a measurement detail, not a different electorate."
  * :func:`periodicity_score` / :func:`dominant_period` — how periodic the signal is (∈ [0, 1]) and the
    period of its dominant cycle **in seconds**, via autocorrelation on the canonical grid. The measured
    period is the self-tuned ``τ`` (≈ refractory ≈ rep-period) — the *only* goal-derived scalar, and it is
    measured from the signal, not read from a table.
  * :func:`robust_amplitude` — the signal's amplitude over a window **in seconds** (a percentile span, so a
    single spike cannot inflate it), one estimate per period rather than per frame.
  * :func:`subsample_stability` — how stable any scorer's answer is when the stream is temporally
    decimated. A metric that survives throwing away half the samples is one that does not depend on fps.

The module is **pure stdlib** (no numpy/scipy) on purpose: ``percept`` core installs with no heavy deps,
and the rhythm/individuation logic must stay cheap, causal-friendly, and inspectable (see ``rhythm.py``).
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Callable, Sequence

# A sensible default canonical rate for resampling pose/rep-scale signals (their cycles are well under a
# few Hz, so 30 Hz is comfortably above the Nyquist floor). Selection should pass an explicit canonical hz
# so every channel is scored on the SAME grid regardless of how each was sampled; this is just the default.
CANON_HZ = 30.0


@dataclass(frozen=True)
class MeasurementSeries:
    """One measured scalar channel as timestamped ``(t_seconds, value)`` samples — never frame indices.

    ``samples`` is time-ordered ``((t, v), ...)`` in seconds and the channel's native ``units`` (a pose
    channel is normalized by body scale, an audio envelope is dB, a cadence is steps/s — the unit is just a
    label carried for explainability; the math is unit-agnostic). ``source_hz`` records the rate the stream
    was *captured* at, when known, but no method relies on it — the helpers read time from the timestamps,
    which is the whole point. Construct via :meth:`from_pairs` to get time-sorting for free.
    """

    channel: str
    samples: tuple[tuple[float, float], ...]
    units: str = ""
    source_hz: float | None = None

    @classmethod
    def from_pairs(
        cls,
        channel: str,
        pairs: Sequence[tuple[float, float]],
        units: str = "",
        source_hz: float | None = None,
    ) -> "MeasurementSeries":
        """Build a series from any iterable of ``(t, value)`` pairs, sorted by time (stable on ties)."""
        ordered = tuple(sorted(((float(t), float(v)) for t, v in pairs), key=lambda p: p[0]))
        return cls(channel=channel, samples=ordered, units=units, source_hz=source_hz)

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def times(self) -> tuple[float, ...]:
        return tuple(t for t, _ in self.samples)

    @property
    def values(self) -> tuple[float, ...]:
        return tuple(v for _, v in self.samples)

    @property
    def span(self) -> tuple[float, float] | None:
        """``(t_first, t_last)`` in seconds, or ``None`` for an empty series."""
        if not self.samples:
            return None
        return self.samples[0][0], self.samples[-1][0]

    @property
    def duration(self) -> float:
        """Wall-clock seconds spanned (``0.0`` for <2 samples)."""
        if len(self.samples) < 2:
            return 0.0
        return self.samples[-1][0] - self.samples[0][0]

    @property
    def effective_hz(self) -> float:
        """The *actual* mean sample rate over the observed span — ``(n-1)/duration`` — which is what the
        canonical layer cares about, not the nominal ``source_hz`` (the fovea drops frames, so the two
        differ). ``0.0`` when there is no span to measure."""
        d = self.duration
        if d > 0:
            return (len(self.samples) - 1) / d
        return self.source_hz or 0.0

    # --- ergonomic method forms; the module functions are the canonical API ----------------------------
    def to_time_grid(self, hz: float, lowpass: bool = True) -> "MeasurementSeries":
        return to_time_grid(self, hz, lowpass=lowpass)

    def periodicity_score(self, band_s: tuple[float, float] | None = None, hz: float | None = None) -> float:
        return periodicity_score(self, band_s=band_s, hz=hz)

    def dominant_period(self, band_s: tuple[float, float] | None = None, hz: float | None = None) -> float | None:
        return dominant_period(self, band_s=band_s, hz=hz)

    def robust_amplitude(self, window_s: float, pct: float = 0.9) -> float:
        return robust_amplitude(self, window_s, pct=pct)

    def subsample_stability(self, scorer: Callable[["MeasurementSeries"], object], factors: Sequence[int] = (2, 3)) -> float:
        return subsample_stability(self, scorer, factors=factors)


# --- internal numeric helpers (pure stdlib) -----------------------------------------------------------

def _percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (``q`` ∈ [0, 1]) — robust amplitude reads percentile spans so one
    outlier spike cannot dominate the estimate. Empty -> 0.0."""
    xs = sorted(values)
    n = len(xs)
    if n == 0:
        return 0.0
    if n == 1:
        return xs[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return xs[lo] + (xs[hi] - xs[lo]) * frac


def _interp(times: Sequence[float], values: Sequence[float], tq: float) -> float:
    """Linear interpolation of ``values`` at query time ``tq`` (clamped to the endpoints)."""
    n = len(times)
    if tq <= times[0]:
        return values[0]
    if tq >= times[-1]:
        return values[-1]
    i = bisect.bisect_right(times, tq)
    t0, t1 = times[i - 1], times[i]
    v0, v1 = values[i - 1], values[i]
    if t1 == t0:
        return v0
    return v0 + (v1 - v0) * (tq - t0) / (t1 - t0)


def _autocorr_peak(
    values: Sequence[float], hz: float, band_s: tuple[float, float] | None
) -> tuple[float, float | None, int | None]:
    """Normalized autocorrelation peak on a UNIFORM grid sampled at ``hz``.

    Returns ``(score, period_s, lag_samples)``: ``score`` ∈ [0, 1] is the height of the dominant
    autocorrelation peak (1.0 = perfectly periodic), ``period_s`` is that peak's lag converted to
    **seconds**, and ``lag_samples`` is the same lag in *samples* (kept only so a caller can show the
    frame-index — i.e. NON-invariant — view for contrast). Mean-removed before correlating so a DC offset
    or slow baseline does not swamp the cycle."""
    n = len(values)
    if n < 4 or hz <= 0:
        return 0.0, None, None
    mean = sum(values) / n
    x = [v - mean for v in values]
    denom = sum(xi * xi for xi in x)
    if denom <= 0:
        return 0.0, None, None
    if band_s is not None:
        kmin = max(1, int(round(band_s[0] * hz)))
        kmax = min(n - 1, int(round(band_s[1] * hz)))
    else:
        kmin, kmax = 1, n // 2
    if kmax < kmin:
        return 0.0, None, None
    ac = [sum(x[i] * x[i + k] for i in range(n - k)) / denom for k in range(kmin, kmax + 1)]
    # Prefer an interior local maximum (the true cycle) over the monotone shoulder near lag 0; fall back
    # to the plain argmax if the band shows no local max.
    best_idx, best_val = None, -2.0
    for j in range(1, len(ac) - 1):
        if ac[j] >= ac[j - 1] and ac[j] >= ac[j + 1] and ac[j] > best_val:
            best_idx, best_val = j, ac[j]
    if best_idx is None:
        best_val = max(ac)
        best_idx = ac.index(best_val)
    lag = kmin + best_idx
    score = max(0.0, min(1.0, best_val))
    return score, lag / hz, lag


# --- canonical layer: the four blueprint helpers ------------------------------------------------------

def to_time_grid(series: MeasurementSeries, hz: float, lowpass: bool = True) -> MeasurementSeries:
    """Resample an irregularly-sampled stream onto a UNIFORM grid at ``hz`` Hz, spanning its time range.

    This is the canonicalization step: scoring and selection happen here, so two captures of the same
    motion at different fps become the *same* representation (above the Nyquist floor) rather than two
    different "electorates."

    When ``lowpass`` (the default) and the grid is coarser than the source, each grid point is the **mean**
    of the source samples within ±half a grid step — a box anti-alias filter, so downsampling cannot fold
    high-frequency jitter down onto the cycle. Where a window is empty (upsampling, or a sparse stretch)
    the grid point is linearly interpolated. With ``lowpass=False`` every grid point is a pure linear
    interpolation. Returns a new series with ``source_hz=hz``; a <2-sample input is returned unchanged."""
    if len(series.samples) < 2 or hz <= 0:
        return series
    times, values = series.times, series.values
    t0, t1 = times[0], times[-1]
    step = 1.0 / hz
    half = 0.5 * step
    n_steps = int(math.floor((t1 - t0) / step + 1e-9))
    grid: list[tuple[float, float]] = []
    for k in range(n_steps + 1):
        tg = t0 + k * step
        if lowpass:
            lo = bisect.bisect_left(times, tg - half)
            hi = bisect.bisect_right(times, tg + half)
            if hi > lo:
                window = values[lo:hi]
                grid.append((tg, sum(window) / len(window)))
                continue
        grid.append((tg, _interp(times, values, tg)))
    return MeasurementSeries(channel=series.channel, samples=tuple(grid), units=series.units, source_hz=hz)


def _periodicity(
    series: MeasurementSeries, band_s: tuple[float, float] | None, hz: float | None
) -> tuple[float, float | None, int | None]:
    """Shared core for :func:`periodicity_score` / :func:`dominant_period`: canonicalize to a uniform grid,
    then read the autocorrelation peak. ``hz=None`` uses the series' own effective rate (already invariant
    because the period comes back in seconds); pass an explicit ``hz`` to force a shared canonical grid."""
    if len(series.samples) < 4:
        return 0.0, None, None
    grid_hz = hz if hz is not None else series.effective_hz
    if grid_hz <= 0:
        return 0.0, None, None
    grid = to_time_grid(series, grid_hz, lowpass=True)
    return _autocorr_peak(grid.values, grid_hz, band_s)


def periodicity_score(
    series: MeasurementSeries, band_s: tuple[float, float] | None = None, hz: float | None = None
) -> float:
    """How periodic the channel is, ∈ [0, 1] — the normalized height of its dominant autocorrelation peak
    (1.0 = a clean cycle, ~0 = noise). ``band_s=(min_period_s, max_period_s)`` restricts the search to a
    plausible period band (e.g. a goal-authored cadence prior); ``hz`` forces a canonical resample rate
    (recommended for cross-channel selection so every channel is scored on the same grid)."""
    return _periodicity(series, band_s, hz)[0]


def dominant_period(
    series: MeasurementSeries, band_s: tuple[float, float] | None = None, hz: float | None = None
) -> float | None:
    """The period of the dominant cycle **in seconds** (``None`` if no cycle is found). This is the
    self-measured ``τ`` — the rep period that scales the counter's seconds-refractory and the selector's
    period band — read from the signal rather than from a goal-type table."""
    return _periodicity(series, band_s, hz)[1]


def robust_amplitude(series: MeasurementSeries, window_s: float, pct: float = 0.9) -> float:
    """Robust amplitude in the channel's native units, measured **per ``window_s`` seconds** (≈ one period),
    not per frame.

    Within each non-overlapping ``window_s`` window the amplitude is a percentile span
    (``pct``=0.9 -> p95−p05), which ignores a lone spike; the channel's amplitude is the **median** of the
    per-window spans, so a couple of anomalous windows cannot dominate. A short series (under one window, or
    no window with enough points) falls back to the global percentile span. This is rate-invariant: it reads
    percentiles over time windows, and more frames only sharpens the same percentiles."""
    vals = series.values
    if not vals:
        return 0.0
    lo_q, hi_q = max(0.0, 0.5 - pct / 2), min(1.0, 0.5 + pct / 2)

    def span(xs: Sequence[float]) -> float:
        return _percentile(xs, hi_q) - _percentile(xs, lo_q)

    if window_s <= 0 or series.duration <= window_s:
        return span(vals)
    t0 = series.samples[0][0]
    buckets: dict[int, list[float]] = {}
    for t, v in series.samples:
        buckets.setdefault(int((t - t0) / window_s), []).append(v)
    spans = [span(xs) for xs in buckets.values() if len(xs) >= 3]
    if not spans:
        return span(vals)
    spans.sort()
    m = len(spans) // 2
    return spans[m] if len(spans) % 2 else 0.5 * (spans[m - 1] + spans[m])


def subsample_stability(
    series: MeasurementSeries,
    scorer: Callable[[MeasurementSeries], object],
    factors: Sequence[int] = (2, 3),
) -> float:
    """How stable ``scorer``'s answer is under temporal decimation — the "does this survive throwing away
    samples?" test, ∈ [0, 1].

    Runs ``scorer`` on the full series and on every phase of each decimation in ``factors`` (factor 2 ->
    even/odd samples, factor 3 -> three phases, …), then measures agreement: for **numeric** answers (a
    period, a count) stability is ``1/(1+CV)`` of the answers (1.0 = identical); for **categorical** answers
    (a selected channel name) it is the modal fraction. A metric that does not depend on fps scores high
    here; one that flips when you decimate (the channel selector's old failure) scores low. Returns ``1.0``
    when there is too little evidence to decide (don't penalize the untestable)."""
    results: list[object] = []
    base = scorer(series)
    if base is not None:
        results.append(base)
    for k in factors:
        if k < 2:
            continue
        sub_hz = (series.source_hz / k) if series.source_hz else None
        for off in range(k):
            sub = MeasurementSeries(
                channel=series.channel, samples=series.samples[off::k], units=series.units, source_hz=sub_hz
            )
            if len(sub) >= 4:
                r = scorer(sub)
                if r is not None:
                    results.append(r)
    if len(results) < 2:
        return 1.0
    numeric = [r for r in results if isinstance(r, (int, float)) and not isinstance(r, bool)]
    if len(numeric) == len(results):
        mean = sum(numeric) / len(numeric)
        var = sum((x - mean) ** 2 for x in numeric) / len(numeric)
        cv = (var ** 0.5) / (abs(mean) + 1e-9)
        return max(0.0, min(1.0, 1.0 / (1.0 + cv)))
    counts: dict[object, int] = {}
    for r in results:
        counts[r] = counts.get(r, 0) + 1
    return max(counts.values()) / len(results)


@dataclass(frozen=True)
class ChannelScore:
    """A channel's selection salience, computed on the canonical (physical-time) grid so it does not depend
    on how densely the channel was sampled. ``score = support × amplitude × regularity × stability`` — the
    Build-3 selection contract, every factor in periods/seconds/[0,1], never in frames."""

    channel: str
    score: float
    support: float  # number of PERIODS observed (duration / period), NOT number of samples — fps-invariant
    amplitude: float  # robust amplitude over one estimated period, in the channel's native units
    regularity: float  # periodicity score (autocorrelation peak height) in [0, 1]
    stability: float  # how stable the period estimate is under temporal subsampling, in [0, 1]
    period_s: float | None


def channel_score(
    series: MeasurementSeries, hz: float | None = None, band_s: tuple[float, float] | None = None
) -> ChannelScore:
    """Score one channel's rep-shapedness on the canonical grid (the Build-3 selection contract).

    Every factor is rate-invariant by construction: ``regularity`` and ``period`` come from the
    autocorrelation in seconds, ``support`` is periods-observed (duration/period, and duration is
    fps-independent), ``amplitude`` is a percentile span per period, and ``stability`` asks whether the
    period survives decimation. Their product rewards a channel that oscillates LARGE, REGULAR, MANY times,
    and ROBUSTLY — and that judgement is the same whether the channel was sampled at 3 or 160 fps. Pass an
    explicit ``hz`` so every channel is scored on the SAME grid; ``band_s`` restricts the plausible period."""
    period = dominant_period(series, band_s=band_s, hz=hz)
    regularity = periodicity_score(series, band_s=band_s, hz=hz)
    if period and period > 0:
        amplitude = robust_amplitude(series, window_s=period)
        support = series.duration / period
    else:  # no resolvable cycle -> no support; amplitude over the whole span for reporting only
        amplitude = robust_amplitude(series, window_s=series.duration or 1.0)
        support = 0.0
    stability = subsample_stability(series, lambda x: dominant_period(x, band_s=band_s, hz=hz))
    score = support * amplitude * regularity * stability
    return ChannelScore(series.channel, score, support, amplitude, regularity, stability, period)


__all__ = [
    "MeasurementSeries",
    "ChannelScore",
    "CANON_HZ",
    "to_time_grid",
    "periodicity_score",
    "dominant_period",
    "robust_amplitude",
    "subsample_stability",
    "channel_score",
]
