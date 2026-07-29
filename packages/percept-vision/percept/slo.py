"""SLO CI asserts: tier-0 gate p95, $/stream-hour, fovea call-reduction — plus the MEASURED
judge round-trip latency distribution (§5), which is the only latency figure Percept publishes.

Three pre-registered budget constants, each computed from telemetry the ALREADY-LANDED runtime/
harness pieces produce — this module adds no new production instrumentation. It wraps:

* :class:`percept.harness.tier0.MotionGate` — the real tier-0 gate (``evaluate()``), timed with
  plain ``time.perf_counter()`` either side of the existing call (no new gate logic).
* :func:`percept.observability.judge_usage` — the already-landed hook that pulls optional token/
  cost figures off a :class:`~percept.contracts.Verdict` (``Verdict.usage`` is an existing field;
  this module does not add it).
* :class:`percept.harness.metrics.CostSample` / :class:`~percept.harness.metrics.CostSummary` —
  the already-landed $/stream-hour aggregator (``percept/harness/metrics.py``). This
  module supplies the one missing piece: turning judge_usage()'s TOKEN COUNTS into a ``cost_usd``
  figure via a per-model price table, so ``CostSample`` has something to ingest.
* :class:`percept.harness.tier0.LoopReport` — ``.reduction`` already computes
  ``1 - vision_calls / frames`` (the fovea's call-suppression fraction); this module just names the
  read a "call reduction" and attaches a floor.

Every fixture the accompanying test drives is OFFLINE and KEYLESS (synthetic frames / a scripted
:class:`~percept.contracts.Verdict`, no network, no API keys) — see ``tests/test_slo_asserts.py``.

--------------------------------------------------------------------------------------------------
THRESHOLD RATIONALE — pre-registered BEFORE inspecting any measured run, preserving the "no
tune-to-green" rule. Each constant states WHY that number, grounded in something the architecture
already committed to elsewhere in the tree (never invented fresh to make a particular run pass).
--------------------------------------------------------------------------------------------------

1. ``REFLEX_P95_THRESHOLD_S`` — the tier-0 gate's OWN decision latency (frame in -> GateDecision
   out, ``MotionGate.evaluate()``), explicitly NOT the VLM judge round-trip (that is §5's
   :data:`JUDGE_ROUND_TRIP_P95_CEILING_S`, about the judge feeling instantaneous to a user — a
   different "reflex"). This §1 number is a LOCAL, in-process measurement with no provider and no
   deadline anywhere near it, so nothing in §5's censoring story touches it; it is unchanged.
   ``percept.sampling.SensePolicy.max_fps`` (default ``15.0``, also ``percept.watchprofile``'s
   default) is the runtime's already-documented COST CEILING — the densest the loop ever asks
   tier-0 to run (``percept/sampling.py`` module docstring: "bound by max_fps — the cost ceiling").
   At that ceiling the frame budget is ``1/15 s ~= 66.7ms``. For a still scene to be "effectively
   free", the reflex path must consume a SMALL, FIXED fraction of that
   budget, leaving headroom for capture/decode/the rest of the loop around it — the bar is set at
   15% of the max_fps frame period: ``0.15 * (1/15) ~= 0.0100s``.

2. ``DOLLARS_PER_STREAM_HOUR_CEILING_USD`` — reuses, rather than invents, the ceiling already
   pre-registered in ``tests/test_p_econ.py``:
   ``PLACEHOLDER_COST_CEILING_USD_PER_STREAM_HOUR = 5.0``, itself a placeholder for
   ``eval/viability/reference_config.yaml``'s ``numeric_bars.cost_ceiling_usd_per_stream_hour``
   (currently ``TODO`` pending ``bars_locked: true``). Using the SAME number here (rather than a
   fresh guess) keeps exactly one "$/stream-hour ceiling" figure live in the tree instead of two
   that could drift apart.

3. ``FOVEA_CALL_REDUCTION_FLOOR`` — the architecture's minimum bar for tier-0 gating to be worth
   its complexity/latency risk AT ALL: gating must suppress AT LEAST half the naive "judge every
   frame" VLM spend, or a dense baseline would be simpler and no more expensive. Measured production
   tracks in ``docs/evaluation/methodology.md`` range from 15.4% to 87.1% depending on
   scene density — this floor is evaluated against a REPRESENTATIVE mostly-still deployment scene
   (the primary economic claim: "a still scene is effectively free"), not the busiest observed
   track, and is a floor, not a target.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

from .harness.metrics import CostSample, CostSummary, PercentileSummary, percentile
from .harness.tier0 import LoopReport, MotionGate

__all__ = [
    "REFLEX_P95_THRESHOLD_S",
    "measure_reflex_latencies",
    "reflex_p95",
    "ModelPrice",
    "PRICE_TABLE",
    "DEFAULT_PRICE",
    "DOLLARS_PER_STREAM_HOUR_CEILING_USD",
    "JudgeCallSample",
    "price_usage",
    "price_usage_or_none",
    "dollars_per_stream_hour",
    "FOVEA_CALL_REDUCTION_FLOOR",
    "fovea_call_reduction",
    "SETTLE_WINDOW_DEFAULT_S",
    "SETTLE_WINDOW_MAX_S",
    "SettleWindow",
    # §5 — the measured judge round-trip latency distribution (the latency PROMISE)
    "LatencyConditions",
    "LatencyDistribution",
    "measure_judge_latency",
    "DELIBERATE_JUDGE_LATENCY",
    "REFLEX_JUDGE_LATENCY",
    "REFLEX_JUDGE_LATENCY_CENSORED",
    "JUDGE_ROUND_TRIP_LATENCY",
    "JUDGE_ROUND_TRIP_P95_CEILING_S",
    "judge_latency_promise",
]

# ---------------------------------------------------------------------------
# 1. REFLEX p95 — the tier-0 gate's own decision latency (frame in -> edge decision out)
# ---------------------------------------------------------------------------

#: See module docstring §1. 15% of the max_fps=15.0 frame period (percept.sampling.SensePolicy /
#: percept.watchprofile's shared cost-ceiling default), NOT fit to any measured run.
REFLEX_P95_THRESHOLD_S = 0.0100


def measure_reflex_latencies(
    gate: MotionGate, frames: Sequence, *, ts_start: float = 0.0, dt: float = 1.0 / 15.0
) -> list[float]:
    """Wall-clock time ``MotionGate.evaluate()`` (the real tier-0 gate) over a frame stream.

    One ``perf_counter()`` sample per frame, around the ALREADY-LANDED gate call — no new gate
    logic, no new production instrumentation. ``dt`` defaults to the ``max_fps=15.0`` cadence (the
    densest the runtime ever drives the gate), matching the threshold's own basis.
    """
    latencies: list[float] = []
    t = ts_start
    for frame in frames:
        start = time.perf_counter()
        gate.evaluate(frame, ts=t)
        latencies.append(time.perf_counter() - start)
        t += dt
    return latencies


def reflex_p95(latencies: Sequence[float]) -> float:
    """p95 of sampled reflex latencies (nearest-rank, :class:`~percept.harness.metrics.PercentileSummary`)."""
    return PercentileSummary.of(latencies).p95


# ---------------------------------------------------------------------------
# 2. $/stream-hour — judge_usage() token counts x a per-model price table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelPrice:
    """$ per 1M tokens, input/output, for one judge model.

    Illustrative catalog-style pricing DATA (not magic numbers inline in the cost math) — swap in
    negotiated/current rates; this module never calls a live pricing API (offline/keyless
    invariant: no network in a CI assertion).
    """

    input_per_mtok: float
    output_per_mtok: float


# Keyed by the model id percept/config.py's Models dataclass already declares as the shipped
# defaults (PERCEPT_VISION_MODEL / PERCEPT_CONDUCTOR_MODEL / PERCEPT_SPECIALIST_MODEL) — the price
# table's keys track the models this codebase actually names elsewhere, not invented labels.
PRICE_TABLE: dict[str, ModelPrice] = {
    "claude-opus-4-8": ModelPrice(input_per_mtok=15.00, output_per_mtok=75.00),  # vision_model default
    "claude-haiku-4-5": ModelPrice(input_per_mtok=1.00, output_per_mtok=5.00),  # conductor/specialist default
}

#: A judge call against a model absent from PRICE_TABLE prices at the most expensive known tier —
#: fail conservative (never silently under-count spend for an un-catalogued model).
DEFAULT_PRICE = ModelPrice(input_per_mtok=15.00, output_per_mtok=75.00)

#: A conservative provisional ceiling for stream-hour cost accounting. Deployments should set
#: their own budget from observed workload and provider pricing.
DOLLARS_PER_STREAM_HOUR_CEILING_USD = 5.0


@dataclass(frozen=True)
class JudgeCallSample:
    """One judge call's usage dict (exactly :func:`percept.observability.judge_usage`'s return
    shape) + the model it was billed against — the minimal input :func:`price_usage` needs."""

    usage: dict[str, float]
    model: str


def price_usage(
    usage: dict[str, float], *, model: str, table: dict[str, ModelPrice] = PRICE_TABLE
) -> float:
    """``judge_usage()``'s token counts x the model's price -> ``cost_usd`` for ONE judge call.

    ``usage`` is exactly what :func:`percept.observability.judge_usage` returns for a
    :class:`~percept.contracts.Verdict`. If the verdict already carried an explicit ``cost_usd`` (a
    backend that reports its own billed cost), that value wins over a token x price estimate — the
    backend's own number is ground truth when present.
    """
    if "cost_usd" in usage:
        return float(usage["cost_usd"])
    price = table.get(model, DEFAULT_PRICE)
    in_tok = usage.get("input_tokens", 0.0)
    out_tok = usage.get("output_tokens", 0.0)
    return (in_tok / 1_000_000.0) * price.input_per_mtok + (out_tok / 1_000_000.0) * price.output_per_mtok


def price_usage_or_none(
    usage: dict[str, float],
    *,
    model: str | None,
    table: dict[str, ModelPrice] = PRICE_TABLE,
) -> float | None:
    """The RUN-PATH pricing rule: tokens -> dollars, or ``None`` when we cannot say honestly.

    :func:`price_usage` is the *offline harness* rule — it falls back to :data:`DEFAULT_PRICE` so a
    CI budget assertion fails conservative on an un-catalogued model. That fallback is exactly wrong
    on the reporting path: a DEFAULT_PRICE-derived figure printed as a run's ``cost_usd`` is a
    fabricated price, indistinguishable from a real one, and an operator with a ``cost_usd`` bar would
    be judging their deployment against a guess. Such estimates must not be reported as measured cost.

    So here:

    * an explicit backend-reported ``cost_usd`` wins — it is the billed number, ground truth;
    * a model PRESENT in ``table`` prices its tokens;
    * anything else returns ``None`` — rendered ``—``. Unknown is safe; a number a bar can pass is not.
    """
    if "cost_usd" in usage:
        return float(usage["cost_usd"])
    price = table.get(model) if model else None
    if price is None:
        return None
    in_tok = float(usage.get("input_tokens", 0.0) or 0.0)
    out_tok = float(usage.get("output_tokens", 0.0) or 0.0)
    return (in_tok / 1_000_000.0) * price.input_per_mtok + (out_tok / 1_000_000.0) * price.output_per_mtok


def dollars_per_stream_hour(
    calls: Sequence[JudgeCallSample],
    *,
    elapsed_s: float,
    stream_hours: float,
    table: dict[str, ModelPrice] = PRICE_TABLE,
) -> float:
    """$/stream-hour over a fixture run's judge calls.

    Prices each call via :func:`price_usage`, then hands the total to the ALREADY-LANDED
    :class:`~percept.harness.metrics.CostSample` / :class:`~percept.harness.metrics.CostSummary`
    aggregator (``percept/harness/metrics.py``) for the actual $/stream-hour division —
    this function supplies only the missing "token counts -> cost_usd" step.
    """
    total_cost = sum(price_usage(c.usage, model=c.model, table=table) for c in calls)
    sample = CostSample(
        judgments=len(calls), cost_usd=total_cost, elapsed_s=elapsed_s, stream_hours=stream_hours
    )
    summary = CostSummary(
        total_judgments=sample.judgments,
        total_cost_usd=sample.cost_usd,
        total_elapsed_s=sample.elapsed_s,
        total_stream_hours=sample.stream_hours,
    )
    return summary.dollars_per_stream_hour


# ---------------------------------------------------------------------------
# 3. Fovea call-reduction floor
# ---------------------------------------------------------------------------

#: See module docstring §3 — a floor for a representative mostly-still deployment scene, not fit to
#: any single measured track.
FOVEA_CALL_REDUCTION_FLOOR = 0.50


def fovea_call_reduction(report: LoopReport) -> float:
    """The fraction of frames tier-0 suppressed vs the ungated "judge every frame" baseline.

    ``LoopReport.reduction`` (``percept/harness/tier0.py``) already computes
    ``1 - vision_calls / frames`` — this is a thin, intention-revealing re-export so the SLO test
    doesn't reach into the harness report's field name directly.
    """
    return report.reduction


# ---------------------------------------------------------------------------
# 4. Settle-window bound — the SLO on a concern's provisional-delivery DELAY
# ---------------------------------------------------------------------------
#
# The delivery choice: a finding is emitted PROVISIONALLY on the rising edge, and a later
# replace/retract on the same episode arrives as a SUPERSEDING revision (linked, never a silent
# swallow) — NOT withheld behind a "settle window" wide enough to catch a late retraction. A real
# correction can land 21-34s after the original claim; withholding EVERY incident for a window that
# wide would destroy latency for every claim to salvage the relation for a few. So the default is
# provisional-immediate, and any settle window a concern DOES declare is bounded here by an explicit
# SLO rather than left to grow to 34s.
#
# THRESHOLD RATIONALE (pre-registered, no tune-to-green — same discipline as §§1-3): a settle window
# is pure ADDED delivery latency on every fire of that concern. 2.0s is the ceiling past which added
# delay stops being "settling" and becomes the wide-window withholding this guard exists to reject;
# 34s — a real late-retraction gap — sits far outside it, so declaring one is a loud ``ValueError``
# at construction, never a silently honoured latency regression.
#
# ANCHOR RETIRED, VALUE HELD. This ceiling used to justify itself by REUSING a "judge round-trip
# reflex budget" of 2.0s — a deadline the runtime enforced. That deadline is gone (see §5: it sat
# below the median call time and cancelled the majority of judgments), so the citation would now be
# a claim about a number that no longer exists. The 2.0s is therefore restated as what it always
# actually measured: a bound on ADDED DELIVERY DELAY — how long a client will wait past a finding
# that is already in hand — which is a property of human patience, not of provider speed.
#
# It is deliberately NOT re-derived upward against §5's measured judge latency. Doing so would widen
# the permitted withholding for no reason connected to what this bound governs: the settle window
# starts AFTER the verdict arrives, so how long the provider took is irrelevant to it. Holding a
# number when its old justification collapses — rather than letting it drift with an unrelated
# measurement — is the conservative reading, and the one that does not quietly cost latency.

#: The DEFAULT settle window: provisional-immediate. A concern that declares nothing emits on the
#: rising edge exactly as today: a concern that declares nothing gets the existing behaviour.
SETTLE_WINDOW_DEFAULT_S = 0.0

#: The hard ceiling on a DECLARED settle window — a bound on ADDED DELIVERY DELAY past a verdict
#: already in hand (see the rationale above; the old "judge round-trip reflex budget" anchor is
#: retired, the value held). A window beyond this is the wide-window withholding this design
#: rejects; declaring one raises.
SETTLE_WINDOW_MAX_S = 2.0


@dataclass(frozen=True)
class SettleWindow:
    """A concern's declared provisional-delivery settle window — CONCERN-SPECIFIC and
    SLO-bounded.

    A settle window is the small delay a concern may spend after the rising edge, letting a same-turn
    correction consolidate BEFORE the incident is emitted, so a very-fast replace/retract collapses
    into a single settled incident instead of an incident-plus-immediate-revision. It is a deliberate,
    bounded latency trade — NOT the wide window that would withhold every fire to catch a 34s
    retraction (that is what the provisional-incident + superseding-revision path exists to avoid).

    :attr:`seconds` defaults to :data:`SETTLE_WINDOW_DEFAULT_S` (``0.0`` — provisional-immediate, the
    default a concern that declares nothing gets). Construction enforces the SLO: a negative window is
    nonsensical and a window past :data:`SETTLE_WINDOW_MAX_S` is the defined latency regression;
    both raise :class:`ValueError` LOUDLY rather than silently degrade latency.
    """

    seconds: float = SETTLE_WINDOW_DEFAULT_S

    def __post_init__(self) -> None:
        if self.seconds < 0.0:
            raise ValueError(f"settle window must be non-negative, got {self.seconds!r}s")
        if self.seconds > SETTLE_WINDOW_MAX_S:
            raise ValueError(
                f"settle window {self.seconds!r}s exceeds the SLO ceiling {SETTLE_WINDOW_MAX_S}s — "
                "a window this wide withholds every incident to catch a late retraction, which "
                "destroys latency for every claim; emit provisionally and supersede on the correction "
                "instead"
            )

    @property
    def is_immediate(self) -> bool:
        """True for the default provisional-immediate window (no delay — fire on the rising edge)."""
        return self.seconds <= 0.0

    def holds(self, elapsed_s: float) -> bool:
        """Whether a fire ``elapsed_s`` past the rising edge is still WITHIN the settle window.

        ``True`` means "keep waiting for the window to close"; ``False`` means "emit now". An immediate
        window never holds, so the default path emits on the rising edge with zero added latency."""
        if self.is_immediate:
            return False
        return elapsed_s < self.seconds

    @classmethod
    def resolve(cls, declared: "float | SettleWindow | None") -> "SettleWindow":
        """Normalise a concern's declared setting into a bounded :class:`SettleWindow`.

        ``None`` (the common case — a concern declared nothing) resolves to the provisional-immediate
        default; a bare ``float`` is wrapped (and SLO-checked); an existing :class:`SettleWindow` passes
        through. The SLO bound is enforced in every path via ``__post_init__``."""
        if declared is None:
            return cls()
        if isinstance(declared, SettleWindow):
            return declared
        return cls(seconds=float(declared))


# ---------------------------------------------------------------------------
# 5. Judge round-trip latency — a measured DISTRIBUTION, not a deadline promise
# ---------------------------------------------------------------------------
#
# WHY THIS SECTION EXISTS. Percept used to state its judge latency as a single deadline figure:
# "reflex answers within 1.5s". That sentence was false in the most expensive possible way. The
# deadline sat BELOW the median provider call time (1.5s against a measured p50 of 1.653s), so over
# half of all reflex calls physically could not finish. On the n=100 corpus (build 2d5097d,
# gemini-3.1-flash-lite) 5,224 of 7,304 reflex calls expired and only 960 returned decisive; reflex
# end-to-end recall was 0.137 against deliberate's 0.684 ON IDENTICAL FRAMES. The gap was
# cancellation, not judgment quality.
#
# Worse, the "measured" reflex latency of that run (p50 1.365s, p95 1.4726s, max 1.5155s) was not a
# measurement at all — it was the deadline's own shadow, because every sample that would have landed
# above 1.5s had been thrown away before it could be recorded. A censored distribution quoted as a
# latency promise is circular: it proves only that the truncation worked.
#
# So the promise is now the DISTRIBUTION: p25 / p50 / p95, plus max, n, and the conditions the
# samples were drawn under. A distribution cannot be met "72% of the time" — it either describes the
# samples or it does not. The deadline that remains
# (:data:`percept.backends.judge_pool.REFLEX_PROVIDER_DEADLINE_S`) is a HANG GUARD, sized so a
# judgment completes; it is not quotable as a latency figure and this module never treats it as one.
#
# WHAT IS MEASURED AND WHAT IS PENDING. Only the deliberate-mode figures below are measured
# uncensored. The reflex-uncensored run has not landed yet, so :data:`REFLEX_JUDGE_LATENCY` is
# ``None`` — deliberately, rather than being filled with a plausible number. Every unmeasured field
# on a :class:`LatencyDistribution` is ``None`` and renders as ``—``: this module never emits a
# percentile it did not observe.


@dataclass(frozen=True)
class LatencyConditions:
    """The measurement conditions a latency distribution is only meaningful WITH.

    A percentile with no conditions attached is not a fact about a system, it is a number. Model,
    transport, frame count and corpus each move judge latency by more than the spread between the
    percentiles being reported, so they travel with the figures rather than beside them in prose.
    """

    model: str
    transport: str
    frames: int | None
    corpus: str
    build: str

    def describe(self) -> str:
        frames = "unbounded" if self.frames is None else f"{self.frames} frames"
        return (
            f"model={self.model}; transport={self.transport}; {frames}; "
            f"corpus={self.corpus}; build={self.build}"
        )


def _fmt_s(value: float | None) -> str:
    """A latency field for display — ``—`` when it was never measured, never a stand-in number."""
    return "—" if value is None else f"{value:g}s"


@dataclass(frozen=True)
class LatencyDistribution:
    """A judge round-trip latency distribution: p25/p50/p95 + max + n + conditions.

    ``None`` on any field means NOT MEASURED — it is never a zero, a default, or an estimate. The
    three percentiles are p25/p50/p95 specifically: p50 is the typical call, p95 is the tail a
    deployment has to survive, and p25 is what stops a reader from reconstructing a "usually
    instant" story out of a median alone.

    ``censored_at_s`` marks a distribution truncated by a deadline. Such a distribution describes
    the deadline, not the provider, so :attr:`is_publishable` is ``False`` for it and
    :meth:`describe` says so on every line it renders. It is retained rather than deleted because
    the censored run is the evidence for why the deadline changed.
    """

    p25_s: float | None
    p50_s: float | None
    p95_s: float | None
    max_s: float | None
    n: int | None
    conditions: LatencyConditions
    censored_at_s: float | None = None
    note: str = ""

    def __post_init__(self) -> None:
        ordered = [v for v in (self.p25_s, self.p50_s, self.p95_s, self.max_s) if v is not None]
        if any(v < 0.0 for v in ordered):
            raise ValueError("latency percentiles must be non-negative")
        if ordered != sorted(ordered):
            raise ValueError(
                "latency fields must be non-decreasing across p25 <= p50 <= p95 <= max, got "
                f"{(self.p25_s, self.p50_s, self.p95_s, self.max_s)!r}"
            )
        if self.n is not None and self.n < 1:
            raise ValueError("latency sample count n must be positive when stated")

    @property
    def is_censored(self) -> bool:
        """Whether a deadline truncated these samples (so they describe the deadline, not the provider)."""
        return self.censored_at_s is not None

    @property
    def is_publishable(self) -> bool:
        """Whether this may be quoted as Percept's latency: uncensored, with all three percentiles."""
        return not self.is_censored and None not in (self.p25_s, self.p50_s, self.p95_s)

    def describe(self) -> str:
        """One human line: the three percentiles, max, n, and the conditions — unmeasured fields ``—``."""
        body = (
            f"p25 {_fmt_s(self.p25_s)} · p50 {_fmt_s(self.p50_s)} · p95 {_fmt_s(self.p95_s)} · "
            f"max {_fmt_s(self.max_s)} · n={'—' if self.n is None else self.n} "
            f"[{self.conditions.describe()}]"
        )
        if self.is_censored:
            body = (
                f"CENSORED at {self.censored_at_s:g}s — NOT A MEASUREMENT of provider latency: "
            ) + body
        if self.note:
            body = f"{body} — {self.note}"
        return body


def measure_judge_latency(
    samples: Sequence[float],
    *,
    conditions: LatencyConditions,
    censored_at_s: float | None = None,
    note: str = "",
) -> LatencyDistribution:
    """Build a :class:`LatencyDistribution` from observed round-trip seconds.

    Uses the same nearest-rank :func:`percept.harness.metrics.percentile` the rest of the tree does,
    so a figure published here is bit-for-bit the figure a harness run reports. Pass ``censored_at_s``
    when a deadline was in force during collection — the result then refuses to be publishable, which
    is the whole point.
    """
    if not samples:
        raise ValueError("measure_judge_latency() requires at least one sample")
    return LatencyDistribution(
        p25_s=percentile(samples, 25),
        p50_s=percentile(samples, 50),
        p95_s=percentile(samples, 95),
        max_s=percentile(samples, 100),
        n=len(samples),
        conditions=conditions,
        censored_at_s=censored_at_s,
        note=note,
    )


_N100_CORPUS = "n=100 action corpus (Perception Test track)"
_N100_BUILD = "2d5097d"

#: DELIBERATE mode, same corpus and same frames as the reflex run, and — critically — UNCENSORED:
#: its 30.0s provider deadline is far above the distribution, so nothing was truncated. p50 and p95
#: are the two figures that run reported; p25, max and n were not, and are not invented here.
DELIBERATE_JUDGE_LATENCY = LatencyDistribution(
    p25_s=None,
    p50_s=1.653,
    p95_s=2.2834,
    max_s=None,
    n=None,
    conditions=LatencyConditions(
        model="gemini-3.1-flash-lite",
        transport="ordered timestamped images",
        frames=None,
        corpus=_N100_CORPUS,
        build=_N100_BUILD,
    ),
    note="uncensored (30.0s provider deadline, far above the distribution)",
)

#: The REFLEX run's recorded latencies — RETAINED AS EVIDENCE, NOT AS A FIGURE. Truncated at the
#: 1.5s deadline that has since been replaced; quoting p95 1.4726s as "reflex latency" would be
#: quoting the deadline back to the reader. 5,224 of 7,304 calls expired; 960 returned decisive.
REFLEX_JUDGE_LATENCY_CENSORED = LatencyDistribution(
    p25_s=None,
    p50_s=1.365,
    p95_s=1.4726,
    max_s=1.5155,
    n=None,
    conditions=LatencyConditions(
        model="gemini-3.1-flash-lite",
        transport="ordered timestamped images",
        frames=3,
        corpus=_N100_CORPUS,
        build=_N100_BUILD,
    ),
    censored_at_s=1.5,
    note="5224/7304 calls expired, 960 decisive; end-to-end recall 0.137 vs deliberate 0.684",
)

#: PENDING. The reflex-uncensored run is collecting now. ``None`` until it lands — this module does
#: not carry a placeholder that a reader could mistake for a measurement.
REFLEX_JUDGE_LATENCY: LatencyDistribution | None = None

#: The figure to quote TODAY, with its caveat stated in the same breath. Reflex sends strictly less
#: than deliberate (bounded frames, bounded reply), so the deliberate distribution is a CONSERVATIVE
#: upper bound on reflex on this corpus — an honest stand-in, and explicitly not a reflex measurement.
JUDGE_ROUND_TRIP_LATENCY = REFLEX_JUDGE_LATENCY or DELIBERATE_JUDGE_LATENCY

#: MEASURED, NOT ASPIRATIONAL. The regression bar on judge round-trip p95, derived from the numbers
#: above rather than from a target: the highest measured UNCENSORED p95 we have (deliberate's
#: 2.2834s) plus the tree's own already-committed p95 run-to-run drift allowance (0.2s — the
#: ``latency_drift_within_tolerance`` default in ``tests/test_p_lat_load.py``), rounded up to the
#: next 0.1s: 2.2834 + 0.2 = 2.4834 -> 2.5s.
#:
#: It is a LOOSENING of the 2.0s bar it replaces, and it is stated as such: 2.0s was a bar the
#: pipeline was never measured to clear, held in place by a deadline that discarded the evidence.
#: A bar above the measured value still asserts — it catches a regression — where a bar below it
#: only guarantees a red build. This is not tuned to make a metric pass; it is the measurement plus
#: a pre-existing tolerance, and it moves DOWN the moment the reflex-uncensored run gives us a
#: tighter uncensored p95 to derive from.
JUDGE_ROUND_TRIP_P95_CEILING_S = 2.5


def judge_latency_promise() -> str:
    """Percept's latency statement, rendered from measured data only.

    The single string any doc, report or CLI should print when asked "how fast is it?". It names the
    distribution, marks what is still pending, and never substitutes the hang-guard deadline for a
    measurement.
    """
    lines = [
        f"judge round-trip latency (best available): {JUDGE_ROUND_TRIP_LATENCY.describe()}",
        f"  p95 regression bar (measured, not aspirational): {JUDGE_ROUND_TRIP_P95_CEILING_S:g}s",
    ]
    if REFLEX_JUDGE_LATENCY is None:
        lines.append(
            "  reflex-uncensored: PENDING — the run collecting it is in flight. The figure above is "
            "the deliberate-mode distribution, a conservative upper bound on reflex (reflex sends "
            "bounded frames and a bounded reply), NOT a reflex measurement."
        )
        lines.append(f"  superseded censored reflex run: {REFLEX_JUDGE_LATENCY_CENSORED.describe()}")
    return "\n".join(lines)
