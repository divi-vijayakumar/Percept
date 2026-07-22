"""M4 — SLO CI asserts: reflex p95, $/stream-hour, fovea call-reduction.

Three pre-registered budget constants, each computed from telemetry the ALREADY-LANDED runtime/
harness pieces produce — this module adds no new production instrumentation. It wraps:

* :class:`percept.harness.tier0.MotionGate` — the real tier-0 gate (``evaluate()``), timed with
  plain ``time.perf_counter()`` either side of the existing call (no new gate logic).
* :func:`percept.observability.judge_usage` — the already-landed hook that pulls optional token/
  cost figures off a :class:`~percept.contracts.Verdict` (``Verdict.usage`` is an existing field;
  this module does not add it).
* :class:`percept.harness.metrics.CostSample` / :class:`~percept.harness.metrics.CostSummary` —
  the already-landed $/stream-hour aggregator (``percept/harness/metrics.py``, card A13). This
  module supplies the one missing piece: turning judge_usage()'s TOKEN COUNTS into a ``cost_usd``
  figure via a per-model price table, so ``CostSample`` has something to ingest.
* :class:`percept.harness.tier0.LoopReport` — ``.reduction`` already computes
  ``1 - vision_calls / frames`` (the fovea's call-suppression fraction); this module just names the
  read a "call reduction" and attaches a floor.

Every fixture the accompanying test drives is OFFLINE and KEYLESS (synthetic frames / a scripted
:class:`~percept.contracts.Verdict`, no network, no API keys) — see ``tests/test_slo_asserts.py``.

--------------------------------------------------------------------------------------------------
THRESHOLD RATIONALE — pre-registered BEFORE inspecting any measured run, per the card's "no
tune-to-green" rule. Each constant states WHY that number, grounded in something the architecture
already committed to elsewhere in the tree (never invented fresh to make a particular run pass).
--------------------------------------------------------------------------------------------------

1. ``REFLEX_P95_THRESHOLD_S`` — the tier-0 gate's OWN decision latency (frame in -> GateDecision
   out, ``MotionGate.evaluate()``), explicitly NOT the VLM judge round-trip (that latency budget is
   ``tests/test_p_lat_base.py``'s unrelated ``REFLEX_INFEASIBLE_THRESHOLD_S = 2.0``, about the judge
   feeling instantaneous to a user — a different "reflex").
   ``percept.sampling.SensePolicy.max_fps`` (default ``15.0``, also ``percept.watchprofile``'s
   default) is the runtime's already-documented COST CEILING — the densest the loop ever asks
   tier-0 to run (``percept/sampling.py`` module docstring: "bound by max_fps — the cost ceiling").
   At that ceiling the frame budget is ``1/15 s ~= 66.7ms``. For a still scene to be "effectively
   free" (the card's own phrase) the reflex path must consume a SMALL, FIXED fraction of that
   budget, leaving headroom for capture/decode/the rest of the loop around it — the bar is set at
   15% of the max_fps frame period: ``0.15 * (1/15) ~= 0.0100s``.

2. ``DOLLARS_PER_STREAM_HOUR_CEILING_USD`` — reuses, rather than invents, the ceiling already
   pre-registered in ``tests/test_p_econ.py`` (Viability §7/G2, card A24):
   ``PLACEHOLDER_COST_CEILING_USD_PER_STREAM_HOUR = 5.0``, itself a placeholder for
   ``eval/viability/reference_config.yaml``'s ``numeric_bars.cost_ceiling_usd_per_stream_hour``
   (currently ``TODO`` pending ``bars_locked: true``). Using the SAME number here (rather than a
   fresh guess) keeps exactly one "$/stream-hour ceiling" figure live in the tree instead of two
   that could drift apart.

3. ``FOVEA_CALL_REDUCTION_FLOOR`` — the architecture's minimum bar for tier-0 gating to be worth
   its complexity/latency risk AT ALL: gating must suppress AT LEAST half the naive "judge every
   frame" VLM spend, or a dense baseline would be simpler and no more expensive. Measured production
   tracks in ``docs/reference/benchmark-methodology.md`` range from 15.4% to 87.1% depending on
   scene density — this floor is evaluated against a REPRESENTATIVE mostly-still deployment scene
   (the primary economic claim: "a still scene is effectively free"), not the busiest observed
   track, and is a floor, not a target.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

from .harness.metrics import CostSample, CostSummary, PercentileSummary
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

#: See module docstring §2 — reused, not reinvented, from tests/test_p_econ.py's
#: PLACEHOLDER_COST_CEILING_USD_PER_STREAM_HOUR (Viability §7/G2 numeric_bars, pending bars_locked).
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
    fabricated price, indistinguishable from a real one, and a client with a ``cost_usd`` bar would
    be judging their deployment against our guess (one such client computed ~$16/stream-hour this way
    and correctly refused to report it).

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
    aggregator (``percept/harness/metrics.py``, card A13) for the actual $/stream-hour division —
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
# 4. Settle-window bound — the SLO on a concern's provisional-delivery DELAY (card Y10)
# ---------------------------------------------------------------------------
#
# Y10's delivery choice: a finding is emitted PROVISIONALLY on the rising edge, and a later
# replace/retract on the same episode arrives as a SUPERSEDING revision (linked, never a silent
# swallow) — NOT withheld behind a "settle window" wide enough to catch a late retraction. A real
# correction can land 21-34s after the original claim; withholding EVERY incident for a window that
# wide would destroy latency for every claim to salvage the relation for a few. So the default is
# provisional-immediate, and any settle window a concern DOES declare is bounded here by an explicit
# SLO rather than left to grow to 34s.
#
# THRESHOLD RATIONALE (pre-registered, no tune-to-green — same discipline as §§1-3): a settle window
# is pure ADDED delivery latency on every fire of that concern. The tree already committed to a
# latency ceiling past which a judged decision stops "feeling instantaneous" to a user:
# ``tests/test_p_lat_base.py``'s ``REFLEX_INFEASIBLE_THRESHOLD_S = 2.0`` (the judge round-trip
# "reflex" budget). A settle window may spend UP TO that already-accepted budget waiting for a
# same-turn correction to consolidate, but no more — beyond it the delay is no longer "settling", it
# is the wide-window withholding this card exists to reject. The ceiling is that same 2.0s (reused,
# not reinvented), and 34s — a real late-retraction gap — sits far outside it, so declaring one is a
# loud ``ValueError`` at construction, never a silently honoured latency regression.

#: The DEFAULT settle window: provisional-immediate. A concern that declares nothing emits on the
#: rising edge exactly as today (the card's "a concern that declares nothing gets today's behaviour").
SETTLE_WINDOW_DEFAULT_S = 0.0

#: The hard ceiling on a DECLARED settle window — the judge round-trip reflex budget
#: (``REFLEX_INFEASIBLE_THRESHOLD_S = 2.0``, ``tests/test_p_lat_base.py``), reused not reinvented.
#: A window beyond this is the wide-window withholding Y10 rejects; declaring one raises.
SETTLE_WINDOW_MAX_S = 2.0


@dataclass(frozen=True)
class SettleWindow:
    """A concern's declared provisional-delivery settle window (card Y10) — CONCERN-SPECIFIC and
    SLO-bounded.

    A settle window is the small delay a concern may spend after the rising edge, letting a same-turn
    correction consolidate BEFORE the incident is emitted, so a very-fast replace/retract collapses
    into a single settled incident instead of an incident-plus-immediate-revision. It is a deliberate,
    bounded latency trade — NOT the wide window that would withhold every fire to catch a 34s
    retraction (that is what the provisional-incident + superseding-revision path exists to avoid).

    :attr:`seconds` defaults to :data:`SETTLE_WINDOW_DEFAULT_S` (``0.0`` — provisional-immediate, the
    default a concern that declares nothing gets). Construction enforces the SLO: a negative window is
    nonsensical and a window past :data:`SETTLE_WINDOW_MAX_S` is the latency regression the card
    rejects — both raise :class:`ValueError` LOUDLY rather than silently degrade latency.
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
