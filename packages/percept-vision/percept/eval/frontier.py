"""The operating-frontier report (§3.2) — the honest form of a quality claim.

This is the report a 1.0.0 release publishes. Its whole purpose is to make a quality claim
*legible and refutable*: not "recall is 0.97", but "over 6.2 scheduled watch-hours and 31 labeled
positive episodes, FIRE recall is >= 0.874 at one-sided 95%, which does NOT clear a 0.90 bar; 29
clean episodes with zero misses would".

What this module deliberately does NOT do
-----------------------------------------

* **It does not re-implement matching.** Every match / miss / false-FIRE / duplicate count comes
  from :func:`percept.harness.golden_eval.score` — the deterministic one-to-one event-level
  matcher. If the frontier report and ``percept eval``'s differential ever disagreed about whether
  a fire matched, that would be two yardsticks, which is the failure mode this codebase keeps
  refusing.
* **It does not re-implement the funnel.** Degraded time, ASK/FIRE/SILENT gate counts and
  unresolved judge work come from :class:`percept.tracing.TraceSummary` (card D8a) — the same
  object ``doctor``, ``eval`` and the visualizer read, so this report cannot disagree with them.
* **It does not approximate its intervals.** Bounds come from :mod:`percept.eval.exact_ci` —
  Garwood/Poisson for rates, Clopper–Pearson for proportions, consistent with
  ``docs/reference/eval-hours-costing.md``. A normal approximation at these counts is wrong, not
  merely loose.
* **It does not invent a bar.** Bars are supplied by the caller. With no bars the report prints
  bounds and says ``NOT_ASSESSED``.

The three honesty rules, stated as code
---------------------------------------

1. **Report the bound, not the point estimate.** Every headline number carries its exact one-sided
   95% bound; :class:`MetricBound` keeps the point estimate and the bound side by side and the
   renderer prints the bound in the claim position.
2. **Refuse to bound what the evidence cannot support.** With zero exposure or zero labeled
   episodes the bound is ``None``, the verdict is ``NO_EVIDENCE``, and the report states how many
   hours / episodes WOULD be required for the declared bar (from the costing doc's formulas).
3. **Keep abstentions and degraded time visible.** ASKs, suppressed decisions and judge work that
   never resolved are itemised in the :class:`AbstentionLedger`; degraded time is subtracted from
   coverage but NEVER from the denominator of a rate. A watch scheduled for an hour that was blind
   for 20 minutes still owes those 20 minutes.

Both denominators, always
-------------------------

Costing doc §4: "per stream-hour" alone hides concurrency and coverage, which is exactly how a
cost/quality claim becomes misleading. Every rate here is published twice — per **scheduled
watch-hour** (the bar-eligible number, concurrency already in the denominator) and per
**source-hour at a stated active-watch count** (the deployment-economics number). Neither
substitutes for the other, and :class:`Exposure` refuses to report the second without the stated
concurrency count.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..contracts import JudgmentEvent
from ..harness.golden_eval import GroundTruthEpisode, MatchOutcome, score
from ..tracing import TraceSummary
from .exact_ci import (
    DEFAULT_CONFIDENCE,
    clopper_pearson_lower,
    poisson_upper_bound,
    required_exposure,
    required_positives,
)

__all__ = [
    "CORE_GATING_ARCHETYPES",
    "SECONDS_PER_HOUR",
    "FrontierInputError",
    "FrontierBars",
    "Exposure",
    "AbstentionLedger",
    "EvidenceVerdict",
    "MetricBound",
    "FixtureEvidence",
    "ArchetypeFrontier",
    "FrontierVerdict",
    "OperatingFrontierReport",
    "evidence_from_fixture",
    "build_frontier",
    "load_frontier_evidence",
    "load_bars",
    "render_text",
]

#: The six core gating archetypes (costing doc §5; mirrors
#: ``eval/viability/ground_truth.py::CORE_ARCHETYPE_ELIGIBLE_AFTER_S``, which lives in the
#: gitignored viability tree and is therefore NOT importable from the shipped package). Declared
#: here so the report can state, per archetype, that an archetype produced NO evidence at all —
#: a silently absent row is how a partial report gets read as a complete one.
CORE_GATING_ARCHETYPES: tuple[str, ...] = (
    "state",
    "transition",
    "brief_action",
    "persistence",
    "absence",
    "repeated_episode",
)

SECONDS_PER_HOUR: float = 3600.0


class FrontierInputError(ValueError):
    """Frontier evidence is missing, malformed, or self-contradictory."""


# ---------------------------------------------------------------------------
# 1. Bars — supplied, never invented
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrontierBars:
    """The operating point being claimed. Both fields optional; absent means "report, don't judge".

    ``false_fire_max`` is per **scheduled watch-hour** (never per stream-hour) and ``recall_min``
    is FIRE recall over labeled positive episodes — the two quantities the costing doc prices.
    """

    false_fire_max: float | None = None
    recall_min: float | None = None
    confidence: float = DEFAULT_CONFIDENCE
    source: str = "(none supplied)"
    declared_at: str = ""

    def __post_init__(self) -> None:
        if self.false_fire_max is not None and self.false_fire_max <= 0:
            raise FrontierInputError(
                f"false_fire_max must be > 0 per scheduled watch-hour, got {self.false_fire_max!r}"
            )
        if self.recall_min is not None and not 0.0 < self.recall_min < 1.0:
            raise FrontierInputError(
                f"recall_min must be in (0, 1), got {self.recall_min!r} — a bar of 1.0 cannot be "
                "supported by any finite exact bound."
            )
        if not 0.0 < self.confidence < 1.0:
            raise FrontierInputError(f"confidence must be in (0, 1), got {self.confidence!r}")

    @property
    def declared(self) -> bool:
        return self.false_fire_max is not None or self.recall_min is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "false_fire_max_per_scheduled_watch_hour": self.false_fire_max,
            "recall_min": self.recall_min,
            "confidence": self.confidence,
            "source": self.source,
            "declared_at": self.declared_at,
        }


# ---------------------------------------------------------------------------
# 2. Exposure — BOTH denominators, explicitly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Exposure:
    """Evaluated time, in both denominators, plus the degraded slice.

    ``scheduled_watch_hours`` — hours a watch was *scheduled to be armed*. This is the bar-eligible
    denominator: concurrency is already inside it (three watches armed for an hour on one camera is
    three scheduled watch-hours), and degraded time stays inside it (a blind watch still owes its
    scheduled time).

    ``source_hours`` + ``active_watches`` — the deployment-economics denominator. ``source_hours``
    alone is refused: a per-source-hour rate without a stated active-watch count is exactly the
    underspecified "per stream-hour" number the costing doc §4 rejects.

    ``degraded_hours`` — scheduled time the watch could not structurally have fired in (source
    unhealthy, judge degraded/dropped). Subtracted from COVERAGE, never from a rate denominator.
    """

    scheduled_watch_hours: float = 0.0
    source_hours: float = 0.0
    active_watches: float = 0.0
    degraded_hours: float = 0.0

    def __post_init__(self) -> None:
        for name in ("scheduled_watch_hours", "source_hours", "active_watches", "degraded_hours"):
            if getattr(self, name) < 0:
                raise FrontierInputError(f"{name} must be >= 0, got {getattr(self, name)!r}")
        if self.degraded_hours > self.scheduled_watch_hours + 1e-9:
            raise FrontierInputError(
                f"degraded_hours ({self.degraded_hours}) exceeds scheduled_watch_hours "
                f"({self.scheduled_watch_hours}) — a watch cannot be blind for longer than it was "
                "scheduled to be armed."
            )

    def __add__(self, other: "Exposure") -> "Exposure":
        return Exposure(
            scheduled_watch_hours=self.scheduled_watch_hours + other.scheduled_watch_hours,
            source_hours=self.source_hours + other.source_hours,
            active_watches=self.active_watches + other.active_watches,
            degraded_hours=self.degraded_hours + other.degraded_hours,
        )

    @property
    def healthy_watch_hours(self) -> float:
        return max(0.0, self.scheduled_watch_hours - self.degraded_hours)

    @property
    def coverage(self) -> float | None:
        """Healthy fraction of SCHEDULED watch-time. ``None`` when nothing was scheduled."""
        if self.scheduled_watch_hours <= 0:
            return None
        return self.healthy_watch_hours / self.scheduled_watch_hours

    @property
    def degraded_fraction(self) -> float | None:
        cov = self.coverage
        return None if cov is None else 1.0 - cov

    @property
    def mean_active_watches(self) -> float | None:
        """Mean concurrently-armed watches per source. ``None`` when not stated."""
        if self.source_hours <= 0 or self.active_watches <= 0:
            return None
        return self.active_watches / self.source_hours

    def per_source_hour(self, count: int) -> float | None:
        """``count`` per source-hour — ``None`` unless the active-watch count was stated."""
        if self.source_hours <= 0 or self.mean_active_watches is None:
            return None
        return count / self.source_hours

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheduled_watch_hours": self.scheduled_watch_hours,
            "healthy_watch_hours": self.healthy_watch_hours,
            "degraded_hours": self.degraded_hours,
            "coverage": self.coverage,
            "source_hours": self.source_hours,
            "mean_active_watches_per_source": self.mean_active_watches,
        }


# ---------------------------------------------------------------------------
# 3. Abstention ledger — visible, never netted out
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AbstentionLedger:
    """Every decision that was NOT a FIRE, itemised. Nothing here is silently dropped.

    ``asks`` and ``silents`` are deliberate abstentions the gate made. ``suppressed`` is the typed
    suppression bucket (start-anchored / refractory) — a silent the baseline gate WOULD have
    delivered, which is a different fact. ``unresolved`` is judge work that entered the funnel and
    produced no decision at all (degraded + dropped): the most important of the four, because it
    is the one a point estimate would quietly erase.
    """

    fires: int = 0
    asks: int = 0
    silents: int = 0
    suppressed: int = 0
    unresolved: int = 0

    def __add__(self, other: "AbstentionLedger") -> "AbstentionLedger":
        return AbstentionLedger(
            fires=self.fires + other.fires,
            asks=self.asks + other.asks,
            silents=self.silents + other.silents,
            suppressed=self.suppressed + other.suppressed,
            unresolved=self.unresolved + other.unresolved,
        )

    @property
    def decisions(self) -> int:
        """Decisions actually emitted (fire + ask + silent). Excludes unresolved work by design."""
        return self.fires + self.asks + self.silents

    @property
    def abstentions(self) -> int:
        """Everything that was not a FIRE, including work that never resolved."""
        return self.asks + self.silents + self.suppressed + self.unresolved

    @property
    def ask_rate(self) -> float | None:
        """ASKs per emitted decision. ``None`` when no decision was emitted."""
        if self.decisions <= 0:
            return None
        return self.asks / self.decisions

    def to_dict(self) -> dict[str, Any]:
        return {
            "fires": self.fires,
            "asks": self.asks,
            "silents": self.silents,
            "suppressed": self.suppressed,
            "unresolved_judge_work": self.unresolved,
            "decisions_emitted": self.decisions,
            "abstentions_total": self.abstentions,
            "ask_rate_per_decision": self.ask_rate,
        }


# ---------------------------------------------------------------------------
# 4. A bounded metric
# ---------------------------------------------------------------------------


class EvidenceVerdict(str, Enum):
    """Whether the evidence supports the claim being made about a metric."""

    #: A bound exists and clears the declared bar.
    SUFFICIENT = "SUFFICIENT"
    #: A bound exists but does NOT clear the declared bar (the bar is not met — say so).
    INSUFFICIENT = "INSUFFICIENT"
    #: A bound exists; no bar was declared, so nothing is asserted.
    NOT_ASSESSED = "NOT_ASSESSED"
    #: No bound exists at all — zero exposure / zero labeled episodes. Never report a number here.
    NO_EVIDENCE = "NO_EVIDENCE"


@dataclass(frozen=True)
class MetricBound:
    """One metric: its point estimate, its exact one-sided bound, and what the evidence supports.

    The point estimate is kept for context and NEVER printed alone in a claim position. ``bound`` is
    the claim. ``required_evidence``/``required_units`` answer "what would it take?" whenever the
    bound fails or does not exist.
    """

    name: str
    point: float | None
    bound: float | None
    direction: str  # "upper" (rate: bound is a ceiling) | "lower" (proportion: bound is a floor)
    method: str
    confidence: float = DEFAULT_CONFIDENCE
    observed: str = ""
    bar: float | None = None
    verdict: EvidenceVerdict = EvidenceVerdict.NOT_ASSESSED
    reason: str = ""
    required_evidence: float | None = None
    required_units: str = ""

    @property
    def clears_bar(self) -> bool | None:
        if self.bar is None or self.bound is None:
            return None
        return self.bound <= self.bar if self.direction == "upper" else self.bound >= self.bar

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "point_estimate": self.point,
            "bound": self.bound,
            "bound_direction": self.direction,
            "confidence": self.confidence,
            "method": self.method,
            "observed": self.observed,
            "bar": self.bar,
            "verdict": self.verdict.value,
            "reason": self.reason,
            "required_evidence": self.required_evidence,
            "required_units": self.required_units,
        }


def _false_fire_bound(
    unmatched: int, exposure: Exposure, bars: FrontierBars
) -> MetricBound:
    """Upper bound on false FIREs per SCHEDULED WATCH-HOUR (Garwood/Poisson exact)."""
    t = exposure.scheduled_watch_hours
    point = (unmatched / t) if t > 0 else None
    bound = poisson_upper_bound(unmatched, t, confidence=bars.confidence)
    observed = f"{unmatched} false FIRE(s) over {t:.4g} scheduled watch-hour(s)"
    bar = bars.false_fire_max

    if bound is None:
        need = required_exposure(bar, unmatched, confidence=bars.confidence) if bar else None
        return MetricBound(
            name="false_fires_per_scheduled_watch_hour",
            point=None,
            bound=None,
            direction="upper",
            method="Poisson exact (Garwood, chi-square)",
            confidence=bars.confidence,
            observed=observed,
            bar=bar,
            verdict=EvidenceVerdict.NO_EVIDENCE,
            reason=(
                "Zero scheduled watch-hours were evaluated. A false-FIRE rate cannot be bounded "
                "against no observed time, and the point estimate does not exist either."
            ),
            required_evidence=need,
            required_units="clean scheduled watch-hours",
        )

    if bar is None:
        return MetricBound(
            name="false_fires_per_scheduled_watch_hour",
            point=point,
            bound=bound,
            direction="upper",
            method="Poisson exact (Garwood, chi-square)",
            confidence=bars.confidence,
            observed=observed,
            verdict=EvidenceVerdict.NOT_ASSESSED,
            reason="No false_fire_max bar was declared; the bound is reported, nothing is asserted.",
        )

    need = required_exposure(bar, unmatched, confidence=bars.confidence)
    if bound <= bar:
        return MetricBound(
            name="false_fires_per_scheduled_watch_hour",
            point=point,
            bound=bound,
            direction="upper",
            method="Poisson exact (Garwood, chi-square)",
            confidence=bars.confidence,
            observed=observed,
            bar=bar,
            verdict=EvidenceVerdict.SUFFICIENT,
            reason=(
                f"{t:.4g} scheduled watch-hour(s) with {unmatched} false FIRE(s) bound the rate at "
                f"<= {bound:.4g}/h, which clears the {bar:.4g}/h bar."
            ),
            required_evidence=need,
            required_units="clean scheduled watch-hours",
        )
    return MetricBound(
        name="false_fires_per_scheduled_watch_hour",
        point=point,
        bound=bound,
        direction="upper",
        method="Poisson exact (Garwood, chi-square)",
        confidence=bars.confidence,
        observed=observed,
        bar=bar,
        verdict=EvidenceVerdict.INSUFFICIENT,
        reason=(
            f"The evidence is too thin to claim <= {bar:.4g}/h: {t:.4g} scheduled watch-hour(s) "
            f"with {unmatched} false FIRE(s) supports only <= {bound:.4g}/h. "
            + (
                f"{need:.4g} clean scheduled watch-hour(s) at this event count would be required."
                if need is not None
                else ""
            )
        ),
        required_evidence=need,
        required_units="clean scheduled watch-hours",
    )


def _recall_bound(
    name: str, matches: int, episodes: int, bars: FrontierBars, *, what: str
) -> MetricBound:
    """Lower bound on a recall proportion (Clopper–Pearson exact)."""
    point = (matches / episodes) if episodes > 0 else None
    bound = clopper_pearson_lower(matches, episodes, confidence=bars.confidence)
    observed = f"{matches}/{episodes} labeled positive episode(s) {what}"
    bar = bars.recall_min
    misses = episodes - matches

    if bound is None:
        need = required_positives(bar, 0, confidence=bars.confidence) if bar else None
        return MetricBound(
            name=name,
            point=None,
            bound=None,
            direction="lower",
            method="Clopper-Pearson exact (beta)",
            confidence=bars.confidence,
            observed=observed,
            bar=bar,
            verdict=EvidenceVerdict.NO_EVIDENCE,
            reason=(
                "Zero labeled positive episodes. Recall is undefined here — it is NOT 1.0, and a "
                "vacuous 1.0 is the single most misleading number this report could print."
            ),
            required_evidence=None if need is None else float(need),
            required_units="labeled positive episodes (assuming zero misses)",
        )

    need_i = required_positives(bar, misses, confidence=bars.confidence) if bar else None
    need = None if need_i is None else float(need_i)
    if bar is None:
        return MetricBound(
            name=name,
            point=point,
            bound=bound,
            direction="lower",
            method="Clopper-Pearson exact (beta)",
            confidence=bars.confidence,
            observed=observed,
            verdict=EvidenceVerdict.NOT_ASSESSED,
            reason="No recall_min bar was declared; the bound is reported, nothing is asserted.",
        )
    if bound >= bar:
        return MetricBound(
            name=name, point=point, bound=bound, direction="lower",
            method="Clopper-Pearson exact (beta)", confidence=bars.confidence,
            observed=observed, bar=bar, verdict=EvidenceVerdict.SUFFICIENT,
            reason=(
                f"{episodes} labeled episode(s) with {misses} miss(es) bound recall at "
                f">= {bound:.4g}, which clears the {bar:.4g} bar."
            ),
            required_evidence=need,
            required_units=f"labeled positive episodes (at {misses} miss(es))",
        )
    return MetricBound(
        name=name, point=point, bound=bound, direction="lower",
        method="Clopper-Pearson exact (beta)", confidence=bars.confidence,
        observed=observed, bar=bar, verdict=EvidenceVerdict.INSUFFICIENT,
        reason=(
            f"The evidence is too thin to claim >= {bar:.4g}: {episodes} labeled episode(s) with "
            f"{misses} miss(es) supports only >= {bound:.4g}. "
            + (
                f"{need_i} labeled positive episode(s) at {misses} miss(es) would be required."
                if need_i is not None
                else "No finite number of episodes supports this bar at this miss count."
            )
        ),
        required_evidence=need,
        required_units=f"labeled positive episodes (at {misses} miss(es))",
    )


# ---------------------------------------------------------------------------
# 5. Evidence — scored ONCE, by golden_eval.score
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FixtureEvidence:
    """One fixture's contribution to one archetype, already scored by ``golden_eval.score``.

    Counts, not pooled episodes: two fixtures of the same archetype are aggregated by summing these
    counts, which keeps ``episode_id`` uniqueness a per-fixture property (``score`` requires it)
    instead of forcing a global rename.
    """

    fixture_id: str
    archetype: str
    episodes: int
    fire_matches: int
    assisted_matches: int
    duplicate_fires: int
    unmatched_false_fires: int
    exposure: Exposure
    ledger: AbstentionLedger

    @property
    def fire_misses(self) -> int:
        return self.episodes - self.fire_matches

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "archetype": self.archetype,
            "episodes": self.episodes,
            "fire_matches": self.fire_matches,
            "fire_misses": self.fire_misses,
            "assisted_matches": self.assisted_matches,
            "duplicate_fires": self.duplicate_fires,
            "unmatched_false_fires": self.unmatched_false_fires,
            "exposure": self.exposure.to_dict(),
            "ledger": self.ledger.to_dict(),
        }


def evidence_from_fixture(
    *,
    fixture_id: str,
    archetype: str,
    episodes: Sequence[GroundTruthEpisode],
    events: Sequence[JudgmentEvent],
    exposure: Exposure,
    summary: TraceSummary | None = None,
) -> FixtureEvidence:
    """Score one fixture into :class:`FixtureEvidence`.

    FIRE recall and assisted recall are two calls into the SAME matcher with different
    ``predicted_actions`` — ``{"fire"}`` and ``{"fire", "ask"}`` — so "assisted" means exactly
    "the episode was surfaced to the user at all, whether as a fire or as a question", scored by
    the identical window rule.

    Degraded time and unresolved judge work are read off ``summary`` (:class:`TraceSummary`) when
    one is supplied, so this report cannot disagree with ``doctor``/``eval``/the visualizer. The
    gate counts come from the events themselves — those ARE the decisions being scored — with the
    trace's ``suppressed`` bucket folded in, since a suppression emits no ``JudgmentEvent``.
    """
    if not archetype:
        raise FrontierInputError(f"fixture {fixture_id!r}: archetype is required (non-empty)")

    fire = score(episodes, events, predicted_actions=frozenset({"fire"}))
    assisted = score(episodes, events, predicted_actions=frozenset({"fire", "ask"}))

    if summary is not None:
        degraded_s = max(0.0, summary.watch_duration_s - summary.source_healthy_s)
        if degraded_s > 0 and exposure.degraded_hours <= 0:
            exposure = Exposure(
                scheduled_watch_hours=exposure.scheduled_watch_hours,
                source_hours=exposure.source_hours,
                active_watches=exposure.active_watches,
                degraded_hours=min(
                    exposure.scheduled_watch_hours, degraded_s / SECONDS_PER_HOUR
                ),
            )

    ledger = AbstentionLedger(
        fires=sum(1 for e in events if e.action == "fire"),
        asks=sum(1 for e in events if e.action == "ask"),
        silents=sum(1 for e in events if e.action == "silent"),
        suppressed=0 if summary is None else int(summary.gate_outcomes.get("suppressed", 0)),
        unresolved=0 if summary is None else summary.judge_degraded + summary.judge_dropped,
    )

    return FixtureEvidence(
        fixture_id=fixture_id,
        archetype=archetype,
        episodes=len(episodes),
        fire_matches=fire.true_positives,
        assisted_matches=assisted.true_positives,
        duplicate_fires=sum(
            1 for p in fire.prediction_scores if p.outcome is MatchOutcome.DUPLICATE
        ),
        unmatched_false_fires=fire.unmatched_false_fires,
        exposure=exposure,
        ledger=ledger,
    )


# ---------------------------------------------------------------------------
# 6. Per-archetype frontier
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArchetypeFrontier:
    """The full §3.2 picture for ONE archetype (or, with ``archetype='(all archetypes)'``, pooled)."""

    archetype: str
    fixtures: tuple[str, ...]
    episodes: int
    fire_matches: int
    assisted_matches: int
    duplicate_fires: int
    unmatched_false_fires: int
    exposure: Exposure
    ledger: AbstentionLedger
    recall: MetricBound
    assisted_recall: MetricBound
    false_fire_rate: MetricBound

    @property
    def fire_misses(self) -> int:
        return self.episodes - self.fire_matches

    @property
    def has_evidence(self) -> bool:
        return bool(self.fixtures)

    @property
    def duplicate_fires_per_episode(self) -> float | None:
        """Duplicate FIREs per labeled episode. ``None`` with no episodes (not 0.0)."""
        if self.episodes <= 0:
            return None
        return self.duplicate_fires / self.episodes

    @property
    def false_fires_per_source_hour(self) -> float | None:
        return self.exposure.per_source_hour(self.unmatched_false_fires)

    @property
    def asks_per_scheduled_watch_hour(self) -> float | None:
        t = self.exposure.scheduled_watch_hours
        return None if t <= 0 else self.ledger.asks / t

    @property
    def verdicts(self) -> tuple[EvidenceVerdict, ...]:
        return (self.recall.verdict, self.false_fire_rate.verdict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "archetype": self.archetype,
            "fixtures": list(self.fixtures),
            "episodes": self.episodes,
            "fire_matches": self.fire_matches,
            "fire_misses": self.fire_misses,
            "assisted_matches": self.assisted_matches,
            "duplicate_fires": self.duplicate_fires,
            "duplicate_fires_per_episode": self.duplicate_fires_per_episode,
            "unmatched_false_fires": self.unmatched_false_fires,
            "exposure": self.exposure.to_dict(),
            "denominators": {
                "primary": "per scheduled watch-hour",
                "false_fires_per_scheduled_watch_hour_bound": self.false_fire_rate.bound,
                "false_fires_per_source_hour": self.false_fires_per_source_hour,
                "mean_active_watches_per_source": self.exposure.mean_active_watches,
                "note": (
                    "A per-source-hour rate is meaningless without the active-watch count; it is "
                    "reported as null when that count was not stated."
                ),
            },
            "coverage": self.exposure.coverage,
            "degraded_hours": self.exposure.degraded_hours,
            "ask_rate_per_decision": self.ledger.ask_rate,
            "asks_per_scheduled_watch_hour": self.asks_per_scheduled_watch_hour,
            "ledger": self.ledger.to_dict(),
            "bounds": {
                "fire_recall": self.recall.to_dict(),
                "assisted_recall": self.assisted_recall.to_dict(),
                "false_fire_rate": self.false_fire_rate.to_dict(),
            },
        }


def _aggregate(
    archetype: str, items: Sequence[FixtureEvidence], bars: FrontierBars
) -> ArchetypeFrontier:
    exposure = Exposure()
    ledger = AbstentionLedger()
    for it in items:
        exposure = exposure + it.exposure
        ledger = ledger + it.ledger
    episodes = sum(it.episodes for it in items)
    fire_matches = sum(it.fire_matches for it in items)
    assisted_matches = sum(it.assisted_matches for it in items)
    duplicates = sum(it.duplicate_fires for it in items)
    unmatched = sum(it.unmatched_false_fires for it in items)
    return ArchetypeFrontier(
        archetype=archetype,
        fixtures=tuple(sorted(it.fixture_id for it in items)),
        episodes=episodes,
        fire_matches=fire_matches,
        assisted_matches=assisted_matches,
        duplicate_fires=duplicates,
        unmatched_false_fires=unmatched,
        exposure=exposure,
        ledger=ledger,
        recall=_recall_bound("fire_recall", fire_matches, episodes, bars, what="FIRED in-window"),
        assisted_recall=_recall_bound(
            "assisted_recall", assisted_matches, episodes, bars,
            what="surfaced in-window (FIRE or ASK)",
        ),
        false_fire_rate=_false_fire_bound(unmatched, exposure, bars),
    )


def _empty_archetype(archetype: str, bars: FrontierBars) -> ArchetypeFrontier:
    return _aggregate(archetype, (), bars)


# ---------------------------------------------------------------------------
# 7. The report
# ---------------------------------------------------------------------------


class FrontierVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_ASSESSED = "NOT_ASSESSED"


@dataclass(frozen=True)
class OperatingFrontierReport:
    """The whole §3.2 report: pooled frontier, per-archetype breakdown, ledger, and the refusals."""

    bars: FrontierBars
    pooled: ArchetypeFrontier
    archetypes: tuple[ArchetypeFrontier, ...]
    verdict: FrontierVerdict
    verdict_reason: str
    evidence: tuple[FixtureEvidence, ...] = ()
    notes: tuple[str, ...] = ()

    def archetype(self, name: str) -> ArchetypeFrontier | None:
        return next((a for a in self.archetypes if a.archetype == name), None)

    @property
    def covered_archetypes(self) -> tuple[str, ...]:
        return tuple(a.archetype for a in self.archetypes if a.has_evidence)

    @property
    def missing_archetypes(self) -> tuple[str, ...]:
        return tuple(
            a for a in CORE_GATING_ARCHETYPES if a not in set(self.covered_archetypes)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "percept.eval.frontier/1",
            "bars": self.bars.to_dict(),
            "denominator_discipline": {
                "primary": "per scheduled watch-hour",
                "secondary": "per source-hour at a stated active-watch count",
                "why": (
                    "'per stream-hour' alone hides concurrency (N watches on one feed) and "
                    "coverage (degraded time that structurally could not fire). Both denominators "
                    "are published; neither substitutes for the other."
                ),
            },
            "evaluated_hours": {
                "scheduled_watch_hours": self.pooled.exposure.scheduled_watch_hours,
                "healthy_watch_hours": self.pooled.exposure.healthy_watch_hours,
                "degraded_hours": self.pooled.exposure.degraded_hours,
                "source_hours": self.pooled.exposure.source_hours,
                "mean_active_watches_per_source": self.pooled.exposure.mean_active_watches,
            },
            "pooled": self.pooled.to_dict(),
            "per_archetype": [a.to_dict() for a in self.archetypes],
            "core_archetypes": list(CORE_GATING_ARCHETYPES),
            "archetypes_without_evidence": list(self.missing_archetypes),
            "fixtures": [e.to_dict() for e in self.evidence],
            "notes": list(self.notes),
            "verdict": self.verdict.value,
            "verdict_reason": self.verdict_reason,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)


def build_frontier(
    evidence: Sequence[FixtureEvidence],
    *,
    bars: FrontierBars | None = None,
    archetypes: Sequence[str] = CORE_GATING_ARCHETYPES,
) -> OperatingFrontierReport:
    """Assemble the operating-frontier report from already-scored fixture evidence.

    Every core archetype gets a row, including the ones with no evidence at all — an absent row is
    how a partial report gets misread as a complete one. Archetypes outside the core six are
    reported too (appended, flagged in the notes), never dropped.
    """
    bars = bars or FrontierBars()
    by_archetype: dict[str, list[FixtureEvidence]] = {}
    for item in evidence:
        by_archetype.setdefault(item.archetype, []).append(item)

    ordered: list[str] = list(archetypes)
    extra = sorted(set(by_archetype) - set(ordered))
    ordered.extend(extra)

    rows = tuple(
        _aggregate(name, by_archetype.get(name, ()), bars) for name in ordered
    )
    pooled = _aggregate("(all archetypes)", tuple(evidence), bars)

    notes: list[str] = []
    missing = [a for a in archetypes if not by_archetype.get(a)]
    if missing:
        notes.append(
            f"{len(missing)} of {len(archetypes)} core gating archetype(s) produced NO evidence in "
            f"this run: {', '.join(missing)}. Their rows are printed empty rather than omitted; "
            f"this report does not describe them, and a claim covering them is not supported here."
        )
    if extra:
        notes.append(
            f"Non-core archetype(s) present and reported separately: {', '.join(extra)}."
        )
    if pooled.exposure.mean_active_watches is None:
        notes.append(
            "No active-watch count was stated, so no per-source-hour rate is published. A bare "
            "per-stream-hour figure is underspecified (costing doc §4) and is omitted rather than "
            "printed with an implied concurrency of 1."
        )
    if pooled.ledger.unresolved:
        notes.append(
            f"{pooled.ledger.unresolved} judge task(s) entered the funnel and never produced a "
            f"decision (degraded + dropped). They are counted as abstentions in the ledger and are "
            f"NOT removed from any denominator."
        )

    verdict, reason = _verdict(bars, rows, archetypes)
    return OperatingFrontierReport(
        bars=bars,
        pooled=pooled,
        archetypes=rows,
        verdict=verdict,
        verdict_reason=reason,
        evidence=tuple(evidence),
        notes=tuple(notes),
    )


def _verdict(
    bars: FrontierBars,
    rows: Sequence[ArchetypeFrontier],
    core: Sequence[str],
) -> tuple[FrontierVerdict, str]:
    """PASS only when EVERY core archetype clears EVERY declared bar. Silence is never a pass."""
    if not bars.declared:
        return (
            FrontierVerdict.NOT_ASSESSED,
            "No operating bars were declared. The frontier is reported with exact one-sided "
            f"{int(bars.confidence * 100)}% bounds; percept does not invent a bar on a client's "
            "behalf.",
        )
    core_rows = [r for r in rows if r.archetype in set(core)]
    no_evidence = [r.archetype for r in core_rows if not r.has_evidence]
    failing: list[str] = []
    thin: list[str] = []
    for r in core_rows:
        if not r.has_evidence:
            continue
        for m in (r.recall, r.false_fire_rate):
            if m.bar is None:
                continue
            if m.verdict is EvidenceVerdict.NO_EVIDENCE:
                thin.append(f"{r.archetype}/{m.name}")
            elif m.verdict is EvidenceVerdict.INSUFFICIENT:
                failing.append(f"{r.archetype}/{m.name}")
    if no_evidence:
        return (
            FrontierVerdict.NOT_ASSESSED,
            f"{len(no_evidence)} core archetype(s) produced no evidence "
            f"({', '.join(no_evidence)}). A frontier claim that skips archetypes is a "
            "cherry-picked read; the run is NOT_ASSESSED rather than partially passed.",
        )
    if thin:
        return (
            FrontierVerdict.NOT_ASSESSED,
            f"{len(thin)} metric(s) could not be bounded at all "
            f"({', '.join(sorted(thin))}); see the required-evidence column.",
        )
    if failing:
        return (
            FrontierVerdict.FAIL,
            f"{len(failing)} metric(s) are not supported by the evidence at "
            f"{int(bars.confidence * 100)}%: {', '.join(sorted(failing))}. The bound, not the "
            "point estimate, is the claim.",
        )
    return (
        FrontierVerdict.PASS,
        f"Every declared bar is cleared by its exact one-sided "
        f"{int(bars.confidence * 100)}% bound in every core archetype.",
    )


# ---------------------------------------------------------------------------
# 8. Loading — reuses percept.eval.differential's fixture layout
# ---------------------------------------------------------------------------


def _exposure_from_payload(payload: Mapping[str, Any], fixture_id: str) -> Exposure:
    raw = payload.get("exposure")
    if raw is None:
        raise FrontierInputError(
            f"fixture {fixture_id!r}: ground_truth.json has no 'exposure' block. The frontier "
            "report cannot be produced without evaluated hours — a recall/false-fire claim with an "
            "unstated denominator is exactly what §3.2 exists to prevent. Expected keys: "
            "scheduled_watch_hours (required), source_hours, active_watches, degraded_hours."
        )
    if "scheduled_watch_hours" not in raw:
        raise FrontierInputError(
            f"fixture {fixture_id!r}: exposure.scheduled_watch_hours is required (the bar-eligible "
            "denominator). source_hours alone is underspecified — see costing doc §4."
        )
    return Exposure(
        scheduled_watch_hours=float(raw["scheduled_watch_hours"]),
        source_hours=float(raw.get("source_hours", 0.0)),
        active_watches=float(raw.get("active_watches", 0.0)),
        degraded_hours=float(raw.get("degraded_hours", 0.0)),
    )


def load_frontier_evidence(
    fixtures_root: str | Path,
    *,
    arm: str = "candidate",
    split_name: str = "test",
    held_out_only: bool = True,
) -> tuple[FixtureEvidence, ...]:
    """Load + score frontier evidence from the ``percept eval`` fixture tree.

    Reuses :mod:`percept.eval.differential`'s loaders verbatim (``load_fixtures``,
    ``load_held_out_split``, the events/trace readers) so a frontier run and a differential run see
    byte-identical inputs. Each fixture's ``ground_truth.json`` additionally declares:

    * ``archetype`` — one of the six core gating archetypes (others are allowed but flagged),
    * ``exposure`` — ``scheduled_watch_hours`` (required), ``source_hours``, ``active_watches``,
      ``degraded_hours``.

    ``held_out_only`` keeps the frontier honest for the same reason the differential does: a bound
    computed on exploratory fixtures is not held-out evidence.
    """
    from . import differential as _diff  # local import: differential's CLI imports this module

    root = Path(fixtures_root)
    fixtures = _diff.load_fixtures(root)
    split = _diff.load_held_out_split(root, fixtures, split_name=split_name)
    held = set(split.held_out)

    out: list[FixtureEvidence] = []
    for fixture in fixtures:
        if held_out_only and fixture.fixture_id not in held:
            continue
        payload = json.loads((Path(fixture.path) / "ground_truth.json").read_text() or "{}")
        archetype = str(payload.get("archetype", "")).strip()
        if not archetype:
            raise FrontierInputError(
                f"fixture {fixture.fixture_id!r}: ground_truth.json declares no 'archetype'. The "
                f"§3.2 report is per-archetype; known core archetypes: "
                f"{', '.join(CORE_GATING_ARCHETYPES)}."
            )
        arm_dir = Path(fixture.path) / arm
        out.append(
            evidence_from_fixture(
                fixture_id=fixture.fixture_id,
                archetype=archetype,
                episodes=fixture.episodes,
                events=_diff._load_events(arm_dir / "events.json"),
                exposure=_exposure_from_payload(payload, fixture.fixture_id),
                summary=_diff._load_summary(arm_dir / "trace.json"),
            )
        )
    return tuple(out)


def load_bars(path: str | Path | None) -> FrontierBars:
    """Load operating bars from a JSON/YAML document. ``None`` -> an undeclared, report-only bars."""
    if path is None:
        return FrontierBars()
    from . import differential as _diff  # local import (see load_frontier_evidence)

    data = _diff._load_structured(path)
    ff = data.get("false_fire_max")
    rc = data.get("recall_min")
    return FrontierBars(
        false_fire_max=None if ff is None else float(ff),
        recall_min=None if rc is None else float(rc),
        confidence=float(data.get("confidence", DEFAULT_CONFIDENCE)),
        source=str(path),
        declared_at=str(data.get("declared_at", "")),
    )


# ---------------------------------------------------------------------------
# 9. Rendering
# ---------------------------------------------------------------------------


def _fmt(value: float | None, spec: str = ".4g") -> str:
    return "—" if value is None else format(value, spec)


def _bound_line(m: MetricBound) -> str:
    arrow = "<=" if m.direction == "upper" else ">="
    return (
        f"    {m.name:<38}{arrow} {_fmt(m.bound):<10}"
        f"(point {_fmt(m.point)}; bar {_fmt(m.bar)}; {m.verdict.value})"
    )


def render_text(report: OperatingFrontierReport) -> str:
    """The human §3.2 report. Every claim on this page is a bound, with its evidence beside it."""
    lines: list[str] = []
    add = lines.append
    conf = int(report.bars.confidence * 100)
    ex = report.pooled.exposure

    add(f"percept eval — OPERATING FRONTIER (§3.2), one-sided {conf}% exact bounds")
    add("=" * 78)
    add("")
    add("BARS")
    add(f"  source              : {report.bars.source}"
        f"{' (declared ' + report.bars.declared_at + ')' if report.bars.declared_at else ''}")
    add(f"  false_fire_max      : {_fmt(report.bars.false_fire_max)} per SCHEDULED WATCH-HOUR")
    add(f"  recall_min          : {_fmt(report.bars.recall_min)} (FIRE recall)")
    add("")
    add("EVALUATED HOURS — BOTH DENOMINATORS (costing doc §4)")
    add(f"  scheduled watch-hours (bar-eligible) : {_fmt(ex.scheduled_watch_hours)}")
    add(f"    of which healthy                   : {_fmt(ex.healthy_watch_hours)}")
    add(f"    of which degraded                  : {_fmt(ex.degraded_hours)} "
        f"(counted against coverage, NOT removed from any denominator)")
    add(f"  source-hours                         : {_fmt(ex.source_hours)}")
    add(f"    mean active watches per source     : {_fmt(ex.mean_active_watches)}")
    add("  A per-source-hour rate without a stated active-watch count is underspecified and is")
    add("  omitted rather than printed with an implied concurrency of 1.")
    add("")
    add("POOLED FRONTIER (all archetypes together — NOT a substitute for the per-archetype rows)")
    p = report.pooled
    add(_bound_line(p.recall))
    add(_bound_line(p.assisted_recall))
    add(_bound_line(p.false_fire_rate))
    add(f"    {'false FIREs per source-hour':<38}"
        f"{_fmt(p.false_fires_per_source_hour)} (second denominator)")
    add(f"    {'coverage':<38}{_fmt(ex.coverage)} "
        f"(degraded {_fmt(ex.degraded_fraction)})")
    add(f"    {'ASK rate (per emitted decision)':<38}{_fmt(p.ledger.ask_rate)}")
    add(f"    {'ASKs per scheduled watch-hour':<38}{_fmt(p.asks_per_scheduled_watch_hour)}")
    add(f"    {'duplicate FIREs per episode':<38}{_fmt(p.duplicate_fires_per_episode)}")
    add("")
    add("ABSTENTION LEDGER (nothing here is netted out of the numbers above)")
    led = p.ledger
    add(f"  fires={led.fires}  asks={led.asks}  silents={led.silents}  "
        f"suppressed={led.suppressed}  unresolved judge work={led.unresolved}")
    add(f"  decisions emitted={led.decisions}  abstentions total={led.abstentions}")
    add("")
    add(f"PER-ARCHETYPE ({len(CORE_GATING_ARCHETYPES)} core gating archetypes)")
    header = (f"  {'archetype':<18}{'eps':<6}{'hit':<6}{'miss':<6}{'recall>=':<11}"
              f"{'ff':<5}{'ff/wh<=':<11}{'cov':<8}{'ask':<8}{'dup/ep':<8}{'hours':<8}")
    add(header)
    for a in report.archetypes:
        if not a.has_evidence:
            add(f"  {a.archetype:<18}{'—':<6}{'—':<6}{'—':<6}{'NO EVIDENCE':<11}"
                f"{'—':<5}{'NO EVIDENCE':<11}{'—':<8}{'—':<8}{'—':<8}{'0':<8}")
            continue
        add(f"  {a.archetype:<18}{a.episodes:<6}{a.fire_matches:<6}{a.fire_misses:<6}"
            f"{_fmt(a.recall.bound):<11}{a.unmatched_false_fires:<5}"
            f"{_fmt(a.false_fire_rate.bound):<11}{_fmt(a.exposure.coverage):<8}"
            f"{_fmt(a.ledger.ask_rate):<8}{_fmt(a.duplicate_fires_per_episode):<8}"
            f"{_fmt(a.exposure.scheduled_watch_hours):<8}")
    add("")
    add("WHAT THE EVIDENCE SUPPORTS (per archetype, per metric)")
    for a in report.archetypes:
        add(f"  {a.archetype}")
        if not a.has_evidence:
            add("      NO EVIDENCE — this archetype was not exercised in this run. No claim about "
                "it is supported by this report.")
            continue
        for m in (a.recall, a.assisted_recall, a.false_fire_rate):
            add(f"      {m.name} [{m.verdict.value}] — {m.observed}")
            add(f"          {m.reason}")
            if m.required_evidence is not None and m.verdict is not EvidenceVerdict.SUFFICIENT:
                add(f"          would require: {_fmt(m.required_evidence)} {m.required_units}")
    add("")
    if report.notes:
        add("NOTES")
        for n in report.notes:
            add(f"  - {n}")
        add("")
    add(f"VERDICT: {report.verdict.value}")
    add(f"  {report.verdict_reason}")
    return "\n".join(lines)
