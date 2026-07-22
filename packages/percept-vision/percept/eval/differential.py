"""``percept eval`` — PAIRED differential A/B over client fixtures.

The question a client with a new model actually has is not "how good is percept" — it is
**"does my OCR detector actually help?"**. That is a paired question: the same fixture, scored
twice, once under a baseline plan and once under a candidate plan, and the interesting quantity is
the per-fixture DELTA, not two independent averages printed side by side.

What this module is careful about, and why:

* **It does not reimplement matching.** Every recall/precision number comes from
  :func:`percept.harness.golden_eval.score` — the deterministic one-to-one greedy matcher that
  implements the one-to-one event-level matching rule. If eval and ``doctor`` ever disagree about
  whether a fire matched, that is a bug in one caller, not two competing definitions.
* **It does not reimplement the funnel.** Every cost/admission/judge number comes from
  :class:`percept.tracing.TraceSummary`, loaded via :func:`percept.tracing.load_trace`.
* **It does not invent pass/fail.** Bars are CLIENT-SUPPLIED. With no bars file the report says
  ``NOT_ASSESSED`` and prints the numbers. It will never manufacture a threshold so that something
  looks green.
* **It refuses to tune to green.** The bars, the held-out split, and the threshold/policy values
  are pre-registered into a :class:`PreRegistration` whose ``registration_hash`` covers all of
  them plus both plan hashes. If the configuration that actually ran differs from the one that was
  declared, :func:`run_differential` raises :class:`ConfigDriftError` and scores nothing.
* **It scores its own scope honestly.** The one-to-one event-level matcher understands point/episode incidents. It
  cannot validate supersession, corrections, settled intervals, speaker-attributed claims or
  document identity changes — those concepts do not exist in the contract yet (ADR-006). Rather
  than silently scoring a richer interaction with the wrong yardstick, every fixture declares a
  ``goal_shape`` and gets one of :class:`ScopeVerdict`: ``SUPPORTED`` /
  ``PARTIALLY_SUPPORTED`` / ``UNSUPPORTED_FOR_THIS_GOAL_SHAPE``. Unsupported fixtures are reported
  but are NEVER folded into a delta or scored against a bar.

Fixture directory layout (everything optional except ``ground_truth.json``)::

    fixtures/
      splits.json                     # optional; percept.eval.load_splits — "test" is the held-out split
      <fixture_id>/
        ground_truth.json             # goal_shape + labeled episodes
        <arm>/events.json             # the arm's predicted JudgmentEvents
        <arm>/trace.json              # the arm's trace artifact (percept.tracing.write_json)

``<arm>`` is ``baseline`` / ``candidate`` by default (see :class:`ArmSpec.fixture_dir`).

CLI::

    percept-eval --baseline baseline-plan.yaml --candidate ocr-plan.yaml \\
                 --fixtures ./fixtures --bars bars.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..contracts import GroundTruth, JudgmentEvent, JudgmentTiming
from ..harness.golden_eval import GoldenEvalResult, GroundTruthEpisode, score
from ..tracing import PlanIdentity, TraceSummary, content_hash, load_trace
from . import SPLIT_NAMES, SplitError, load_splits

__all__ = [
    "ScopeVerdict",
    "ScopeAssessment",
    "GOAL_SHAPE_SUPPORT",
    "scope_for_goal_shape",
    "Bar",
    "BarResult",
    "BarsDocument",
    "ArmSpec",
    "Fixture",
    "ArmOutcome",
    "PairedFixtureResult",
    "Delta",
    "HeldOutSplit",
    "PreRegistration",
    "DifferentialReport",
    "ConfigDriftError",
    "EvalInputError",
    "load_plan",
    "load_bars",
    "load_fixtures",
    "load_held_out_split",
    "evaluate_bars",
    "run_differential",
    "render_text",
    "main",
]


# ---------------------------------------------------------------------------
# 0. Errors
# ---------------------------------------------------------------------------


class EvalInputError(ValueError):
    """A fixture/plan/bars file is missing, malformed, or self-contradictory."""


class ConfigDriftError(EvalInputError):
    """The configuration that ran is not the configuration that was pre-registered.

    Raised INSTEAD of producing numbers. A differential whose thresholds, split, plans or bars
    moved after the bars were declared is not evidence — it is tuning, and this is the refusal.
    """


# ---------------------------------------------------------------------------
# 1. Scope — what the one-to-one event-level matcher can and cannot honestly validate (ADR-006)
# ---------------------------------------------------------------------------


class ScopeVerdict(str, Enum):
    """Whether the existing one-to-one event-level matcher can validate a given goal shape AT ALL."""

    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED_FOR_THIS_GOAL_SHAPE = "UNSUPPORTED_FOR_THIS_GOAL_SHAPE"


#: goal shape -> (verdict, why). The "why" is quoted verbatim into the report so a client reading
#: it at 2am learns what percept does not model, not merely that a row is blank.
GOAL_SHAPE_SUPPORT: dict[str, tuple[ScopeVerdict, str]] = {
    "point": (
        ScopeVerdict.SUPPORTED,
        "A point incident is exactly what the one-to-one event-level matcher scores: one fire, "
        "one labeled instant, a closed match window.",
    ),
    "episode": (
        ScopeVerdict.SUPPORTED,
        "A continuous true-episode with a rising edge is the contract's native shape "
        "(ADR-004 §1: at most one outward event per (goal, entity) per episode).",
    ),
    "recurrence": (
        ScopeVerdict.PARTIALLY_SUPPORTED,
        "Repeat occurrences of the same state are scored as independent episodes. Counts and "
        "windows are right; the RELATION between occurrences is not modelled, so a run that "
        "merges two occurrences and a run that splits one are indistinguishable here.",
    ),
    "supersession": (
        ScopeVerdict.UNSUPPORTED_FOR_THIS_GOAL_SHAPE,
        "No Incident references any other Incident (ADR-006 §1). A later claim superseding an "
        "earlier one would be scored as an unrelated peer or eaten as 'refractory'.",
    ),
    "correction": (
        ScopeVerdict.UNSUPPORTED_FOR_THIS_GOAL_SHAPE,
        "A retraction is a new claim ABOUT an old claim, not a confident-false on the original "
        "condition (ADR-006 §2). The gate cannot see it, so scoring it would be a well-logged "
        "wrong answer.",
    ),
    "settled_interval": (
        ScopeVerdict.UNSUPPORTED_FOR_THIS_GOAL_SHAPE,
        "There is no settle window in the contract (ADR-006 §2); 'the value that stood at the end' "
        "is not a thing percept emits, so it cannot be matched.",
    ),
    "speaker_attributed_claim": (
        ScopeVerdict.UNSUPPORTED_FOR_THIS_GOAL_SHAPE,
        "Incidents carry no speaker attribution (ADR-006). Matching on time alone would credit a "
        "fire that attributed the claim to the wrong person.",
    ),
    "document_identity_change": (
        ScopeVerdict.UNSUPPORTED_FOR_THIS_GOAL_SHAPE,
        "Document identity changes are relations between findings about the same subject; percept "
        "models the world, not its own findings (ADR-006).",
    ),
}


@dataclass(frozen=True)
class ScopeAssessment:
    """The scope verdict for one goal shape, with the reason quoted into the report."""

    goal_shape: str
    verdict: ScopeVerdict
    reason: str

    @property
    def scorable(self) -> bool:
        """Only SUPPORTED / PARTIALLY_SUPPORTED shapes contribute to deltas and bars."""
        return self.verdict is not ScopeVerdict.UNSUPPORTED_FOR_THIS_GOAL_SHAPE

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_shape": self.goal_shape,
            "verdict": self.verdict.value,
            "reason": self.reason,
            "scorable": self.scorable,
        }


def scope_for_goal_shape(goal_shape: str) -> ScopeAssessment:
    """Map a declared goal shape onto a :class:`ScopeVerdict`.

    An UNKNOWN shape is ``UNSUPPORTED_FOR_THIS_GOAL_SHAPE``, never optimistically SUPPORTED — the
    whole point of this axis is that silence about scope is the failure mode.
    """
    shape = (goal_shape or "").strip()
    known = GOAL_SHAPE_SUPPORT.get(shape)
    if known is None:
        return ScopeAssessment(
            goal_shape=shape or "(undeclared)",
            verdict=ScopeVerdict.UNSUPPORTED_FOR_THIS_GOAL_SHAPE,
            reason=(
                f"Unknown goal shape {shape or '(undeclared)'!r}. eval refuses to assume the "
                f"one-to-one event-level point/episode matcher is the right yardstick for a shape "
                f"it has never been shown. "
                f"Known shapes: {', '.join(sorted(GOAL_SHAPE_SUPPORT))}."
            ),
        )
    verdict, reason = known
    return ScopeAssessment(goal_shape=shape, verdict=verdict, reason=reason)


# ---------------------------------------------------------------------------
# 2. Bars — CLIENT-SUPPLIED. Absent bars means "report", never "invent".
# ---------------------------------------------------------------------------

#: Metrics a bar may be declared against. ``delta`` metrics are candidate-minus-baseline.
BAR_METRICS: tuple[str, ...] = (
    "recall_delta",
    "precision_delta",
    "false_fire_delta",
    "cost_usd_delta",
    "candidate_recall",
    "candidate_precision",
)


@dataclass(frozen=True)
class Bar:
    """One client-declared quality bar: ``metric`` must be ``>= value`` (min) or ``<= value`` (max).

    ``require_ci_excludes_zero`` (delta metrics only) additionally demands that the paired
    confidence interval on the delta exclude zero — i.e. the improvement is not merely a point
    estimate. It is opt-in because it is the client's call, not ours.
    """

    metric: str
    direction: str  # "min" | "max"
    value: float
    require_ci_excludes_zero: bool = False

    def __post_init__(self) -> None:
        if self.metric not in BAR_METRICS:
            raise EvalInputError(
                f"unknown bar metric {self.metric!r}; known: {', '.join(BAR_METRICS)}"
            )
        if self.direction not in ("min", "max"):
            raise EvalInputError(f"bar direction must be 'min' or 'max', got {self.direction!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "direction": self.direction,
            "value": self.value,
            "require_ci_excludes_zero": self.require_ci_excludes_zero,
        }


@dataclass(frozen=True)
class BarsDocument:
    """The client's bars file: where it came from, and when it was declared."""

    source: str
    bars: tuple[Bar, ...] = ()
    declared_by: str = ""
    declared_at: str = ""

    @property
    def declared(self) -> bool:
        return bool(self.bars)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "declared_by": self.declared_by,
            "declared_at": self.declared_at,
            "bars": [b.to_dict() for b in self.bars],
        }


@dataclass(frozen=True)
class BarResult:
    """One bar evaluated against the run. ``observed`` is ``None`` when the metric was not measurable."""

    bar: Bar
    observed: float | None
    passed: bool | None  # None = not assessable (metric unmeasurable)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "bar": self.bar.to_dict(),
            "observed": self.observed,
            "passed": self.passed,
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# 3. Arms, fixtures, outcomes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmSpec:
    """One side of the pair: its plan identity, its adapter/model identities, its thresholds."""

    name: str
    plan_path: str
    plan: PlanIdentity
    components: tuple[dict[str, Any], ...] = ()
    thresholds: Mapping[str, Any] = field(default_factory=dict)
    fixture_dir: str = ""

    def __post_init__(self) -> None:
        if not self.fixture_dir:
            object.__setattr__(self, "fixture_dir", self.name)

    @property
    def plan_hash(self) -> str | None:
        return self.plan.plan_hash

    def identity_payload(self) -> dict[str, Any]:
        """The part of an arm that MUST NOT change after bars are declared."""
        return {
            "name": self.name,
            "plan": self.plan.to_dict(),
            "components": [dict(sorted(c.items())) for c in self.components],
            "thresholds": dict(sorted(self.thresholds.items())),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["plan_path"] = self.plan_path
        payload["fixture_dir"] = self.fixture_dir
        return payload


@dataclass(frozen=True)
class Fixture:
    """One labeled fixture: its episodes, its declared goal shape, and its scope assessment."""

    fixture_id: str
    goal_shape: str
    episodes: tuple[GroundTruthEpisode, ...]
    scope: ScopeAssessment
    path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "goal_shape": self.goal_shape,
            "episodes": len(self.episodes),
            "scope": self.scope.to_dict(),
            "path": self.path,
        }


@dataclass(frozen=True)
class ArmOutcome:
    """One arm's result on one fixture: one-to-one event-level matching scoring plus (when traced) the funnel."""

    arm: str
    result: GoldenEvalResult
    summary: TraceSummary | None = None

    @property
    def cost_usd(self) -> float | None:
        return None if self.summary is None else self.summary.cost_usd

    @property
    def input_tokens(self) -> float | None:
        return None if self.summary is None else self.summary.input_tokens

    @property
    def judge_calls(self) -> int | None:
        return None if self.summary is None else self.summary.judge_completed

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "recall": self.result.recall,
            "precision": self.result.precision,
            "true_positives": self.result.true_positives,
            "misses": self.result.misses,
            "false_fires": self.result.false_fires,
            "duplicate_false_fires": self.result.duplicate_false_fires,
            "unmatched_false_fires": self.result.unmatched_false_fires,
            "missed_episode_ids": list(self.result.missed_episode_ids),
            "funnel": None if self.summary is None else self.summary.to_dict(),
        }


@dataclass(frozen=True)
class PairedFixtureResult:
    """Both arms on ONE fixture — the unit of the paired comparison."""

    fixture: Fixture
    baseline: ArmOutcome
    candidate: ArmOutcome
    in_held_out: bool

    @property
    def scored(self) -> bool:
        """Contributes to deltas/bars only if the shape is scorable AND it is in the held-out split."""
        return self.fixture.scope.scorable and self.in_held_out

    def delta(self, metric: str) -> float | None:
        b, c = _fixture_metric(self.baseline, metric), _fixture_metric(self.candidate, metric)
        if b is None or c is None:
            return None
        return c - b

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture": self.fixture.to_dict(),
            "in_held_out": self.in_held_out,
            "scored": self.scored,
            "baseline": self.baseline.to_dict(),
            "candidate": self.candidate.to_dict(),
            "deltas": {
                "recall": self.delta("recall"),
                "precision": self.delta("precision"),
                "false_fires": self.delta("false_fires"),
                "cost_usd": self.delta("cost_usd"),
                "input_tokens": self.delta("input_tokens"),
                "judge_calls": self.delta("judge_calls"),
            },
        }


def _fixture_metric(outcome: ArmOutcome, metric: str) -> float | None:
    if metric == "recall":
        return outcome.result.recall
    if metric == "precision":
        return outcome.result.precision
    if metric == "false_fires":
        return float(outcome.result.false_fires)
    if metric == "misses":
        return float(outcome.result.misses)
    if metric == "cost_usd":
        return outcome.cost_usd
    if metric == "input_tokens":
        return outcome.input_tokens
    if metric == "judge_calls":
        return None if outcome.judge_calls is None else float(outcome.judge_calls)
    raise EvalInputError(f"unknown fixture metric {metric!r}")


# ---------------------------------------------------------------------------
# 4. Paired statistics
# ---------------------------------------------------------------------------

#: Bootstrap resamples for the CI on a paired delta. Fixed (not a knob) so two runs over the same
#: inputs produce the same interval; the RNG is seeded from the registration hash, never from time.
BOOTSTRAP_RESAMPLES = 2000
CI_LEVEL = 0.95


@dataclass(frozen=True)
class Delta:
    """A paired delta with its confidence interval and the reason if there isn't one."""

    metric: str
    n_pairs: int
    baseline_mean: float | None
    candidate_mean: float | None
    mean_delta: float | None
    ci_low: float | None
    ci_high: float | None
    ci_level: float = CI_LEVEL
    method: str = "paired percentile bootstrap"
    note: str = ""
    per_fixture: tuple[tuple[str, float], ...] = ()

    @property
    def ci_excludes_zero(self) -> bool | None:
        if self.ci_low is None or self.ci_high is None:
            return None
        return self.ci_low > 0.0 or self.ci_high < 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "n_pairs": self.n_pairs,
            "baseline_mean": self.baseline_mean,
            "candidate_mean": self.candidate_mean,
            "mean_delta": self.mean_delta,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_level": self.ci_level,
            "ci_excludes_zero": self.ci_excludes_zero,
            "method": self.method,
            "note": self.note,
            "per_fixture": [{"fixture_id": f, "delta": d} for f, d in self.per_fixture],
        }


def paired_delta(
    pairs: Sequence[PairedFixtureResult], metric: str, *, seed: int
) -> Delta:
    """Mean paired delta for ``metric`` plus a deterministic percentile-bootstrap CI.

    PAIRED, not two independent samples: the statistic is the mean of the per-fixture
    ``candidate - baseline`` differences, and the bootstrap resamples FIXTURES (carrying both arms
    together), which is what makes the interval an interval on the difference.
    """
    usable = [(p.fixture.fixture_id, p.delta(metric)) for p in pairs]
    per_fixture = tuple((fid, d) for fid, d in usable if d is not None)
    dropped = len(usable) - len(per_fixture)
    n = len(per_fixture)

    base_vals = [_fixture_metric(p.baseline, metric) for p in pairs]
    cand_vals = [_fixture_metric(p.candidate, metric) for p in pairs]
    base_mean = _mean([v for v in base_vals if v is not None])
    cand_mean = _mean([v for v in cand_vals if v is not None])

    note = ""
    if dropped:
        note = (
            f"{dropped} fixture(s) contributed no {metric} delta (metric unmeasurable on at least "
            f"one arm — e.g. no trace artifact); they are excluded rather than imputed."
        )

    if n == 0:
        return Delta(metric, 0, base_mean, cand_mean, None, None, None,
                     note=note or f"no fixture yielded a {metric} delta", per_fixture=())

    deltas = [d for _, d in per_fixture]
    mean_delta = _mean(deltas)

    if n < 2:
        return Delta(
            metric, n, base_mean, cand_mean, mean_delta, None, None,
            note=(note + " " if note else "")
            + "no confidence interval from a single pair — one fixture is an anecdote, not an interval.",
            per_fixture=per_fixture,
        )

    rng = random.Random(seed)
    means = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(math.floor((1.0 - CI_LEVEL) / 2.0 * (len(means) - 1)))]
    hi = means[int(math.ceil((1.0 - (1.0 - CI_LEVEL) / 2.0) * (len(means) - 1)))]
    return Delta(metric, n, base_mean, cand_mean, mean_delta, lo, hi, note=note,
                 per_fixture=per_fixture)


def _mean(values: Sequence[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


# ---------------------------------------------------------------------------
# 5. Pre-registration — the anti-tune-to-green record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeldOutSplit:
    """The EXACT held-out split, and where it came from.

    ``declared`` is False when no ``splits.json`` was found. An undeclared split is not an error —
    the numbers are still reported — but bars are NOT applied to it, because a bar scored on a
    split chosen after the fact is precisely the thing pre-registration exists to prevent.
    """

    declared: bool
    source: str
    held_out: tuple[str, ...] = ()
    exploratory: tuple[str, ...] = ()
    split_name: str = "test"

    def to_dict(self) -> dict[str, Any]:
        return {
            "declared": self.declared,
            "source": self.source,
            "split_name": self.split_name,
            "held_out": list(self.held_out),
            "exploratory": list(self.exploratory),
        }


@dataclass(frozen=True)
class PreRegistration:
    """Everything that had to be fixed BEFORE the run, content-addressed.

    ``registration_hash`` covers both plan hashes, both arms' components and thresholds, the exact
    held-out split, and the bars. :func:`run_differential` recomputes it from the configuration it
    is actually handed; a mismatch against ``expected_registration_hash`` raises
    :class:`ConfigDriftError` and NOTHING is scored.
    """

    baseline: ArmSpec
    candidate: ArmSpec
    split: HeldOutSplit
    bars: BarsDocument
    declared_at: str = ""
    expected_registration_hash: str | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline.identity_payload(),
            "candidate": self.candidate.identity_payload(),
            "split": self.split.to_dict(),
            "bars": [b.to_dict() for b in self.bars.bars],
        }

    @property
    def registration_hash(self) -> str:
        return content_hash(self.payload())

    @property
    def seed(self) -> int:
        """Deterministic bootstrap seed derived from the registration — never from wall time."""
        return int(self.registration_hash[:12], 16)

    def assert_no_drift(self) -> None:
        if self.expected_registration_hash is None:
            return
        actual = self.registration_hash
        if actual != self.expected_registration_hash:
            raise ConfigDriftError(
                "REFUSING TO SCORE: the configuration changed after the bars were declared.\n"
                f"  pre-registered registration_hash: {self.expected_registration_hash}\n"
                f"  configuration presented now:      {actual}\n"
                "Plan hashes, component/model identities, thresholds, the held-out split and the "
                "bars are all covered by this hash. Re-declare the bars against the new "
                "configuration and re-run; do not adjust the configuration until it passes."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "declared_at": self.declared_at,
            "registration_hash": self.registration_hash,
            "expected_registration_hash": self.expected_registration_hash,
            "bootstrap_seed": self.seed,
            **self.payload(),
            "bars_source": self.bars.source,
            "bars_declared_by": self.bars.declared_by,
        }


# ---------------------------------------------------------------------------
# 6. The report
# ---------------------------------------------------------------------------


class OverallVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_ASSESSED = "NOT_ASSESSED"


@dataclass(frozen=True)
class DifferentialReport:
    """The whole answer: identities, per-fixture results, paired deltas + CIs, cost, scope, bars."""

    registration: PreRegistration
    pairs: tuple[PairedFixtureResult, ...]
    deltas: tuple[Delta, ...]
    bar_results: tuple[BarResult, ...]
    verdict: OverallVerdict
    verdict_reason: str
    scope_summary: tuple[ScopeAssessment, ...] = ()

    def delta(self, metric: str) -> Delta | None:
        return next((d for d in self.deltas if d.metric == metric), None)

    @property
    def scored_pairs(self) -> tuple[PairedFixtureResult, ...]:
        return tuple(p for p in self.pairs if p.scored)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "percept.eval.differential/1",
            "pre_registration": self.registration.to_dict(),
            "scope": [s.to_dict() for s in self.scope_summary],
            "fixtures": [p.to_dict() for p in self.pairs],
            "paired_deltas": [d.to_dict() for d in self.deltas],
            "bars": {
                "declared": self.registration.bars.declared,
                "document": self.registration.bars.to_dict(),
                "results": [r.to_dict() for r in self.bar_results],
            },
            "verdict": self.verdict.value,
            "verdict_reason": self.verdict_reason,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)


# ---------------------------------------------------------------------------
# 7. Loading
# ---------------------------------------------------------------------------


def _load_structured(path: str | Path) -> dict[str, Any]:
    """Load a ``.json`` (always) or ``.yaml``/``.yml`` (needs PyYAML) config document."""
    p = Path(path)
    if not p.exists():
        raise EvalInputError(f"file not found: {p}")
    text = p.read_text()
    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # noqa: PLC0415 - optional; percept core has zero runtime deps
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise EvalInputError(
                f"{p} is YAML but PyYAML is not installed. Install it (`pip install pyyaml`) or "
                f"pass the same document as JSON — percept core deliberately has no runtime deps."
            ) from exc
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text or "{}")
    if not isinstance(data, dict):
        raise EvalInputError(f"{p}: expected a mapping at the top level, got {type(data).__name__}")
    return data


class _PlanGoal:
    """Duck-typed goal for :meth:`PlanIdentity.from_goals` (id/condition/say/modality/state).

    ``state`` is carried because the per-goal dwell (``state['hold_s']``) is a resolved policy value
    the plan hash must cover — two arms whose only difference is a sustain window are two different
    configurations.
    """

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.id = str(payload.get("id", ""))
        self.condition = str(payload.get("condition", ""))
        self.say = str(payload.get("say", ""))
        self.modality = payload.get("modality")
        raw_state = payload.get("state")
        self.state: dict[str, Any] = dict(raw_state) if isinstance(raw_state, Mapping) else {}


def _arm_policy(data: Mapping[str, Any]) -> dict[str, Any]:
    """The arm's RESOLVED policy — everything under ``thresholds`` plus an optional ``policy`` block.

    Both are merged into one flat mapping for :meth:`PlanIdentity.from_goals`; ``policy`` wins on a
    collision because it is the more specific block. A key present in only one arm is still a
    difference (the other arm simply lacks it), so the two hashes diverge — which is the point.
    """
    merged: dict[str, Any] = {}
    merged.update(dict(data.get("thresholds") or {}))
    merged.update(dict(data.get("policy") or {}))
    return merged


def load_plan(path: str | Path, *, name: str, fixture_dir: str = "") -> ArmSpec:
    """Load one arm's plan file into an :class:`ArmSpec` with a content-addressed ``plan_hash``.

    The hash is computed by :meth:`percept.tracing.PlanIdentity.from_goals` — the SAME function the
    runtime uses to stamp a trace manifest, so an eval's quoted plan hash is comparable with a
    production run's.

    The arm's RESOLVED thresholds and policy go into the hash alongside the goals. The
    commonest paired eval on this harness is a threshold A/B — both arms share a goals list by
    construction — and until this, both arms printed the same ``plan_hash``, so the "quote these
    identities" block could not tell a reader which configuration produced which numbers. Key order in
    the plan file never matters (the hash is taken over sorted keys, exactly as
    :func:`percept.profile_versioning.content_hash` does for profiles).
    """
    data = _load_structured(path)
    goals = [_PlanGoal(g) for g in data.get("goals") or ()]
    if not goals:
        raise EvalInputError(f"{path}: plan declares no goals (expected a non-empty 'goals' list)")
    plan = PlanIdentity.from_goals(
        goals,
        revision=data.get("revision"),
        plan_mode=data.get("plan_mode"),
        policy=_arm_policy(data),
    )
    components = tuple(
        {
            "role": str(c.get("role", "")),
            "name": str(c.get("name", "")),
            "version": c.get("version"),
            "model": c.get("model"),
            "config_hash": c.get("config_hash"),
        }
        for c in data.get("components") or ()
    )
    thresholds = dict(data.get("thresholds") or {})
    return ArmSpec(
        name=name,
        plan_path=str(path),
        plan=plan,
        components=components,
        thresholds=thresholds,
        fixture_dir=fixture_dir or str(data.get("fixture_dir") or name),
    )


def load_bars(path: str | Path | None) -> BarsDocument:
    """Load the CLIENT-SUPPLIED bars file. ``None`` -> an empty, undeclared document."""
    if path is None:
        return BarsDocument(source="(none supplied)")
    data = _load_structured(path)
    raw = data.get("bars")
    if raw is None:
        raise EvalInputError(f"{path}: bars file has no 'bars' list")
    bars = tuple(
        Bar(
            metric=str(b["metric"]),
            direction=str(b.get("direction", "min")),
            value=float(b["value"]),
            require_ci_excludes_zero=bool(b.get("require_ci_excludes_zero", False)),
        )
        for b in raw
    )
    return BarsDocument(
        source=str(path),
        bars=bars,
        declared_by=str(data.get("declared_by", "")),
        declared_at=str(data.get("declared_at", "")),
    )


def _load_episodes(payload: Mapping[str, Any], fixture_id: str) -> tuple[GroundTruthEpisode, ...]:
    episodes = []
    for e in payload.get("episodes") or ():
        gt = GroundTruth(
            event_start_ts=float(e["event_start_ts"]),
            decision_eligible_ts=float(e["decision_eligible_ts"]),
            annotation_version=str(e["annotation_version"]),
            annotator_or_fixture_id=str(e["annotator_or_fixture_id"]),
        )
        episodes.append(
            GroundTruthEpisode(
                episode_id=str(e["episode_id"]),
                ground_truth=gt,
                episode_end_ts=float(e["episode_end_ts"]),
                allowed_lateness_s=float(e.get("allowed_lateness_s", 0.0)),
            )
        )
    if not episodes:
        raise EvalInputError(f"fixture {fixture_id!r}: ground_truth.json declares no episodes")
    return tuple(episodes)


def _load_events(path: Path) -> tuple[JudgmentEvent, ...]:
    if not path.exists():
        return ()
    data = json.loads(path.read_text() or "[]")
    if isinstance(data, dict):
        data = data.get("events") or []
    events = []
    for e in data:
        t = e.get("timing") or {}
        emitted = float(t["decision_emitted_ts"])
        timing = JudgmentTiming(
            ingest_observed_ts=float(t.get("ingest_observed_ts", emitted)),
            admitted_ts=float(t.get("admitted_ts", emitted)),
            judge_started_ts=float(t.get("judge_started_ts", emitted)),
            judge_completed_ts=float(t.get("judge_completed_ts", emitted)),
            decision_emitted_ts=emitted,
            source_media_ts=t.get("source_media_ts"),
        )
        events.append(
            JudgmentEvent(
                key=str(e.get("key", "")),
                evidence_seq=int(e.get("evidence_seq", 0)),
                action=str(e.get("action", "fire")),
                timing=timing,
            )
        )
    return tuple(events)


def _load_summary(path: Path) -> TraceSummary | None:
    if not path.exists():
        return None
    _manifest, _records, summary = load_trace(path.read_text())
    return summary


def load_fixtures(root: str | Path) -> tuple[Fixture, ...]:
    """Load every fixture under ``root`` (one directory per fixture, ``ground_truth.json`` inside)."""
    base = Path(root)
    if not base.is_dir():
        raise EvalInputError(f"fixtures root is not a directory: {base}")
    fixtures = []
    for child in sorted(base.iterdir()):
        gt_path = child / "ground_truth.json"
        if not child.is_dir() or not gt_path.exists():
            continue
        payload = json.loads(gt_path.read_text() or "{}")
        shape = str(payload.get("goal_shape", ""))
        fixtures.append(
            Fixture(
                fixture_id=child.name,
                goal_shape=shape,
                episodes=_load_episodes(payload, child.name),
                scope=scope_for_goal_shape(shape),
                path=str(child),
            )
        )
    if not fixtures:
        raise EvalInputError(
            f"no fixtures under {base} — expected <fixtures>/<fixture_id>/ground_truth.json"
        )
    return tuple(fixtures)


def load_held_out_split(
    root: str | Path, fixtures: Sequence[Fixture], *, split_name: str = "test"
) -> HeldOutSplit:
    """Read ``splits.json`` (via :func:`percept.eval.load_splits`, which enforces disjointness)."""
    base = Path(root)
    path = base / "splits.json"
    ids = {f.fixture_id for f in fixtures}
    if not path.exists():
        return HeldOutSplit(
            declared=False,
            source=f"(no {path} — split undeclared)",
            held_out=tuple(sorted(ids)),
            exploratory=(),
            split_name=split_name,
        )
    if split_name not in SPLIT_NAMES:
        raise EvalInputError(f"split_name must be one of {SPLIT_NAMES}, got {split_name!r}")
    splits = load_splits(path)  # raises SplitError if the splits overlap
    held = tuple(sorted(set(splits.get(split_name, ())) & ids))
    unknown = sorted(set(splits.get(split_name, ())) - ids)
    if unknown:
        raise EvalInputError(
            f"{path}: split {split_name!r} names fixtures that do not exist: {unknown}"
        )
    return HeldOutSplit(
        declared=True,
        source=str(path),
        held_out=held,
        exploratory=tuple(sorted(ids - set(held))),
        split_name=split_name,
    )


# ---------------------------------------------------------------------------
# 8. The run
# ---------------------------------------------------------------------------

#: The deltas always computed and reported (quality first, then cost).
REPORTED_DELTAS: tuple[str, ...] = (
    "recall",
    "precision",
    "false_fires",
    "cost_usd",
    "input_tokens",
    "judge_calls",
)


def _score_arm(fixture: Fixture, arm: ArmSpec) -> ArmOutcome:
    arm_dir = Path(fixture.path) / arm.fixture_dir
    events = _load_events(arm_dir / "events.json")
    summary = _load_summary(arm_dir / "trace.json")
    return ArmOutcome(arm=arm.name, result=score(fixture.episodes, events), summary=summary)


def run_differential(
    *,
    fixtures_root: str | Path,
    baseline: ArmSpec,
    candidate: ArmSpec,
    bars: BarsDocument,
    split_name: str = "test",
    expected_registration_hash: str | None = None,
    declared_at: str = "",
) -> DifferentialReport:
    """Run the paired differential and produce the report.

    Order matters: the pre-registration is assembled and drift-checked BEFORE anything is scored,
    so a drifted configuration produces an exception rather than a number someone might quote.
    """
    root = Path(fixtures_root)
    fixtures = load_fixtures(root)
    split = load_held_out_split(root, fixtures, split_name=split_name)

    registration = PreRegistration(
        baseline=baseline,
        candidate=candidate,
        split=split,
        bars=bars,
        declared_at=declared_at,
        expected_registration_hash=expected_registration_hash,
    )
    registration.assert_no_drift()  # refuse to score a drifted config

    held = set(split.held_out)
    pairs = tuple(
        PairedFixtureResult(
            fixture=f,
            baseline=_score_arm(f, baseline),
            candidate=_score_arm(f, candidate),
            in_held_out=f.fixture_id in held,
        )
        for f in fixtures
    )

    scored = tuple(p for p in pairs if p.scored)
    deltas = tuple(paired_delta(scored, m, seed=registration.seed) for m in REPORTED_DELTAS)

    bar_results, verdict, reason = _apply_bars(registration, deltas, scored)

    seen: dict[str, ScopeAssessment] = {}
    for f in fixtures:
        seen.setdefault(f.scope.goal_shape, f.scope)
    scope_summary = tuple(seen[k] for k in sorted(seen))

    return DifferentialReport(
        registration=registration,
        pairs=pairs,
        deltas=deltas,
        bar_results=bar_results,
        verdict=verdict,
        verdict_reason=reason,
        scope_summary=scope_summary,
    )


_BAR_SOURCES: dict[str, tuple[str, str]] = {
    # bar metric -> (kind, underlying metric); kind is "delta" or "candidate"
    "recall_delta": ("delta", "recall"),
    "precision_delta": ("delta", "precision"),
    "false_fire_delta": ("delta", "false_fires"),
    "cost_usd_delta": ("delta", "cost_usd"),
    "candidate_recall": ("candidate", "recall"),
    "candidate_precision": ("candidate", "precision"),
}


def evaluate_bars(
    bars: Sequence[Bar],
    deltas: Sequence[Delta],
    scored: Sequence[PairedFixtureResult],
) -> tuple[BarResult, ...]:
    """Evaluate client-supplied ``bars`` against one comparison's deltas + scored pairs.

    Split out of :func:`_apply_bars` so the profile-derivation job (:mod:`percept.eval.derivation`)
    applies bars through the SAME code path rather than a second, drifting implementation. It
    evaluates the bars it is handed and nothing else: it never invents a bar, and an unmeasurable
    metric yields ``passed=None`` (unassessed), never an optimistic pass.
    """
    by_metric = {d.metric: d for d in deltas}
    results: list[BarResult] = []
    for bar in bars:
        kind, metric = _BAR_SOURCES[bar.metric]
        note = ""
        if kind == "delta":
            d = by_metric.get(metric)
            observed = d.mean_delta if d else None
        else:
            vals = [v for v in (_fixture_metric(p.candidate, metric) for p in scored)
                    if v is not None]
            observed = _mean(vals)
            d = None
        if observed is None:
            results.append(
                BarResult(bar, None, None,
                          f"{bar.metric} was not measurable on this run (no arm reported it); "
                          f"reported as unassessed rather than assumed to pass.")
            )
            continue
        passed = observed >= bar.value if bar.direction == "min" else observed <= bar.value
        if bar.require_ci_excludes_zero:
            if d is None or d.ci_excludes_zero is None:
                passed = False
                note = "no confidence interval available, and this bar requires one to exclude zero"
            elif not d.ci_excludes_zero:
                passed = False
                note = f"CI [{d.ci_low:.4g}, {d.ci_high:.4g}] includes zero"
        results.append(BarResult(bar, observed, passed, note))
    return tuple(results)


def _apply_bars(
    registration: PreRegistration,
    deltas: Sequence[Delta],
    scored: Sequence[PairedFixtureResult],
) -> tuple[tuple[BarResult, ...], OverallVerdict, str]:
    """Apply the client's bars — or explicitly decline to, stating why. NEVER invents a bar."""
    if not registration.bars.declared:
        return (
            (),
            OverallVerdict.NOT_ASSESSED,
            "No bars were supplied. Results are reported; percept does not invent a pass/fail "
            "threshold on a client's behalf. Supply --bars to get a verdict.",
        )
    if not registration.split.declared:
        return (
            (),
            OverallVerdict.NOT_ASSESSED,
            "Bars were supplied but the held-out split is UNDECLARED (no splits.json under the "
            "fixtures root). Scoring bars against a split chosen after the run is tune-to-green; "
            "declare train/dev/test and re-run.",
        )
    if not scored:
        return (
            (),
            OverallVerdict.NOT_ASSESSED,
            "Bars were supplied but no fixture in the held-out split has a scorable goal shape "
            "(see the scope table). Nothing was scored, so nothing passed or failed.",
        )

    results = list(evaluate_bars(registration.bars.bars, deltas, scored))

    assessed = [r for r in results if r.passed is not None]
    failed = [r for r in results if r.passed is False]
    unassessed = [r for r in results if r.passed is None]
    if failed:
        return tuple(results), OverallVerdict.FAIL, (
            f"{len(failed)} of {len(results)} client bar(s) failed on the held-out split "
            f"({len(scored)} fixtures)."
        )
    if not assessed:
        return tuple(results), OverallVerdict.NOT_ASSESSED, (
            "No supplied bar was measurable on this run."
        )
    if unassessed:
        return tuple(results), OverallVerdict.NOT_ASSESSED, (
            f"{len(assessed)} bar(s) passed but {len(unassessed)} could not be measured; a partial "
            f"pass is not a PASS."
        )
    return tuple(results), OverallVerdict.PASS, (
        f"All {len(results)} client bar(s) passed on the held-out split ({len(scored)} fixtures)."
    )


# ---------------------------------------------------------------------------
# 9. Rendering
# ---------------------------------------------------------------------------


def _fmt(value: float | None, spec: str = ".4g") -> str:
    return "—" if value is None else format(value, spec)


def render_text(report: DifferentialReport) -> str:
    """The human report. Everything required is on this page, including the refusals."""
    reg = report.registration
    lines: list[str] = []
    add = lines.append

    add("percept eval — paired differential A/B")
    add("=" * 72)
    add("")
    add("IDENTITIES (quote these; two runs are comparable only if they match)")
    for arm in (reg.baseline, reg.candidate):
        add(f"  {arm.name}:")
        add(f"    plan file    : {arm.plan_path}")
        add(f"    plan_hash    : {arm.plan_hash}")
        add(f"    goal_ids     : {', '.join(arm.plan.goal_ids) or '—'}")
        for c in arm.components:
            add(f"    {c['role']:<9}: {c['name']} version={c.get('version') or '—'} "
                f"model={c.get('model') or '—'} config_hash={c.get('config_hash') or '—'}")
        add(f"    thresholds   : {json.dumps(dict(sorted(arm.thresholds.items())), sort_keys=True)}")
        # The resolved policy is IN the plan_hash, so print it right under the hash it moves.
        # A reader who sees two different hashes must be able to see WHICH knob differs without
        # opening the plan files.
        add(f"    policy       : {json.dumps(arm.plan.policy_dict, sort_keys=True)}")
    if reg.baseline.plan_hash == reg.candidate.plan_hash:
        add("  NOTE: both arms have the SAME plan_hash — goals, thresholds and policy are identical.")
        add("        Any difference between them is outside the plan identity; a delta here is noise,")
        add("        not a configuration effect.")
    add("")
    add("PRE-REGISTERED BEFORE THE RUN")
    add(f"  registration_hash : {reg.registration_hash}")
    add(f"  declared_at       : {reg.declared_at or '—'}")
    add(f"  bars              : {reg.bars.source}"
        f"{' (declared by ' + reg.bars.declared_by + ')' if reg.bars.declared_by else ''}")
    add(f"  split source      : {reg.split.source}")
    add(f"  held-out ({reg.split.split_name}) : "
        f"{', '.join(reg.split.held_out) or '—'}")
    add(f"  exploratory       : {', '.join(reg.split.exploratory) or '—'} "
        f"(reported, never scored against bars)")
    add(f"  bootstrap seed    : {reg.seed} (derived from registration_hash; not wall time)")
    add("")
    add("SCOPE — what this matcher can honestly validate (ADR-006)")
    for s in report.scope_summary:
        add(f"  {s.goal_shape:<26} {s.verdict.value}")
        add(f"      {s.reason}")
    add("")
    add("PER-FIXTURE (baseline -> candidate)")
    header = (f"  {'fixture':<22}{'shape':<14}{'recall':<18}{'prec':<18}{'ff':<12}{'cost usd':<20}")
    add(header)
    for p in report.pairs:
        flag = "" if p.scored else ("  [not scored: "
                                    + ("unsupported shape" if not p.fixture.scope.scorable
                                       else "not in held-out split") + "]")
        add(f"  {p.fixture.fixture_id:<22}{p.fixture.goal_shape:<14}"
            f"{_fmt(p.baseline.result.recall)+' -> '+_fmt(p.candidate.result.recall):<18}"
            f"{_fmt(p.baseline.result.precision)+' -> '+_fmt(p.candidate.result.precision):<18}"
            f"{str(p.baseline.result.false_fires)+' -> '+str(p.candidate.result.false_fires):<12}"
            f"{_fmt(p.baseline.cost_usd)+' -> '+_fmt(p.candidate.cost_usd):<20}{flag}")
    add("")
    add(f"PAIRED DELTAS (candidate - baseline, {int(CI_LEVEL*100)}% percentile bootstrap, "
        f"{BOOTSTRAP_RESAMPLES} resamples)")
    for d in report.deltas:
        ci = "—" if d.ci_low is None else f"[{_fmt(d.ci_low)}, {_fmt(d.ci_high)}]"
        add(f"  {d.metric:<14} n={d.n_pairs:<4} mean Δ={_fmt(d.mean_delta):<12} CI={ci:<26}"
            f"{'' if d.ci_excludes_zero is None else ('excludes 0' if d.ci_excludes_zero else 'includes 0')}")
        if d.note:
            add(f"      note: {d.note}")
    add("")
    add("BARS (client-supplied)")
    if not reg.bars.declared:
        add("  none supplied — results reported, no pass/fail invented.")
    else:
        for r in report.bar_results:
            state = "—" if r.passed is None else ("PASS" if r.passed else "FAIL")
            add(f"  {r.bar.metric:<20}{r.bar.direction} {r.bar.value:<10} "
                f"observed={_fmt(r.observed):<12}{state}")
            if r.note:
                add(f"      {r.note}")
        if not report.bar_results:
            add("  supplied, but not applied (see verdict).")
    add("")
    add(f"VERDICT: {report.verdict.value}")
    add(f"  {report.verdict_reason}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 10. CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="percept eval",
        description="Paired differential A/B of two plans over client fixtures.",
    )
    p.add_argument("--baseline", required=True, help="baseline plan file (.yaml/.json)")
    p.add_argument("--candidate", required=True, help="candidate plan file (.yaml/.json)")
    p.add_argument("--fixtures", required=True, help="fixtures root directory")
    p.add_argument("--bars", default=None, help="CLIENT-SUPPLIED bars file; omit to report only")
    p.add_argument("--split", default="test", choices=list(SPLIT_NAMES),
                   help="which split is held out (default: test)")
    p.add_argument("--expect-registration-hash", default=None,
                   help="fail (refuse to score) unless the configuration hashes to this")
    p.add_argument("--json", dest="json_out", default=None, help="write the full report as JSON")
    p.add_argument("--declared-at", default="", help="when the bars/split were declared")
    # --- operating frontier (§3.2) — the SAME entry point, not a second CLI -------------
    p.add_argument("--frontier", action="store_true",
                   help="also emit the operating-frontier report (§3.2): exact one-sided 95% "
                        "bounds on FIRE recall and false FIREs per scheduled watch-hour, ASK rate, "
                        "coverage, assisted recall, degraded time, duplicate FIREs per episode, "
                        "per-archetype, both denominators")
    p.add_argument("--frontier-arm", default="candidate",
                   help="which arm the frontier is computed on (default: candidate)")
    p.add_argument("--frontier-bars", default=None,
                   help="operating bars file: false_fire_max (per scheduled watch-hour) and/or "
                        "recall_min; omit to report bounds without a verdict")
    p.add_argument("--frontier-json", default=None,
                   help="write the operating-frontier report as JSON")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """``percept-eval`` entrypoint. 0 = pass/reported, 1 = a client bar failed, 2 = refused."""
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        report = run_differential(
            fixtures_root=args.fixtures,
            baseline=load_plan(args.baseline, name="baseline"),
            candidate=load_plan(args.candidate, name="candidate"),
            bars=load_bars(args.bars),
            split_name=args.split,
            expected_registration_hash=args.expect_registration_hash,
            declared_at=args.declared_at,
        )
    except (EvalInputError, SplitError) as exc:
        print(f"percept eval: REFUSED — {exc}", file=sys.stderr)
        return 2
    print(render_text(report))
    if args.json_out:
        Path(args.json_out).write_text(report.to_json())
    failed = report.verdict is OverallVerdict.FAIL

    if args.frontier:
        from . import frontier as _frontier  # local import: frontier imports this module

        try:
            evidence = _frontier.load_frontier_evidence(
                args.fixtures, arm=args.frontier_arm, split_name=args.split
            )
            frontier_report = _frontier.build_frontier(
                evidence, bars=_frontier.load_bars(args.frontier_bars)
            )
        except (EvalInputError, SplitError, _frontier.FrontierInputError) as exc:
            print(f"percept eval --frontier: REFUSED — {exc}", file=sys.stderr)
            return 2
        print()
        print(_frontier.render_text(frontier_report))
        if args.frontier_json:
            Path(args.frontier_json).write_text(frontier_report.to_json())
        failed = failed or frontier_report.verdict is _frontier.FrontierVerdict.FAIL

    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
