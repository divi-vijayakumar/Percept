"""The E-1 metric set for the subject-identity instruction — with the false-bind asymmetry encoded.

E-1 measures what adding an identity instruction to the judge does. The review rejected a naive
"accuracy went up" read and pinned a stronger, safety-weighted metric set:

* **exact-key accuracy when legible** — of the sightings whose key was actually readable, how often
  the predicted key matched the truth;
* **false-bound rate** — how often the judge bound to a key it should NOT have: the wrong key on a
  legible sighting, or a fabricated key on an illegible one;
* **unknown rate** — how often it abstained (``binding = unknown``);
* **normalized-key collision rate** — how often normalisation merged two GENUINELY distinct subjects
  into one key (a measurement hazard, not a win);
* **cross-run consistency** — how often repeated runs of the same input produced the same binding;
* **judgment-quality delta** — the change in downstream judgment quality caused by the instruction
  (fed through :mod:`percept.eval.noninferiority`, because a quality *regression* is the real risk);
* **token / latency delta** — the cost the instruction adds.

THE LOAD-BEARING ASYMMETRY: **a false bind is far more dangerous than an abstention.** Binding an
invoice to the wrong subject silently corrupts everything downstream; abstaining (``unknown``) is
loud and safe. So the acceptance criterion must NOT trade them one-for-one. Two mechanisms encode
that here, and both are tested:

1. a **hard false-bind ceiling** — no amount of accuracy buys past it; and
2. a **weighted harm score** where a false bind costs many times what an abstention costs.

Swapping one abstention for one false bind must therefore strictly worsen the verdict, never improve
it. That property is the essential metric invariant, and :mod:`tests` asserts it directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .noninferiority import NIVerdict, NonInferiorityResult

__all__ = [
    "BindOutcome",
    "classify_binding",
    "E1Metrics",
    "compute_e1_metrics",
    "normalized_key_collision_rate",
    "cross_run_consistency",
    "AcceptancePolicy",
    "AcceptanceResult",
    "assess_acceptance",
]


class BindOutcome(Enum):
    """How one identity decision landed against ground truth.

    The two ``*_false_bind`` outcomes are the dangerous ones; the two ``abstained_*`` are safe.
    """

    EXACT = "exact"                          # legible, predicted the correct key
    WRONG_KEY_FALSE_BIND = "wrong_key"       # legible, predicted a DIFFERENT key  (dangerous)
    FABRICATED_FALSE_BIND = "fabricated"     # illegible, predicted a key at all    (dangerous)
    ABSTAINED_WHEN_LEGIBLE = "abstained_legible"    # legible, said unknown          (safe miss)
    ABSTAINED_WHEN_ILLEGIBLE = "abstained_illegible"  # illegible, said unknown      (correct)

    @property
    def is_false_bind(self) -> bool:
        return self in (BindOutcome.WRONG_KEY_FALSE_BIND, BindOutcome.FABRICATED_FALSE_BIND)

    @property
    def is_abstention(self) -> bool:
        return self in (BindOutcome.ABSTAINED_WHEN_LEGIBLE, BindOutcome.ABSTAINED_WHEN_ILLEGIBLE)


def classify_binding(*, legible: bool, true_key: str | None, predicted_key: str | None) -> BindOutcome:
    """Classify one identity decision. ``predicted_key is None`` means the judge abstained (unknown).

    ``legible`` is ground truth: whether the key was actually readable. When legible, ``true_key`` is
    the correct answer. The comparison is on the RAW key here — normalisation collisions are measured
    separately by :func:`normalized_key_collision_rate`, so a wrong bind is not excused by a lenient
    canonicaliser.
    """
    if predicted_key is None:
        return BindOutcome.ABSTAINED_WHEN_ILLEGIBLE if not legible else BindOutcome.ABSTAINED_WHEN_LEGIBLE
    if not legible:
        return BindOutcome.FABRICATED_FALSE_BIND
    if true_key is not None and predicted_key == true_key:
        return BindOutcome.EXACT
    return BindOutcome.WRONG_KEY_FALSE_BIND


def normalized_key_collision_rate(
    bound_pairs: Sequence[tuple[str, str]],
    *,
    normalize,
) -> float | None:
    """Fraction of bound sightings whose NORMALISED predicted key collides across distinct truths.

    ``bound_pairs`` is ``(true_key, predicted_key)`` for the sightings that actually bound. A
    collision is a normalised predicted key shared by more than one distinct ``true_key`` — i.e.
    normalisation has merged subjects that should stay apart. Returns ``None`` when nothing bound.
    """
    if not bound_pairs:
        return None
    buckets: dict[str, set[str]] = {}
    for true_key, pred in bound_pairs:
        buckets.setdefault(normalize(pred), set()).add(true_key)
    colliding = sum(1 for _, pred in bound_pairs if len(buckets[normalize(pred)]) > 1)
    return colliding / len(bound_pairs)


def cross_run_consistency(runs: Sequence[Sequence[object]]) -> float | None:
    """Fraction of positions where every run agreed. ``runs`` are per-run sequences of equal length.

    Each inner sequence is one run's per-sighting predicted binding (any hashable — a key string, or
    a sentinel for unknown). Returns ``None`` for fewer than two runs (consistency is undefined with
    one observation of the process).
    """
    if len(runs) < 2:
        return None
    lengths = {len(r) for r in runs}
    if len(lengths) != 1:
        raise ValueError(f"cross_run_consistency needs equal-length runs, got lengths {sorted(lengths)}")
    n = lengths.pop()
    if n == 0:
        return None
    agree = sum(1 for i in range(n) if len({r[i] for r in runs}) == 1)
    return agree / n


@dataclass(frozen=True)
class E1Metrics:
    """The full E-1 metric set for one arm. Rates are ``None`` when their denominator is empty."""

    n: int
    legible_n: int
    exact_hits: int
    false_binds: int
    abstentions: int
    exact_key_accuracy_when_legible: float | None
    false_bound_rate: float
    unknown_rate: float
    normalized_key_collision_rate: float | None = None
    cross_run_consistency: float | None = None
    judgment_quality: NonInferiorityResult | None = None
    token_delta: float | None = None
    latency_delta_s: float | None = None

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "legible_n": self.legible_n,
            "exact_hits": self.exact_hits,
            "false_binds": self.false_binds,
            "abstentions": self.abstentions,
            "exact_key_accuracy_when_legible": self.exact_key_accuracy_when_legible,
            "false_bound_rate": self.false_bound_rate,
            "unknown_rate": self.unknown_rate,
            "normalized_key_collision_rate": self.normalized_key_collision_rate,
            "cross_run_consistency": self.cross_run_consistency,
            "judgment_quality": None if self.judgment_quality is None else self.judgment_quality.to_dict(),
            "token_delta": self.token_delta,
            "latency_delta_s": self.latency_delta_s,
        }


def compute_e1_metrics(
    outcomes: Sequence[BindOutcome],
    *,
    normalized_collision_rate: float | None = None,
    cross_run: float | None = None,
    judgment_quality: NonInferiorityResult | None = None,
    token_delta: float | None = None,
    latency_delta_s: float | None = None,
) -> E1Metrics:
    """Aggregate per-sighting :class:`BindOutcome`s into the E-1 metric set."""
    n = len(outcomes)
    legible_n = sum(
        1 for o in outcomes
        if o in (BindOutcome.EXACT, BindOutcome.WRONG_KEY_FALSE_BIND, BindOutcome.ABSTAINED_WHEN_LEGIBLE)
    )
    exact_hits = sum(1 for o in outcomes if o is BindOutcome.EXACT)
    false_binds = sum(1 for o in outcomes if o.is_false_bind)
    abstentions = sum(1 for o in outcomes if o.is_abstention)
    return E1Metrics(
        n=n,
        legible_n=legible_n,
        exact_hits=exact_hits,
        false_binds=false_binds,
        abstentions=abstentions,
        exact_key_accuracy_when_legible=(exact_hits / legible_n) if legible_n else None,
        false_bound_rate=(false_binds / n) if n else 0.0,
        unknown_rate=(abstentions / n) if n else 0.0,
        normalized_key_collision_rate=normalized_collision_rate,
        cross_run_consistency=cross_run,
        judgment_quality=judgment_quality,
        token_delta=token_delta,
        latency_delta_s=latency_delta_s,
    )


@dataclass(frozen=True)
class AcceptancePolicy:
    """Predeclared acceptance thresholds for E-1, encoding the false-bind > abstention asymmetry.

    ``false_bind_ceiling`` is the HARD cap: a rate above it fails outright, no matter how high exact
    accuracy climbs. ``false_bind_weight`` >> ``unknown_weight`` makes the ``harm_score`` punish a
    false bind many times harder than an abstention, so trading an abstention for a false bind always
    worsens the score. ``require_quality_non_inferior`` makes a refused or inferior judgment-quality
    verdict block acceptance (a silent quality regression is not acceptable collateral).
    """

    false_bind_ceiling: float = 0.02
    false_bind_weight: float = 25.0
    unknown_weight: float = 1.0
    harm_budget: float = 0.10
    min_exact_accuracy_when_legible: float = 0.90
    require_quality_non_inferior: bool = True
    max_normalized_collision_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.false_bind_weight <= self.unknown_weight:
            raise ValueError(
                "false_bind_weight must exceed unknown_weight — a false bind is more dangerous than an "
                f"abstention (got {self.false_bind_weight} <= {self.unknown_weight})."
            )
        for name, v in (("false_bind_ceiling", self.false_bind_ceiling), ("harm_budget", self.harm_budget)):
            if v < 0.0:
                raise ValueError(f"{name} must be >= 0, got {v!r}")

    def harm_score(self, metrics: E1Metrics) -> float:
        """Asymmetric harm: ``false_bind_weight * false_bound_rate + unknown_weight * unknown_rate``."""
        return (
            self.false_bind_weight * metrics.false_bound_rate
            + self.unknown_weight * metrics.unknown_rate
        )


@dataclass(frozen=True)
class AcceptanceResult:
    accepted: bool
    harm_score: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"accepted": self.accepted, "harm_score": self.harm_score, "reasons": list(self.reasons)}


def assess_acceptance(metrics: E1Metrics, policy: AcceptancePolicy | None = None) -> AcceptanceResult:
    """Apply the asymmetric acceptance criterion. Every failing gate is reported, not just the first."""
    policy = policy or AcceptancePolicy()
    harm = policy.harm_score(metrics)
    reasons: list[str] = []

    # 1. Hard false-bind ceiling — cannot be bought back by accuracy.
    if metrics.false_bound_rate > policy.false_bind_ceiling:
        reasons.append(
            f"false_bound_rate {metrics.false_bound_rate:.4g} exceeds the hard ceiling "
            f"{policy.false_bind_ceiling:.4g} — a false bind corrupts downstream state silently, so no "
            f"accuracy gain buys past this."
        )
    # 2. Weighted harm budget (asymmetric: a false bind costs far more than an abstention).
    if harm > policy.harm_budget:
        reasons.append(
            f"harm_score {harm:.4g} exceeds budget {policy.harm_budget:.4g} "
            f"(false binds weighted {policy.false_bind_weight:g}x an abstention)."
        )
    # 3. Exact accuracy floor, when there were legible sightings to be accurate on.
    if metrics.exact_key_accuracy_when_legible is not None and (
        metrics.exact_key_accuracy_when_legible < policy.min_exact_accuracy_when_legible
    ):
        reasons.append(
            f"exact_key_accuracy_when_legible {metrics.exact_key_accuracy_when_legible:.4g} "
            f"below floor {policy.min_exact_accuracy_when_legible:.4g}."
        )
    # 4. Normalisation must not silently merge distinct subjects beyond the tolerance.
    if metrics.normalized_key_collision_rate is not None and (
        metrics.normalized_key_collision_rate > policy.max_normalized_collision_rate
    ):
        reasons.append(
            f"normalized_key_collision_rate {metrics.normalized_key_collision_rate:.4g} exceeds "
            f"{policy.max_normalized_collision_rate:.4g} — normalisation is merging distinct subjects."
        )
    # 5. Judgment quality must be established non-inferior (a refusal or inferiority blocks).
    if policy.require_quality_non_inferior:
        q = metrics.judgment_quality
        if q is None:
            reasons.append("judgment-quality non-inferiority was required but not supplied.")
        elif q.verdict is NIVerdict.UNDERPOWERED:
            reasons.append(f"judgment-quality assessment was refused (underpowered): {q.note}")
        elif q.verdict is NIVerdict.INFERIOR:
            reasons.append(f"judgment quality is inferior: {q.note}")

    return AcceptanceResult(accepted=not reasons, harm_score=harm, reasons=tuple(reasons))
