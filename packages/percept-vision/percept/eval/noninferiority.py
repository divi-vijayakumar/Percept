"""Non-inferiority verdicts for paired A/B deltas — the correction to "the CI includes zero".

WHY THIS MODULE EXISTS (experiment-design review, blocker on E-1..E-4):

    A 95% confidence interval that *includes zero* does NOT establish "no regression". A wide
    interval can straddle zero while still permitting a materially negative effect — you have simply
    failed to measure precisely enough to rule harm out. "CI includes zero" is an argument from
    ignorance dressed up as evidence of safety.

The honest test is **non-inferiority**, and it has two predeclared pieces the caller must commit to
*before* seeing data:

1. a tolerable margin ``delta > 0`` — the largest regression you are willing to call "not worse";
2. a power target — enough samples that, if the true effect really were at the margin, you would
   detect it.

The verdict is then:

* ``NON_INFERIOR`` — the one-sided lower confidence bound on the effect is **strictly above
  ``-delta``**. Not "the interval touches zero"; the whole plausible range of harm is inside the
  tolerated margin.
* ``INFERIOR`` — the lower bound sits at or below ``-delta``: a regression as large as the margin is
  not ruled out.
* ``UNDERPOWERED`` — **the sample is too small to detect ``delta`` at all**, so NO verdict is
  issued. This is the piece the review insisted on: with too few samples, a bound that happens to
  clear ``-delta`` is luck, not evidence, and a bound that doesn't is un-interpretable. Refusing is
  the correct, honest output — it tells the client *how many* samples the claim would need.

The lower bound reuses the same deterministic paired percentile bootstrap the differential report
uses (:mod:`percept.eval.differential`), seeded so two runs over identical inputs agree to the bit.
The power/sample-size gate uses a pure-Python normal quantile (Acklam's inverse-CDF rational
approximation) so the module keeps ``percept`` core's zero-runtime-dependency promise.

This module MEASURES; it never spends API budget. E-1..E-4 feed it their per-fixture effects.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

__all__ = [
    "NIVerdict",
    "NonInferiorityResult",
    "norm_ppf",
    "required_n_for_margin",
    "assess_non_inferiority",
    "DEFAULT_ALPHA",
    "DEFAULT_POWER",
    "NI_BOOTSTRAP_RESAMPLES",
]

#: One-sided type-I error for the lower confidence bound (95% one-sided).
DEFAULT_ALPHA: float = 0.05
#: Target power — the probability of correctly declaring non-inferiority when the true effect is 0.
DEFAULT_POWER: float = 0.80
#: Fixed so the bound is reproducible; the RNG is seeded from the caller, never from the wall clock.
NI_BOOTSTRAP_RESAMPLES = 4000


class NIVerdict(str, Enum):
    """The three honest outcomes. ``UNDERPOWERED`` is a REFUSAL, not a soft pass."""

    NON_INFERIOR = "non_inferior"
    INFERIOR = "inferior"
    UNDERPOWERED = "underpowered"


# ---------------------------------------------------------------------------
# Normal quantile — pure Python (Acklam), so no SciPy dependency.
# ---------------------------------------------------------------------------

# Coefficients for Peter Acklam's inverse normal CDF rational approximation.
_A = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
      1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
_B = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
      6.680131188771972e01, -1.328068155288572e01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
      -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)
_P_LOW = 0.02425
_P_HIGH = 1.0 - _P_LOW


def norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF ``Phi^-1(p)`` for ``0 < p < 1`` (Acklam, ~1e-9 accurate).

    ``norm_ppf(0.975) ~= 1.959964``, ``norm_ppf(0.95) ~= 1.644854``, ``norm_ppf(0.80) ~= 0.841621``.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"norm_ppf needs 0 < p < 1, got {p!r}")
    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        )
    if p <= _P_HIGH:
        q = p - 0.5
        r = q * q
        return (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / (
            ((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
        (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
    )


def required_n_for_margin(
    margin: float,
    sigma: float,
    *,
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
) -> int:
    """Smallest paired-sample size that can DETECT a regression of size ``margin``.

    One-sided sample size for a mean under a non-inferiority margin, at the null-favourable point
    where the true effect is exactly 0::

        n = ceil( ( (z_{1-alpha} + z_{power}) * sigma / margin )^2 )

    ``sigma`` is the standard deviation of the per-fixture *effects* (paired differences). A
    ``sigma`` of 0 means the effect is measured without noise, so ``n = 2`` (the floor below which
    there is no interval at all) suffices. ``margin`` must be strictly positive — a margin of zero
    demands infinite samples and is a specification error, not a large number.
    """
    if margin <= 0.0:
        raise ValueError(f"non-inferiority margin must be > 0, got {margin!r}")
    if sigma < 0.0:
        raise ValueError(f"sigma must be >= 0, got {sigma!r}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0, 1), got {power!r}")
    if sigma == 0.0:
        return 2
    z = norm_ppf(1.0 - alpha) + norm_ppf(power)
    n = (z * sigma / margin) ** 2
    return max(2, int(math.ceil(n)))


@dataclass(frozen=True)
class NonInferiorityResult:
    """A non-inferiority assessment of one metric's paired effect. Serialises for the report."""

    metric: str
    margin: float
    higher_is_better: bool
    n: int
    mean_effect: float | None
    lower_bound: float | None
    sigma: float | None
    required_n: int | None
    alpha: float
    power: float
    verdict: NIVerdict
    note: str = ""
    method: str = "one-sided paired percentile bootstrap"

    @property
    def refused(self) -> bool:
        """True when the sample was too small to detect the margin — no verdict was issued."""
        return self.verdict is NIVerdict.UNDERPOWERED

    @property
    def passed(self) -> bool | None:
        """``True``/``False`` for a real verdict; ``None`` when the assessment was refused."""
        if self.refused:
            return None
        return self.verdict is NIVerdict.NON_INFERIOR

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "margin": self.margin,
            "higher_is_better": self.higher_is_better,
            "n": self.n,
            "mean_effect": self.mean_effect,
            "lower_bound": self.lower_bound,
            "sigma": self.sigma,
            "required_n": self.required_n,
            "alpha": self.alpha,
            "power": self.power,
            "verdict": self.verdict.value,
            "refused": self.refused,
            "passed": self.passed,
            "note": self.note,
            "method": self.method,
        }


def _stdev(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def _bootstrap_lower_bound(effects: Sequence[float], *, alpha: float, seed: int, resamples: int) -> float:
    """One-sided lower confidence bound on the mean via a deterministic percentile bootstrap."""
    n = len(effects)
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        means.append(sum(effects[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    idx = int(math.floor(alpha * (len(means) - 1)))
    return means[idx]


def assess_non_inferiority(
    effects: Sequence[float],
    *,
    margin: float,
    higher_is_better: bool = True,
    metric: str = "effect",
    alpha: float = DEFAULT_ALPHA,
    power: float = DEFAULT_POWER,
    sigma: float | None = None,
    seed: int,
    resamples: int = NI_BOOTSTRAP_RESAMPLES,
) -> NonInferiorityResult:
    """Assess whether a candidate is non-inferior to a baseline within a predeclared ``margin``.

    ``effects`` are the per-fixture PAIRED differences (candidate - baseline). ``margin > 0`` is the
    predeclared tolerable regression, in the metric's own units. ``higher_is_better`` orients the
    metric: for recall it is ``True`` (a drop is a regression); for cost/latency/false-fire rate pass
    ``False`` (a *rise* is the regression) and the effects are flipped internally so the test always
    asks the same question — "is the lower bound on the improvement above ``-margin``?".

    The power gate runs FIRST and can only be cleared, never bought back: if ``n`` is below the
    number of samples needed to detect ``margin`` (:func:`required_n_for_margin`, using ``sigma`` —
    the caller's predeclared value if given, else the sample estimate), the result is
    ``UNDERPOWERED`` and carries ``required_n`` so the caller knows the shortfall. Only a
    sufficiently-powered sample earns a ``NON_INFERIOR`` / ``INFERIOR`` verdict.
    """
    if margin <= 0.0:
        raise ValueError(f"non-inferiority margin must be predeclared > 0, got {margin!r}")

    oriented = [float(e) if higher_is_better else -float(e) for e in effects]
    n = len(oriented)

    if n == 0:
        return NonInferiorityResult(
            metric, margin, higher_is_better, 0, None, None, None, None, alpha, power,
            NIVerdict.UNDERPOWERED, note="no paired effects — nothing to assess.",
        )

    mean_effect = sum(oriented) / n
    est_sigma = _stdev(oriented) if sigma is None else float(sigma)
    required_n = required_n_for_margin(margin, est_sigma, alpha=alpha, power=power)

    # Power gate first: too few samples => refuse, regardless of where a bound would land.
    if n < required_n or n < 2:
        need = max(required_n, 2)
        return NonInferiorityResult(
            metric, margin, higher_is_better, n, mean_effect, None, est_sigma, need, alpha, power,
            NIVerdict.UNDERPOWERED,
            note=(
                f"underpowered: n={n} < {need} required to detect a margin of {margin:g} at "
                f"sigma={est_sigma:g}, alpha={alpha:g}, power={power:g}. No verdict — a bound from "
                f"this few samples is not evidence of non-inferiority. Collect {need - n} more pair(s)."
            ),
        )

    lower = _bootstrap_lower_bound(oriented, alpha=alpha, seed=seed, resamples=resamples)
    if lower > -margin:
        verdict = NIVerdict.NON_INFERIOR
        note = (
            f"lower bound {lower:.4g} > -margin ({-margin:g}): a regression of {margin:g} or more is "
            f"ruled out at the {1 - alpha:.0%} one-sided level."
        )
    else:
        verdict = NIVerdict.INFERIOR
        note = (
            f"lower bound {lower:.4g} <= -margin ({-margin:g}): a regression as large as the margin is "
            f"NOT ruled out. This is not non-inferior, whatever the point estimate suggests."
        )
    return NonInferiorityResult(
        metric, margin, higher_is_better, n, mean_effect, lower, est_sigma, required_n, alpha, power,
        verdict, note=note,
    )
