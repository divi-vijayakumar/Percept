"""Exact one-sided 95% confidence bounds — Poisson (Garwood) and binomial (Clopper–Pearson).

These are the ONLY interval methods the operating-frontier report (§3.2) is allowed to use. At the
event counts this report actually operates at — a handful of false FIREs over ~6 held-out hours, a
few dozen labeled positive episodes — a normal approximation is not "slightly conservative", it is
simply wrong (it can return a negative lower bound on a recall of 1.0, or a false-fire bound
tighter than the rule of three permits). So: no Wald, no Wilson, no ``1.96 * sqrt(p(1-p)/n)``.

Pure Python, no SciPy — ``percept`` core has zero runtime dependencies. The two special functions
are computed directly:

* **Poisson upper bound** (``docs/evaluation/sizing-a-run.md`` §1.1–1.2)::

      lambda_upper(k, T) = chi2_{0.95, 2(k+1)} / (2T) = factor(k) / T

  For even degrees of freedom ``2m`` the chi-square quantile has a closed-form CDF — the
  regularized upper incomplete gamma with integer shape ``m`` is a finite sum,
  ``Q(m, x) = e^-x * sum_{j<m} x^j / j!`` — so ``factor(k)`` is found by bisecting
  ``Q(k+1, x) = 0.05`` to machine tolerance. ``factor(0) = 2.9957`` reproduces the rule of three.

* **Clopper–Pearson lower bound** (§1.3)::

      recall_lower(x, n) = Beta^-1(0.05; x, n - x + 1)

  computed by bisecting the regularized incomplete beta ``I_t(a, b)`` (modified-Lentz continued
  fraction). For the all-hits case this collapses to ``0.05 ** (1/n)``, which is the property that
  makes a 10-for-10 run support only ``recall >= 0.741``.

Both inverse helpers (:func:`required_exposure`, :func:`required_positives`) answer the question the
honesty requirement forces whenever evidence is thin: *how many hours / episodes WOULD this claim
need?*
"""

from __future__ import annotations

import math

__all__ = [
    "DEFAULT_CONFIDENCE",
    "poisson_factor",
    "poisson_upper_bound",
    "required_exposure",
    "clopper_pearson_lower",
    "required_positives",
    "regularized_incomplete_beta",
]

#: One-sided confidence level for every bound in the §3.2 report.
DEFAULT_CONFIDENCE: float = 0.95

_TOL = 1e-12
_MAX_ITER = 400


# ---------------------------------------------------------------------------
# Poisson — rates per scheduled watch-hour
# ---------------------------------------------------------------------------


def _q_gamma_int(m: int, x: float) -> float:
    """Regularized UPPER incomplete gamma ``Q(m, x)`` for integer shape ``m >= 1``.

    Closed form for integer shape: ``Q(m, x) = e^-x * sum_{j=0}^{m-1} x^j / j!`` — i.e. the Poisson
    probability of observing fewer than ``m`` events with mean ``x``. Summed with a running term to
    avoid computing large factorials.
    """
    term = 1.0
    total = 1.0
    for j in range(1, m):
        term *= x / j
        total += term
    return math.exp(-x) * total


def poisson_factor(k: int, *, confidence: float = DEFAULT_CONFIDENCE) -> float:
    """``factor(k) = chi2_{conf, 2(k+1)} / 2`` — the exposure-free part of the Poisson upper bound.

    ``lambda_upper = factor(k) / T``. ``factor(0) = 2.9957`` (the rule of three), ``factor(1) =
    4.7439``, ``factor(2) = 6.2958`` — the values tabulated in the costing doc §1.2.
    """
    if k < 0:
        raise ValueError(f"k (observed events) must be >= 0, got {k!r}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")
    alpha = 1.0 - confidence
    m = k + 1
    # Q(m, x) decreases monotonically from 1 to 0; bracket then bisect.
    lo, hi = 0.0, max(1.0, float(m))
    while _q_gamma_int(m, hi) > alpha:
        hi *= 2.0
        if hi > 1e12:  # pragma: no cover - unreachable for sane confidence levels
            raise ValueError("failed to bracket the Poisson quantile")
    for _ in range(_MAX_ITER):
        mid = 0.5 * (lo + hi)
        if _q_gamma_int(m, mid) > alpha:
            lo = mid
        else:
            hi = mid
        if hi - lo < _TOL * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


def poisson_upper_bound(
    k: int, exposure: float, *, confidence: float = DEFAULT_CONFIDENCE
) -> float | None:
    """Exact one-sided upper bound on a Poisson RATE: ``factor(k) / exposure``.

    ``exposure`` is in whatever unit the rate is quoted in — for §3.2 that is **scheduled
    watch-hours**, never raw stream-hours. Returns ``None`` for zero/negative exposure: no observed
    time means no bound exists, and inventing one (or reporting the point estimate ``0/0``) is
    exactly the dishonesty this module is here to prevent.
    """
    if exposure <= 0:
        return None
    return poisson_factor(k, confidence=confidence) / float(exposure)


def required_exposure(
    bar: float, k: int, *, confidence: float = DEFAULT_CONFIDENCE
) -> float | None:
    """Exposure needed for ``k`` observed events to still bound the rate at ``<= bar``.

    ``T_required(b, k) = factor(k) / b`` (costing doc §1.2). ``None`` when ``bar <= 0`` — a bar of
    zero events per hour is unbounded evidence, not a large number of hours.
    """
    if bar <= 0:
        return None
    return poisson_factor(k, confidence=confidence) / float(bar)


# ---------------------------------------------------------------------------
# Binomial — proportions (recall) over labeled positive episodes
# ---------------------------------------------------------------------------


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (modified Lentz, Numerical Recipes §6.4)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, _MAX_ITER):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-16:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """``I_x(a, b)`` — the CDF of a Beta(a, b) distribution evaluated at ``x``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def clopper_pearson_lower(
    successes: int, trials: int, *, confidence: float = DEFAULT_CONFIDENCE
) -> float | None:
    """Exact one-sided LOWER bound on a proportion: ``Beta^-1(1 - conf; x, n - x + 1)``.

    Returns ``None`` when ``trials == 0`` — zero labeled episodes supports no recall claim at all,
    not a vacuous ``1.0``. Returns ``0.0`` when ``successes == 0`` (the honest floor).
    """
    if trials < 0 or successes < 0:
        raise ValueError("successes and trials must be >= 0")
    if successes > trials:
        raise ValueError(f"successes ({successes}) cannot exceed trials ({trials})")
    if trials == 0:
        return None
    if successes == 0:
        return 0.0
    alpha = 1.0 - confidence
    a = float(successes)
    b = float(trials - successes + 1)
    lo, hi = 0.0, 1.0
    for _ in range(_MAX_ITER):
        mid = 0.5 * (lo + hi)
        if regularized_incomplete_beta(a, b, mid) < alpha:
            lo = mid
        else:
            hi = mid
        if hi - lo < _TOL:
            break
    return 0.5 * (lo + hi)


def required_positives(
    bar: float,
    misses: int,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    max_trials: int = 100_000,
) -> int | None:
    """Smallest ``n`` labeled positive episodes whose exact lower bound clears ``bar``, given ``misses``.

    Costing doc §3: ``recall_min = 0.90`` with zero misses needs ``n = 29``; one miss needs 46.
    Returns ``None`` if no ``n <= max_trials`` suffices (``bar >= 1.0`` is unreachable by an exact
    bound at any finite ``n``).
    """
    if not 0.0 < bar < 1.0:
        return None
    if misses < 0:
        raise ValueError(f"misses must be >= 0, got {misses!r}")
    n = max(misses + 1, 1)
    while n <= max_trials:
        lower = clopper_pearson_lower(n - misses, n, confidence=confidence)
        if lower is not None and lower >= bar:
            return n
        n += 1
    return None
