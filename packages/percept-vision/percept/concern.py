"""Concern logic — the upstream adapters that feed the UNCHANGED gate (spec §3,§4,§5).

Belief is **derived** from a concern's observation log against its expectations — computed on read,
never stored (spec §3). The two adapters here (``derive_belief`` -> ``(Verdict, freshness)`` and
``threshold_for`` -> ``float``) produce exactly the inputs ``percept.gate.classify`` / ``Gate.evaluate``
already consume, which is how "the gate stays literally unchanged" goes from aspiration to invariant
(plan R2a). The bounded log keeps a 24/7 agent from leaking memory (plan R14).

A0 introduces these as pure, unit-tested functions; the agent does not route through them yet (A1/A2
wire them, each guarded by the byte-for-byte invariant). Defaults are pinned to the measured gate
numbers (plan R15): ``threshold_for(Consequence()) == 0.7`` == ``Gate().threshold``.
"""

from __future__ import annotations

from .adapters import Concern, Consequence, Evidence, Verdict

# Pinned to the measured current gate defaults (S2 capture; plan R15).
DEFAULT_THRESHOLD = 0.7
# Bounded observation log — a baby monitor runs for days (plan R14).
OBS_CAP = 256


def threshold_for(consequence: Consequence) -> float:
    """Per-concern firing threshold from the cost asymmetry (spec §5).

    R3 (honest): at today's NON-discriminative confidence (~constant 0.95) this job is **inert** —
    "fire the child at a lower threshold" is a no-op, so v1 returns the gate default for everything
    and consequence earns its keep via attention / escalation / verification (A3) instead. The
    asymmetry logic slots in here only once the labeling pipeline (B3) shows confidence discriminates.
    The DEFAULT MUST equal the gate's threshold so existing goals are byte-for-byte unchanged.
    """
    return DEFAULT_THRESHOLD


def observe(concern: Concern, ev: Evidence) -> None:
    """Append evidence to the concern's log, bounded to the most-recent ``OBS_CAP`` (plan R14)."""
    obs = concern.observations
    obs.append(ev)
    extra = len(obs) - OBS_CAP
    if extra > 0:
        del obs[:extra]


def derive_belief(
    concern: Concern, now: float, *, half_life_s: float = 30.0
) -> tuple[Verdict, float]:
    """Collapse the observation log into the ``(Verdict, freshness)`` the EXISTING gate eats.

    Belief is derived on read, never stored (spec §3). ``freshness`` decays with the recency of the
    most-recent supporting observation, so **observability falls out for free** (spec §3): a
    confident-but-stale belief yields low freshness, which the unchanged ``classify`` already maps to
    "unknown" -> ASK. With no observations, coverage is zero -> not-fresh.
    """
    obs = concern.observations
    if not obs:
        return Verdict(satisfied=False, confidence=0.0, evidence="no-observations"), 0.0
    last = obs[-1]
    age = max(0.0, now - last.t)
    if half_life_s <= 0:
        freshness = 1.0 if age == 0.0 else 0.0
    else:
        freshness = 0.5 ** (age / half_life_s)
    freshness *= last.coverage
    return (
        Verdict(satisfied=last.satisfied, confidence=last.confidence, evidence=f"obs@{last.t:.3f}"),
        freshness,
    )


__all__ = ["DEFAULT_THRESHOLD", "OBS_CAP", "threshold_for", "observe", "derive_belief"]
