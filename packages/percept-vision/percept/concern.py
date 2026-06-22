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

from .adapters import Concern, Consequence, Evidence, Goal, Verdict

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


# --- Consequence's three CALIBRATION-FREE jobs (A3, spec §5) -------------------------------------
# These are how a coarse consequence earns its keep TODAY, while the threshold job (threshold_for)
# stays inert (R3): none of them needs calibrated confidence — they read stakes, not a 0..1 number.

_MISS_RANK = {"trivial": 0, "mild": 1, "severe": 2}
_FF_RANK = {"trivial": 0, "mild": 1, "severe": 2, "irreversible": 3}


def consequence_of(goal: Goal) -> Consequence:
    """Read a goal's consequence from its state (the conductor authors these; default mild/mild).

    ``protected_floor`` PINS ``miss_cost`` at maximum — refinement may relax convenience but never
    that (spec §5). So a protected-floor concern is always ``severe`` miss, regardless of the stored tag.
    """
    floor = bool(goal.state.get("protected_floor", False))
    miss = "severe" if floor else goal.state.get("miss_cost", "mild")
    return Consequence(
        miss_cost=miss,
        false_fire_cost=goal.state.get("false_fire_cost", "mild"),
        protected_floor=floor,
    )


def attention_weight(c: Consequence) -> int:
    """Job — attention allocation (spec §5): perception budget ∝ stakes. Pure miss-cost rank, no
    confidence. High-consequence concerns are tracked attentively; low ones loosely."""
    return _MISS_RANK[c.miss_cost]


def is_hard_quadrant(c: Consequence) -> bool:
    """Job — hard-quadrant routing (spec §5): both costs high => no cheap threshold is safe =>
    escalate to the deliberative tier / ASK the human instead of auto-firing."""
    return _MISS_RANK[c.miss_cost] >= 2 and _FF_RANK[c.false_fire_cost] >= 2


def requires_verification(c: Consequence) -> bool:
    """Job — verification-required (spec §5,§7): a high false-fire cost must be verified before/after
    acting (consumed by the action layer, A4). Irreversible always requires it."""
    return _FF_RANK[c.false_fire_cost] >= 2


__all__ = [
    "DEFAULT_THRESHOLD",
    "OBS_CAP",
    "threshold_for",
    "observe",
    "derive_belief",
    "consequence_of",
    "attention_weight",
    "is_hard_quadrant",
    "requires_verification",
]
