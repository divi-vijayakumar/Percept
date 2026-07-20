"""Posture-B deliberative escalation — the consequence×novelty dial.

"As close to an LLM as each concern's stakes × novelty justify, and no closer." Recurrent AMBIGUITY (a
run of ``unknown`` reads — the novelty/confusion signal in :class:`~percept.behavior.ConfusionSignal`)
is the trigger to re-invoke the LLM instead of re-asking the human every tick; the concern's CONSEQUENCE
(stakes) sets how SOON; a per-concern cooldown is the deliberative analog of the gate's refractory, so
the LLM can't be thrashed.

Pure decision functions — no LLM, no agent import. The agent (``Percept.deliberate``) reads these and
performs the re-invocation. ``ctx`` is duck-typed (``ctx.confusion.consecutive_unknown`` +
``ctx.last_escalated_at``) so this module stays decoupled from ``behavior``.
"""

from __future__ import annotations

from .adapters import Consequence

_MISS_RANK = {"trivial": 0, "mild": 1, "severe": 2}

# Consecutive AMBIGUOUS reads before re-invoking the LLM, by miss-cost stakes. Higher stakes escalate
# SOONER; trivial never escalates (an LLM call isn't worth it). This IS the dial.
_ESCALATE_AFTER = {2: 2, 1: 4}  # severe: after 2 unknowns; mild: after 4; trivial: never


def escalation_threshold(consequence: Consequence) -> int | None:
    """How many consecutive ambiguous reads warrant an LLM re-invocation for this concern's stakes —
    or ``None`` if its stakes never justify one."""
    return _ESCALATE_AFTER.get(_MISS_RANK[consequence.miss_cost])


def should_escalate(ctx, consequence: Consequence, now: float, *, cooldown_s: float) -> bool:
    """Re-invoke the LLM iff the concern is recurrently ambiguous past its stakes-set threshold AND it
    is outside the per-concern cooldown (the deliberative refractory that prevents thrash)."""
    threshold = escalation_threshold(consequence)
    if threshold is None:
        return False
    if ctx.confusion.consecutive_unknown < threshold:
        return False
    return (now - ctx.last_escalated_at) >= cooldown_s


MAX_PERCEPTION_LEVEL = 2  # 0=single · 1=strip · 2=+audio


def escalate_perception(
    ctx, consequence: Consequence, now: float, *, cooldown_s: float, max_level: int = MAX_PERCEPTION_LEVEL
) -> bool:
    """The CHEAP rung of the dial: when a concern is recurrently ambiguous, escalate its PERCEPTION
    (single→strip→+audio) BEFORE re-invoking the LLM — better looking is cheaper than more thinking.

    Same stakes-set ambiguity threshold as :func:`should_escalate`, but a SHORTER cooldown (so perception
    climbs first) and capped at ``max_level``. Bumps ``ctx.perception_level`` + stamps the perception
    refractory; returns ``True`` iff it escalated this tick. The agent maxes this ladder before the LLM
    rung, and decays the level back to 0 on the next confident read."""
    threshold = escalation_threshold(consequence)
    if threshold is None:
        return False
    if ctx.confusion.consecutive_unknown < threshold:
        return False
    if ctx.perception_level >= max_level:
        return False
    if (now - ctx.last_perception_escalate_at) < cooldown_s:
        return False
    ctx.perception_level += 1
    ctx.last_perception_escalate_at = now
    return True


__all__ = ["escalation_threshold", "should_escalate", "escalate_perception", "MAX_PERCEPTION_LEVEL"]
