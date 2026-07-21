"""Concern logic — the upstream adapters that feed the UNCHANGED gate.

LIVE surface (imported by agent.py / behavior.py, on the real firing path):
  * ``consequence_of`` — read a goal's consequence (agent.py, behavior.py)
  * ``attention_weight`` / ``is_hard_quadrant`` / ``requires_verification`` — Consequence's three
    calibration-free jobs (agent.py, behavior.py)
  * ``resolution_ready`` — the process-QA "is this running answer ready?" predicate (agent.py)

INERT-but-LIVE: ``threshold_for`` is imported by no production module today (it is only exercised by
tests). It is kept as the pinned per-concern threshold seam — its DEFAULT must equal the gate's 0.7
so wiring it later is byte-for-byte safe. At today's non-discriminative confidence it would
be a no-op, which is why the runtime does not route through it yet.

REMOVED (2026-07-12 OSS hardening): the ``derive_belief`` / ``observe`` / ``OBS_CAP`` belief-log
spike. It was a pure-function pair the agent never routed through (docstring admitted "the agent
does not route through them yet"); it wrote to / read from ``Concern.observations``, which nothing in
the runtime populates. Deleted rather than left as architecture-shaped dead code. The gate-unchanged
invariant it claimed to enforce is already held structurally: ``behavior.py`` is the only driver of
the gate and never edits it (see ``test_program_equivalence``).
"""

from __future__ import annotations

from .adapters import Consequence, Goal

# Pinned to the measured current gate defaults.
DEFAULT_THRESHOLD = 0.7


def threshold_for(consequence: Consequence) -> float:
    """Per-concern firing threshold from the cost asymmetry.

    Honest: at today's NON-discriminative confidence (~constant 0.95) this job is **inert** —
    "fire the child at a lower threshold" is a no-op, so v1 returns the gate default for everything
    and consequence earns its keep via attention / escalation / verification instead. The
    asymmetry logic slots in here only once the labeling pipeline shows confidence discriminates.
    The DEFAULT MUST equal the gate's threshold so existing goals are byte-for-byte unchanged.
    """
    return DEFAULT_THRESHOLD


# --- Consequence's three CALIBRATION-FREE jobs -------------------------------------
# These are how a coarse consequence earns its keep TODAY, while the threshold job (threshold_for)
# stays inert: none of them needs calibrated confidence — they read stakes, not a 0..1 number.

_MISS_RANK = {"trivial": 0, "mild": 1, "severe": 2}
_FF_RANK = {"trivial": 0, "mild": 1, "severe": 2, "irreversible": 3}


def consequence_of(goal: Goal) -> Consequence:
    """Read a goal's consequence from its state (the conductor authors these; default mild/mild).

    ``protected_floor`` PINS ``miss_cost`` at maximum — refinement may relax convenience but never
    that. So a protected-floor concern is always ``severe`` miss, regardless of the stored tag.
    """
    floor = bool(goal.state.get("protected_floor", False))
    miss = "severe" if floor else goal.state.get("miss_cost", "mild")
    return Consequence(
        miss_cost=miss,
        false_fire_cost=goal.state.get("false_fire_cost", "mild"),
        protected_floor=floor,
    )


def attention_weight(c: Consequence) -> int:
    """Job — attention allocation: perception budget ∝ stakes. Pure miss-cost rank, no
    confidence. High-consequence concerns are tracked attentively; low ones loosely."""
    return _MISS_RANK[c.miss_cost]


def is_hard_quadrant(c: Consequence) -> bool:
    """Job — hard-quadrant routing: both costs high => no cheap threshold is safe =>
    escalate to the deliberative tier / ASK the human instead of auto-firing."""
    return _MISS_RANK[c.miss_cost] >= 2 and _FF_RANK[c.false_fire_cost] >= 2


def requires_verification(c: Consequence) -> bool:
    """Job — verification-required: a high false-fire cost must be verified before/after
    acting (consumed by the action layer). Irreversible always requires it."""
    return _FF_RANK[c.false_fire_cost] >= 2


# --- process-QA resolution: when a running answer is COMPLETE enough to resolve (Item 2 mvp2) ----
# A process-QA concern accumulates a per-concern ``running_answer`` over many escalations (the model
# rewrites its summary each time it sees more evidence). The shipped path NEVER resolves it: the gate
# only fires on a confident-YES ``satisfied`` verdict, which an evidence-gathering judge returns rarely.
# So the running answer grows but is never EMITTED as a recognized final answer — every turn reads as
# unresolved. These pure adapters expose the missing "is this answer ready?" decision WITHOUT touching
# the gate: resolution is an ANSWER-PATH concern (accumulated coverage + confidence), de-duped by the
# agent through the same synthetic-verdict-on-a-dedicated-key mechanism as coverage/episode/overdue.

# Resolution bar — accumulated COVERAGE (how many evidence-bearing observations folded into the answer)
# and SUBSTANCE (the answer is non-trivial). Tuned to protect precision: a single thin fragment must
# NOT resolve. All general — driven by accumulation, never by the question text.
RESOLVE_MIN_EVIDENCE = 2  # at least two evidence-bearing observations must agree into the answer
RESOLVE_MIN_CHARS = 12  # the running answer must carry real substance (not an empty/stub fragment)
RESOLVE_MIN_CONFIDENCE = 0.6  # the latest read must not be a coin-flip guess


def resolution_ready(
    running_answer: str,
    evidence_count: int,
    confidence: float,
    *,
    min_evidence: int = RESOLVE_MIN_EVIDENCE,
    min_chars: int = RESOLVE_MIN_CHARS,
    min_confidence: float = RESOLVE_MIN_CONFIDENCE,
    at_boundary: bool = False,
) -> bool:
    """Has an accumulated process-QA ``running_answer`` crossed the bar to RESOLVE into a final answer?

    GENERAL (no per-question logic): resolution is driven by accumulated COVERAGE (``evidence_count`` —
    how many evidence-bearing observations the running answer folds in), SUBSTANCE (the answer is long
    enough to be real), and CONFIDENCE (the latest read isn't a coin-flip). An unseen process question
    resolves with zero new branches — the predicate only reads accumulation.

    Precision guard: a thin running answer (one fragment, or empty/stub) never resolves — a real partial
    with numbers beats a hallucinated fix. ``at_boundary`` relaxes the coverage floor by ONE (an episode
    that ended with a single substantive observation is allowed to resolve what little it confidently saw,
    but still never resolves on empty/no-confidence)."""
    answer = (running_answer or "").strip()
    if not answer or len(answer) < min_chars:
        return False  # nothing substantive accumulated — never hallucinate an answer
    if confidence < min_confidence:
        return False  # the latest read is too uncertain to commit to
    floor = max(1, min_evidence - 1) if at_boundary else min_evidence
    return evidence_count >= floor


__all__ = [
    "DEFAULT_THRESHOLD",
    "RESOLVE_MIN_EVIDENCE",
    "RESOLVE_MIN_CHARS",
    "RESOLVE_MIN_CONFIDENCE",
    "threshold_for",
    "consequence_of",
    "attention_weight",
    "is_hard_quadrant",
    "requires_verification",
    "resolution_ready",
]
