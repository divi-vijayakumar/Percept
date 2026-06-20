"""The V0.0 routing lexicon — an ordered, data-only rule table.

This module is **pure data + regexes**. It carries no control flow: the evaluator in
``engine.py`` walks it, first confident match wins, and a structural miss DEFERS to the
conductor (returns ``None``). The router fast-paths only confident, *non-authoring* turns —
everything goal-authoring / conditional / ambiguous / a fragment is intentionally absent from
the confident table and therefore defers (invariant 2: "when in doubt, defer").

Each :class:`Rule` carries ``examples`` (utterances that should HIT it) and ``defers``
(utterances that look superficially similar but must DEFER); the L1 suite auto-generates a
case per example/defer string so the table is self-testing. Regex lexicons are a V0.0
expedient — a learned tier (``LearnedRouteLayer``) replaces them in V0.1 behind the same
``RouteLayer`` seam, so nothing here is load-bearing structure.

A rule never *authors a goal*: authoring is the conductor's sole privilege (it is the only
tier that writes standing state). The router can only pick a confident **route**; goal ops
always come back from the conductor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# A "route" string here must be a member of adapters.Decision.route. We keep the literal set
# local (as data, not imported logic) so this module stays import-light and the table is
# legible; engine.py validates against the frozen Decision at construction time.
Route = str


@dataclass(frozen=True)
class Rule:
    """One confident routing fast-path: a named lexicon -> a route, with self-test fixtures.

    ``patterns`` are matched case-insensitively as *search* (substring-anchored) regexes.
    ``confidence`` is the score the layer reports on a match; the engine defers anything below
    the configured threshold, so a deliberately low-confidence rule is itself a defer. ``look``
    mirrors :class:`~percept.adapters.Decision.look` for routes that want a frame.
    """

    name: str
    route: Route
    patterns: tuple[str, ...]
    confidence: float = 0.9
    look: str = "none"  # "none" | "request_frame"
    examples: tuple[str, ...] = field(default_factory=tuple)  # must HIT this rule
    defers: tuple[str, ...] = field(default_factory=tuple)  # must DEFER (no confident match)

    def compiled(self) -> list[re.Pattern[str]]:
        return [re.compile(p, re.IGNORECASE) for p in self.patterns]


# ---------------------------------------------------------------------------
# DEFAULT_RULES — ordered, first confident match wins.
#
# Only confident, non-authoring intents live here. Conspicuously ABSENT (and therefore
# deferred): goal authoring ("watch my coffee", "remind me when..."), conditionals ("if the
# water boils..."), and bare fragments ("the cup", "uh"). Those carry standing-state or
# ambiguity and must reach the conductor.
# ---------------------------------------------------------------------------

DEFAULT_RULES: tuple[Rule, ...] = (
    # --- spatial / perceptual questions -> ground the reply on a frame -------------------
    Rule(
        name="vision_question",
        route="respond_with_vision",
        patterns=(
            r"\bwhere('?s| is| are| did)\b",
            r"\bwhat('?s| is| are| am i| are these| are those)\b.*\b(this|that|here|holding|see|looking|wearing|color|colou?r)\b",
            r"\bwhich (one|cup|color|colou?r)\b",
            r"\b(can|do) you see\b",
            r"\bhow many\b.*\b(here|there|see|on|in)\b",
            r"\bis (this|that|it|he|she|there)\b.*\?",
            r"\bread (this|that|it|the)\b",
        ),
        confidence=0.92,
        look="request_frame",
        examples=(
            "where is my phone?",
            "where's the remote",
            "what is this thing I'm holding?",
            "what color is that?",
            "which cup is mine?",
            "can you see the stove?",
            "how many cups are on the table?",
            "read this label for me",
        ),
        defers=(
            "where should I go on vacation?",  # not perceptual -> conductor
            "what do you think?",  # opinion, not grounded -> conductor
        ),
    ),
    # --- general chit-chat / direct questions -> answer without a frame ------------------
    Rule(
        name="direct_question",
        route="respond_direct",
        patterns=(
            r"^\s*(who|why|when|how)\b",
            r"\bwhat('?s| is| are| does| do)\b",
            r"\bcan you (tell|explain|help|give)\b",
            r"\bdo you (know|think)\b",
            r"\?\s*$",
        ),
        confidence=0.8,
        examples=(
            "why is the sky blue?",
            "how does this recipe work?",
            "what does mise en place mean?",
            "can you explain that?",
            "do you know the time?",
        ),
        defers=(
            "what should I watch for?",  # borderline authoring -> defer
        ),
    ),
    # --- greetings / acknowledgements / thanks -> a direct, frame-free reply -------------
    Rule(
        name="social",
        route="respond_direct",
        patterns=(
            r"^\s*(hi|hey|hello|yo|good (morning|afternoon|evening))\b",
            r"^\s*(thanks|thank you|cheers|appreciate it)\b",
            r"^\s*(bye|goodbye|see you|good night)\b",
        ),
        confidence=0.9,
        examples=(
            "hey there",
            "good morning",
            "thanks so much",
            "goodbye for now",
        ),
        defers=(
            "thanks, now watch the pan",  # carries an authoring tail -> defer
        ),
    ),
    # --- explicit dismissals / stop -> ignore (no model turn) ----------------------------
    Rule(
        name="dismiss",
        route="ignore",
        patterns=(
            r"^\s*(never ?mind|nevermind|forget it|stop|cancel that|nothing)\s*[.!]?\s*$",
            r"^\s*(no|nope|nah)\s*[.!]?\s*$",
        ),
        confidence=0.9,
        examples=(
            "never mind",
            "forget it",
            "stop",
            "no",
        ),
        defers=(
            "stop watching the coffee",  # references a goal -> conductor decides
        ),
    ),
)


# ---------------------------------------------------------------------------
# DEFER_GUARDS — a *veto* lexicon. If any guard matches, NO confident rule may fire and the
# engine defers, even if a fast-path rule also matched. This is the structural realization of
# "when in doubt, defer": the four defer classes (goal-authoring, conditional, ambiguous,
# opinion/underspecified) are recognized by their markers and routed to the conductor — the
# only tier allowed to author standing state. A false fast-path could skip goal authoring, so
# the guard is intentionally broad; a false defer only costs latency (invariant 2).
#
# Like the rules, this is a V0.0 regex expedient; the V0.1 LearnedRouteLayer subsumes it by
# returning a low confidence on these utterances, which the same threshold gates.
# ---------------------------------------------------------------------------

DEFER_GUARDS: tuple[str, ...] = (
    # goal authoring — watch / remind / count / keep an eye / let me know / tell me when
    r"\bwatch(es|ing)?\b",
    r"\bkeep an eye\b",
    r"\bremind(s|ing)?\b\s+me",
    r"\bcount\b.*\b(how many|times|every)\b",
    r"\blet me know\b",
    r"\btell me (when|if|whenever)\b",
    r"\bnotify\b",
    # conditional — if / when / whenever / as soon as <something> (then) <do>
    r"^\s*(if|when|whenever|as soon as)\b",
    r"\b(if|when|whenever|as soon as)\b.*\b(let me know|tell me|say|notify|warn|alert)\b",
    # opinion / underspecified — not perceptual, the conductor should reason
    r"\bshould i\b",
    r"\bwhat (do|would) you (think|recommend|suggest)\b",
    r"\bdo the thing\b",
    r"\byou know what i mean\b",
)


# ---------------------------------------------------------------------------
# DEFER fixtures the table as a whole must punt on (used by the L1 suite). These are the
# four classes the spec calls out: goal-authoring, conditional, ambiguous, fragment.
# They live here (not in the test) so the contract is versioned with the lexicon.
# ---------------------------------------------------------------------------

DEFER_EXAMPLES: tuple[str, ...] = (
    # goal authoring (standing state — conductor only)
    "watch my coffee and tell me if I drink it",
    "remind me when the water boils",
    "let me know if someone comes to the door",
    "keep an eye on the stove",
    "count how many times I check my phone",
    # conditional
    "if the pan starts smoking say something",
    "when the timer goes off let me know",
    # ambiguous / underspecified
    "do the thing",
    "you know what I mean",
    # fragment / disfluency
    "the cup",
    "uh",
    "so like",
)


__all__ = ["Rule", "Route", "DEFAULT_RULES", "DEFER_GUARDS", "DEFER_EXAMPLES"]
