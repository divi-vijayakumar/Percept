"""Corpus 4 — EPISODE COUNTING, the cup problem: "how many cups of coffee did I drink?"

The failure is sampling: an activity observed 50 times is not 50 occurrences. Counting has to be
over EPISODES, and an episode boundary is a rule about absence, not about frames.

This is testable SYNTHETICALLY and needs no real footage — the input that matters is the *pattern of
presence and absence over time*, which a generated stream expresses exactly. Three cases pin the
boundary rule itself (separation exactly at the gap, and at gap ± epsilon), so the rule cannot drift
without a fixture going red.

Boundary rule as labelled here: a separation (gap between the END of one occurrence and the START of
the next) of **>= BOUNDARY_GAP_S** starts a new episode; anything shorter continues.
"""

from __future__ import annotations

from ._common import frame_step, labels_doc, obs, stream_doc

NAME = "episode_counting"
GENERATOR = __name__

#: The episode boundary. Stated in every expectation so the rule is pinned by the fixture.
BOUNDARY_GAP_S = 20.0
EPSILON_S = 0.1
_ACTIVITY = "drinking_from_cup"


def _samples(start: float, end: float, step_s: float, *, present: bool) -> list[dict]:
    out: list[dict] = []
    t = start
    # Integer tick arithmetic — no float accumulation drift, so the JSON is byte-stable.
    ticks = int(round((end - start) / step_s))
    for i in range(ticks + 1):
        t = round(start + i * step_s, 6)
        out.append(
            frame_step(
                t,
                f"{'sip' if present else 'idle'}@{t:g}",
                observations=[obs("cup", confidence=0.9)] if present else [],
                verdict={
                    "satisfied": present,
                    "confidence": 0.93 if present else 0.91,
                    "evidence": f"scripted: {_ACTIVITY}={'yes' if present else 'no'} at t={t:g}",
                    "action_present": present,
                    "subject_key": "activity:drinking_from_cup",
                },
            )
        )
    return out


def _occurrences(spans: list[tuple[float, float]], *, step_s: float = 1.0) -> list[dict]:
    """Sample only WITHIN each occurrence — the space between spans is genuine absence."""
    steps: list[dict] = []
    for start, end in spans:
        steps.extend(_samples(start, end, step_s, present=True))
    return steps


def build() -> tuple[dict, dict]:
    g, eps = BOUNDARY_GAP_S, EPSILON_S

    three = [(0.0, 2.0), (40.0, 42.0), (80.0, 82.0)]
    exact = [(0.0, 2.0), (2.0 + g, 4.0 + g)]                    # separation == 20.0
    under = [(0.0, 2.0), (2.0 + g - eps, 4.0 + g - eps)]        # separation == 19.9
    over = [(0.0, 2.0), (2.0 + g + eps, 4.0 + g + eps)]         # separation == 20.1

    cases = [
        {
            "id": "three_separated_occurrences",
            "description": "Three drinking occurrences separated by ~38s of absence each.",
            "steps": _occurrences(three),
        },
        {
            "id": "one_continuous_occurrence_sampled_fifty_times",
            "description": "ONE continuous drinking occurrence, sampled 50 times at 0.5s intervals.",
            "steps": _samples(0.0, 24.5, 0.5, present=True),
        },
        {
            "id": "boundary_separation_exactly_gap",
            "description": f"Two occurrences separated by EXACTLY the {g:g}s boundary gap.",
            "steps": _occurrences(exact),
        },
        {
            "id": "boundary_separation_gap_minus_epsilon",
            "description": f"Two occurrences separated by {g - eps:g}s — one epsilon UNDER the boundary.",
            "steps": _occurrences(under),
        },
        {
            "id": "boundary_separation_gap_plus_epsilon",
            "description": f"Two occurrences separated by {g + eps:g}s — one epsilon OVER the boundary.",
            "steps": _occurrences(over),
        },
    ]

    rule = "separation >= boundary_gap_s starts a NEW episode; a shorter separation CONTINUES"

    def _lab(cid: str, count: int, spans: list, proves: str, does_not: str, **extra) -> dict:
        return {
            "id": cid,
            "expect": {
                "count": count,
                "boundaries": [[float(a), float(b)] for a, b in spans],
                "boundary_gap_s": g,
                "boundary_rule": rule,
                **extra,
            },
            "proves": proves,
            "does_not_prove": does_not,
        }

    labels = [
        _lab(
            "three_separated_occurrences", 3, three,
            "Separated occurrences of the same activity count as distinct episodes.",
            "That a real judge detects each occurrence — detection recall is a benchmark question.",
            wrong_answer={"count": 9, "why": "counting satisfied frames instead of episodes"},
        ),
        _lab(
            "one_continuous_occurrence_sampled_fifty_times", 1, [(0.0, 24.5)],
            "Sampling rate must not affect the count: 50 observations of one occurrence is ONE episode.",
            "That the activity is really continuous in real footage, where detection flickers.",
            sample_count=50,
            wrong_answer={"count": 50, "why": "one increment per satisfied observation — the cup problem"},
        ),
        _lab(
            "boundary_separation_exactly_gap", 2, exact,
            f"At a separation of EXACTLY {g:g}s the rule resolves to a NEW episode (>= is inclusive).",
            "That this particular gap value is correct for any real activity.",
            separation_s=g,
        ),
        _lab(
            "boundary_separation_gap_minus_epsilon", 1, [(0.0, 4.0 + g - eps)],
            f"A separation of {g - eps:g}s is inside the gap — the occurrences are ONE episode.",
            "That epsilon-scale timing survives real jitter, dropped frames, or clock skew.",
            separation_s=round(g - eps, 6),
        ),
        _lab(
            "boundary_separation_gap_plus_epsilon", 2, over,
            f"A separation of {g + eps:g}s is past the gap — TWO episodes.",
            "That epsilon-scale timing survives real jitter, dropped frames, or clock skew.",
            separation_s=round(g + eps, 6),
        ),
    ]

    return (
        stream_doc(NAME, "frames", GENERATOR, cases, boundary_gap_s=g, epsilon_s=eps),
        labels_doc(NAME, labels, boundary_gap_s=g, epsilon_s=eps, boundary_rule=rule),
    )
