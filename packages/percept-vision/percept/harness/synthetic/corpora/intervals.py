"""Corpus 5 — INTERVALS: coverage comes from observation, never from interpolation.

"How long was the door open?" is answerable only over spans you actually observed. Two sightings 19
seconds apart do NOT make a 19-second interval — that number is manufactured by forward-filling an
unobserved gap, which the design explicitly rejects. Coverage must come from observed spans or from
a declared validity horizon; everything else is a GAP and must be reported as one.

The third case is the one that bites in production: the judge degraded (backpressure, a failed
call), so nothing was observed. An unobserved stretch must be marked as a gap, NOT carried forward
as continued truth — silence is not evidence.
"""

from __future__ import annotations

from ._common import frame_step, labels_doc, obs, stream_doc

NAME = "intervals"
GENERATOR = __name__


def _look(t: float, *, satisfied: bool = True, degraded: bool = False) -> dict:
    if degraded:
        verdict = {
            "satisfied": False,
            "confidence": 0.0,
            "evidence": "scripted: judge degraded — nothing was observed at this tick",
            "degraded": True,
            "observed": False,
        }
    else:
        verdict = {
            "satisfied": satisfied,
            "confidence": 0.94,
            "evidence": f"scripted: door_open={'yes' if satisfied else 'no'} at t={t:g}",
            "observed": True,
            "subject_key": "door:front",
        }
    extra = {"degraded": True} if degraded else {}
    return frame_step(
        t,
        f"{'degraded' if degraded else 'door'}@{t:g}",
        observations=[] if degraded else [obs("door", confidence=0.94)],
        verdict=verdict,
        **extra,
    )


def build() -> tuple[dict, dict]:
    cases = [
        {
            "id": "two_sparse_observations_are_not_a_span",
            "description": "The door is observed open at t=1 and again at t=20, with NOTHING observed between.",
            "steps": [_look(1.0), _look(20.0)],
        },
        {
            "id": "continuous_observation_is_the_span",
            "description": "The door is observed open on every tick from t=0 to t=10 (1s cadence).",
            "steps": [_look(float(t)) for t in range(0, 11)],
        },
        {
            "id": "degraded_judge_gap_is_not_continued_truth",
            "description": (
                "Observed open t=0..3; the judge is then DEGRADED and observes nothing t=4..8; observed "
                "open again t=9..10."
            ),
            "steps": (
                [_look(float(t)) for t in range(0, 4)]
                + [_look(float(t), degraded=True) for t in range(4, 9)]
                + [_look(float(t)) for t in range(9, 11)]
            ),
        },
    ]

    labels = [
        {
            "id": "two_sparse_observations_are_not_a_span",
            "expect": {
                "intervals": [
                    {"start": 1.0, "end": 1.0, "kind": "observed"},
                    {"start": 20.0, "end": 20.0, "kind": "observed"},
                ],
                "gaps": [{"start": 1.0, "end": 20.0, "reason": "unobserved"}],
                "forward_fill_forbidden": True,
                "observed_total_s": 0.0,
                "wrong_answer": {
                    "interval_s": 19.0,
                    "why": "forward-filling the unobserved gap between two sightings into one span",
                },
            },
            "proves": (
                "Two sparse observations yield two point observations and one declared GAP — never a "
                "19-second interval."
            ),
            "does_not_prove": (
                "What the door was actually doing between t=1 and t=20. That is unknowable from this "
                "stream, which is precisely the point."
            ),
        },
        {
            "id": "continuous_observation_is_the_span",
            "expect": {
                "intervals": [{"start": 0.0, "end": 10.0, "kind": "observed"}],
                "gaps": [],
                "forward_fill_forbidden": True,
                "observed_total_s": 10.0,
                "sample_cadence_s": 1.0,
                "wrong_answer": {
                    "interval_count": 11,
                    "why": "treating each tick as its own interval instead of merging a continuous span",
                },
            },
            "proves": "A genuinely continuous observation IS the interval — merged, not fragmented per tick.",
            "does_not_prove": "That 1s cadence is dense enough for any real question about duration.",
        },
        {
            "id": "degraded_judge_gap_is_not_continued_truth",
            "expect": {
                "intervals": [
                    {"start": 0.0, "end": 3.0, "kind": "observed"},
                    {"start": 9.0, "end": 10.0, "kind": "observed"},
                ],
                "gaps": [{"start": 3.0, "end": 9.0, "reason": "judge_degraded"}],
                "forward_fill_forbidden": True,
                "observed_total_s": 4.0,
                "degraded_step_indices": [4, 5, 6, 7, 8],
                "wrong_answers": [
                    {"observed_total_s": 10.0, "why": "carrying the last true verdict across the outage"},
                    {"gaps": [], "why": "a degraded tick recorded as a confident false"},
                ],
            },
            "proves": (
                "An outage is reported as a GAP with its reason, not as continued truth and not as a "
                "confident negative."
            ),
            "does_not_prove": "How the runtime should recover, backfill, or re-judge after the outage ends.",
        },
    ]

    return stream_doc(NAME, "frames", GENERATOR, cases), labels_doc(NAME, labels)
