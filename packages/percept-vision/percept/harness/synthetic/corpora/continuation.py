"""Corpus 2 — CONTINUATION vs NEW EPISODE: the distinction the current design collapses.

Same subject, seen twice. Is that one thing still happening, or the same thing happening AGAIN?
The answer decides whether the user is told once or twice, and the design currently has no seam that
separates the two. Every case here is the same shape — a subject re-appearing — and the labelled
answer differs only by *when*, and by *what happened to some other subject in between*.

``continuity_gap_s`` is stated in every expectation so the boundary rule is pinned by the fixture,
not inherited from whatever default the implementation happens to pick.
"""

from __future__ import annotations

from ._common import CONTINUITY_GAP_S, frame_step, labels_doc, obs, stream_doc

NAME = "continuation"
GENERATOR = __name__

_A = "po:PO-88"
_B = "po:PO-91"


def _sight(t: float, subject: str, *, satisfied: bool = True, confidence: float = 0.92,
           token: str | None = None) -> dict:
    short = subject.split(":")[-1]
    return frame_step(
        t,
        token or f"{short}@{t:g}",
        observations=[obs("purchase_order", content_key=short, confidence=confidence)],
        verdict={
            "satisfied": satisfied,
            "confidence": confidence,
            "evidence": f"scripted: {subject} visible at t={t:g}",
            "subject_key": subject,
            "binding": "resolved",
        },
    )


def build() -> tuple[dict, dict]:
    gap = CONTINUITY_GAP_S

    cases = [
        {
            "id": "same_subject_inside_gap_is_one_episode",
            "description": f"PO-88 seen at t=0 and t=10 — 10s apart, inside the {gap:g}s continuity gap.",
            "steps": [_sight(0.0, _A), _sight(10.0, _A)],
        },
        {
            "id": "same_subject_after_gap_is_two_episodes",
            "description": f"PO-88 seen at t=0 and t=45 — 45s apart, past the {gap:g}s gap (closed, then reopened).",
            "steps": [_sight(0.0, _A), _sight(45.0, _A)],
        },
        {
            "id": "interleaved_a_b_a_continues_a",
            "description": "PO-88, then PO-91, then PO-88 again — all inside the continuity gap.",
            "steps": [_sight(0.0, _A), _sight(5.0, _B), _sight(10.0, _A)],
        },
        {
            "id": "b_stuck_in_ask_must_not_refire_a",
            "description": (
                "PO-88 fires. PO-91 opens a NEW episode but its evidence is weak — it becomes ASK and "
                "never fires. PO-88 then returns inside its own continuity gap."
            ),
            "steps": [
                _sight(0.0, _A),
                _sight(5.0, _B, confidence=0.41),
                _sight(10.0, _B, confidence=0.44),
                _sight(15.0, _A),
            ],
        },
    ]

    labels = [
        {
            "id": "same_subject_inside_gap_is_one_episode",
            "expect": {
                "continuity_gap_s": gap,
                "episode_count": 1,
                "episodes": [{"subject": _A, "start_t": 0.0, "end_t": 10.0, "step_indices": [0, 1], "continued": True}],
                "actions": ["fire", "silent"],
                "fire_count": 1,
                "wrong_answer": {"episode_count": 2, "why": "treating every sighting as an event"},
            },
            "proves": "A re-sighting inside the gap CONTINUES the open episode and does not re-notify.",
            "does_not_prove": "That 30s is the right gap for any real domain — the gap is a policy input, not a fact.",
        },
        {
            "id": "same_subject_after_gap_is_two_episodes",
            "expect": {
                "continuity_gap_s": gap,
                "episode_count": 2,
                "episodes": [
                    {"subject": _A, "start_t": 0.0, "end_t": 0.0, "step_indices": [0], "continued": False},
                    {"subject": _A, "start_t": 45.0, "end_t": 45.0, "step_indices": [1], "continued": False},
                ],
                "actions": ["fire", "fire"],
                "fire_count": 2,
                "wrong_answer": {
                    "episode_count": 1,
                    "why": "durable-key dedup with no time bound — the same key is forever one episode, "
                           "so a PO that closed and reopened is never reported again",
                },
            },
            "proves": (
                "A durable subject key must NOT imply permanent suppression: past the gap, the same "
                "subject is a NEW episode and fires again."
            ),
            "does_not_prove": "Whether a real reopened PO/person/door is semantically the same thing to a user.",
        },
        {
            "id": "interleaved_a_b_a_continues_a",
            "expect": {
                "continuity_gap_s": gap,
                "episode_count": 2,
                "episodes": [
                    {"subject": _A, "start_t": 0.0, "end_t": 10.0, "step_indices": [0, 2], "continued": True},
                    {"subject": _B, "start_t": 5.0, "end_t": 5.0, "step_indices": [1], "continued": False},
                ],
                "actions": ["fire", "fire", "silent"],
                "fire_count": 2,
                "spurious_third_fire_forbidden": True,
                "wrong_answer": {
                    "fire_count": 3,
                    "why": "episode state keyed by recency rather than by subject — B displaces A, so A's "
                           "return looks new",
                },
            },
            "proves": "Episode state is per-SUBJECT: an interleaved subject does not displace A's open episode.",
            "does_not_prove": "That a real judge attributes each sighting to the right subject in the first place.",
        },
        {
            "id": "b_stuck_in_ask_must_not_refire_a",
            "expect": {
                "continuity_gap_s": gap,
                "episode_count": 2,
                "episodes": [
                    {"subject": _A, "start_t": 0.0, "end_t": 15.0, "step_indices": [0, 3], "continued": True},
                    {"subject": _B, "start_t": 5.0, "end_t": 10.0, "step_indices": [1, 2], "continued": True},
                ],
                "actions": ["fire", "ask", "silent", "silent"],
                "fire_count": 1,
                "subject_states_independent": True,
                "wrong_answer": {
                    "fire_count": 2,
                    "why": "duplicate suppression governed by a single global 'last evaluation' — B's "
                           "unresolved ASK clears the armed state, so A refires on return",
                },
            },
            "proves": (
                "One subject's evaluation state (a pending ASK / failed dwell) must not govern another "
                "subject's duplicate suppression."
            ),
            "does_not_prove": "That the ASK threshold or dwell window is correctly tuned for any real stream.",
        },
    ]

    return stream_doc(NAME, "frames", GENERATOR, cases, continuity_gap_s=gap), labels_doc(
        NAME, labels, continuity_gap_s=gap
    )
