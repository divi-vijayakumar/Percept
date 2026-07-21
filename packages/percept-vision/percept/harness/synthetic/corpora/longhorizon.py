"""Corpus 7 — LONG HORIZON: episode memory over a 15-minute SYNTHETIC stream.

READ THIS BEFORE USING THE CORPUS.

This is a **CAPABILITY fixture, not a quality benchmark**. It proves the MECHANISM can link
occurrence 1 (minute ~2) to occurrence 3 (minute ~14) across two long absences — that the memory
survives the horizon and the subject key is still the same key twelve minutes later. It proves
NOTHING about real-world accuracy: no real footage, no appearance drift, no lighting change, no
occlusion, no look-alike person walking through. A green run here means "the plumbing reaches
minute 14", never "we re-identify people over 15 minutes".

Real long-horizon accuracy belongs in the end-to-end release benchmark on real footage. (See also
the standing note that episode memory should not be evaluated on short clips — it is for
long-running, multi-occurrence tasks, which is exactly what this stream simulates.)
"""

from __future__ import annotations

import random

from ..schema import SEED
from ._common import CONTINUITY_GAP_S, frame_step, labels_doc, obs, stream_doc

NAME = "longhorizon"
GENERATOR = __name__

DURATION_S = 900.0          # 15 minutes
SAMPLE_S = 15.0             # one look every 15s -> 61 steps
SUBJECT = "person:badge-7741"

#: The three occurrences, at roughly minutes 2, 7 and 14.
OCCURRENCES: list[tuple[float, float]] = [(120.0, 150.0), (420.0, 450.0), (840.0, 870.0)]


def build() -> tuple[dict, dict]:
    # Seeded, never wall-clock: the only randomness is confidence jitter, drawn from SEED so the
    # generated JSON is byte-identical on every machine and every run.
    rng = random.Random(SEED)

    steps: list[dict] = []
    present_indices: list[list[int]] = [[] for _ in OCCURRENCES]
    ticks = int(DURATION_S / SAMPLE_S)
    for i in range(ticks + 1):
        t = round(i * SAMPLE_S, 6)
        occ = next((k for k, (a, b) in enumerate(OCCURRENCES) if a <= t <= b), None)
        present = occ is not None
        if present:
            present_indices[occ].append(i)
        conf = round(0.80 + rng.random() * 0.15, 4) if present else round(0.88 + rng.random() * 0.08, 4)
        steps.append(
            frame_step(
                t,
                f"{'badge7741' if present else 'empty'}@{t:g}",
                observations=[obs("person", content_key="badge-7741", confidence=conf)] if present else [],
                verdict={
                    "satisfied": present,
                    "confidence": conf,
                    "evidence": f"scripted: {SUBJECT if present else 'nobody'} at t={t:g}",
                    "subject_key": SUBJECT if present else None,
                    "binding": "resolved" if present else "unknown",
                },
            )
        )

    cases = [
        {
            "id": "same_subject_recurs_at_minutes_2_7_and_14",
            "description": (
                f"A {DURATION_S / 60:g}-minute SYNTHETIC stream sampled every {SAMPLE_S:g}s. The same "
                f"subject ({SUBJECT}) appears three times — around minutes 2, 7 and 14 — separated by "
                "long absences."
            ),
            "synthetic_note": "Generated, not recorded. Capability fixture only.",
            "steps": steps,
        }
    ]

    labels = [
        {
            "id": "same_subject_recurs_at_minutes_2_7_and_14",
            "expect": {
                "episode_count": 3,
                "boundaries": [[a, b] for a, b in OCCURRENCES],
                "duration_s": DURATION_S,
                "capability_fixture": True,
                "subject": SUBJECT,
                "same_subject_key_all_occurrences": True,
                "continuity_gap_s": CONTINUITY_GAP_S,
                "separations_s": [
                    round(OCCURRENCES[1][0] - OCCURRENCES[0][1], 6),
                    round(OCCURRENCES[2][0] - OCCURRENCES[1][1], 6),
                ],
                "occurrence_step_indices": present_indices,
                "links_first_to_last": {"from_t": OCCURRENCES[0][0], "to_t": OCCURRENCES[2][0],
                                        "span_s": OCCURRENCES[2][0] - OCCURRENCES[0][0]},
                "wrong_answers": [
                    {"episode_count": 1, "why": "durable key with no gap bound — three visits collapse to one"},
                    {"episode_count": 61, "why": "counting ticks"},
                    {"why": "memory evicted before minute 14, so occurrence 3 reads as a brand-new subject",
                     "symptom": "subject key differs between occurrence 1 and occurrence 3"},
                ],
            },
            "proves": (
                "THE MECHANISM ONLY: episode memory retains the subject across a >=10-minute horizon and "
                "links occurrence 1 to occurrence 3 with the same durable key, producing exactly 3 episodes."
            ),
            "does_not_prove": (
                "ANY real-world accuracy. This is a generated stream with a perfect scripted judge — no "
                "appearance drift, occlusion, lighting change or look-alikes. It is NOT a quality "
                "benchmark and must never be cited as long-horizon re-identification performance."
            ),
        }
    ]

    return (
        stream_doc(NAME, "frames", GENERATOR, cases, duration_s=DURATION_S, sample_s=SAMPLE_S,
                   capability_fixture=True),
        labels_doc(NAME, labels, capability_fixture=True,
                   warning="CAPABILITY fixture on a SYNTHETIC stream — not a quality benchmark."),
    )
