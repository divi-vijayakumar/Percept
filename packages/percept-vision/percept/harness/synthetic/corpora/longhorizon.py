"""Corpus 7 — LONG HORIZON: EpisodeLog semantics over a 15-minute SYNTHETIC stream.

READ THIS BEFORE USING THE CORPUS.

This is a **CAPABILITY fixture, not a quality benchmark**. It feeds exact timestamps and a supplied
subject key through one in-process EpisodeLog and proves that its recurrence/boundary rules produce
the expected episodes through minute 14. It does not exercise processing-window handoff, worker
restart, persistence, or subject recognition. It proves NOTHING about real-world accuracy: no real
footage, no appearance drift, no lighting change, no occlusion, no look-alike person walking
through. A green run means "one EpisodeLog handled these scripted timestamps", never "we
re-identify people over 15 minutes".

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

# A separate boundedness case uses distinct, scripted subjects throughout one session. The values are
# fixture inputs, not release-suite counts: the release track reads the cases from its manifest and
# derives its coverage from whatever the corpus contains.
BOUNDED_TIMES_S: tuple[float, ...] = (0.0, 150.0, 300.0, 450.0, 600.0, 750.0, 900.0)
BOUNDED_MAX_EPISODES = 3


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

    bounded_steps = [
        frame_step(
            t,
            f"bounded-subject-{i}@{t:g}",
            observations=[obs("person", content_key=f"bounded-{i}", confidence=0.95)],
            verdict={
                "satisfied": True,
                "confidence": 0.95,
                "evidence": f"scripted: distinct subject {i} at t={t:g}",
                "subject_key": f"person:bounded-{i}",
                "binding": "resolved",
            },
        )
        for i, t in enumerate(BOUNDED_TIMES_S)
    ]

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
        },
        {
            "id": "bounded_episode_log_over_long_session",
            "description": (
                f"A {DURATION_S / 60:g}-minute SYNTHETIC session presents one distinct scripted "
                f"subject at each labelled time while EpisodeLog is capped at "
                f"{BOUNDED_MAX_EPISODES} retained episodes."
            ),
            "synthetic_note": "Generated, not recorded. Bounded-state mechanism fixture only.",
            "steps": bounded_steps,
        },
    ]

    bounded_boundaries = [[t, t] for t in BOUNDED_TIMES_S]
    labels = [
        {
            "id": "same_subject_recurs_at_minutes_2_7_and_14",
            "expect": {
                "episode_count": 3,
                "boundaries": [[a, b] for a, b in OCCURRENCES],
                "duration_s": DURATION_S,
                "capability_fixture": True,
                "subject": SUBJECT,
                "continuity_gap_s": CONTINUITY_GAP_S,
                "separations_s": [
                    round(OCCURRENCES[1][0] - OCCURRENCES[0][1], 6),
                    round(OCCURRENCES[2][0] - OCCURRENCES[1][1], 6),
                ],
                "occurrence_step_indices": present_indices,
                "wrong_answers": [
                    {"episode_count": 1, "why": "durable key with no gap bound — three visits collapse to one"},
                    {"episode_count": 61, "why": "counting ticks"},
                    {"why": "memory evicted before minute 14, so occurrence 3 reads as a brand-new subject",
                     "symptom": "subject key differs between occurrence 1 and occurrence 3"},
                ],
            },
            "proves": (
                "ONE IN-PROCESS EPISODELOG ONLY: exact timestamps and a supplied subject key recur "
                "through minute 14 and produce the expected 3 episode boundaries."
            ),
            "does_not_prove": (
                "Processing-window handoff, worker restart, persistence, or ANY real-world accuracy. "
                "This is a generated stream with a perfect scripted judge — no appearance drift, "
                "occlusion, lighting change or look-alikes. It is NOT a quality benchmark and must "
                "never be cited as long-horizon re-identification performance."
            ),
        },
        {
            "id": "bounded_episode_log_over_long_session",
            "expect": {
                "episode_count": len(BOUNDED_TIMES_S),
                "boundaries": bounded_boundaries,
                "duration_s": DURATION_S,
                "capability_fixture": True,
                "max_episodes": BOUNDED_MAX_EPISODES,
                "retained_episode_count": BOUNDED_MAX_EPISODES,
                "dropped_episode_count": len(BOUNDED_TIMES_S) - BOUNDED_MAX_EPISODES,
                "drop_reason": "max_episodes",
                "retained_subjects": [
                    f"person:bounded-{i}"
                    for i in range(len(BOUNDED_TIMES_S) - BOUNDED_MAX_EPISODES, len(BOUNDED_TIMES_S))
                ],
            },
            "proves": (
                "Within one process session, the episode log enforces its authored count bound, "
                "retains the newest episodes, and reports each capacity eviction."
            ),
            "does_not_prove": (
                "Restart durability, storage RPO/RTO, real subject identification, or that this "
                "fixture's cap is appropriate for any production Watch."
            ),
        },
    ]

    return (
        stream_doc(NAME, "frames", GENERATOR, cases, duration_s=DURATION_S, sample_s=SAMPLE_S,
                   capability_fixture=True),
        labels_doc(NAME, labels, capability_fixture=True,
                   warning="CAPABILITY fixture on a SYNTHETIC stream — not a quality benchmark."),
    )
