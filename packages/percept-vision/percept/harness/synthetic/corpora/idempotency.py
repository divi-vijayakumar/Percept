"""Corpus 6 — IDEMPOTENCY: one real-world happening is one user-visible incident.

Two ways a system re-notifies about something the user already knows:

1. **Re-confirmation.** The condition is still true on the next tick. Nothing changed; nothing new
   should be said.
2. **Housekeeping.** The gate's key TTL sweeps a key for memory reasons while the episode is STILL
   OPEN. A bookkeeping eviction is not a real-world event, and must never surface as one — this is
   the subtle one, because the refire looks legitimate from inside the gate.

``incident_count`` is the number of user-visible incidents the whole stream may produce, and the
validator cross-checks it against the per-step ``actions``.
"""

from __future__ import annotations

from ._common import frame_step, labels_doc, obs, stream_doc

NAME = "idempotency"
GENERATOR = __name__

#: The gate key TTL the second case is labelled against (a key untouched this long is dropped).
KEY_TTL_S = 60.0
_SUBJECT = "spill:aisle-4"


def _sighting(t: float, *, evidence: str) -> dict:
    return frame_step(
        t,
        f"spill@{t:g}",
        observations=[obs("spill", content_key="aisle-4", confidence=0.95)],
        verdict={
            "satisfied": True,
            "confidence": 0.95,
            "evidence": f"scripted: {evidence} at t={t:g}",
            "subject_key": _SUBJECT,
            "binding": "resolved",
        },
    )


def build() -> tuple[dict, dict]:
    cases = [
        {
            "id": "unchanged_confirmation_does_not_reincident",
            "description": "The same spill is re-confirmed, unchanged, on two later ticks.",
            "steps": [
                _sighting(0.0, evidence="spill first seen"),
                _sighting(5.0, evidence="same spill, unchanged"),
                _sighting(10.0, evidence="same spill, unchanged"),
            ],
        },
        {
            "id": "key_ttl_expiry_while_episode_open_does_not_refire",
            "description": (
                f"The same spill persists across a {KEY_TTL_S:g}s gate key TTL — the key is swept for "
                "memory reasons while the episode is still open, then the spill is seen again."
            ),
            "steps": [
                _sighting(0.0, evidence="spill first seen"),
                _sighting(KEY_TTL_S + 1.0, evidence="still the same spill, after the TTL sweep"),
                _sighting(120.0, evidence="still the same spill"),
            ],
        },
    ]

    labels = [
        {
            "id": "unchanged_confirmation_does_not_reincident",
            "expect": {
                "incident_count": 1,
                "actions": ["fire", "silent", "silent"],
                "subject": _SUBJECT,
                "episode_open_throughout": True,
                "wrong_answer": {
                    "incident_count": 3,
                    "why": "an incident per satisfied evaluation instead of per rising edge",
                },
            },
            "proves": "Re-confirming an already-reported episode produces no second incident.",
            "does_not_prove": "That a real judge returns a stable verdict tick to tick on real footage.",
        },
        {
            "id": "key_ttl_expiry_while_episode_open_does_not_refire",
            "expect": {
                "incident_count": 1,
                "actions": ["fire", "silent", "silent"],
                "subject": _SUBJECT,
                "key_ttl_s": KEY_TTL_S,
                "ttl_sweep_between_step_indices": [0, 1],
                "episode_open_throughout": True,
                "housekeeping_is_not_an_event": True,
                "wrong_answer": {
                    "incident_count": 2,
                    "why": "the TTL sweep dropped the armed key, so the still-open episode read as a new "
                           "rising edge — a memory bound leaking into user-visible behaviour",
                },
            },
            "proves": (
                "A gate-key TTL expiry is bookkeeping: while the episode is still open it must not "
                "manufacture a second incident."
            ),
            "does_not_prove": (
                "That the TTL/max-keys bounds are correctly sized for a 24/7 stream — that is a soak "
                "question, not a capability one."
            ),
        },
    ]

    return (
        stream_doc(NAME, "frames", GENERATOR, cases, key_ttl_s=KEY_TTL_S),
        labels_doc(NAME, labels, key_ttl_s=KEY_TTL_S),
    )
