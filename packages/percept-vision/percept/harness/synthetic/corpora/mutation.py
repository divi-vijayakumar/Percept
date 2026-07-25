"""Corpus 3 — MUTATION / SPEECH ACT: containing a value is not the same as setting it.

The conversational case, so this corpus is TURNS, not frames. Four of the five utterances in the
main case contain the string ``$25K``; exactly ONE of them changes the goal's value. A system that
extracts values by pattern-matching the utterance gets four wrong answers out of five.

The three ways a value can be present without being asserted:

* a **question** — "Could we do $25K?" (proposed, not stated)
* an **attribution** — "Another customer paid $25K" (someone else's number)
* a **negation** — "The budget is not $25K" (explicitly denied)

Plus the two acts that change state without proposing a number at all: a valueless **retract**, and
a **replace** blocked because the speaker lacks AUTHORITY for the goal's condition.
"""

from __future__ import annotations

from ._common import labels_doc, stream_doc, turn_step

NAME = "mutation"
GENERATOR = __name__

#: The goal every case in this corpus is evaluated against. Note the explicit speaker-side condition:
#: authority is part of the condition, not an afterthought.
CONDITION = "a BUYER-side speaker states the approved budget"


def build() -> tuple[dict, dict]:
    cases = [
        {
            "id": "only_one_of_four_25k_utterances_replaces",
            "description": (
                "Five buyer turns. Four contain '$25K'; only the last one — 'Actually, the approved "
                "budget is $25K' — replaces the value."
            ),
            "condition": CONDITION,
            "steps": [
                turn_step(0.0, "buyer", "The budget is $40K"),
                turn_step(12.0, "buyer", "Could we do $25K?"),
                turn_step(24.0, "buyer", "Another customer paid $25K"),
                turn_step(36.0, "buyer", "The budget is not $25K"),
                turn_step(48.0, "buyer", "Actually, the approved budget is $25K"),
            ],
        },
        {
            "id": "valueless_retraction_clears_the_value",
            "description": "A buyer asserts $40K and then retracts it without naming any replacement value.",
            "condition": CONDITION,
            "steps": [
                turn_step(0.0, "buyer", "The budget is $40K"),
                turn_step(15.0, "buyer", "Scratch what I said about the budget"),
            ],
        },
        {
            "id": "seller_lacks_authority_to_replace_buyer_value",
            "description": (
                "The buyer states $40K; the SELLER then utters a textbook replace ('Actually, the "
                "approved budget is $25K') on a goal whose condition requires a buyer-side speaker."
            ),
            "condition": CONDITION,
            "steps": [
                turn_step(0.0, "buyer", "The budget is $40K"),
                turn_step(20.0, "seller", "Actually, the approved budget is $25K"),
            ],
        },
    ]

    labels = [
        {
            "id": "only_one_of_four_25k_utterances_replaces",
            "expect": {
                "condition": CONDITION,
                "value_before": None,
                "acts": ["assert", "mention", "mention", "negation", "replace"],
                "value_after": ["$40K", "$40K", "$40K", "$40K", "$25K"],
                "final_value": "$25K",
                "supersedes": {"turn_index": 4, "supersedes_turn_index": 0, "superseded_value": "$40K"},
                "utterances_containing_value": [1, 2, 3, 4],
                "utterances_mutating_value": [4],
                "authorized": [True, True, True, True, True],
                "claim_id": "budget",
                "revision_chain": [
                    {"turn_index": 0, "revision_id": "budget#r1", "supersedes": None},
                    {"turn_index": 4, "revision_id": "budget#r2", "supersedes": "budget#r1"},
                ],
                "wrong_answer": {
                    "final_value": "$25K after turn 1",
                    "why": "value extracted by pattern-match on the utterance, with no speech act",
                },
            },
            "proves": (
                "Containing a value is not asserting it: a question, an attribution and a negation all "
                "leave the goal's value UNCHANGED, and only an explicit replacement supersedes it."
            ),
            "does_not_prove": (
                "That a real model classifies these speech acts correctly — act classification accuracy "
                "is a statistical question for the release benchmark. This fixes only the state machine "
                "downstream of the classification."
            ),
        },
        {
            "id": "valueless_retraction_clears_the_value",
            "expect": {
                "condition": CONDITION,
                "value_before": None,
                "acts": ["assert", "retract"],
                "value_after": ["$40K", None],
                "final_value": None,
                "authorized": [True, True],
                "claim_id": "budget",
                "revision_chain": [
                    {"turn_index": 0, "revision_id": "budget#r1", "supersedes": None},
                    {"turn_index": 1, "revision_id": None, "supersedes": "budget#r1"},
                ],
                "retraction_is_valueless": True,
                "wrong_answer": {
                    "final_value": "$40K",
                    "why": "a mutation model that can only overwrite with a new value cannot express "
                           "'the value is now unknown'",
                },
            },
            "proves": "Retraction is a first-class operation: a turn with NO value can still clear one.",
            "does_not_prove": "Whether the prior value should be restorable, or how long a retraction persists.",
        },
        {
            "id": "seller_lacks_authority_to_replace_buyer_value",
            "expect": {
                "condition": CONDITION,
                "value_before": None,
                "acts": ["assert", "replace"],
                "value_after": ["$40K", "$40K"],
                "final_value": "$40K",
                "authority_required": "buyer",
                "authorized": [True, False],
                "blocked_by_authority": [False, True],
                "claim_id": "budget",
                "revision_chain": [
                    {"turn_index": 0, "revision_id": "budget#r1", "supersedes": None},
                ],
                "wrong_answer": {
                    "final_value": "$25K",
                    "why": "the utterance IS a replace; without a speaker-authority check the state machine "
                           "applies it",
                },
            },
            "proves": (
                "A well-formed replace from an UNAUTHORIZED speaker must not mutate the value — authority "
                "is evaluated separately from the speech act."
            ),
            "does_not_prove": "That speaker diarization/role attribution is reliable on real audio.",
        },
    ]

    return stream_doc(NAME, "turns", GENERATOR, cases, condition=CONDITION), labels_doc(NAME, labels)
