"""Claim-mode state machine — a value DIFFERENCE is not a REPLACEMENT (card Y2).

The two paying client teams that ran percept 0.7.0 could PERCEIVE correctly — their judges scored
recall 1.00 on documents, handled retractions and counterfactuals — but the SDK gave the executor no
safe way to turn a stream of value-bearing findings into a single authoritative value. So the
executor was left to parse values out of free-text ``evidence`` (unstable, vendor-dependent) and,
worse, had no way to tell a value that merely APPEARS ("Could we do $25K?") from one that is ASSERTED
("The budget is $40K") from one that REPLACES ("Actually, the approved budget is $25K"). Four of the
five decisive utterances contain ``$25K``; exactly ONE of them changes the value.

This module folds the :data:`~percept.contracts.Operation` speech-act channel (which the judge emits
LOCALLY, from the current turn alone — the prior value never enters the satisfaction prompt, which is
what preserves the anchoring mitigation) into an authoritative value, with two hard rules:

1. **Containing a value is not asserting it.** ``mention``/``confirm`` leave the value untouched;
   ``assert`` ESTABLISHES an empty slot but never overwrites an established value; only ``replace``
   supersedes, and only ``retract``/``close`` clear it. A value difference is therefore never mistaken
   for a replacement — the speech act decides, not a diff of the numbers.

2. **Authority is evaluated separately from the speech act.** A well-formed ``replace`` from a speaker
   who lacks authority for the concern (a seller against a buyer-authoritative budget) is DOWNGRADED
   to a non-mutating mention — the value does not move — and the downgrade is TRACED
   (``claim.authority_downgrade``), never silent. See :class:`~percept.contracts.ConcernAuthority`.

The state machine is a pure fold over ``(operation, value, asserted_by)``; the only side effect is the
authority-downgrade trace line, which a caller may silence with ``trace=False``. It knows nothing
about the gate, and the gate reads nothing from it: this is the value channel, GATE-IRRELEVANT by
construction, exactly like every other card-Y2 Verdict addition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .behavior import revision_id
from .contracts import ConcernAuthority, Operation, Verdict
from .observability import get_logger, log

__all__ = [
    "ClaimOutcome",
    "ClaimState",
    "apply_operation",
    "fold_verdicts",
    "VALUE_SETTING",
    "VALUE_CLEARING",
]

#: Acts that WRITE the slot value (a subset of :data:`~percept.contracts.OPERATIONS`). ``assert``
#: establishes an empty slot; ``replace`` supersedes; ``retract``/``close`` clear. ``confirm``/``mention``
#: never write. Distinct from :data:`~percept.contracts.MUTATING_OPERATIONS` (the authority-gated set):
#: ``assert`` writes but is NOT gated, because it can only establish an empty slot, never overwrite.
VALUE_SETTING: frozenset[str] = frozenset({"assert", "replace"})
VALUE_CLEARING: frozenset[str] = frozenset({"retract", "close"})


@dataclass(frozen=True)
class ClaimOutcome:
    """The typed result of folding ONE operation into a :class:`ClaimState` — the WHY, made explicit.

    - ``requested`` — the operation the judge emitted.
    - ``effective`` — what was actually applied (equals ``requested`` unless an authority downgrade
      rewrote a mutating op to ``mention``).
    - ``value_after`` / ``normalized_after`` — the slot value after this operation.
    - ``applied`` — did the slot value actually change (a real write)?
    - ``downgraded`` — was a mutating op refused for lack of authority (then ``effective == 'mention'``)?
    - ``supersedes`` — for an applied ``replace``/``retract``/``close``, the revision id it acted on
      (the prior authoritative revision), or ``None`` when there was nothing to supersede.
    - ``revision_id`` — the NEW revision minted by an applied ``assert``/``replace``, or ``None``.
    """

    requested: Operation
    effective: Operation
    value_after: Any
    normalized_after: Any
    applied: bool
    downgraded: bool
    supersedes: str | None
    revision_id: str | None


@dataclass
class ClaimState:
    """The authoritative value of ONE claim slot, folded from a stream of speech acts (card Y2).

    ``value`` is ``None`` when the slot is genuinely empty — never ``""`` (an empty string is a value,
    not an absence, and conflating them is exactly the bug the retraction case guards against). The
    revision ledger (``revisions`` / ``revision_id``) mints ``<claim_id>#r<n>`` on each write so a
    ``replace`` can name the revision it superseded and a downstream episode revision lines up.
    """

    claim_id: str = "claim"
    value: Any = None
    normalized_value: Any = None
    revisions: int = 0
    revision_id: str | None = None

    def apply(
        self,
        operation: Operation,
        *,
        value: Any = None,
        normalized_value: Any = None,
        asserted_by: "str | None" = None,
        authority: "ConcernAuthority | None" = None,
        trace: bool = True,
    ) -> ClaimOutcome:
        """Fold one operation into this state (in place). Delegates to :func:`apply_operation`."""
        return apply_operation(
            self,
            operation,
            value=value,
            normalized_value=normalized_value,
            asserted_by=asserted_by,
            authority=authority,
            trace=trace,
        )


def apply_operation(
    state: ClaimState,
    operation: Operation,
    *,
    value: Any = None,
    normalized_value: Any = None,
    asserted_by: "str | None" = None,
    authority: "ConcernAuthority | None" = None,
    trace: bool = True,
) -> ClaimOutcome:
    """Fold one operation into ``state`` (in place); return its typed :class:`ClaimOutcome`.

    AUTHORITY FIRST: a mutating op (``replace``/``retract``/``close``) from an asserter the concern's
    :class:`~percept.contracts.ConcernAuthority` does not permit is DOWNGRADED to ``mention`` and traced
    (unless ``trace=False``). VALUE FOLD SECOND, on the EFFECTIVE op:

    * ``assert``  — establish the slot IFF empty (never overwrite; overwriting is ``replace``'s job).
    * ``replace`` — supersede the current revision with the stated value.
    * ``retract``/``close`` — clear the slot to ``None`` (valueless).
    * ``confirm``/``mention`` — leave the value untouched.
    """
    requested = operation
    effective = operation
    downgraded = False
    if authority is not None and not authority.permits(asserted_by, operation):
        effective = "mention"
        downgraded = True
        if trace:
            log(
                get_logger("percept.claims"),
                "claim.authority_downgrade",
                claim_id=state.claim_id,
                requested=requested,
                effective=effective,
                asserted_by=asserted_by,
                authorized=sorted(authority.asserters),
                value=value,
            )

    before = state.value
    supersedes: str | None = None
    minted: str | None = None

    if effective == "assert":
        if state.value is None:  # establish ONLY into an empty slot — never overwrite
            state.value = value
            state.normalized_value = normalized_value
    elif effective == "replace":
        supersedes = state.revision_id  # the revision being superseded (None if never established)
        state.value = value
        state.normalized_value = normalized_value
    elif effective in VALUE_CLEARING:
        supersedes = state.revision_id  # the revision being withdrawn
        state.value = None
        state.normalized_value = None
    # confirm / mention (and any downgraded mutating op) leave the value untouched.

    applied = state.value != before
    if applied and effective in VALUE_SETTING:
        state.revisions += 1
        minted = revision_id(state.claim_id, state.revisions)
        state.revision_id = minted
    elif applied and effective in VALUE_CLEARING:
        state.revision_id = None  # a withdrawal ends the revision line; next establish starts fresh

    return ClaimOutcome(
        requested=requested,
        effective=effective,
        value_after=state.value,
        normalized_after=state.normalized_value,
        applied=applied,
        downgraded=downgraded,
        supersedes=supersedes,
        revision_id=minted,
    )


def fold_verdicts(
    verdicts: "list[Verdict]",
    *,
    claim_id: str = "claim",
    authority: "ConcernAuthority | None" = None,
    trace: bool = True,
) -> "tuple[ClaimState, list[ClaimOutcome]]":
    """Fold a whole sequence of :class:`~percept.contracts.Verdict`\\ s into one authoritative value.

    Each verdict contributes its ``operation``/``value``/``normalized_value``/``asserted_by``. Returns
    the final :class:`ClaimState` and the per-verdict :class:`ClaimOutcome` list (aligned 1:1), so a
    caller gets both the final value and the full, explainable trace of how it got there."""
    state = ClaimState(claim_id=claim_id)
    outcomes = [
        apply_operation(
            state,
            v.operation,
            value=v.value,
            normalized_value=v.normalized_value,
            asserted_by=v.asserted_by,
            authority=authority,
            trace=trace,
        )
        for v in verdicts
    ]
    return state, outcomes
