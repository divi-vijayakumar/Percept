"""``apply_watch_plan`` — resolve + compile + arm a :class:`~percept.contracts.WatchPlan` in one call
(ADR-008's configuration-ownership resolution/compilation boundary, applied).

This is the structured, TRUSTED counterpart of the conductor's ``arm_from_language`` /
``apply_goal_ops`` path (``percept.agent.Percept.apply_goal_ops``): a caller hands a
:class:`~percept.contracts.watchplan.WatchPlan` (or several — see "multi-goal plan" below), it is
resolved through :class:`percept.resolver.ConfigResolver` (ownership + precedence + provenance),
compiled through :func:`percept.runtime_compiler.runtime_compile` (the exact ``Goal``/``Settings``
shapes the runtime already accepts), and ARMED via the existing, unchanged
:meth:`percept.agent.Percept.add_goal` — trust rules unchanged (``add_goal`` still stamps
``GoalOrigin.STRUCTURED_TRUSTED``, overwriting whatever the caller supplied).

**Coercion happens here, at APPLY time**, not in :mod:`percept.runtime_compiler`:
``runtime_compile`` assembles ``Goal.state``'s ``"behavior"``/``"rhythm"``/``"episode"``/``"sense"``/
``"perception"`` sub-blocks in the RAW shape :mod:`percept.behavior`'s ``coerce_authored_*`` firewall
already expects as input (the same firewall the conductor's untrusted language path runs through);
this module runs that SAME firewall over the compiled state and diffs input vs. output into
:class:`~percept.contracts.watchplan.FieldDiagnostic` entries (``coerced_fields``) instead of letting
a malformed authored value silently vanish.

**Multi-goal plans and partial application**: :func:`apply_watch_plan` accepts either
one :class:`WatchPlan` or an iterable of them (a "multi-goal plan" is a batch — each ``WatchPlan``
still names exactly one goal condition, per its own DTO shape). Each plan is resolved/compiled/armed
independently; the aggregate :class:`~percept.contracts.watchplan.WatchReceipt` reflects the batch:
``"applied"`` (every plan armed cleanly), ``"applied_with_warnings"`` (armed, but at least one loud
non-field-loss diagnostic — e.g. env overriding an explicitly-selected profile),
``"partial"`` (at least one plan in the batch was rejected, or at least one armed plan had a field
dropped/coerced, while at least one goal did arm), ``"rejected"`` (every plan in the batch was
rejected — nothing armed at all).

**Idempotency by plan_hash**: a plan whose :attr:`ResolvedWatch.plan_hash` was already
applied to THIS ``percept`` core re-arms nothing and returns the SAME receipt as the first
application — a no-op, not a re-arm, not a second goal. The applied-hash ledger is scoped to the
``percept`` core it was applied to (an attribute set dynamically on the instance, not a new
dataclass field on :class:`~percept.agent.Percept` — this module owns its own bookkeeping rather than
editing that class's frozen field list).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable

from .behavior import (
    coerce_authored_behavior,
    coerce_authored_episode,
    coerce_authored_perception,
    coerce_authored_rhythm,
    coerce_authored_sense,
)
from .contracts.watchplan import FieldDiagnostic, ReceiptStatus, WatchPlan, WatchReceipt
from .resolver import ConfigError, ConfigResolver
from .runtime_compiler import runtime_compile

#: ``Goal.state["behavior"]`` RAW-shape leaf keys that sit flat (not nested under "accumulate"/
#: "consequence") — mirrors ``coerce_authored_behavior``'s own reads.
_BEHAVIOR_LEAF_KEYS: tuple[str, ...] = ("type", "rearm", "cadence_s", "deadline_after_s", "from", "to")

#: goal.state sub-block key -> its coercer, for the four coercers whose RAW input and coerced OUTPUT
#: are both wrapped under the same sub-key (unlike "behavior", which unwraps to flat fields).
_WRAPPED_COERCERS: tuple[tuple[str, Any], ...] = (
    ("rhythm", coerce_authored_rhythm),
    ("episode", coerce_authored_episode),
    ("sense", coerce_authored_sense),
    ("perception", coerce_authored_perception),
)

#: The dynamic attribute name this module stamps on a ``Percept`` core to remember applied plan
#: hashes (idempotency ledger). Not a dataclass field on ``Percept`` — see module docstring.
_LEDGER_ATTR = "_watchplan_applied_hashes"


def _behavior_leaf_checks(raw: dict) -> list[tuple[str, str, Any]]:
    """``(diagnostic leaf path, coerced-output key to check, raw authored value)`` triples for every
    leaf ``coerce_authored_behavior`` actually reads out of ``raw`` — the RAW-vs-coerced diff basis."""
    checks: list[tuple[str, str, Any]] = []
    for key in _BEHAVIOR_LEAF_KEYS:
        if key in raw:
            checks.append((key, key, raw[key]))
    accumulate = raw.get("accumulate")
    if isinstance(accumulate, dict):
        for key in ("fire_when", "window_s"):
            if key in accumulate:
                checks.append((f"accumulate.{key}", key, accumulate[key]))
    consequence = raw.get("consequence")
    if isinstance(consequence, dict):
        for key in ("miss_cost", "false_fire_cost", "protected_floor"):
            if key in consequence:
                checks.append((f"consequence.{key}", key, consequence[key]))
    return checks


def _coerce_goal_state(state: dict, *, goal_id: str) -> tuple[dict, list[FieldDiagnostic]]:
    """Run the EXISTING ``coerce_authored_*`` firewall (:mod:`percept.behavior`) over the compiled,
    still-raw sub-blocks of ``state``, diffing input vs. output into ``FieldDiagnostic`` entries.
    Never invents a new coercion rule — every accept/drop/clamp decision is the coercer's own."""
    state = dict(state)
    diagnostics: list[FieldDiagnostic] = []

    behavior_raw = state.pop("behavior", None)
    if behavior_raw is not None:
        result = coerce_authored_behavior(behavior_raw)
        for leaf_path, out_key, raw_value in _behavior_leaf_checks(behavior_raw):
            if out_key not in result:
                diagnostics.append(FieldDiagnostic(
                    path=f"Goal.state.{leaf_path}", code="coerced_dropped", goal_id=goal_id,
                    message="authored value failed validation and was dropped.",
                ))
            elif result[out_key] != raw_value:
                diagnostics.append(FieldDiagnostic(
                    path=f"Goal.state.{leaf_path}", code="coerced_clamped", goal_id=goal_id,
                    message=f"{raw_value!r} -> {result[out_key]!r}",
                ))
        state.update(result)

    for block_key, coercer in _WRAPPED_COERCERS:
        raw = state.pop(block_key, None)
        if raw is None:
            continue
        result = coercer(raw)
        after = result.get(block_key, {})
        for key, raw_value in raw.items():
            if key not in after:
                diagnostics.append(FieldDiagnostic(
                    path=f"Goal.state.{block_key}.{key}", code="coerced_dropped", goal_id=goal_id,
                    message="authored value failed validation and was dropped.",
                ))
            elif after[key] != raw_value:
                diagnostics.append(FieldDiagnostic(
                    path=f"Goal.state.{block_key}.{key}", code="coerced_clamped", goal_id=goal_id,
                    message=f"{raw_value!r} -> {after[key]!r}",
                ))
        state.update(result)

    return state, diagnostics


def _apply_one(
    percept: Any, plan: WatchPlan, resolver: ConfigResolver, resolve_kwargs: dict
) -> tuple[str | None, WatchReceipt]:
    """Resolve + compile + arm ONE plan. Returns ``(goal_id_or_None, receipt)`` — ``goal_id`` is
    ``None`` only when the plan was rejected outright (nothing armed)."""
    try:
        resolved = resolver.resolve(plan, **resolve_kwargs)
    except ConfigError as exc:
        return None, WatchReceipt(
            status="rejected",
            dropped_fields=(FieldDiagnostic(
                path="WatchPlan", code="ownership_violation", goal_id=plan.id, message=str(exc),
            ),),
        )

    ledger: dict[str, tuple[str, WatchReceipt]] = getattr(percept, _LEDGER_ATTR, None)
    if ledger is None:
        ledger = {}
        setattr(percept, _LEDGER_ATTR, ledger)
    cached = ledger.get(resolved.plan_hash)
    if cached is not None:
        return cached  # idempotent no-op: identical plan already applied to this core

    compiled = runtime_compile(resolved)
    state, coerced = _coerce_goal_state(compiled.goal.state, goal_id=compiled.goal.id)
    goal = replace(compiled.goal, state=state)
    percept.add_goal(goal)

    base = resolver.last_receipt  # the resolution-time WatchReceipt (accepted_fields + warnings)
    warnings = base.warnings if base is not None else ()
    accepted_fields = base.accepted_fields if base is not None else ()

    status: ReceiptStatus
    if coerced:
        status = "partial"  # "some fields dropped/coerced, the plan still armed"
    elif warnings:
        status = "applied_with_warnings"
    else:
        status = "applied"

    receipt = WatchReceipt(
        status=status,
        watch_ids=(goal.id,),
        accepted_fields=accepted_fields,
        coerced_fields=tuple(coerced),
        warnings=warnings,
    )
    ledger[resolved.plan_hash] = (goal.id, receipt)
    return goal.id, receipt


def apply_watch_plan(
    percept: Any,
    plans: "WatchPlan | Iterable[WatchPlan]",
    *,
    resolver: ConfigResolver | None = None,
    source_profile: Any = None,
    source_profile_explicit: bool = True,
    domain_pack: Any = None,
    execution_policy: Any = None,
    optimization_lock: Any = None,
    situation: tuple[Any, Any, Any] | None = None,
    registry: Any = None,
    settings: Any = None,
) -> WatchReceipt:
    """Resolve, compile, and arm ``plans`` on ``percept`` — see module docstring for the full
    semantics (coercion-at-apply, multi-goal batching, idempotency).

    ``percept`` is a :class:`~percept.agent.Percept` core (typed ``Any`` here to avoid importing
    ``percept.agent`` — this module is a leaf the agent module can import FROM, never the reverse).
    Every keyword beyond ``resolver``/``plans`` is passed straight through to
    :meth:`percept.resolver.ConfigResolver.resolve` for EVERY plan in the batch (the same six-object
    configuration inputs apply to every goal a single call arms).
    """
    resolver = resolver if resolver is not None else ConfigResolver()
    resolve_kwargs = dict(
        source_profile=source_profile,
        source_profile_explicit=source_profile_explicit,
        domain_pack=domain_pack,
        execution_policy=execution_policy,
        optimization_lock=optimization_lock,
        situation=situation,
        registry=registry,
        settings=settings,
    )
    plan_list = [plans] if isinstance(plans, WatchPlan) else list(plans)
    if not plan_list:
        raise ValueError("apply_watch_plan requires at least one WatchPlan")

    watch_ids: list[str] = []
    accepted: list[str] = []
    dropped: list[FieldDiagnostic] = []
    coerced: list[FieldDiagnostic] = []
    warnings: list[FieldDiagnostic] = []
    n_rejected = 0
    n_partial = 0
    n_warned = 0

    for plan in plan_list:
        goal_id, receipt = _apply_one(percept, plan, resolver, resolve_kwargs)
        if goal_id is not None:
            watch_ids.append(goal_id)
        accepted.extend(receipt.accepted_fields)
        dropped.extend(receipt.dropped_fields)
        coerced.extend(receipt.coerced_fields)
        warnings.extend(receipt.warnings)
        if receipt.status == "rejected":
            n_rejected += 1
        elif receipt.status == "partial":
            n_partial += 1
        elif receipt.status == "applied_with_warnings":
            n_warned += 1

    total = len(plan_list)
    status: ReceiptStatus
    if n_rejected == total:
        status = "rejected"
    elif n_rejected > 0 or n_partial > 0:
        status = "partial"
    elif n_warned > 0:
        status = "applied_with_warnings"
    else:
        status = "applied"

    return WatchReceipt(
        status=status,
        watch_ids=tuple(watch_ids),
        accepted_fields=tuple(dict.fromkeys(accepted)),  # dedupe, preserve first-seen order
        dropped_fields=tuple(dropped),
        coerced_fields=tuple(coerced),
        warnings=tuple(warnings),
    )


__all__ = ["apply_watch_plan"]
