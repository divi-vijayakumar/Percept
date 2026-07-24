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

from dataclasses import dataclass, replace
from typing import Any, Iterable

from .behavior import (
    coerce_authored_behavior,
    coerce_authored_episode,
    coerce_authored_perception,
    coerce_authored_rhythm,
    coerce_authored_sense,
)
from .contracts.watchplan import FieldDiagnostic, ReceiptStatus, WatchPlan, WatchReceipt
from .deployment_judges import load_deployment_judges
from .judge_context import JudgeContextError, JudgeContextStore
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

#: The class-attribute marker a vision backend sets to declare it PARSES the claim-mode correction
#: channel (operation/value). The three real backends (claude/gemini/openrouter) and the L1 fakes
#: declare it; any other selected backend (incl. a client's custom one) cannot fold a live correction,
#: so arming a ``claim_slot`` plan on it earns a (non-rejecting) receipt warning.
_CLAIM_PARSING_MARKER = "PARSES_CLAIM_CHANNEL"


def _claim_backend_warning(percept: Any, goal: Any) -> "FieldDiagnostic | None":
    """A receipt WARNING (never a rejection) when ``goal`` arms ``identity_policy='claim_slot'`` but the
    core's selected vision backend does not parse the correction channel.

    The claim/supersession fold is fully wired, but it only mutates when a backend emits
    ``operation``/``value`` — a backend that does not (no :data:`_CLAIM_PARSING_MARKER`) will still
    arm and run, silently never producing a revision chain. That is exactly a loud-but-non-fatal
    condition: warn, don't reject."""
    if goal.state.get("identity_policy") != "claim_slot":
        return None
    vision = getattr(percept, "vision", None)
    if vision is None or getattr(vision, _CLAIM_PARSING_MARKER, False):
        return None
    return FieldDiagnostic(
        path="Goal.state.identity_policy",
        code="claim_mode_backend_unverified",
        goal_id=goal.id,
        message=(
            f"identity_policy='claim_slot' needs a claim-channel-parsing vision backend "
            f"(claude/gemini/openrouter) to fold live corrections into a revision chain; the selected "
            f"{type(vision).__name__} does not parse operation/value, so corrections will not supersede. "
            "The plan is armed — this is a capability warning, not a rejection."
        ),
    )


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


class _JudgePrepareError(Exception):
    """A judge PREPARE-phase failure — adapter construction or static-context verification.

    Raised INSIDE Phase 1 (prepare) so :func:`apply_watch_plan` commits NOTHING: an atomic apply
    arms every goal or none. Carries a structured :class:`FieldDiagnostic` the batch surfaces on the
    rejected :class:`WatchReceipt` (the file's existing diagnostic idiom), not a bare string.
    """

    def __init__(self, diagnostic: FieldDiagnostic) -> None:
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic


@dataclass
class _PreparedGoal:
    """One goal fully prepared in Phase 1 — resolved/compiled/coerced + its judge binding (if any),
    with the receipt it will land. Holds NO runtime mutation: nothing is armed until Phase 2 commit."""

    goal: Any
    binding: Any | None  # a BoundJudge-shaped object (``.spec``/``.adapter``/``.static_context``), or None
    plan_hash: str
    receipt: WatchReceipt


def _prepare_judge_binding(
    resolved_judge: Any, percept: Any, deployment_judges: Any, *, goal_id: str | None
) -> Any:
    """Build a BoundJudge-shaped binding for a resolved judge WITHOUT mutating runtime state.

    Lazily imports :mod:`percept.judge_registry` (a sibling built by the parallel wave — NOT importable
    at module top until the merge) so a plan that requests no judge never touches it. Constructs the
    adapter via ``get_or_create_adapter(spec, cfg)`` and, when the judge binds a static context, verifies
    the exact bytes through :class:`percept.judge_context.JudgeContextStore` against the resolved digest.
    ANY failure raises :class:`_JudgePrepareError` so the whole batch arms nothing.
    """
    try:
        from . import judge_registry  # noqa: PLC0415 — LAZY: only when a judge was actually requested.
    except ImportError as exc:
        raise _JudgePrepareError(FieldDiagnostic(
            path="ResolvedWatch.judge", code="judge_registry_unavailable", goal_id=goal_id,
            message=f"a judge was requested but percept.judge_registry is not importable: {exc}",
        )) from exc

    cfg = getattr(percept, "settings", None)
    try:
        adapter = judge_registry.get_or_create_adapter(resolved_judge, cfg)
    except Exception as exc:  # noqa: BLE001 — any factory/config failure fails closed, arms nothing.
        raise _JudgePrepareError(FieldDiagnostic(
            path="ResolvedWatch.judge", code="judge_adapter_unavailable", goal_id=goal_id,
            message=f"judge adapter could not be constructed: {exc}",
        )) from exc

    static_context = None
    if resolved_judge.context_ref is not None:
        if deployment_judges is None:
            deployment_judges = load_deployment_judges()
        root = getattr(deployment_judges, "context_store_root", None)
        if root is None:
            raise _JudgePrepareError(FieldDiagnostic(
                path="ResolvedWatch.judge.context_ref", code="judge_context_store_unconfigured",
                goal_id=goal_id,
                message="judge binds a static context but the deployment has no context_store_root",
            ))
        try:
            static_context = JudgeContextStore.from_dir(root).resolve(
                resolved_judge.context_ref, resolved_judge.context_sha256 or "",
            )
        except JudgeContextError as exc:
            raise _JudgePrepareError(FieldDiagnostic(
                path="ResolvedWatch.judge.context_ref", code="judge_context_unverified",
                goal_id=goal_id, message=f"judge static context failed verification: {exc}",
            )) from exc

    return judge_registry.BoundJudge(spec=resolved_judge, adapter=adapter, static_context=static_context)


def _prepare_one(
    percept: Any,
    plan: WatchPlan,
    resolver: ConfigResolver,
    resolve_kwargs: dict,
    deployment_judges: Any,
    known: dict,
) -> tuple[str, Any]:
    """Phase 1 PREPARE for ONE plan — resolve + compile + coerce + build the judge binding, WITHOUT any
    runtime mutation. Returns one of:

    * ``("rejected", receipt)`` — a resolution ownership violation (per-plan; the batch continues);
    * ``("cached", (goal_id, receipt))`` — an idempotent no-op (this plan_hash already applied, or a
      within-batch duplicate);
    * ``("prepared", _PreparedGoal)`` — ready to commit.

    Raises :class:`_JudgePrepareError` on a judge prepare failure (aborts the WHOLE batch)."""
    try:
        resolved = resolver.resolve(plan, **resolve_kwargs)
    except ConfigError as exc:
        return "rejected", WatchReceipt(
            status="rejected",
            dropped_fields=(FieldDiagnostic(
                path="WatchPlan", code="ownership_violation", goal_id=plan.id, message=str(exc),
            ),),
        )

    cached = known.get(resolved.plan_hash)
    if cached is not None:
        return "cached", cached  # idempotent no-op: identical plan already applied / duplicated in-batch

    compiled = runtime_compile(resolved)
    state, coerced = _coerce_goal_state(compiled.goal.state, goal_id=compiled.goal.id)
    goal = replace(compiled.goal, state=state)

    # JUDGE PREPARE (no runtime mutation): construct the adapter + verify any static context and build
    # the binding NOW, so a bad judge never arms a goal. resolved.judge is None for every judge-less plan
    # (the default), so this whole branch is skipped and judge_registry is never imported.
    binding = None
    if getattr(resolved, "judge", None) is not None:
        binding = _prepare_judge_binding(resolved.judge, percept, deployment_judges, goal_id=goal.id)

    base = resolver.last_receipt  # the resolution-time WatchReceipt (accepted_fields + warnings)
    warnings = base.warnings if base is not None else ()
    accepted_fields = base.accepted_fields if base is not None else ()

    # Capability warning assembled HERE (watchplan_apply's own channel — NOT resolver.py's binding
    # function): a claim_slot plan armed on a non-parsing vision backend can never form a revision chain.
    backend_warning = _claim_backend_warning(percept, goal)
    if backend_warning is not None:
        warnings = (*warnings, backend_warning)

    # Judge provenance echoed onto the receipt (additive; all None when no judge was requested).
    judge_fields: dict[str, Any] = {}
    if getattr(resolved, "judge", None) is not None:
        j = resolved.judge
        judge_fields = dict(
            judge_requested=j.requested_ref, judge_resolved=j.resolved_ref,
            judge_context_ref=j.context_ref, judge_context_sha256=j.context_sha256,
        )

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
        **judge_fields,
    )
    return "prepared", _PreparedGoal(
        goal=goal, binding=binding, plan_hash=resolved.plan_hash, receipt=receipt
    )


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
    deployment_judges: Any = None,
) -> WatchReceipt:
    """Resolve, compile, and arm ``plans`` on ``percept`` — see module docstring for the full
    semantics (coercion-at-apply, multi-goal batching, idempotency).

    ``percept`` is a :class:`~percept.agent.Percept` core (typed ``Any`` here to avoid importing
    ``percept.agent`` — this module is a leaf the agent module can import FROM, never the reverse).
    Every keyword beyond ``resolver``/``plans``/``deployment_judges`` is passed straight through to
    :meth:`percept.resolver.ConfigResolver.resolve` for EVERY plan in the batch (the same six-object
    configuration inputs apply to every goal a single call arms).

    **Two-phase atomic apply**: the batch is PREPARED in full first — every plan resolved,
    compiled, coerced, and (for a plan that requested a judge) its adapter constructed and any static
    context verified — with NO runtime mutation. Only once every plan prepared cleanly does Phase 2
    COMMIT arm the goals and register their judge bindings. A judge PREPARE failure (adapter cannot be
    built, or a static context fails its digest check) arms NOTHING and surfaces a rejected receipt with
    a structured diagnostic; prior runtime state is untouched. ``deployment_judges`` (a
    :class:`~percept.deployment_judges.DeploymentJudges`) supplies the ``context_store_root`` a bound
    static context resolves against; ``None`` defers to :func:`percept.deployment_judges.load_deployment_judges`
    and is only consulted when a plan actually binds a context.
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
        # Thread the SAME deployment catalog into the resolve-side authorization (ADR-009) that the
        # prepare-side binding uses. Without this the resolver reconciles WatchPlan.judge against the
        # DEFAULT (env) catalog — empty unless PERCEPT_JUDGE_CATALOG is set — so a caller that supplies
        # its catalog to apply_watch_plan (rather than via env) would have EVERY judge request rejected
        # as unknown before the binding step is ever reached. ``None`` defers to load_deployment_judges,
        # exactly as resolve() does on its own, so a judge-less batch is byte-for-byte unchanged.
        deployment_judges=deployment_judges,
    )
    plan_list = [plans] if isinstance(plans, WatchPlan) else list(plans)
    if not plan_list:
        raise ValueError("apply_watch_plan requires at least one WatchPlan")

    # -- Phase 1 PREPARE: resolve/compile/coerce every plan + build every judge binding. No runtime
    # mutation yet. ``known`` seeds idempotency from the core's applied-hash ledger and grows as each
    # plan prepares, so a within-batch duplicate is a no-op even though the ledger is written only at
    # commit. A _JudgePrepareError from ANY plan aborts the whole batch — arm nothing.
    ledger: dict[str, tuple[str, WatchReceipt]] = getattr(percept, _LEDGER_ATTR, None) or {}
    known: dict[str, tuple[str, WatchReceipt]] = dict(ledger)
    per_plan_receipts: list[WatchReceipt] = []
    prepared: list[_PreparedGoal] = []
    try:
        for plan in plan_list:
            kind, payload = _prepare_one(
                percept, plan, resolver, resolve_kwargs, deployment_judges, known
            )
            if kind == "prepared":
                prepared.append(payload)
                per_plan_receipts.append(payload.receipt)
                known[payload.plan_hash] = (payload.goal.id, payload.receipt)
            elif kind == "cached":
                per_plan_receipts.append(payload[1])
            else:  # "rejected"
                per_plan_receipts.append(payload)
    except _JudgePrepareError as exc:
        # Fail-closed atomic apply: a judge prepare failed — COMMIT nothing, leave the runtime untouched
        # (no goal armed, no binding registered, the ledger unwritten). Surface the structured failure.
        return WatchReceipt(status="rejected", dropped_fields=(exc.diagnostic,))

    # -- Phase 2 COMMIT: every plan prepared cleanly — NOW arm each goal, register its judge binding, and
    # write the idempotency ledger. This is the ONLY place this call mutates runtime state.
    if prepared:
        real_ledger = getattr(percept, _LEDGER_ATTR, None)
        if real_ledger is None:
            real_ledger = {}
            setattr(percept, _LEDGER_ATTR, real_ledger)
        # Thread the RESOLVED SourceProfile OBJECT (the actual ``resolve_kwargs["source_profile"]`` this
        # batch was called with — never ``resolved.source_profile``, a plain JSON snapshot) onto the core,
        # the same additive field ``Percept.configure`` populates. Only when THIS call named one; a
        # profile-less arm leaves whatever the core already carried untouched.
        if resolve_kwargs.get("source_profile") is not None:
            percept.source_profile = resolve_kwargs["source_profile"]
        for p in prepared:
            percept.add_goal(p.goal)
            if p.binding is not None:
                percept._register_judge_binding(p.goal.id, p.binding)
            real_ledger[p.plan_hash] = (p.goal.id, p.receipt)

    watch_ids: list[str] = []
    accepted: list[str] = []
    dropped: list[FieldDiagnostic] = []
    coerced: list[FieldDiagnostic] = []
    warnings: list[FieldDiagnostic] = []
    n_rejected = 0
    n_partial = 0
    n_warned = 0

    for receipt in per_plan_receipts:
        watch_ids.extend(receipt.watch_ids)
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
