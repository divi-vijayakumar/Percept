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

Ordinary per-plan resolution or field-support rejections remain independent and may therefore
produce a ``"partial"`` batch receipt. Prepare invariants that would make an armed plan unsafe are
batch-atomic instead: an unavailable/invalid judge binding or an unsupported runtime behavior route
aborts preparation and arms **none** of the batch.

**Idempotency by executable identity**: a plan whose :attr:`ResolvedWatch.plan_hash` AND
:attr:`ResolvedWatch.component_snapshot_hash` were already applied to THIS ``percept`` core re-arms
nothing and returns the SAME receipt as the first application — a no-op, not a re-arm, not a second
goal. The component snapshot matters because the same portable intent resolved against a different
SourceProfile/policy is a different executable Watch. The ledger is scoped to the ``percept`` core
it was applied to. A changed executable may not replace an existing goal id in place: state migration
for gate/episode/executor/detector memory is not defined, so the apply is rejected and the caller must
use a new id (or explicitly end the old goal first).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any, Iterable

from .behavior import (
    coerce_authored_behavior,
    coerce_authored_episode,
    coerce_authored_perception,
    coerce_authored_rhythm,
    coerce_authored_sense,
    program_of,
)
from .behavior_routes import (
    authored_behavior_type_display,
    behavior_route,
    worker_can_drive,
)
from .audio_input import prepare_audio_judge_binding
from .contracts import GoalOrigin
from .contracts.watchplan import FieldDiagnostic, ReceiptStatus, WatchPlan, WatchReceipt
from .deployment_judges import load_deployment_judges
from .duration_routes import duration_route_issues
from .judge_context import JudgeContextError, JudgeContextStore
from .resolver import ConfigError, ConfigResolver
from .runtime_compiler import (
    modality_matches_source,
    runtime_compile,
    sense_matches_modalities,
    supports_sense,
)
from .tracing import content_hash

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
_RUNTIME_MUTABLE_GOAL_STATE_KEYS = frozenset(
    {"count", "subject", "deadline_at", "deadline_at_epoch", "deadline_clock_domain"}
)


def _claim_backend_warning(
    percept: Any, goal: Any, *, adapter: Any = None
) -> "FieldDiagnostic | None":
    """A receipt WARNING (never a rejection) when ``goal`` arms ``identity_policy='claim_slot'`` but the
    core's selected vision backend does not parse the correction channel.

    The claim/supersession fold is fully wired, but it only mutates when a backend emits
    ``operation``/``value`` — a backend that does not (no :data:`_CLAIM_PARSING_MARKER`) will still
    arm and run, silently never producing a revision chain. That is exactly a loud-but-non-fatal
    condition: warn, don't reject."""
    if goal.state.get("identity_policy") != "claim_slot":
        return None
    vision = adapter if adapter is not None else getattr(percept, "vision", None)
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

    Raised INSIDE Phase 1 (prepare) so :func:`apply_watch_plan` commits no goal. Carries a structured
    :class:`FieldDiagnostic` the batch surfaces on the rejected :class:`WatchReceipt` (the file's
    existing diagnostic idiom), not a bare string.
    """

    def __init__(self, diagnostic: FieldDiagnostic) -> None:
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic


class _RoutePrepareError(Exception):
    """An unsupported frame-worker route found before any plan in the batch is armed."""

    def __init__(self, diagnostic: FieldDiagnostic) -> None:
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic


def _runtime_route_diagnostic(state: Any, *, goal_id: str | None) -> FieldDiagnostic | None:
    """Return the fail-closed apply diagnostic for a non-frame-worker behavior state."""

    route = behavior_route(state)
    if worker_can_drive(route):
        return None
    authored_type = authored_behavior_type_display(state)
    guidance = (
        " Use the matching direct event, boundary, rhythm, or scheduler primitive."
        if route.value != "unsupported"
        else ""
    )
    return FieldDiagnostic(
        path="Goal.state.type",
        code="unsupported_runtime_route",
        goal_id=goal_id,
        message=(
            f"behavior type {authored_type} resolves to route {route.value!r}, which "
            f"percept.run cannot execute; no Watch was armed.{guidance}"
        ),
    )


def _duration_route_diagnostic(state: Any, *, goal_id: str | None) -> FieldDiagnostic | None:
    """Return the shared fail-closed duration diagnostic for worker-bound WatchPlans."""

    issues = duration_route_issues(
        state,
        goal_id=goal_id,
        require_worker_cadence=True,
        canonical_goal=False,
        validate_cadence_without_hold=True,
    )
    if not issues:
        return None
    issue = issues[0]
    return FieldDiagnostic(
        path=issue.path,
        code=issue.code,
        goal_id=goal_id,
        message=issue.message,
    )


@dataclass
class _PreparedGoal:
    """One goal fully prepared in Phase 1 — resolved/compiled/coerced + its judge binding (if any),
    with the receipt it will land. Holds NO runtime mutation: nothing is armed until Phase 2 commit."""

    goal: Any
    binding: Any | None  # a BoundJudge-shaped object (``.spec``/``.adapter``/``.static_context``), or None
    audio_binding: Any | None
    compiled: Any
    source_profile: Any | None
    application_key: str
    receipt: WatchReceipt


_PER_WATCH_SETTINGS_SECTIONS: frozenset[str] = frozenset({"edge", "episodes"})


def _application_key(resolved: Any) -> str:
    """Stable identity of the portable plan plus the configuration it actually compiled against."""
    effective_hash = content_hash(dict(resolved.effective))
    return f"{resolved.plan_hash}:{effective_hash}:{resolved.component_snapshot_hash}"


def _cached_runtime_intact(
    percept: Any,
    *,
    goal_id: str,
    application_key: str,
    compiled: Any,
    expected_goal: Any,
    source_profile: Any,
    resolved_judge: Any,
) -> bool:
    """Whether a prior receipt still describes the exact live executable state."""
    current_goal = percept.goals.get(goal_id)
    if current_goal is None or percept.goals.status(goal_id) != "active":
        return False
    if getattr(percept, "_watch_runtime_keys", {}).get(goal_id) != application_key:
        return False
    runtime = getattr(percept, "_watch_runtime_bindings", {}).get(goal_id)
    if runtime is None or len(runtime) != 2:
        return False
    bound_compiled, bound_source_profile = runtime
    if bound_compiled != compiled or bound_source_profile != source_profile:
        return False
    # These Goal fields are immutable executable/presentation inputs for an applied Watch. Runtime-
    # owned counters and deadline stamps live in state and are intentionally not compared here.
    static_fields = ("id", "condition", "say", "source", "modality", "expiry")
    if any(
        getattr(current_goal, name, None) != getattr(expected_goal, name, None)
        for name in static_fields
    ):
        return False
    if getattr(current_goal, "origin", None) is not GoalOrigin.STRUCTURED_TRUSTED:
        return False
    current_state = {
        key: value
        for key, value in current_goal.state.items()
        if key not in _RUNTIME_MUTABLE_GOAL_STATE_KEYS
    }
    expected_state = {
        key: value
        for key, value in expected_goal.state.items()
        if key not in _RUNTIME_MUTABLE_GOAL_STATE_KEYS
    }
    if current_state != expected_state:
        return False
    modality = program_of(expected_goal).perception.modality
    audio_binding = getattr(percept, "_audio_judge_bindings", {}).get(goal_id)
    if modality in ("audio", "fused"):
        selected_adapter = (
            getattr(getattr(percept, "_judge_bindings", {}).get(goal_id), "adapter", None)
            if resolved_judge is not None
            else getattr(audio_binding, "adapter", None)
        )
        expected_mode = "audio_only" if modality == "audio" else "fused"
        if (
            audio_binding is None
            or getattr(audio_binding, "adapter", None) is not selected_adapter
            or getattr(audio_binding, "mode", None) != expected_mode
            or getattr(audio_binding, "application_key", None) != application_key
            or getattr(percept, "_audio_judge_binding_tokens", {}).get(goal_id)
            is not getattr(audio_binding, "binding_token", None)
        ):
            return False
    elif audio_binding is not None:
        return False
    judge_bindings = getattr(percept, "_judge_bindings", {})
    judge_requesting = getattr(percept, "_judge_requesting_goal_ids", set())
    bound_judge = judge_bindings.get(goal_id)
    if resolved_judge is None:
        return bound_judge is None and goal_id not in judge_requesting
    if (
        goal_id not in judge_requesting
        or bound_judge is None
        or getattr(bound_judge, "spec", None) != resolved_judge
        or getattr(bound_judge, "adapter", None) is None
        or getattr(percept, "_judge_binding_tokens", {}).get(goal_id) != id(bound_judge)
    ):
        return False
    static_context = getattr(bound_judge, "static_context", None)
    if resolved_judge.context_ref is None:
        return resolved_judge.context_sha256 is None and static_context is None
    if static_context is None or resolved_judge.context_sha256 is None:
        return False
    expected_sha = resolved_judge.context_sha256.strip().lower().removeprefix("sha256:")
    context_sha = (
        str(getattr(static_context, "sha256", "")).strip().lower().removeprefix("sha256:")
    )
    content = getattr(static_context, "content_bytes", None)
    if not isinstance(content, bytes):
        return False
    try:
        canonical_text = content.decode("utf-8")
    except UnicodeDecodeError:
        # JudgeContextStore.resolve uses the same strict UTF-8 contract. A hand-built/tampered bound
        # context must not bypass it on the cached path.
        return False
    return (
        getattr(static_context, "ref", None) == resolved_judge.context_ref
        and context_sha == expected_sha
        and hashlib.sha256(content).hexdigest() == expected_sha
        and getattr(static_context, "decoded_text", None) == canonical_text
    )


def _named_profile_candidates(name: str, watch_registry: Any) -> tuple[Any | None, tuple[Any, ...]]:
    """Resolve the two profile namespaces without guessing which one ``WatchPlan.profile`` meant."""
    source_candidate = None
    try:
        from .sourceprofile import (  # noqa: PLC0415
            SourceProfileNotFoundError,
            get_source_profile,
        )

        source_candidate = get_source_profile(name)
    except SourceProfileNotFoundError:  # registry miss: WatchProfile may still own the name
        pass

    if watch_registry is None:
        from . import profiles_default  # noqa: F401, PLC0415 - seed the default registry
        from .watchprofile import DEFAULT_REGISTRY  # noqa: PLC0415

        watch_registry = DEFAULT_REGISTRY
    profiles = getattr(watch_registry, "_profiles", {})
    watch_candidates = tuple(
        profile for profile in getattr(profiles, "values", lambda: ())()
        if getattr(profile, "name", None) == name
    )
    return source_candidate, watch_candidates


def _unsupported_plan_fields(plan: WatchPlan) -> tuple[FieldDiagnostic, ...]:
    """Typed WatchPlan fields that this runtime cannot execute yet.

    These fields are useful contract reservations, but accepting one today would produce an inert
    configuration. Refuse before resolution/arm so the public receipt never says a Watch carrying an
    ignored admission strategy, evidence strategy, extractor selection, or uncertainty policy
    was applied.
    """
    values = {
        "admission": plan.admission,
        "evidence": plan.evidence,
        "extractors": plan.extractors,
        "uncertainty": plan.uncertainty,
    }
    return tuple(
        FieldDiagnostic(
            path=f"WatchPlan.{name}",
            code="unsupported_watchplan_field",
            goal_id=plan.id,
            message=(
                f"WatchPlan.{name} is not executable by this runtime build; refusing to arm "
                "instead of accepting an inert field."
            ),
        )
        for name, value in values.items()
        if value not in (None, (), [])
    )


def _unsupported_core_setting(percept: Any, compiled: Any, *, goal_id: str) -> FieldDiagnostic | None:
    """Return the first compiled setting this shared core cannot isolate per Watch.

    Admission (``edge``) and episode bounds already have per-goal consumers.  The remaining
    ``Settings`` sections construct core-wide components (gate, entity graph, cache, providers,
    routing, evidence windows, and consent) and cannot truthfully be changed by arming one goal on an
    existing core.  Refuse a differing value instead of accepting configuration that never executes.
    """
    core_settings = getattr(percept, "settings", None)
    if core_settings is None:
        return None
    for section_name, compiled_section in vars(compiled.settings).items():
        if section_name in _PER_WATCH_SETTINGS_SECTIONS:
            continue
        core_section = getattr(core_settings, section_name, None)
        if core_section == compiled_section:
            continue
        compiled_values = vars(compiled_section)
        core_values = vars(core_section) if core_section is not None else {}
        field_name = next(
            (
                name
                for name, value in compiled_values.items()
                if core_values.get(name) != value
            ),
            "<section>",
        )
        return FieldDiagnostic(
            path=f"Settings.{section_name}.{field_name}",
            code="core_scoped_setting_requires_separate_runtime",
            goal_id=goal_id,
            message=(
                f"Settings.{section_name}.{field_name} resolves differently for this Watch, but "
                "the current runtime constructs that setting once per Percept core. Build the core "
                "with Percept.configure(...) or use a separate core; refusing to arm an inert "
                "per-Watch override."
            ),
        )
    return None


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

    A judge or runtime-route prepare failure aborts the WHOLE batch."""
    explicit_type = plan.explicit.get("Goal.state.type")
    if "Goal.state.type" in plan.explicit:
        authored = {"type": explicit_type}
        for key in ("from", "to"):
            path = f"Goal.state.{key}"
            if path in plan.explicit:
                authored[key] = plan.explicit[path]
        route_error = _runtime_route_diagnostic(
            {"type": "watch", "behavior": authored}, goal_id=plan.id
        )
        if route_error is not None:
            raise _RoutePrepareError(route_error)

    inert = _unsupported_plan_fields(plan)
    if inert:
        return "rejected", WatchReceipt(status="rejected", dropped_fields=inert)

    plan_resolve_kwargs = dict(resolve_kwargs)
    selected_source_profile = plan_resolve_kwargs.get("source_profile")
    if plan.profile is not None:
        source_candidate, watch_candidates = _named_profile_candidates(
            plan.profile, plan_resolve_kwargs.get("registry")
        )
        selected_name = getattr(selected_source_profile, "name", None)
        source_selection_explicit = bool(
            selected_source_profile is not None
            and plan_resolve_kwargs.get("source_profile_explicit")
        )
        if selected_source_profile is not None and selected_name != plan.profile:
            return "rejected", WatchReceipt(
                status="rejected",
                dropped_fields=(FieldDiagnostic(
                    path="WatchPlan.profile",
                    code="profile_kind_or_selection_conflict",
                    goal_id=plan.id,
                    message=(
                        f"WatchPlan.profile={plan.profile!r} conflicts with selected SourceProfile "
                        f"{selected_name!r}. The field spans SourceProfile and WatchProfile "
                        "namespaces, so the runtime will not guess."
                    ),
                ),),
            )
        if (
            not source_selection_explicit
            and source_candidate is not None
            and watch_candidates
        ):
            return "rejected", WatchReceipt(
                status="rejected",
                dropped_fields=(FieldDiagnostic(
                    path="WatchPlan.profile",
                    code="ambiguous_profile_kind",
                    goal_id=plan.id,
                    message=(
                        f"profile name {plan.profile!r} exists in both SourceProfile and "
                        "WatchProfile registries; select the SourceProfile explicitly or select a "
                        "WatchProfile through situation=(goal_type, environment, state)."
                    ),
                ),),
            )
        if not source_selection_explicit and watch_candidates:
            return "rejected", WatchReceipt(
                status="rejected",
                dropped_fields=(FieldDiagnostic(
                    path="WatchPlan.profile",
                    code="watch_profile_requires_situation",
                    goal_id=plan.id,
                    message=(
                        f"{plan.profile!r} names a WatchProfile, which is selected by situation="
                        "(goal_type, environment, state), not bound as a SourceProfile."
                    ),
                ),),
            )
        if source_candidate is None and not source_selection_explicit:
            return "rejected", WatchReceipt(
                status="rejected",
                dropped_fields=(FieldDiagnostic(
                    path="WatchPlan.profile",
                    code="profile_unavailable",
                    goal_id=plan.id,
                    message=(
                        f"profile {plan.profile!r} is absent from both SourceProfile and "
                        "WatchProfile registries."
                    ),
                ),),
            )
        if not source_selection_explicit:
            if (
                selected_source_profile is not None
                and selected_source_profile != source_candidate
            ):
                return "rejected", WatchReceipt(
                    status="rejected",
                    dropped_fields=(FieldDiagnostic(
                        path="WatchPlan.profile",
                        code="profile_kind_or_selection_conflict",
                        goal_id=plan.id,
                        message=(
                            f"WatchPlan.profile={plan.profile!r} resolves to a SourceProfile that "
                            "differs from the same-named SourceProfile inherited from the core."
                        ),
                    ),),
                )
            selected_source_profile = source_candidate
            plan_resolve_kwargs["source_profile"] = selected_source_profile
            plan_resolve_kwargs["source_profile_explicit"] = True

    if (
        selected_source_profile is not None
        and not sense_matches_modalities(plan.sense, selected_source_profile.modalities)
    ):
        return "rejected", WatchReceipt(
            status="rejected",
            dropped_fields=(FieldDiagnostic(
                path="WatchPlan.sense",
                code="source_profile_modality_mismatch",
                goal_id=plan.id,
                message=(
                    f"WatchPlan.sense={plan.sense!r} is incompatible with source profile "
                    f"{selected_source_profile.name!r} modalities "
                    f"{sorted(selected_source_profile.modalities)!r}; refusing to arm a Watch "
                    "whose requested channel is absent."
                ),
            ),),
        )

    try:
        resolved = resolver.resolve(plan, **plan_resolve_kwargs)
    except ConfigError as exc:
        return "rejected", WatchReceipt(
            status="rejected",
            dropped_fields=(FieldDiagnostic(
                path="WatchPlan",
                code="ownership_violation",
                goal_id=plan.id,
                message=str(exc),
            ),),
        )
    except ValueError as exc:
        unsupported_sense = not supports_sense(plan.sense)
        return "rejected", WatchReceipt(
            status="rejected",
            dropped_fields=(FieldDiagnostic(
                path="WatchPlan.sense" if unsupported_sense else "WatchPlan",
                code="unsupported_media_sense" if unsupported_sense else "configuration_invalid",
                goal_id=plan.id,
                message=str(exc),
            ),),
        )

    application_key = _application_key(resolved)
    cached = known.get(application_key)
    # The third tuple element is a prepare-only marker for an identical duplicate in this SAME batch.
    # Its first copy is still being prepared and will commit atomically with it.
    if cached is not None and len(cached) > 2:
        return "cached", (cached[0], cached[1])

    try:
        compiled = runtime_compile(resolved)
    except ValueError as exc:
        return "rejected", WatchReceipt(
            status="rejected",
            dropped_fields=(FieldDiagnostic(
                path="WatchPlan.sense",
                code="unsupported_media_sense",
                goal_id=plan.id,
                message=str(exc),
            ),),
        )
    duration_error = _duration_route_diagnostic(compiled.goal.state, goal_id=compiled.goal.id)
    if duration_error is not None:
        raise _RoutePrepareError(duration_error)
    route_error = _runtime_route_diagnostic(compiled.goal.state, goal_id=compiled.goal.id)
    if route_error is not None:
        raise _RoutePrepareError(route_error)
    unsupported = _unsupported_core_setting(percept, compiled, goal_id=compiled.goal.id)
    if unsupported is not None:
        return "rejected", WatchReceipt(status="rejected", dropped_fields=(unsupported,))
    state, coerced = _coerce_goal_state(compiled.goal.state, goal_id=compiled.goal.id)
    goal = replace(compiled.goal, state=state)
    if cached is not None and _cached_runtime_intact(
        percept,
        goal_id=cached[0],
        application_key=application_key,
        compiled=compiled,
        expected_goal=goal,
        source_profile=selected_source_profile,
        resolved_judge=getattr(resolved, "judge", None),
    ):
        return "cached", (cached[0], cached[1])
    existing = percept.goals.get(goal.id)
    if existing is not None:
        existing_status = percept.goals.status(goal.id)
        status_note = (
            f"existing lifecycle status is {existing_status!r}; " if existing_status != "active"
            else ""
        )
        return "rejected", WatchReceipt(
            status="rejected",
            dropped_fields=(FieldDiagnostic(
                path="WatchPlan.id",
                code="active_goal_id_reconfiguration_unsupported",
                goal_id=goal.id,
                message=(
                    f"goal id {goal.id!r} already exists. {status_note}The prior applied receipt is "
                    "not an exact match for intact live runtime state, and same-id replacement cannot "
                    "safely reset gate, executor, episode, detector, and tier-0 state. Use a new goal "
                    "id; completion or cancellation does not make the old id reusable."
                ),
            ),),
        )
    modality = program_of(goal).perception.modality
    if (
        selected_source_profile is not None
        and not modality_matches_source(modality, selected_source_profile.modalities)
    ):
        return "rejected", WatchReceipt(
            status="rejected",
            dropped_fields=(FieldDiagnostic(
                path="WatchPlan.sense",
                code="source_profile_modality_mismatch",
                goal_id=goal.id,
                message=(
                    f"compiled modality {modality!r} is incompatible with source profile "
                    f"{selected_source_profile.name!r} modalities "
                    f"{sorted(selected_source_profile.modalities)!r}; refusing to arm a Watch "
                    "whose required channel is absent."
                ),
            ),),
        )

    # JUDGE PREPARE (no runtime mutation): construct the adapter + verify any static context and build
    # the binding NOW, so a bad judge never arms a goal. resolved.judge is None for every judge-less plan
    # (the default), so this whole branch is skipped and judge_registry is never imported.
    binding = None
    if getattr(resolved, "judge", None) is not None:
        binding = _prepare_judge_binding(resolved.judge, percept, deployment_judges, goal_id=goal.id)
    selected_adapter = binding.adapter if binding is not None else getattr(percept, "vision", None)
    audio_binding = None
    if modality in ("audio", "fused"):
        try:
            audio_binding = prepare_audio_judge_binding(
                selected_adapter,
                mode="audio_only" if modality == "audio" else "fused",
                application_key=application_key,
            )
        except (TypeError, ValueError) as exc:
            raise _JudgePrepareError(FieldDiagnostic(
                path="WatchPlan.sense",
                code="audio_input_backend_unsupported",
                goal_id=goal.id,
                message=f"compiled {modality} Watch cannot bind trusted audio input: {exc}",
            )) from exc

    base = resolver.last_receipt  # the resolution-time WatchReceipt (accepted_fields + warnings)
    warnings = base.warnings if base is not None else ()
    accepted_fields = base.accepted_fields if base is not None else ()

    # Capability warning assembled HERE (watchplan_apply's own channel — NOT resolver.py's binding
    # function): a claim_slot plan armed on a non-parsing vision backend can never form a revision chain.
    backend_warning = _claim_backend_warning(percept, goal, adapter=selected_adapter)
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
        goal=goal,
        binding=binding,
        audio_binding=audio_binding,
        compiled=compiled,
        source_profile=selected_source_profile,
        application_key=application_key,
        receipt=receipt,
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

    **Prepared apply with bounded rollback**: the batch is PREPARED in full first — every plan
    resolved, compiled, coerced, and (for a plan that requested a judge) its adapter constructed and
    any static context verified — with NO runtime mutation. Cache invalidation is then preflighted
    before any goal is armed. An unavailable/invalid judge binding, unsupported runtime behavior
    route, or cache PREPARE failure aborts the batch atomically and arms nothing. Ordinary per-plan
    resolution and field-support rejections remain independent, so a batch may still return
    ``"partial"`` when at least one other plan arms. An unexpected COMMIT failure restores the goal
    registry plus Watch/judge bindings and apply ledger; cache eviction and revision notifications
    are intentionally not claimed transactional. ``deployment_judges`` (a
    :class:`~percept.deployment_judges.DeploymentJudges`) supplies the ``context_store_root`` a bound
    static context resolves against; ``None`` defers to :func:`percept.deployment_judges.load_deployment_judges`
    and is only consulted when a plan actually binds a context.
    """
    resolver = resolver if resolver is not None else ConfigResolver()
    if source_profile is None:
        source_profile = getattr(percept, "source_profile", None)
        if source_profile is not None:
            # The core's selected profile is an inherited deployment binding, not a new explicit
            # apply-call choice. Resolver provenance and warning behavior must retain that distinction.
            source_profile_explicit = False
    resolve_kwargs = dict(
        source_profile=source_profile,
        source_profile_explicit=source_profile_explicit,
        domain_pack=domain_pack,
        execution_policy=execution_policy,
        optimization_lock=optimization_lock,
        situation=situation,
        registry=registry,
        # The core's actual deployment settings are the default base. Reading env again here could
        # compile a Watch against a different configuration than the already-constructed runtime.
        settings=settings if settings is not None else getattr(percept, "settings", None),
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
    prepared_by_goal_id: dict[str, _PreparedGoal] = {}
    try:
        for plan in plan_list:
            kind, payload = _prepare_one(
                percept, plan, resolver, resolve_kwargs, deployment_judges, known
            )
            if kind == "prepared":
                prior = prepared_by_goal_id.get(payload.goal.id)
                if prior is not None and prior.application_key != payload.application_key:
                    return WatchReceipt(
                        status="rejected",
                        dropped_fields=(FieldDiagnostic(
                            path="WatchPlan.id",
                            code="duplicate_goal_id_in_batch",
                            goal_id=payload.goal.id,
                            message=(
                                f"batch contains distinct executable Watches for goal id "
                                f"{payload.goal.id!r}; no member of the batch was armed."
                            ),
                        ),),
                    )
                prepared.append(payload)
                prepared_by_goal_id[payload.goal.id] = payload
                per_plan_receipts.append(payload.receipt)
                known[payload.application_key] = (
                    payload.goal.id,
                    payload.receipt,
                    "prepared_in_batch",
                )
            elif kind == "cached":
                per_plan_receipts.append(payload[1])
            else:  # "rejected"
                per_plan_receipts.append(payload)
    except (_JudgePrepareError, _RoutePrepareError) as exc:
        # A prepare invariant failed — COMMIT nothing, leave the runtime untouched
        # (no goal armed, no binding registered, the ledger unwritten). Surface the structured failure.
        return WatchReceipt(status="rejected", dropped_fields=(exc.diagnostic,))

    # Cache invalidation may be supplied by a client adapter and can fail. Preflight every prepared
    # goal before mutating the registry/bindings so that failure cannot strand a half-armed batch.
    if prepared:
        try:
            for p in prepared:
                percept.invalidate_cache(p.goal.id)
        except Exception as exc:  # noqa: BLE001 - fail closed before goal/binding mutation
            return WatchReceipt(
                status="rejected",
                dropped_fields=(FieldDiagnostic(
                    path="WatchPlan",
                    code="commit_preflight_failed",
                    message=f"cache invalidation preflight failed; no Watch was armed: {exc}",
                ),),
            )

        # -- Phase 2 COMMIT. Snapshot only the state this function owns and can honestly restore.
        # GoalRegistry.restore bumps revision and cache evictions remain evicted, hence the bounded
        # (not fully transactional) contract documented above.
        goals_before = percept.goals.snapshot()
        watch_bindings_before = dict(getattr(percept, "_watch_runtime_bindings", {}))
        watch_keys_before = dict(getattr(percept, "_watch_runtime_keys", {}))
        judge_bindings_before = dict(getattr(percept, "_judge_bindings", {}))
        judge_tokens_before = dict(getattr(percept, "_judge_binding_tokens", {}))
        judge_requesting_before = set(getattr(percept, "_judge_requesting_goal_ids", set()))
        audio_bindings_before = dict(getattr(percept, "_audio_judge_bindings", {}))
        audio_tokens_before = dict(getattr(percept, "_audio_judge_binding_tokens", {}))
        ledger_existed = hasattr(percept, _LEDGER_ATTR)
        ledger_before = dict(getattr(percept, _LEDGER_ATTR, {}) or {})
        try:
            real_ledger = getattr(percept, _LEDGER_ATTR, None)
            if real_ledger is None:
                real_ledger = {}
                setattr(percept, _LEDGER_ATTR, real_ledger)
            for p in prepared:
                percept.add_goal(p.goal, _cache_preinvalidated=True)
                percept._register_watch_runtime(
                    p.goal.id,
                    p.compiled,
                    p.source_profile,
                    application_key=p.application_key,
                )
                if p.binding is not None:
                    percept._register_judge_binding(p.goal.id, p.binding)
                if p.audio_binding is not None:
                    percept._register_audio_judge_binding(p.goal.id, p.audio_binding)
            # Publish idempotency only after every goal and binding landed.
            for p in prepared:
                real_ledger[p.application_key] = (p.goal.id, p.receipt)
        except Exception as exc:  # noqa: BLE001 - restore every state owner mutated above
            percept.goals.restore(goals_before)
            percept._watch_runtime_bindings.clear()
            percept._watch_runtime_bindings.update(watch_bindings_before)
            percept._watch_runtime_keys.clear()
            percept._watch_runtime_keys.update(watch_keys_before)
            percept._judge_bindings.clear()
            percept._judge_bindings.update(judge_bindings_before)
            percept._judge_binding_tokens.clear()
            percept._judge_binding_tokens.update(judge_tokens_before)
            percept._judge_requesting_goal_ids.clear()
            percept._judge_requesting_goal_ids.update(judge_requesting_before)
            percept._audio_judge_bindings.clear()
            percept._audio_judge_bindings.update(audio_bindings_before)
            percept._audio_judge_binding_tokens.clear()
            percept._audio_judge_binding_tokens.update(audio_tokens_before)
            if ledger_existed:
                restored_ledger = getattr(percept, _LEDGER_ATTR, None)
                if not isinstance(restored_ledger, dict):
                    restored_ledger = {}
                    setattr(percept, _LEDGER_ATTR, restored_ledger)
                restored_ledger.clear()
                restored_ledger.update(ledger_before)
            else:
                try:
                    delattr(percept, _LEDGER_ATTR)
                except AttributeError:
                    pass
            return WatchReceipt(
                status="rejected",
                dropped_fields=(FieldDiagnostic(
                    path="WatchPlan",
                    code="commit_failed_rolled_back",
                    message=f"Watch commit failed and owned runtime state was restored: {exc}",
                ),),
            )

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
