"""Judge authorization — reconcile a ``WatchPlan.judge`` REQUEST against the deployment catalog.

Per ADR-009 (``.adr/ADR-009-operational-judge-request.md``) ``WatchPlan.judge`` is an
operational *request*: the user names the specialist judge they want, but the user does not decide
whether it is reachable. Availability is a **deployment** concern
(:mod:`percept.deployment_judges`). This module is the single atomic authorization step the
resolver calls PRE-ARM, and it is **fail closed**: an unknown ref, a ref whose vision-seam factory
is not installed, or a static-context binding a judge does not accept each REFUSE — there is no
silent fallback to the generalist vision model (ADR-009's "no generalist fallback for a specific
request" invariant).

Three failures, in the order they are checked (:func:`resolve_authorized`):

1. **catalog membership** — the ref must be a key in the deployment catalog
   (:meth:`percept.deployment_judges.DeploymentJudges.spec_for`). A miss is
   :class:`UnknownJudgeError`: the deployment must EXPLICITLY authorize the judge. An empty catalog
   (the default) therefore refuses every request.
2. **installed factory** — the authorized spec's ``factory_ref`` must be a registered ``"vision"``
   seam factory (:func:`percept.registry.has`, a non-instantiating lookup). A catalog entry that
   names a factory this build does not ship is :class:`JudgeFactoryUnavailableError`.
3. **context capability** — if the plan bound a static ``judge_context`` but the spec declares
   ``supports_static_context=False``, that is :class:`JudgeContextUnsupportedError`.

All three subclass :class:`percept.resolver.ConfigError` so a caller that already fails closed on a
resolver ``ConfigError`` catches a judge refusal the same way, and the refusal propagates pre-arm.

The module also owns the ADAPTER CACHE (:func:`get_or_create_adapter`) — a process-level memo so
the same resolved judge does not reconstruct its adapter per dispatch — and :class:`BoundJudge`, the
shape the dispatch side binds a resolved spec, its constructed adapter, and any verified static
context into.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from . import registry
from .contracts.watchplan import ArtifactBinding, ResolvedJudgeSpec
from .deployment_judges import DeploymentJudges
from .resolver import ConfigError

if TYPE_CHECKING:  # pragma: no cover — typing only, no runtime import cost / cycle
    from .judge_context import ResolvedJudgeContext


class JudgeAuthorizationError(ConfigError):
    """Base for every judge-authorization refusal — fail closed, raised pre-arm.

    Subclasses :class:`percept.resolver.ConfigError` so a resolver caller that already treats a
    ``ConfigError`` as "the watch cannot arm" catches a judge refusal identically.
    """


class UnknownJudgeError(JudgeAuthorizationError):
    """The requested judge ref is not in the deployment catalog.

    A refusal, never a downgrade: the deployment must EXPLICITLY authorize a judge before it can be
    armed (an empty catalog authorizes none). No fallback to the generalist vision model.
    """


class JudgeFactoryUnavailableError(JudgeAuthorizationError):
    """The catalog authorizes the judge, but its ``factory_ref`` is not an installed vision factory.

    The deployment named a judge whose implementing ``"vision"``-seam factory this build does not
    register (nor advertise via the ``percept.backends`` entry-point group) — so the judge cannot be
    constructed. Fail closed rather than arm a watch that would break at first dispatch.
    """


class JudgeContextUnsupportedError(JudgeAuthorizationError):
    """A static ``judge_context`` binding was supplied for a judge that does not accept one.

    The judge's spec declares ``supports_static_context=False``; binding a static context to it is a
    configuration error, refused pre-arm.
    """


def resolve_authorized(
    requested_ref: str,
    deployment: DeploymentJudges,
    *,
    judge_context: "ArtifactBinding | None" = None,
) -> ResolvedJudgeSpec:
    """Authorize ``requested_ref`` against ``deployment`` in ONE atomic, fail-closed step.

    Checks, in order (each a hard refusal): catalog membership → installed vision-seam factory →
    static-context capability. On success returns the :class:`~percept.contracts.watchplan.ResolvedJudgeSpec`
    the resolver records as the judge component. ``prompt_envelope_sha256`` is left ``None`` in this
    wheel (a deferred field). The returned spec mirrors any ``judge_context`` binding's ref + digest
    onto ``context_ref`` / ``context_sha256`` (the binding's bytes are verified elsewhere — this step
    only records the content address).
    """
    spec = deployment.spec_for(requested_ref)
    if spec is None:
        raise UnknownJudgeError(
            f"judge {requested_ref!r} is not in the deployment catalog — the deployment must "
            "explicitly authorize it before it can be armed (fail closed, pre-arm; there is no "
            "fallback to the generalist vision model for a failed specific request)"
        )

    if not registry.has("vision", spec.factory_ref):
        raise JudgeFactoryUnavailableError(
            f"judge {requested_ref!r} is authorized but its vision-seam factory "
            f"{spec.factory_ref!r} is not installed/registered in this build "
            f"(available: {registry.available('vision')}) — the watch is refused pre-arm rather "
            "than armed to break at first dispatch"
        )

    if judge_context is not None and not spec.supports_static_context:
        raise JudgeContextUnsupportedError(
            f"judge {requested_ref!r} does not accept a static judge_context binding "
            "(its deployment spec declares supports_static_context=False), but a judge_context was "
            f"bound (ref={judge_context.ref!r}) — refused pre-arm"
        )

    binding = judge_context
    return ResolvedJudgeSpec(
        requested_ref=requested_ref,
        resolved_ref=requested_ref,
        factory_ref=spec.factory_ref,
        provider_ref=spec.provider_ref,
        model_requested=spec.model_ref,
        model_resolved=spec.model_ref,
        prompt_envelope_sha256=None,
        context_ref=binding.ref if binding is not None else None,
        context_sha256=binding.sha256 if binding is not None else None,
    )


@dataclass(frozen=True)
class BoundJudge:
    """A resolved judge bound to its constructed adapter and (optionally) its verified static context.

    The dispatch side assembles exactly this from a :class:`~percept.contracts.watchplan.ResolvedJudgeSpec`:
    the authorized ``spec``, the ``adapter`` :func:`get_or_create_adapter` returned for it, and the
    ``static_context`` a :class:`percept.judge_context.JudgeContextStore` verified (``None`` when the
    plan bound no context).
    """

    spec: ResolvedJudgeSpec
    adapter: Any
    static_context: "ResolvedJudgeContext | None"


#: Process-level adapter memo, keyed by the resolved judge's identity dimensions (ref + factory +
#: provider + model). A judge that resolves the same way across dispatches reuses one adapter rather
#: than reconstructing (and re-importing) it each time. Never keyed on a plan or a frame — only on
#: the judge's own identity.
_ADAPTER_CACHE: dict[tuple[str, str, str, str], Any] = {}


def _adapter_cache_key(spec: ResolvedJudgeSpec) -> tuple[str, str, str, str]:
    return (spec.resolved_ref, spec.factory_ref, spec.provider_ref, spec.model_resolved)


def get_or_create_adapter(spec: ResolvedJudgeSpec, cfg: Any) -> Any:
    """Return the cached adapter for ``spec``, constructing it via the vision seam on a miss.

    Keyed by ``(resolved_ref, factory_ref, provider_ref, model_resolved)`` — the judge's identity —
    so two dispatches of the same resolved judge share one adapter. On a miss the adapter is built
    with :func:`percept.registry.make` (``"vision"`` seam, the spec's ``factory_ref``) and ``cfg``.
    Use :func:`clear_adapter_cache` to reset (tests, or a deployment reconfiguration).
    """
    key = _adapter_cache_key(spec)
    adapter = _ADAPTER_CACHE.get(key)
    if adapter is None:
        adapter = registry.make("vision", spec.factory_ref, cfg)
        _ADAPTER_CACHE[key] = adapter
    return adapter


def clear_adapter_cache() -> None:
    """Drop every cached adapter — the next :func:`get_or_create_adapter` reconstructs."""
    _ADAPTER_CACHE.clear()


__all__ = [
    "JudgeAuthorizationError",
    "UnknownJudgeError",
    "JudgeFactoryUnavailableError",
    "JudgeContextUnsupportedError",
    "resolve_authorized",
    "BoundJudge",
    "get_or_create_adapter",
    "clear_adapter_cache",
]
