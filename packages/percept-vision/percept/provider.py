"""Provider identity: a name resolves to a factory; a model id never does.

Percept already has a name -> factory table (:mod:`percept.registry`). What it did NOT have is a
statement of *what a provider name is*, and a guard on the one mistake that keeps re-appearing in
LLM-facing SDKs: **using the model id as the adapter's identity**.

Model ids are the most volatile string in this system — ``"qwen/qwen3-vl-32b-instruct"`` becomes
``"qwen/qwen3.5-vl"`` in a quarter, ``"claude-opus-4-8"`` gets a successor, an OpenRouter id can name
*any* vendor (``"anthropic/claude-3-opus"`` routed through OpenRouter is still the **openrouter**
adapter). If the model id selects the adapter, then every model rename is a wiring change, and a
cross-vendor id silently picks the wrong client. So this module fixes the split:

    adapter identity  = the PROVIDER NAME  ("openrouter", "claude", "gemini", "scripted")
    model id          = CONFIGURATION carried to that adapter (``settings.models.*_model``)

Hence the shape used — ``judge="openrouter", model="qwen/..."`` — two arguments, never one
fused string. :func:`resolve_vision` (and the general :func:`resolve`) is that door.

Three properties this module enforces, each with a test:

1. **Name -> factory.** Resolution goes through :func:`percept.registry.make`; this module adds no
   second construction path and no vendor import (invariant 7).
2. **A model id is never an adapter name.** ``judge="qwen/qwen3-vl-32b-instruct"`` does not fall through
   to "unknown vision backend" — it raises :class:`ProviderError` naming the two-argument fix. A
   *registered* name always wins the lookup, so this heuristic can never shadow a real provider.
3. **An explicit model is honored or refused, never silently swapped.** Some adapters coerce an id they
   cannot route (``OpenRouterVision._resolve_model`` replaces a non-namespaced id with its default).
   Silently judging on a different model than you asked for corrupts any sweep, so when the caller
   passes ``model=`` explicitly and the constructed adapter reports a different one, this raises.

**Manual adapter construction stays first-class.** Pass an already-built adapter instead of a name and
it is used verbatim — no registry, no settings rewrite, no model check (you built it, you own its
model). That is the escape hatch for anything the built-in table does not cover.

**Capability manifests are explicitly DEFERRED.** There is deliberately no "what can this provider do"
declaration here (multi-image? audio? tool use? JSON mode?). A manifest is only worth shipping once it
is *enforced* — checked before a call and honestly degraded on a miss — and unenforced manifests rot
into documentation that lies. Until that lands, capability differences stay where they already are: in
each adapter's honest degradation (e.g. ``OpenRouterVision.judge`` ignoring ``audio``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

from . import registry
from .config import Settings, load_settings

#: seam -> the ``ModelSettings`` field carrying that seam's model id. Seams absent here (stt, tts,
#: transport) take no model id through this door; passing one is an error rather than a silent no-op.
_MODEL_FIELD: dict[str, str] = {
    "vision": "vision_model",
    "specialist": "specialist_model",
    "conductor": "conductor_model",
}

#: Model-id shapes, used ONLY to turn an unknown name into an actionable error (never to select an
#: adapter). ``vendor/model`` is the OpenRouter/HF convention; the bare form catches the vendor ids that
#: get typed into the wrong slot (``claude-opus-4-8``, ``gemini-2.5-flash``, ``gpt-5-mini``).
_MODEL_ID_PREFIXES = ("claude", "gemini", "gpt", "qwen", "llama", "mistral", "internvl", "molmo")
_VERSIONED_ID = re.compile(r"^[a-z0-9]+(?:[-.][a-z0-9]+)*[-.]\d")


class ProviderError(ValueError):
    """A provider name / model id was used in a way that would make model ids load-bearing identity."""


def looks_like_model_id(name: str) -> bool:
    """True if ``name`` reads as a MODEL id rather than a provider name.

    Diagnostic only. Callers must check the registry first — a registered provider name is a provider
    name no matter what it looks like.
    """
    if not isinstance(name, str) or not name:
        return False
    low = name.strip().lower()
    if "/" in low:  # vendor/model — never a provider name in this table
        return True
    head = re.split(r"[-.]", low)[0]
    return head in _MODEL_ID_PREFIXES and bool(_VERSIONED_ID.match(low))


@dataclass(frozen=True)
class ProviderSpec:
    """A resolved provider selection: **identity** (the provider name) + **configuration** (the model).

    ``ProviderSpec("openrouter", "qwen/qwen3-vl-32b-instruct")`` and
    ``ProviderSpec("openrouter", "opengvlab/internvl3-78b")`` are two configurations of ONE adapter
    identity — :attr:`identity` is equal for both. That is the whole rule in one assertion.
    """

    provider: str
    model: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ProviderError("provider must be a non-empty name, e.g. 'openrouter'.")

    @property
    def identity(self) -> str:
        """The adapter identity — the provider name, and ONLY the provider name."""
        return self.provider


def settings_with_model(settings: Settings, seam: str, model: str | None) -> Settings:
    """Return ``settings`` with ``seam``'s model id set to ``model`` (a copy; the input is frozen)."""
    if model is None:
        return settings
    field = _MODEL_FIELD.get(seam)
    if field is None:
        raise ProviderError(
            f"seam {seam!r} carries no model id (it has no {seam}_model setting); "
            f"model={model!r} would be silently ignored. Model-bearing seams: {sorted(_MODEL_FIELD)}."
        )
    return replace(settings, models=replace(settings.models, **{field: model}))


def _adapter_model(adapter: Any) -> str | None:
    """The model id an adapter actually resolved to, if it reports one. No manifest, no protocol —
    the built-in adapters all keep it on ``_model``; anything that doesn't simply isn't checked."""
    value = getattr(adapter, "_model", None) or getattr(adapter, "model", None)
    return value if isinstance(value, str) and value else None


def resolve(
    seam: str,
    provider: Any,
    *,
    model: str | None = None,
    settings: Settings | None = None,
) -> Any:
    """Resolve ``provider`` for ``seam`` to a constructed adapter, carrying ``model`` as configuration.

    Args:
        seam: A :mod:`percept.registry` seam — ``"vision"``, ``"specialist"``, ``"conductor"``, ...
        provider: A registered provider NAME (``"openrouter"``, ``"claude"``, ``"scripted"``) — or an
            already-constructed adapter, which is returned verbatim (manual construction).
        model: OPTIONAL model id, e.g. ``"qwen/qwen3-vl-32b-instruct"``. Configuration only: it never
            selects the adapter. Passing it with a manually built adapter is an error (that object's
            model was fixed when you built it — a silently ignored ``model=`` is the bug this guards
            against).
        settings: Base :class:`~percept.config.Settings`; defaults to :func:`load_settings`.

    Raises:
        ProviderError: if a model id was passed where a provider name belongs, if the seam carries no
            model id, or if the adapter did not honor an explicitly requested model.
    """
    if not isinstance(provider, str):
        if provider is None:
            raise ProviderError(
                f"no {seam} provider given. Pass a provider name (e.g. {seam}='openrouter') plus an "
                f"optional model id, or an already-constructed adapter."
            )
        if model is not None:
            raise ProviderError(
                f"model={model!r} was passed alongside an already-constructed "
                f"{type(provider).__name__}; its model was fixed at construction and this would be "
                f"silently ignored. Either pass the provider NAME plus model=, or build the adapter "
                f"with the model you want."
            )
        return provider  # manual adapter construction — used verbatim, no registry, no rewrite

    name = provider.strip()
    if name not in registry.available(seam) and looks_like_model_id(name):
        raise ProviderError(
            f"{name!r} is a MODEL id, not a {seam} provider name — model ids are configuration, never "
            f"adapter identity (they change every release, and one id can name any vendor). "
            f"Fix: {seam}='openrouter', model={name!r} (or whichever provider serves it). "
            f"Available {seam} providers: {registry.available(seam)}."
        )

    cfg = settings if settings is not None else load_settings()
    adapter = registry.make(seam, name, settings_with_model(cfg, seam, model))

    if model is not None:
        actual = _adapter_model(adapter)
        if actual is not None and actual != model:
            raise ProviderError(
                f"{seam} provider {name!r} did not accept model={model!r} — it resolved to {actual!r} "
                f"instead. Refusing to judge on a model you did not ask for (a silent swap corrupts "
                f"any cost/quality comparison). Fix: pass a model id this provider can route "
                f"(for 'openrouter' that is a 'vendor/model' id), or drop model= to take its default."
            )
    return adapter


def resolve_vision(judge: Any, *, model: str | None = None, settings: Settings | None = None) -> Any:
    """Resolve the public provider/model shape: ``resolve_vision("openrouter", model="qwen/...")``.

    ``judge`` is a provider name or an already-constructed ``Vision`` adapter. See :func:`resolve`.
    """
    return resolve(  # judge is the vision seam's user-facing word
        "vision", judge, model=model, settings=settings
    )


__all__ = [
    "ProviderError",
    "ProviderSpec",
    "looks_like_model_id",
    "resolve",
    "resolve_vision",
    "settings_with_model",
]
