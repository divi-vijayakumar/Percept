"""Name -> backend factory resolution (the open-core hook).

The core never imports a vendor SDK directly (invariant 7). Instead every seam resolves a
*name* ("fake", "claude", "gemini", ...) to a factory here. Built-ins are registered below;
third-party backends register via the ``percept.backends`` entry-point group; plugins may also
register at runtime with :func:`register`.

In Phase 0 only the ``fake`` backends exist. Real backends resolve lazily and raise a clear,
actionable error until their module ships (Phase 1+).
"""

from __future__ import annotations

from importlib import import_module
from importlib import metadata as importlib_metadata
from typing import Any, Callable

from .backends import fake as _fake

# seam -> {name -> factory(settings) -> adapter}
_REGISTRY: dict[str, dict[str, Callable[[Any], Any]]] = {
    "vision": {},
    "specialist": {},
    "conductor": {},
    "stt": {},
    "tts": {},
}

_SEAMS = tuple(_REGISTRY)


def register(seam: str, name: str, factory: Callable[[Any], Any]) -> None:
    """Register (or override) a backend factory for a seam."""
    if seam not in _REGISTRY:
        raise ValueError(f"unknown seam {seam!r}; expected one of {_SEAMS}")
    _REGISTRY[seam][name] = factory


def _lazy(module: str, attr: str) -> Callable[[Any], Any]:
    """A factory that imports a vendor adapter on first use, with a clear error if absent."""

    def _factory(settings: Any) -> Any:
        try:
            mod = import_module(f".backends.{module}", package=__package__)
            cls = getattr(mod, attr)
        except (ModuleNotFoundError, AttributeError) as exc:
            raise RuntimeError(
                f"backend {module!r} ({attr}) is not available in this build — it ships in a "
                f"later phase or needs an optional extra. Use 'fake' for now. (cause: {exc})"
            ) from exc
        return cls(settings)

    return _factory


# ---- built-in registrations -------------------------------------------------
register("vision", "fake", lambda s: _fake.FakeVision())
register("vision", "claude", _lazy("anthropic", "AnthropicVision"))
register("vision", "gemini", _lazy("gemini", "GeminiVision"))
register("vision", "gemini-live", _lazy("gemini_live", "GeminiLiveVision"))
register("vision", "openrouter", _lazy("openrouter", "OpenRouterVision"))

register("specialist", "fake", lambda s: _fake.FakeSpecialist())
register("specialist", "claude", _lazy("anthropic_text", "AnthropicSpecialist"))
register("specialist", "gemini", _lazy("gemini", "GeminiSpecialist"))

register("conductor", "fake", lambda s: _fake.FakeConductor())
register("conductor", "claude", _lazy("anthropic_text", "AnthropicConductor"))
register("conductor", "gemini", _lazy("gemini", "GeminiConductor"))

register("stt", "fake", lambda s: _fake.FakeSTT())
register("stt", "deepgram", _lazy("deepgram", "DeepgramSTT"))

register("tts", "fake", lambda s: _fake.FakeTTS())
register("tts", "deepgram", _lazy("deepgram", "DeepgramTTS"))


def discover_entrypoints() -> None:
    """Load third-party backends advertised under the ``percept.backends`` entry-point group.

    Each entry point's value is called with the registry's ``register`` so plugins can wire
    themselves in. Failures in one plugin never break resolution of the others.
    """
    try:
        eps = importlib_metadata.entry_points(group="percept.backends")
    except TypeError:  # pragma: no cover — older importlib.metadata signature
        eps = importlib_metadata.entry_points().get("percept.backends", [])  # type: ignore[attr-defined]
    for ep in eps:
        try:
            ep.load()(register)
        except Exception:  # noqa: BLE001 — a bad plugin must not break the core
            continue


def make(seam: str, name: str, settings: Any = None) -> Any:
    """Resolve ``name`` for ``seam`` to a constructed adapter, or raise a clear error."""
    if seam not in _REGISTRY:
        raise ValueError(f"unknown seam {seam!r}; expected one of {_SEAMS}")
    table = _REGISTRY[seam]
    if name not in table:
        raise ValueError(
            f"unknown {seam} backend {name!r}; available: {sorted(table)}"
        )
    return table[name](settings)


def make_vision(name: str, settings: Any = None) -> Any:
    return make("vision", name, settings)


def make_specialist(name: str, settings: Any = None) -> Any:
    return make("specialist", name, settings)


def make_conductor(name: str, settings: Any = None) -> Any:
    return make("conductor", name, settings)


def make_stt(name: str, settings: Any = None) -> Any:
    return make("stt", name, settings)


def make_tts(name: str, settings: Any = None) -> Any:
    return make("tts", name, settings)


def available(seam: str) -> list[str]:
    return sorted(_REGISTRY.get(seam, {}))


__all__ = [
    "register",
    "discover_entrypoints",
    "make",
    "make_vision",
    "make_specialist",
    "make_conductor",
    "make_stt",
    "make_tts",
    "available",
]
