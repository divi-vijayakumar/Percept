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
from .extractors import fields as _fields
from .extractors import ocr as _ocr

# seam -> {name -> factory(settings) -> adapter}
_REGISTRY: dict[str, dict[str, Callable[[Any], Any]]] = {
    "vision": {},
    "specialist": {},
    "conductor": {},
    "stt": {},
    "tts": {},
    "transport": {},
    "extractor": {},
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


def _lazy_fixture_vision() -> Any:
    """Construct the bundled-fixture vision replayer (imported on use, not at registry import)."""
    from .offline import FixtureVision

    return FixtureVision()


# ---- built-in registrations -------------------------------------------------
# "scripted" is the CANONICAL selector for the in-core doubles (RFC: offline mode selects scripted
# backends). "fake" is kept as a deprecated alias so existing wiring keeps resolving; both land on the
# same in-core double (the class name still reads Fake* — internal implementation, not user-facing text).
register("vision", "fake", lambda s: _fake.FakeVision())
register("vision", "scripted", lambda s: _fake.FakeVision())
# The FIXTURE replayer: the keyless first run's vision seam. Unlike "scripted", it
# answers ONLY the frames its bundled fixture scripts and raises on anything else — a real stream
# pointed at it fails loudly instead of producing plausible-looking judgements. Lazy (percept.offline
# imports adapters/config only, but keep the registry import-graph flat either way).
register("vision", "fixture", lambda s: _lazy_fixture_vision())
register("vision", "claude", _lazy("anthropic", "AnthropicVision"))
register("vision", "gemini", _lazy("gemini", "GeminiVision"))
register("vision", "gemini-live", _lazy("gemini_live", "GeminiLiveVision"))
register("vision", "openrouter", _lazy("openrouter", "OpenRouterVision"))

register("specialist", "fake", lambda s: _fake.FakeSpecialist())
register("specialist", "scripted", lambda s: _fake.FakeSpecialist())
register("specialist", "claude", _lazy("anthropic_text", "AnthropicSpecialist"))
register("specialist", "gemini", _lazy("gemini", "GeminiSpecialist"))

register("conductor", "fake", lambda s: _fake.FakeConductor())
register("conductor", "scripted", lambda s: _fake.FakeConductor())
# The HONEST scripted conductor double: maps an utterance -> a real create GoalOp, and its no-match
# fallback is respond_direct (never "ignore"). Opt-in — pass a script or select this selector.
register("conductor", "scripted-conductor", lambda s: _fake.ScriptedConductor())
register("conductor", "claude", _lazy("anthropic_text", "AnthropicConductor"))
register("conductor", "gemini", _lazy("gemini", "GeminiConductor"))

register("stt", "fake", lambda s: _fake.FakeSTT())
register("stt", "scripted", lambda s: _fake.FakeSTT())
register("stt", "deepgram", _lazy("deepgram", "DeepgramSTT"))

register("tts", "fake", lambda s: _fake.FakeTTS())
register("tts", "scripted", lambda s: _fake.FakeTTS())
register("tts", "deepgram", _lazy("deepgram", "DeepgramTTS"))

# The VOICE TRANSPORT seam (mic+speaker+AEC+endpointing+barge-in). Only the fake ships in-core; a real
# Pipecat/LiveKit transport is a vendor dep + a WebRTC room to test, so it's client-side — registered via
# the `percept.backends` entry-point group (see discover_entrypoints), never imported here.
register("transport", "fake", lambda s: _fake.FakeVoiceTransport())
register("transport", "scripted", lambda s: _fake.FakeVoiceTransport())

# The EXTRACTOR seam (percept.extraction.EvidenceExtractor) — structured facts OUT of a frame/roi,
# distinct from a detector's admission boundary (WHEN to look). Only the scripted double ships
# in-core; a real OCR/LPR/ASR specialist is client-side, registered the same way a vendor vision
# backend is (a plugin entry point, or a direct `register("extractor", name, factory)` call).
register("extractor", "fake", lambda s: _fake.ScriptedExtractor())
register("extractor", "scripted", lambda s: _fake.ScriptedExtractor())
# OCR (percept.extractors.ocr): "ocr-scripted" is the deterministic offline double (same
# import-cost profile as the generic "scripted" extractor above — zero heavy deps, CI-safe).
# "ocr" is the real pytesseract+Pillow backend; constructing it never imports either (only
# TesseractOCR.extract does), so this registration is safe even when the optional
# percept-vision[ocr] extra is not installed — the typed error surfaces at first actual use.
register("extractor", "ocr-scripted", lambda s: _ocr.ScriptedOCR(s if isinstance(s, dict) else None))
register("extractor", "ocr", lambda s: _ocr.TesseractOCR(s))
# Fields (percept.extractors.fields): typed numeric/currency/date/frequency normalization out of
# OCR-style text, so a fused compare judges NORMALIZED values, not raw surface text (the seam that
# closes the "25,000 dollars" vs "threshold: 25,000 dollars" false-mismatch incident). "fields" is
# the REAL backend — pure-Python regex/format logic, no vendor dependency, no lazy import (unlike
# TesseractOCR, nothing optional to install). "fields-scripted" is the deterministic offline double
# for CI, scripted by frame token the same way "ocr-scripted" is. "ocr-fields-scripted" is the
# convenience composition (percept.extractors.fields.chain_ocr_then_fields) wired to the scripted
# OCR+fields backends in one registry name, so a caller gets {"text", "regions", "fields"} facts
# out of a single extractor= without hand-wiring the chain themselves.
register("extractor", "fields-scripted", lambda s: _fields.ScriptedFields(s if isinstance(s, dict) else None))
register("extractor", "fields", lambda s: _fields.FieldNormalizer())
register(
    "extractor",
    "ocr-fields-scripted",
    lambda s: _fields.chain_ocr_then_fields(
        _ocr.ScriptedOCR(s if isinstance(s, dict) else None), _fields.FieldNormalizer()
    ),
)


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


def make_transport(name: str, settings: Any = None) -> Any:
    return make("transport", name, settings)


def make_extractor(name: str, settings: Any = None) -> Any:
    return make("extractor", name, settings)


def available(seam: str) -> list[str]:
    return sorted(_REGISTRY.get(seam, {}))


def seams() -> tuple[str, ...]:
    """Every seam name this registry knows (``"vision"``, ``"specialist"``, ...), declaration order.

    The public, read-only way to enumerate seams — ``percept.catalog`` uses this instead of
    reaching into the private ``_SEAMS``/``_REGISTRY`` module state."""
    return _SEAMS


__all__ = [
    "register",
    "discover_entrypoints",
    "make",
    "make_vision",
    "make_specialist",
    "make_conductor",
    "make_stt",
    "make_tts",
    "make_transport",
    "make_extractor",
    "available",
    "seams",
]
