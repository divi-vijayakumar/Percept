"""FallbackVision — wrap a primary ``Vision`` backend with an optional secondary, fail-closed.

**The seam this closes.** A judge call can fail for reasons that have nothing to do with the
condition being watched: a vendor outage, a rate limit, a tripped :class:`~percept.backends._resilience.
CircuitBreaker` (:class:`~percept.backends._resilience.CircuitOpen`). Every real backend already retries
(`with_retry`) and degrades a *parse* miss to a low-confidence verdict — but an exhausted retry or an open
circuit still RAISES today, and that exception propagates straight through the judge seam into the agent's
hot path. ``FallbackVision`` is the seam that stops that: it wraps one primary backend and an OPTIONAL
secondary, and on a primary failure it either hands off to the secondary or returns a typed low-confidence
``Verdict`` — it never lets a backend exception cross the judge seam.

**Composition, not registration.** This module deliberately does NOT touch ``percept/registry.py`` or
``percept/config.py`` (outside this backend module's scope). Call
:func:`wrap_if_configured` with the settings tree and an already-constructed primary backend to get a
transparently-wrapped ``Vision`` when a fallback is configured (e.g. ``settings.models.vision_fallback =
"gemini"``), or the primary unchanged otherwise — a one-line hook for whoever owns the registry choke
point.
"""

from __future__ import annotations

from typing import Any

from ..adapters import Goal, Observation, Verdict

_LOW_CONFIDENCE_EVIDENCE = "vision backend unavailable"


class FallbackVision:
    """``Vision`` adapter: try ``primary``; on error, try ``fallback`` (if given); never raise.

    Both ``primary`` and ``fallback`` are constructed ``Vision`` adapters (any backend — Claude, Gemini,
    OpenRouter, a fake). A primary failure — a network error, an exhausted retry, or a tripped circuit
    breaker (:class:`~percept.backends._resilience.CircuitOpen` is just another exception here; no special
    casing needed) — delegates to ``fallback``. If ``fallback`` is absent, or it ALSO raises, the seam
    degrades to a typed low-confidence :class:`~percept.adapters.Verdict` (``satisfied=False,
    confidence=0.0``) for ``judge`` and an empty list for ``detect``/``detect_events`` — the same
    "unparsed -> low-confidence unknown" shape the individual backends already use for a parse miss, so
    nothing downstream of the eye needs to know a fallback happened.
    """

    RUNTIME_EXECUTION_BOUNDARY = "process"

    def __init__(self, primary: Any, fallback: Any = None) -> None:
        self._primary = primary
        self._fallback = fallback

    async def judge(
        self,
        condition: str,
        frame: Any,
        *,
        roi: Any = None,
        exemplar: Any = None,
        corrections: dict | None = None,
        frames: list | None = None,
        audio: Any = None,
        context: dict | None = None,
    ) -> Verdict:
        kwargs = dict(roi=roi, exemplar=exemplar, corrections=corrections, frames=frames,
                      audio=audio, context=context)
        try:
            return await self._primary.judge(condition, frame, **kwargs)
        except Exception as primary_exc:  # noqa: BLE001 — fail-closed seam: nothing may cross it
            if self._fallback is not None:
                try:
                    return await self._fallback.judge(condition, frame, **kwargs)
                except Exception as fallback_exc:  # noqa: BLE001 — both backends down: degrade, don't raise
                    return Verdict(
                        satisfied=False, confidence=0.0,
                        evidence=f"{_LOW_CONFIDENCE_EVIDENCE}: primary={primary_exc}; fallback={fallback_exc}",
                    )
            return Verdict(satisfied=False, confidence=0.0,
                           evidence=f"{_LOW_CONFIDENCE_EVIDENCE}: {primary_exc}")

    async def detect(self, frame: Any, goals: list[Goal]) -> list[Observation]:
        try:
            return await self._primary.detect(frame, goals)
        except Exception:  # noqa: BLE001 — fail-closed
            if self._fallback is not None:
                try:
                    return await self._fallback.detect(frame, goals)
                except Exception:  # noqa: BLE001
                    pass
            return []

    async def detect_events(self, frames: list, goals: list[Goal]) -> list:
        try:
            return await self._primary.detect_events(frames, goals)
        except Exception:  # noqa: BLE001 — fail-closed
            if self._fallback is not None:
                try:
                    return await self._fallback.detect_events(frames, goals)
                except Exception:  # noqa: BLE001
                    pass
            return []


def wrap_if_configured(primary: Any, settings: Any, *, make_vision: Any = None) -> Any:
    """Return ``primary`` wrapped in :class:`FallbackVision`, or ``primary`` unchanged.

    Reads ``settings.models.vision_fallback`` (a backend name, e.g. ``"gemini"``) — when it is a
    non-empty string, resolves it to a constructed adapter and wraps ``primary``; when it is absent/None
    (the default — no field needs to exist on ``settings`` for this to be a no-op), ``primary`` is
    returned untouched, byte-for-byte today's behaviour.

    ``make_vision`` is the resolver (``percept.registry.make_vision`` in production) — injected rather
    than imported at module scope so this module never needs to import the registry (keeps the wrapper
    trivially unit-testable and avoids an import-order coupling). This is the one-line hook a registry
    choke point wires in: ``vision = wrap_if_configured(vision, settings, make_vision=registry.make_vision)``.
    """
    name = getattr(getattr(settings, "models", settings), "vision_fallback", None)
    if not name:
        return primary
    if make_vision is None:
        from ..registry import make_vision as _make_vision  # noqa: PLC0415 — lazy: avoid a hard import cycle

        make_vision = _make_vision
    fallback = make_vision(name, settings)
    return FallbackVision(primary, fallback)


__all__ = ["FallbackVision", "wrap_if_configured"]
