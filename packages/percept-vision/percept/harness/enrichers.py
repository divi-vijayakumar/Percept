"""Enricher registry — the specialist-model handoff (ticket F1).

A Tier-0 signal is cheap and dumb (motion magnitude, an acoustic onset, a ``plate-presence`` cue). Some
signals want a NARROW specialist model to read out *structured facts* the general VLM judge can't cheaply
produce — an LPR model for a plate, a fire/smoke classifier for a hot-surface cue, a face-id model for a
known-person cue. This module is the seam that routes such a signal to its specialist and feeds the
specialist's STRUCTURED output into the judge's context:

    signal/detector name  --register_enricher-->  Enricher  --enrich_signal-->  {facts}  -->  Vision.judge(context=...)

It is the third registry beside ``detectors.py`` (cheap signal SKILLS) and ``concerns.py`` (standing
things-to-watch): same deliberate ``register`` / teardown lifecycle, a different keyspace (a *signal name*
-> the *specialist model* it hands off to). The boundary that keeps it honest is the same one the detector
registry draws: adding a specialist is a deliberate ``register_enricher(...)`` event — Percept core is
NEVER edited to add an LPR/fire/face model. Register an :class:`~percept.adapters.Enricher`, and any goal
whose cue carries that signal name gets the specialist's facts in its judge context, with zero core changes.

The :class:`~percept.adapters.Enricher` protocol (in ``percept.contracts``, beside ``Specialist``):
``enrich(signal, frame) -> dict``. The returned dict is injected as ``Vision.judge(context=...)``; a
``{"verdict": Verdict}`` key SHORT-CIRCUITS the judge (the brain uses that verdict and skips the VLM call).
This module ships only fakes (``NullEnricher`` / ``FakeEnricher``) — no real model; F1 is the SEAM, not the
LPR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from percept.adapters import Enricher, Verdict


# ---------------------------------------------------------------------------
# Fakes — prove the seam offline, no keys, no model (standing rule 2: wiring, not intelligence)
# ---------------------------------------------------------------------------


class NullEnricher:
    """An :class:`~percept.adapters.Enricher` that enriches nothing — the no-op default.

    Returns ``{}`` for every signal, so wiring it changes nothing: the judge proceeds exactly as it would
    untriggered. Useful as a registered placeholder for a signal whose real specialist hasn't shipped yet
    (the seam is wired; the model is a stub), and as the safe fallback the driver returns on a miss.
    """

    def enrich(self, signal: Any, frame: Any = None) -> dict:
        del signal, frame
        return {}


class FakeEnricher:
    """A scripted :class:`~percept.adapters.Enricher` — returns fixed structured facts (no model).

    Stands in for a real specialist (an LPR model, a fire/smoke classifier) in offline tests: it returns a
    pre-set ``facts`` dict so a test can prove the structured output actually *reaches* the judge context,
    without claiming the fake read anything. Pass ``verdict=`` to script the SHORT-CIRCUIT path: the
    enricher then returns ``{"verdict": verdict, **facts}`` and the driver hands that verdict straight to
    the brain (the VLM judge is skipped). ``raises=True`` simulates a flaky specialist — the driver must
    swallow it and degrade to a no-op (an enrichment failure never breaks the judge path).
    """

    def __init__(
        self,
        facts: dict | None = None,
        *,
        verdict: Verdict | None = None,
        raises: bool = False,
    ) -> None:
        self._facts = dict(facts or {})
        self._verdict = verdict
        self._raises = raises

    def enrich(self, signal: Any, frame: Any = None) -> dict:
        del signal, frame
        if self._raises:
            raise RuntimeError("scripted enricher failure")
        out = dict(self._facts)
        if self._verdict is not None:
            out["verdict"] = self._verdict
        return out


# ---------------------------------------------------------------------------
# The result of running an enricher (structured context + optional short-circuit verdict)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Enrichment:
    """What a specialist produced for a triggering signal: ``context`` for the judge, ``verdict`` to skip it.

    - ``context`` — the structured facts to pass to :meth:`~percept.adapters.Vision.judge` as ``context=``
      (never includes the reserved ``verdict`` key; that is split out below). ``{}`` means no-op.
    - ``verdict`` — present iff the enricher SHORT-CIRCUITED (returned a ``{"verdict": Verdict}`` key): the
      brain should use this verdict directly and skip the VLM judge call.

    ``is_noop`` is the cheap test the brain uses to stay byte-for-byte when nothing fired.
    """

    context: dict
    verdict: Verdict | None = None

    @property
    def is_noop(self) -> bool:
        """True when there is nothing to inject and no short-circuit — the un-enriched path."""
        return not self.context and self.verdict is None


# ---------------------------------------------------------------------------
# The registry — register a specialist under a signal name; the brain resolves by name
# ---------------------------------------------------------------------------
#
# PUBLIC, SEMVER-STABLE API, mirroring detectors.py: ``register_enricher`` (add a specialist),
# ``enricher_for`` (resolve, or None), ``enrich_signal`` (run it, swallowing failures),
# ``registered_enrichers`` (introspect), ``unregister_enricher`` / ``clear_enrichers`` (teardown).
# The registry is a process-global singleton keyed by SIGNAL/detector NAME.

_REGISTRY: dict[str, Enricher] = {}

#: Reserved key in an enricher's output dict that SHORT-CIRCUITS the judge.
VERDICT_KEY = "verdict"


def register_enricher(signal: str, enricher: Enricher) -> Enricher:
    """Register a specialist :class:`~percept.adapters.Enricher` under a ``signal``/detector name.

    The deliberate-registration lifecycle for specialists (mirrors
    :func:`~percept.harness.detectors.register`): after this, a cue carrying ``signal`` resolves to
    ``enricher`` via :func:`enricher_for`, and :func:`enrich_signal` runs it to produce judge context.
    Re-registering the same ``signal`` replaces the prior specialist (last-writer-wins — how a product
    swaps a stub for a calibrated model). Returns the enricher for convenient chaining.

    This is the WHOLE integration cost of a new specialist: an LPR model for a ``plate-presence`` cue is
    added by ``register_enricher("plate-presence", LprEnricher())`` — NO change to Percept core, the judge,
    or the contract. Adding a SECOND specialist is a second independent ``register_enricher`` call.
    """
    _REGISTRY[signal] = enricher
    return enricher


def enricher_for(signal: str) -> Enricher | None:
    """The specialist registered for ``signal``, or ``None`` if no specialist is registered for it.

    ``None`` (not a raise) is the un-enriched answer: most signals have no specialist and run the plain
    judge. The brain calls this per cue to decide whether a handoff exists before spending the model.
    """
    return _REGISTRY.get(signal)


def registered_enrichers() -> list[str]:
    """Registered signal names (sorted) — the specialists the brain can hand off to."""
    return sorted(_REGISTRY)


def unregister_enricher(signal: str) -> bool:
    """Tear down the specialist for ``signal``. Returns ``True`` if one was registered, else ``False``.

    The teardown half of the lifecycle: after this, ``signal`` resolves to ``None`` again and
    :func:`enrich_signal` no-ops for it. Idempotent (gone -> ``False``)."""
    return _REGISTRY.pop(signal, None) is not None


def clear_enrichers() -> None:
    """Empty the enricher registry (test isolation / a clean re-registration pass).

    Unlike the detector registry there are NO built-in enrichers to restore — F1 ships the seam, not any
    specialist — so this simply empties the table."""
    _REGISTRY.clear()


def enrich_signal(signal_name: str, signal: Any = None, frame: Any = None) -> Enrichment:
    """Run the specialist registered for ``signal_name`` and split its output into an :class:`Enrichment`.

    The driver the brain calls between a Tier-0 trigger and the judge:

    1. resolve ``signal_name`` -> specialist (no specialist -> a no-op :class:`Enrichment`, untouched path);
    2. ``enricher.enrich(signal, frame)`` -> a structured ``dict``;
    3. split a reserved ``verdict`` key out as the SHORT-CIRCUIT, the rest becomes judge ``context``.

    Robustness is part of the contract: a specialist that RAISES (a flaky model, a bad payload) degrades to
    a no-op — an enrichment failure must never break the judge path. ``signal`` is the raw trigger object
    handed to the specialist (a :class:`~percept.adapters.Tier0Signal`, a detector boundary, or any
    enricher-defined payload); ``frame`` is the current frame when available.
    """
    enricher = _REGISTRY.get(signal_name)
    if enricher is None:
        return Enrichment(context={})
    try:
        out = enricher.enrich(signal, frame)
    except Exception:  # noqa: BLE001 — a flaky specialist must never break the judge path
        return Enrichment(context={})
    if not isinstance(out, dict):  # defensive: a misbehaving enricher returned a non-dict
        return Enrichment(context={})
    verdict = out.get(VERDICT_KEY)
    context = {k: v for k, v in out.items() if k != VERDICT_KEY}
    return Enrichment(context=context, verdict=verdict if isinstance(verdict, Verdict) else None)


__all__ = [
    "NullEnricher",
    "FakeEnricher",
    "Enrichment",
    "VERDICT_KEY",
    "register_enricher",
    "enricher_for",
    "registered_enrichers",
    "unregister_enricher",
    "clear_enrichers",
    "enrich_signal",
]
