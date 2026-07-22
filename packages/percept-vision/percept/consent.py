"""Consent — the SDK primitives (spec §8). The UI lives in project 2 (the assistant).

This module keeps the trust posture minimal but real:

* a **profile flag** gates the higher-trust ``act`` (computer-use) modality — opt-in via
  ``allow_computer_use``, scoped to a declared **action set**; the SDK *enforces* it;
* an **action log** records every fire and every act (the audit seed + a future feedback corpus).

The SDK exposes the events; the assistant renders the consent strip / "what I can see" affordance —
no UI is built here. Nothing acts silently: a blocked ``act`` is logged too.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .envelope import (
    ACTUATION_CLASSES,
    PHYSICAL_ACTUATION,
    EnvelopeCheck,
    actuation_check,
    validate_actuation,
)


class ConsentDenied(PermissionError):
    """Raised when an ``act`` is attempted without the flag / outside the declared action set."""


@dataclass(frozen=True)
class ConsentProfile:
    """The act-modality gate. Default: computer-use OFF, no actions permitted.

    Consent is **per-source**: ``allowed_sources`` scopes acting to specific
    ``source_id``s. Empty (the default) = no source restriction, so existing single-source configs are
    byte-for-byte unchanged. A non-empty set is the single-node safety contract — a source NOT in it
    (e.g. another node's camera) can never act, which is exactly the "no cross-node bind" guarantee.
    """

    allow_computer_use: bool = False
    allowed_actions: frozenset = frozenset()
    allowed_sources: frozenset = frozenset()
    #: ADR-007 — the TYPED envelope declaration for acting, following the ADR-005
    #: ``UNTUNED`` pattern (a field a tool reads and reports, never a paragraph in a README).
    #: ``"none"`` (default, unchanged behaviour) / ``"screen"`` (advisory software acts — the
    #: narrowed survivor) / ``"physical"`` (actuation with physical consequence). Declaring
    #: ``"physical"`` declares this deployment OUTSIDE the envelope: :meth:`permits` then refuses
    #: every action, because the gate's honest failure mode is ASK and a control loop cannot ask.
    actuation: str = "none"

    def __post_init__(self) -> None:
        validate_actuation(self.actuation)

    def permits(self, action: str, source_id: str | None = None) -> bool:
        if self.actuation == PHYSICAL_ACTUATION:
            # Structural, not advisory: a deployment that DECLARES physical consequence gets no acts
            # from percept at all. See percept.envelope.OUT_OF_ENVELOPE["robotics_control_loop"].
            return False
        if not (self.allow_computer_use and action in self.allowed_actions):
            return False
        if self.allowed_sources and source_id not in self.allowed_sources:
            return False  # per-source consent: an un-consented source can't act (no cross-node bind)
        return True

    def envelope(self, *, modality: str = "act") -> EnvelopeCheck:
        """The typed envelope verdict for this profile — what ``percept doctor`` prints."""
        return actuation_check(modality=modality, actuation=self.actuation)

    @property
    def actuation_classes(self) -> tuple[str, ...]:
        """The declarable vocabulary, re-exported so a caller does not import two modules to
        validate one field."""
        return ACTUATION_CLASSES


@dataclass
class ActionRecord:
    """One audited event — a fire, an ask, or an act (allowed or blocked)."""

    kind: str  # "fire" | "ask" | "act" | "act_blocked"
    goal_id: str
    detail: dict = field(default_factory=dict)
    t: float = 0.0
    allowed: bool = True


class ActionLog:
    """Append-only audit log of fires + acts. Doubles as a future feedback corpus."""

    def __init__(self) -> None:
        self._records: list[ActionRecord] = []

    def record(self, record: ActionRecord) -> ActionRecord:
        self._records.append(record)
        return record

    def entries(self, kind: str | None = None) -> list[ActionRecord]:
        return [r for r in self._records if kind is None or r.kind == kind]

    def __len__(self) -> int:
        return len(self._records)


__all__ = [
    "ConsentDenied",
    "ConsentProfile",
    "ActionRecord",
    "ActionLog",
    "ACTUATION_CLASSES",
    "PHYSICAL_ACTUATION",
]
