"""Consent — the SDK primitives (spec §8). The UI lives in project 2 (the assistant).

V0 keeps the trust posture minimal but real:

* a **profile flag** gates the higher-trust ``act`` (computer-use) modality — opt-in via
  ``allow_computer_use``, scoped to a declared **action set**; the SDK *enforces* it;
* an **action log** records every fire and every act (the audit seed + the V1 feedback corpus).

The SDK exposes the events; the assistant renders the consent strip / "what I can see" affordance —
no UI is built here. Nothing acts silently: a blocked ``act`` is logged too.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class ConsentDenied(PermissionError):
    """Raised when an ``act`` is attempted without the flag / outside the declared action set."""


@dataclass(frozen=True)
class ConsentProfile:
    """The act-modality gate. Default: computer-use OFF, no actions permitted.

    Consent is **per-source** (spec §9, A5): ``allowed_sources`` scopes acting to specific
    ``source_id``s. Empty (the default) = no source restriction, so existing single-source configs are
    byte-for-byte unchanged. A non-empty set is the single-node safety contract — a source NOT in it
    (e.g. another node's camera) can never act, which is exactly the "no cross-node bind" guarantee.
    """

    allow_computer_use: bool = False
    allowed_actions: frozenset = frozenset()
    allowed_sources: frozenset = frozenset()

    def permits(self, action: str, source_id: str | None = None) -> bool:
        if not (self.allow_computer_use and action in self.allowed_actions):
            return False
        if self.allowed_sources and source_id not in self.allowed_sources:
            return False  # per-source consent: an un-consented source can't act (no cross-node bind)
        return True


@dataclass
class ActionRecord:
    """One audited event — a fire, an ask, or an act (allowed or blocked)."""

    kind: str  # "fire" | "ask" | "act" | "act_blocked"
    goal_id: str
    detail: dict = field(default_factory=dict)
    t: float = 0.0
    allowed: bool = True


class ActionLog:
    """Append-only audit log of fires + acts. Doubles as the V1 feedback corpus."""

    def __init__(self) -> None:
        self._records: list[ActionRecord] = []

    def record(self, record: ActionRecord) -> ActionRecord:
        self._records.append(record)
        return record

    def entries(self, kind: str | None = None) -> list[ActionRecord]:
        return [r for r in self._records if kind is None or r.kind == kind]

    def __len__(self) -> int:
        return len(self._records)


__all__ = ["ConsentDenied", "ConsentProfile", "ActionRecord", "ActionLog"]
