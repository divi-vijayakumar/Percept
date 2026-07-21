"""A reference identity state machine — the deterministic ORACLE the E-1 experiment scores against.

E-1 asks whether adding a subject-identity instruction to the judge prompt makes the runtime bind,
count, and re-fire subjects correctly. To *score* a real judge's output you need a reference answer
that is not itself a model — a pure, seeded, inspectable state machine that says, for a given
sequence of sightings, exactly how many distinct subjects there were, which ticks should fire, and
when a binding is honestly ``unknown``. This module is that oracle.

It is deliberately NOT a runtime (it never sees pixels, never calls a model). It consumes the same
scripted-verdict shape the synthetic corpora emit — ``subject_key`` / ``binding`` / ``operation`` —
and applies the identity rules the review pinned:

* **Identity is content-derived, never a session ordinal.** The same content key resolves to the
  same subject even after an unrelated subject held the region, after an episode closed, after a
  TTL horizon lapsed, and across a restart. Minting a key per sighting is the collapse this catches.
* **Normalization is symmetric.** Two formatting-equivalent values ("INV-2291" / "inv 2291") are the
  same subject (idempotent, no false split); but normalization that merges two GENUINELY distinct
  keys is a *collision* — a measurement hazard the metrics track, not a feature.
* **``unknown`` is a first-class state.** An illegible key yields ``unknown`` — never a fabricated
  key, never a silent disappearance.
* **Firing is a rising edge.** A subject fires when it OPENS (first sighting) or RE-opens (returns
  after close / TTL expiry / restart). Re-observing an already-open subject does not re-fire. Denser
  observation cannot manufacture incidents.
* **Non-binding speech acts do not bind.** A ``mention`` / ``negation`` / ``hypothetical`` sighting
  neither creates a subject nor fires — the counterfactual "if this were INV-9000" must not enter
  the ledger as a real subject.

Pure stdlib, no wall clock, no randomness. Two runs over the same sightings agree to the bit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

__all__ = [
    "normalize_key",
    "Binding",
    "NON_BINDING_OPS",
    "BINDING_OPS",
    "Sighting",
    "FireKind",
    "FireEvent",
    "Observed",
    "IdentityLedger",
]


class Binding(str, Enum):
    RESOLVED = "resolved"
    UNKNOWN = "unknown"


#: Speech acts that MUST NOT bind a subject or fire: a mention, a negation, or a hypothetical.
NON_BINDING_OPS = frozenset({"mention", "negation", "hypothetical"})
#: Speech acts that DO bind.
BINDING_OPS = frozenset({"assert", "replace", "retract"})


def normalize_key(key: str) -> str:
    """Canonicalise a content key for identity comparison: uppercase, alphanumerics only.

    ``"INV-2291"``, ``"inv 2291"`` and ``"INV2291"`` all normalise to ``"INV2291"`` — the same
    subject. This is the SAME mechanism whose over-eagerness the collision metric watches for: if two
    truly distinct keys share a normalisation, that is a hazard, reported, not silently accepted.
    """
    return "".join(ch for ch in key.upper() if ch.isalnum())


@dataclass(frozen=True)
class Sighting:
    """One thing seen at time ``t`` — the oracle's input unit, mirroring a scripted verdict.

    ``subject_key`` is the raw content key (``None`` when nothing legible was read). ``binding`` says
    whether the key resolved or is honestly unknown. ``operation`` is the speech act; only the
    binding ops actually bind. ``ttl`` overrides the ledger's default validity horizon for this
    sighting (``None`` = use the ledger default).
    """

    t: float
    subject_key: str | None = None
    binding: Binding = Binding.RESOLVED
    operation: str = "assert"
    ttl: float | None = None

    @classmethod
    def from_verdict(cls, t: float, verdict) -> "Sighting":
        """Build a sighting from a scripted ``Verdict`` (reads the additive ``subject_key`` / ``binding``).

        Accepts either a mapping (a raw verdict spec) or a ``Verdict`` instance carrying the additive
        attributes the ``ScriptedJudge`` attaches. A missing/None binding with a key present is read
        as ``resolved``; a null key is read as ``unknown``.
        """
        if isinstance(verdict, dict):
            key = verdict.get("subject_key")
            raw_binding = verdict.get("binding")
            op = verdict.get("operation", "assert")
            ttl = verdict.get("ttl")
        else:
            key = getattr(verdict, "subject_key", None)
            raw_binding = getattr(verdict, "binding", None)
            op = getattr(verdict, "operation", "assert")
            ttl = getattr(verdict, "ttl", None)
        if raw_binding is None:
            binding = Binding.RESOLVED if key else Binding.UNKNOWN
        else:
            binding = Binding(raw_binding)
        return cls(t=float(t), subject_key=key, binding=binding, operation=op or "assert", ttl=ttl)


class FireKind(str, Enum):
    OPEN = "open"      # a subject seen for the first time
    REOPEN = "reopen"  # a known subject returning after close / TTL expiry / restart


@dataclass(frozen=True)
class FireEvent:
    t: float
    subject_key: str      # the NORMALISED durable key
    kind: FireKind


@dataclass(frozen=True)
class Observed:
    """The oracle's answer for one sighting: did it fire, and how was it classified."""

    t: float
    normalized_key: str | None
    binding: Binding
    fired: bool
    fire_kind: FireKind | None
    non_binding: bool = False


@dataclass
class _Subject:
    key: str            # normalised durable key
    first_key_seen: str  # the raw key on first sighting (for collision diagnostics)
    open: bool
    last_seen_t: float
    ttl: float | None


@dataclass
class IdentityLedger:
    """The reference identity + firing state machine. Feed it :class:`Sighting`s in time order.

    ``default_ttl`` is the validity horizon (seconds) after which an unrefreshed open subject lapses
    and its next sighting counts as a re-open (a new episode). ``None`` disables TTL expiry entirely.
    """

    default_ttl: float | None = None
    _subjects: dict[str, _Subject] = field(default_factory=dict)
    _fires: list[FireEvent] = field(default_factory=list)
    _observed: list[Observed] = field(default_factory=list)
    _unknown_count: int = 0

    # -- introspection --------------------------------------------------------------------------
    @property
    def distinct_subject_count(self) -> int:
        """Distinct durable subjects resolved so far. ``unknown`` sightings never inflate this."""
        return len(self._subjects)

    @property
    def fires(self) -> list[FireEvent]:
        return list(self._fires)

    @property
    def fire_count(self) -> int:
        return len(self._fires)

    @property
    def observations(self) -> list[Observed]:
        return list(self._observed)

    @property
    def unknown_count(self) -> int:
        return self._unknown_count

    def subject_keys(self) -> set[str]:
        return set(self._subjects)

    def is_open(self, raw_key: str) -> bool:
        s = self._subjects.get(normalize_key(raw_key))
        return bool(s and s.open)

    # -- transitions ----------------------------------------------------------------------------
    def _expire(self, now: float) -> None:
        """Close any open subject whose validity horizon lapsed before ``now`` (never forward-fill)."""
        for s in self._subjects.values():
            if s.open and s.ttl is not None and s.last_seen_t + s.ttl < now:
                s.open = False

    def observe(self, sighting: Sighting) -> Observed:
        """Apply one sighting and return the oracle's classification of it."""
        self._expire(sighting.t)

        # A non-binding speech act must not bind or fire, whatever key it names.
        if sighting.operation in NON_BINDING_OPS:
            out = Observed(sighting.t, None, sighting.binding, False, None, non_binding=True)
            self._observed.append(out)
            return out

        # An unknown binding is the honest third answer: no key, no fire, no fabrication.
        if sighting.binding is Binding.UNKNOWN or not sighting.subject_key:
            self._unknown_count += 1
            out = Observed(sighting.t, None, Binding.UNKNOWN, False, None)
            self._observed.append(out)
            return out

        norm = normalize_key(sighting.subject_key)
        ttl = sighting.ttl if sighting.ttl is not None else self.default_ttl
        subj = self._subjects.get(norm)

        if subj is None:
            self._subjects[norm] = _Subject(
                key=norm, first_key_seen=sighting.subject_key, open=True,
                last_seen_t=sighting.t, ttl=ttl,
            )
            self._fires.append(FireEvent(sighting.t, norm, FireKind.OPEN))
            out = Observed(sighting.t, norm, Binding.RESOLVED, True, FireKind.OPEN)
        elif subj.open:
            # Same subject, still ongoing — refresh the horizon, do NOT re-fire (rising edge only).
            subj.last_seen_t = sighting.t
            subj.ttl = ttl
            out = Observed(sighting.t, norm, Binding.RESOLVED, False, None)
        else:
            # Known subject returning after a close / expiry / restart: re-fire, same durable key.
            subj.open = True
            subj.last_seen_t = sighting.t
            subj.ttl = ttl
            self._fires.append(FireEvent(sighting.t, norm, FireKind.REOPEN))
            out = Observed(sighting.t, norm, Binding.RESOLVED, True, FireKind.REOPEN)

        self._observed.append(out)
        return out

    def close(self, raw_key: str) -> None:
        """Close a subject's current episode. Its next sighting re-fires; its identity is unchanged."""
        s = self._subjects.get(normalize_key(raw_key))
        if s is not None:
            s.open = False

    def restart(self) -> None:
        """Model a process restart: all episodes close, but durable identities SURVIVE.

        After a restart the same content key resolves to the same subject (no new subject minted),
        and its next sighting re-fires as a fresh episode. This is the property that separates a
        durable content key from a volatile session ordinal.
        """
        for s in self._subjects.values():
            s.open = False

    def run(self, sightings: Iterable[Sighting]) -> list[Observed]:
        """Convenience: observe a whole sequence in order."""
        return [self.observe(s) for s in sightings]
