"""ExecutionPolicy (ADR-008) — a named, versioned bundle of EXISTING resource/scheduling knobs.

ADR-008 names :class:`ExecutionPolicy` as "resource/scheduling posture: judge-call budget, cadence
ceilings, concurrency, degrade-under-load behavior... new — not yet in code" and assigns it, via
:data:`percept.tunables.REGISTRY`, ownership of exactly 13 tunables-registry paths:

- ``WatchProfile.cadence.{base_fps,max_fps,escalate_threshold,hold_s}`` — the fovea's
  processing-density envelope (:class:`percept.watchprofile.CadenceProfile` /
  :class:`percept.sampling.SensePolicy`).
- ``Settings.cache.{enabled,threshold,ttl_s,tau_bits,audit_rate}`` — the judge-dispatch cost
  cascade (:mod:`percept.cache`'s FrameHashLayer short-circuit / backpressure).
- ``Settings.routing.confidence_threshold`` — the local-router defer-on-miss cutoff (judge
  dispatch mode: route locally vs. defer to the conductor).
- ``Settings.entity.max_entities`` / ``Settings.episodes.max_episodes`` /
  ``Settings.events.preroll_max_bytes`` — resource/retention budgets (eviction ceilings, the
  incident pre-roll ring's memory bound).

This module ships exactly that: a typed :class:`ExecutionPolicy` dataclass (an ``id``/``version``/
``values`` bundle, same shape :mod:`percept.resolver` already accepted as a placeholder — this
module now OWNS that class; :mod:`percept.resolver` imports it from here) plus four named,
versioned instances built ONLY from the 13 paths above. No new temporal shapes, no
``event_capsule`` defaults (validated-but-not-defaulted per the capsule card) — every value here is
an existing ``Settings``/``WatchProfile.cadence`` field, nothing invented.

Ownership is enforced where it always has been: :meth:`percept.resolver.ConfigResolver.resolve`
checks every ``execution_policy.values`` path against :data:`percept.tunables.REGISTRY`'s owner for
that path (:func:`percept.resolver._check_ownership`) and raises :class:`percept.resolver.ConfigError`
for a path this module does not own — this module does not re-implement that check, only supplies
values a real :class:`ExecutionPolicy` is allowed to carry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterator, Mapping


class ExecutionPolicyError(ValueError):
    """An :class:`ExecutionPolicy` (or the registry) failed a shape/consistency check."""


class ExecutionPolicyNotFoundError(KeyError):
    """No :class:`ExecutionPolicy` matches a registry lookup."""


def _freeze_values(values: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if values is None:
        return MappingProxyType({})
    if not isinstance(values, Mapping):
        raise ExecutionPolicyError(f"values must be a mapping, got {type(values).__name__}")
    for key in values:
        if not isinstance(key, str) or not key.strip():
            raise ExecutionPolicyError(f"values keys must be non-empty strings, got {key!r}")
    return MappingProxyType(dict(values))


@dataclass(frozen=True)
class ExecutionPolicy:
    """A named, versioned bundle of EXISTING ``ExecutionPolicy``-owned tunables (ADR-008).

    - ``id`` / ``version`` — together name the policy (:attr:`qualified_name` renders
      ``f"{id}@{version}"``, e.g. ``"balanced@1"``); this is a label, not itself a resolved value.
    - ``values`` — ``{tunables-registry path: value}``, the SAME dotted vocabulary
      :attr:`~percept.contracts.WatchPlan.explicit` uses. Every path here must resolve to owner
      ``"ExecutionPolicy"`` (or ``"unclassified"``) in :data:`percept.tunables.REGISTRY` —
      enforced at :meth:`percept.resolver.ConfigResolver.resolve` time, not construction time (this
      object does not import the registry, mirroring :class:`percept.resolver.DomainPack`).
    - ``notes`` — one plain-language line describing the posture this bundle encodes; never a
      measured claim (mirrors :attr:`percept.sourceprofile.SourceProfile.notes`).
    """

    id: str
    version: str = "1"
    values: Mapping[str, Any] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ExecutionPolicyError(f"ExecutionPolicy.id must be a non-empty string, got {self.id!r}")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ExecutionPolicyError(
                f"ExecutionPolicy.version must be a non-empty string, got {self.version!r}"
            )
        object.__setattr__(self, "values", _freeze_values(self.values))

    @property
    def qualified_name(self) -> str:
        """``"{id}@{version}"`` — the registry key and the card-vocabulary name (``"balanced@1"``)."""
        return f"{self.id}@{self.version}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "qualified_name": self.qualified_name,
            "values": dict(self.values),
            "notes": self.notes,
        }


@dataclass
class ExecutionPolicyRegistry:
    """An OPEN, ``qualified_name``-keyed table of :class:`ExecutionPolicy` — same shape and same
    fail-loud-on-silent-replace rule as :class:`percept.sourceprofile.SourceProfileRegistry` /
    :class:`percept.watchprofile.WatchProfileRegistry`."""

    _policies: dict[str, ExecutionPolicy] = field(default_factory=dict)

    def register(self, policy: ExecutionPolicy, *, overwrite: bool = False) -> None:
        if not isinstance(policy, ExecutionPolicy):
            raise ExecutionPolicyError(f"expected an ExecutionPolicy, got {type(policy).__name__}")
        existing = self._policies.get(policy.qualified_name)
        if existing is not None and existing != policy and not overwrite:
            raise ExecutionPolicyError(
                f"an execution policy named {policy.qualified_name!r} is already registered; pass "
                "overwrite=True to replace it"
            )
        self._policies[policy.qualified_name] = policy

    def unregister(self, qualified_name: str) -> None:
        self._policies.pop(qualified_name, None)

    def get(self, qualified_name: str) -> ExecutionPolicy | None:
        return self._policies.get(qualified_name)

    def resolve(self, qualified_name: str) -> ExecutionPolicy:
        """Name lookup; raises :class:`ExecutionPolicyNotFoundError` (listing what IS registered)."""
        policy = self._policies.get(qualified_name)
        if policy is None:
            raise ExecutionPolicyNotFoundError(
                f"no execution policy named {qualified_name!r}; registered: {sorted(self._policies)}"
            )
        return policy

    def names(self) -> list[str]:
        return sorted(self._policies)

    def all(self) -> list[ExecutionPolicy]:
        return list(self._policies.values())

    def __contains__(self, qualified_name: object) -> bool:
        return qualified_name in self._policies

    def __len__(self) -> int:
        return len(self._policies)

    def __iter__(self) -> Iterator[ExecutionPolicy]:
        return iter(self._policies.values())


#: The process-wide registry. ``percept.execution_policies_default`` seeds the shipped policies
#: into it at import; third parties add to it with :func:`register_execution_policy`.
DEFAULT_EXECUTION_POLICY_REGISTRY = ExecutionPolicyRegistry()


def register_execution_policy(policy: ExecutionPolicy, *, overwrite: bool = False) -> None:
    """Add ``policy`` to :data:`DEFAULT_EXECUTION_POLICY_REGISTRY` — the third-party extension point."""
    DEFAULT_EXECUTION_POLICY_REGISTRY.register(policy, overwrite=overwrite)


def get_execution_policy(qualified_name: str) -> ExecutionPolicy:
    """Resolve ``qualified_name`` (e.g. ``"balanced@1"``) against
    :data:`DEFAULT_EXECUTION_POLICY_REGISTRY`, seeding the shipped policies first so the shipped
    names always resolve, whatever import order the caller used."""
    from . import execution_policies_default  # noqa: F401, PLC0415 — import seeds the registry

    return DEFAULT_EXECUTION_POLICY_REGISTRY.resolve(qualified_name)


__all__ = [
    "ExecutionPolicyError",
    "ExecutionPolicyNotFoundError",
    "ExecutionPolicy",
    "ExecutionPolicyRegistry",
    "DEFAULT_EXECUTION_POLICY_REGISTRY",
    "register_execution_policy",
    "get_execution_policy",
]
