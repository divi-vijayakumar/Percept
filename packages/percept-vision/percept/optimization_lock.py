"""OptimizationLock (ADR-008) — a validated, hashed selection of tunables an optimizer proved safe.

``docs/adr/ADR-008-configuration-ownership-resolution-provenance.md`` names :class:`OptimizationLock`
as "a validated, hashed selection of tunables an optimizer previously proved safe — never authored
by hand... new — not yet in code" and assigns it a precedence slot between ``ExecutionPolicy`` and
``Deployment``. :mod:`percept.resolver` previously carried a MINIMAL placeholder (``id``/``version``/
``values``/``locked_hash`` only — "the precedence chain is real and testable now; its full contract
(optimizer validation logic) was deliberately not built there"). This module now
OWNS that class (mirroring :class:`percept.domain_pack.DomainPack` / :class:`percept.execution_policy.
ExecutionPolicy`'s own earlier graduation out of the same placeholder set); :mod:`percept.resolver`
imports it from here and still owns the two enforcement gates that depend on the tunables registry
(unchanged in behavior, not duplicated in this module):

- **Ownership**: an :class:`OptimizationLock` may cover ANY path ``tunables.json`` marks
  ``safe_to_optimize: true``, regardless of that path's nominal owner — checked at RESOLUTION time
  (:func:`percept.resolver._check_optimization_lock_ownership`), not at construction, because it
  depends on :data:`percept.tunables.REGISTRY`, the single source of truth this object does not
  duplicate (unchanged from the placeholder's own documented reasoning).
- **Invalidation** (ADR-008 Corrections(d)): a ``Deployment``/env change touching a field this lock
  covers must surface as a loud diagnostic, never a silent outranking — enforced by
  :meth:`percept.resolver.ConfigResolver.resolve`'s merge loop, which now also compares a fresh
  :attr:`content_hash` against a supplied :attr:`locked_hash` (a stale claim on the lock's OWN
  values) *and* watches for a ``Deployment``-tier contribution silently overriding a
  ``OptimizationLock``-tier value (an outside change to the SAME field).

This graduation carries the placeholder's shape forward into the real one ADR-008 §OptimizationLock
names: a lock now also carries the PROVENANCE an optimizer run actually produces —
``base_plan_hash``/``catalog_component_hash``/``fixture_hash`` (what this selection was validated
against) and ``held_out_metrics``/``created_at`` (what it measured, and when the caller says it was
measured — never wall-clock, the same "created_at from caller" discipline
:func:`percept.profile_versioning.stamp` already holds). ``values``/``locked_hash`` keep their exact
placeholder meaning and shape — no existing caller of the placeholder constructor breaks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from .tracing import content_hash

#: Schema id + version stamped into a lock's on-disk/wire JSON (``percept optimize --out-lock``) —
#: ADR-008's version ledger assigns this schema to the OptimizationLock implementation.
LOCK_SCHEMA = "percept.optimization_lock/1"
LOCK_SCHEMA_VERSION = 1


class OptimizationLockError(ValueError):
    """An :class:`OptimizationLock` failed a shape/consistency check."""


def _freeze_values(values: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if values is None:
        return MappingProxyType({})
    if not isinstance(values, Mapping):
        raise OptimizationLockError(f"values must be a mapping, got {type(values).__name__}")
    for key in values:
        if not isinstance(key, str) or not key.strip():
            raise OptimizationLockError(f"values keys must be non-empty strings, got {key!r}")
    return MappingProxyType(dict(values))


def _freeze_metrics(metrics: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if metrics is None:
        return MappingProxyType({})
    if not isinstance(metrics, Mapping):
        raise OptimizationLockError(f"held_out_metrics must be a mapping, got {type(metrics).__name__}")
    return MappingProxyType(dict(metrics))


@dataclass(frozen=True)
class OptimizationLock:
    """A validated, hashed selection of ``safe_to_optimize`` tunables (ADR-008).

    - ``id`` / ``version`` — identity; ``version`` defaults to ``"v1"`` (the placeholder's own
      default, unchanged).
    - ``values`` — ``{tunables-path: value}``. Every path must be ``safe_to_optimize: true`` in
      :data:`percept.tunables.REGISTRY` — checked at RESOLUTION time (module docstring), not here.
    - ``locked_hash`` — the hash this lock CLAIMS its ``values`` validated under; compared against a
      freshly computed :attr:`content_hash` by the resolver (a mismatch is a loud diagnostic, never
      a silently trusted stale claim — ADR-008 Corrections(d)).
    - ``base_plan_hash`` — the :class:`~percept.contracts.watchplan.WatchPlan` content hash this
      selection was optimized/validated against (``percept.tracing.content_hash(plan.to_dict())``,
      the same function :class:`~percept.resolver.ConfigResolver` uses for ``plan_hash``).
    - ``catalog_component_hash`` — :func:`percept.catalog.snapshot().component_hash` at validation
      time: which registered component SET this selection was proved against.
    - ``fixture_hash`` — :func:`percept.eval.derivation.fixture_set_hash` over the labeled fixtures
      this selection was scored on (the SAME anti-tune-to-green content address a
      :class:`~percept.eval.derivation.DerivationPreRegistration` stamps).
    - ``held_out_metrics`` — ``{metric: value}`` the winning candidate measured on the held-out split
      (e.g. ``recall``/``precision``/``false_fires``/``cost_usd`` — the same vocabulary
      :meth:`percept.eval.derivation.RungResult.mean` reports).
    - ``created_at`` — supplied by the CALLER, never wall-clock (mirrors
      :func:`percept.profile_versioning.stamp`'s own ``created_at`` discipline: a lock's provenance
      must be reproducible from its inputs, not from when the process happened to run).
    """

    id: str
    version: str = "v1"
    values: Mapping[str, Any] = field(default_factory=dict)
    locked_hash: str | None = None
    base_plan_hash: str = ""
    catalog_component_hash: str = ""
    fixture_hash: str = ""
    held_out_metrics: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise OptimizationLockError(f"OptimizationLock.id must be a non-empty string, got {self.id!r}")
        if not isinstance(self.version, str) or not self.version.strip():
            raise OptimizationLockError(
                f"OptimizationLock.version must be a non-empty string, got {self.version!r}"
            )
        object.__setattr__(self, "values", _freeze_values(self.values))
        object.__setattr__(self, "held_out_metrics", _freeze_metrics(self.held_out_metrics))

    @property
    def content_hash(self) -> str:
        """The hash this lock's OWN ``values`` hash to right now — compared against
        :attr:`locked_hash` by the resolver (ADR-008 Corrections(d)). Deliberately over ``values``
        alone (not the provenance fields): the invalidation contract is about the TUNABLE SELECTION
        moving underneath the lock, not about its descriptive metadata."""
        return content_hash(dict(sorted(self.values.items())))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": LOCK_SCHEMA,
            "schema_version": LOCK_SCHEMA_VERSION,
            "id": self.id,
            "version": self.version,
            "values": dict(self.values),
            "locked_hash": self.locked_hash,
            "content_hash": self.content_hash,
            "base_plan_hash": self.base_plan_hash,
            "catalog_component_hash": self.catalog_component_hash,
            "fixture_hash": self.fixture_hash,
            "held_out_metrics": dict(self.held_out_metrics),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OptimizationLock":
        return cls(
            id=data["id"],
            version=str(data.get("version") or "v1"),
            values=data.get("values") or {},
            locked_hash=data.get("locked_hash"),
            base_plan_hash=str(data.get("base_plan_hash") or ""),
            catalog_component_hash=str(data.get("catalog_component_hash") or ""),
            fixture_hash=str(data.get("fixture_hash") or ""),
            held_out_metrics=data.get("held_out_metrics") or {},
            created_at=str(data.get("created_at") or ""),
        )


__all__ = ["LOCK_SCHEMA", "LOCK_SCHEMA_VERSION", "OptimizationLockError", "OptimizationLock"]
