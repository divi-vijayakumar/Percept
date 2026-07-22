"""ComponentCatalog — a READ-ONLY facade over percept's five selectable-component registries.

A client asking "what can I plug in here?" today has to know five different modules exist and
read each one's own registry shape. This module is a single, uniform listing surface over all
five, and it WRITES to none of them — every function here reads a live registry and returns
plain, JSON-shaped :class:`ComponentDescriptor` values; nothing registers, overrides, or tears
down through this module.

The five registries (name -> where it lives -> what it selects):

1. **detectors** (:mod:`percept.harness.detectors`) — cheap, high-fps edge skills a concern
   composes by name (``register_detector`` / ``registered`` / ``list_manifests``).
2. **judges/backends** (:mod:`percept.registry`) — the name -> factory table behind every seam
   (``vision``/``specialist``/``conductor``/``stt``/``tts``/``transport``/``extractor``);
   ``register`` / ``available`` / ``make``.
3. **identity resolvers** (:mod:`percept.envelope`) — :class:`~percept.envelope.SubjectResolver`
   entries in :data:`~percept.envelope.DEFAULT_SUBJECT_RESOLVERS`, refused at registration if any
   would identify a person from pixels.
4. **senses** (:mod:`percept.sourceprofile`) — :class:`~percept.sourceprofile.SourceProfile`
   entries in :data:`~percept.sourceprofile.DEFAULT_SOURCE_REGISTRY`: what KIND of world a stream
   is, and so which modalities/admission strategy it reads with (screen/camera/conversation, ...).
5. **profiles** (:mod:`percept.watchprofile`) — :class:`~percept.watchprofile.WatchProfile`
   entries in :data:`~percept.watchprofile.DEFAULT_REGISTRY`: the empirically-derived
   ``Q(p | environment, goal_type, state)`` viability tuning.

**Measured performance is never a descriptor field.** A :class:`ComponentDescriptor` names and
describes a component; it carries no recall/precision/latency numbers. Anything MEASURED belongs
in a :class:`QualificationRecord`, which is stamped to the exact fixture that produced it
(``fixture_hash``) — auditable provenance, not a floating claim. This module defines the type but
populates none; qualifying a component against a fixture is a later card's concern.

:func:`snapshot` returns a :class:`CatalogSnapshot` — every descriptor plus one stable
``content_hash`` over all of them (:func:`percept.tracing.content_hash`, the same hashing this
package already uses for ``plan_hash``/``component_snapshot_hash``). Two snapshots hash equal iff
the registered COMPONENT SET is byte-identical; this is a value a caller can quote and diff, not
a summary of runtime activity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from . import registry as _backend_registry
from .envelope import DEFAULT_SUBJECT_RESOLVERS
from .harness import detectors as _detectors
from .tracing import content_hash

#: The five roles a :class:`ComponentDescriptor` may carry. Closed here (unlike the registries
#: themselves, which stay open/name-keyed) because this tuple describes THIS module's own survey,
#: not the extensibility of what it surveys.
ROLE_DETECTOR = "detector"
ROLE_BACKEND = "backend"
ROLE_IDENTITY_RESOLVER = "identity_resolver"
ROLE_SENSE = "sense"
ROLE_PROFILE = "profile"

ROLES: tuple[str, ...] = (
    ROLE_DETECTOR,
    ROLE_BACKEND,
    ROLE_IDENTITY_RESOLVER,
    ROLE_SENSE,
    ROLE_PROFILE,
)


@dataclass(frozen=True)
class ComponentDescriptor:
    """One selectable component, named by its registry. Describes IDENTITY, never MEASUREMENT.

    - ``role`` — one of :data:`ROLES`.
    - ``name`` — the registry key this component is selected by.
    - ``seam`` — for a backend, which seam it registers under (``"vision"``, ``"stt"``, ...);
      ``""`` for every other role (a detector/resolver/sense/profile has no seam axis).
    - ``category`` — a coarse class within the role when the source registry has one (a
      detector's sense category, a resolver's ``subject_class``); ``""`` when none applies.
    - ``summary`` — one plain-language line pulled from the source registry's own docs field
      (never authored here).
    - ``metadata`` — whatever else the source registry exposes, plain JSON-shaped. Descriptive
      only — never a measured number.
    """

    role: str
    name: str
    seam: str = ""
    category: str = ""
    summary: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"ComponentDescriptor.role must be one of {ROLES}, got {self.role!r}")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("ComponentDescriptor.name must be a non-empty string")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "name": self.name,
            "seam": self.seam,
            "category": self.category,
            "summary": self.summary,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class QualificationRecord:
    """A MEASURED claim about one component, stamped to the fixture that produced it.

    Deliberately separate from :class:`ComponentDescriptor` — identity and measurement must never
    share a type, or a descriptor becomes a place stale numbers quietly rot. ``fixture_hash`` is a
    :func:`percept.tracing.content_hash` over whatever fixture/eval track produced ``metrics``, so
    a reader can tell exactly which run a number came from (the same provenance discipline
    :mod:`percept.profiles_default` documents for ``WatchProfile.notes``). This module defines the
    shape; nothing here populates one — qualifying a component against a fixture is a later card.
    """

    component_role: str
    component_name: str
    fixture_hash: str
    metrics: Mapping[str, float] = field(default_factory=dict)
    measured_at: float | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.component_role not in ROLES:
            raise ValueError(
                f"QualificationRecord.component_role must be one of {ROLES}, got {self.component_role!r}"
            )
        if not isinstance(self.fixture_hash, str) or not self.fixture_hash:
            raise ValueError("QualificationRecord.fixture_hash must be a non-empty string")
        object.__setattr__(self, "metrics", dict(self.metrics))

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_role": self.component_role,
            "component_name": self.component_name,
            "fixture_hash": self.fixture_hash,
            "metrics": dict(self.metrics),
            "measured_at": self.measured_at,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class CatalogSnapshot:
    """Every :class:`ComponentDescriptor` the catalog can currently see, plus one stable hash.

    ``component_hash`` feeds :attr:`percept.contracts.ResolvedWatch.component_snapshot_hash` in
    spirit (that field is computed by :mod:`percept.resolver` today; wiring the resolver to READ
    this snapshot is a follow-up, not this module's job — see the module docstring). Two snapshots
    hash equal iff their component sets are byte-identical.
    """

    components: tuple[ComponentDescriptor, ...]
    component_hash: str

    def by_role(self, role: str) -> tuple[ComponentDescriptor, ...]:
        return tuple(c for c in self.components if c.role == role)

    def names(self, role: str | None = None) -> list[str]:
        src = self.components if role is None else self.by_role(role)
        return sorted(c.name for c in src)

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_hash": self.component_hash,
            "components": [c.to_dict() for c in self.components],
        }


def _detector_descriptors() -> list[ComponentDescriptor]:
    return [
        ComponentDescriptor(
            role=ROLE_DETECTOR,
            name=manifest.name,
            category=manifest.category,
            summary=manifest.when_to_use,
            metadata={"senses": manifest.senses, "output": manifest.output},
        )
        for manifest in sorted(_detectors.list_manifests(), key=lambda m: m.name)
    ]


def _backend_descriptors() -> list[ComponentDescriptor]:
    out: list[ComponentDescriptor] = []
    for seam in _backend_registry.seams():
        for name in _backend_registry.available(seam):
            out.append(ComponentDescriptor(role=ROLE_BACKEND, name=name, seam=seam))
    return out


def _identity_resolver_descriptors() -> list[ComponentDescriptor]:
    return [
        ComponentDescriptor(
            role=ROLE_IDENTITY_RESOLVER,
            name=resolver.name,
            category=resolver.subject_class,
            summary=resolver.notes,
            metadata={
                "derives_from": resolver.derives_from,
                "binding": resolver.binding,
                "persists": resolver.persists,
            },
        )
        for resolver in DEFAULT_SUBJECT_RESOLVERS.all()
    ]


def _sense_descriptors() -> list[ComponentDescriptor]:
    # Import for its side effect (seeds DEFAULT_SOURCE_REGISTRY) before reading it — mirrors
    # percept.sourceprofile.get_source_profile's own seed-then-resolve pattern.
    from . import source_profiles_default  # noqa: F401
    from .sourceprofile import DEFAULT_SOURCE_REGISTRY

    return [
        ComponentDescriptor(
            role=ROLE_SENSE,
            name=profile.name,
            category=profile.temporal_unit,
            summary=profile.notes,
            metadata={
                "modalities": sorted(profile.modalities),
                "admission_strategy": profile.admission_strategy,
                "evidence_strategy": profile.evidence_strategy,
                "identity_strategy": profile.identity_strategy,
                "stability": profile.stability,
            },
        )
        for profile in DEFAULT_SOURCE_REGISTRY.all()
    ]


def _profile_descriptors() -> list[ComponentDescriptor]:
    # Same seed-then-resolve pattern as _sense_descriptors: percept.profiles_default seeds
    # percept.watchprofile.DEFAULT_REGISTRY as an import side effect.
    from . import profiles_default  # noqa: F401
    from .watchprofile import DEFAULT_REGISTRY

    return [
        ComponentDescriptor(
            role=ROLE_PROFILE,
            name=profile.name,
            category=f"{profile.environment}/{profile.goal_type}/{profile.state}",
            summary=profile.notes,
            metadata={
                "environment": profile.environment,
                "goal_type": profile.goal_type,
                "state": profile.state,
                "plan_mode": profile.plan_mode,
            },
        )
        for profile in DEFAULT_REGISTRY.all()
    ]


_BUILDERS = (
    _detector_descriptors,
    _backend_descriptors,
    _identity_resolver_descriptors,
    _sense_descriptors,
    _profile_descriptors,
)


def list_components(role: str | None = None) -> list[ComponentDescriptor]:
    """Every currently-registered component, optionally filtered to one ``role`` (:data:`ROLES`).

    Reads all five registries fresh on every call — this is a live view, not a cache, so a
    component a client just registered shows up immediately. Sorted by ``(role, seam, name)`` for
    a deterministic listing."""
    if role is not None and role not in ROLES:
        raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")
    out: list[ComponentDescriptor] = []
    for build in _BUILDERS:
        out.extend(build())
    if role is not None:
        out = [c for c in out if c.role == role]
    out.sort(key=lambda c: (c.role, c.seam, c.name))
    return out


def snapshot() -> CatalogSnapshot:
    """The full :class:`CatalogSnapshot` — every component plus one stable content hash."""
    components = tuple(list_components())
    component_hash = content_hash([c.to_dict() for c in components])
    return CatalogSnapshot(components=components, component_hash=component_hash)


__all__ = [
    "ROLES",
    "ROLE_DETECTOR",
    "ROLE_BACKEND",
    "ROLE_IDENTITY_RESOLVER",
    "ROLE_SENSE",
    "ROLE_PROFILE",
    "ComponentDescriptor",
    "QualificationRecord",
    "CatalogSnapshot",
    "list_components",
    "snapshot",
]
