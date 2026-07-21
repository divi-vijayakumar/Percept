"""SourceProfile (card D7, ADR-005) — what KIND of world is on the other end of the stream.

:class:`~percept.watchprofile.WatchProfile` answers ``Q(p | environment, goal, state)`` for a CAMERA:
its ``environment`` axis is literally a camera mount (``fixed_camera`` / ``handheld`` / ``wearable``),
so it cannot express "this stream is a screen recording" or "this stream is a conversation". A
:class:`SourceProfile` is the missing, orthogonal axis: it names the SOURCE CLASS and selects the
strategies that only make sense per source class.

What a SourceProfile selects
----------------------------

- ``admission_strategy`` — WHICH cheap signal decides a frame is worth judging (tier-0 motion,
  a page-change signal, an utterance boundary, ...).
- ``modalities`` / ``temporal_unit`` — what the stream carries and what "one unit of time" means
  on it (a frame, a page, a turn).
- ``evidence_strategy`` — how evidence is packaged for the judge and for the incident.
- ``identity_strategy`` — how two observations are decided to be THE SAME THING.
- ``default_timebase`` — the honest default clock for this source class.
- episode segmentation and sampling shape, carried in :attr:`SourceProfile.parameters`.

What a SourceProfile MUST NOT change (architecturally load-bearing)
-------------------------------------------------------------------

**Gate semantics.** ``FIRE`` / ``ASK`` / ``SILENT``, the rising edge, ``refractory_s`` and dwell mean
exactly the same thing on every source. There is ONE :class:`percept.gate.Gate` and ONE incident
truth — never a conversation-Gate, a document-Gate and a camera-Gate (see
``docs/reference/consequence-gate.md``). This is enforced structurally: this dataclass has no gate
field at all (:data:`FORBIDDEN_FIELDS`, and ``tests/test_sourceprofile.py`` asserts it), so there is
nowhere for a source class to put a special-case firing rule.

An OPEN registry, never a closed enum
-------------------------------------

``SourceProfile.name`` is free text and :class:`SourceProfileRegistry` is keyed by that name. A third
party ships ``industrial_line`` / ``medical_imaging`` / ``satellite_change`` / ``code_editor`` /
``financial_dashboard`` by constructing a ``SourceProfile`` and calling
:meth:`SourceProfileRegistry.register` (or via the ``percept.source_profiles`` entry-point group) —
never by editing a ``Literal`` in this file. The same rule ``WatchProfile``'s free-text axes follow.

Scope of contract v1.4 — narrow on purpose (ADR-006 §Consequences, card Y9)
--------------------------------------------------------------------------

The open registry above is a SEAM, not a promise of coverage. For contract v1.4 percept's subject
handling is scoped to exactly two kinds, and that scope is a decision, not a gap waiting to close:

1. **Screen documents with stable content keys** — the subject is keyed off CONTENT the integrator
   supplies (invoice number, ticket id, file path), never off pixel continuity.
2. **Conversational claim slots with explicit mutation semantics** — a named dialogue slot whose
   value is asserted / revised / retracted, with the mutation DECLARED rather than inferred from
   timestamp order.

**Explicitly EXCLUDED from v1.4** — scope, not backlog: biometric re-identification (deciding two
observations are the same *person*), growth or trajectory prediction (inferring a subject's future
value), and robotics control (closing a loop from a finding back onto an actuator). Percept does not
ship these and does not partially ship them; ``identity_strategy`` being free text must not be read
as a step toward them.

The resolver interface stays open so a future scheme can be REGISTERED rather than merged into this
file — but v1.4 deliberately does not generalise ahead of evidence. Two client probes are the
evidence we have, and the two kinds above are exactly what they established.

Identifier stability is not restart-safety
------------------------------------------

Where a subject reference carries a ``scope='durable'``-style marker, that describes **IDENTIFIER
STABILITY** — the string ``PO-4471`` denotes the same real-world subject next week as it does today —
and NOTHING ELSE. It is not a claim that any runtime state about that subject survives a process
restart: episode logs are session-only and are not carried by ``Percept.snapshot()`` (see
:mod:`percept.episodes`). Conflating the two reads as continuous 24/7 operation and shows up as
silent under-counting after a redeploy.

Honesty about admission
-----------------------

Some source classes have NO admission signal percept can ship. Screen/document content is the
measured case (ADR-005 §Context): the tier-0 gate frame is 48x36 greyscale and ``block_max`` reads an
8x6 grid over it, so a document-sized edit on a 1920x1080 screen recording is diluted below every
calibrated cutoff. A profile in that position declares itself :data:`UNTUNED` rather than pretending
to a threshold nobody measured — ``percept doctor`` reads that declaration and says "this profile is
running on coast alone, supply a change signal" instead of leaving the client to guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterator, Mapping

#: ``parameters["change_signal"] == UNTUNED`` means: percept ships NO calibrated admission signal for
#: this source class and the profile says so out loud. It is not a placeholder to be quietly filled by
#: a guessed threshold — it is the honest state until the CLIENT registers a detector on THEIR content
#: (docs/how-to/tune-admission.md). Doctor reports it; nothing silently "fixes" it.
UNTUNED = "untuned"

#: Stability vocabulary carried on :attr:`SourceProfile.stability`. ``experimental`` profiles are
#: shipped for evaluation only: their strategies may change without a contract bump.
STABILITIES: tuple[str, ...] = ("stable", "experimental")

#: Timebase vocabulary — the same three values ``percept.runtime.worker._resolve_timebase`` accepts.
TIMEBASES: tuple[str, ...] = ("auto", "media", "wall")

#: Names a SourceProfile may never carry. Gate semantics are source-independent by construction
#: (module docstring, ADR-005 §Decision 2); a profile that could carry one of these would be a
#: per-source gate in disguise.
FORBIDDEN_FIELDS: frozenset[str] = frozenset(
    {
        "gate",
        "require_false_first",
        "refractory_s",
        "dwell_s",
        "gate_action",
        "fire",
        "ask",
        "silent",
        "rising_edge",
    }
)


class SourceProfileError(ValueError):
    """A :class:`SourceProfile` (or a registry operation on one) failed validation."""


class SourceProfileNotFoundError(KeyError):
    """No :class:`SourceProfile` is registered under a requested name."""


def _freeze(parameters: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return an immutable view of ``parameters`` (a frozen dataclass with a mutable dict inside is
    not frozen). Nested dict/list values are frozen one level down for the same reason."""
    if parameters is None:
        return MappingProxyType({})
    frozen: dict[str, Any] = {}
    for key, value in parameters.items():
        if not isinstance(key, str) or not key.strip():
            raise SourceProfileError(f"parameters keys must be non-empty strings, got {key!r}")
        if isinstance(value, Mapping):
            frozen[key] = MappingProxyType(dict(value))
        elif isinstance(value, (list, set)):
            frozen[key] = tuple(value)
        else:
            frozen[key] = value
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class SourceProfile:
    """A named, typed description of a SOURCE CLASS and the strategies it selects.

    Every field is free text on purpose (see the module docstring): the registry is open, so adding
    ``industrial_line`` never requires editing this file. Validation here is about SHAPE (non-empty
    names, a real timebase, immutable parameters), never about membership in a closed vocabulary.

    - ``name`` — the registry key, e.g. ``"physical_scene"``, ``"screen_document"``.
    - ``modalities`` — what the stream carries, e.g. ``frozenset({"video"})``,
      ``frozenset({"audio", "text"})``.
    - ``temporal_unit`` — what one unit of time IS on this source: ``"frame"``, ``"page"``, ``"turn"``.
    - ``admission_strategy`` — the name of the cheap signal that decides "worth judging".
    - ``evidence_strategy`` — how evidence is packaged for judge + incident.
    - ``identity_strategy`` — how two observations are decided to be the same entity.
    - ``default_timebase`` — ``"auto"`` / ``"media"`` / ``"wall"``.
    - ``parameters`` — everything strategy-specific (coast interval, episode unit, change signal,
      ...). Immutable; read by name, never by position.
    - ``stability`` — ``"stable"`` or ``"experimental"``; experimental profiles ship for evaluation.
    - ``notes`` — provenance/honesty text. Doctor and the ADR both quote it.
    """

    name: str
    modalities: frozenset[str]
    temporal_unit: str
    admission_strategy: str
    evidence_strategy: str
    identity_strategy: str
    default_timebase: str = "auto"
    parameters: Mapping[str, Any] = field(default_factory=dict)
    stability: str = "stable"
    notes: str = ""

    def __post_init__(self) -> None:
        for label in (
            "name",
            "temporal_unit",
            "admission_strategy",
            "evidence_strategy",
            "identity_strategy",
        ):
            value = getattr(self, label)
            if not isinstance(value, str) or not value.strip():
                raise SourceProfileError(
                    f"SourceProfile.{label} must be a non-empty string, got {value!r}"
                )
        modalities = self.modalities
        if isinstance(modalities, (set, list, tuple)):
            modalities = frozenset(modalities)
        if not isinstance(modalities, frozenset) or not modalities:
            raise SourceProfileError(
                f"SourceProfile.modalities must be a non-empty frozenset of strings, got "
                f"{self.modalities!r}"
            )
        for m in modalities:
            if not isinstance(m, str) or not m.strip():
                raise SourceProfileError(f"modality must be a non-empty string, got {m!r}")
        object.__setattr__(self, "modalities", modalities)
        if self.default_timebase not in TIMEBASES:
            raise SourceProfileError(
                f"SourceProfile.default_timebase must be one of {TIMEBASES}, got "
                f"{self.default_timebase!r}"
            )
        if self.stability not in STABILITIES:
            raise SourceProfileError(
                f"SourceProfile.stability must be one of {STABILITIES}, got {self.stability!r}"
            )
        params = _freeze(self.parameters)
        clashes = FORBIDDEN_FIELDS & set(params)
        if clashes:
            raise SourceProfileError(
                f"SourceProfile.parameters may not carry gate semantics {sorted(clashes)}: "
                "FIRE/ASK/SILENT, the rising edge, refractory and dwell mean the same thing on "
                "every source (ADR-005 Decision 2). A source profile selects admission, sampling, "
                "evidence, identity, episodes and timebase — nothing else."
            )
        object.__setattr__(self, "parameters", params)

    # -- introspection ------------------------------------------------------

    @property
    def admission_tuned(self) -> bool:
        """``False`` when this profile DECLARES that percept ships no calibrated admission signal
        for it (``parameters["change_signal"] == UNTUNED``, or the strategy itself is
        :data:`UNTUNED`). A false here is a fact to report, never a threshold to invent."""
        return not (
            self.admission_strategy == UNTUNED or self.parameters.get("change_signal") == UNTUNED
        )

    @property
    def coast_interval_s(self) -> float | None:
        """The fixed low-rate sampling floor, if this profile declares one. The coast is what keeps a
        static-but-important source observed when the change signal is absent or untuned."""
        value = self.parameters.get("coast_interval_s")
        return None if value is None else float(value)

    def admission_note(self) -> str:
        """One sentence a client can act on, for doctor and for logs."""
        if self.admission_tuned:
            return f"admission via '{self.admission_strategy}'"
        coast = self.coast_interval_s
        coast_txt = (
            f"running on the fixed coast alone (every {coast:g}s)"
            if coast
            else "running with NO admission signal at all"
        )
        return (
            f"admission signal for source profile '{self.name}' is UNTUNED — {coast_txt}. percept "
            "ships no calibrated change signal for this source class; supply one with "
            "register_detector(...) and A/B it with `percept eval` "
            "(docs/how-to/tune-admission.md)."
        )

    def with_parameters(self, **overrides: Any) -> "SourceProfile":
        """A copy with ``parameters`` updated — how a client tunes a shipped profile for THEIR
        content without editing percept (``SCREEN_DOCUMENT.with_parameters(change_signal="my-phash",
        coast_interval_s=1.0)``)."""
        merged = dict(self.parameters)
        merged.update(overrides)
        return SourceProfile(
            name=self.name,
            modalities=self.modalities,
            temporal_unit=self.temporal_unit,
            admission_strategy=self.admission_strategy,
            evidence_strategy=self.evidence_strategy,
            identity_strategy=self.identity_strategy,
            default_timebase=self.default_timebase,
            parameters=merged,
            stability=self.stability,
            notes=self.notes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "modalities": sorted(self.modalities),
            "temporal_unit": self.temporal_unit,
            "admission_strategy": self.admission_strategy,
            "evidence_strategy": self.evidence_strategy,
            "identity_strategy": self.identity_strategy,
            "default_timebase": self.default_timebase,
            "parameters": {k: (dict(v) if isinstance(v, Mapping) else v)
                           for k, v in self.parameters.items()},
            "stability": self.stability,
            "admission_tuned": self.admission_tuned,
            "notes": self.notes,
        }


@dataclass
class SourceProfileRegistry:
    """An OPEN, name-keyed table of :class:`SourceProfile`.

    Open means: no closed enum anywhere, and no central literal to edit. ``register()`` refuses to
    silently replace a DIFFERENT profile under a name that is already taken (pass ``overwrite=True``
    to mean it) — the same fail-loud rule
    :meth:`percept.watchprofile.WatchProfileRegistry.register` follows.
    """

    _profiles: dict[str, SourceProfile] = field(default_factory=dict)

    def register(self, profile: SourceProfile, *, overwrite: bool = False) -> None:
        if not isinstance(profile, SourceProfile):
            raise SourceProfileError(f"expected a SourceProfile, got {type(profile).__name__}")
        existing = self._profiles.get(profile.name)
        if existing is not None and existing != profile and not overwrite:
            raise SourceProfileError(
                f"a source profile named {profile.name!r} is already registered; pass "
                "overwrite=True to replace it"
            )
        self._profiles[profile.name] = profile

    def unregister(self, name: str) -> None:
        self._profiles.pop(name, None)

    def get(self, name: str) -> SourceProfile | None:
        return self._profiles.get(name)

    def resolve(self, name: str) -> SourceProfile:
        """Name lookup; raises :class:`SourceProfileNotFoundError` (listing what IS registered) so a
        typo at 2am is a readable error, not a silent fallback to the wrong source class."""
        profile = self._profiles.get(name)
        if profile is None:
            raise SourceProfileNotFoundError(
                f"no source profile named {name!r}; registered: {sorted(self._profiles)}"
            )
        return profile

    def names(self) -> list[str]:
        return sorted(self._profiles)

    def all(self) -> list[SourceProfile]:
        return list(self._profiles.values())

    def __contains__(self, name: object) -> bool:
        return name in self._profiles

    def __len__(self) -> int:
        return len(self._profiles)

    def __iter__(self) -> Iterator[SourceProfile]:
        return iter(self._profiles.values())


#: The process-wide registry. ``percept.source_profiles_default`` seeds the shipped profiles into it
#: at import; third parties add to it with :func:`register_source_profile`.
DEFAULT_SOURCE_REGISTRY = SourceProfileRegistry()


def register_source_profile(profile: SourceProfile, *, overwrite: bool = False) -> None:
    """Add ``profile`` to :data:`DEFAULT_SOURCE_REGISTRY` — the third-party extension point."""
    DEFAULT_SOURCE_REGISTRY.register(profile, overwrite=overwrite)


def get_source_profile(name: str) -> SourceProfile:
    """Resolve ``name`` against :data:`DEFAULT_SOURCE_REGISTRY` (seeding the shipped profiles first
    so the shipped names always resolve, whatever import order the caller used)."""
    from . import source_profiles_default  # noqa: F401, PLC0415 — import seeds the registry

    return DEFAULT_SOURCE_REGISTRY.resolve(name)


def coast_sense_prior(profile: SourceProfile) -> dict[str, float] | None:
    """The goal ``state["sense"]`` fragment that ARMS this profile's fixed low-rate coast on the seam
    the runtime already has — or ``None`` when the profile declares no coast.

    The coast is not a new mechanism invented by this card. ``percept.runtime.worker._capture_plan``
    already builds a :class:`percept.sampling.SamplingController` from a goal's authored
    ``sense.base_fps`` (via :func:`percept.sensorplan.sensor_plan_for`), and the worker's
    ``coast_admit`` branch forces a judge on a QUIET, sub-threshold frame the motion gate would skip.
    That is exactly what a static-but-important page needs. This function only translates
    ``parameters["coast_interval_s"]`` into that existing prior::

        goal.state["sense"] = coast_sense_prior(SCREEN_DOCUMENT)   # -> {"base_fps": 0.5, ...}

    ``max_fps`` is left at the coast rate: the coast is a FLOOR that guarantees coverage, not a
    licence to spend — a real change signal, once the client registers one, is what should escalate
    above it.
    """
    coast = profile.coast_interval_s
    if not coast or coast <= 0:
        return None
    fps = 1.0 / float(coast)
    return {"base_fps": fps, "max_fps": fps}


def discover_source_profile_entrypoints() -> None:
    """Load third-party source profiles advertised under the ``percept.source_profiles`` entry-point
    group. Each entry point is called with :func:`register_source_profile`. One bad plugin never
    breaks the others (the same contract :func:`percept.registry.discover_entrypoints` follows)."""
    from importlib import metadata as importlib_metadata  # noqa: PLC0415

    try:
        eps = importlib_metadata.entry_points(group="percept.source_profiles")
    except TypeError:  # pragma: no cover — older importlib.metadata signature
        eps = importlib_metadata.entry_points().get("percept.source_profiles", [])  # type: ignore[attr-defined]
    for ep in eps:
        try:
            ep.load()(register_source_profile)
        except Exception:  # noqa: BLE001 — a bad plugin must not break the core
            continue


__all__ = [
    "UNTUNED",
    "STABILITIES",
    "TIMEBASES",
    "FORBIDDEN_FIELDS",
    "SourceProfile",
    "SourceProfileError",
    "SourceProfileNotFoundError",
    "SourceProfileRegistry",
    "DEFAULT_SOURCE_REGISTRY",
    "register_source_profile",
    "get_source_profile",
    "coast_sense_prior",
    "discover_source_profile_entrypoints",
]
