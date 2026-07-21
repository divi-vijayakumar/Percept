"""Envelope — the three OUT verdicts, enforced in CONFIG rather than in prose (card Z7, ADR-007).

The seven-domain evaluation returned **OUT** for three domains. Each OUT has a NARROWED survivor that
is often one config change away from the excluded form, which is exactly why a paragraph in a README
is not enforcement. This module is where the refusal lives as TYPE and CODE:

===========================  =========================================================================
domain                       why OUT / what survives
===========================  =========================================================================
``security_person_id``       24/7 security **identifying people**: biometric re-identification of
                             non-consenting third parties, AND a safety mechanism on a clock percept
                             cannot guarantee. Survivor: scene-local, non-persisting individuation
                             ("someone is at the door", "the same person is still there") with NO
                             cross-session person identity. **Structurally enforced here**:
                             :class:`SubjectResolver` refuses to construct a person-class resolver
                             that derives identity from pixels or persists it across sessions, and
                             :data:`DEFAULT_SUBJECT_RESOLVERS` ships none.
``robotics_control_loop``    Goal.modality ``"act"`` closing a loop onto PHYSICAL actuation: physical
                             consequence on an unguaranteeable clock — and mechanically incoherent,
                             because the gate's honest failure mode is **ASK** and a control loop
                             cannot ask. Survivor: advisory acts (screen/software) and human-in-the-
                             loop notification. **Declared + enforced**: a
                             :class:`~percept.consent.ConsentProfile` carries a typed ``actuation``
                             field; ``actuation="physical"`` makes ``permits()`` return ``False`` for
                             every action, and ``percept doctor`` reports it.
``passive_analytics``        Passive customer analytics: the beneficiary is the business, not the
                             observed person. Survivor: an operator-facing watch on a condition the
                             OBSERVED person also benefits from (a queue that got too long, a spill,
                             an unattended child) with no per-person record. The line is the person
                             resolver — see ``security_person_id``.
===========================  =========================================================================

The precedent is ADR-005 ``FORBIDDEN_FIELDS`` and :data:`~percept.sourceprofile.UNTUNED`: structural
enforcement where the mechanism allows it, and a TYPED declaration a tool can read where it does not.
Nothing here is a lint rule over prose.

What percept will never do
--------------------------

percept never derives PERSON IDENTITY from pixels. A client's own identity — a badge scan, their own
re-ID service, a login — may enter as a **client-asserted key**::

    key = client_asserted_subject(value="emp-4417", asserted_by="acme-badge-reader")
    key.binding      # "bound"  -- because THEIR system asserted it, not because percept guessed

``binding="bound"`` is reachable ONLY via ``derives_from="client_asserted"`` with a non-empty
``asserted_by``; a pixel-derived person key cannot be constructed at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterator, Mapping

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

#: How much of an envelope refusal is machine-enforced. ``structural`` — the excluded thing cannot be
#: constructed/registered at all; ``declared`` — it is a typed field a tool reads and reports, and the
#: enforcement it drives is deterministic; ``docs`` — honest prose only, with no config surface yet
#: (see :data:`ATTENTION_CLASS_ENFORCEMENT`).
ENFORCEMENTS: tuple[str, ...] = ("structural", "declared", "docs")

#: Where a subject key came from. ``pixels`` — derived by percept from the image; ``scene_local`` —
#: transient within-run individuation (tracking/diarization) that dies with the run; ``client_asserted``
#: — supplied by the CLIENT's own system.
IDENTITY_SOURCES: tuple[str, ...] = ("pixels", "scene_local", "client_asserted")

#: ``unbound`` — a handle that correlates observations and means nothing outside this run.
#: ``bound`` — a key that names a real-world subject. Reachable only from ``client_asserted``.
IDENTITY_BINDINGS: tuple[str, ...] = ("unbound", "bound")

#: Actuation classes a deployment may declare. ``none`` — percept only speaks. ``screen`` — advisory
#: software/computer-use acts (the survivor). ``physical`` — actuation with physical consequence,
#: which is OUT of the envelope (:data:`OUT_OF_ENVELOPE`\\ ``["robotics_control_loop"]``).
ACTUATION_CLASSES: tuple[str, ...] = ("none", "screen", "physical")

#: The one actuation class that is out of the envelope. Named so callers compare to a constant.
PHYSICAL_ACTUATION = "physical"

#: Subject classes that carry a person. A resolver over these is the one percept refuses to ship.
PERSON_SUBJECT_CLASSES: frozenset[str] = frozenset({"person", "face", "voice", "biometric"})


class EnvelopeViolation(ValueError):
    """A construction/registration that would put percept outside its declared envelope.

    Raised, never warned: the whole point of card Z7 is that the refusal survives a client wiring
    something at 2am with no access to us.
    """


# ---------------------------------------------------------------------------
# The three verdicts, as data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvelopeVerdict:
    """One OUT verdict from the seven-domain evaluation, in a shape a tool can print.

    ``narrowed_survivor`` is the whole reason this is typed: the survivor is usually ONE config change
    from the excluded form, so a client needs the boundary spelled out beside the refusal, not in a
    different document.
    """

    domain: str
    verdict: str
    reason: str
    narrowed_survivor: str
    boundary: str
    enforcement: str

    def __post_init__(self) -> None:
        if self.enforcement not in ENFORCEMENTS:
            raise EnvelopeViolation(
                f"EnvelopeVerdict.enforcement must be one of {ENFORCEMENTS}, got "
                f"{self.enforcement!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "verdict": self.verdict,
            "reason": self.reason,
            "narrowed_survivor": self.narrowed_survivor,
            "boundary": self.boundary,
            "enforcement": self.enforcement,
        }

    def sentence(self) -> str:
        """One line for doctor/logs: what is refused, what survives, and where the line is."""
        return (
            f"OUT OF ENVELOPE [{self.domain}] ({self.enforcement}): {self.reason} "
            f"NARROWED SURVIVOR: {self.narrowed_survivor} BOUNDARY: {self.boundary}"
        )


_SECURITY_PERSON_ID = EnvelopeVerdict(
    domain="security_person_id",
    verdict="out",
    reason=(
        "24/7 security that IDENTIFIES PEOPLE is out: it is biometric re-identification of "
        "non-consenting third parties, and it is a safety mechanism running on a clock percept "
        "cannot guarantee (percept is best-effort; it has no bounded detection latency to offer)."
    ),
    narrowed_survivor=(
        "scene-local, non-persisting individuation — 'someone is at the door', 'the same person is "
        "still standing there' — with no cross-session person identity and no per-person record."
    ),
    boundary=(
        "the person resolver. A subject key for a person may never be derived from pixels and may "
        "never persist across sessions; a client's OWN identity (badge scan, their re-ID service) "
        "may enter as a client-asserted bound key. percept ships no person-identifying resolver."
    ),
    enforcement="structural",
)

_ROBOTICS_CONTROL_LOOP = EnvelopeVerdict(
    domain="robotics_control_loop",
    verdict="out",
    reason=(
        "robotics IN THE CONTROL LOOP is out: physical consequence on an unguaranteeable clock, and "
        "mechanically incoherent — the gate's honest failure mode is ASK, and a control loop cannot "
        "ask. A perception system whose correct answer is sometimes 'I need a human' cannot be the "
        "thing that decides whether an actuator moves."
    ),
    narrowed_survivor=(
        "advisory acts with no physical consequence (actuation='screen': software/computer-use "
        "under ConsentProfile) and human-in-the-loop notification, where an ASK reaches a person."
    ),
    boundary=(
        "ConsentProfile.actuation. 'physical' is a declaration that this deployment is outside the "
        "envelope: ConsentProfile.permits() then returns False for every action and percept doctor "
        "reports OUT_OF_ENVELOPE_ACTUATION."
    ),
    enforcement="declared",
)

_PASSIVE_ANALYTICS = EnvelopeVerdict(
    domain="passive_analytics",
    verdict="out",
    reason=(
        "passive customer analytics is out: the beneficiary is the business and the observed person "
        "neither consented nor gains. The perception works; the arrangement is the problem."
    ),
    narrowed_survivor=(
        "an operator-facing watch on a condition the OBSERVED person also benefits from — a queue "
        "that got too long, a spill, an unattended child — reported as a scene condition with no "
        "per-person record and no per-person history."
    ),
    boundary=(
        "the same person resolver as security_person_id: the narrowed form becomes the excluded one "
        "the moment observations are keyed to a persisting person identity. That key cannot be "
        "constructed from pixels here, so the config change that would cross the line does not exist."
    ),
    enforcement="structural",
)

#: The three OUT verdicts, keyed by domain. Read by ``percept doctor`` and by ADR-007's tests.
OUT_OF_ENVELOPE: Mapping[str, EnvelopeVerdict] = MappingProxyType(
    {
        v.domain: v
        for v in (_SECURITY_PERSON_ID, _ROBOTICS_CONTROL_LOOP, _PASSIVE_ANALYTICS)
    }
)

#: Honest status of "assistant-class delivery only" (card Z7, the reported item).
#:
#: :class:`~percept.contracts.AttentionIntent` has ONE policy-controlled field, ``preempt``, and
#: :class:`~percept.incidents.AttentionPolicy` clamps it deterministically from durable provenance +
#: consequence + allowlists. That is real config enforcement — of INTERRUPT AUTHORITY. It is NOT
#: enforcement of the CONSUMER CLASS: percept has no typed notion of what is on the other end of a
#: delivery. ``preferred_channel`` is free text and advisory; a sink that drives a horn instead of an
#: assistant is indistinguishable from one that speaks. So: assistant-class-only is enforced in DOCS.
ATTENTION_CLASS_ENFORCEMENT = "docs"

#: What it would take to move :data:`ATTENTION_CLASS_ENFORCEMENT` from ``docs`` to ``declared``.
ATTENTION_CLASS_GAP = (
    "To make 'assistant-class only' enforceable in config rather than docs, three additive things "
    "are needed: (1) a typed consumer class on the delivery side — a sink declares "
    "channel_class in ('assistant', 'operator_console', 'machine') the way SourceProfile declares "
    "its strategies, instead of today's free-text AttentionIntent.preferred_channel; (2) a registry "
    "of delivery sinks keyed by that class, so the class is a property of the WIRING and not of a "
    "string a caller passes; (3) an AttentionPolicy rule that refuses 'machine' consumers outright "
    "and clamps them the way an unauthorized interrupt is clamped today. Until (1) exists there is "
    "nothing for a policy to read: AttentionPolicy.clamp() sees an incident, a goal origin and a "
    "consequence, and none of those says who listens."
)


# ---------------------------------------------------------------------------
# Subject resolvers — the structural refusal
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubjectResolver:
    """A typed declaration of HOW a subject key is obtained — the surface card Z7 (a) locks down.

    Construction itself is the refusal. A resolver whose ``subject_class`` is a person class
    (:data:`PERSON_SUBJECT_CLASSES`) may not:

    - derive from ``pixels`` at all (that is biometric identification, whatever it is called), or
    - ``persists`` across sessions unless the key is ``client_asserted`` (a persisting pixel-or-scene
      key IS a re-identification record).

    ``binding="bound"`` — the key names a real subject in the world — is reachable only from
    ``derives_from="client_asserted"`` with a non-empty ``asserted_by``. THEIR system asserts it;
    percept never promotes a handle it computed into an identity.
    """

    name: str
    subject_class: str
    derives_from: str
    binding: str = "unbound"
    asserted_by: str = ""
    persists: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        for label in ("name", "subject_class"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value.strip():
                raise EnvelopeViolation(
                    f"SubjectResolver.{label} must be a non-empty string, got {value!r}"
                )
        if self.derives_from not in IDENTITY_SOURCES:
            raise EnvelopeViolation(
                f"SubjectResolver.derives_from must be one of {IDENTITY_SOURCES}, got "
                f"{self.derives_from!r}"
            )
        if self.binding not in IDENTITY_BINDINGS:
            raise EnvelopeViolation(
                f"SubjectResolver.binding must be one of {IDENTITY_BINDINGS}, got {self.binding!r}"
            )
        if self.binding == "bound":
            if self.derives_from != "client_asserted":
                raise EnvelopeViolation(
                    f"SubjectResolver {self.name!r} declares binding='bound' from "
                    f"{self.derives_from!r}: a BOUND key names a real subject and may only come "
                    "from the client's own system (derives_from='client_asserted'). percept never "
                    "promotes a handle it computed into an identity."
                )
            if not self.asserted_by.strip():
                raise EnvelopeViolation(
                    f"SubjectResolver {self.name!r} declares binding='bound' but names no asserting "
                    "system; set asserted_by to the client system that asserts the key (a badge "
                    "reader, their re-ID service, an SSO subject)."
                )
        if self.is_person_class:
            if self.derives_from == "pixels":
                raise EnvelopeViolation(
                    f"SubjectResolver {self.name!r} would derive PERSON identity from pixels. That "
                    "is biometric re-identification of (in the deployments this was refused for) "
                    "non-consenting third parties, and it is OUT of percept's envelope: "
                    + OUT_OF_ENVELOPE["security_person_id"].sentence()
                )
            if self.persists and self.derives_from != "client_asserted":
                raise EnvelopeViolation(
                    f"SubjectResolver {self.name!r} would PERSIST a person key percept computed "
                    f"({self.derives_from!r}). A persisting person key is a re-identification "
                    "record however it was produced. Scene-local individuation must die with the "
                    "run; only a client-asserted key may persist. "
                    + OUT_OF_ENVELOPE["security_person_id"].sentence()
                )

    @property
    def is_person_class(self) -> bool:
        return self.subject_class.strip().lower() in PERSON_SUBJECT_CLASSES

    @property
    def identifies_persons(self) -> bool:
        """``True`` when this resolver produces a person identity percept itself derived. Always
        ``False`` for anything constructible here — it is the invariant, kept as a readable property
        so doctor and the tests can ASSERT it rather than trust the constructor."""
        return self.is_person_class and self.derives_from != "client_asserted"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "subject_class": self.subject_class,
            "derives_from": self.derives_from,
            "binding": self.binding,
            "asserted_by": self.asserted_by,
            "persists": self.persists,
            "identifies_persons": self.identifies_persons,
            "notes": self.notes,
        }


def client_asserted_subject(
    *, value: str, asserted_by: str, subject_class: str = "person", notes: str = ""
) -> SubjectResolver:
    """The ONLY door a person identity comes through: the client's own system asserts it.

    Shipped as a FACTORY, never as a registered resolver — there is no such thing as a default
    asserting system, and a percept release must not contain a person resolver anyone can simply
    switch on. ``value`` is the client's key (a badge id, their re-ID handle); percept treats it as
    opaque and never computes, verifies or extends it.
    """
    if not isinstance(value, str) or not value.strip():
        raise EnvelopeViolation("client_asserted_subject(value=...) must be a non-empty key")
    return SubjectResolver(
        name=f"client_asserted:{value}",
        subject_class=subject_class,
        derives_from="client_asserted",
        binding="bound",
        asserted_by=asserted_by,
        persists=True,
        notes=notes or "client-asserted subject key; percept derived nothing from pixels",
    )


@dataclass
class SubjectResolverRegistry:
    """Name-keyed registry of :class:`SubjectResolver` with a second, independent refusal.

    The constructor already makes a percept-derived person key unconstructible. The registry re-checks
    :attr:`SubjectResolver.identifies_persons` anyway, because the two guards fail independently: a
    future field could widen what the constructor accepts, and a client's own subclass/duck-typed
    object could reach ``register`` without going through ``__post_init__``.
    """

    _resolvers: dict[str, SubjectResolver] = field(default_factory=dict)

    def register(self, resolver: SubjectResolver, *, overwrite: bool = False) -> None:
        if not isinstance(resolver, SubjectResolver):
            raise EnvelopeViolation(
                f"expected a SubjectResolver, got {type(resolver).__name__}"
            )
        if resolver.identifies_persons:
            raise EnvelopeViolation(
                f"refusing to register subject resolver {resolver.name!r}: it identifies persons "
                "from a key percept derived. "
                + OUT_OF_ENVELOPE["security_person_id"].sentence()
            )
        existing = self._resolvers.get(resolver.name)
        if existing is not None and existing != resolver and not overwrite:
            raise EnvelopeViolation(
                f"a subject resolver named {resolver.name!r} is already registered; pass "
                "overwrite=True to replace it"
            )
        self._resolvers[resolver.name] = resolver

    def unregister(self, name: str) -> None:
        self._resolvers.pop(name, None)

    def get(self, name: str) -> SubjectResolver | None:
        return self._resolvers.get(name)

    def names(self) -> list[str]:
        return sorted(self._resolvers)

    def all(self) -> list[SubjectResolver]:
        return list(self._resolvers.values())

    def person_identifying(self) -> list[SubjectResolver]:
        """Every registered resolver that identifies persons. MUST be empty — doctor reports the
        length, and ``tests/test_envelope.py`` asserts it, so a resolver added by a later card fails
        the suite instead of review."""
        return [r for r in self._resolvers.values() if r.identifies_persons]

    def __contains__(self, name: object) -> bool:
        return name in self._resolvers

    def __len__(self) -> int:
        return len(self._resolvers)

    def __iter__(self) -> Iterator[SubjectResolver]:
        return iter(self._resolvers.values())


def _default_registry() -> SubjectResolverRegistry:
    """The resolvers percept SHIPS. Each mirrors an ``identity_strategy`` a shipped
    :class:`~percept.sourceprofile.SourceProfile` already names, so this registry describes reality
    rather than inventing a parallel one — and none of them identifies a person."""
    reg = SubjectResolverRegistry()
    reg.register(
        SubjectResolver(
            name="track_iou_label",
            subject_class="object",
            derives_from="scene_local",
            binding="unbound",
            persists=False,
            notes=(
                "IoU + label continuity (percept.identity.ContinuityIdentity). A within-run handle "
                "for a physical object; it means nothing after the run ends."
            ),
        )
    )
    reg.register(
        SubjectResolver(
            name="appearance_recent_set",
            subject_class="object",
            derives_from="scene_local",
            binding="unbound",
            persists=False,
            notes=(
                "continuity, then appearance match over the RECENT set only "
                "(percept.identity.AppearanceIdentity) — re-acquisition across a tracking gap "
                "inside one run. Deliberately not a person resolver and deliberately not persisted: "
                "persisted appearance matching over people is re-identification."
            ),
        )
    )
    reg.register(
        SubjectResolver(
            name="content_key",
            subject_class="document",
            derives_from="scene_local",
            binding="unbound",
            persists=False,
            notes="page/content identity for the screen_document source profile (ADR-005).",
        )
    )
    reg.register(
        SubjectResolver(
            name="speaker_label",
            subject_class="speaker_slot",
            derives_from="scene_local",
            binding="unbound",
            persists=False,
            notes=(
                "diarization SLOT ('speaker 1'), not a person. It is an ordering label within one "
                "conversation and is never carried across runs or matched to a voice print."
            ),
        )
    )
    return reg


#: The process-wide subject-resolver registry. Ships FOUR resolvers, none person-identifying.
DEFAULT_SUBJECT_RESOLVERS: SubjectResolverRegistry = _default_registry()


def register_subject_resolver(resolver: SubjectResolver, *, overwrite: bool = False) -> None:
    """Third-party extension point — subject to the same refusal as the shipped ones."""
    DEFAULT_SUBJECT_RESOLVERS.register(resolver, overwrite=overwrite)


def assert_no_person_identifying_resolver(
    registry: SubjectResolverRegistry | None = None,
) -> None:
    """Raise :class:`EnvelopeViolation` if any resolver in ``registry`` identifies persons.

    Called by the suite and by ``percept doctor``: the invariant is checked against the LIVE registry
    (including whatever a client registered) rather than asserted about the source file.
    """
    reg = DEFAULT_SUBJECT_RESOLVERS if registry is None else registry
    bad = reg.person_identifying()
    if bad:
        raise EnvelopeViolation(
            "person-identifying subject resolvers are registered: "
            f"{sorted(r.name for r in bad)}. "
            + OUT_OF_ENVELOPE["security_person_id"].sentence()
        )


# ---------------------------------------------------------------------------
# Actuation — the typed declaration (card Z7 (b))
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvelopeCheck:
    """The answer to "is this configuration inside the envelope?" — what doctor prints.

    ``in_envelope`` false always carries a ``verdict``; ``note`` is the human sentence.
    """

    in_envelope: bool
    note: str
    verdict: EnvelopeVerdict | None = None
    modality: str = "speak"
    actuation: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "in_envelope": self.in_envelope,
            "note": self.note,
            "modality": self.modality,
            "actuation": self.actuation,
            "verdict": self.verdict.to_dict() if self.verdict else None,
        }


def validate_actuation(actuation: str) -> str:
    """Normalize + validate a declared actuation class, or raise :class:`EnvelopeViolation`."""
    if not isinstance(actuation, str) or actuation not in ACTUATION_CLASSES:
        raise EnvelopeViolation(
            f"actuation must be one of {ACTUATION_CLASSES}, got {actuation!r}. It is a DECLARATION "
            "of what an act can physically do, not a preference — percept doctor reports it and "
            "ConsentProfile enforces it."
        )
    return actuation


def actuation_check(*, modality: str = "speak", actuation: str = "none") -> EnvelopeCheck:
    """Fold a goal ``modality`` and a declared ``actuation`` class into an :class:`EnvelopeCheck`.

    ``modality="act"`` PLUS ``actuation="physical"`` is the pair that is out of the envelope. Either
    one alone is not: a speaking goal in a plant with robots is fine, and an ``act`` goal that clicks
    a button is the narrowed survivor.
    """
    actuation = validate_actuation(actuation)
    if actuation != PHYSICAL_ACTUATION:
        return EnvelopeCheck(
            in_envelope=True,
            note=f"actuation='{actuation}' — advisory only; no physical consequence declared",
            modality=modality,
            actuation=actuation,
        )
    verdict = OUT_OF_ENVELOPE["robotics_control_loop"]
    if modality != "act":
        return EnvelopeCheck(
            in_envelope=True,
            note=(
                "actuation='physical' is DECLARED but no goal uses modality='act', so percept "
                "closes no loop onto it. ConsentProfile.permits() returns False for every action "
                "while this declaration stands. " + verdict.sentence()
            ),
            verdict=verdict,
            modality=modality,
            actuation=actuation,
        )
    return EnvelopeCheck(
        in_envelope=False,
        note=(
            "modality='act' with actuation='physical' — percept is being wired INTO a control loop. "
            + verdict.sentence()
        ),
        verdict=verdict,
        modality=modality,
        actuation=actuation,
    )


def envelope_summary() -> dict[str, Any]:
    """Everything a tool needs to REPORT the envelope, in one dict (``percept doctor --json``)."""
    return {
        "schema": "percept.envelope/1",
        "out_of_envelope": {k: v.to_dict() for k, v in OUT_OF_ENVELOPE.items()},
        "subject_resolvers": [r.to_dict() for r in DEFAULT_SUBJECT_RESOLVERS.all()],
        "person_identifying_resolvers": [
            r.name for r in DEFAULT_SUBJECT_RESOLVERS.person_identifying()
        ],
        "attention_class_enforcement": ATTENTION_CLASS_ENFORCEMENT,
        "attention_class_gap": ATTENTION_CLASS_GAP,
    }


__all__ = [
    "ENFORCEMENTS",
    "IDENTITY_SOURCES",
    "IDENTITY_BINDINGS",
    "ACTUATION_CLASSES",
    "PHYSICAL_ACTUATION",
    "PERSON_SUBJECT_CLASSES",
    "EnvelopeViolation",
    "EnvelopeVerdict",
    "OUT_OF_ENVELOPE",
    "ATTENTION_CLASS_ENFORCEMENT",
    "ATTENTION_CLASS_GAP",
    "SubjectResolver",
    "SubjectResolverRegistry",
    "DEFAULT_SUBJECT_RESOLVERS",
    "client_asserted_subject",
    "register_subject_resolver",
    "assert_no_person_identifying_resolver",
    "EnvelopeCheck",
    "validate_actuation",
    "actuation_check",
    "envelope_summary",
]
