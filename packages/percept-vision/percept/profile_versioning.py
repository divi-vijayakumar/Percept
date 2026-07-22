"""Profile versioning + provenance: content-addressed, auditable, reversible.

:mod:`percept.watchprofile` gave ``Q(p | environment, goal, state)`` a typed SHAPE, and
:mod:`percept.profiles_default` seeded the empirically-derived DATA. Both are in-process Python objects: a
profile that only exists as a module-level constant cannot be shipped, diffed, or audited, and "recall
went 0.68 -> 0.90 when we flipped these three knobs" cannot be expressed as an artifact. This module adds
the three things that turn a profile into config you can reason about over time:

- **Serialization** — :func:`profile_to_dict` / :func:`profile_from_dict` are an exact, lossless
  round-trip (``profile_from_dict(profile_to_dict(p)) == p``) over plain JSON-safe types. Unknown or
  missing keys raise :class:`~percept.watchprofile.WatchProfileError` rather than being silently dropped;
  a profile that fails to round-trip is a bug, not a warning.
- **Content addressing** — :func:`content_hash` is ``sha256`` over the CANONICAL serialization
  (:func:`canonical_json`: sorted keys, no insignificant whitespace, UTF-8). Two profiles with the same
  knobs hash the same regardless of construction order or dict insertion order; changing any single knob
  changes the hash. That hash is the profile's identity — it is what an eval run, a fire event, or a
  release note can cite unambiguously.
- **Provenance** — :class:`ProfileProvenance` stamps a profile with the eval run that produced it
  (``eval_run_id``), the harness that measured it (``harness_version``), and when (``created_at``, ISO-8601
  UTC). :class:`VersionedProfile` is the shippable envelope: ``{schema_version, content_hash, provenance,
  profile}``. The stamp deliberately does NOT participate in the content hash — re-stamping the same knobs
  from a later eval run must not change the profile's identity, otherwise "did the config actually change?"
  becomes unanswerable.

And the payoff the continuous-learning bridge needs: :func:`diff_registries` turns two
registry versions into an explicit, field-level :class:`RegistryDiff` — added / removed / changed profiles,
each change a dotted-path ``before -> after`` :class:`FieldDelta`. That is *learnings-as-config-deltas*: an
eval run's output is a diff you can read, review and — via :meth:`RegistryDiff.inverse` — reverse.

Stdlib-only (``hashlib`` / ``json`` / ``datetime``), no network, no I/O. This module READS
:mod:`percept.watchprofile`; it changes nothing there and nothing in the runtime path.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from .watchprofile import (
    CadenceProfile,
    GateProfile,
    GoalDensityGuidance,
    WatchProfile,
    WatchProfileError,
    WatchProfileRegistry,
)

#: Envelope schema version for :class:`VersionedProfile`. Bumped only when the serialized SHAPE changes
#: incompatibly (a new WatchProfile field is additive and does not require a bump; a renamed or removed
#: one does). It is envelope metadata and is deliberately NOT part of :func:`content_hash`.
PROFILE_SCHEMA_VERSION = 1

#: Field name -> nested dataclass type, for the three sub-profiles a WatchProfile carries. Serialization
#: and deserialization both drive off this table so the two directions cannot drift apart.
_NESTED_TYPES: dict[str, type] = {
    "cadence": CadenceProfile,
    "gate": GateProfile,
    "goal_density_guidance": GoalDensityGuidance,
}

#: The scalar (non-nested) WatchProfile fields, in canonical declaration order.
_SCALAR_FIELDS: tuple[str, ...] = (
    "name",
    "environment",
    "goal_type",
    "state",
    "salience_stat",
    "plan_mode",
    "notes",
)


# ---------------------------------------------------------------------------
# Serialization — lossless, JSON-safe, exact round-trip.
# ---------------------------------------------------------------------------


def profile_to_dict(profile: WatchProfile) -> dict[str, Any]:
    """Serialize ``profile`` to a plain, JSON-safe dict — every field, nothing else.

    Nested sub-profiles (``cadence``/``gate``/``goal_density_guidance``) become nested dicts;
    ``stat_thresholds`` is emitted with sorted keys and ``float`` values so the mapping's insertion order
    can never leak into :func:`content_hash`.
    """
    if not isinstance(profile, WatchProfile):
        raise WatchProfileError(f"profile_to_dict expects a WatchProfile, got {type(profile).__name__}")
    out: dict[str, Any] = {name: getattr(profile, name) for name in _SCALAR_FIELDS}
    out["stat_thresholds"] = {k: float(profile.stat_thresholds[k]) for k in sorted(profile.stat_thresholds)}
    for name, _cls in _NESTED_TYPES.items():
        nested = getattr(profile, name)
        out[name] = {f: getattr(nested, f) for f in nested.__dataclass_fields__}
    return out


def profile_from_dict(payload: Mapping[str, Any]) -> WatchProfile:
    """Rebuild a :class:`~percept.watchprofile.WatchProfile` from :func:`profile_to_dict` output.

    Strict on purpose: an unknown top-level or nested key, or a missing required field, raises
    :class:`~percept.watchprofile.WatchProfileError`. Silently ignoring an unrecognized knob would let a
    profile authored against a newer schema load as a *different* profile with the same name — exactly the
    failure content addressing exists to prevent. All value-level validation (vocabularies, ranges) is left
    to the dataclasses' own ``__post_init__``, so there is one validation authority, not two.
    """
    if not isinstance(payload, Mapping):
        raise WatchProfileError(f"profile_from_dict expects a mapping, got {type(payload).__name__}")
    known = set(_SCALAR_FIELDS) | {"stat_thresholds"} | set(_NESTED_TYPES)
    unknown = sorted(set(payload) - known)
    if unknown:
        raise WatchProfileError(f"unknown WatchProfile field(s) in payload: {unknown}")

    kwargs: dict[str, Any] = {}
    for name in _SCALAR_FIELDS:
        if name in payload:
            kwargs[name] = payload[name]
        elif name in ("name", "environment", "goal_type", "state"):
            raise WatchProfileError(f"WatchProfile payload is missing required field {name!r}")

    thresholds = payload.get("stat_thresholds", {})
    if not isinstance(thresholds, Mapping):
        raise WatchProfileError(f"stat_thresholds must be a mapping, got {type(thresholds).__name__}")
    kwargs["stat_thresholds"] = dict(thresholds)

    for name, cls in _NESTED_TYPES.items():
        nested = payload.get(name)
        if nested is None:
            continue
        if not isinstance(nested, Mapping):
            raise WatchProfileError(f"{name} must be a mapping, got {type(nested).__name__}")
        allowed = set(cls.__dataclass_fields__)
        extra = sorted(set(nested) - allowed)
        if extra:
            raise WatchProfileError(f"unknown {name} field(s) in payload: {extra}")
        kwargs[name] = cls(**dict(nested))

    return WatchProfile(**kwargs)


def canonical_json(payload: Any) -> str:
    """The canonical serialization every hash is taken over: sorted keys, compact separators, UTF-8
    preserved (``ensure_ascii=False``). Deterministic across processes and insertion orders."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash(profile: WatchProfile | Mapping[str, Any]) -> str:
    """``sha256`` hex digest of the profile's canonical dict — the profile's content address.

    Accepts a :class:`~percept.watchprofile.WatchProfile` or an already-serialized dict. Provenance is
    NOT included (see :class:`ProfileProvenance`): the hash addresses the *knobs*, so re-stamping the same
    configuration from a later eval run leaves its identity untouched.
    """
    payload = profile_to_dict(profile) if isinstance(profile, WatchProfile) else dict(profile)
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def short_hash(profile: WatchProfile | Mapping[str, Any], length: int = 12) -> str:
    """First ``length`` hex chars of :func:`content_hash` — for logs and diff summaries, never for
    equality checks (use the full hash for those)."""
    if length < 1 or length > 64:
        raise WatchProfileError(f"short_hash length must be in 1..64, got {length}")
    return content_hash(profile)[:length]


# ---------------------------------------------------------------------------
# Provenance + the shippable envelope.
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    """Current UTC instant as ``YYYY-MM-DDTHH:MM:SS+00:00`` (second resolution — a provenance stamp is
    an audit breadcrumb, not a timer)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class ProfileProvenance:
    """Where a profile's numbers CAME FROM — the audit stamp that makes it evidence instead of a guess.

    - ``eval_run_id`` — the identifier of the eval run whose measurements produced these knobs.
    - ``harness_version`` — the harness/package version that ran it (a number measured by a different
      harness is a different number).
    - ``created_at`` — ISO-8601 UTC instant the stamp was taken (defaults to now, at second resolution).
    - ``source`` — optional: the track/bench that was run (e.g. ``"charades-e2e"``).
    - ``notes`` — optional free text: the finding, in one sentence.
    """

    eval_run_id: str
    harness_version: str
    created_at: str = field(default_factory=_utc_now_iso)
    source: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        for label in ("eval_run_id", "harness_version", "created_at"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value.strip():
                raise WatchProfileError(
                    f"ProfileProvenance.{label} must be a non-empty string, got {value!r}"
                )
        for label in ("source", "notes"):
            if not isinstance(getattr(self, label), str):
                raise WatchProfileError(f"ProfileProvenance.{label} must be a string")

    def to_dict(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProfileProvenance":
        if not isinstance(payload, Mapping):
            raise WatchProfileError(
                f"ProfileProvenance.from_dict expects a mapping, got {type(payload).__name__}"
            )
        extra = sorted(set(payload) - set(cls.__dataclass_fields__))
        if extra:
            raise WatchProfileError(f"unknown ProfileProvenance field(s) in payload: {extra}")
        return cls(**dict(payload))


@dataclass(frozen=True)
class VersionedProfile:
    """A :class:`~percept.watchprofile.WatchProfile` plus its provenance stamp — the shippable unit.

    :meth:`to_dict` writes ``{schema_version, content_hash, provenance, profile}``. The stored hash is a
    CHECKSUM, not an input: :meth:`from_dict` recomputes it from the profile body and raises if the two
    disagree, so a hand-edited knob can never masquerade under an old address.
    """

    profile: WatchProfile
    provenance: ProfileProvenance
    schema_version: int = PROFILE_SCHEMA_VERSION

    @property
    def content_hash(self) -> str:
        """The wrapped profile's content address (provenance-independent)."""
        return content_hash(self.profile)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "content_hash": self.content_hash,
            "provenance": self.provenance.to_dict(),
            "profile": profile_to_dict(self.profile),
        }

    def to_json(self) -> str:
        """Canonical JSON of :meth:`to_dict` — byte-stable for the same content."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "VersionedProfile":
        if not isinstance(payload, Mapping):
            raise WatchProfileError(
                f"VersionedProfile.from_dict expects a mapping, got {type(payload).__name__}"
            )
        extra = sorted(set(payload) - {"schema_version", "content_hash", "provenance", "profile"})
        if extra:
            raise WatchProfileError(f"unknown VersionedProfile field(s) in payload: {extra}")
        schema_version = payload.get("schema_version", PROFILE_SCHEMA_VERSION)
        if schema_version != PROFILE_SCHEMA_VERSION:
            raise WatchProfileError(
                f"unsupported profile schema_version {schema_version!r}; this build reads "
                f"{PROFILE_SCHEMA_VERSION}"
            )
        if "profile" not in payload:
            raise WatchProfileError("VersionedProfile payload is missing 'profile'")
        if "provenance" not in payload:
            raise WatchProfileError("VersionedProfile payload is missing 'provenance'")
        profile = profile_from_dict(payload["profile"])
        stated = payload.get("content_hash")
        if stated is not None:
            actual = content_hash(profile)
            if stated != actual:
                raise WatchProfileError(
                    f"content_hash mismatch for profile {profile.name!r}: payload states {stated!r} but "
                    f"the profile body hashes to {actual!r} (the profile was edited without re-stamping)"
                )
        return cls(
            profile=profile,
            provenance=ProfileProvenance.from_dict(payload["provenance"]),
            schema_version=PROFILE_SCHEMA_VERSION,
        )

    @classmethod
    def from_json(cls, text: str) -> "VersionedProfile":
        return cls.from_dict(json.loads(text))

    def restamp(self, provenance: ProfileProvenance) -> "VersionedProfile":
        """Same knobs, new stamp — and therefore the SAME :attr:`content_hash`. This is the operation
        that proves provenance is metadata: a re-run that confirms the existing numbers records a new
        eval run without inventing a new configuration identity."""
        return VersionedProfile(profile=self.profile, provenance=provenance)


def stamp(
    profile: WatchProfile,
    *,
    eval_run_id: str,
    harness_version: str,
    created_at: str | None = None,
    source: str = "",
    notes: str = "",
) -> VersionedProfile:
    """Convenience constructor: wrap ``profile`` in a fresh :class:`ProfileProvenance` stamp."""
    prov = ProfileProvenance(
        eval_run_id=eval_run_id,
        harness_version=harness_version,
        **({} if created_at is None else {"created_at": created_at}),
        source=source,
        notes=notes,
    )
    return VersionedProfile(profile=profile, provenance=prov)


# ---------------------------------------------------------------------------
# Registry versioning + diff — learnings as config deltas.
# ---------------------------------------------------------------------------


def registry_to_dicts(registry: WatchProfileRegistry) -> list[dict[str, Any]]:
    """Every profile in ``registry`` as canonical dicts, ordered by registry key so two dumps of the same
    content compare equal regardless of registration order."""
    return [profile_to_dict(p) for p in sorted(registry, key=lambda p: p.key)]


def registry_hash(registry: WatchProfileRegistry) -> str:
    """``sha256`` over the whole registry's canonical dump — one content address for a config VERSION.
    Two registries holding the same profiles hash identically whatever order they were seeded in."""
    return hashlib.sha256(canonical_json(registry_to_dicts(registry)).encode("utf-8")).hexdigest()


def load_registry(
    payloads: Iterable[Mapping[str, Any]],
    registry: WatchProfileRegistry | None = None,
    *,
    overwrite: bool = False,
) -> WatchProfileRegistry:
    """Rebuild a registry from :func:`registry_to_dicts` output. Defaults to a NEW registry — this never
    touches :data:`percept.watchprofile.DEFAULT_REGISTRY` unless you pass it explicitly."""
    target = registry if registry is not None else WatchProfileRegistry()
    for payload in payloads:
        target.register(profile_from_dict(payload), overwrite=overwrite)
    return target


def _flatten(payload: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a profile dict to dotted paths (``cadence.base_fps``, ``stat_thresholds.block_max``)."""
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(_flatten(value, prefix=f"{path}."))
        else:
            flat[path] = value
    return flat


#: Sentinel for a dotted path that exists on only ONE side of a field-level diff (e.g. a
#: ``stat_thresholds`` entry that was added). Distinct from ``None``, which is a legitimate value
#: (``goal_density_guidance.max_recommended_goals``).
ABSENT = "<absent>"


@dataclass(frozen=True)
class FieldDelta:
    """One knob's change: a dotted ``path`` and its ``before`` / ``after`` values (:data:`ABSENT` when the
    path exists on only one side)."""

    path: str
    before: Any
    after: Any

    def inverse(self) -> "FieldDelta":
        return FieldDelta(path=self.path, before=self.after, after=self.before)

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "before": self.before, "after": self.after}

    def describe(self) -> str:
        return f"{self.path}: {self.before!r} -> {self.after!r}"


@dataclass(frozen=True)
class ProfileDelta:
    """A profile that exists on both sides of a diff but whose knobs changed. Carries both full bodies
    (so the delta is self-contained and reversible) plus the field-level breakdown."""

    key: tuple[str, str, str]
    before: dict[str, Any]
    after: dict[str, Any]
    fields: tuple[FieldDelta, ...]

    @property
    def before_hash(self) -> str:
        return content_hash(self.before)

    @property
    def after_hash(self) -> str:
        return content_hash(self.after)

    def inverse(self) -> "ProfileDelta":
        return ProfileDelta(
            key=self.key,
            before=self.after,
            after=self.before,
            fields=tuple(d.inverse() for d in self.fields),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": list(self.key),
            "before_hash": self.before_hash,
            "after_hash": self.after_hash,
            "before": self.before,
            "after": self.after,
            "fields": [d.to_dict() for d in self.fields],
        }


@dataclass(frozen=True)
class RegistryDiff:
    """The difference between two registry versions — the reviewable artifact an eval run produces.

    ``added`` / ``removed`` map a registry key to a full profile dict; ``changed`` holds a
    :class:`ProfileDelta` per key present on both sides with different content; ``unchanged`` lists the
    keys whose content hash is identical. :meth:`inverse` returns the diff that undoes this one — the
    "reversible" half of the card.
    """

    added: dict[tuple[str, str, str], dict[str, Any]] = field(default_factory=dict)
    removed: dict[tuple[str, str, str], dict[str, Any]] = field(default_factory=dict)
    changed: tuple[ProfileDelta, ...] = ()
    unchanged: tuple[tuple[str, str, str], ...] = ()

    @property
    def is_empty(self) -> bool:
        """True when the two registries are content-identical."""
        return not (self.added or self.removed or self.changed)

    def __bool__(self) -> bool:
        return not self.is_empty

    def inverse(self) -> "RegistryDiff":
        """The diff that turns the *after* registry back into the *before* one."""
        return RegistryDiff(
            added=dict(self.removed),
            removed=dict(self.added),
            changed=tuple(d.inverse() for d in self.changed),
            unchanged=self.unchanged,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "added": {"|".join(k): v for k, v in self.added.items()},
            "removed": {"|".join(k): v for k, v in self.removed.items()},
            "changed": [d.to_dict() for d in self.changed],
            "unchanged": ["|".join(k) for k in self.unchanged],
        }

    def summary(self) -> str:
        """A one-screen, human-readable rendering — what a review comment or release note pastes."""
        if self.is_empty:
            return "no profile changes (registries are content-identical)"
        lines: list[str] = []
        for key, body in sorted(self.added.items()):
            lines.append(f"+ {'/'.join(key)} [{short_hash(body)}] {body.get('name', '')}")
        for key, body in sorted(self.removed.items()):
            lines.append(f"- {'/'.join(key)} [{short_hash(body)}] {body.get('name', '')}")
        for delta in self.changed:
            lines.append(
                f"~ {'/'.join(delta.key)} [{short_hash(delta.before)} -> {short_hash(delta.after)}]"
            )
            lines.extend(f"    {d.describe()}" for d in delta.fields)
        return "\n".join(lines)


def diff_profiles(before: WatchProfile, after: WatchProfile) -> tuple[FieldDelta, ...]:
    """Field-level delta between two profiles, as dotted-path :class:`FieldDelta` in sorted path order.
    Empty when the two are content-identical."""
    flat_before = _flatten(profile_to_dict(before))
    flat_after = _flatten(profile_to_dict(after))
    deltas: list[FieldDelta] = []
    for path in sorted(set(flat_before) | set(flat_after)):
        old = flat_before.get(path, ABSENT)
        new = flat_after.get(path, ABSENT)
        if old != new:
            deltas.append(FieldDelta(path=path, before=old, after=new))
    return tuple(deltas)


def diff_registries(before: WatchProfileRegistry, after: WatchProfileRegistry) -> RegistryDiff:
    """Diff two registry versions into a reviewable, reversible :class:`RegistryDiff`.

    Keys are compared by ``(environment, goal_type, state)`` — the identity a lookup actually uses — and
    bodies by :func:`content_hash`, so a profile whose only difference is dict insertion order counts as
    unchanged.
    """
    before_by_key = {p.key: p for p in before}
    after_by_key = {p.key: p for p in after}

    added = {k: profile_to_dict(after_by_key[k]) for k in sorted(set(after_by_key) - set(before_by_key))}
    removed = {k: profile_to_dict(before_by_key[k]) for k in sorted(set(before_by_key) - set(after_by_key))}

    changed: list[ProfileDelta] = []
    unchanged: list[tuple[str, str, str]] = []
    for key in sorted(set(before_by_key) & set(after_by_key)):
        old, new = before_by_key[key], after_by_key[key]
        old_body, new_body = profile_to_dict(old), profile_to_dict(new)
        if content_hash(old_body) == content_hash(new_body):
            unchanged.append(key)
            continue
        changed.append(
            ProfileDelta(key=key, before=old_body, after=new_body, fields=diff_profiles(old, new))
        )

    return RegistryDiff(
        added=added, removed=removed, changed=tuple(changed), unchanged=tuple(unchanged)
    )


__all__ = [
    "PROFILE_SCHEMA_VERSION",
    "ABSENT",
    "profile_to_dict",
    "profile_from_dict",
    "canonical_json",
    "content_hash",
    "short_hash",
    "ProfileProvenance",
    "VersionedProfile",
    "stamp",
    "registry_to_dicts",
    "registry_hash",
    "load_registry",
    "FieldDelta",
    "ProfileDelta",
    "RegistryDiff",
    "diff_profiles",
    "diff_registries",
]
