"""Evidence dereference — the opaque handle, its resolver, typed failures and a retention seam.

This module implements ADR-003 (``docs/adr/ADR-003-evidence-artifacts-retention.md``). Before it,
a webhook payload was all-or-nothing: ``include_media=False`` left the consumer with **no** way to
reach the evidence a fire was gated on, and ``include_media=True`` inlined base64 media into the POST
(an exfiltration vector for a client watching an accounting screen). This module adds the missing
third option — a **dereferenceable handle**: the payload may carry an opaque ``ref`` string that the
consumer resolves *against this runtime* to metadata (and, locally only, to bytes), rather than
carrying the pixels themselves.

What ADR-003 fixes, and what this module ships:

* **The opaque handle shape.** :class:`EvidenceArtifact` is the resolved metadata head — ``ref``,
  ``kind``, ``media_type``, capture timestamps, ``sha256``, ``expires_at`` (§2). It carries **no
  bytes**; the pixels live in the store keyed by ``ref``. The handle is opaque: a caller MUST NOT
  parse it, construct one, or assume it is a filesystem path (§1, §4).
* **Egress-safe head / local-only tail**, reusing :class:`~percept.contracts.LabelRecord`'s exact
  construction (an :attr:`~EvidenceArtifact.EGRESS_FIELDS` whitelist that structurally excludes the
  ``ref``). The no-handle-by-default rule is a *type-level fact*, not a runtime filter someone can
  forget (§4). A sink MAY opt in to carrying the bare handle string — :meth:`EvidenceArtifact.to_egress`
  takes ``include_ref=True`` — but that is a separate, narrower decision than dereferencing to bytes.
* **The resolver.** :func:`resolve_evidence` (and :meth:`EvidenceStore.resolve`) is the only sanctioned
  path from a handle to an artifact. It accepts either a bare ``ref`` string or a Y14
  :class:`~percept.episodes.EvidenceHandle` (the same opaque-handle shape — reused, not forked).
* **Four typed failures** (§3): :class:`EvidenceExpired`, :class:`EvidenceUnavailable`,
  :class:`EvidenceAccessDenied`, :class:`EvidenceCorrupt`. Resolution *never* returns ``None`` or an
  empty artifact — a silent empty is indistinguishable from "there genuinely was no evidence," a
  different fact a caller must not conflate with a resolution failure.
* **A retention seam** (§5) — TTL and eviction *mechanism only*. Every retention **number** is a
  caller-supplied parameter (:class:`RetentionPolicy`); this module invents none (no tune-to-green,
  per ADR §5's seven still-open owner decisions). What it *does* fix, because ADR-003 fixes it
  regardless of the numbers chosen, is the cross-class **eviction order**
  (:data:`EVICTION_ORDER`: SILENT-debug → sampled-debug → ASK → FIRE, i.e. FIRE evicted last).

Media inclusion is **opt-in and default OFF** throughout: the store holds bytes locally, egress
payloads carry neither bytes nor (by default) the handle, and reading bytes is a distinct local
operation (:meth:`EvidenceStore.open_bytes`) from resolving metadata.

Pure stdlib. No policy numbers are baked in.
"""

from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, ClassVar, Iterable, Literal, Mapping

#: The clock seam — a zero-arg callable returning production-domain seconds. Structurally identical to
#: ``percept.clock.Clock``, but named locally and defaulted to bare ``time.monotonic`` so this
#: production module keeps ADR-001 §5's rule (``percept/clock.py`` stays harness-only; production time
#: defaults are bare ``time.monotonic``, never ``clock.WallClock``). A test injects a ``FakeClock``.
Clock = Callable[[], float]

#: "argument not supplied" sentinel — distinct from ``None``, which is itself a meaningful value for
#: ``expires_at`` (None = never expires). Not part of the public surface.
_UNSET: Any = object()

# ------------------------------------------------------------------------------------------------
# Typed failures (ADR-003 §3) — resolution never returns a silent empty result.
# ------------------------------------------------------------------------------------------------


class EvidenceError(Exception):
    """Base for every evidence-resolution failure. Callers catch the four concrete subclasses; this
    exists only so a caller MAY catch the whole family without also catching unrelated exceptions."""


class EvidenceExpired(EvidenceError):
    """The retention window closed and the artifact was deleted on schedule (or evicted under byte
    pressure). This is an *expected* outcome in normal operation — retention did its job."""


class EvidenceUnavailable(EvidenceError):
    """The artifact is not (yet) written, the store is unreachable, a transient I/O failure occurred,
    or the bytes were redacted out from under an otherwise-live metadata head. Also expected in normal
    operation — distinct from :class:`EvidenceExpired`, which means *permanently gone by schedule*."""


class EvidenceAccessDenied(EvidenceError):
    """The caller lacks the authority to resolve this class of evidence. Exceptional — MUST be
    surfaced, never swallowed into a generic failure."""


class EvidenceCorrupt(EvidenceError):
    """The bytes behind the handle do not match the recorded ``sha256``. Exceptional and load-bearing:
    it is the tamper-evidence signal, and MUST be surfaced, never treated as 'no evidence'."""


# ------------------------------------------------------------------------------------------------
# The artifact shape (ADR-003 §2) — resolved metadata head; bytes live in the store, not here.
# ------------------------------------------------------------------------------------------------

#: The four capture shapes a handle may resolve to (§2). ``"frame"`` is a single still, ``"crop"`` a
#: region-of-interest still, ``"clip"`` a temporal window (an ordered frame burst OR a short encoded
#: video — both "more than one instant, one track"), ``"bundle"`` a heterogeneous multi-artifact
#: capture (overview + hi-res crop, or clip + adjacent audio). A ``"bundle"`` resolves to an ordered
#: ``list[EvidenceArtifact]`` sharing one ``ref``'s namespace — see :meth:`EvidenceStore.resolve_bundle`.
EvidenceKind = Literal["frame", "crop", "clip", "bundle"]


@dataclass(frozen=True)
class EvidenceArtifact:
    """The resolved metadata head behind an opaque ``evidence_ref`` (ADR-003 §2).

    This is what :func:`resolve_evidence` returns. It carries **no bytes** — the pixels live in the
    :class:`EvidenceStore` keyed by ``ref`` and are reached, locally only, through
    :meth:`EvidenceStore.open_bytes`. The artifact is deliberately the *metadata head*: it MAY egress
    (a downstream health dashboard wants ``kind``/``media_type``/timestamps), while the ``ref`` — the
    handle to the bytes — is the *local-only tail* excluded from egress by default (§4).

    ``ref`` is opaque in the ADR-003 sense: a caller MUST NOT parse it, construct one, assume it is a
    filesystem path, or infer capture shape from its format. Branch on :attr:`kind`, never on the ref
    string. ``capture_start_ts``/``capture_end_ts`` are in the same monotonic production domain as
    ``JudgmentTiming`` (ADR-001 §5) — never the untrusted ``source_media_ts`` — so evidence age and
    judgment latency compare directly. ``expires_at`` is ``None`` when the artifact does not expire
    under its active retention class.

    **Egress split**, mirroring :class:`~percept.contracts.LabelRecord` exactly: :meth:`to_egress`
    is built from the :attr:`EGRESS_FIELDS` whitelist, which omits ``ref`` — so the handle is
    *structurally* un-emittable unless a sink explicitly opts in via ``include_ref=True``. There is no
    ``**asdict`` / field-iteration path by which the handle leaks into an egress payload.
    """

    ref: str  # the opaque handle — the LOCAL-ONLY tail; NOT in EGRESS_FIELDS
    kind: EvidenceKind
    media_type: str  # e.g. "image/jpeg", "video/mp4", "application/x-percept-bundle"
    capture_start_ts: float
    capture_end_ts: float  # == capture_start_ts for an instantaneous single frame
    sha256: str  # integrity + dedup; also the tamper-evidence anchor for audit
    expires_at: float | None = None  # None = does not expire under the active retention class

    #: The egress whitelist — the metadata head, in declaration order, EXCLUDING ``ref`` (the local
    #: tail). :meth:`to_egress` emits exactly these names, so ``ref`` is structurally un-emittable. A
    #: ``ClassVar`` (not a dataclass field): it never serializes and never reaches a JSON schema. This
    #: is the same construction :class:`~percept.contracts.LabelRecord` uses for ``frame_ref``.
    EGRESS_FIELDS: ClassVar[tuple[str, ...]] = (
        "kind",
        "media_type",
        "capture_start_ts",
        "capture_end_ts",
        "sha256",
        "expires_at",
    )

    def to_egress(self, *, include_ref: bool = False) -> dict[str, Any]:
        """The metadata-head dict that MAY leave the machine — structurally cannot contain ``ref``
        unless a sink explicitly opts in (ADR-003 §4).

        Built from the :attr:`EGRESS_FIELDS` whitelist (which omits ``ref``), so there is no code path
        — no ``**asdict``, no field iteration — by which the opaque handle leaks by default. Passing
        ``include_ref=True`` carries the *bare handle string* for a trusted, co-located, or
        authenticated downstream resolver (mirroring today's ``WebhookSink(include_media=True)`` opt-in).
        Carrying the handle is a strictly narrower decision than embedding the resolved bytes — this
        method never emits bytes under any flag; that would require reading :meth:`EvidenceStore.open_bytes`."""
        head = {name: getattr(self, name) for name in self.EGRESS_FIELDS}
        if include_ref:
            head["ref"] = self.ref
        return head

    def is_expired(self, *, now: float) -> bool:
        """Has this artifact's retention window closed at ``now`` (production-domain seconds)? Always
        ``False`` when ``expires_at`` is ``None`` (does not expire under the active class)."""
        return self.expires_at is not None and now >= self.expires_at


def evidence_ref_of(handle: Any) -> str:
    """Extract the opaque ``ref`` string from a handle, reusing (not forking) Y14's shape.

    Accepts a bare ``ref`` string, an :class:`~percept.episodes.EvidenceHandle`, an
    :class:`EvidenceArtifact`, or anything else exposing a ``.ref`` attribute — so
    :func:`resolve_evidence` can be handed whatever the caller has without a conversion dance. Raises
    :class:`TypeError` if no ref can be read, and :class:`ValueError` if the ref is empty (a handle
    that never named a stored artifact)."""
    if isinstance(handle, str):
        ref = handle
    elif hasattr(handle, "ref"):
        # A handle that exposes ``.ref`` but whose ref is None/empty NAMED no stored artifact — that is
        # a value problem (ValueError), distinct from an object that is not a handle at all (TypeError).
        ref = handle.ref
    else:
        raise TypeError(f"cannot read an evidence ref from {type(handle).__name__}: {handle!r}")
    if not isinstance(ref, str) or not ref:
        raise ValueError(f"evidence ref must be a non-empty string, got {ref!r}")
    return ref


# ------------------------------------------------------------------------------------------------
# Retention seam (ADR-003 §5) — MECHANISM ONLY. Every number is a caller parameter; none is baked in.
# ------------------------------------------------------------------------------------------------

#: The four retention classes (§5). These name the *policy classes* whose TTLs the owner sets; this
#: module bakes in none of those TTLs.
RetentionClass = Literal["fire", "ask", "sampled_debug", "silent_debug"]

#: The cross-class eviction order when the global byte ceiling is hit before per-class TTLs would
#: evict (§5 "Eviction order"). This is the one thing ADR-003 fixes *regardless of the numbers
#: chosen*: SILENT-debug is shed first, FIRE last — a byte-pressure event must never be the reason a
#: FIRE capsule disappears before an ASK or debug one does. Do not reorder without amending ADR-003.
EVICTION_ORDER: tuple[RetentionClass, ...] = ("silent_debug", "sampled_debug", "ask", "fire")


@dataclass(frozen=True)
class RetentionPolicy:
    """Caller-supplied retention numbers — the seam ADR-003 §5 leaves to the owner.

    **This class invents no policy.** Every field defaults to "no constraint" (an empty TTL map, no
    byte ceiling, no labelled ceiling), so a store constructed with the default :class:`RetentionPolicy`
    never expires or evicts anything until the owner supplies numbers. That is deliberate: ADR-003 §5
    lists seven owner decisions (FIRE window, ASK window, sampled/SILENT debug flags, the global
    ceiling and per-class split, the labelled ceiling, webhook dereference) that MUST be set before
    the ADR leaves Proposed. This module refuses to guess any of them (no tune-to-green); it only
    provides the mechanism the numbers plug into.

    * ``ttls`` — ``{retention_class: ttl_seconds_or_None}``. A class absent from the map, or mapped to
      ``None``, means *no TTL* (does not expire under that class). Seconds, production domain.
    * ``max_bytes`` — global ceiling; ``None`` disables byte-pressure eviction entirely.
    * ``class_max_bytes`` — optional per-class sub-budgets so a debug-capture storm cannot starve FIRE
      of its share (§5). A class absent from the map has no sub-budget.
    * ``labelled_ceiling_s`` — the *bounded* extension a ``record_label()``'d artifact gets past its
      class default (§5 "Does labelling extend retention? Yes, bounded"). ``None`` means the mechanism
      is unconfigured and :meth:`EvidenceStore.extend_for_label` will refuse rather than extend
      indefinitely.
    """

    ttls: Mapping[RetentionClass, float | None] = field(default_factory=dict)
    max_bytes: int | None = None
    class_max_bytes: Mapping[RetentionClass, int] = field(default_factory=dict)
    labelled_ceiling_s: float | None = None

    def ttl_for(self, retention_class: RetentionClass) -> float | None:
        """The TTL in seconds for ``retention_class``, or ``None`` (no expiry) if the owner set none."""
        return self.ttls.get(retention_class)


# ------------------------------------------------------------------------------------------------
# The store — the ref → (metadata + bytes) index with the retention mechanism behind the resolver.
# ------------------------------------------------------------------------------------------------


@dataclass
class _Entry:
    """One stored artifact: its metadata head, its retention class, and its bytes (or ``None`` once
    redacted). Internal + mutable — never egressed, never part of the public surface."""

    artifact: EvidenceArtifact
    retention_class: RetentionClass
    data: bytes | None  # None once redacted in place (bytes cleared, metadata head kept)
    labelled: bool = False


#: A store-level authorizer: given a resolved artifact + its retention class, return ``True`` to allow
#: resolution, ``False`` to deny (:class:`EvidenceAccessDenied`). ``None`` (the default) allows all —
#: the store does not invent an access policy, it only provides the seam one plugs into.
Authorizer = Callable[[EvidenceArtifact, "RetentionClass"], bool]


class EvidenceStore:
    """A ``ref → (metadata head + local bytes)`` index with ADR-003's retention mechanism behind the
    resolver. In-memory by default; the index is serializable (:meth:`snapshot`/:meth:`restore`) so a
    caller can persist it across process restarts as §5 requires — this module provides the mechanism,
    not a chosen backend.

    The store is the *local* side of the handle: it holds bytes, and reading them
    (:meth:`open_bytes`) is a distinct operation from resolving metadata (:meth:`resolve`). Neither
    the bytes nor (by default) the handle crosses the network — see :meth:`EvidenceArtifact.to_egress`.

    Retention is caller-parameterised (:class:`RetentionPolicy`): the store expires on schedule
    (:meth:`reconcile`), evicts under byte pressure in the fixed cross-class order
    (:meth:`enforce_budget`), redacts on operator/legal request (:meth:`redact`, independent of TTL),
    and extends bounded retention for a labelled artifact (:meth:`extend_for_label`) — but every
    *number* those use comes from the policy, never from this class."""

    def __init__(
        self,
        *,
        policy: RetentionPolicy | None = None,
        clock: Clock = time.monotonic,
        authorizer: Authorizer | None = None,
    ) -> None:
        self._policy = policy if policy is not None else RetentionPolicy()
        self._clock = clock
        self._authorizer = authorizer
        self._entries: dict[str, _Entry] = {}
        #: Tombstones for refs deleted on schedule / under pressure — so :meth:`resolve` can return
        #: :class:`EvidenceExpired` (permanently gone) rather than :class:`EvidenceUnavailable`
        #: (never seen). ADR-003 §3 turns on this distinction.
        self._expired: set[str] = set()

    # -- write side ------------------------------------------------------------------------------

    def put(
        self,
        data: bytes,
        *,
        kind: EvidenceKind,
        media_type: str,
        capture_start_ts: float,
        capture_end_ts: float | None = None,
        retention_class: RetentionClass = "fire",
        ref: str | None = None,
        expires_at: Any = _UNSET,
    ) -> EvidenceArtifact:
        """Store ``data`` and return its :class:`EvidenceArtifact` handle (media inclusion is a local
        write here — nothing egresses).

        ``ref`` is generated as an opaque key when not supplied (a caller SHOULD let the store mint it
        — a caller-chosen ref must still be treated as opaque downstream). ``sha256`` is computed over
        ``data``. ``expires_at`` defaults to ``now + policy.ttl_for(retention_class)`` (``None`` TTL →
        never expires); pass an explicit ``expires_at`` (including ``None``) to override the policy for
        this artifact. ``capture_end_ts`` defaults to ``capture_start_ts`` (an instantaneous frame)."""
        if capture_end_ts is None:
            capture_end_ts = capture_start_ts
        if ref is None:
            ref = _mint_ref()
        if expires_at is _UNSET:
            ttl = self._policy.ttl_for(retention_class)
            expires_at = None if ttl is None else self._clock() + ttl
        artifact = EvidenceArtifact(
            ref=ref,
            kind=kind,
            media_type=media_type,
            capture_start_ts=capture_start_ts,
            capture_end_ts=capture_end_ts,
            sha256=_sha256(data),
            expires_at=expires_at,
        )
        self._entries[ref] = _Entry(artifact=artifact, retention_class=retention_class, data=data)
        self._expired.discard(ref)
        return artifact

    def put_bundle(
        self,
        items: Iterable[tuple[bytes, str]],
        *,
        capture_start_ts: float,
        capture_end_ts: float | None = None,
        retention_class: RetentionClass = "fire",
        ref: str | None = None,
        media_type: str = "application/x-percept-bundle",
        expires_at: Any = _UNSET,
    ) -> EvidenceArtifact:
        """Store a heterogeneous multi-artifact capture as a ``kind="bundle"`` parent whose children
        share the parent ref's namespace (ADR-003 §1/§2 — "bundle contents are ``list[EvidenceArtifact]``
        sharing one ref's namespace, not a new type").

        ``items`` is an ordered iterable of ``(data, child_media_type)`` — e.g. an overview frame plus
        a hi-res crop, or a clip plus adjacent audio. Each child is stored under ``"{parent}/{i}"`` and
        resolved as an ordered list by :meth:`resolve_bundle`. The parent artifact's ``sha256`` covers
        the concatenated child hashes so the bundle as a whole is tamper-evident. Every child inherits
        the parent's capture window, retention class and expiry, so the bundle expires and evicts as a
        unit."""
        if ref is None:
            ref = _mint_ref()
        items = list(items)
        child_refs: list[str] = []
        child_hashes: list[str] = []
        # Resolve the shared expiry ONCE so parent and children agree to the exact second.
        if expires_at is _UNSET:
            ttl = self._policy.ttl_for(retention_class)
            expires_at = None if ttl is None else self._clock() + ttl
        for i, (data, child_media_type) in enumerate(items):
            child_ref = f"{ref}/{i}"
            child = self.put(
                data,
                kind="frame",
                media_type=child_media_type,
                capture_start_ts=capture_start_ts,
                capture_end_ts=capture_end_ts,
                retention_class=retention_class,
                ref=child_ref,
                expires_at=expires_at,
            )
            child_refs.append(child_ref)
            child_hashes.append(child.sha256)
        parent = self.put(
            _sha256_hex_join(child_hashes),
            kind="bundle",
            media_type=media_type,
            capture_start_ts=capture_start_ts,
            capture_end_ts=capture_end_ts,
            retention_class=retention_class,
            ref=ref,
            expires_at=expires_at,
        )
        return parent

    # -- read side (the resolver) ----------------------------------------------------------------

    def resolve(self, handle: Any) -> EvidenceArtifact:
        """The only sanctioned path from a handle to its metadata (ADR-003 §2). Accepts a bare ref
        string or any ``.ref``-bearing handle. Returns the :class:`EvidenceArtifact`, or raises exactly
        one of the four typed failures (§3) — it NEVER returns ``None`` or an empty artifact.

        Lazily honours expiry: an artifact whose window has closed at the current clock reads as
        :class:`EvidenceExpired` even before a :meth:`reconcile` sweep has physically removed it."""
        ref = evidence_ref_of(handle)
        entry = self._entries.get(ref)
        if entry is None:
            if ref in self._expired:
                raise EvidenceExpired(f"evidence {ref!r} was deleted on schedule (retention closed)")
            raise EvidenceUnavailable(f"no evidence recorded for ref {ref!r}")
        now = self._clock()
        if entry.artifact.is_expired(now=now):
            # Window closed but the sweep hasn't run yet — collect it now and report the true reason.
            self._delete(ref, tombstone=True)
            raise EvidenceExpired(f"evidence {ref!r} expired at {entry.artifact.expires_at}")
        if self._authorizer is not None and not self._authorizer(entry.artifact, entry.retention_class):
            raise EvidenceAccessDenied(
                f"caller lacks authority to resolve {entry.retention_class!r}-class evidence {ref!r}"
            )
        return entry.artifact

    def resolve_bundle(self, handle: Any) -> list[EvidenceArtifact]:
        """Resolve a ``kind="bundle"`` handle to its ordered child artifacts (ADR-003 §2). Raises
        :class:`ValueError` if the ref does not name a bundle, and the same typed failures as
        :meth:`resolve` for the parent (expired/unavailable/denied)."""
        parent = self.resolve(handle)
        if parent.kind != "bundle":
            raise ValueError(f"ref {parent.ref!r} is a {parent.kind!r}, not a bundle")
        children: list[EvidenceArtifact] = []
        i = 0
        while True:
            entry = self._entries.get(f"{parent.ref}/{i}")
            if entry is None:
                break
            children.append(entry.artifact)
            i += 1
        return children

    def open_bytes(self, handle: Any) -> bytes:
        """Read the bytes behind a handle — the LOCAL dereference, never an egress path.

        Verifies the stored bytes against the recorded ``sha256`` and raises :class:`EvidenceCorrupt`
        on mismatch (the tamper-evidence check). Raises :class:`EvidenceUnavailable` if the bytes were
        redacted out from under a still-live metadata head, and the same expired/denied failures as
        :meth:`resolve`. This is the media path ADR-003 keeps opt-in and off the wire: obtaining an
        artifact never obtains its pixels; a caller must ask for them explicitly, and only locally."""
        artifact = self.resolve(handle)  # applies expiry + access checks and typed failures
        entry = self._entries[artifact.ref]
        if entry.data is None:
            raise EvidenceUnavailable(f"evidence {artifact.ref!r} was redacted; bytes are gone")
        if _sha256(entry.data) != artifact.sha256:
            raise EvidenceCorrupt(
                f"evidence {artifact.ref!r} failed integrity check (bytes do not match recorded sha256)"
            )
        return entry.data

    # -- retention mechanism (numbers all come from the policy, never from here) -----------------

    def redact(self, handle: Any) -> None:
        """Clear an artifact's bytes IN PLACE while keeping its metadata head (ADR-003 §5 "Privacy
        policy", reusing the ``IncidentSnippet`` precedent — redaction clears ``media``/``frames`` but
        keeps ``kind``/``mime``/timestamps for the health dashboard). Redaction is a *distinct trigger*
        from TTL closure and can fire ahead of it — e.g. on an operator/legal request. A subsequent
        :meth:`resolve` still returns the head; :meth:`open_bytes` then raises
        :class:`EvidenceUnavailable`. No-op if the ref is unknown or already gone."""
        entry = self._entries.get(evidence_ref_of(handle))
        if entry is not None:
            entry.data = None

    def extend_for_label(self, handle: Any) -> EvidenceArtifact:
        """Extend a labelled artifact's retention to the *bounded* labelled ceiling (ADR-003 §5 "Does
        labelling extend retention? Yes, bounded"). A human already spent attention confirming this
        incident, so its evidence outlives its class default — but only to
        ``policy.labelled_ceiling_s`` past now, never indefinitely, so an active labelling program
        cannot turn "labelled" into unbounded accumulation.

        Raises :class:`ValueError` if the policy set no ``labelled_ceiling_s`` (the mechanism is
        unconfigured — refusing is correct; guessing a ceiling would be tune-to-green), and the typed
        resolution failures if the artifact is already gone."""
        ceiling = self._policy.labelled_ceiling_s
        if ceiling is None:
            raise ValueError(
                "RetentionPolicy.labelled_ceiling_s is unset; refusing to extend retention "
                "indefinitely (ADR-003 §5 requires a bounded labelled ceiling)"
            )
        artifact = self.resolve(handle)
        entry = self._entries[artifact.ref]
        extended = replace(artifact, expires_at=self._clock() + ceiling)
        entry.artifact = extended
        entry.labelled = True
        return extended

    def reconcile(self, *, now: float | None = None) -> list[str]:
        """Sweep everything past its expiry and delete it, returning the evicted refs (ADR-003 §5
        "Persistence across restarts" — on restart a reconciliation sweep clears anything past expiry
        that a crash prevented from being evicted on schedule). Within this sweep, order is
        oldest-past-expiry irrelevant: everything expired is gone. Idempotent."""
        t = self._clock() if now is None else now
        doomed = [ref for ref, e in self._entries.items() if e.artifact.is_expired(now=t)]
        for ref in doomed:
            self._delete(ref, tombstone=True)
        return doomed

    def enforce_budget(self, *, now: float | None = None) -> list[str]:
        """Evict under byte pressure until the store fits ``policy.max_bytes`` (and any per-class
        sub-budget), in ADR-003 §5's fixed cross-class order (:data:`EVICTION_ORDER`: SILENT-debug →
        sampled-debug → ASK → FIRE). Within a class, oldest-past-expiry first, then oldest by
        ``capture_start_ts``. Returns the evicted refs. A no-op when ``max_bytes`` is ``None`` and no
        per-class sub-budgets are set — the store never evicts under pressure the owner did not price.

        FIRE is evicted last by construction: a byte-pressure event must never be the reason a FIRE
        capsule disappears before an ASK or debug one does."""
        t = self._clock() if now is None else now
        evicted: list[str] = []

        # Reconcile expired entries first so pressure eviction never sheds a live capsule while a
        # dead one still counts against the budget.
        self.reconcile(now=t)

        # Per-class sub-budgets next: a debug-capture storm must not starve FIRE of its share.
        for cls, cap in self._policy.class_max_bytes.items():
            evicted += self._evict_class_to(cls, cap)

        # Then the global ceiling, shedding classes in the fixed order (FIRE last).
        ceiling = self._policy.max_bytes
        if ceiling is not None:
            for cls in EVICTION_ORDER:
                if self._total_bytes() <= ceiling:
                    break
                evicted += self._evict_class_until_global(cls, ceiling)
        return evicted

    # -- persistence seam (§5 "Persistence across restarts") -------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """The persistable ``ref → metadata + retention`` index (NOT the bytes) — the part ADR-003 §5
        requires to survive process restarts so a crash cannot silently orphan or leak an
        uncollectible artifact. Bytes live wherever the deployment keyed them by ``ref`` and are not
        part of this index; :meth:`restore` reconstructs metadata + tombstones, not pixels."""
        return {
            "entries": {
                ref: {
                    "artifact": e.artifact.to_egress(include_ref=True),
                    "retention_class": e.retention_class,
                    "labelled": e.labelled,
                    "has_bytes": e.data is not None,
                }
                for ref, e in self._entries.items()
            },
            "expired": sorted(self._expired),
        }

    # -- introspection (observability, not egress) ----------------------------------------------

    def total_bytes(self) -> int:
        """Bytes currently retained across all live (un-redacted) artifacts — for the health
        dashboard and for testing byte-pressure eviction. Not an egress field."""
        return self._total_bytes()

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, handle: Any) -> bool:
        try:
            return evidence_ref_of(handle) in self._entries
        except (TypeError, ValueError):
            return False

    # -- internals -------------------------------------------------------------------------------

    def _total_bytes(self) -> int:
        return sum(len(e.data) for e in self._entries.values() if e.data is not None)

    def _delete(self, ref: str, *, tombstone: bool) -> None:
        self._entries.pop(ref, None)
        if tombstone:
            self._expired.add(ref)

    def _class_entries_evict_order(self, cls: RetentionClass) -> list[str]:
        """Refs of one class in eviction order within the class: oldest-past-expiry first, then oldest
        by capture start."""
        now = self._clock()
        members = [(ref, e) for ref, e in self._entries.items() if e.retention_class == cls]
        members.sort(
            key=lambda kv: (
                not kv[1].artifact.is_expired(now=now),  # expired (False sorts first)
                kv[1].artifact.capture_start_ts,
            )
        )
        return [ref for ref, _ in members]

    def _evict_class_to(self, cls: RetentionClass, cap: int) -> list[str]:
        evicted: list[str] = []
        for ref in self._class_entries_evict_order(cls):
            if self._class_bytes(cls) <= cap:
                break
            self._delete(ref, tombstone=True)
            evicted.append(ref)
        return evicted

    def _evict_class_until_global(self, cls: RetentionClass, ceiling: int) -> list[str]:
        evicted: list[str] = []
        for ref in self._class_entries_evict_order(cls):
            if self._total_bytes() <= ceiling:
                break
            self._delete(ref, tombstone=True)
            evicted.append(ref)
        return evicted

    def _class_bytes(self, cls: RetentionClass) -> int:
        return sum(
            len(e.data)
            for e in self._entries.values()
            if e.retention_class == cls and e.data is not None
        )


# ------------------------------------------------------------------------------------------------
# The module-level resolver — the sanctioned function form (ADR-003 §2 ``w.resolve_evidence(...)``).
# ------------------------------------------------------------------------------------------------


def resolve_evidence(handle: Any, *, store: EvidenceStore) -> EvidenceArtifact:
    """Resolve an opaque handle to its :class:`EvidenceArtifact` against ``store`` (ADR-003 §2).

    This is the function form of the runtime's ``w.resolve_evidence(event.evidence_ref)`` seam: a
    runtime holds one :class:`EvidenceStore` and exposes this as a bound method. It accepts a bare ref
    string or any ``.ref``-bearing handle (a Y14 :class:`~percept.episodes.EvidenceHandle`, an
    :class:`EvidenceArtifact`), and raises exactly one of the four typed failures on failure — never a
    silent ``None``. The store may be local disk, an embedded DB, or a future remote blob store; that
    is an implementation detail behind the resolver, not a caller concern."""
    return store.resolve(handle)


# ------------------------------------------------------------------------------------------------
# small stdlib helpers
# ------------------------------------------------------------------------------------------------


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_hex_join(hex_digests: list[str]) -> bytes:
    """Bytes whose sha256 tamper-evidences an ordered set of child hashes (bundle integrity)."""
    return "\n".join(hex_digests).encode("utf-8")


def _mint_ref() -> str:
    """Mint an opaque store key. The ``ev_`` prefix is a namespace marker for humans reading logs, NOT
    a parseable schema — a caller MUST NOT split on it, and the body is unguessable so it cannot be
    constructed (ADR-003 §1/§4)."""
    return f"ev_{secrets.token_hex(16)}"


__all__ = [
    "EvidenceArtifact",
    "EvidenceKind",
    "EvidenceStore",
    "RetentionPolicy",
    "RetentionClass",
    "EVICTION_ORDER",
    "Authorizer",
    "resolve_evidence",
    "evidence_ref_of",
    "EvidenceError",
    "EvidenceExpired",
    "EvidenceUnavailable",
    "EvidenceAccessDenied",
    "EvidenceCorrupt",
]
