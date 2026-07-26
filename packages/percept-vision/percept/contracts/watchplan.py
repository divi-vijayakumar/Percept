"""WatchPlan / ResolvedWatch / provenance DTOs (ADR-008: configuration ownership, resolution,
provenance — ``docs/adr/ADR-008-configuration-ownership-resolution-provenance.md``).

ADR-008 names **six independently constructible configuration objects** and a two-step binder
(``ConfigResolver.resolve`` -> ``ResolvedWatch`` -> ``RuntimeCompiler.compile`` -> runtime
primitives). This module ships the DTOs for the FIRST half of that boundary — the portable,
typed shapes a resolver reads and produces. It does not implement the resolver itself (that is
``percept.resolver.ConfigResolver``, outside this zero-dependency package) and it performs no
I/O, no env read, and no runtime execution: constructing any DTO here is pure shape validation,
the same discipline every other ``percept.contracts`` DTO follows.

Six DTOs ship here:

- :class:`WatchPlan` — the user's stated intent, portable and typed: the condition VERBATIM
  (never compiled to a rigid predicate — the same cardinal rule the module docstring above
  states for ``Vision.judge``), plus the per-field ``explicit`` overrides a caller names. This is
  the ADR's ``WatchPlan`` ownership-table object: "the goal text, an explicit
  ``salience_stat="block_max"`` override".
- :class:`SettingOrigin` — one resolved field's provenance: which layer/object supplied the
  winning value, and why.
- :class:`ResolvedWatch` — the resolver's OUTPUT: one concrete value per resolvable field, each
  carrying a :class:`SettingOrigin`. Immutable (frozen) by construction — ADR-008's own
  invariant: "every field on it is written exactly once, at resolution, with its provenance
  attached in the same assignment."
- :class:`FieldDiagnostic` — a single accepted/dropped/coerced/warned field, always carrying a
  closed ``code`` (never prose-only), so a receipt is machine-actionable, not just readable.
- :class:`WatchReceipt` — what actually got applied when a plan was armed: status + per-field
  diagnostics.

``source_profile`` / ``domain_pack`` / ``execution_policy`` / ``components`` on
:class:`ResolvedWatch` are typed ``Mapping[str, Any] | None`` (plain, JSON-shaped snapshots), not
the live ``percept.sourceprofile.SourceProfile`` / (not-yet-existing) ``DomainPack`` /
``ExecutionPolicy`` objects. This is deliberate: ``percept.contracts`` is the frozen, stdlib-only
membrane extracted so non-vision consumers can import it standalone (see the package docstring)
— it does not import the rest of ``percept`` and the rest of ``percept`` imports FROM it, never
the other way. ``percept.resolver`` (outside this package) holds the real objects and fills these
snapshot fields from them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping

#: The closed status vocabulary a :class:`WatchReceipt` may report. Mirrors the shape (not the
#: vocabulary) of ``percept.gate``'s closed FIRE/ASK/SILENT set: a receipt's status is always one
#: of these four, never a free-form string.
ReceiptStatus = Literal["applied", "applied_with_warnings", "partial", "rejected"]
RECEIPT_STATUSES: tuple[str, ...] = ("applied", "applied_with_warnings", "partial", "rejected")


@dataclass(frozen=True)
class FieldDiagnostic:
    """One field-level fact about a resolution/authoring attempt — never prose-only.

    - ``path`` — the dotted, stable locator the field is known by (the same identifier space
      :class:`percept.tunables.TunableSpec.path` uses, e.g. ``"Settings.edge.salience_stat"``,
      ``"Goal.state.rhythm.cadence_s"``) so a diagnostic and a tunable-registry entry always
      speak the same vocabulary.
    - ``code`` — a short, closed-ish machine code (``"ownership_violation"``, ``"unknown_key"``,
      ``"coerced"``, ``"env_overrides_explicit_profile"``, ...). This is the field that MUST be
      present and MUST be checked by calling code — ``message`` is a human convenience layered on
      top, never a substitute for it.
    - ``goal_id`` — which goal/concern this diagnostic is scoped to, or ``None`` for a plan-level
      (not per-goal) fact.
    - ``message`` — an optional one-line human-readable elaboration. May be empty; ``code`` alone
      must always be enough for a caller to branch on.
    """

    path: str
    code: str
    goal_id: str | None = None
    message: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValueError(f"FieldDiagnostic.path must be a non-empty string, got {self.path!r}")
        if not isinstance(self.code, str) or not self.code.strip():
            raise ValueError(
                f"FieldDiagnostic.code must be a non-empty string (never prose-only), got {self.code!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "code": self.code, "goal_id": self.goal_id, "message": self.message}


@dataclass(frozen=True)
class SettingOrigin:
    """One resolved field's provenance — WHO won, and why.

    - ``value`` — the winning value itself (repeated here, not just in :class:`ResolvedWatch`,
      so a :class:`SettingOrigin` is self-contained and quotable on its own, e.g. in a log line).
    - ``owner_layer`` — which of ADR-008's declared objects (or precedence tiers) supplied it:
      one of ``"SDK"``, ``"SourceProfile"``, ``"DomainPack"``, ``"ExecutionPolicy"``,
      ``"OptimizationLock"``, ``"Deployment"``, ``"WatchPlan"``, or ``"Gate"`` (a field the ADR's
      Invariants wall off from every configuration object — see
      :data:`percept.tunables.OWNERS`). Not itself validated against a closed set here (that
      would duplicate :mod:`percept.tunables`'s own registry, which is the source of truth) but
      every value :mod:`percept.resolver` emits is drawn from exactly that vocabulary.
    - ``source_id`` — the concrete, named thing that supplied it within that layer (a profile
      name, ``"env:PERCEPT_SALIENCE_STAT"``, ``"watch_plan.explicit"``, ``"sdk_default"``, ...).
    - ``reason`` — one line of why this source won at this field (e.g. ``"explicit WatchPlan
      override beats env"``, ``"env beats SourceProfile default"``).
    - ``validated_by`` — the id of an :class:`~percept.resolver.OptimizationLock`-shaped object
      that validated this value, if any; ``None`` when no lock covers this field.
    """

    value: Any
    owner_layer: str
    source_id: str
    reason: str
    validated_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "owner_layer": self.owner_layer,
            "source_id": self.source_id,
            "reason": self.reason,
            "validated_by": self.validated_by,
        }


#: The grammar an :class:`ArtifactBinding.ref` must satisfy — a flat, path-separator-free name token
#: (``[A-Za-z0-9._@-]+``). Deliberately excludes ``/`` and ``\\`` so a ref can never smuggle a path
#: traversal into a resolver that reads it back; the digest, not the ref, is the integrity anchor.
_ARTIFACT_REF_RE = re.compile(r"[A-Za-z0-9._@-]+")
#: A bare, lowercase, 64-char SHA-256 hex digest (no ``sha256:`` prefix — that is stripped on input).
_SHA256_HEX_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ArtifactBinding:
    """An immutable, content-addressed pointer to a judge-context artifact.

    A :class:`WatchPlan` may bind a piece of judge context (a prompt envelope, a rubric, a policy
    doc) by *identity + content hash* so the exact bytes a plan was authored against are pinned:
    resolution refuses a binding whose digest does not match the artifact the deployment resolves,
    and the applied digest rides through to the :class:`WatchReceipt` for audit. It carries no bytes
    itself — only the ref that names the artifact and the digest that fixes its content.

    - ``ref`` — the artifact name, VERBATIM. Must be non-empty and match ``[A-Za-z0-9._@-]+``
      (:data:`_ARTIFACT_REF_RE`): a flat token with no path separators, so a ref cannot encode a
      traversal.
    - ``sha256`` — the artifact's SHA-256, stored as a bare 64-char lowercase hex digest. Input
      normalization accepts an optional ``"sha256:"`` prefix and any letter case and reduces both to
      the canonical bare-lowercase form, so ``"sha256:ABC…"`` and ``"abc…"`` bind identically.
    - ``media_type`` — the artifact's MIME type (default ``text/plain; charset=utf-8``); carried so a
      consumer reads the bytes with the right decoder without a second lookup.
    - ``schema_version`` — an OPTIONAL version tag for the artifact's own schema, or ``None``.
    """

    ref: str
    sha256: str
    media_type: str = "text/plain; charset=utf-8"
    schema_version: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.ref, str) or not self.ref.strip():
            raise ValueError(f"ArtifactBinding.ref must be a non-empty string, got {self.ref!r}")
        if not _ARTIFACT_REF_RE.fullmatch(self.ref):
            raise ValueError(
                f"ArtifactBinding.ref must match [A-Za-z0-9._@-]+ (no path separators), got {self.ref!r}"
            )
        raw = self.sha256
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"ArtifactBinding.sha256 must be a non-empty hex digest, got {raw!r}")
        digest = raw.strip()
        if digest[:7].lower() == "sha256:":
            digest = digest[7:]
        digest = digest.lower()
        if not _SHA256_HEX_RE.fullmatch(digest):
            raise ValueError(
                "ArtifactBinding.sha256 must be a 64-char SHA-256 hex digest "
                f"(optionally 'sha256:'-prefixed, any case, on input), got {raw!r}"
            )
        object.__setattr__(self, "sha256", digest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "sha256": self.sha256,
            "media_type": self.media_type,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class ResolvedJudgeSpec:
    """The resolver's OUTPUT for the judge slot — the concrete judge a requested ref bound to.

    Where :attr:`WatchPlan.judge` is a *request* (a name the user asked for), this is the *result*:
    which factory/provider/model actually resolved, plus the content addresses that pin the prompt
    envelope and any bound judge context. It is a provenance record — the ``components["judge"]`` block
    on :class:`ResolvedWatch` documents the nested wire projection of these same facts.

    - ``requested_ref`` — the judge ref the plan asked for, or ``None`` when the plan named none.
    - ``resolved_ref`` — the judge ref that actually resolved (may differ from the request under a
      documented fallback; see :attr:`WatchReceipt.fallback_used`).
    - ``factory_ref`` — the registered factory identity that constructed the judge.
    - ``provider_ref`` — the backend/provider identity the judge runs against.
    - ``model_requested`` / ``model_resolved`` — the model asked for vs. the model that bound.
    - ``prompt_envelope_sha256`` — content address of the prompt envelope the judge ran under, or
      ``None`` when not tracked.
    - ``context_ref`` / ``context_sha256`` — the bound judge-context artifact's ref + digest (mirrors
      the applied :class:`ArtifactBinding`), or ``None`` when the plan bound no context.
    """

    requested_ref: str | None
    resolved_ref: str
    factory_ref: str
    provider_ref: str
    model_requested: str
    model_resolved: str
    prompt_envelope_sha256: str | None
    context_ref: str | None
    context_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_ref": self.requested_ref,
            "resolved_ref": self.resolved_ref,
            "factory_ref": self.factory_ref,
            "provider_ref": self.provider_ref,
            "model_requested": self.model_requested,
            "model_resolved": self.model_resolved,
            "prompt_envelope_sha256": self.prompt_envelope_sha256,
            "context_ref": self.context_ref,
            "context_sha256": self.context_sha256,
        }


@dataclass(frozen=True)
class WatchPlan:
    """Portable, typed intent — the user's stated ask, before any resolution happens.

    ``condition`` is carried VERBATIM (the module docstring's cardinal rule: a plain-language
    condition is never compiled to a rigid ``attr_key=value`` at this layer either — that stays
    the judge's job downstream). Every other field is OPTIONAL — an unset field does not
    participate in resolution at all (ADR-008: "An *unset* WatchPlan field does not participate —
    it is not 'explicit,' so the next value down the chain applies untouched").

    - ``condition`` — the plain-language condition, verbatim.
    - ``sense`` — which physical channel/modality this plan expects to read (free text; mirrors
      :data:`percept.behavior.Modality` loosely without importing it — this package is stdlib-only
      and imports nothing from the rest of ``percept``).
    - ``target`` — what the condition is about (free text; entity/subject hint).
    - ``admission`` — a requested admission-strategy name, or ``None`` to defer to the resolved
      ``SourceProfile``.
    - ``evidence`` — a requested evidence-strategy name, or ``None``.
    - ``detectors`` — cheap edge-signal detector names this plan requests.
    - ``extractors`` — extractor names this plan requests (free text; open registry, mirrors
      ``percept.harness.detectors``'s own openness).
    - ``judge`` — a requested judge reference, or ``None``. Resolution validates it against the
      deployment-owned judge catalog and fails closed (a pre-arm error) if the requested judge is
      unavailable — it is a REQUEST, not an override: a plan may ask for a judge, but the deployment
      owns which judges exist, and an ask for one that does not resolve stops the arm rather than
      silently substituting.
    - ``judge_context`` — an OPTIONAL :class:`ArtifactBinding` pinning the exact judge-context
      artifact (prompt envelope / rubric / policy doc) this plan was authored against, by ref + SHA-256.
      ``None`` (the default) binds no context and — critically — makes :meth:`to_dict` emit NOTHING new,
      so an existing plan's wire bytes (and its ``plan_hash``) are byte-for-byte what they were before
      this field existed. When set, the binding is projected onto the wire as the flat
      ``judge_context_ref`` / ``judge_context_sha256`` / ``judge_context_media_type`` keys (plus
      ``judge_context_schema_version`` only when the binding carries one); :meth:`from_dict` reconstructs
      it and REFUSES a half-specified binding (a ref without a hash, or a hash without a ref).
    - ``consequence`` — the ``(miss_cost, false_fire_cost, protected_floor)`` triple, expressed as
      a plain mapping (not the ``percept.contracts.Consequence`` frozen dataclass — this module
      does not import from ``percept.contracts``'s own sibling definitions to avoid a second DTO
      needing to agree with the first; a resolver composing both is expected to build a real
      ``Consequence`` from this mapping).
    - ``rearm`` — the requested re-arm policy name, or ``None``.
    - ``uncertainty`` — a requested ASK/verification posture hint, or ``None``.
    - ``schedule`` — a plain ``{"windows": [...], "days": [...]}`` mapping shaped like
      ``percept.contracts.Schedule``, or ``None`` (always armed).
    - ``explicit`` — the field-level overrides this plan authors, keyed by the SAME dotted path
      vocabulary :class:`percept.tunables.TunableSpec.path` uses (e.g.
      ``{"WatchProfile.salience_stat": "block_max"}``). This is explicit per-Watch intent and is
      what :mod:`percept.resolver` reads to apply the highest-precedence tier; the corresponding
      deployment runtime path remains SourceProfile-owned.
    - ``source`` — the frame-source key (``watch.source``'s own WatchPlan-ownership example).
    - ``profile`` — the ``WatchProfile``/``SourceProfile`` selection name a caller passes, or
      ``None``.
    - ``id`` — a caller-supplied plan identifier, or ``None`` (the resolver mints one if absent).
    """

    condition: str
    sense: str | None = None
    target: str | None = None
    admission: str | None = None
    evidence: str | None = None
    detectors: tuple[str, ...] = ()
    extractors: tuple[str, ...] = ()
    judge: str | None = None
    consequence: Mapping[str, Any] | None = None
    rearm: str | None = None
    uncertainty: str | None = None
    schedule: Mapping[str, Any] | None = None
    explicit: Mapping[str, Any] = field(default_factory=dict)
    source: str = "camera"
    profile: str | None = None
    id: str | None = None
    # Appended LAST + defaulted None so field order (and thus positional construction) of every
    # pre-1.10 field is untouched; an unset binding contributes nothing to the wire (see to_dict).
    judge_context: ArtifactBinding | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.condition, str) or not self.condition.strip():
            raise ValueError(
                f"WatchPlan.condition must be a non-empty string (verbatim, never compiled), "
                f"got {self.condition!r}"
            )
        object.__setattr__(self, "detectors", tuple(self.detectors))
        object.__setattr__(self, "extractors", tuple(self.extractors))
        explicit = self.explicit
        if not isinstance(explicit, Mapping):
            raise ValueError(f"WatchPlan.explicit must be a mapping, got {type(explicit).__name__}")
        for key in explicit:
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"WatchPlan.explicit keys must be non-empty strings, got {key!r}")
        object.__setattr__(self, "explicit", MappingProxyType(dict(explicit)))
        if self.consequence is not None and not isinstance(self.consequence, Mapping):
            raise ValueError("WatchPlan.consequence must be a mapping or None")
        if self.schedule is not None and not isinstance(self.schedule, Mapping):
            raise ValueError("WatchPlan.schedule must be a mapping or None")
        if self.judge_context is not None and not isinstance(self.judge_context, ArtifactBinding):
            raise ValueError(
                "WatchPlan.judge_context must be an ArtifactBinding or None, got "
                f"{type(self.judge_context).__name__}"
            )

    def to_dict(self) -> dict[str, Any]:
        d = {
            "condition": self.condition,
            "sense": self.sense,
            "target": self.target,
            "admission": self.admission,
            "evidence": self.evidence,
            "detectors": list(self.detectors),
            "extractors": list(self.extractors),
            "judge": self.judge,
            "consequence": dict(self.consequence) if self.consequence is not None else None,
            "rearm": self.rearm,
            "uncertainty": self.uncertainty,
            "schedule": dict(self.schedule) if self.schedule is not None else None,
            "explicit": dict(self.explicit),
            "source": self.source,
            "profile": self.profile,
            "id": self.id,
        }
        # BACK-COMPAT INVARIANT: when no judge context is bound, emit NOTHING new — the key set (and
        # thus the serialized bytes and plan_hash) is identical to the pre-1.10 output. The binding
        # rides the wire as flat keys, added only when present so an unset plan is unchanged.
        if self.judge_context is not None:
            b = self.judge_context
            d["judge_context_ref"] = b.ref
            d["judge_context_sha256"] = b.sha256
            d["judge_context_media_type"] = b.media_type
            if b.schema_version is not None:
                d["judge_context_schema_version"] = b.schema_version
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WatchPlan":
        """Reconstruct a :class:`WatchPlan` from its :meth:`to_dict` wire form — the plan-read-back path.

        Folds the flat ``judge_context_*`` wire keys back into an :class:`ArtifactBinding` and REFUSES a
        half-specified binding: a ``judge_context_ref`` without a ``judge_context_sha256`` (or vice versa)
        raises ``ValueError`` rather than binding an unpinned or unnamed artifact. A payload carrying no
        ``judge_context_*`` keys reconstructs exactly as a pre-1.10 plan did (``judge_context=None``).
        """
        payload = dict(data)
        ref = payload.pop("judge_context_ref", None)
        sha = payload.pop("judge_context_sha256", None)
        media = payload.pop("judge_context_media_type", None)
        schema_version = payload.pop("judge_context_schema_version", None)
        existing = payload.pop("judge_context", None)
        if (ref is None) != (sha is None):
            raise ValueError(
                "WatchPlan judge_context wire fields must appear together: judge_context_ref requires "
                "judge_context_sha256 and vice versa (a ref without a hash, or a hash without a ref, is "
                "a half-specified binding)"
            )
        binding: ArtifactBinding | None = None
        if ref is not None and sha is not None:
            kwargs: dict[str, Any] = {"ref": ref, "sha256": sha}
            if media is not None:
                kwargs["media_type"] = media
            if schema_version is not None:
                kwargs["schema_version"] = schema_version
            binding = ArtifactBinding(**kwargs)
        elif isinstance(existing, ArtifactBinding):
            binding = existing
        return cls(judge_context=binding, **payload)


@dataclass(frozen=True)
class ResolvedWatch:
    """The resolver's OUTPUT — one concrete value per resolvable field, each with its
    :class:`SettingOrigin`. Immutable: every field is written exactly once, at resolution, by
    :meth:`percept.resolver.ConfigResolver.resolve` — nothing may construct one field-by-field
    and claim it went through resolution (ADR-008 §I/O contract).

    - ``plan`` — the :class:`WatchPlan` this was resolved from.
    - ``source_profile`` / ``domain_pack`` / ``execution_policy`` — plain, JSON-shaped snapshots
      (``to_dict()``-style mappings) of whichever real objects contributed, or ``None`` when that
      layer contributed nothing. Snapshots, not live objects, so this package stays
      zero-dependency (see module docstring).
    - ``components`` — a snapshot of whichever runtime-component identity information the
      resolver has at resolution time: ``{"providers": {...}, "catalog_component_hash": "..."}``,
      where ``providers`` is the provider-NAME identities (``vision``/``conductor``/``specialist``/
      ``stt``/``tts``) read back from :func:`percept.runtime_compiler.runtime_compile`, and
      ``catalog_component_hash`` is :attr:`percept.catalog.CatalogSnapshot.component_hash` — the
      registered component SET's own content address at resolution time.
      When a plan requested a judge (:attr:`WatchPlan.judge`), ``components`` ALSO carries an additive
      ``components["judge"]`` block — the nested wire projection of :attr:`ResolvedWatch.judge` — present
      ONLY when a judge was requested::

          {"requested_ref", "resolved_ref", "factory_ref", "provider", "model_requested",
           "model_resolved", "prompt_envelope_sha256", "context": {"ref", "sha256"}}

      A plan that named no judge omits the block entirely, so a pre-1.10 ``components`` snapshot is
      byte-for-byte unchanged.
    - ``effective`` — the plain resolved value per field, keyed by the same dotted-path
      vocabulary as :class:`percept.tunables.TunableSpec.path`. This is the "what" view.
    - ``provenance`` — the same keys, each mapped to its full :class:`SettingOrigin`. This is the
      "why" view; ``effective[k] == provenance[k].value`` always holds.
    - ``plan_hash`` / ``component_snapshot_hash`` / ``optimization_lock_hash`` — content
      addresses (``percept.tracing.content_hash``-shaped hex digests) over the plan, the
      ``components`` snapshot above, and whichever ``OptimizationLock`` was applied (``None`` when
      no lock applied). Hashing is NOT reimplemented here — :mod:`percept.resolver` computes these
      via the existing :func:`percept.tracing.content_hash` / :class:`percept.tracing.PlanIdentity`
      seams and hands the digests to this constructor.
    """

    plan: WatchPlan
    source_profile: Mapping[str, Any] | None
    domain_pack: Mapping[str, Any] | None
    execution_policy: Mapping[str, Any] | None
    components: Mapping[str, Any]
    effective: Mapping[str, Any]
    provenance: Mapping[str, SettingOrigin]
    plan_hash: str
    component_snapshot_hash: str
    optimization_lock_hash: str | None = None
    # Appended LAST + defaulted None so pre-1.10 field order/construction is untouched; a resolution
    # that requested no judge leaves this None and its serialization (see to_dict) emits nothing new.
    judge: ResolvedJudgeSpec | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.plan, WatchPlan):
            raise ValueError(f"ResolvedWatch.plan must be a WatchPlan, got {type(self.plan).__name__}")
        if self.judge is not None and not isinstance(self.judge, ResolvedJudgeSpec):
            raise ValueError(
                f"ResolvedWatch.judge must be a ResolvedJudgeSpec or None, got {type(self.judge).__name__}"
            )
        object.__setattr__(self, "source_profile", _freeze_snapshot(self.source_profile))
        object.__setattr__(self, "domain_pack", _freeze_snapshot(self.domain_pack))
        object.__setattr__(self, "execution_policy", _freeze_snapshot(self.execution_policy))
        object.__setattr__(self, "components", _freeze_snapshot(self.components) or MappingProxyType({}))
        effective = dict(self.effective)
        provenance = dict(self.provenance)
        if set(effective) != set(provenance):
            raise ValueError(
                "ResolvedWatch.effective and .provenance must carry exactly the same field keys; "
                f"effective-only={sorted(set(effective) - set(provenance))} "
                f"provenance-only={sorted(set(provenance) - set(effective))}"
            )
        for path, origin in provenance.items():
            if not isinstance(origin, SettingOrigin):
                raise ValueError(f"ResolvedWatch.provenance[{path!r}] must be a SettingOrigin")
        object.__setattr__(self, "effective", MappingProxyType(effective))
        object.__setattr__(self, "provenance", MappingProxyType(provenance))
        if not isinstance(self.plan_hash, str) or not self.plan_hash:
            raise ValueError("ResolvedWatch.plan_hash must be a non-empty string")
        if not isinstance(self.component_snapshot_hash, str) or not self.component_snapshot_hash:
            raise ValueError("ResolvedWatch.component_snapshot_hash must be a non-empty string")

    def explain(self) -> str:
        """Render every effective field with its owner layer + source — one line per field, sorted
        by path so the output is deterministic. Meant for a human at 2am asking "where did this
        threshold come from?" (ADR-008 §Context), not for machine parsing (use :attr:`provenance`
        for that)."""
        lines = [f"ResolvedWatch (plan_hash={self.plan_hash[:12]}):"]
        for path in sorted(self.provenance):
            origin = self.provenance[path]
            note = f" [{origin.reason}]" if origin.reason else ""
            lock = f" validated_by={origin.validated_by}" if origin.validated_by else ""
            lines.append(
                f"  {path} = {origin.value!r}  <- {origin.owner_layer} ({origin.source_id}){note}{lock}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "plan": self.plan.to_dict(),
            "source_profile": dict(self.source_profile) if self.source_profile is not None else None,
            "domain_pack": dict(self.domain_pack) if self.domain_pack is not None else None,
            "execution_policy": dict(self.execution_policy) if self.execution_policy is not None else None,
            "components": dict(self.components),
            "effective": dict(self.effective),
            "provenance": {k: v.to_dict() for k, v in self.provenance.items()},
            "plan_hash": self.plan_hash,
            "component_snapshot_hash": self.component_snapshot_hash,
            "optimization_lock_hash": self.optimization_lock_hash,
        }
        # Additive + present ONLY when a judge was requested, so a pre-1.10 ResolvedWatch serializes
        # byte-for-byte as before.
        if self.judge is not None:
            d["judge"] = self.judge.to_dict()
        return d


def _freeze_snapshot(snapshot: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if snapshot is None:
        return None
    if not isinstance(snapshot, Mapping):
        raise ValueError(f"expected a mapping snapshot or None, got {type(snapshot).__name__}")
    return MappingProxyType(dict(snapshot))


@dataclass(frozen=True)
class WatchReceipt:
    """What actually happened when a :class:`WatchPlan` was applied — status + per-field facts.

    - ``status`` — ``"applied"`` (everything the plan asked for landed), ``"applied_with_warnings"``
      (landed, but at least one loud diagnostic was raised — e.g. env overriding an explicitly
      selected profile), ``"partial"`` (some fields dropped/coerced, the plan still armed),
      ``"rejected"`` (the plan could not be applied at all — an ownership violation or unknown
      key raised before anything armed).
    - ``watch_ids`` — the goal/concern ids this plan resolved to, once armed.
    - ``plan_revision`` — the registry revision this application landed at, if applicable.
    - ``accepted_fields`` — dotted paths that resolved and applied cleanly.
    - ``dropped_fields`` — :class:`FieldDiagnostic` entries for fields that were refused/dropped.
    - ``coerced_fields`` — :class:`FieldDiagnostic` entries for fields that were accepted after
      being coerced/clamped (mirrors the ``coerce_authored_*`` family's own DROP-don't-guess
      posture, but reported here instead of silently disappearing).
    - ``warnings`` — :class:`FieldDiagnostic` entries that are loud but non-fatal (the
      env-overrides-explicit-profile case ADR-008 names).
    - ``judge_requested`` / ``judge_resolved`` — the judge ref the plan asked for vs. the judge that
      actually bound at arm time; ``None`` when the plan named no judge. They differ only
      under a documented fallback — see ``fallback_used``.
    - ``judge_context_ref`` / ``judge_context_sha256`` — the applied judge-context binding's ref +
      canonical (bare-lowercase) SHA-256 digest, echoed onto the receipt for audit; ``None`` when the
      plan bound no judge context.
    - ``fallback_used`` — ``True`` when the resolved judge (or its model) is NOT the one requested
      because a documented fallback applied; ``False`` (the default) when the request bound exactly.
    """

    status: ReceiptStatus
    watch_ids: tuple[str, ...] = ()
    plan_revision: int | None = None
    accepted_fields: tuple[str, ...] = ()
    dropped_fields: tuple[FieldDiagnostic, ...] = ()
    coerced_fields: tuple[FieldDiagnostic, ...] = ()
    warnings: tuple[FieldDiagnostic, ...] = ()
    # Judge provenance — all appended LAST + defaulted so a pre-1.10 receipt is unchanged.
    judge_requested: str | None = None
    judge_resolved: str | None = None
    judge_context_ref: str | None = None
    judge_context_sha256: str | None = None
    fallback_used: bool = False

    def __post_init__(self) -> None:
        if self.status not in RECEIPT_STATUSES:
            raise ValueError(f"WatchReceipt.status must be one of {RECEIPT_STATUSES}, got {self.status!r}")
        object.__setattr__(self, "watch_ids", tuple(self.watch_ids))
        object.__setattr__(self, "accepted_fields", tuple(self.accepted_fields))
        object.__setattr__(self, "dropped_fields", tuple(self.dropped_fields))
        object.__setattr__(self, "coerced_fields", tuple(self.coerced_fields))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        for group_name in ("dropped_fields", "coerced_fields", "warnings"):
            for item in getattr(self, group_name):
                if not isinstance(item, FieldDiagnostic):
                    raise ValueError(f"WatchReceipt.{group_name} must contain only FieldDiagnostic entries")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "watch_ids": list(self.watch_ids),
            "plan_revision": self.plan_revision,
            "accepted_fields": list(self.accepted_fields),
            "dropped_fields": [d.to_dict() for d in self.dropped_fields],
            "coerced_fields": [d.to_dict() for d in self.coerced_fields],
            "warnings": [d.to_dict() for d in self.warnings],
            "judge_requested": self.judge_requested,
            "judge_resolved": self.judge_resolved,
            "judge_context_ref": self.judge_context_ref,
            "judge_context_sha256": self.judge_context_sha256,
            "fallback_used": self.fallback_used,
        }


__all__ = [
    "ReceiptStatus",
    "RECEIPT_STATUSES",
    "ArtifactBinding",
    "ResolvedJudgeSpec",
    "FieldDiagnostic",
    "SettingOrigin",
    "WatchPlan",
    "ResolvedWatch",
    "WatchReceipt",
]
