"""ConfigResolver — the ADR-008 resolution boundary, step 1 of the two-step binder.

``docs/adr/ADR-008-configuration-ownership-resolution-provenance.md`` names six independently
constructible configuration objects and draws the boundary as two functions:

    (SourceProfile, DomainPack, ExecutionPolicy, DeploymentConfig, OptimizationLock, WatchPlan)
            |
            v
       ConfigResolver.resolve(...)   <- THIS MODULE
            |
            v
       ResolvedWatch
            |
            v
       RuntimeCompiler.compile(...)  <- A LATER CARD, NOT THIS ONE

This module ships ONLY :class:`ConfigResolver` (step 1). It performs **no runtime execution**:
``resolve()`` reads configuration inputs and returns an in-memory :class:`~percept.contracts.ResolvedWatch`.
It never touches ``percept.runtime.worker``, ``percept.agent``'s runtime paths, ``percept.gate``,
or ``percept.contracts.WatchSpec`` — turning a ``ResolvedWatch`` into those runtime primitives is
``RuntimeCompiler``'s job, out of scope here.

Precedence, low to high (ADR-008 §Precedence)::

    SDK defaults < SourceProfile < DomainPack < ExecutionPolicy < OptimizationLock
                 < Deployment (incl. PERCEPT_* env) < WatchPlan explicit

**Ownership gates precedence, not the reverse.** A field write from a layer that does not own
that field's class is a :class:`ConfigError`, raised before any value is produced — never
last-writer-wins. Ownership is read from :data:`percept.tunables.REGISTRY` (the tunables inventory),
the single source of truth this module does not duplicate.

``DomainPack`` / ``ExecutionPolicy`` / ``OptimizationLock`` do not exist as real objects yet (ADR-008
§Six declared objects: "new — not yet in code"). This module defines them here as MINIMAL
placeholder dataclasses — an ``id``/``version``/``values`` mapping, nothing more — so the
precedence chain is real and testable now; their full contracts (pack loading, template YAML,
optimizer logic) are later cards' scope, deliberately not built here.

``DeploymentConfig`` also does not exist as a real object yet; ``percept.config.Settings`` /
``load_settings()`` continue to play its role informally (ADR-008 §Consequences), which is what
this resolver composes directly — no new deployment object is invented.

Two "asserting" layers deliberately have no literal field on today's real objects for many of the
Settings-shaped tunables the ADR ownership table assigns them (e.g. no ``SourceProfile.motion_gate``
attribute exists): the registry's ``owner`` assignment is the ADR's TARGET ownership, not a claim
that every owning object already carries every field. Where a real object's own attributes ARE the
field (e.g. ``SourceProfile.admission_strategy``), this resolver reads them directly. Where the
"default" for a Settings-shaped field is contributed today through :func:`percept.profile_resolution.resolve_situation`
(called VERBATIM, unchanged — same 3-axis lookup, same wildcard fallback), this resolver stamps
those contributed values at the tunables-registry owner their PATH names (mostly ``"WatchPlan"``
for ``salience_stat``/``stat_thresholds``/``plan_mode``, ``"SourceProfile"`` for ``ego_compensate``,
``"Gate"`` for ``gate.require_false_first`` — Gate fields ride along for transparency but are never
ownership-gated, per the Invariants' "One Gate" rule: no configuration object may carry gate
semantics).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from .config import Settings, load_settings
from .contracts.watchplan import (
    FieldDiagnostic,
    ResolvedWatch,
    SettingOrigin,
    WatchPlan,
    WatchReceipt,
)
from .profile_resolution import resolve_profile, resolve_situation
from .sourceprofile import SourceProfile, coast_sense_prior
from .tracing import content_hash
from .tunables import REGISTRY, TunableSpec
from .watchprofile import CadenceProfile, WatchProfileRegistry

# ---------------------------------------------------------------------------
# Precedence tiers (ADR-008 §Precedence). Strings, not an Enum, so they are the SAME vocabulary
# percept.tunables.OWNERS already uses for "owner" (plus "SDK" and "Deployment", the two tiers
# that are not one of the six declared objects/escape hatches but ARE precedence positions).
# ---------------------------------------------------------------------------

LAYER_SDK = "SDK"
LAYER_SOURCE_PROFILE = "SourceProfile"
LAYER_DOMAIN_PACK = "DomainPack"
LAYER_EXECUTION_POLICY = "ExecutionPolicy"
LAYER_OPTIMIZATION_LOCK = "OptimizationLock"
LAYER_DEPLOYMENT = "Deployment"
LAYER_WATCH_PLAN = "WatchPlan"
LAYER_GATE = "Gate"

#: Precedence order, low to high — index is rank (higher index wins a conflict at the same path).
PRECEDENCE: tuple[str, ...] = (
    LAYER_SDK,
    LAYER_SOURCE_PROFILE,
    LAYER_DOMAIN_PACK,
    LAYER_EXECUTION_POLICY,
    LAYER_OPTIMIZATION_LOCK,
    LAYER_DEPLOYMENT,
    LAYER_WATCH_PLAN,
)

_RANK: dict[str, int] = {name: i for i, name in enumerate(PRECEDENCE)}


class ConfigError(ValueError):
    """A configuration layer attempted to set a field outside its ownership, or named an unknown
    tunable path. Raised by :meth:`ConfigResolver.resolve` BEFORE any value is produced (ADR-008
    §Ownership table: "not last-writer-wins... a ConfigError, raised... before any value is
    produced")."""


#: path -> TunableSpec, built once from the tunables registry (the single source of truth for
#: ownership this module reads and never duplicates).
_BY_PATH: dict[str, TunableSpec] = {spec.path: spec for spec in REGISTRY}


# ---------------------------------------------------------------------------
# Minimal placeholder resolver inputs (ADR-008: "new — not yet in code" for all three). Only the
# shape a precedence chain needs: id/version/values. Real pack loading / scheduling posture /
# optimizer validation are later cards.
# ---------------------------------------------------------------------------


def _freeze_values(values: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if values is None:
        return MappingProxyType({})
    if not isinstance(values, Mapping):
        raise ConfigError(f"values must be a mapping, got {type(values).__name__}")
    for key in values:
        if not isinstance(key, str) or not key.strip():
            raise ConfigError(f"values keys must be non-empty strings, got {key!r}")
    return MappingProxyType(dict(values))


@dataclass(frozen=True)
class DomainPack:
    """Placeholder resolver input — ADR-008: "domain vocabulary, phrase templates, per-domain
    recommended WatchProfile/SourceProfile pairings... new, not yet in code". ``values`` is keyed
    by the same dotted tunables-registry path vocabulary as :attr:`WatchPlan.explicit`. No tunable
    in :data:`percept.tunables.REGISTRY` is owned by ``"DomainPack"`` today — any non-empty
    ``values`` therefore raises :class:`ConfigError` until a card assigns it real ownership; this
    is intentional, not a bug (see the ADR's own "open registry" framing)."""

    id: str
    version: str = "v1"
    values: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ConfigError(f"DomainPack.id must be a non-empty string, got {self.id!r}")
        object.__setattr__(self, "values", _freeze_values(self.values))


@dataclass(frozen=True)
class ExecutionPolicy:
    """Placeholder resolver input — ADR-008: "resource/scheduling posture: judge-call budget,
    cadence ceilings, concurrency, degrade-under-load behavior... new, not yet in code"."""

    id: str
    version: str = "v1"
    values: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ConfigError(f"ExecutionPolicy.id must be a non-empty string, got {self.id!r}")
        object.__setattr__(self, "values", _freeze_values(self.values))


@dataclass(frozen=True)
class OptimizationLock:
    """Placeholder resolver input — ADR-008: "a validated, hashed selection of tunables an
    optimizer previously proved safe — never authored by hand... new, not yet in code".

    Every path in ``values`` must be ``safe_to_optimize: true`` in :data:`percept.tunables.REGISTRY`
    (Invariants: "The optimizer tunes only declared-safe paths... a lock covering any other field is
    invalid to construct") — this is checked at RESOLUTION time (:meth:`ConfigResolver.resolve`),
    not at construction, because it depends on the registry, the single source of truth this object
    does not duplicate. ``locked_hash``, if given, is the hash this lock claims to have validated
    its ``values`` under; :meth:`ConfigResolver.resolve` compares it against a freshly computed
    :func:`percept.tracing.content_hash` and reports a mismatch as a loud diagnostic rather than
    silently trusting a stale claim (ADR-008 §Corrections(d))."""

    id: str
    version: str = "v1"
    values: Mapping[str, Any] = field(default_factory=dict)
    locked_hash: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ConfigError(f"OptimizationLock.id must be a non-empty string, got {self.id!r}")
        object.__setattr__(self, "values", _freeze_values(self.values))

    @property
    def content_hash(self) -> str:
        """The hash this lock's OWN ``values`` hash to right now — compared against
        :attr:`locked_hash` by the resolver (ADR-008 §Corrections(d))."""
        return content_hash(dict(sorted(self.values.items())))


# ---------------------------------------------------------------------------
# Snapshot helpers — turn a real object into a {tunables-path: value} mapping. Pure functions,
# no I/O beyond the Settings/env read `load_settings()` itself already does.
# ---------------------------------------------------------------------------


def _settings_paths(settings: Settings) -> dict[str, Any]:
    """Every ``Settings.*`` tunables-registry path, read off a real :class:`Settings` instance via
    the SAME dotted path the registry names it by (``"Settings.gate.threshold"`` ->
    ``settings.gate.threshold``). Generic over the registry so a new ``Settings`` field picked up
    by the tunables inventory is read here with no per-field code."""
    out: dict[str, Any] = {}
    for spec in REGISTRY:
        if not spec.path.startswith("Settings."):
            continue
        parts = spec.path.split(".")[1:]
        obj: Any = settings
        try:
            for p in parts:
                obj = getattr(obj, p)
        except AttributeError:
            continue
        out[spec.path] = obj
    return out


def _source_profile_paths(profile: SourceProfile) -> dict[str, Any]:
    """A real :class:`SourceProfile`'s own fields, keyed by the tunables-registry paths that name
    them. Only :attr:`SourceProfile.parameters` sub-keys the registry actually names are included —
    an unrecognized parameter key is not silently promoted to a tunables path (it stays a plain
    SourceProfile-internal parameter the resolver has no opinion on)."""
    out: dict[str, Any] = {
        "SourceProfile.name": profile.name,
        "SourceProfile.modalities": profile.modalities,
        "SourceProfile.temporal_unit": profile.temporal_unit,
        "SourceProfile.admission_strategy": profile.admission_strategy,
        "SourceProfile.evidence_strategy": profile.evidence_strategy,
        "SourceProfile.identity_strategy": profile.identity_strategy,
        "SourceProfile.default_timebase": profile.default_timebase,
        "SourceProfile.stability": profile.stability,
        "SourceProfile.notes": profile.notes,
    }
    for key, value in profile.parameters.items():
        path = f"SourceProfile.parameters.{key}"
        if path in _BY_PATH:
            out[path] = value
    return out


#: Which ``Settings.edge.*`` threshold field a salience_stat's cutoff rides on — mirrors
#: ``_expand_explicit_path``'s ``stat_thresholds`` fan-out, applied here to a SourceProfile's OWN
#: declared ``parameters['salience_stat']``/``parameters['salience_threshold']`` (card C2: "salience
#: params -> edge settings").
_SALIENCE_THRESHOLD_PATH: dict[str, str] = {
    "mean": "Settings.edge.motion_gate",
    "block_max": "Settings.edge.motion_gate_block_max",
    "p95": "Settings.edge.motion_gate_p95",
}


def _source_profile_runtime_bindings(profile: SourceProfile) -> dict[str, Any]:
    """Card C2: bind a resolved :class:`SourceProfile`'s DECLARED strategy onto the runtime fields
    that actually implement it — the missing half of ADR-005's "SourceProfile selects strategies"
    promise, which until this card stopped at the profile object and never reached
    :class:`~percept.config.Settings` / ``Goal.state`` (this is why passing ``source_profile=`` to
    ``percept doctor`` used to change only the REPORTED note, never the replayed admission).

    - **salience params -> edge settings.** When the profile DECLARES a tuned admission signal
      (:attr:`SourceProfile.admission_tuned`) via ``parameters['salience_stat']``, that statistic and
      its ``parameters['salience_threshold']`` become ``Settings.edge.salience_stat`` and the
      matching per-stat cutoff (:data:`_SALIENCE_THRESHOLD_PATH`) — the same fields
      :func:`percept.harness.tier0.compile_watchspec` already reads. An UNTUNED profile contributes
      nothing here (there is no calibrated statistic to bind).
    - **coast floor -> the existing runtime seams.** A declared ``coast_interval_s``
      (:attr:`SourceProfile.coast_interval_s`) always contributes a SAMPLING floor
      (:func:`percept.sourceprofile.coast_sense_prior` -> ``Goal.state.sense.{base_fps,max_fps}``),
      so the capture loop keeps producing frames at least that often. When the profile is NOT
      admission_tuned, the coast is additionally the ONLY defensible admission mechanism, so it also
      forces ``Goal.state.cadence = {min_judge_interval: coast, gate_on: 'fixed'}`` — the tier-0
      gate's existing "look every cadence_s regardless of pixel motion" floor
      (``percept.harness.tier0._coerce_cadence`` / ``MotionGate.evaluate``). A TUNED profile (its own
      salience stat admits on its own) does NOT force ``gate_on='fixed'`` — that would SILENCE the
      real signal (the gate ignores motion entirely in fixed mode), not floor it — so once a profile
      is promoted to tuned the coast stays a pure sampling floor underneath the real admission.

    Both contributions are layered at the SourceProfile precedence tier (:data:`LAYER_SOURCE_PROFILE`,
    same as :func:`_source_profile_paths`), so an explicit WatchPlan/env value for the same path still
    wins per the normal precedence chain — this function only supplies the DEFAULT a profile implies.
    """
    out: dict[str, Any] = {}

    if profile.admission_tuned:
        stat = profile.parameters.get("salience_stat")
        if stat in _SALIENCE_THRESHOLD_PATH:
            out["Settings.edge.salience_stat"] = stat
            threshold = profile.parameters.get("salience_threshold")
            if isinstance(threshold, (int, float)) and not isinstance(threshold, bool):
                out[_SALIENCE_THRESHOLD_PATH[stat]] = float(threshold)

    coast = profile.coast_interval_s
    if coast:
        prior = coast_sense_prior(profile)
        if prior:
            out["Goal.state.sense.base_fps"] = prior["base_fps"]
            out["Goal.state.sense.max_fps"] = prior["max_fps"]
        if not profile.admission_tuned:
            out["Goal.state.cadence.min_judge_interval"] = coast
            out["Goal.state.cadence.gate_on"] = "fixed"

    return out


#: The situation-resolver override keys (percept.profile_resolution.SituationResolution.overrides)
#: mapped onto the tunables-registry path + owner tier they correspond to. Built from the REAL
#: registry ownership (WatchProfile.salience_stat/stat_thresholds/plan_mode -> "WatchPlan";
#: Settings.edge.ego_compensate -> "SourceProfile"; Settings.gate.require_false_first -> "Gate"),
#: not re-decided here.
_SITUATION_PATHS: tuple[tuple[str, str], ...] = (
    ("salience_stat", "WatchProfile.salience_stat"),
    ("stat_thresholds", "WatchProfile.stat_thresholds"),
    ("plan_mode", "WatchProfile.plan_mode"),
    ("ego_compensate", "Settings.edge.ego_compensate"),
)


def _situation_paths(overrides: Mapping[str, Any]) -> dict[str, Any]:
    """:func:`percept.profile_resolution.resolve_situation`'s ``overrides`` dict, translated to
    tunables-registry paths. ``overrides["gate"]["require_false_first"]`` is included under its own
    registry path (``"Settings.gate.require_false_first"``, owner ``"Gate"``) for transparency in
    :attr:`~percept.contracts.ResolvedWatch.provenance`, even though Gate-owned fields are never
    ownership-gated against the six objects (see module docstring)."""
    out: dict[str, Any] = {}
    for key, path in _SITUATION_PATHS:
        if key in overrides:
            out[path] = overrides[key]
    gate = overrides.get("gate")
    if isinstance(gate, Mapping) and "require_false_first" in gate:
        out["Settings.gate.require_false_first"] = gate["require_false_first"]
    return out


def _cadence_paths(cadence: CadenceProfile) -> dict[str, Any]:
    """A registry-resolved :class:`~percept.watchprofile.WatchProfile`'s ``cadence`` fields, keyed by
    their :data:`percept.tunables.REGISTRY` paths (``"WatchProfile.cadence.*"``, owner
    ``"ExecutionPolicy"``) — the ONLY situation-tier contribution that needs a real
    :class:`~percept.watchprofile.WatchProfileRegistry` lookup rather than pure axis derivation
    (:func:`percept.profile_resolution.resolve_situation` has no cadence opinion on its own)."""
    return {
        "WatchProfile.cadence.base_fps": cadence.base_fps,
        "WatchProfile.cadence.max_fps": cadence.max_fps,
        "WatchProfile.cadence.escalate_threshold": cadence.escalate_threshold,
        "WatchProfile.cadence.hold_s": cadence.hold_s,
    }


def _owner_layer_for_situation_path(path: str) -> str:
    spec = _BY_PATH.get(path)
    return spec.owner if spec is not None else LAYER_SOURCE_PROFILE


# ---------------------------------------------------------------------------
# Ownership gate
# ---------------------------------------------------------------------------


def _check_ownership(values: Mapping[str, Any], *, layer: str, expected_owner: str) -> None:
    """Raise :class:`ConfigError` for any path in ``values`` that is unknown to the registry, or
    whose registry owner is neither ``expected_owner`` nor ``"unclassified"`` (the two documented
    escape hatches from :mod:`percept.tunables` — an unclassified field resists ownership
    classification by design, so it is not gated against any one layer)."""
    for path in values:
        spec = _BY_PATH.get(path)
        if spec is None:
            raise ConfigError(
                f"{layer} named an unknown tunable path {path!r} — not in percept.tunables.REGISTRY "
                "(unknown keys raise loud, at authoring time, per ADR-008)"
            )
        if spec.owner not in (expected_owner, "unclassified"):
            raise ConfigError(
                f"{layer} may not set {path!r}: owned by {spec.owner!r}, not {expected_owner!r} "
                "(ADR-008: ownership gates precedence, not the reverse — this is a ConfigError, "
                "never a silent last-writer-wins)"
            )


def _check_optimization_lock_ownership(values: Mapping[str, Any]) -> None:
    """An :class:`OptimizationLock` is gated differently from the other five layers: it may cover
    ANY path marked ``safe_to_optimize: true`` in the registry, regardless of that path's nominal
    owner (Invariants: "The optimizer tunes only declared-safe paths... a lock covering any other
    field is invalid to construct") — never gated by owner-name equality the way SourceProfile/
    DomainPack/ExecutionPolicy/WatchPlan are."""
    for path in values:
        spec = _BY_PATH.get(path)
        if spec is None:
            raise ConfigError(
                f"OptimizationLock named an unknown tunable path {path!r} — not in "
                "percept.tunables.REGISTRY"
            )
        if not spec.safe_to_optimize:
            raise ConfigError(
                f"OptimizationLock may not cover {path!r}: safe_to_optimize=False in the registry "
                "(Invariants: 'the optimizer tunes only declared-safe paths')"
            )


def _expand_explicit_path(path: str, value: Any) -> dict[str, Any]:
    """Some ``owner="WatchPlan"`` registry paths are the EXPLICIT-OVERRIDE registration for an
    underlying ``Settings.*`` runtime field, not the runtime field itself — tunables.py Section D's
    own distinction between a DEFAULT-tier registration (owned by whichever object sets the
    underlying Settings-tree field, e.g. ``Settings.edge.salience_stat`` -> ``"SourceProfile"``) and
    an EXPLICIT per-context override (``WatchProfile.salience_stat`` -> ``"WatchPlan"``, the ADR's
    own worked example: "an explicit ``salience_stat='block_max'`` override").

    Both a :class:`WatchPlan`'s explicit fields AND :func:`percept.profile_resolution.resolve_situation`'s
    contributed overrides are asserted against the EXPLICIT-tier path (passes ownership / is the
    path the situation resolver's own vocabulary names); this translates it, AFTER the ownership
    check, to the runtime field it actually resolves — otherwise the value would land on
    ``effective["WatchProfile.salience_stat"]``, a key nothing else ever contests, instead of
    actually outranking a lower-precedence layer on ``Settings.edge.salience_stat``. One registry
    path may expand to several runtime paths (``stat_thresholds`` fans out to up to three
    ``motion_gate*`` fields, mirroring :func:`percept.profile_wiring.resolve_watch_profile`'s own
    field-by-field translation). A path with no special-cased expansion passes through unchanged."""
    if path == "WatchProfile.salience_stat":
        return {"Settings.edge.salience_stat": value}
    if path == "WatchProfile.stat_thresholds" and isinstance(value, Mapping):
        out: dict[str, Any] = {}
        if "mean" in value:
            out["Settings.edge.motion_gate"] = value["mean"]
        if "block_max" in value:
            out["Settings.edge.motion_gate_block_max"] = value["block_max"]
        if "p95" in value:
            out["Settings.edge.motion_gate_p95"] = value["p95"]
        return out
    return {path: value}


# ---------------------------------------------------------------------------
# ConfigResolver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LayerContribution:
    layer: str
    values: Mapping[str, Any]
    source_id: str
    reason: str


class ConfigResolver:
    """Resolves a subset of the ADR-008 configuration inputs into an immutable
    :class:`~percept.contracts.ResolvedWatch`. Stateless across calls except for
    :attr:`last_receipt`, a side-channel populated by the most recent :meth:`resolve` call so a
    caller can inspect loud diagnostics without changing ``resolve``'s return type (which mirrors
    the ADR's own ``ConfigResolver.resolve(...) -> ResolvedWatch`` arrow exactly)."""

    def __init__(self) -> None:
        self.last_receipt: WatchReceipt | None = None

    def resolve(
        self,
        watch_plan: WatchPlan,
        *,
        source_profile: SourceProfile | None = None,
        source_profile_explicit: bool = True,
        domain_pack: DomainPack | None = None,
        execution_policy: ExecutionPolicy | None = None,
        optimization_lock: OptimizationLock | None = None,
        situation: tuple[Any, Any, Any] | None = None,
        registry: "WatchProfileRegistry | None" = None,
        settings: Settings | None = None,
    ) -> ResolvedWatch:
        """Resolve ``watch_plan`` (plus whichever of the other five inputs are given — any may be
        omitted, an omitted object contributes nothing, it does not error) into a
        :class:`~percept.contracts.ResolvedWatch`.

        - ``source_profile_explicit`` — was ``source_profile`` a caller-named selection (the
          default) rather than some resolved fallback default? Feeds the "loud diagnostic on
          profile override" rule (ADR-008 §Precedence): only an EXPLICITLY selected profile earns
          the loud warning when Deployment/env overrides one of its fields.
        - ``situation`` — an optional ``(goal_temporality, camera_mount, start_state)`` triple fed
          VERBATIM to :func:`percept.profile_resolution.resolve_situation` (same 3-axis lookup,
          same wildcard fallback, unchanged) — passing it is itself an explicit choice, so its
          SourceProfile-tier contribution (``ego_compensate``) is treated the same as an explicit
          ``source_profile`` for the loud-diagnostic rule.
        - ``registry`` — OPTIONAL, a :class:`~percept.watchprofile.WatchProfileRegistry`. Only
          consulted when ``situation`` is ALSO given (ADR-008 open question 2: "a caller with
          (environment, goal_type, state) gets the registry profile through the same precedence
          chain"). When both are present, the REGISTERED profile for the resolved situation key is
          looked up (:func:`percept.profile_resolution.resolve_profile`, exact match then
          :data:`~percept.watchprofile.WILDCARD` fallback) and its ``cadence`` fields additionally
          contribute at the situation precedence slot (owner ``"ExecutionPolicy"`` per
          :data:`percept.tunables.REGISTRY`'s own ``WatchProfile.cadence.*`` entries), provenance-
          stamped with the winning profile's name. ``situation`` without ``registry`` (the default)
          is UNCHANGED from the earlier resolver: pure axis-derived overrides only, no profile
          lookup, no cadence contribution — byte-for-byte identical to omitting this parameter.
        - ``settings`` — a test-injection seam standing in for what :func:`percept.config.load_settings`
          would return (the Deployment/env layer). ``None`` (the default) calls
          :func:`~percept.config.load_settings` for real, reading ``PERCEPT_*`` env vars.

        Raises :class:`ConfigError` for an out-of-ownership write or an unknown tunable path,
        BEFORE any value is produced (ADR-008).
        """
        sdk_settings = Settings()
        sdk_snapshot = _settings_paths(sdk_settings)

        deployment_settings = settings if settings is not None else load_settings()
        deployment_snapshot = _settings_paths(deployment_settings)
        # Only a path whose deployment value actually DIFFERS from the bare SDK default is a real
        # Deployment-tier contribution; an unchanged value is the SDK default passing through, not
        # something Deployment "did" (ADR-008: "env beats profile default" is about a REAL override).
        deployment_contribution = {
            path: value for path, value in deployment_snapshot.items()
            if value != sdk_snapshot.get(path)
        }

        source_profile_values = _source_profile_paths(source_profile) if source_profile else {}
        source_profile_bindings = (
            _source_profile_runtime_bindings(source_profile) if source_profile else {}
        )

        situation_profile_name: str | None = None
        situation_source_id = "profile_resolution.resolve_situation"
        if situation is not None and registry is not None:
            registered_profile, situation_overrides = resolve_profile(*situation, registry=registry)
            situation_profile_name = registered_profile.name
            situation_source_id = f"profile_resolution.resolve_profile({registered_profile.name!r})"
        elif situation is not None:
            situation_overrides = resolve_situation(*situation).overrides
        else:
            situation_overrides = None
        situation_raw = _situation_paths(situation_overrides) if situation_overrides else {}
        if situation_profile_name is not None:
            situation_raw.update(_cadence_paths(registered_profile.cadence))
        # Expand each situation-contributed path to the runtime field it actually resolves (the
        # same translation a WatchPlan's explicit fields go through below), while remembering the
        # ORIGINAL path's registry owner for provenance — see _expand_explicit_path.
        situation_values: dict[str, Any] = {}
        situation_owner_by_path: dict[str, str] = {}
        for raw_path, raw_value in situation_raw.items():
            owner = _owner_layer_for_situation_path(raw_path)
            for epath, evalue in _expand_explicit_path(raw_path, raw_value).items():
                situation_values[epath] = evalue
                situation_owner_by_path[epath] = owner
        # An explicitly-requested situation resolution is, like an explicit source_profile, an
        # EXPLICIT selection for the loud-diagnostic rule (module docstring).
        situation_is_explicit = situation is not None

        domain_pack_values = domain_pack.values if domain_pack else {}
        execution_policy_values = execution_policy.values if execution_policy else {}
        optimization_lock_values = optimization_lock.values if optimization_lock else {}
        watch_plan_values = dict(watch_plan.explicit)

        # -- ownership gate: raised BEFORE any value is produced -----------------------------
        _check_ownership(domain_pack_values, layer="DomainPack", expected_owner=LAYER_DOMAIN_PACK)
        _check_ownership(
            execution_policy_values, layer="ExecutionPolicy", expected_owner=LAYER_EXECUTION_POLICY
        )
        _check_optimization_lock_ownership(optimization_lock_values)
        _check_ownership(watch_plan_values, layer="WatchPlan", expected_owner=LAYER_WATCH_PLAN)
        # source_profile_values / situation_values are produced by this module from trusted,
        # already-validated objects (a real SourceProfile, or resolve_situation's own pure
        # mapping) — not caller-authored path dicts — so they are not re-gated here.

        # AFTER ownership passes (against the ASSERTED path, e.g. "WatchProfile.salience_stat"),
        # expand to the runtime path it actually resolves (e.g. "Settings.edge.salience_stat") for
        # the merge below — see _expand_explicit_path.
        watch_plan_effective: dict[str, Any] = {}
        for path, value in watch_plan_values.items():
            watch_plan_effective.update(_expand_explicit_path(path, value))

        optimization_lock_hash: str | None = None
        warnings: list[FieldDiagnostic] = []
        if optimization_lock is not None:
            optimization_lock_hash = optimization_lock.content_hash
            if optimization_lock.locked_hash is not None and optimization_lock.locked_hash != optimization_lock_hash:
                warnings.append(FieldDiagnostic(
                    path=f"OptimizationLock.{optimization_lock.id}",
                    code="optimization_lock_hash_mismatch",
                    message=(
                        f"OptimizationLock {optimization_lock.id!r} claims locked_hash="
                        f"{optimization_lock.locked_hash!r} but its values hash to "
                        f"{optimization_lock_hash!r} now — the lock's validity claim is void "
                        "(ADR-008 Corrections(d)); its values still apply per precedence, but are "
                        "no longer a validated selection."
                    ),
                ))

        # -- merge, low to high; each layer's write to a path OVERWRITES a lower layer's ---------
        contributions: tuple[_LayerContribution, ...] = (
            _LayerContribution(LAYER_SDK, sdk_snapshot, "sdk_default", "registry default"),
            _LayerContribution(
                LAYER_SOURCE_PROFILE, source_profile_values,
                source_profile.name if source_profile else "source_profile",
                "source profile declaration",
            ),
            _LayerContribution(
                LAYER_SOURCE_PROFILE, situation_values,
                situation_source_id,
                "derived from goal-temporality/camera-mount/start-state situation axes",
            ),
            _LayerContribution(
                LAYER_SOURCE_PROFILE, source_profile_bindings,
                source_profile.name if source_profile else "source_profile",
                "source profile declared strategy bound to runtime fields (salience/coast, card C2)",
            ),
            _LayerContribution(
                LAYER_DOMAIN_PACK, domain_pack_values,
                domain_pack.id if domain_pack else "domain_pack", "domain pack declaration",
            ),
            _LayerContribution(
                LAYER_EXECUTION_POLICY, execution_policy_values,
                execution_policy.id if execution_policy else "execution_policy",
                "execution policy declaration",
            ),
            _LayerContribution(
                LAYER_OPTIMIZATION_LOCK, optimization_lock_values,
                optimization_lock.id if optimization_lock else "optimization_lock",
                "validated optimizer selection",
            ),
            _LayerContribution(
                LAYER_DEPLOYMENT, deployment_contribution, "env", "deployment/env override",
            ),
            _LayerContribution(
                LAYER_WATCH_PLAN, watch_plan_effective, "watch_plan.explicit", "explicit plan override",
            ),
        )

        effective: dict[str, Any] = {}
        provenance: dict[str, SettingOrigin] = {}
        explicit_profile_owned: set[str] = set()  # paths an EXPLICIT profile-like source touched

        for contribution in contributions:
            for path, value in contribution.values.items():
                # Situation-tier contributions carry their REAL registry owner (mostly WatchPlan/
                # Gate, not literally "SourceProfile") even though they are layered in at the
                # SourceProfile precedence slot above — see module docstring.
                owner_layer = (
                    situation_owner_by_path.get(path, LAYER_SOURCE_PROFILE)
                    if contribution.values is situation_values
                    else contribution.layer
                )
                if contribution.layer in (LAYER_SOURCE_PROFILE,) and (
                    (contribution.values is source_profile_values and source_profile_explicit)
                    or (contribution.values is source_profile_bindings and source_profile_explicit)
                    or (contribution.values is situation_values and situation_is_explicit)
                ):
                    explicit_profile_owned.add(path)

                if (
                    contribution.layer == LAYER_DEPLOYMENT
                    and path in explicit_profile_owned
                    and path in effective
                    and effective[path] != value
                ):
                    warnings.append(FieldDiagnostic(
                        path=path,
                        code="env_overrides_explicit_profile",
                        message=(
                            f"deployment/env value {value!r} for {path!r} overrides an "
                            f"explicitly-selected profile's value {effective[path]!r} "
                            f"({provenance[path].source_id})"
                        ),
                    ))

                env_var = _BY_PATH[path].env_var if path in _BY_PATH else None
                if contribution.layer == LAYER_DEPLOYMENT and env_var:
                    source_id = f"env:{env_var}"
                    reason = "env beats lower-precedence default"
                else:
                    source_id = contribution.source_id
                    reason = contribution.reason

                effective[path] = value
                provenance[path] = SettingOrigin(
                    value=value,
                    owner_layer=owner_layer,
                    source_id=source_id,
                    reason=reason,
                    validated_by=(
                        optimization_lock.id
                        if contribution.layer == LAYER_OPTIMIZATION_LOCK and optimization_lock is not None
                        else None
                    ),
                )

        plan_hash = content_hash(watch_plan.to_dict())
        component_snapshot_hash = content_hash(dict(sorted(effective.items(), key=lambda kv: kv[0])))

        resolved = ResolvedWatch(
            plan=watch_plan,
            source_profile=(source_profile_values if source_profile else None),
            domain_pack=(dict(domain_pack.values) if domain_pack else None),
            execution_policy=(dict(execution_policy.values) if execution_policy else None),
            components={},
            effective=effective,
            provenance=provenance,
            plan_hash=plan_hash,
            component_snapshot_hash=component_snapshot_hash,
            optimization_lock_hash=optimization_lock_hash,
        )

        status = "applied_with_warnings" if warnings else "applied"
        self.last_receipt = WatchReceipt(
            status=status,
            accepted_fields=tuple(sorted(effective)),
            warnings=tuple(warnings),
        )
        return resolved


__all__ = [
    "ConfigError",
    "PRECEDENCE",
    "LAYER_SDK",
    "LAYER_SOURCE_PROFILE",
    "LAYER_DOMAIN_PACK",
    "LAYER_EXECUTION_POLICY",
    "LAYER_OPTIMIZATION_LOCK",
    "LAYER_DEPLOYMENT",
    "LAYER_WATCH_PLAN",
    "LAYER_GATE",
    "DomainPack",
    "ExecutionPolicy",
    "OptimizationLock",
    "ConfigResolver",
]
