"""The tunables inventory (ADR-008) — every behavior-affecting knob, typed and owned.

ADR-008 (``.adr/ADR-008-configuration-ownership-resolution-provenance.md``) names six places a
behavior-affecting value can be set today (the ``Settings`` tree, ``watch()``/``Percept.create()``
kwargs, ``Goal.state`` keys, ``WatchProfile`` fields, ``SourceProfile`` fields, a detector's
``DetectorManifest``) and six OBJECTS the target configuration architecture will resolve them
through (``SourceProfile``, ``DomainPack``, ``ExecutionPolicy``, ``DeploymentConfig``,
``OptimizationLock``, ``WatchPlan``). This module is the accounting: every knob from the six existing
places, enumerated as a typed :class:`TunableSpec` and assigned an owner from the ADR's ownership
table (.adr/ADR-008-configuration-ownership-resolution-provenance.md#ownership-table-per-field-class).

Two owner values are not among the ADR's six declared objects, both deliberate:

- ``"Gate"`` — fields that tune :mod:`percept.gate` itself (thresholds, dwell, refractory, freshness).
  The ADR's Invariants section is explicit: "One Gate... no configuration object may carry gate
  semantics" — the same rule :data:`percept.sourceprofile.FORBIDDEN_FIELDS` already enforces
  structurally for ``SourceProfile``. A gate-tuning field is therefore not ownable by any of the six
  objects; it stays with the Gate.
- ``"unclassified"`` — a field that genuinely resists classification against the ownership table as
  written. ADR-008's own paradigm case was a "coverage floor" field (§Corrections(b), §Open questions
  #1) — resolved here to ``"SourceProfile"`` (a per-source-class default, the same object that
  already declares ``coast_interval_s``/``coast_required`` for the same "is this source legitimately
  quiet" question): ``SourceProfile.parameters["coverage_floor"]`` below now carries that owner.

This module performs no I/O at import (the same "no import-time semantic config" invariant ADR-008
holds every one of the six objects to): building :data:`REGISTRY` is pure data construction, and
nothing here reads an environment variable or touches the filesystem. Only :func:`main` (``python -m
percept.tunables``) writes ``tunables.json``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

#: Schema version of the *tunables.json* artifact itself (the shape of a serialized TunableSpec).
#: Bump on any change to :class:`TunableSpec`'s fields — independent of every other version ADR-008's
#: version ledger names (SDK package version, WATCH_VERSION, CONTRACT_VERSION, ...).
SCHEMA_VERSION = 1

#: The six objects ADR-008 declares, plus the two deliberate escape hatches described in the module
#: docstring. A :class:`TunableSpec` whose ``owner`` is not one of these is a bug in this registry.
OWNERS: tuple[str, ...] = (
    "SourceProfile",
    "DomainPack",
    "ExecutionPolicy",
    "DeploymentConfig",
    "OptimizationLock",
    "WatchPlan",
    "Gate",
    "unclassified",
)

VISIBILITIES: tuple[str, ...] = ("public", "internal", "advisory", "key", "descriptive")
RISKS: tuple[str, ...] = ("low", "medium", "high")
TYPES: tuple[str, ...] = (
    "float", "int", "bool", "str", "enum", "dict", "callable", "frozenset", "list", "object",
)


class TunableError(ValueError):
    """A :class:`TunableSpec` (or the registry as a whole) failed a shape/consistency check."""


@dataclass(frozen=True)
class TunableSpec:
    """One typed, owned knob.

    - ``path`` — a dotted, human-readable locator (e.g. ``"Settings.edge.motion_gate"``,
      ``"Goal.state.rhythm.channel"``). Not a live attribute lookup path (several sources — Goal.state,
      SourceProfile.parameters — are free-form dicts, not fixed attributes) — a stable IDENTIFIER a
      human or a resolver can cite.
    - ``owner`` — one of :data:`OWNERS`; which of ADR-008's six objects (or the two escape hatches)
      is allowed to set this field's class.
    - ``visibility`` — ``"public"`` (a documented, caller-facing knob), ``"internal"`` (wiring/testing
      seam, not a tuning surface), ``"advisory"`` (guidance nothing enforces), ``"key"`` (an identifying
      string, not a tuned value), or ``"descriptive"`` (self-describing registry metadata).
    - ``safe_to_optimize`` — may an :class:`~percept.tunables.TunableSpec`-covering ``OptimizationLock``
      ever include this field? ADR-008's Invariants: "The optimizer tunes only declared-safe paths...
      a lock covering any other field is invalid to construct." Conservative by construction: ``False``
      unless a field is a bounded numeric calibration knob with no gate/safety/credential involvement.
    - ``default`` — the shipped default value (``None`` when there isn't one, e.g. required kwargs).
    - ``value_type`` — one of :data:`TYPES`; the Python shape the resolved value takes.
    - ``bounds`` — ``(low, high)`` for a numeric field IF the source code enforces or clearly implies
      that range; ``None`` when no such range is enforced (documented as such in ``note``).
    - ``choices`` — the closed vocabulary for an enum-shaped field; ``None`` for open-ended fields.
    - ``env_var`` — the ``PERCEPT_*`` variable that overrides this field via ``load_settings()`` or a
      backend's own env read, if any.
    - ``requires_recompile`` — changing this value needs ``Percept.create()`` /
      ``RuntimeCompiler.compile()`` re-run to take effect (it is read once at core-construction time,
      not hot-reloaded mid-run).
    - ``requires_restart`` — changing this value needs the Python process itself restarted (it is read
      at import time or by an external/native layer this process cannot re-initialize).
    - ``risk`` — ``"low"`` / ``"medium"`` / ``"high"``: the blast radius of getting this field wrong.
    - ``note`` — one line of provenance/reasoning: why this owner, what module/line grounds it.
    """

    path: str
    owner: str
    visibility: str
    safe_to_optimize: bool
    default: Any = None
    value_type: str = "float"
    bounds: tuple[float, float] | None = None
    choices: tuple[str, ...] | None = None
    env_var: str | None = None
    requires_recompile: bool = False
    requires_restart: bool = False
    risk: str = "low"
    note: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise TunableError(f"TunableSpec.path must be a non-empty string, got {self.path!r}")
        if self.owner not in OWNERS:
            raise TunableError(f"{self.path}: owner must be one of {OWNERS}, got {self.owner!r}")
        if self.visibility not in VISIBILITIES:
            raise TunableError(
                f"{self.path}: visibility must be one of {VISIBILITIES}, got {self.visibility!r}"
            )
        if self.value_type not in TYPES:
            raise TunableError(f"{self.path}: value_type must be one of {TYPES}, got {self.value_type!r}")
        if self.risk not in RISKS:
            raise TunableError(f"{self.path}: risk must be one of {RISKS}, got {self.risk!r}")
        if self.bounds is not None:
            lo, hi = self.bounds
            if lo > hi:
                raise TunableError(f"{self.path}: bounds low > high ({self.bounds!r})")
        if self.safe_to_optimize and self.owner == "Gate":
            raise TunableError(
                f"{self.path}: a Gate-owned field cannot be safe_to_optimize=True "
                "(ADR-008 Invariants: 'One Gate... no configuration object may carry gate semantics')"
            )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["bounds"] = list(self.bounds) if self.bounds is not None else None
        d["choices"] = list(self.choices) if self.choices is not None else None
        return d


# ======================================================================================================
# Section A — the ``Settings`` tree (``percept.config``, 10 dataclass sections, ``load_settings()``).
# Read ONCE at ``Percept.create()`` time (config.py module docstring: "Read once via load_settings()")
# -> every Settings field ``requires_recompile=True``, never ``requires_restart`` (no import-time I/O).
# ======================================================================================================

_SETTINGS: tuple[TunableSpec, ...] = (
    # -- GateSettings — owner="Gate": ADR-008 Invariants wall gate semantics off from all six objects,
    # the same structural rule percept.sourceprofile.FORBIDDEN_FIELDS already enforces for SourceProfile.
    TunableSpec(
        path="Settings.gate.threshold", owner="Gate", visibility="public", safe_to_optimize=False,
        default=0.7, value_type="float", bounds=(0.0, 1.0), env_var="PERCEPT_GATE_THRESHOLD",
        requires_recompile=True, risk="high",
        note="confident-yes cutoff; the Gate is the sole firing authority (ADR-001, ADR-008 Invariants).",
    ),
    TunableSpec(
        path="Settings.gate.validity_window_s", owner="Gate", visibility="public", safe_to_optimize=False,
        default=90.0, value_type="float", env_var="PERCEPT_GATE_VALIDITY_S",
        requires_recompile=True, risk="medium", note="asserted-fact freshness window.",
    ),
    TunableSpec(
        path="Settings.gate.key_ttl_s", owner="Gate", visibility="public", safe_to_optimize=False,
        default=3600.0, value_type="float", env_var="PERCEPT_GATE_KEY_TTL_S",
        requires_recompile=True, risk="medium",
        note="R-1 growth bound: evict a gate key untouched this long (0/neg = off).",
    ),
    TunableSpec(
        path="Settings.gate.max_keys", owner="Gate", visibility="public", safe_to_optimize=False,
        default=0, value_type="int", bounds=(0.0, float("inf")), env_var="PERCEPT_GATE_MAX_KEYS",
        requires_recompile=True, risk="medium", note="hard LRU ceiling on gate keys (0 = no ceiling).",
    ),
    TunableSpec(
        path="Settings.gate.stale_ask", owner="Gate", visibility="public", safe_to_optimize=False,
        default=False, value_type="bool", env_var="PERCEPT_GATE_STALE_ASK",
        requires_recompile=True, risk="high",
        note="A-3b: folds perception recency into freshness; the ONE switch that changes WHAT FIRES.",
    ),
    TunableSpec(
        path="Settings.gate.require_false_first", owner="Gate", visibility="public", safe_to_optimize=False,
        default=False, value_type="bool", env_var="PERCEPT_GATE_REQUIRE_FALSE_FIRST",
        requires_recompile=True, risk="high",
        note="require a confident-false baseline before a key can fire (v0.1 hero matrix).",
    ),
    TunableSpec(
        path="Settings.gate.refractory_s", owner="Gate", visibility="public", safe_to_optimize=False,
        default=0.0, value_type="float", env_var="PERCEPT_GATE_REFRACTORY_S",
        requires_recompile=True, risk="high",
        note="Refractory re-fire suppression window (precision-tuning finding's RE-FIRES source).",
    ),
    # -- EntitySettings — split: iou_match is identity-strategy calibration (SourceProfile); max_entities
    # is a resource/eviction ceiling (ExecutionPolicy).
    TunableSpec(
        path="Settings.entity.iou_match", owner="SourceProfile", visibility="public", safe_to_optimize=True,
        default=0.5, value_type="float", bounds=(0.0, 1.0), env_var="PERCEPT_IOU_MATCH",
        requires_recompile=True, risk="medium",
        note="frame-to-frame continuity threshold; identity_strategy calibration (SourceProfile row).",
    ),
    TunableSpec(
        path="Settings.entity.max_entities", owner="ExecutionPolicy", visibility="public",
        safe_to_optimize=False, default=64, value_type="int", bounds=(1.0, float("inf")),
        env_var="PERCEPT_MAX_ENTITIES", requires_recompile=True, risk="low",
        note="eviction bound; resource ceiling (ExecutionPolicy row: 'concurrency ceiling'-shaped).",
    ),
    # -- CacheSettings — cost/perf cascade tuning -> ExecutionPolicy (resource/scheduling posture).
    TunableSpec(
        path="Settings.cache.enabled", owner="ExecutionPolicy", visibility="public", safe_to_optimize=False,
        default=False, value_type="bool", env_var="PERCEPT_CACHE_ENABLED", requires_recompile=True,
        risk="medium", note="wires the FrameHashLayer cost-cascade short-circuit into the pipeline.",
    ),
    TunableSpec(
        path="Settings.cache.threshold", owner="ExecutionPolicy", visibility="public", safe_to_optimize=True,
        default=0.0, value_type="float", bounds=(0.0, 1.0), env_var="PERCEPT_CACHE_THRESHOLD",
        requires_recompile=True, risk="medium", note="cascade short-circuit cutoff.",
    ),
    TunableSpec(
        path="Settings.cache.ttl_s", owner="ExecutionPolicy", visibility="public", safe_to_optimize=True,
        default=8.0, value_type="float", env_var="PERCEPT_CACHE_TTL_S", requires_recompile=True,
        risk="low", note="hard ceiling on slow drift before a cache HIT is distrusted.",
    ),
    TunableSpec(
        path="Settings.cache.tau_bits", owner="ExecutionPolicy", visibility="public", safe_to_optimize=True,
        default=6, value_type="int", bounds=(0.0, 64.0), env_var="PERCEPT_CACHE_TAU_BITS",
        requires_recompile=True, risk="low", note="dHash Hamming near-match radius.",
    ),
    TunableSpec(
        path="Settings.cache.audit_rate", owner="ExecutionPolicy", visibility="public", safe_to_optimize=True,
        default=0.05, value_type="float", bounds=(0.0, 1.0), env_var="PERCEPT_CACHE_AUDIT_RATE",
        requires_recompile=True, risk="low", note="fraction of cache HITs re-perceived for audit.",
    ),
    # -- ModelSettings — provider/model selection -> DeploymentConfig (ownership table's own example row;
    # Invariants: "Model id != adapter identity... DeploymentConfig may own model-id fields").
    TunableSpec(
        path="Settings.models.vision", owner="DeploymentConfig", visibility="public", safe_to_optimize=False,
        default=None, value_type="str", env_var="PERCEPT_VISION_BACKEND", requires_recompile=True,
        risk="high", note="provider NAME (never a model id — percept.provider.ProviderError guards this).",
    ),
    TunableSpec(
        path="Settings.models.conductor", owner="DeploymentConfig", visibility="public",
        safe_to_optimize=False, default=None, value_type="str", env_var="PERCEPT_CONDUCTOR_BACKEND",
        requires_recompile=True, risk="medium", note="provider NAME for the conductor seam.",
    ),
    TunableSpec(
        path="Settings.models.specialist", owner="DeploymentConfig", visibility="public",
        safe_to_optimize=False, default=None, value_type="str", env_var="PERCEPT_SPECIALIST_BACKEND",
        requires_recompile=True, risk="medium", note="provider NAME for the specialist seam.",
    ),
    TunableSpec(
        path="Settings.models.stt", owner="DeploymentConfig", visibility="public", safe_to_optimize=False,
        default="fake", value_type="str", env_var="PERCEPT_STT_BACKEND", requires_recompile=True,
        risk="medium", note="speech-to-text provider NAME.",
    ),
    TunableSpec(
        path="Settings.models.tts", owner="DeploymentConfig", visibility="public", safe_to_optimize=False,
        default="fake", value_type="str", env_var="PERCEPT_TTS_BACKEND", requires_recompile=True,
        risk="medium", note="text-to-speech provider NAME.",
    ),
    TunableSpec(
        path="Settings.models.vision_model", owner="DeploymentConfig", visibility="public",
        safe_to_optimize=False, default="claude-opus-4-8", value_type="str", env_var="PERCEPT_VISION_MODEL",
        requires_recompile=True, risk="medium",
        note="model id CARRIED to the vision adapter, never a selector (Invariants).",
    ),
    TunableSpec(
        path="Settings.models.conductor_model", owner="DeploymentConfig", visibility="public",
        safe_to_optimize=False, default="claude-haiku-4-5", value_type="str",
        env_var="PERCEPT_CONDUCTOR_MODEL", requires_recompile=True, risk="low",
        note="model id carried to the conductor adapter.",
    ),
    TunableSpec(
        path="Settings.models.specialist_model", owner="DeploymentConfig", visibility="public",
        safe_to_optimize=False, default="claude-haiku-4-5", value_type="str",
        env_var="PERCEPT_SPECIALIST_MODEL", requires_recompile=True, risk="low",
        note="model id carried to the specialist adapter.",
    ),
    # -- RoutingSettings — local-router defer-on-miss cutoff: a cost/quality tradeoff -> ExecutionPolicy.
    TunableSpec(
        path="Settings.routing.confidence_threshold", owner="ExecutionPolicy", visibility="public",
        safe_to_optimize=True, default=0.6, value_type="float", bounds=(0.0, 1.0),
        env_var="PERCEPT_ROUTE_THRESHOLD", requires_recompile=True, risk="low",
        note="below this, the local router defers to the conductor (defer-on-miss is structural).",
    ),
    # -- IdentitySettings — re-id calibration -> SourceProfile (identity_strategy, ownership table row).
    TunableSpec(
        path="Settings.identity.appearance_threshold", owner="SourceProfile", visibility="public",
        safe_to_optimize=True, default=0.8, value_type="float", bounds=(0.0, 1.0),
        env_var="PERCEPT_APPEARANCE_MATCH", requires_recompile=True, risk="medium",
        note="cosine sim to re-id across a gap; identity_strategy calibration.",
    ),
    # -- EpisodeSettings — split: window_s is episode-segmentation shape (SourceProfile, per its module
    # docstring's "episode segmentation and sampling shape, carried in parameters"); max_episodes is a
    # resource eviction ceiling (ExecutionPolicy).
    TunableSpec(
        path="Settings.episodes.window_s", owner="SourceProfile", visibility="public",
        safe_to_optimize=True, default=3600.0, value_type="float", env_var="PERCEPT_EPISODE_WINDOW_S",
        requires_recompile=True, risk="medium", note="counting window ('today's cups'); episode shape.",
    ),
    TunableSpec(
        path="Settings.episodes.max_episodes", owner="ExecutionPolicy", visibility="public",
        safe_to_optimize=False, default=64, value_type="int", bounds=(1.0, float("inf")),
        env_var="PERCEPT_MAX_EPISODES", requires_recompile=True, risk="low",
        note="bound on the per-goal episode log; resource ceiling.",
    ),
    # -- EventSettings — split: window/merge_gap_s/audio_window_s are source-stream sampling shape
    # (SourceProfile); preroll_max_bytes is an explicit memory ceiling (ExecutionPolicy).
    TunableSpec(
        path="Settings.events.window", owner="SourceProfile", visibility="public", safe_to_optimize=True,
        default=12, value_type="int", bounds=(1.0, float("inf")), env_var="PERCEPT_FRAME_WINDOW",
        requires_recompile=True, risk="low", note="rolling frames per source for temporal perception.",
    ),
    TunableSpec(
        path="Settings.events.preroll_max_bytes", owner="ExecutionPolicy", visibility="public",
        safe_to_optimize=False, default=134_217_728, value_type="int", bounds=(0.0, float("inf")),
        env_var="PERCEPT_PREROLL_MAX_BYTES", requires_recompile=True, risk="medium",
        note="byte budget for the incident pre-roll ring; the measured 'runaway memory' fix.",
    ),
    TunableSpec(
        path="Settings.events.merge_gap_s", owner="SourceProfile", visibility="public",
        safe_to_optimize=True, default=1.5, value_type="float", env_var="PERCEPT_EVENT_MERGE_GAP_S",
        requires_recompile=True, risk="low", note="same-type detections within this = 1 event.",
    ),
    TunableSpec(
        path="Settings.events.audio_window_s", owner="SourceProfile", visibility="public",
        safe_to_optimize=True, default=4.0, value_type="float", env_var="PERCEPT_AUDIO_WINDOW_S",
        requires_recompile=True, risk="low", note="rolling seconds of audio per source.",
    ),
    # -- ConsentSettings — deployment-wide actuation/consent posture (ADR-007) -> DeploymentConfig.
    TunableSpec(
        path="Settings.consent.allow_computer_use", owner="DeploymentConfig", visibility="public",
        safe_to_optimize=False, default=False, value_type="bool", env_var="PERCEPT_ALLOW_COMPUTER_USE",
        requires_recompile=True, risk="high", note="gates the `act` modality; deployment permission.",
    ),
    TunableSpec(
        path="Settings.consent.allowed_actions", owner="DeploymentConfig", visibility="public",
        safe_to_optimize=False, default=frozenset(), value_type="frozenset",
        env_var="PERCEPT_ACT_ALLOWED_ACTIONS", requires_recompile=True, risk="high",
        note="comma-separated action scope for the `act` modality.",
    ),
    TunableSpec(
        path="Settings.consent.actuation", owner="DeploymentConfig", visibility="public",
        safe_to_optimize=False, default="none", value_type="enum",
        choices=("none", "screen", "physical"), env_var="PERCEPT_ACTUATION", requires_recompile=True,
        risk="high",
        note="'physical' is a DECLARATION every act is refused (ADR-007; the gate cannot ask a loop).",
    ),
    # -- EdgeSettings — tier-0 salience admission tuning -> SourceProfile ("admission_strategy" row;
    # EdgeSettings' own docstring: tracks the camera mount, the deployment-wide admission-signal default).
    TunableSpec(
        path="Settings.edge.motion_gate", owner="SourceProfile", visibility="public", safe_to_optimize=True,
        default=0.10, value_type="float", bounds=(0.0, 1.0), env_var="PERCEPT_MOTION_GATE",
        requires_recompile=True, risk="medium", note="mean-stat motion escalation cutoff.",
    ),
    TunableSpec(
        path="Settings.edge.salience_stat", owner="SourceProfile", visibility="public",
        safe_to_optimize=False, default="mean", value_type="enum", choices=("mean", "block_max", "p95"),
        env_var="PERCEPT_SALIENCE_STAT", requires_recompile=True, risk="medium",
        note="deployment-wide default localizing statistic (empirical tuning: block_max ~doubled recall).",
    ),
    TunableSpec(
        path="Settings.edge.ego_compensate", owner="SourceProfile", visibility="public",
        safe_to_optimize=False, default=False, value_type="bool", env_var="PERCEPT_EGO_COMPENSATE",
        requires_recompile=True, risk="medium", note="remove global camera motion before diffing.",
    ),
    TunableSpec(
        path="Settings.edge.motion_gate_block_max", owner="SourceProfile", visibility="public",
        safe_to_optimize=True, default=0.20, value_type="float", bounds=(0.0, 1.0),
        env_var="PERCEPT_MOTION_GATE_BLOCK_MAX", requires_recompile=True, risk="medium",
        note="block_max-scale escalation cutoff (calibrated: 0.20 -> 0.85 recall).",
    ),
    TunableSpec(
        path="Settings.edge.motion_gate_p95", owner="SourceProfile", visibility="public",
        safe_to_optimize=True, default=0.15, value_type="float", bounds=(0.0, 1.0),
        env_var="PERCEPT_MOTION_GATE_P95", requires_recompile=True, risk="medium",
        note="p95-scale escalation cutoff (calibrated: 0.15 -> 0.79 recall).",
    ),
    # coast_enabled is a scheduling/cost posture, not an admission-signal calibration -> ExecutionPolicy:
    # whether a SourceProfile-declared coast floor is bound into the runtime at all. The strict_admission
    # policy sets it False for signal-only admission (percept.execution_policies_default).
    TunableSpec(
        path="Settings.edge.coast_enabled", owner="ExecutionPolicy", visibility="public",
        safe_to_optimize=False, default=True, value_type="bool", env_var="PERCEPT_COAST_ENABLED",
        requires_recompile=True, risk="medium",
        note="whether a SourceProfile-declared coast floor is bound into the runtime; strict deployments "
        "disable via ExecutionPolicy.",
    ),
)

# ======================================================================================================
# Section B — ``watch()`` / ``Percept.create()`` keyword parameters.
# ======================================================================================================

_RUN_PARAMETERS: tuple[TunableSpec, ...] = (
    # -- watch() (percept/watch.py) — source=/profile= are literally the WatchPlan row's own example
    # ("the source=/profile= selection a caller passes to watch()").
    TunableSpec(
        path="watch.source", owner="WatchPlan", visibility="public", safe_to_optimize=False,
        default="camera", value_type="str", risk="low",
        note="frame source key; WatchPlan ownership row's own example.",
    ),
    TunableSpec(
        path="watch.profile", owner="WatchPlan", visibility="public", safe_to_optimize=False,
        default=None, value_type="object", risk="medium",
        note="the WatchProfile/Settings selection a caller passes to watch(); WatchPlan's own example.",
    ),
    TunableSpec(
        path="watch.percept", owner="unclassified", visibility="internal", safe_to_optimize=False,
        default=None, value_type="object", risk="low",
        note="pre-built Percept core reuse; a wiring seam, not a configuration value.",
    ),
    # -- Percept.create() (percept/agent.py) — backend selection mirrors ModelSettings' ownership.
    TunableSpec(
        path="Percept.create.vision", owner="DeploymentConfig", visibility="public", safe_to_optimize=False,
        default=None, value_type="str", requires_recompile=True, risk="high",
        note="explicit-kwarg mirror of Settings.models.vision; same DeploymentConfig ownership.",
    ),
    TunableSpec(
        path="Percept.create.specialist", owner="DeploymentConfig", visibility="public",
        safe_to_optimize=False, default=None, value_type="str", requires_recompile=True, risk="medium",
        note="explicit-kwarg mirror of Settings.models.specialist.",
    ),
    TunableSpec(
        path="Percept.create.conductor", owner="DeploymentConfig", visibility="public",
        safe_to_optimize=False, default=None, value_type="str", requires_recompile=True, risk="medium",
        note="explicit-kwarg mirror of Settings.models.conductor.",
    ),
    TunableSpec(
        path="Percept.create.stt", owner="DeploymentConfig", visibility="public", safe_to_optimize=False,
        default=None, value_type="str", requires_recompile=True, risk="medium",
        note="explicit-kwarg mirror of Settings.models.stt.",
    ),
    TunableSpec(
        path="Percept.create.tts", owner="DeploymentConfig", visibility="public", safe_to_optimize=False,
        default=None, value_type="str", requires_recompile=True, risk="medium",
        note="explicit-kwarg mirror of Settings.models.tts.",
    ),
    TunableSpec(
        path="Percept.create.settings", owner="WatchPlan", visibility="public", safe_to_optimize=False,
        default=None, value_type="object", requires_recompile=True, risk="high",
        note="an explicit Settings override — ADR-008's own silent-last-writer-wins motivating example.",
    ),
    TunableSpec(
        path="Percept.create.clock", owner="unclassified", visibility="internal", safe_to_optimize=False,
        default=None, value_type="callable", requires_recompile=True, risk="low",
        note="test/infra clock injection seam, not a tuning knob.",
    ),
    TunableSpec(
        path="Percept.create.discover_plugins", owner="DeploymentConfig", visibility="public",
        safe_to_optimize=False, default=True, value_type="bool", requires_recompile=True, risk="low",
        note="entry-point plugin discovery; deployment posture.",
    ),
    TunableSpec(
        path="Percept.create.audit", owner="unclassified", visibility="internal", safe_to_optimize=False,
        default=None, value_type="callable", requires_recompile=True, risk="low",
        note="audit hook; wiring seam.",
    ),
    TunableSpec(
        path="Percept.create.identity", owner="DeploymentConfig", visibility="public",
        safe_to_optimize=False, default=None, value_type="object", requires_recompile=True, risk="medium",
        note="identity backend/provider wiring.",
    ),
    TunableSpec(
        path="Percept.create.embedder", owner="DeploymentConfig", visibility="public",
        safe_to_optimize=False, default=None, value_type="callable", requires_recompile=True, risk="medium",
        note="embedding provider wiring.",
    ),
    TunableSpec(
        path="Percept.create.on_trace", owner="unclassified", visibility="internal", safe_to_optimize=False,
        default=None, value_type="callable", requires_recompile=True, risk="low",
        note="observability hook; not an ownership-table concern.",
    ),
    TunableSpec(
        path="Percept.create.on_health", owner="unclassified", visibility="internal", safe_to_optimize=False,
        default=None, value_type="callable", requires_recompile=True, risk="low",
        note="observability hook; not an ownership-table concern.",
    ),
    TunableSpec(
        path="Percept.create.mode", owner="DeploymentConfig", visibility="public", safe_to_optimize=False,
        default=None, value_type="enum", choices=("offline", "live"), env_var="PERCEPT_MODE",
        requires_recompile=True, risk="high",
        note="offline/live: selects scripted vs real backends; deployment/environment posture.",
    ),
    TunableSpec(
        path="Percept.create.allow_scripted_brain", owner="DeploymentConfig", visibility="public",
        safe_to_optimize=False, default=False, value_type="bool", requires_recompile=True, risk="high",
        note="safety guard for a LIVE run with a scripted conductor.",
    ),
    TunableSpec(
        path="Percept.create.sense_policy", owner="WatchPlan", visibility="public", safe_to_optimize=False,
        default=None, value_type="object", requires_recompile=True, risk="medium",
        note="explicit cadence override, bypassing WatchProfile resolution; explicit user intent.",
    ),
    TunableSpec(
        path="Percept.create.plan_mode", owner="WatchPlan", visibility="public", safe_to_optimize=False,
        default=None, value_type="enum", choices=("single", "strip", "window", "sprite"),
        requires_recompile=True, risk="medium",
        note="explicit temporal-shape override (Corrections(a): must never default to 'sprite').",
    ),
)

# ======================================================================================================
# Section C — ``Goal.state`` keys (percept/behavior.py: coerce_authored_behavior/rhythm/episode/sense/
# perception, and the plain program_of() top-level reads). ALL owner=WatchPlan: goal.state is literally
# "the user's stated intent" (ADR-008 WatchPlan row: "owns: ... explicit per-field overrides").
# Every authored value here is a conductor/author-supplied override, safe_to_optimize=False across the
# board — an optimizer tuning a client's stated goal condition/consequence is out of scope by
# construction, not merely undecided.
# ======================================================================================================

_GOAL_STATE: tuple[TunableSpec, ...] = (
    TunableSpec(
        path="Goal.state.type", owner="WatchPlan", visibility="public", safe_to_optimize=False,
        default="watch", value_type="enum", choices=("watch", "transition", "count", "rhythm"),
        risk="medium", note="which BehaviorProgram sense program_of() compiles to (behavior.py).",
    ),
    TunableSpec(
        path="Goal.state.rearm", owner="WatchPlan", visibility="public", safe_to_optimize=False,
        default="on_false", value_type="enum",
        choices=("on_false", "never", "interval", "deadline", "each_event"), risk="medium",
        note="re-arm policy after a fire.",
    ),
    TunableSpec(
        path="Goal.state.cadence_s", owner="WatchPlan", visibility="public", safe_to_optimize=False,
        default=None, value_type="float", bounds=(1.0, float("inf")), risk="low",
        note="interval-rearm period; clamped to _MIN_CADENCE_S=1.0 so it can't busy-loop.",
    ),
    TunableSpec(
        path="Goal.state.deadline_after_s", owner="WatchPlan", visibility="public", safe_to_optimize=False,
        default=None, value_type="float", bounds=(0.0, float("inf")), risk="low",
        note="deadline-rearm DURATION from arm time (never an absolute stamp — A-2).",
    ),
    TunableSpec(
        path="Goal.state.fire_when", owner="WatchPlan", visibility="public", safe_to_optimize=False,
        default=None, value_type="str", risk="high",
        note="nth-distinct predicate, AST-whitelisted (compile_fire_when); authored, never optimized.",
    ),
    TunableSpec(
        path="Goal.state.window_s", owner="WatchPlan", visibility="public", safe_to_optimize=False,
        default=None, value_type="float", bounds=(0.0, float("inf")), risk="low",
        note="accumulate window for a fire_when count.",
    ),
    TunableSpec(
        path="Goal.state.miss_cost", owner="WatchPlan", visibility="public", safe_to_optimize=False,
        default=None, value_type="enum", choices=("trivial", "mild", "severe"), risk="high",
        note="consequence declaration; closed vocab, never optimizer-authored.",
    ),
    TunableSpec(
        path="Goal.state.false_fire_cost", owner="WatchPlan", visibility="public", safe_to_optimize=False,
        default=None, value_type="enum", choices=("trivial", "mild", "severe", "irreversible"), risk="high",
        note="consequence declaration; closed vocab, never optimizer-authored.",
    ),
    TunableSpec(
        path="Goal.state.protected_floor", owner="WatchPlan", visibility="public", safe_to_optimize=False,
        default=None, value_type="bool", risk="high",
        note="can only be raised true, never authored false-down.",
    ),
    TunableSpec(
        path="Goal.state.hold_s", owner="WatchPlan", visibility="public", safe_to_optimize=False,
        default=None, value_type="float", bounds=(0.0, float("inf")), risk="medium",
        note="R-1 sustained-duration dwell before firing.",
    ),
    TunableSpec(
        path="Goal.state.from", owner="WatchPlan", visibility="public", safe_to_optimize=False,
        default=None, value_type="str", risk="low", note="transition-type from-predicate (plain language).",
    ),
    TunableSpec(
        path="Goal.state.to", owner="WatchPlan", visibility="public", safe_to_optimize=False,
        default=None, value_type="str", risk="low", note="transition-type to-predicate (plain language).",
    ),
    # -- goal.state["rhythm"] (coerce_authored_rhythm)
    TunableSpec(
        path="Goal.state.rhythm.channel", owner="WatchPlan", visibility="public", safe_to_optimize=False,
        default=None, value_type="str", risk="low", note="channel hint biasing RhythmTracker selection.",
    ),
    TunableSpec(
        path="Goal.state.rhythm.cadence_s", owner="WatchPlan", visibility="public", safe_to_optimize=False,
        default=None, value_type="float", bounds=(0.0, 30.0), risk="low",
        note="rough expected rep period, clamped to _MAX_WINDOW_S=30.0.",
    ),
    TunableSpec(
        path="Goal.state.rhythm.whole_body", owner="WatchPlan", visibility="public", safe_to_optimize=False,
        default=None, value_type="bool", risk="low", note="flags a multi-limb action.",
    ),
    TunableSpec(
        path="Goal.state.rhythm.mode", owner="WatchPlan", visibility="public", safe_to_optimize=False,
        default=None, value_type="enum", choices=("discrete", "continuous"), risk="low",
        note="rep regime; continuous is a recorded research seam, not yet counted.",
    ),
    TunableSpec(
        path="Goal.state.rhythm.window_s", owner="WatchPlan", visibility="public", safe_to_optimize=False,
        default=None, value_type="float", bounds=(0.0, 30.0), risk="low",
        note="endurance window for windowed rep counts.",
    ),
    TunableSpec(
        path="Goal.state.rhythm.expected_rom", owner="WatchPlan", visibility="public",
        safe_to_optimize=False, default=None, value_type="float", bounds=(0.0, 100.0), risk="low",
        note="expected range-of-motion, channel-native units; only honoured with a named channel.",
    ),
    TunableSpec(
        path="Goal.state.rhythm.expected_rom_space", owner="WatchPlan", visibility="public",
        safe_to_optimize=False, default="channel_native", value_type="enum",
        choices=("channel_native",), risk="low",
        note="unit-contract tag; only 'channel_native' is honoured today.",
    ),
    TunableSpec(
        path="Goal.state.rhythm.amp_gate_frac", owner="WatchPlan", visibility="public",
        safe_to_optimize=False, default=None, value_type="float", bounds=(0.0, 1.0), risk="low",
        note="fraction of expected_rom a rep must reach to count.",
    ),
    TunableSpec(
        path="Goal.state.rhythm.smooth_s", owner="WatchPlan", visibility="public", safe_to_optimize=False,
        default=None, value_type="float", bounds=(0.0, 1.0), risk="low",
        note="opt-in causal-smoothing span; rate-invariance knob.",
    ),
    TunableSpec(
        path="Goal.state.rhythm.canonical_select", owner="WatchPlan", visibility="public",
        safe_to_optimize=False, default=False, value_type="bool", risk="low",
        note="opt-in canonical-grid channel scoring vs legacy medoid.",
    ),
    # -- goal.state["episode"] (coerce_authored_episode)
    TunableSpec(
        path="Goal.state.episode.policy", owner="WatchPlan", visibility="public", safe_to_optimize=False,
        default=None, value_type="enum", risk="medium",
        note="must name a shipped percept.episodes.EPISODE_POLICIES preset.",
    ),
    TunableSpec(
        path="Goal.state.episode.continuity_gap_s", owner="WatchPlan", visibility="public",
        safe_to_optimize=False, default=None, value_type="float", bounds=(0.0, 86_400.0), risk="low",
        note="clamped under a day; beyond that is not a 'gap'.",
    ),
    TunableSpec(
        path="Goal.state.episode.allow_reopen", owner="WatchPlan", visibility="public",
        safe_to_optimize=False, default=None, value_type="bool", risk="low",
        note="whether a closed episode may reopen.",
    ),
    TunableSpec(
        path="Goal.state.episode.terminal_statuses", owner="WatchPlan", visibility="public",
        safe_to_optimize=False, default=None, value_type="list", risk="low",
        note="bounded set (<=16, <=40 chars each) of statuses that close an episode.",
    ),
    # -- goal.state["sense"] (coerce_authored_sense)
    TunableSpec(
        path="Goal.state.sense.base_fps", owner="WatchPlan", visibility="public", safe_to_optimize=False,
        default=None, value_type="float", bounds=(0.0, 60.0), risk="low",
        note="per-concern quiet-floor fps; explicit author override of the fovea's cadence.",
    ),
    TunableSpec(
        path="Goal.state.sense.max_fps", owner="WatchPlan", visibility="public", safe_to_optimize=False,
        default=None, value_type="float", bounds=(0.0, 60.0), risk="low",
        note="per-concern motion-escalated ceiling fps; never inverted below base_fps.",
    ),
    TunableSpec(
        path="Goal.state.sense.escalate_threshold", owner="WatchPlan", visibility="public",
        safe_to_optimize=False, default=None, value_type="float", bounds=(0.0, 1.0), risk="low",
        note="motion level that triggers the per-concern saccade to max_fps.",
    ),
    # -- goal.state["perception"] (coerce_authored_perception)
    TunableSpec(
        path="Goal.state.perception.modality", owner="WatchPlan", visibility="public",
        safe_to_optimize=False, default=None, value_type="enum",
        choices=("vision", "audio", "fused"), risk="medium", note="which channel the judge reads.",
    ),
    TunableSpec(
        path="Goal.state.perception.temporal", owner="WatchPlan", visibility="public",
        safe_to_optimize=False, default=None, value_type="enum",
        choices=("single", "strip", "window", "sprite"), risk="medium",
        note="evidence-gathering shape; Corrections(a): screen/document defaults must be 'window', "
        "never 'sprite' (percept.agent._normalize_plan_mode normalizes a stale 'sprite' with a warning).",
    ),
    TunableSpec(
        path="Goal.state.perception.roi", owner="WatchPlan", visibility="public", safe_to_optimize=False,
        default=None, value_type="enum", choices=("focus", "hint", "full"), risk="low",
        note="region-of-interest framing for the judge.",
    ),
    TunableSpec(
        path="Goal.state.perception.sampling", owner="WatchPlan", visibility="public",
        safe_to_optimize=False, default=None, value_type="enum", choices=("recent", "uniform"), risk="low",
        note="which frames a strip/window samples.",
    ),
    TunableSpec(
        path="Goal.state.perception.n_frames", owner="WatchPlan", visibility="public",
        safe_to_optimize=False, default=None, value_type="int", bounds=(1.0, 6.0), risk="low",
        note="strip frame count, clamped to _MAX_STRIP=6 (token-bomb ceiling).",
    ),
    TunableSpec(
        path="Goal.state.perception.audio_s", owner="WatchPlan", visibility="public",
        safe_to_optimize=False, default=None, value_type="float", bounds=(0.0, 5.0), risk="low",
        note="authored audio clip length, clamped to _MAX_AUDIO_S=5.0.",
    ),
    TunableSpec(
        path="Goal.state.perception.window_s", owner="WatchPlan", visibility="public",
        safe_to_optimize=False, default=None, value_type="float", bounds=(0.0, 30.0), risk="low",
        note="clip window seconds, clamped to _MAX_WINDOW_S=30.0; implies temporal='window'.",
    ),
    # -- goal.state["identity_policy"] (percept/identity_policy.py, STATE_KEY) — a concern-level
    # override of SourceProfile.identity_strategy; explicit per-goal author intent (WatchPlan row's
    # own "explicit per-field overrides").
    TunableSpec(
        path="Goal.state.identity_policy", owner="WatchPlan", visibility="public",
        safe_to_optimize=False, default=None, value_type="str", risk="medium",
        note="percept.identity_policy.identity_policy_of; overrides SourceProfile.identity_strategy "
        "for this concern only (percept.identity_policy.select_identity_policy).",
    ),
    # -- goal.state["judge_mode"] (percept/episode_memory.py, episode_memory_enabled) — opt-in flag
    # selecting the episode-memory judge mode; a future WatchProfile-driven seam, author-set today.
    TunableSpec(
        path="Goal.state.judge_mode", owner="WatchPlan", visibility="public", safe_to_optimize=False,
        default=None, value_type="enum", choices=("episode_memory",), risk="medium",
        note="percept.episode_memory.episode_memory_enabled; additive opt-in, no default-path change.",
    ),
    # -- goal.state["delivery"] (percept/incidents.py: RequestedDelivery/parse_requested_delivery) —
    # the legacy AUTHORED delivery-intent dict, clamped by AttentionPolicy before it becomes effective.
    TunableSpec(
        path="Goal.state.delivery.preempt", owner="WatchPlan", visibility="public",
        safe_to_optimize=False, default="queue", value_type="enum",
        choices=("interrupt", "queue", "drop"), risk="medium",
        note="requested attention priority; AttentionPolicy CLAMPS 'interrupt' absent earned authority "
        "(percept.incidents.AttentionPolicy.clamp) — the request alone never grants it.",
    ),
    TunableSpec(
        path="Goal.state.delivery.ttl_s", owner="WatchPlan", visibility="public", safe_to_optimize=False,
        default=None, value_type="float", risk="low", note="drop-if-stale bound on a queued delivery.",
    ),
    TunableSpec(
        path="Goal.state.delivery.verbatim", owner="WatchPlan", visibility="public",
        safe_to_optimize=False, default=False, value_type="bool", risk="low",
        note="speak the authored text verbatim vs. paraphrase.",
    ),
    TunableSpec(
        path="Goal.state.delivery.preferred_channel", owner="WatchPlan", visibility="public",
        safe_to_optimize=False, default=None, value_type="str", risk="low",
        note="additive app hint, absent in the legacy dict -> None.",
    ),
)

# ======================================================================================================
# Section D — ``WatchProfile`` fields (percept/watchprofile.py). Per the module's own docstring:
# "It invents NO new knobs: every field is a typed handle onto a setting that already exists" — so each
# field's owner tracks the underlying knob it is a handle onto, distinguishing a DEFAULT (owned by the
# object that sets the underlying Settings-tree field) from an EXPLICIT per-context override (WatchPlan,
# per ADR-008's own "explicit salience_stat='block_max' override" example).
# ======================================================================================================

_WATCH_PROFILE: tuple[TunableSpec, ...] = (
    TunableSpec(
        path="WatchProfile.name", owner="WatchPlan", visibility="key", safe_to_optimize=False,
        default=None, value_type="str", risk="low", note="registry identity, not a tuned value.",
    ),
    TunableSpec(
        path="WatchProfile.environment", owner="WatchPlan", visibility="key", safe_to_optimize=False,
        default=None, value_type="str", risk="low",
        note="registry key segment (free-text, e.g. 'fixed_camera'/'handheld'/'wearable').",
    ),
    TunableSpec(
        path="WatchProfile.goal_type", owner="WatchPlan", visibility="key", safe_to_optimize=False,
        default=None, value_type="str", risk="low",
        note="registry key segment (free-text, e.g. 'stative'/'dynamic'/'periodic').",
    ),
    TunableSpec(
        path="WatchProfile.state", owner="WatchPlan", visibility="key", safe_to_optimize=False,
        default=None, value_type="str", risk="low",
        note="registry key segment (free-text, e.g. 'start_anchored'/'transition').",
    ),
    TunableSpec(
        path="WatchProfile.salience_stat", owner="WatchPlan", visibility="public", safe_to_optimize=False,
        default="mean", value_type="enum", choices=("mean", "block_max", "p95"), risk="medium",
        note="explicit per-context override of Settings.edge.salience_stat; the ADR's own "
        "'explicit salience_stat=\"block_max\" override' WatchPlan example.",
    ),
    TunableSpec(
        path="WatchProfile.stat_thresholds", owner="WatchPlan", visibility="public", safe_to_optimize=False,
        default=None, value_type="dict", risk="medium",
        note="per-stat escalation cutoffs (0..1 each); explicit companion to salience_stat.",
    ),
    TunableSpec(
        path="WatchProfile.plan_mode", owner="WatchPlan", visibility="public", safe_to_optimize=False,
        default="single", value_type="enum", choices=("single", "strip", "window", "sprite"), risk="medium",
        note="explicit per-context evidence-gathering shape; Corrections(a) 'sprite' deprecation applies.",
    ),
    TunableSpec(
        path="WatchProfile.cadence.base_fps", owner="ExecutionPolicy", visibility="public",
        safe_to_optimize=True, default=4.0, value_type="float", bounds=(0.0, float("inf")), risk="low",
        note="fovea quiet floor; ExecutionPolicy row's own example ('cadence ceilings').",
    ),
    TunableSpec(
        path="WatchProfile.cadence.max_fps", owner="ExecutionPolicy", visibility="public",
        safe_to_optimize=True, default=15.0, value_type="float", bounds=(0.0, float("inf")), risk="low",
        note="fovea cost ceiling; must be >= base_fps (CadenceProfile.__post_init__).",
    ),
    TunableSpec(
        path="WatchProfile.cadence.escalate_threshold", owner="ExecutionPolicy", visibility="public",
        safe_to_optimize=True, default=0.15, value_type="float", bounds=(0.0, 1.0), risk="low",
        note="motion level that triggers a saccade toward max_fps.",
    ),
    TunableSpec(
        path="WatchProfile.cadence.hold_s", owner="ExecutionPolicy", visibility="public",
        safe_to_optimize=True, default=0.5, value_type="float", bounds=(0.0, float("inf")), risk="low",
        note="keeps an escalation alive briefly after a spike.",
    ),
    TunableSpec(
        path="WatchProfile.gate.require_false_first", owner="Gate", visibility="public",
        safe_to_optimize=False, default=False, value_type="bool", risk="high",
        note="typed handle onto Settings.gate.require_false_first; same Gate-only ownership.",
    ),
    TunableSpec(
        path="WatchProfile.gate.refractory_s", owner="Gate", visibility="public", safe_to_optimize=False,
        default=0.5, value_type="float", bounds=(0.0, float("inf")), risk="high",
        note="typed handle onto the re-arm interval; same Gate-only ownership.",
    ),
    TunableSpec(
        path="WatchProfile.goal_density_guidance.max_recommended_goals", owner="unclassified",
        visibility="advisory", safe_to_optimize=False, default=3, value_type="int",
        bounds=(1.0, float("inf")), risk="low",
        note="advisory only, NOTHING enforces it (module docstring); empirical tuning held <=3.",
    ),
    TunableSpec(
        path="WatchProfile.goal_density_guidance.note", owner="unclassified", visibility="advisory",
        safe_to_optimize=False, default="", value_type="str", risk="low",
        note="free-text guidance carried alongside max_recommended_goals.",
    ),
    TunableSpec(
        path="WatchProfile.notes", owner="unclassified", visibility="descriptive", safe_to_optimize=False,
        default="", value_type="str", risk="low", note="free-text provenance, not a runtime value.",
    ),
)

# ======================================================================================================
# Section E — ``SourceProfile`` fields (percept/sourceprofile.py, ADR-005). All owner="SourceProfile"
# except the one field ADR-008 explicitly flags as unresolved (see the module docstring and
# ADR-008 §Corrections(b) / §Open questions #1).
# ======================================================================================================

_SOURCE_PROFILE: tuple[TunableSpec, ...] = (
    TunableSpec(
        path="SourceProfile.name", owner="SourceProfile", visibility="key", safe_to_optimize=False,
        default=None, value_type="str", risk="low", note="open-registry identity key.",
    ),
    TunableSpec(
        path="SourceProfile.modalities", owner="SourceProfile", visibility="public",
        safe_to_optimize=False, default=None, value_type="frozenset", risk="medium",
        note="what the stream carries, e.g. {'video'}, {'audio','text'}.",
    ),
    TunableSpec(
        path="SourceProfile.temporal_unit", owner="SourceProfile", visibility="public",
        safe_to_optimize=False, default=None, value_type="str", risk="medium",
        note="what one unit of time IS on this source: 'frame'/'page'/'turn'.",
    ),
    TunableSpec(
        path="SourceProfile.admission_strategy", owner="SourceProfile", visibility="public",
        safe_to_optimize=False, default=None, value_type="str", risk="high",
        note="the ownership table's own 'source-class strategy' example.",
    ),
    TunableSpec(
        path="SourceProfile.evidence_strategy", owner="SourceProfile", visibility="public",
        safe_to_optimize=False, default=None, value_type="str", risk="medium",
        note="how evidence is packaged for the judge and the incident.",
    ),
    TunableSpec(
        path="SourceProfile.identity_strategy", owner="SourceProfile", visibility="public",
        safe_to_optimize=False, default=None, value_type="str", risk="high",
        note="the ownership table's own 'source-class strategy' example.",
    ),
    TunableSpec(
        path="SourceProfile.default_timebase", owner="SourceProfile", visibility="public",
        safe_to_optimize=False, default="auto", value_type="enum",
        choices=("auto", "media", "wall"), risk="medium", note="the honest default clock for this source.",
    ),
    TunableSpec(
        path="SourceProfile.parameters.change_signal", owner="SourceProfile", visibility="public",
        safe_to_optimize=False, default=None, value_type="str", risk="medium",
        note="UNTUNED marker or a registered detector name; honesty-about-admission mechanism.",
    ),
    TunableSpec(
        path="SourceProfile.parameters.coast_interval_s", owner="SourceProfile", visibility="public",
        safe_to_optimize=True, default=None, value_type="float", bounds=(0.0, float("inf")), risk="low",
        note="fixed low-rate sampling floor when the change signal is absent/untuned.",
    ),
    TunableSpec(
        path="SourceProfile.parameters.coast_required", owner="SourceProfile", visibility="descriptive",
        safe_to_optimize=False, default=False, value_type="bool", risk="low",
        note="the profile's own assertion that its coast floor is not optional; read by the resolver and "
        "doctor so a strict_admission override of the coast is reported LOUDLY, never silently dropped.",
    ),
    TunableSpec(
        path="SourceProfile.parameters.unknown_continuity_gap_s", owner="SourceProfile",
        visibility="public", safe_to_optimize=True, default=None, value_type="float",
        bounds=(0.0, float("inf")), risk="low",
        note="default_episode_policy input: unknown-to-unknown temporal-proximity continuity window "
        "(percept.sourceprofile.SourceProfile.default_episode_policy); off by default.",
    ),
    TunableSpec(
        path="SourceProfile.parameters.unknown_attaches_to_active_bound", owner="SourceProfile",
        visibility="public", safe_to_optimize=False, default=False, value_type="bool", risk="low",
        note="default_episode_policy input: single-focus attach; off by default.",
    ),
    TunableSpec(
        path="SourceProfile.parameters.coverage_floor", owner="SourceProfile", visibility="advisory",
        safe_to_optimize=False, default=None, value_type="float", risk="medium",
        note="ADR-008 §Corrections(b)/§Open questions #1, resolved: a per-source-class minimum "
        "admission count over a run, declared explicitly and opt-in (default None = nothing "
        "declared). Must be explicit before a zero-admit run is ever reported 'degraded' rather "
        "than quiet (percept.doctor's own rule) -- an undeclared floor leaves "
        "RuntimeHealthEvent('zero_admission_coverage') exactly as diagnostic as it always was; a "
        "declared-and-unmet floor is what makes it read as degraded (percept.runtime.worker.run's "
        "coverage_floor= kwarg, sourced from this field via percept.runtime_compiler).",
    ),
    TunableSpec(
        path="SourceProfile.stability", owner="SourceProfile", visibility="public", safe_to_optimize=False,
        default="stable", value_type="enum", choices=("stable", "experimental"), risk="low",
        note="experimental profiles may change strategy without a contract bump.",
    ),
    TunableSpec(
        path="SourceProfile.notes", owner="SourceProfile", visibility="descriptive", safe_to_optimize=False,
        default="", value_type="str", risk="low", note="provenance/honesty text (doctor quotes it).",
    ),
)

# ======================================================================================================
# Section F — detector manifests, edge-detector instance knobs, and backend/model selection env vars
# living OUTSIDE the Settings tree (percept/harness/detectors.py, percept/harness/pose.py, and the
# per-backend PERCEPT_VISION_TIMEOUT_S / PERCEPT_STRIP_PROMPT / PERCEPT_HOME reads).
# ======================================================================================================

_DETECTOR_AND_BACKEND: tuple[TunableSpec, ...] = (
    # -- DetectorManifest (self-describing registry entry, harness/detectors.py) — descriptive metadata,
    # not tuned values; category is the one field with a real taxonomy the ownership table can reach.
    TunableSpec(
        path="DetectorManifest.name", owner="unclassified", visibility="descriptive",
        safe_to_optimize=False, default=None, value_type="str", risk="low",
        note="self-describing registry metadata; no owner among the six exists yet (DomainPack-shaped).",
    ),
    TunableSpec(
        path="DetectorManifest.senses", owner="unclassified", visibility="descriptive",
        safe_to_optimize=False, default=None, value_type="str", risk="low",
        note="the physical signal read; descriptive metadata.",
    ),
    TunableSpec(
        path="DetectorManifest.when_to_use", owner="unclassified", visibility="descriptive",
        safe_to_optimize=False, default=None, value_type="str", risk="low",
        note="conductor-authoring hint; DomainPack-shaped ('domain vocabulary') but DomainPack "
        "does not exist in code yet.",
    ),
    TunableSpec(
        path="DetectorManifest.output", owner="unclassified", visibility="descriptive",
        safe_to_optimize=False, default=None, value_type="str", risk="low",
        note="the boundary shape it emits; descriptive metadata.",
    ),
    TunableSpec(
        path="DetectorManifest.category", owner="SourceProfile", visibility="descriptive",
        safe_to_optimize=False, default="", value_type="enum",
        choices=(
            "motion", "audio", "frame", "presence", "pose", "audio_event", "change",
            "semantic_relevance",
        ),
        risk="low", note="coarse sense CLASS; the one manifest field with a source-class-shaped taxonomy.",
    ),
    # -- edge-detector instance knobs (harness/detectors.py) — cheap admission/wake calibration ->
    # SourceProfile, same reasoning as Settings.edge.*. refractory_s here is a DETECTOR debounce,
    # distinct from Settings.gate.refractory_s (the Gate re-fire suppression window).
    TunableSpec(
        path="MotionPeriodicity.min_height", owner="SourceProfile", visibility="public",
        safe_to_optimize=True, default=0.02, value_type="float", bounds=(0.0, 1.0), risk="low",
        note="ignores still-scene jitter below this motion magnitude.",
    ),
    TunableSpec(
        path="MotionPeriodicity.rel_drop", owner="SourceProfile", visibility="public",
        safe_to_optimize=True, default=0.5, value_type="float", bounds=(0.0, 1.0), risk="low",
        note="a bump confirms only once the signal falls to (1-rel_drop) of its peak.",
    ),
    TunableSpec(
        path="MotionPeriodicity.refractory_s", owner="SourceProfile", visibility="public",
        safe_to_optimize=True, default=0.0, value_type="float", bounds=(0.0, float("inf")), risk="low",
        note="detector-level debounce (distinct from Gate.refractory_s, the fire-suppression window).",
    ),
    TunableSpec(
        path="AcousticOnset.high", owner="SourceProfile", visibility="public", safe_to_optimize=True,
        default=0.3, value_type="float", bounds=(0.0, 1.0), risk="low",
        note="hysteresis-crossing high threshold (energy onset).",
    ),
    TunableSpec(
        path="AcousticOnset.low", owner="SourceProfile", visibility="public", safe_to_optimize=True,
        default=0.15, value_type="float", bounds=(0.0, 1.0), risk="low",
        note="hysteresis-crossing low threshold (re-arm).",
    ),
    TunableSpec(
        path="AcousticOnset.refractory_s", owner="SourceProfile", visibility="public",
        safe_to_optimize=True, default=0.0, value_type="float", bounds=(0.0, float("inf")), risk="low",
        note="detector-level debounce (distinct from Gate.refractory_s).",
    ),
    TunableSpec(
        path="PoseOpenness.min_height", owner="SourceProfile", visibility="public", safe_to_optimize=True,
        default=0.6, value_type="float", bounds=(0.0, float("inf")), risk="low",
        note="inner MotionPeriodicity floor on the openness waveform.",
    ),
    TunableSpec(
        path="PoseOpenness.rel_drop", owner="SourceProfile", visibility="public", safe_to_optimize=True,
        default=0.4, value_type="float", bounds=(0.0, 1.0), risk="low",
        note="inner MotionPeriodicity descent-confirmation fraction.",
    ),
    TunableSpec(
        path="PoseOpenness.refractory_s", owner="SourceProfile", visibility="public",
        safe_to_optimize=True, default=0.4, value_type="float", bounds=(0.0, float("inf")), risk="low",
        note="detector-level debounce between counted reps.",
    ),
    TunableSpec(
        path="PoseOpenness.provider_name", owner="DeploymentConfig", visibility="public",
        safe_to_optimize=False, default=None, value_type="str", choices=("rtmo", "blazepose"),
        env_var="PERCEPT_POSE_PROVIDER", risk="medium",
        note="pose provider selection; lazy + opt-in (percept-vision[pose]).",
    ),
    TunableSpec(
        path="WakePolicy.threshold", owner="SourceProfile", visibility="public", safe_to_optimize=True,
        default=0.1, value_type="float", bounds=(0.0, 1.0), risk="medium",
        note="fused wake score cutoff before spending a heavy vision call; admission-strategy shape.",
    ),
    TunableSpec(
        path="WakePolicy.weights", owner="SourceProfile", visibility="public", safe_to_optimize=False,
        default=None, value_type="dict", risk="medium",
        note="per-channel weights in the calibrated weighted-OR fusion.",
    ),
    TunableSpec(
        path="WakePolicy.floor", owner="SourceProfile", visibility="public", safe_to_optimize=True,
        default=0.0, value_type="float", bounds=(0.0, 1.0), risk="low",
        note="minimum fused score floor.",
    ),
    # -- backend/provider env vars living OUTSIDE the Settings tree — DeploymentConfig ("env vars live
    # here, nowhere else").
    TunableSpec(
        path="backend.vision_timeout_s", owner="DeploymentConfig", visibility="public",
        safe_to_optimize=False, default=None, value_type="float", bounds=(0.0, float("inf")),
        env_var="PERCEPT_VISION_TIMEOUT_S", risk="medium",
        note="per-backend request timeout (anthropic/gemini_live/openrouter backends).",
    ),
    TunableSpec(
        path="backend.strip_prompt_variant", owner="DeploymentConfig", visibility="internal",
        safe_to_optimize=False, default="change", value_type="enum", choices=("change", "neutral"),
        env_var="PERCEPT_STRIP_PROMPT", risk="low",
        note="gemini backend strip-judging prompt variant switch.",
    ),
    TunableSpec(
        path="backend.pose_model_path", owner="DeploymentConfig", visibility="public",
        safe_to_optimize=False, default="", value_type="str", env_var="PERCEPT_POSE_MODEL", risk="medium",
        note="on-disk/URI pose model override.",
    ),
    TunableSpec(
        path="backend.rtmo_model", owner="DeploymentConfig", visibility="public", safe_to_optimize=False,
        default="rtmo", value_type="str", env_var="PERCEPT_RTMO_MODEL", risk="medium",
        note="RTMO pose model variant selection.",
    ),
    TunableSpec(
        path="backend.pose_device", owner="DeploymentConfig", visibility="public", safe_to_optimize=False,
        default="cpu", value_type="str", env_var="PERCEPT_POSE_DEVICE", risk="low",
        note="inference device for the pose provider.",
    ),
    TunableSpec(
        path="backend.pose_channels", owner="DeploymentConfig", visibility="public",
        safe_to_optimize=False, default=None, value_type="str", env_var="PERCEPT_POSE_CHANNELS",
        risk="low", note="channel-selection override for pose decomposition.",
    ),
    TunableSpec(
        path="backend.percept_home", owner="DeploymentConfig", visibility="public", safe_to_optimize=False,
        default=None, value_type="str", env_var="PERCEPT_HOME", requires_restart=True, risk="low",
        note="on-disk home directory for the label store; read at process/module init, needs a restart.",
    ),
)


def _build_registry() -> tuple[TunableSpec, ...]:
    """Concatenate every section into one registry, sorted by ``path`` for a stable, reviewable order.

    Sorting (rather than relying on section-concatenation order) means adding a spec to any section
    above, in any position, produces the same serialized ``tunables.json`` — the registry's own
    internal grouping is for human readability, never for the artifact's ordering.
    """
    all_specs = (
        _SETTINGS + _RUN_PARAMETERS + _GOAL_STATE + _WATCH_PROFILE + _SOURCE_PROFILE
        + _DETECTOR_AND_BACKEND
    )
    seen: dict[str, TunableSpec] = {}
    for spec in all_specs:
        if spec.path in seen:
            raise TunableError(f"duplicate TunableSpec.path: {spec.path!r}")
        seen[spec.path] = spec
    return tuple(sorted(all_specs, key=lambda s: s.path))


#: The full, typed tunable registry — every knob from the six existing configuration surfaces
#: ADR-008 names, each carrying an owner from the ADR's ownership table (or one of the two documented
#: escape hatches). Building this performs no I/O (module docstring's "no import-time semantic config").
REGISTRY: tuple[TunableSpec, ...] = _build_registry()


def owners_summary(registry: tuple[TunableSpec, ...] = REGISTRY) -> dict[str, int]:
    """Tunable count by owner — the shape of the summary report this module hands back."""
    out: dict[str, int] = {}
    for spec in registry:
        out[spec.owner] = out.get(spec.owner, 0) + 1
    return out


def generate(registry: tuple[TunableSpec, ...] = REGISTRY) -> str:
    """Render ``registry`` to the canonical ``tunables.json`` text (stable across runs and machines:
    fixed key order via ``sort_keys``, a trailing newline, no floating hash-seed-dependent ordering)."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "tunables": [spec.to_dict() for spec in registry],
    }
    return json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"


#: ``tunables.json`` lives at the package root (sibling of ``pyproject.toml``), not inside ``percept/``
#: — it is a generated ARTIFACT for tooling/clients to read, not importable package data.
ARTIFACT_PATH = Path(__file__).resolve().parent.parent / "tunables.json"


def write_tunables_json(path: Path = ARTIFACT_PATH, registry: tuple[TunableSpec, ...] = REGISTRY) -> Path:
    """Write the generated artifact to ``path`` (default :data:`ARTIFACT_PATH`). Returns the path."""
    path.write_text(generate(registry), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:  # noqa: ARG001 — CLI entry-point signature convention
    """``python -m percept.tunables`` — (re)generate ``tunables.json`` at the package root."""
    out = write_tunables_json()
    counts = owners_summary()
    total = len(REGISTRY)
    print(f"wrote {out} ({total} tunables)")
    for owner in sorted(counts):
        print(f"  {owner}: {counts[owner]}")
    return 0


if __name__ == "__main__":  # pragma: no cover — exercised via main() in tests
    raise SystemExit(main())


__all__ = [
    "SCHEMA_VERSION",
    "OWNERS",
    "VISIBILITIES",
    "RISKS",
    "TYPES",
    "TunableError",
    "TunableSpec",
    "REGISTRY",
    "ARTIFACT_PATH",
    "owners_summary",
    "generate",
    "write_tunables_json",
    "main",
]
