"""RuntimeCompiler — the ADR-008 resolution boundary, step 2 of the two-step binder.

``docs/adr/ADR-008-configuration-ownership-resolution-provenance.md`` draws the boundary as two
functions::

    (SourceProfile, DomainPack, ExecutionPolicy, DeploymentConfig, OptimizationLock, WatchPlan)
            |
            v
       ConfigResolver.resolve(...)   <- percept.resolver
            |
            v
       ResolvedWatch
            |
            v
       RuntimeCompiler.compile(...)  <- THIS MODULE
            |
            v
    (Goal / Goal.state, WatchSpec inputs, SensePolicy, Settings)

This module is a THIN adaptor, nothing more: :func:`runtime_compile` reads a
:class:`~percept.contracts.ResolvedWatch` and returns the exact shapes the EXISTING runtime already
knows how to consume — a :class:`~percept.config.Settings` tree for :meth:`Percept.create`, a
:class:`~percept.contracts.Goal` for :meth:`Percept.add_goal`, and the keyword arguments
:func:`percept.harness.tier0.compile_watchspec` already accepts. It never re-decides configuration
(no new threshold, no new precedence rule, no new vocabulary) — every value it produces is either
read directly off ``resolved.effective`` (itself produced by :class:`percept.resolver.ConfigResolver`)
or re-shaped from one existing dict form into another existing dict form.

Two things this module deliberately does NOT do (both stay with the callers of this module, per the
ADR's own boundary):

- **No adapter construction.** ``Percept.create`` (unchanged) still resolves provider NAMES to
  constructed backends via ``percept.provider``/``percept.registry`` — this module only carries the
  provider *names* forward (in :attr:`CompiledWatch.settings.models` and, for observability,
  :attr:`CompiledWatch.components`), never a model id as adapter identity (ADR-008 Invariants: "Model
  id != adapter identity").
- **No goal.state coercion.** The ``Goal`` this module returns carries an assembled but NOT YET
  validated ``state`` (its ``"behavior"``/``"rhythm"``/``"episode"``/``"sense"``/``"perception"``
  sub-blocks are in the RAW shape :mod:`percept.behavior`'s ``coerce_authored_*`` firewall already
  expects as input). Running that firewall is :func:`percept.watchplan_apply.apply_watch_plan`'s job
  (coercion happens at APPLY time, per ADR-008), not this module's — a bare
  :func:`runtime_compile` call performs no validation and raises nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .config import (
    CacheSettings,
    ConsentSettings,
    EdgeSettings,
    EntitySettings,
    EpisodeSettings,
    EventSettings,
    GateSettings,
    IdentitySettings,
    ModelSettings,
    RoutingSettings,
    Settings,
)
from .contracts import Goal
from .contracts.watchplan import ResolvedWatch
from .goalsugar import goal_id_for
from .sampling import SensePolicy
from .watchprofile import CadenceProfile

#: ``Goal.state`` flat field names that belong to ``coerce_authored_behavior``'s RAW input shape
#: (percept/behavior.py) — everything else under ``Goal.state.*`` in ``resolved.effective`` is either
#: a coerced sub-block (rhythm/episode/sense/perception) or an uncoerced pass-through the runtime
#: already reads directly (e.g. ``hold_s``, ``identity_policy``, ``judge_mode``, ``delivery.*``).
_BEHAVIOR_FLAT_KEYS: tuple[str, ...] = (
    "type", "rearm", "cadence_s", "deadline_after_s", "from", "to",
    "fire_when", "window_s", "miss_cost", "false_fire_cost", "protected_floor",
)

# ``WatchPlan.sense`` is the portable media spelling at the configuration boundary.  The existing
# behavior runtime calls the image modality ``vision``; ``video`` is accepted as the source-facing
# synonym and normalized at compile time.  No other media shape is silently approximated.
_SENSE_MODALITIES: dict[str, str] = {
    "video": "vision",
    "vision": "vision",
    "audio": "audio",
    "fused": "fused",
}
_SENSE_SOURCE_REQUIREMENTS: dict[str, frozenset[str]] = {
    "video": frozenset({"video"}),
    "vision": frozenset({"video"}),
    "audio": frozenset({"audio"}),
    "fused": frozenset({"video", "audio"}),
}
_EXECUTABLE_SENSES: frozenset[str] = frozenset({"video", "vision", "audio", "fused"})


def supports_sense(sense: str | None) -> bool:
    """Whether the runtime can execute a WatchPlan media sense."""
    return sense is None or sense in _EXECUTABLE_SENSES


def sense_matches_modalities(sense: str | None, modalities: Any) -> bool:
    """Whether a SourceProfile actually carries every channel requested by a media sense."""
    if sense is None:
        return True
    required = _SENSE_SOURCE_REQUIREMENTS.get(sense)
    return required is not None and required.issubset(frozenset(modalities))


def modality_matches_source(modality: str, modalities: Any) -> bool:
    """Whether a compiled perception modality is executable on a SourceProfile's channels."""
    required = {
        "vision": frozenset({"video"}),
        "audio": frozenset({"audio"}),
        "fused": frozenset({"video", "audio"}),
    }.get(modality)
    return required is not None and required.issubset(frozenset(modalities))


@dataclass(frozen=True)
class CompiledWatch:
    """What :func:`runtime_compile` produces — the exact inputs ``Percept.create`` /
    ``Percept.add_goal`` / :func:`percept.harness.tier0.compile_watchspec` already accept.

    - ``settings`` — a full :class:`~percept.config.Settings`, built from ``resolved.effective``'s
      ``Settings.*`` paths (the resolver's own SDK-default snapshot overlaid with every layer that
      won a field — this is byte-identical to ``load_settings()`` when nothing overrode anything).
    - ``goal`` — one :class:`~percept.contracts.Goal` for ``resolved.plan``. Its ``state`` is
      ASSEMBLED, not yet coerced (see module docstring) — pass it through
      :func:`percept.watchplan_apply.apply_watch_plan`, not straight to ``add_goal``, unless the
      plan authored no behavior/rhythm/episode/sense/perception fields at all (then there is nothing
      to coerce and the state is already final).
    - ``sense_policy`` — a real :class:`~percept.sampling.SensePolicy`, or ``None`` when nothing in
      ``resolved.effective`` named a cadence override (byte-for-byte pre-ADR-008: no override).
    - ``plan_mode`` — the resolved temporal-shape string, or ``None`` (no override).
    - ``timebase`` — the resolved :class:`~percept.sourceprofile.SourceProfile.default_timebase`
      (``"auto"``/``"media"``/``"wall"``), or ``None`` when no profile contributed one — a caller
      threads this straight into :func:`percept.runtime.worker.run`'s own ``timebase=`` kwarg
      (``run(core, source, timebase=compiled.timebase or "auto")``), never a new timebase concept
      ("default_timebase -> run(timebase=)").
    - ``watchspec_kwargs`` — ``threshold``/``salience_stat``/``ego_compensate``/``stat_thresholds``,
      exactly :func:`percept.harness.tier0.compile_watchspec`'s deployment-default kwargs.
    - ``coverage_floor`` — the resolved ``SourceProfile.parameters['coverage_floor']`` (ADR-008
      §Corrections(b)), or ``None`` when no profile declared one — a caller threads this into
      :func:`percept.runtime.worker.run`'s own ``coverage_floor=`` kwarg
      (``run(core, source, coverage_floor=compiled.coverage_floor)``) so a declared-and-unmet floor
      classifies that run's ``zero_admission_coverage`` health event as degraded; an undeclared floor
      (the default) leaves it exactly the diagnostic-only reading it always was.
    - ``components`` — provider NAME identities the compiler carried forward (``vision``,
      ``conductor``, ``specialist``, ``stt``, ``tts``), never a model id (ADR-008 Invariants). This is
      the fuller snapshot ``ResolvedWatch.components`` deliberately stays minimal for — observability
      only, nothing downstream branches on it.
    """

    settings: Settings
    goal: Goal
    sense_policy: SensePolicy | None
    plan_mode: str | None
    timebase: str | None
    coverage_floor: float | None
    watchspec_kwargs: Mapping[str, Any]
    components: Mapping[str, Any]


def _settings_from_effective(effective: Mapping[str, Any]) -> Settings:
    """Rebuild a full :class:`Settings` tree from ``resolved.effective``'s ``Settings.*`` paths.

    ``percept.tunables.REGISTRY`` names all 38 ``Settings.*`` leaf fields (Section A), so a real
    :class:`~percept.resolver.ConfigResolver` output always carries every one of them (the SDK-default
    layer alone contributes the full snapshot). The ``base.<field>`` fallback below only matters for a
    hand-built/partial ``ResolvedWatch`` (e.g. in a unit test) that omits some paths.
    """
    base = Settings()

    def get(path: str, default: Any) -> Any:
        return effective.get(path, default)

    return Settings(
        gate=GateSettings(
            threshold=get("Settings.gate.threshold", base.gate.threshold),
            validity_window_s=get("Settings.gate.validity_window_s", base.gate.validity_window_s),
            key_ttl_s=get("Settings.gate.key_ttl_s", base.gate.key_ttl_s),
            max_keys=get("Settings.gate.max_keys", base.gate.max_keys),
            stale_ask=get("Settings.gate.stale_ask", base.gate.stale_ask),
            require_false_first=get(
                "Settings.gate.require_false_first", base.gate.require_false_first
            ),
            refractory_s=get("Settings.gate.refractory_s", base.gate.refractory_s),
        ),
        entity=EntitySettings(
            iou_match=get("Settings.entity.iou_match", base.entity.iou_match),
            max_entities=get("Settings.entity.max_entities", base.entity.max_entities),
        ),
        cache=CacheSettings(
            enabled=get("Settings.cache.enabled", base.cache.enabled),
            threshold=get("Settings.cache.threshold", base.cache.threshold),
            ttl_s=get("Settings.cache.ttl_s", base.cache.ttl_s),
            tau_bits=get("Settings.cache.tau_bits", base.cache.tau_bits),
            audit_rate=get("Settings.cache.audit_rate", base.cache.audit_rate),
        ),
        models=ModelSettings(
            vision=get("Settings.models.vision", base.models.vision),
            conductor=get("Settings.models.conductor", base.models.conductor),
            specialist=get("Settings.models.specialist", base.models.specialist),
            stt=get("Settings.models.stt", base.models.stt),
            tts=get("Settings.models.tts", base.models.tts),
            vision_model=get("Settings.models.vision_model", base.models.vision_model),
            conductor_model=get("Settings.models.conductor_model", base.models.conductor_model),
            specialist_model=get("Settings.models.specialist_model", base.models.specialist_model),
        ),
        routing=RoutingSettings(
            confidence_threshold=get(
                "Settings.routing.confidence_threshold", base.routing.confidence_threshold
            ),
        ),
        identity=IdentitySettings(
            appearance_threshold=get(
                "Settings.identity.appearance_threshold", base.identity.appearance_threshold
            ),
        ),
        episodes=EpisodeSettings(
            window_s=get("Settings.episodes.window_s", base.episodes.window_s),
            max_episodes=get("Settings.episodes.max_episodes", base.episodes.max_episodes),
        ),
        events=EventSettings(
            window=get("Settings.events.window", base.events.window),
            preroll_max_bytes=get(
                "Settings.events.preroll_max_bytes", base.events.preroll_max_bytes
            ),
            merge_gap_s=get("Settings.events.merge_gap_s", base.events.merge_gap_s),
            audio_window_s=get("Settings.events.audio_window_s", base.events.audio_window_s),
        ),
        consent=ConsentSettings(
            allow_computer_use=get(
                "Settings.consent.allow_computer_use", base.consent.allow_computer_use
            ),
            allowed_actions=get("Settings.consent.allowed_actions", base.consent.allowed_actions),
            actuation=get("Settings.consent.actuation", base.consent.actuation),
        ),
        edge=EdgeSettings(
            motion_gate=get("Settings.edge.motion_gate", base.edge.motion_gate),
            salience_stat=get("Settings.edge.salience_stat", base.edge.salience_stat),
            ego_compensate=get("Settings.edge.ego_compensate", base.edge.ego_compensate),
            motion_gate_block_max=get(
                "Settings.edge.motion_gate_block_max", base.edge.motion_gate_block_max
            ),
            motion_gate_p95=get("Settings.edge.motion_gate_p95", base.edge.motion_gate_p95),
            coast_enabled=get("Settings.edge.coast_enabled", base.edge.coast_enabled),
        ),
    )


def _nested_goal_state(effective: Mapping[str, Any]) -> dict[str, Any]:
    """Every ``Goal.state.*`` path in ``effective`` (WatchPlan-explicit overrides only — no other
    resolver layer writes this namespace, see ``percept.tunables`` Section C), rebuilt into a nested
    dict keyed exactly like :attr:`~percept.contracts.Goal.state`. Generic over depth so a new
    ``Goal.state.*`` registry path needs no new code here.
    """
    out: dict[str, Any] = {}
    for path, value in effective.items():
        if not path.startswith("Goal.state."):
            continue
        parts = path[len("Goal.state.") :].split(".")
        node = out
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def _goal_state_from_resolved(resolved: ResolvedWatch) -> dict[str, Any]:
    """Assemble ``Goal.state`` from ``resolved.plan`` + ``resolved.effective``'s ``Goal.state.*``
    paths. The ``"behavior"``/``"rhythm"``/``"episode"``/``"sense"``/``"perception"`` sub-blocks
    this returns are in the RAW shape :mod:`percept.behavior`'s ``coerce_authored_*`` firewall
    expects as INPUT (not yet coerced) — see the module docstring. Everything else (``hold_s``,
    ``identity_policy``, ``judge_mode``, ``delivery.*``, ...) is an existing goal.state key the
    runtime already reads directly, with no coercer of its own, so it passes through unchanged.
    """
    plan = resolved.plan
    nested = _nested_goal_state(resolved.effective)

    flat_behavior = {k: nested.pop(k) for k in _BEHAVIOR_FLAT_KEYS if k in nested}
    # An explicit "Goal.state.*" path is the more specific ask; a WatchPlan typed field (`rearm`,
    # `consequence`) only fills a gap the explicit path left open — never overrides it.
    if "rearm" not in flat_behavior and plan.rearm is not None:
        flat_behavior["rearm"] = plan.rearm
    if plan.consequence:
        cons = dict(plan.consequence)
        for key in ("miss_cost", "false_fire_cost", "protected_floor"):
            if key not in flat_behavior and key in cons:
                flat_behavior[key] = cons[key]

    behavior_raw: dict[str, Any] = {}
    for key in ("type", "rearm", "cadence_s", "deadline_after_s", "from", "to"):
        if key in flat_behavior:
            behavior_raw[key] = flat_behavior[key]
    if "fire_when" in flat_behavior or "window_s" in flat_behavior:
        behavior_raw["accumulate"] = {
            k: flat_behavior[k] for k in ("fire_when", "window_s") if k in flat_behavior
        }
    if any(k in flat_behavior for k in ("miss_cost", "false_fire_cost", "protected_floor")):
        behavior_raw["consequence"] = {
            k: flat_behavior[k]
            for k in ("miss_cost", "false_fire_cost", "protected_floor")
            if k in flat_behavior
        }

    state: dict[str, Any] = {"type": "watch", "count": 0}
    if plan.target is not None:
        # Target is an inspectable entity/subject hint carried with the executable Goal. It does not
        # override the plain-language condition or invent an identity resolver.
        state["target"] = plan.target
    if behavior_raw:
        state["behavior"] = behavior_raw
    state.update(nested)  # rhythm/episode/sense/perception raw sub-blocks + any flat pass-through
    if plan.detectors:
        state["detectors"] = list(plan.detectors)  # coerced downstream by detectors_for() itself
    if plan.schedule is not None:
        state["schedule"] = dict(plan.schedule)  # coerced downstream by compile_watchspec itself
    if plan.sense is not None:
        modality = _SENSE_MODALITIES.get(plan.sense)
        if modality is None:
            raise ValueError(
                f"WatchPlan.sense {plan.sense!r} is not executable by this runtime; "
                f"supported media senses are {tuple(sorted(_EXECUTABLE_SENSES))}"
            )
        perception = state.setdefault("perception", {})
        perception.setdefault("modality", modality)
    return state


def _sense_policy_from_effective(effective: Mapping[str, Any]) -> SensePolicy | None:
    """An explicit ``Percept.create.sense_policy`` object wins outright (the caller built one by
    hand); otherwise assemble one from ``WatchProfile.cadence.*`` fields (a registered-profile
    lookup fed through ``percept.resolver``'s ``situation``/``registry`` seam) when any are present.
    ``None`` when neither contributed anything — byte-for-byte pre-ADR-008 (no cadence override).
    """
    explicit = effective.get("Percept.create.sense_policy")
    if explicit is not None:
        return explicit
    cadence_fields = {
        path[len("WatchProfile.cadence.") :]: value
        for path, value in effective.items()
        if path.startswith("WatchProfile.cadence.")
    }
    if not cadence_fields:
        return None
    default = CadenceProfile()  # no literal knobs here — read the shipped default, don't invent one
    return SensePolicy(
        base_fps=cadence_fields.get("base_fps", default.base_fps),
        max_fps=cadence_fields.get("max_fps", default.max_fps),
        escalate_threshold=cadence_fields.get("escalate_threshold", default.escalate_threshold),
        hold_s=cadence_fields.get("hold_s", default.hold_s),
    )


def runtime_compile(resolved: ResolvedWatch) -> CompiledWatch:
    """Turn a :class:`~percept.contracts.ResolvedWatch` into the runtime primitives
    ``Percept.create``/``Percept.add_goal``/``compile_watchspec`` already accept.

    Performs no I/O, no ownership re-check (that already happened in
    :meth:`percept.resolver.ConfigResolver.resolve`), and constructs no adapter. Pure translation.
    """
    settings = _settings_from_effective(resolved.effective)
    state = _goal_state_from_resolved(resolved)
    plan = resolved.plan
    perception = state.get("perception")
    if isinstance(perception, dict) and perception.get("modality") in ("audio", "fused"):
        # A modality label without a clip duration strands the already-wired AudioWindow: the agent's
        # audio assembly correctly returns no payload when ``audio_s`` is absent. Bind the resolved
        # event-window duration as the default so a compiled audio/fused Watch actually hands audio to
        # a capable judge. An explicit Goal.state.perception.audio_s remains more specific.
        perception.setdefault("audio_s", settings.events.audio_window_s)
    sense_policy = _sense_policy_from_effective(resolved.effective)
    if sense_policy is not None:
        # Cadence already has a runtime primitive: the authored ``Goal.state.sense`` block consumed
        # by ``sensor_plan_for``/``runtime.worker._capture_plan``.  Put the compiled per-Watch policy
        # there so two Watches can carry different capture floors without mutating a shared core.
        sense = state.setdefault("sense", {})
        sense.setdefault("base_fps", sense_policy.base_fps)
        sense.setdefault("max_fps", sense_policy.max_fps)
        sense.setdefault("escalate_threshold", sense_policy.escalate_threshold)
    goal_id = plan.id if plan.id else goal_id_for(plan.condition)
    goal = Goal(
        id=goal_id, condition=plan.condition, say=plan.condition, source=plan.source, state=state
    )

    # plan_mode: an explicit "Percept.create.plan_mode" override wins over a situation-derived
    # "WatchProfile.plan_mode" (see percept.resolver's situation-tier contribution) — both are
    # WatchPlan-owned per the tunables registry, but the direct kwarg-shaped path is the more
    # specific ask.
    plan_mode = resolved.effective.get("Percept.create.plan_mode")
    if plan_mode is None:
        plan_mode = resolved.effective.get("WatchProfile.plan_mode")

    # timebase: an explicit "Percept.create.timebase"/"run.timebase" override (if a caller ever
    # authors one via WatchPlan.explicit) wins over the profile's own default_timebase — same
    # more-specific-ask precedent as plan_mode above.
    timebase = resolved.effective.get("Percept.create.timebase")
    if timebase is None:
        timebase = resolved.effective.get("SourceProfile.default_timebase")

    # coverage_floor: the resolved SourceProfile's declared minimum admission count, if any (ADR-008
    # Corrections(b)) -- None when nothing declared one, byte-for-byte diagnostic-only.
    coverage_floor = resolved.effective.get("SourceProfile.parameters.coverage_floor")

    watchspec_kwargs: dict[str, Any] = {
        "threshold": settings.edge.motion_gate,
        "salience_stat": settings.edge.salience_stat,
        "ego_compensate": settings.edge.ego_compensate,
        "stat_thresholds": {
            "mean": settings.edge.motion_gate,
            "block_max": settings.edge.motion_gate_block_max,
            "p95": settings.edge.motion_gate_p95,
        },
    }

    components: dict[str, Any] = {
        "vision": settings.models.vision,
        "conductor": settings.models.conductor,
        "specialist": settings.models.specialist,
        "stt": settings.models.stt,
        "tts": settings.models.tts,
    }

    return CompiledWatch(
        settings=settings,
        goal=goal,
        sense_policy=sense_policy,
        plan_mode=plan_mode,
        timebase=timebase,
        coverage_floor=coverage_floor,
        watchspec_kwargs=watchspec_kwargs,
        components=components,
    )


__all__ = [
    "CompiledWatch",
    "modality_matches_source",
    "runtime_compile",
    "sense_matches_modalities",
    "supports_sense",
]
