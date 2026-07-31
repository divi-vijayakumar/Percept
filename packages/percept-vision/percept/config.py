"""Settings tree — every tunable env-overridable, no literals in logic (standing rule 5).

Read once via :func:`load_settings`. Each field maps to a ``PERCEPT_*`` env var so behaviour
is reproducible from the environment and nothing is hard-coded in the cognition logic.
"""

from __future__ import annotations

import logging
import os
import warnings
from contextvars import ContextVar
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any

from .envelope import ACTUATION_CLASSES  # ("none", "screen", "physical") — PERCEPT_ACTUATION
from .watchprofile import SALIENCE_STATS  # ("mean", "block_max", "p95") — PERCEPT_SALIENCE_STAT

#: The one diagnostic channel this module owns. Paired with :mod:`warnings` on every rejection:
#: a library that only warns is silent under ``-W ignore``, and one that only logs is silent until
#: someone configures logging. Both, or the operator learns nothing.
_LOG = logging.getLogger("percept.config")


class ModeError(RuntimeError):
    """A run would silently fake, and we refuse to let it look like success.

    Raised for the two configurations the offline/scripted RFC makes impossible to reach by
    accident: a LIVE run with no backend selected (or a selected backend missing its credential),
    and a SCRIPTED brain wired behind a LIVE voice path. The message always names the fix.
    """


# Which env var(s) supply each LIVE backend's credential. A name absent here (``scripted``/``fake``/a
# passed adapter instance) needs no credential. Do NOT hardcode a single vendor — every wired backend
# is checked against its OWN keys. (Grepped from percept.backends.*.)
_BACKEND_CREDENTIALS: dict[str, tuple[str, ...]] = {
    "claude": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_AISTUDIO_API_KEY"),
    "gemini-live": ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_AISTUDIO_API_KEY"),
    "openrouter": ("OPENROUTER_API_KEY", "OPENROUTER_KEY"),
    "deepgram": ("DEEPGRAM_API_KEY",),
}


def resolve_mode(mode: str | None = None) -> str:
    """Resolve the run mode. Precedence: explicit ``mode`` kwarg > ``PERCEPT_MODE`` env > ``"live"``.

    Only ``"offline"`` is special-cased downstream (it selects scripted seams); any other value is
    treated as live (real backends, credential-checked). Returned lowercased/stripped.
    """
    if mode is not None:
        return mode.strip().lower()
    env = os.environ.get("PERCEPT_MODE")
    if env:
        return env.strip().lower()
    return "live"


def missing_credential(name: Any) -> tuple[str, ...] | None:
    """For a LIVE backend *name*, the credential env keys if NONE is set, else ``None``.

    Returns ``None`` (no requirement) for a scripted/fake/unknown name or a non-string (an adapter
    instance the caller wired directly — that object raises its own error if it needs a key).
    """
    if not isinstance(name, str):
        return None
    keys = _BACKEND_CREDENTIALS.get(name)
    if not keys:
        return None
    if any(os.environ.get(k) for k in keys):
        return None
    return keys


@dataclass(frozen=True)
class _EnvRejection:
    """One ``PERCEPT_*`` override the loader refused, and the value it stood down to.

    ``reason`` is one of the four :data:`_REJECTION_DETAIL` keys, in the order :func:`_env` tests
    them: ``"empty"``, ``"uncast"``, ``"not_in_choices"``, ``"out_of_bounds"``.
    """

    var: str
    raw: str
    reason: str
    substituted: Any


#: The operator-facing half of each rejection reason. Keys are the closed ``reason`` vocabulary.
_REJECTION_DETAIL: dict[str, str] = {
    "empty": "is set but empty",
    "uncast": "is not readable as this field's type",
    "not_in_choices": "is outside this field's declared vocabulary",
    "out_of_bounds": "is outside this field's declared bounds",
}

#: The three bound shapes the tunables registry declares for ``Settings.*`` fields, named once so a
#: call site reads as the constraint rather than as two numbers. ``docs/configuration/tunables.md``
#: is generated FROM that registry, so these are a published promise, not a private preference:
#: ``PERCEPT_GATE_THRESHOLD=5.0`` used to load verbatim and leave a gate that classifies every
#: verdict "unknown" — a watch that can never fire, silently.
_UNIT_INTERVAL: tuple[float, float] = (0.0, 1.0)
_NON_NEGATIVE: tuple[float, float] = (0.0, float("inf"))
_AT_LEAST_ONE: tuple[float, float] = (1.0, float("inf"))

#: The ledger :func:`load_settings` installs for the duration of one load, so the rejections can be
#: attached to the returned :class:`Settings` (``env_rejections``) rather than only logged. A log
#: line that scrolled past is not an observability contract. ``None`` outside a load — a direct
#: ``_env`` call still emits on both channels, it simply has no tree to hang the record on.
#:
#: A :class:`~contextvars.ContextVar`, not a plain global: a ``ContextVar`` value does not propagate
#: into a newly started thread, so two runtimes loading settings concurrently cannot land entries in
#: each other's ledger or clobber each other's restore.
_ACTIVE_REJECTIONS: ContextVar[list[_EnvRejection] | None] = ContextVar(
    "percept_config_env_rejections", default=None
)


def _reject(name: str, raw: str, reason: str, default: Any, sink) -> Any:
    """Record, announce, and substitute the declared default. Never raises.

    A typo in a shell profile must not be fatal to a 24/7 process, so this is loud, not lethal:
    the ledger for machines, ``warnings`` for a developer, ``logging`` for a deployment.
    """
    record = _EnvRejection(var=name, raw=raw, reason=reason, substituted=default)
    ledger = _ACTIVE_REJECTIONS.get()
    if ledger is not None:
        ledger.append(record)
    if sink is not None:
        sink(record)
    message = (
        f"{name}={raw!r} {_REJECTION_DETAIL[reason]}; the override was discarded and "
        f"{name} fell back to its declared default {default!r}"
    )
    warnings.warn(message, RuntimeWarning, stacklevel=3)
    _LOG.warning(message)
    return default


def _env(
    name: str,
    default: Any,
    cast,
    *,
    choices: tuple[str, ...] | None = None,
    bounds: tuple[float, float] | None = None,
    sink=None,
) -> Any:
    """Read one ``PERCEPT_*`` override, or report why it was not adopted.

    The order of the five terms is the contract, not an implementation detail:

    1. unset -> the declared default, silently (nothing was asked for);
    2. **empty** -> rejected BEFORE the cast, because ``str`` and ``_as_bool`` cannot raise and
       would otherwise read ``PERCEPT_X=`` as a successful instruction (``""`` / ``False`` /
       ``frozenset()``) rather than as the absent value it is;
    3. uncastable -> rejected;
    4. outside ``choices`` -> rejected (the closed vocabularies the tunables registry declares);
    5. outside ``bounds`` -> rejected (ditto for the declared ranges).

    ``choices``/``bounds`` are duplicated at the call sites rather than imported from
    :mod:`percept.tunables` — that module imports ``behavior_routes`` and this one is imported by
    ``percept/__init__.py``, so the import would be a cycle. ``tests/test_config_tunables_drift.py``
    is what keeps the two copies honest, the same "duplicated, checked to agree by a test" pattern
    ``percept.resolver._MISS_COST_ORDER`` already uses.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    if raw.strip() == "":
        return _reject(name, raw, "empty", default, sink)
    try:
        value = cast(raw)
    except (TypeError, ValueError):
        return _reject(name, raw, "uncast", default, sink)
    if choices is not None and value not in choices:
        return _reject(name, raw, "not_in_choices", default, sink)
    if bounds is not None and not bounds[0] <= value <= bounds[1]:
        return _reject(name, raw, "out_of_bounds", default, sink)
    return value


def _as_bool(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _as_actions(raw: str) -> frozenset:
    return frozenset(a.strip() for a in raw.split(",") if a.strip())


@dataclass(frozen=True)
class GateSettings:
    threshold: float = 0.7  # PERCEPT_GATE_THRESHOLD — confident-yes cutoff
    validity_window_s: float = 90.0  # PERCEPT_GATE_VALIDITY_S — asserted-fact freshness
    # R-1 growth bound for the 24/7 gate. A per-entity edge key untouched (unevaluated) longer than the TTL
    # is swept; ``max_keys`` is the hard LRU backstop. ``run()`` installs ``key_ttl_s`` as the batteries-
    # included default (an unbounded gate is not a sane always-on default); a bare ``Percept`` stays
    # unbounded (byte-for-byte) unless a bound is passed. Never-goals + per-goal singletons are exempt.
    key_ttl_s: float = 3600.0  # PERCEPT_GATE_KEY_TTL_S — evict a gate key untouched this long (0/neg = off)
    max_keys: int = 0  # PERCEPT_GATE_MAX_KEYS — hard LRU ceiling on gate keys (0 = no ceiling)
    # A-3b: fold PERCEPTION recency into freshness, not just user-asserted facts. A goal not confidently
    # read within ``validity_window_s`` contributes freshness 0.0 -> ASK (invariant 2's stale->ASK arm,
    # which is otherwise dead because ``_freshness`` only measured asserted-fact age). OFF is byte-for-byte
    # the asserted-only path; this is the ONE switch in the hardening pass that changes WHAT FIRES (goals
    # begin ASK-ing on stale perception where they stayed SILENT), so it lands INERT — enabling it is a
    # separate human decision with its own eval pass.
    stale_ask: bool = False  # PERCEPT_GATE_STALE_ASK — perception-recency freshness (opt-in)
    # "already present when the runtime starts -> SILENT" (v0.1 hero matrix): require a confident-false
    # baseline before a key can fire, so a condition already true at start is not a fire. Opt-in; off is
    # byte-for-byte the historical rising-edge gate (fire on first true).
    require_false_first: bool = False  # PERCEPT_GATE_REQUIRE_FALSE_FIRST
    # Refractory suppression (precision track): suppress a re-fire landing within this many seconds of the
    # same key's last real fire, even across a spurious confident-false/confident-true blip mid-episode
    # (the "RE-FIRES" false-positive source found during precision tuning). A WatchProfile-settable knob
    # (``gate.refractory_s`` in the WatchProfile schema); off by default (0/neg = byte-for-byte
    # the historical gate, no refractory).
    refractory_s: float = 0.0  # PERCEPT_GATE_REFRACTORY_S


@dataclass(frozen=True)
class EntitySettings:
    iou_match: float = 0.5  # PERCEPT_IOU_MATCH — frame-to-frame continuity threshold
    max_entities: int = 64  # PERCEPT_MAX_ENTITIES — eviction bound


@dataclass(frozen=True)
class CacheSettings:
    enabled: bool = False  # PERCEPT_CACHE_ENABLED — wire the FrameHashLayer into the cascade
    threshold: float = 0.0  # PERCEPT_CACHE_THRESHOLD — cascade short-circuit cutoff
    ttl_s: float = 8.0  # PERCEPT_CACHE_TTL_S — hard ceiling on slow drift
    tau_bits: int = 6  # PERCEPT_CACHE_TAU_BITS — dHash Hamming near-match radius
    audit_rate: float = 0.05  # PERCEPT_CACHE_AUDIT_RATE — fraction of HITs re-perceived


@dataclass(frozen=True)
class ModelSettings:
    # None (not "fake") is the default: you do NOT land on a scripted backend by NOT typing something.
    # The mode (offline -> scripted) or an explicit selection fills these; a bare LIVE run with them
    # unset raises ModeError instead of silently faking. env (PERCEPT_*_BACKEND) still overrides.
    vision: str | None = None  # PERCEPT_VISION_BACKEND — set claude/gemini in prod, or mode="offline"
    conductor: str | None = None  # PERCEPT_CONDUCTOR_BACKEND
    specialist: str | None = None  # PERCEPT_SPECIALIST_BACKEND
    stt: str = "fake"  # PERCEPT_STT_BACKEND — set deepgram for voice
    tts: str = "fake"  # PERCEPT_TTS_BACKEND
    vision_model: str = "claude-opus-4-8"  # PERCEPT_VISION_MODEL
    conductor_model: str = "claude-haiku-4-5"  # PERCEPT_CONDUCTOR_MODEL
    specialist_model: str = "claude-haiku-4-5"  # PERCEPT_SPECIALIST_MODEL


@dataclass(frozen=True)
class ConsentSettings:
    allow_computer_use: bool = False  # PERCEPT_ALLOW_COMPUTER_USE — gates the `act` modality
    allowed_actions: frozenset = frozenset()  # PERCEPT_ACT_ALLOWED_ACTIONS — comma-separated scope
    # PERCEPT_ACTUATION — ADR-007. "none" (default) | "screen" (advisory software acts)
    # | "physical" (a DECLARATION that this deployment is outside percept's envelope: every act is
    # then refused, because the gate's honest failure mode is ASK and a control loop cannot ask).
    # Typed, so `percept doctor` reports it; see percept.envelope.ACTUATION_CLASSES.
    actuation: str = "none"


@dataclass(frozen=True)
class EventSettings:
    window: int = 12  # PERCEPT_FRAME_WINDOW — rolling frames per source for temporal perception
    # PERCEPT_PREROLL_MAX_BYTES — byte budget for the incident snippet's pre-roll ring
    # (percept/runtime/worker.py). The count bound (`snippet_size`, 48) does NOT bound memory: at
    # 1080p RGB those 48 frames are ~300 MB resident, matching the observed "runaway memory"
    # plateau. 128 MB still covers the default 5 s pre-roll at 4 fps / 1080p; beyond it the
    # OLDEST frames are evicted and every eviction is reported (ADR-001 §9), never silent.
    preroll_max_bytes: int = 134_217_728
    merge_gap_s: float = 1.5  # PERCEPT_EVENT_MERGE_GAP_S — same-type detections within this = 1 event
    audio_window_s: float = 4.0  # PERCEPT_AUDIO_WINDOW_S — rolling seconds of audio per source for fused/acoustic perception


@dataclass(frozen=True)
class IdentitySettings:
    appearance_threshold: float = 0.8  # PERCEPT_APPEARANCE_MATCH — cosine sim to re-id across a gap


@dataclass(frozen=True)
class EpisodeSettings:
    window_s: float = 3600.0  # PERCEPT_EPISODE_WINDOW_S — counting window ("today's cups")
    max_episodes: int = 64  # PERCEPT_MAX_EPISODES — bound on the per-goal episode log


@dataclass(frozen=True)
class EdgeSettings:
    # The on-device / server-side tier-0 salience gate. The coarse
    # motion gate decides *when* to spend a cloud vision call: escalate only when the frame-diff
    # motion score clears this threshold. A shipped default from an auditable sweep, with optional
    # per-goal bias. (PERCEPT_MOTION_GATE)
    motion_gate: float = 0.10  # PERCEPT_MOTION_GATE — motion salience escalation cutoff (0..1)
    # Deployment-wide default salience MODE (tracks the camera mount; overridable per goal, and by a
    # SourceProfile's own ``parameters['salience_stat']``).
    #
    # DEFAULT RATIONALE (pre-registered, no tune-to-green — same discipline as percept/slo.py).
    # A whole-frame `mean` dilutes a small mover against a still background: on a mostly-still scene
    # the average barely moves, so the gate admits nothing and the judge never sees the event at all.
    # A localized statistic keeps the signal where it is strong. Measured on annotated action
    # material at matched economy, `block_max` at its own 0.20 cutoff recovers substantially more of
    # the true events than `mean` at 0.10, and reduces zero-admission scenes by roughly an order of
    # magnitude. `mean` remains available for whole-frame-change watches and byte-for-byte parity
    # with the pre-1.0 gate.
    # Selecting this per goal shape was evaluated against a pre-registered bar and NOT supported —
    # the effect did not survive refitting — so it stays one global default plus the SourceProfile
    # override rather than becoming a per-goal lookup.
    salience_stat: str = "block_max"  # PERCEPT_SALIENCE_STAT — "mean" | "block_max" | "p95"
    ego_compensate: bool = False  # PERCEPT_EGO_COMPENSATE — remove global camera motion before diffing
    # Per-stat escalation cutoffs. block_max/p95 live on a HIGHER value scale than the whole-frame mean
    # (a single active block >= the frame average), so they need their own thresholds — the mean-tuned
    # 0.10 over-fires them. Calibrated on Perception Test action events (N=80, 7.8k frames): block_max
    # @0.20 -> 0.85 recall, p95 @0.15 -> 0.79, vs the mean's 0.06 at 0.10. See docs/admission/wake-policy.md.
    motion_gate_block_max: float = 0.20  # PERCEPT_MOTION_GATE_BLOCK_MAX
    motion_gate_p95: float = 0.15  # PERCEPT_MOTION_GATE_P95
    # Whether a SourceProfile-declared coast floor is bound into the runtime. A profile such as
    # SCREEN_DOCUMENT declares a fixed low-rate coast (``coast_interval_s``) that admits a judge call
    # on a fixed cadence regardless of the tuned salience signal — the floor that keeps an
    # illegible-to-tier-0 document from going wholly unwatched. Strict deployments that want
    # SIGNAL-ONLY admission (judge calls spent only on tuned-signal escalations and audio wakes) set
    # this False via the ``strict_admission`` ExecutionPolicy; the resolver then skips binding the
    # coast contributions and a profile-declared ``coast_required`` is overridden LOUDLY (receipt
    # warning), never silently. Off changes nothing for profiles that declare no coast.
    coast_enabled: bool = True  # PERCEPT_COAST_ENABLED — bind a SourceProfile coast floor into the runtime


@dataclass(frozen=True)
class RoutingSettings:
    # Below this confidence the local router DEFERS to the conductor. Defer-on-miss is
    # structural: a match under the bar is indistinguishable from no match. Raise it to make
    # the fast-path quieter; lower it (carefully) to route more locally. (PERCEPT_ROUTE_THRESHOLD)
    confidence_threshold: float = 0.6


@dataclass(frozen=True)
class Settings:
    gate: GateSettings = GateSettings()
    entity: EntitySettings = EntitySettings()
    cache: CacheSettings = CacheSettings()
    models: ModelSettings = ModelSettings()
    routing: RoutingSettings = RoutingSettings()
    identity: IdentitySettings = IdentitySettings()
    episodes: EpisodeSettings = EpisodeSettings()
    events: EventSettings = EventSettings()
    consent: ConsentSettings = ConsentSettings()
    edge: EdgeSettings = EdgeSettings()
    # Every ``PERCEPT_*`` override this tree DECLINED to adopt, in load order (empty tuple when the
    # environment was clean). Carried on the tree — not merely logged — so ``/health``, the receipt
    # and any caller holding the settings can tell "the operator typo'd this" from "nobody ever
    # configured it", which is otherwise indistinguishable once the default is substituted.
    env_rejections: tuple[_EnvRejection, ...] = ()


def load_settings() -> Settings:
    """Build a Settings tree, overlaying ``PERCEPT_*`` env vars over the defaults.

    Any override that could not be adopted is substituted by its declared default and recorded on
    :attr:`Settings.env_rejections` (as well as announced through ``warnings`` and ``logging``).
    Loading never raises on a bad override: an always-on process must survive a typo'd shell profile.
    """
    ledger: list[_EnvRejection] = []
    token = _ACTIVE_REJECTIONS.set(ledger)
    try:
        tree = _load_settings_tree()
    finally:
        _ACTIVE_REJECTIONS.reset(token)
    return replace(tree, env_rejections=tuple(ledger))


def _load_settings_tree() -> Settings:
    """The env overlay itself — one ``_env`` call per field, no other reads of ``os.environ``."""
    return Settings(
        gate=GateSettings(
            threshold=_env("PERCEPT_GATE_THRESHOLD", 0.7, float, bounds=_UNIT_INTERVAL),
            validity_window_s=_env("PERCEPT_GATE_VALIDITY_S", 90.0, float),
            key_ttl_s=_env("PERCEPT_GATE_KEY_TTL_S", 3600.0, float),
            max_keys=_env("PERCEPT_GATE_MAX_KEYS", 0, int, bounds=_NON_NEGATIVE),
            stale_ask=_env("PERCEPT_GATE_STALE_ASK", False, _as_bool),
            require_false_first=_env("PERCEPT_GATE_REQUIRE_FALSE_FIRST", False, _as_bool),
            refractory_s=_env("PERCEPT_GATE_REFRACTORY_S", 0.0, float),
        ),
        entity=EntitySettings(
            iou_match=_env("PERCEPT_IOU_MATCH", 0.5, float, bounds=_UNIT_INTERVAL),
            max_entities=_env("PERCEPT_MAX_ENTITIES", 64, int, bounds=_AT_LEAST_ONE),
        ),
        cache=CacheSettings(
            enabled=_env("PERCEPT_CACHE_ENABLED", False, _as_bool),
            threshold=_env("PERCEPT_CACHE_THRESHOLD", 0.0, float, bounds=_UNIT_INTERVAL),
            ttl_s=_env("PERCEPT_CACHE_TTL_S", 8.0, float),
            tau_bits=_env("PERCEPT_CACHE_TAU_BITS", 6, int, bounds=(0.0, 64.0)),
            audit_rate=_env("PERCEPT_CACHE_AUDIT_RATE", 0.05, float, bounds=_UNIT_INTERVAL),
        ),
        models=ModelSettings(
            vision=_env("PERCEPT_VISION_BACKEND", None, str),
            conductor=_env("PERCEPT_CONDUCTOR_BACKEND", None, str),
            specialist=_env("PERCEPT_SPECIALIST_BACKEND", None, str),
            stt=_env("PERCEPT_STT_BACKEND", "fake", str),
            tts=_env("PERCEPT_TTS_BACKEND", "fake", str),
            vision_model=_env("PERCEPT_VISION_MODEL", "claude-opus-4-8", str),
            conductor_model=_env("PERCEPT_CONDUCTOR_MODEL", "claude-haiku-4-5", str),
            specialist_model=_env("PERCEPT_SPECIALIST_MODEL", "claude-haiku-4-5", str),
        ),
        routing=RoutingSettings(
            confidence_threshold=_env("PERCEPT_ROUTE_THRESHOLD", 0.6, float, bounds=_UNIT_INTERVAL),
        ),
        identity=IdentitySettings(
            appearance_threshold=_env("PERCEPT_APPEARANCE_MATCH", 0.8, float, bounds=_UNIT_INTERVAL),
        ),
        episodes=EpisodeSettings(
            window_s=_env("PERCEPT_EPISODE_WINDOW_S", 3600.0, float),
            max_episodes=_env("PERCEPT_MAX_EPISODES", 64, int, bounds=_AT_LEAST_ONE),
        ),
        events=EventSettings(
            window=_env("PERCEPT_FRAME_WINDOW", 12, int, bounds=_AT_LEAST_ONE),
            preroll_max_bytes=_env(
                "PERCEPT_PREROLL_MAX_BYTES", 134_217_728, int, bounds=_NON_NEGATIVE
            ),
            merge_gap_s=_env("PERCEPT_EVENT_MERGE_GAP_S", 1.5, float),
            audio_window_s=_env("PERCEPT_AUDIO_WINDOW_S", 4.0, float),
        ),
        consent=ConsentSettings(
            allow_computer_use=_env("PERCEPT_ALLOW_COMPUTER_USE", False, _as_bool),
            allowed_actions=_env("PERCEPT_ACT_ALLOWED_ACTIONS", frozenset(), _as_actions),
            actuation=_env("PERCEPT_ACTUATION", "none", str, choices=ACTUATION_CLASSES),
        ),
        edge=EdgeSettings(
            motion_gate=_env("PERCEPT_MOTION_GATE", 0.10, float, bounds=_UNIT_INTERVAL),
            # An out-of-vocabulary value here is the one config typo that silently changes what the
            # eye ADMITS: worker.py hands it to ``_coerce_salience_stat`` as the *trusted* default,
            # whose last line falls back to "mean", so the deployment runs mean@0.10 where the
            # shipped tree gives block_max@0.20. Refuse it where it is typed.
            salience_stat=_env("PERCEPT_SALIENCE_STAT", "block_max", str, choices=SALIENCE_STATS),
            ego_compensate=_env("PERCEPT_EGO_COMPENSATE", False, _as_bool),
            motion_gate_block_max=_env(
                "PERCEPT_MOTION_GATE_BLOCK_MAX", 0.20, float, bounds=_UNIT_INTERVAL
            ),
            motion_gate_p95=_env("PERCEPT_MOTION_GATE_P95", 0.15, float, bounds=_UNIT_INTERVAL),
            coast_enabled=_env("PERCEPT_COAST_ENABLED", True, _as_bool),
        ),
    )


def as_dict(settings: Any) -> dict:
    """Recursively render a Settings tree to a plain dict (for logging / /health).

    Sequences are walked too, so ``env_rejections`` renders as a list of ``{var, raw, reason,
    substituted}`` records instead of a row of opaque dataclass reprs.
    """
    if is_dataclass(settings) and not isinstance(settings, type):
        return {f.name: as_dict(getattr(settings, f.name)) for f in fields(settings)}
    if isinstance(settings, (tuple, list)):
        return [as_dict(item) for item in settings]
    return settings


__all__ = [
    "Settings",
    "GateSettings",
    "EntitySettings",
    "CacheSettings",
    "ModelSettings",
    "RoutingSettings",
    "IdentitySettings",
    "EpisodeSettings",
    "EventSettings",
    "ConsentSettings",
    "EdgeSettings",
    "ModeError",
    "resolve_mode",
    "missing_credential",
    "load_settings",
    "as_dict",
]
