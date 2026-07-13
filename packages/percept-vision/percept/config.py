"""Settings tree — every tunable env-overridable, no literals in logic (standing rule 5).

Read once via :func:`load_settings`. Each field maps to a ``PERCEPT_*`` env var so behaviour
is reproducible from the environment and nothing is hard-coded in the cognition logic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, is_dataclass
from typing import Any


def _env(name: str, default: Any, cast) -> Any:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return cast(raw)
    except (TypeError, ValueError):
        return default


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
    vision: str = "fake"  # PERCEPT_VISION_BACKEND — set claude/gemini in prod
    conductor: str = "fake"  # PERCEPT_CONDUCTOR_BACKEND
    specialist: str = "fake"  # PERCEPT_SPECIALIST_BACKEND
    stt: str = "fake"  # PERCEPT_STT_BACKEND — set deepgram for voice
    tts: str = "fake"  # PERCEPT_TTS_BACKEND
    vision_model: str = "claude-opus-4-8"  # PERCEPT_VISION_MODEL
    conductor_model: str = "claude-haiku-4-5"  # PERCEPT_CONDUCTOR_MODEL
    specialist_model: str = "claude-haiku-4-5"  # PERCEPT_SPECIALIST_MODEL


@dataclass(frozen=True)
class ConsentSettings:
    allow_computer_use: bool = False  # PERCEPT_ALLOW_COMPUTER_USE — gates the `act` modality
    allowed_actions: frozenset = frozenset()  # PERCEPT_ACT_ALLOWED_ACTIONS — comma-separated scope


@dataclass(frozen=True)
class EventSettings:
    window: int = 12  # PERCEPT_FRAME_WINDOW — rolling frames per source for temporal perception
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
    # The on-device / server-side tier-0 salience gate (spec §6.3, impl plan §8). The coarse
    # motion gate decides *when* to spend a cloud vision call: escalate only when the frame-diff
    # motion score clears this threshold. A shipped default from an auditable sweep, with optional
    # per-goal bias. (PERCEPT_MOTION_GATE)
    motion_gate: float = 0.10  # PERCEPT_MOTION_GATE — motion salience escalation cutoff (0..1)
    # Deployment-wide default salience MODE (tracks the camera mount; overridable per goal). Default
    # "mean" / False preserve the legacy whole-frame gate byte-for-byte.
    salience_stat: str = "mean"  # PERCEPT_SALIENCE_STAT — "mean" | "block_max" | "p95" (localized recall)
    ego_compensate: bool = False  # PERCEPT_EGO_COMPENSATE — remove global camera motion before diffing
    # Per-stat escalation cutoffs. block_max/p95 live on a HIGHER value scale than the whole-frame mean
    # (a single active block >= the frame average), so they need their own thresholds — the mean-tuned
    # 0.10 over-fires them. Calibrated on Perception Test action events (N=80, 7.8k frames): block_max
    # @0.20 -> 0.85 recall, p95 @0.15 -> 0.79, vs the mean's 0.06 at 0.10. See docs/reference/edge-policy.
    motion_gate_block_max: float = 0.20  # PERCEPT_MOTION_GATE_BLOCK_MAX
    motion_gate_p95: float = 0.15  # PERCEPT_MOTION_GATE_P95


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


def load_settings() -> Settings:
    """Build a Settings tree, overlaying ``PERCEPT_*`` env vars over the defaults."""
    return Settings(
        gate=GateSettings(
            threshold=_env("PERCEPT_GATE_THRESHOLD", 0.7, float),
            validity_window_s=_env("PERCEPT_GATE_VALIDITY_S", 90.0, float),
            key_ttl_s=_env("PERCEPT_GATE_KEY_TTL_S", 3600.0, float),
            max_keys=_env("PERCEPT_GATE_MAX_KEYS", 0, int),
            stale_ask=_env("PERCEPT_GATE_STALE_ASK", False, _as_bool),
        ),
        entity=EntitySettings(
            iou_match=_env("PERCEPT_IOU_MATCH", 0.5, float),
            max_entities=_env("PERCEPT_MAX_ENTITIES", 64, int),
        ),
        cache=CacheSettings(
            enabled=_env("PERCEPT_CACHE_ENABLED", False, _as_bool),
            threshold=_env("PERCEPT_CACHE_THRESHOLD", 0.0, float),
            ttl_s=_env("PERCEPT_CACHE_TTL_S", 8.0, float),
            tau_bits=_env("PERCEPT_CACHE_TAU_BITS", 6, int),
            audit_rate=_env("PERCEPT_CACHE_AUDIT_RATE", 0.05, float),
        ),
        models=ModelSettings(
            vision=_env("PERCEPT_VISION_BACKEND", "fake", str),
            conductor=_env("PERCEPT_CONDUCTOR_BACKEND", "fake", str),
            specialist=_env("PERCEPT_SPECIALIST_BACKEND", "fake", str),
            stt=_env("PERCEPT_STT_BACKEND", "fake", str),
            tts=_env("PERCEPT_TTS_BACKEND", "fake", str),
            vision_model=_env("PERCEPT_VISION_MODEL", "claude-opus-4-8", str),
            conductor_model=_env("PERCEPT_CONDUCTOR_MODEL", "claude-haiku-4-5", str),
            specialist_model=_env("PERCEPT_SPECIALIST_MODEL", "claude-haiku-4-5", str),
        ),
        routing=RoutingSettings(
            confidence_threshold=_env("PERCEPT_ROUTE_THRESHOLD", 0.6, float),
        ),
        identity=IdentitySettings(
            appearance_threshold=_env("PERCEPT_APPEARANCE_MATCH", 0.8, float),
        ),
        episodes=EpisodeSettings(
            window_s=_env("PERCEPT_EPISODE_WINDOW_S", 3600.0, float),
            max_episodes=_env("PERCEPT_MAX_EPISODES", 64, int),
        ),
        events=EventSettings(
            window=_env("PERCEPT_FRAME_WINDOW", 12, int),
            merge_gap_s=_env("PERCEPT_EVENT_MERGE_GAP_S", 1.5, float),
            audio_window_s=_env("PERCEPT_AUDIO_WINDOW_S", 4.0, float),
        ),
        consent=ConsentSettings(
            allow_computer_use=_env("PERCEPT_ALLOW_COMPUTER_USE", False, _as_bool),
            allowed_actions=_env("PERCEPT_ACT_ALLOWED_ACTIONS", frozenset(), _as_actions),
        ),
        edge=EdgeSettings(
            motion_gate=_env("PERCEPT_MOTION_GATE", 0.10, float),
            salience_stat=_env("PERCEPT_SALIENCE_STAT", "mean", str),
            ego_compensate=_env("PERCEPT_EGO_COMPENSATE", False, _as_bool),
            motion_gate_block_max=_env("PERCEPT_MOTION_GATE_BLOCK_MAX", 0.20, float),
            motion_gate_p95=_env("PERCEPT_MOTION_GATE_P95", 0.15, float),
        ),
    )


def as_dict(settings: Any) -> dict:
    """Recursively render a Settings tree to a plain dict (for logging / /health)."""
    if is_dataclass(settings) and not isinstance(settings, type):
        return {f.name: as_dict(getattr(settings, f.name)) for f in fields(settings)}
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
    "load_settings",
    "as_dict",
]
