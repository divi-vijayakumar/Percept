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


@dataclass(frozen=True)
class GateSettings:
    threshold: float = 0.7  # PERCEPT_GATE_THRESHOLD — confident-yes cutoff
    validity_window_s: float = 90.0  # PERCEPT_GATE_VALIDITY_S — asserted-fact freshness


@dataclass(frozen=True)
class EntitySettings:
    iou_match: float = 0.5  # PERCEPT_IOU_MATCH — frame-to-frame continuity threshold
    max_entities: int = 64  # PERCEPT_MAX_ENTITIES — eviction bound


@dataclass(frozen=True)
class CacheSettings:
    threshold: float = 0.0  # PERCEPT_CACHE_THRESHOLD — cascade short-circuit cutoff
    ttl_s: float = 8.0  # PERCEPT_CACHE_TTL_S — hard ceiling on slow drift
    tau_bits: int = 6  # PERCEPT_CACHE_TAU_BITS — dHash Hamming near-match radius
    audit_rate: float = 0.05  # PERCEPT_CACHE_AUDIT_RATE — fraction of HITs re-perceived


@dataclass(frozen=True)
class ModelSettings:
    vision: str = "claude"  # PERCEPT_VISION_BACKEND
    conductor: str = "claude"  # PERCEPT_CONDUCTOR_BACKEND
    vision_model: str = "claude-opus-4-8"  # PERCEPT_VISION_MODEL
    conductor_model: str = "claude-haiku-4-5"  # PERCEPT_CONDUCTOR_MODEL


@dataclass(frozen=True)
class Settings:
    gate: GateSettings = GateSettings()
    entity: EntitySettings = EntitySettings()
    cache: CacheSettings = CacheSettings()
    models: ModelSettings = ModelSettings()


def load_settings() -> Settings:
    """Build a Settings tree, overlaying ``PERCEPT_*`` env vars over the defaults."""
    return Settings(
        gate=GateSettings(
            threshold=_env("PERCEPT_GATE_THRESHOLD", 0.7, float),
            validity_window_s=_env("PERCEPT_GATE_VALIDITY_S", 90.0, float),
        ),
        entity=EntitySettings(
            iou_match=_env("PERCEPT_IOU_MATCH", 0.5, float),
            max_entities=_env("PERCEPT_MAX_ENTITIES", 64, int),
        ),
        cache=CacheSettings(
            threshold=_env("PERCEPT_CACHE_THRESHOLD", 0.0, float),
            ttl_s=_env("PERCEPT_CACHE_TTL_S", 8.0, float),
            tau_bits=_env("PERCEPT_CACHE_TAU_BITS", 6, int),
            audit_rate=_env("PERCEPT_CACHE_AUDIT_RATE", 0.05, float),
        ),
        models=ModelSettings(
            vision=_env("PERCEPT_VISION_BACKEND", "claude", str),
            conductor=_env("PERCEPT_CONDUCTOR_BACKEND", "claude", str),
            vision_model=_env("PERCEPT_VISION_MODEL", "claude-opus-4-8", str),
            conductor_model=_env("PERCEPT_CONDUCTOR_MODEL", "claude-haiku-4-5", str),
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
    "load_settings",
    "as_dict",
]
