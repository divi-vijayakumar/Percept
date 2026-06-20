"""The entity graph — remembers the *things* (identity below the LLM, handed up as symbols).

Identity is established here from continuity (IoU + label) and handed up as a stable id; it is
never re-derived from pixels turn to turn (invariant 4). Attributes carry provenance and
freshness, and the load-bearing rule is enforced here: **assertion beats observation** — a user
"this is water" overrides any detection, and an observed value never clobbers an asserted one.

Appearance re-id (recognizing a thing across a tracking gap) is Phase 4; this module ships the
continuity reference and retains an ``appearance`` slot on every entity for forward-compat.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .adapters import Attribute, Box, Observation
from .identity import ContinuityIdentity


@dataclass
class Entity:
    id: str
    label: str
    box: Box
    first_seen: float
    last_seen: float
    attributes: dict[str, Attribute] = field(default_factory=dict)
    sources: set[str] = field(default_factory=set)
    appearance: object | None = None  # Phase 4 re-id signature; retained early for forward-compat

    def area(self) -> float:
        return self.box.w * self.box.h

    def asserted(self) -> dict[str, str]:
        """The user-asserted facts on this entity (the corrections passed to vision.judge)."""
        return {k: a.value for k, a in self.attributes.items() if a.source == "asserted"}


class EntityGraph:
    """Continuity-based entity tracking + attribute provenance/freshness."""

    def __init__(
        self,
        iou_match: float = 0.5,
        max_entities: int = 64,
        clock: Callable[[], float] | None = None,
        *,
        identity: Any = None,
        embedder: Callable[[Any, Box], Any] | None = None,
    ) -> None:
        self._iou = iou_match
        self._max = max_entities
        self._clock = clock or time.monotonic
        # The matcher decides "same thing?" — continuity by default; AppearanceIdentity re-ids
        # across gaps when an embedder is supplied. Either way, ids are assigned here, below the LLM.
        self._identity = identity if identity is not None else ContinuityIdentity(iou_match)
        self._embedder = embedder
        self._entities: dict[str, Entity] = {}
        self._counter = 0
        self._last_frame_ids: list[str] = []

    # -- ingestion -----------------------------------------------------------
    def ingest(
        self,
        observations: list[Observation],
        frame: object = None,
        source: str = "camera",
        *,
        t: float | None = None,
    ) -> list[Entity]:
        """Match observations to entities by IoU+label; mint/update; record observed attrs."""
        now = self._clock() if t is None else t
        touched: list[str] = []
        for obs in observations:
            signature = self._embedder(frame, obs.box) if self._embedder is not None else None
            match_id = self._match_id(obs, signature)
            entity = self._entities.get(match_id) if match_id else None
            if entity is None:
                entity = self._mint(obs, now, source)
            else:
                entity.box = obs.box
                entity.last_seen = now
                entity.sources.add(source)
            if signature is not None:
                entity.appearance = signature  # retain on every entity (forward-compat re-id)
            self._record_observed(entity, obs, now)
            touched.append(entity.id)
        self._last_frame_ids = touched
        self._evict()
        return [self._entities[i] for i in touched]

    def _match_id(self, obs: Observation, signature: Any) -> str | None:
        """Ask the identity strategy for the matching entity id (None -> mint a new one)."""
        known = list(self._entities.values())
        try:
            return self._identity.match(obs, known, signature=signature)
        except TypeError:
            # a custom Identity that conforms only to the frozen match(obs, known) signature
            return self._identity.match(obs, known)

    def _mint(self, obs: Observation, now: float, source: str) -> Entity:
        self._counter += 1
        eid = f"e{self._counter}"
        entity = Entity(
            id=eid,
            label=obs.label,
            box=obs.box,
            first_seen=now,
            last_seen=now,
            sources={source},
        )
        self._entities[eid] = entity
        return entity

    def _record_observed(self, entity: Entity, obs: Observation, now: float) -> None:
        for key, value in obs.attrs.items():
            existing = entity.attributes.get(key)
            if existing is not None and existing.source == "asserted":
                continue  # assertion beats observation — never clobber a user fact
            entity.attributes[key] = Attribute(
                value=value, source="observed", confidence=obs.confidence, last_confirmed=now
            )

    def _evict(self) -> None:
        while len(self._entities) > self._max:
            oldest = min(self._entities.values(), key=lambda e: e.last_seen)
            del self._entities[oldest.id]

    # -- assertions / fusion -------------------------------------------------
    def assert_attribute(
        self, entity_id: str, key: str, value: str, *, t: float | None = None
    ) -> None:
        """Bind a user-asserted fact to an entity (beats observation, fully confident)."""
        now = self._clock() if t is None else t
        entity = self._entities.get(entity_id)
        if entity is None:
            raise KeyError(f"no entity {entity_id!r}")
        entity.attributes[key] = Attribute(
            value=value, source="asserted", confidence=1.0, last_confirmed=now
        )

    def note_mention(self, text: str, *, t: float | None = None) -> str | None:
        """Fuse a voice mention onto the in-focus entity (voice -> visible-entity).

        Returns the entity id the mention was bound to, or None if nothing is in focus.
        """
        focus = self.in_focus()
        if focus is None:
            return None
        now = self._clock() if t is None else t
        focus.attributes["mention"] = Attribute(
            value=text, source="asserted", confidence=1.0, last_confirmed=now
        )
        return focus.id

    # -- queries -------------------------------------------------------------
    def in_focus(self) -> Entity | None:
        """The largest entity in the most recent frame (the deictic 'this')."""
        candidates = [self._entities[i] for i in self._last_frame_ids if i in self._entities]
        if not candidates:
            return None
        return max(candidates, key=lambda e: e.area())

    def get(self, entity_id: str) -> Entity | None:
        return self._entities.get(entity_id)

    def active(self) -> list[Entity]:
        return list(self._entities.values())

    def __len__(self) -> int:
        return len(self._entities)


__all__ = ["Entity", "EntityGraph"]
