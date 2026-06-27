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
    # Motion model for predictive continuity. ``vel`` is the box's (centre-x, centre-y, w, h) velocity per
    # second, estimated from successive OBSERVED boxes; the anchor is the last observed box + its time, so a
    # prediction can be made for any later moment. This is what lets continuity survive a fast displacement:
    # match against where the subject *will be*, not its stale last box (the memoryless-IoU failure that
    # fragments a moving hero).
    vel: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    anchor_box: Box | None = None
    anchor_t: float = 0.0

    def area(self) -> float:
        return self.box.w * self.box.h

    def predicted_box(self, now: float) -> Box:
        """Where the box should be at ``now``, advancing the last observed box by its velocity (constant
        velocity). Falls back to the current box when there is no motion anchor yet."""
        if self.anchor_box is None:
            return self.box
        dt = now - self.anchor_t
        if dt <= 0:
            return self.anchor_box
        a, (vcx, vcy, vw, vh) = self.anchor_box, self.vel
        w = max(0.02, a.w + vw * dt)
        h = max(0.02, a.h + vh * dt)
        cx = a.x + a.w / 2 + vcx * dt
        cy = a.y + a.h / 2 + vcy * dt
        return Box(cx - w / 2, cy - h / 2, w, h)

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
        vel_alpha: float = 0.5,
        predict_iou: float = 0.3,
        coast_s: float = 1.0,
    ) -> None:
        self._iou = iou_match
        self._max = max_entities
        self._clock = clock or time.monotonic
        # The matcher decides "same thing?" — continuity by default; AppearanceIdentity re-ids
        # across gaps when an embedder is supplied. Either way, ids are assigned here, below the LLM.
        self._identity = identity if identity is not None else ContinuityIdentity(iou_match)
        self._embedder = embedder
        # Predictive continuity: each entity learns a velocity from successive observations, and when normal
        # continuity+appearance fails to match, the recovery path re-links to where the entity is PREDICTED
        # to be. AUGMENT-only (it never overrides a successful match) and self-disabling on slow/static
        # objects (velocity ~0 -> predicted box == last box), so it costs nothing until a subject moves fast.
        self._vel_alpha = vel_alpha
        self._predict_iou = predict_iou  # min IoU(predicted, obs) for the recovery match
        self._coast_s = coast_s          # don't predict-match a track unseen longer than this (stale)
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
            if match_id is None:
                # AUGMENT, don't replace: only when normal continuity+appearance fails do we try where the
                # entity is PREDICTED to be. So a held subject (crowd) is untouched; a fast displacement that
                # broke memoryless IoU re-links instead of minting a new id.
                match_id = self._predict_match(obs, now, exclude=set(touched))
            entity = self._entities.get(match_id) if match_id else None
            if entity is None:
                entity = self._mint(obs, now, source)
                entity.anchor_box, entity.anchor_t = obs.box, now
            else:
                self._update_motion(entity, obs.box, now)
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

    def _predict_match(self, obs: Observation, now: float, *, exclude: set) -> str | None:
        """Recovery path (motion mode): match ``obs`` against each entity's PREDICTED box (constant
        velocity), so a fast displacement re-links to the same id. Coasts only within ``_coast_s`` of the
        last real observation — a stale track doesn't reach out and grab a passer-by."""
        best, best_iou = None, self._predict_iou
        for e in self._entities.values():
            if e.id in exclude or e.label != obs.label or e.anchor_box is None:
                continue
            if now - e.anchor_t > self._coast_s:
                continue
            iou = e.predicted_box(now).iou(obs.box)
            if iou >= best_iou:
                best, best_iou = e, iou
        return best.id if best is not None else None

    def _update_motion(self, entity: Entity, new_box: Box, now: float) -> None:
        """Re-estimate the entity's velocity from its last observed box → this one, then re-anchor."""
        a = entity.anchor_box
        dt = now - entity.anchor_t
        if a is not None and dt > 0:
            measured = (
                (new_box.x + new_box.w / 2 - (a.x + a.w / 2)) / dt,
                (new_box.y + new_box.h / 2 - (a.y + a.h / 2)) / dt,
                (new_box.w - a.w) / dt,
                (new_box.h - a.h) / dt,
            )
            al = self._vel_alpha
            entity.vel = tuple(al * m + (1 - al) * v for m, v in zip(measured, entity.vel))
        entity.anchor_box, entity.anchor_t = new_box, now

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

    # -- persistence (durable entities; rehydrate on reconnect) --------------
    def snapshot(self) -> list[dict]:
        return [
            {
                "id": e.id,
                "label": e.label,
                "box": {"x": e.box.x, "y": e.box.y, "w": e.box.w, "h": e.box.h},
                "first_seen": e.first_seen,
                "last_seen": e.last_seen,
                "attributes": {
                    k: {
                        "value": a.value,
                        "source": a.source,
                        "confidence": a.confidence,
                        "last_confirmed": a.last_confirmed,
                    }
                    for k, a in e.attributes.items()
                },
                "sources": sorted(e.sources),
                "appearance": list(e.appearance) if e.appearance is not None else None,
            }
            for e in self._entities.values()
        ]

    def restore(self, dumps: list[dict]) -> None:
        self._entities = {}
        self._last_frame_ids = []
        max_n = 0
        for d in dumps:
            entity = Entity(
                id=d["id"],
                label=d["label"],
                box=Box(**d["box"]),
                first_seen=d["first_seen"],
                last_seen=d["last_seen"],
                attributes={k: Attribute(**a) for k, a in d.get("attributes", {}).items()},
                sources=set(d.get("sources", [])),
                appearance=d.get("appearance"),
            )
            self._entities[entity.id] = entity
            if entity.id.startswith("e") and entity.id[1:].isdigit():
                max_n = max(max_n, int(entity.id[1:]))
        self._counter = max_n  # new mints continue past the restored ids


__all__ = ["Entity", "EntityGraph"]
