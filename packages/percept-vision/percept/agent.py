"""The Percept orchestrator — the reactive watch loop (Phase 1).

One firing path: for each active goal, judge the plain-language condition over the in-focus
ROI (with any user-asserted corrections), run the verdict through the three-state gate, and emit
fires/asks on the rising edge. Later phases add *beside* this — the spoken turn (Phase 3),
episodes (Phase 4), events (Phase 5) — without editing the gate/graph contracts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from . import registry
from .adapters import Box, Goal, Verdict
from .cache import CacheCascade
from .config import Settings, load_settings
from .entity import EntityGraph
from .gate import ASK, FIRE, Gate, GateAction
from .goals import GoalOp, GoalRegistry
from .observability import Metrics, health


@dataclass
class FireEvent:
    """What the watch loop emits when the gate acts."""

    goal_id: str
    action: GateAction  # "fire" | "ask"
    text: str
    entity_id: str | None = None
    verdict: Verdict | None = None


def _resolve(seam: str, value: Any, cfg: Settings):
    """Accept either a backend *name* (resolved via the registry) or an adapter instance."""
    if isinstance(value, str):
        return registry.make(seam, value, cfg)
    return value  # already an adapter (handy for tests / custom wiring)


@dataclass
class Percept:
    """The cognition core: backends behind the seams + the memory structures + the gate."""

    vision: Any
    specialist: Any
    conductor: Any
    stt: Any
    tts: Any
    settings: Settings
    metrics: Metrics
    graph: EntityGraph
    gate: Gate
    goals: GoalRegistry
    cache: CacheCascade
    clock: Callable[[], float] = field(default=time.monotonic)

    # -- construction --------------------------------------------------------
    @classmethod
    def create(
        cls,
        *,
        vision: Any = "fake",
        specialist: Any = "fake",
        conductor: Any = "fake",
        stt: Any = "fake",
        tts: Any = "fake",
        settings: Settings | None = None,
        clock: Callable[[], float] | None = None,
        discover_plugins: bool = True,
    ) -> "Percept":
        cfg = settings or load_settings()
        if discover_plugins:
            registry.discover_entrypoints()
        clk = clock or time.monotonic
        return cls(
            vision=_resolve("vision", vision, cfg),
            specialist=_resolve("specialist", specialist, cfg),
            conductor=_resolve("conductor", conductor, cfg),
            stt=_resolve("stt", stt, cfg),
            tts=_resolve("tts", tts, cfg),
            settings=cfg,
            metrics=Metrics(),
            graph=EntityGraph(cfg.entity.iou_match, cfg.entity.max_entities, clock=clk),
            gate=Gate(threshold=cfg.gate.threshold),
            goals=GoalRegistry(),
            cache=CacheCascade(layers=[], threshold=cfg.cache.threshold),
            clock=clk,
        )

    # -- goal authoring ------------------------------------------------------
    def add_goal(self, goal: Goal) -> Goal:
        return self.goals.create(goal)

    def apply_goal_ops(self, ops: list[GoalOp]) -> None:
        self.goals.apply(ops)

    # -- the watch path ------------------------------------------------------
    async def perceive_judged(
        self, frame: Any, source: str = "camera", *, t: float | None = None
    ) -> list[FireEvent]:
        """Judge every active goal against ``frame`` and emit gate actions on rising edges."""
        now = self.clock() if t is None else t
        observations = await self.vision.detect(frame, self.goals.active())
        self.graph.ingest(observations, frame, source, t=now)
        focus = self.graph.in_focus()

        events: list[FireEvent] = []
        for goal in self.goals.active():
            roi: Box | None = focus.box if focus else None
            corrections = focus.asserted() if focus else {}
            verdict = await self._judge(goal, frame, roi, corrections)
            freshness = self._freshness(focus, now)
            key = (goal.id, focus.id if focus else None)
            action = self.gate.evaluate(key, verdict, freshness=freshness)
            self.metrics.incr(f"gate.{action}")
            if action == FIRE:
                events.append(FireEvent(goal.id, FIRE, goal.say, key[1], verdict))
            elif action == ASK:
                events.append(FireEvent(goal.id, ASK, self._ask_text(goal), key[1], verdict))
        return events

    async def _judge(
        self, goal: Goal, frame: Any, roi: Box | None, corrections: dict[str, str]
    ) -> Verdict:
        cache_key = (goal.id, str(frame))
        hit = self.cache.resolve(cache_key)
        if hit.status == "HIT" and hit.verdict is not None:
            self.metrics.incr("cache.hit")
            return hit.verdict
        verdict = await self.vision.judge(
            goal.condition, frame, roi=roi, corrections=corrections or None
        )
        self.cache.store(cache_key, verdict, verdict.confidence)
        return verdict

    def _freshness(self, focus, now: float) -> float:
        """1.0 if no asserted fact or it's still within the validity window, else 0.0 (stale)."""
        if focus is None:
            return 1.0
        confirmed = [a.last_confirmed for a in focus.attributes.values() if a.source == "asserted"]
        if not confirmed:
            return 1.0
        age = now - max(confirmed)
        return 1.0 if age < self.settings.gate.validity_window_s else 0.0

    def _ask_text(self, goal: Goal) -> str:
        return f'About "{goal.condition}" — can you confirm?'

    # -- corrections (voice -> memory) --------------------------------------
    def assert_attribute(
        self, key: str, value: str, *, entity_id: str | None = None, t: float | None = None
    ) -> str | None:
        """Bind a user correction to an entity (defaults to in-focus), then re-open judgment.

        Asserted beats observed; the goal's edge state is cleared so the corrected fact is
        re-judged rather than suppressed by a stale rising edge, and the cache is invalidated.
        """
        now = self.clock() if t is None else t
        target = entity_id
        if target is None:
            focus = self.graph.in_focus()
            target = focus.id if focus else None
        if target is None:
            return None
        self.graph.assert_attribute(target, key, value, t=now)
        for goal in self.goals.active():
            self.gate.reset_goal(goal.id)  # a correction re-opens the edge
        # cache invalidation: empty cascade is a no-op today; FrameHashLayer (Phase 2) honors it
        self.cache = CacheCascade(layers=self.cache.layers, threshold=self.cache.threshold)
        return target

    def note_mention(self, text: str, *, t: float | None = None) -> str | None:
        """Fuse a spoken mention onto the in-focus entity (voice -> visible-entity)."""
        now = self.clock() if t is None else t
        return self.graph.note_mention(text, t=now)

    # -- introspection -------------------------------------------------------
    def health(self) -> dict:
        backends = {
            "vision": type(self.vision).__name__,
            "specialist": type(self.specialist).__name__,
            "conductor": type(self.conductor).__name__,
            "stt": type(self.stt).__name__,
            "tts": type(self.tts).__name__,
        }
        from . import __version__

        return health(__version__, backends, self.metrics)


__all__ = ["Percept", "FireEvent"]
