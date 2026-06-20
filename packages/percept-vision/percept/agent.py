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
from .cache import CacheCascade, CacheKey, FrameHashLayer
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


class DeterministicAuditor:
    """Decides which cache HITs to re-perceive (false-serve audit) — **no RNG** (rule 2).

    False-serve is the cache's analog of false-fire (spec §6.2): re-perceive an
    ``audit_rate`` fraction of HITs through real vision and compare; a divergence is a
    **false-serve** and the fresh read wins. The decision is a deterministic *counter* so the L1
    lane is reproducible: with ``audit_rate = r`` we audit whenever the running HIT count crosses
    the next ``1/r`` boundary (a stratified 1-in-N, not a coin flip). ``r <= 0`` never audits;
    ``r >= 1`` audits every HIT. Inject a custom predicate for other policies.
    """

    def __init__(self, audit_rate: float) -> None:
        self._rate = audit_rate
        self._hits = 0
        self._audited = 0

    def __call__(self, key: object) -> bool:
        if self._rate <= 0.0:
            self._hits += 1
            return False
        if self._rate >= 1.0:
            self._hits += 1
            self._audited += 1
            return True
        self._hits += 1
        # audit when the ideal audited-count (hits*rate) has advanced past what we've done.
        if self._hits * self._rate >= self._audited + 1:
            self._audited += 1
            return True
        return False


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
    # Which cache HITs to re-perceive for false-serve auditing. Deterministic by default
    # (no RNG on the L1 path); injectable for custom policies. ``None`` -> never audit.
    audit: Callable[[object], bool] | None = None

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
        audit: Callable[[object], bool] | None = None,
    ) -> "Percept":
        cfg = settings or load_settings()
        if discover_plugins:
            registry.discover_entrypoints()
        clk = clock or time.monotonic
        # Cache ships a contract, not a policy (invariant 6): the cascade is EMPTY (always DEFERs)
        # unless the FrameHashLayer is explicitly enabled via settings — then it sits in front of
        # the VLM. The false-serve auditor defaults to deterministic (no RNG on the L1 path).
        layers = (
            [FrameHashLayer(cfg.cache, clock=clk)] if cfg.cache.enabled else []
        )
        auditor = audit if audit is not None else DeterministicAuditor(cfg.cache.audit_rate)
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
            cache=CacheCascade(layers=layers, threshold=cfg.cache.threshold),
            clock=clk,
            audit=auditor,
        )

    # -- goal authoring ------------------------------------------------------
    def add_goal(self, goal: Goal) -> Goal:
        return self.goals.create(goal)

    def apply_goal_ops(self, ops: list[GoalOp]) -> None:
        self.goals.apply(ops)
        # cache invalidation — goal-change (1st in the order): a goal whose condition/lifecycle
        # changed must not be served from a verdict cached under its old meaning.
        for op in ops:
            gid = op.goal.id if op.op == "create" and op.goal is not None else op.goal_id
            if op.op in ("update", "complete", "cancel") and gid is not None:
                self.invalidate_cache(gid)

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
            verdict = await self._judge(
                goal, frame, roi, corrections, entity_id=focus.id if focus else None
            )
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
        self,
        goal: Goal,
        frame: Any,
        roi: Box | None,
        corrections: dict[str, str],
        *,
        entity_id: str | None = None,
    ) -> Verdict:
        # A structured key so the FrameHashLayer can ROI-crop + dHash the frame and scope its
        # entry to (goal, entity); the legacy stub cascade ignores the extra fields harmlessly.
        cache_key = CacheKey(
            goal_id=goal.id, frame=frame, roi=roi, entity_id=entity_id
        )
        hit = self.cache.resolve(cache_key)
        if hit.status == "HIT" and hit.verdict is not None:
            self.metrics.incr("cache.hit")
            return await self._audit_or_use(goal, frame, roi, corrections, cache_key, hit.verdict)
        verdict = await self.vision.judge(
            goal.condition, frame, roi=roi, corrections=corrections or None
        )
        self.cache.store(cache_key, verdict, verdict.confidence)
        return verdict

    async def _audit_or_use(
        self,
        goal: Goal,
        frame: Any,
        roi: Box | None,
        corrections: dict[str, str],
        cache_key: Any,
        cached: Verdict,
    ) -> Verdict:
        """Trust-but-verify a cache HIT (spec §4.1, §6.2).

        On the audited fraction (decided by ``self.audit`` — deterministic, no RNG on the L1
        path), re-perceive through real vision and compare to the cached verdict. A divergence in
        the ``satisfied`` bit is a **false-serve**: the fresh read wins and replaces the stale
        cache entry. Un-audited HITs are served as-is. ``audit is None`` -> never audit.
        """
        if self.audit is None or not self.audit(cache_key):
            return cached
        self.metrics.incr("cache.audit")
        fresh = await self.vision.judge(
            goal.condition, frame, roi=roi, corrections=corrections or None
        )
        if fresh.satisfied != cached.satisfied:
            self.metrics.incr("cache.false_serve")  # the cache's analog of false-fire
            self.cache.store(cache_key, fresh, fresh.confidence)  # fresh read wins
            return fresh
        return cached

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
        # cache invalidation — assertion-override (2nd in the order goal-change -> assertion ->
        # TTL -> L0): drop every cached verdict resting on the corrected entity so the asserted
        # fact is re-judged, never served stale. Empty/stub cascades are a harmless no-op.
        self.invalidate_cache_entity(target)
        return target

    def invalidate_cache(self, goal_id: str | None = None) -> None:
        """Goal-change invalidation: drop a goal's cached verdicts (``None`` clears all)."""
        for layer in self.cache.layers:
            inv = getattr(layer, "invalidate", None)
            if callable(inv):
                inv(goal_id)

    def invalidate_cache_entity(self, entity_id: str | None) -> None:
        """Assertion-override invalidation: drop cached verdicts resting on an entity."""
        for layer in self.cache.layers:
            inv = getattr(layer, "invalidate_entity", None)
            if callable(inv):
                inv(entity_id)

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
