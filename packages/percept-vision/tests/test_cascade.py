"""L1: cache cascade composition — empty defers, HIT>=threshold short-circuits.

Plus, below, the ONE place that walks every documented condition under which the verdict cache is
bypassed. The cache is a cost optimisation with a safety contract: a bypass costs money (loud, in
``cache.hit``), a *wrong serve* costs correctness (silent). Twelve separate conditions bypass, spread
across three modules, and no single test ever accounted for all of them — so a condition could quietly
stop bypassing and nothing would notice. ``TestTheTwelveBypassConditions`` is that accounting.
"""

from __future__ import annotations

import asyncio
import unittest
import warnings
from dataclasses import dataclass
from typing import Any, Callable

from percept import UndrivenBehaviorRouteWarning
from percept.adapters import Goal, Verdict
from percept.agent import Percept
from percept.cache import CacheCascade, CacheResult
from percept.cache.framehash import CacheKey, FrameHashLayer, dhash, hamming
from percept.config import CacheSettings, GateSettings, Settings
from percept.direct_judge import DirectJudge, DirectResult, DirectRouting
from percept.source_profiles_default import PHYSICAL_SCENE


class _StubLayer:
    """A scripted cache layer for testing cascade control flow (not intelligence).

    ``conf=None`` -> DEFER; otherwise a HIT stamped with this layer's own name (mirroring how a
    real layer tags ``source_layer`` on its own result).
    """

    def __init__(self, name, conf):
        self.name = name
        self._conf = conf
        self.lookups = 0

    def lookup(self, key):
        self.lookups += 1
        if self._conf is None:
            return CacheResult.defer()
        return CacheResult(
            status="HIT",
            verdict=Verdict(satisfied=True, confidence=self._conf),
            confidence=self._conf,
            source_layer=self.name,
        )

    def store(self, key, verdict, confidence):
        pass


class TestCascade(unittest.TestCase):
    def test_empty_cascade_defers(self):
        cascade = CacheCascade(layers=[], threshold=0.5)
        self.assertEqual(cascade.resolve("k").status, "DEFER")

    def test_hit_above_threshold_short_circuits(self):
        first = _StubLayer("a", 0.9)
        second = _StubLayer("b", 0.95)
        cascade = CacheCascade(layers=[first, second], threshold=0.5)
        result = cascade.resolve("k")
        self.assertEqual(result.status, "HIT")
        self.assertEqual(result.source_layer, "a")
        self.assertEqual(second.lookups, 0)  # short-circuited

    def test_hit_below_threshold_falls_through_to_defer(self):
        only = _StubLayer("a", 0.3)
        cascade = CacheCascade(layers=[only], threshold=0.5)
        self.assertEqual(cascade.resolve("k").status, "DEFER")

    def test_below_threshold_then_above_uses_second(self):
        first = _StubLayer("a", 0.3)
        second = _StubLayer("b", 0.8)
        cascade = CacheCascade(layers=[first, second], threshold=0.5)
        result = cascade.resolve("k")
        self.assertEqual(result.status, "HIT")
        self.assertEqual(result.source_layer, "b")
        self.assertEqual(first.lookups, 1)

    def test_all_defer_defers(self):
        a = _StubLayer("a", None)
        b = _StubLayer("b", None)
        cascade = CacheCascade(layers=[a, b], threshold=0.0)
        self.assertEqual(cascade.resolve("k").status, "DEFER")


# ==================================================================================================
# The twelve bypass conditions, as one matrix.
#
# WHICH FENCE THIS EXERCISES. The judging test spec names ``agent.py:3478-3519`` as "the single
# fence". On this commit there are two structurally equivalent fences: that one lives in
# ``prepare_frame_judgment`` (the split/prepared-job pipeline), and a second lives in
# ``Percept._judge`` (``if frames or audio or process_mode or claim_mode or event_mode or
# force_context or unkeyed_context``). The DEFAULT ``perceive_judged`` path dispatches through
# ``_judge_slot`` -> ``_judge``, so the ``_judge`` fence is the one every case below drives, and it
# is the one a shipped watch actually reaches today.
#
# HOW EACH CASE PROVES ITS BYPASS. Two layers, because the twelve conditions live at two layers:
#
#   * the six cheap ones are decided inside ``percept/cache`` and are proved against a real
#     ``FrameHashLayer`` / ``CacheCascade`` — the observable is the returned ``CacheResult``;
#   * the six runtime ones are decided above the cache, so they are proved by wrapping a real core's
#     cascade in ``_CountingCascade`` and asserting the cache was never so much as ASKED.
#
# Every runtime case carries its own CONTROL: the identical drive with the bypass condition removed
# must consult the cache. Without that, "no lookups happened" would also pass on a core whose cache
# is switched off, which proves nothing at all.
# ==================================================================================================

_YES = Verdict(satisfied=True, confidence=0.95, evidence="a person is at the door")


def _still() -> list:
    """One unchanging griddable frame — the case the ROI dHash is supposed to recognise."""
    return [[float(x * 8) for x in range(9)] for _ in range(8)]


def _moved() -> list:
    """A frame whose luminance gradient is inverted — maximally distant from ``_still`` in dHash."""
    return [[float(200 - x * 8) for x in range(9)] for _ in range(8)]


def _nudged() -> list:
    """``_still`` with one cell changed — inside any sane near-match radius."""
    grid = _still()
    grid[0][0] = 255.0
    return grid


class _StepClock:
    """A hand-advanced clock in SECONDS (the unit ``FrameHashLayer`` reads)."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _CountingVision:
    """A scripted judge that counts the calls it was actually asked to make."""

    def __init__(self, script: "list | None" = None) -> None:
        self.script = list(script or [_YES])
        self.calls = 0

    async def judge(self, condition, frame, **kw):  # noqa: D401 — a double, not an API
        verdict = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return verdict

    async def detect(self, frame, goals):
        return []

    async def detect_events(self, frames, goals):
        return []


class _CountingCascade:
    """A pass-through recorder around a real cascade — the "was the cache consulted at all?" probe.

    It delegates everything, so behaviour is unchanged; it only records that ``resolve``/``store``
    were reached. ``layers`` is delegated as a property because ``Percept.invalidate_cache`` walks
    it directly.
    """

    def __init__(self, inner: CacheCascade) -> None:
        self.inner = inner
        self.lookups: list = []
        self.stores: list = []

    @property
    def layers(self):
        return self.inner.layers

    @property
    def threshold(self):
        return self.inner.threshold

    def resolve(self, key):
        self.lookups.append(key)
        return self.inner.resolve(key)

    def store(self, key, verdict, confidence):
        self.stores.append(key)
        self.inner.store(key, verdict, confidence)


def _core(
    vision,
    *,
    enabled: bool = True,
    ttl_s: float = 1000.0,
    tau_bits: int = 6,
    threshold: float = 0.0,
    audit: "Callable[[Any], bool] | None" = None,
    direct_judge: "DirectJudge | None" = None,
) -> Percept:
    """A core whose cascade is wrapped in ``_CountingCascade`` (reachable as ``core.cache``)."""
    settings = Settings(
        gate=GateSettings(threshold=0.7, validity_window_s=1000.0),
        cache=CacheSettings(
            enabled=enabled, ttl_s=ttl_s, tau_bits=tau_bits,
            threshold=threshold, audit_rate=0.0,
        ),
    )
    extra: dict = {}
    if audit is not None:
        extra["audit"] = audit
    if direct_judge is not None:
        extra["direct_judge"] = direct_judge
    core = Percept.create(
        vision=vision, settings=settings, discover_plugins=False,
        source_profile=PHYSICAL_SCENE, **extra,
    )
    core.cache = _CountingCascade(core.cache)
    return core


def _arm(core: Percept, goal: Goal) -> Goal:
    with warnings.catch_warnings():
        # A ``count`` goal is driven by ``perceive_boundary``, not the frame worker; arming one
        # without that driver is exactly what the warning is for and exactly what case 6 wants.
        warnings.simplefilter("ignore", UndrivenBehaviorRouteWarning)
        core.add_goal(goal)
    return core.goals.get(goal.id)


def _watch(core: Percept, goal_id: str = "g1", condition: str = "a person is at the door") -> Goal:
    return _arm(core, Goal(
        id=goal_id, condition=condition, say="someone is at the door",
        state={"type": "watch", "count": 0},
    ))


def _layer(clock, *, ttl_s: float = 100.0, tau_bits: int = 4) -> FrameHashLayer:
    return FrameHashLayer(
        CacheSettings(enabled=True, ttl_s=ttl_s, tau_bits=tau_bits), clock=clock
    )


def _drive_judge(core: Percept, goal: Goal, ticks: int, **judge_kw) -> None:
    """Drive ``_judge`` directly — the seam the fence lives on, with its inputs under test control."""
    frame = _still()
    for _ in range(ticks):
        asyncio.run(core._judge(goal, frame, None, {}, **judge_kw))


def _drive_frames(core: Percept, ticks: int, frame=None) -> list:
    out = []
    still = _still() if frame is None else frame
    for i in range(ticks):
        out += asyncio.run(core.perceive_judged(still, t=float(i)))
    return out


# -- 1 ---------------------------------------------------------------------------------------------
def _cache_disabled(t: unittest.TestCase) -> None:
    """``cfg.cache.enabled`` false builds a ZERO-LAYER cascade, so ``resolve`` can only DEFER.

    The bypass is structural, not a branch: ``agent.py:1496`` builds ``layers=[]`` and
    ``cascade.py:53-55`` has nothing to iterate. The fence above is not even involved — which is why
    the assertion is on the cascade's own output as well as on the judge-call count.
    """
    vision = _CountingVision()
    core = _core(vision, enabled=False)
    goal = _watch(core)
    t.assertEqual(list(core.cache.layers), [], "cache.enabled=False must build no layers")

    _drive_judge(core, goal, 6)

    t.assertEqual(vision.calls, 6, "a zero-layer cascade served a verdict it could not hold")
    t.assertEqual(len(core.cache.lookups), 6, "the fence itself did not bypass — the cascade is empty")
    t.assertTrue(
        all(core.cache.inner.resolve(k).status == "DEFER" for k in core.cache.lookups),
        "an empty cascade returned something other than DEFER",
    )

    control_vision = _CountingVision()
    control = _core(control_vision, enabled=True)
    _drive_judge(control, _watch(control), 6)
    t.assertEqual(control_vision.calls, 1, "control: the same drive with the cache on must serve")


# -- 2 ---------------------------------------------------------------------------------------------
def _confident_direct_judge(t: unittest.TestCase) -> None:
    """A confident ``DirectJudge`` answer short-circuits the ``elif`` chain — no key is ever built.

    ``agent.py`` takes the direct branch and ``_judge`` is never entered, so neither ``cache_key``
    nor a lookup exists. The user-visible property is that the answer is fresh for THIS frame and no
    slot is filled that a later frame could be served from.
    """
    class _Rule:
        applies_to = DirectRouting(condition_namespace="screen.ocr.")

        def evaluate(self, condition, frame, roi, facts):
            return DirectResult(
                outcome="confident",
                verdict={"satisfied": True, "confidence": 0.97, "evidence": "direct"},
            )

    vision = _CountingVision()
    core = _core(vision, direct_judge=DirectJudge([_Rule()]))
    _watch(core, condition="screen.ocr.total_exceeds")

    events = _drive_frames(core, 1)

    t.assertEqual(core.metrics.get("direct_judge.hit"), 1, "the premise: the direct rule answered")
    t.assertEqual(vision.calls, 0, "the premise: no provider was consulted")
    t.assertEqual([e.action for e in events], ["fire"], "the gate still governed the direct verdict")
    t.assertEqual(core.cache.lookups, [], "a confident direct hit must not consult the cache")
    t.assertEqual(core.cache.stores, [], "a confident direct hit must not fill a cache slot")

    control_vision = _CountingVision()
    control = _core(control_vision)  # same drive, no direct judge
    _watch(control, condition="screen.ocr.total_exceeds")
    _drive_frames(control, 1)
    t.assertEqual(len(control.cache.lookups), 1, "control: the same goal otherwise IS cacheable")


# -- 3 ---------------------------------------------------------------------------------------------
def _multi_frame_strip(t: unittest.TestCase) -> None:
    """A non-empty ``frames`` strip bypasses: the key dHashes exactly ONE frame.

    Serving a strip verdict out of a single-frame slot (or filling one from a strip) is how a
    temporal answer leaks onto a still.
    """
    vision = _CountingVision()
    core = _core(vision)
    goal = _watch(core)

    _drive_judge(core, goal, 5, frames=[_still(), _nudged(), _still()])

    t.assertEqual(vision.calls, 5, "a strip judge was served from the single-frame cache")
    t.assertEqual(core.cache.lookups, [], "a multi-frame strip must not consult the cache")
    t.assertEqual(core.cache.stores, [], "a multi-frame strip must not pollute a single-frame slot")

    control_vision = _CountingVision()
    control = _core(control_vision)
    _drive_judge(control, _watch(control), 5)  # identical, minus the strip
    t.assertEqual(control_vision.calls, 1, "control: the same drive without a strip is cached")


# -- 4 ---------------------------------------------------------------------------------------------
def _audio_evidence(t: unittest.TestCase) -> None:
    """``audio is not None`` bypasses: a frame hash says nothing about what was heard."""
    vision = _CountingVision()
    core = _core(vision)
    goal = _watch(core)

    _drive_judge(core, goal, 5, audio=b"pcm-bytes")

    t.assertEqual(vision.calls, 5, "an audio judge was served from a frame-hashed slot")
    t.assertEqual(core.cache.lookups, [], "audio evidence must not consult the frame cache")
    t.assertEqual(core.cache.stores, [], "audio evidence must not fill a frame-hashed slot")

    control_vision = _CountingVision()
    control = _core(control_vision)
    _drive_judge(control, _watch(control), 5)
    t.assertEqual(control_vision.calls, 1, "control: the same drive without audio is cached")


# -- 5 ---------------------------------------------------------------------------------------------
def _uncovered_context_key(t: unittest.TestCase) -> None:
    """A context key the episode fingerprint does not cover bypasses.

    ``episode_bundle.covers(context)`` is asked of the PRODUCER of the fingerprint, so widening the
    projection and widening the fence are one edit. An unrecognised key is, by definition,
    answer-affecting state that no part of the key saw.
    """
    vision = _CountingVision()
    core = _core(vision)
    goal = _watch(core)

    _drive_judge(core, goal, 5, context={"a_context_nobody_keyed": 1})

    t.assertEqual(vision.calls, 5, "a verdict was served across context the key never covered")
    t.assertEqual(core.cache.lookups, [], "an uncovered context key must not consult the cache")
    t.assertEqual(core.cache.stores, [], "an uncovered context key must not fill a slot")

    control_vision = _CountingVision()
    control = _core(control_vision)
    _drive_judge(control, _watch(control), 5)
    t.assertEqual(control_vision.calls, 1, "control: the same drive with no context is cached")


# -- 6 ---------------------------------------------------------------------------------------------
def _sense_that_builds_no_key(t: unittest.TestCase) -> None:
    """The non-frame-watch senses never build a ``CacheKey`` — proved by an absence, per sense.

    An absence is only worth asserting when the sense DID something: each drive below is checked to
    have spent judge calls, confirmed reps, or emitted a fire, and only then is the cache asserted
    untouched. Only ``BehaviorRoute.FRAME_WATCH`` reaches a cache key at all.
    """
    # transition — two real judge calls (the from-half and the to-half), zero cache
    vision = _CountingVision()
    core = _core(vision)
    _arm(core, Goal(
        id="tr", condition="the scene lights up", say="good morning",
        state={"type": "transition", "from": "the scene is dark",
               "to": "something bright is present", "count": 0},
    ))
    _drive_frames(core, 1)
    t.assertEqual(vision.calls, 2, "the premise: the transition sense spent its two judge calls")
    t.assertEqual(core.cache.lookups, [], "the transition sense must build no cache key")
    t.assertEqual(core.cache.stores, [], "the transition sense must fill no cache slot")

    # rhythm — the tracker confirmed reps off measured channels, zero cache
    core = _core(_CountingVision())
    _arm(core, Goal(id="rh", condition="repetitions", say="ten",
                    state={"type": "rhythm", "count": 0}))
    for i in range(60):
        phase = (i % 12) / 12.0
        core.perceive_rhythm("rh", t=float(i) * 0.1, channels={"knee": abs(phase - 0.5)})
    t.assertGreater(core.rhythm["rh"].count, 0, "the premise: the rhythm sense counted something")
    t.assertEqual(core.cache.lookups, [], "the rhythm sense must build no cache key")

    # boundary — a counted occurrence fired, zero cache
    core = _core(_CountingVision())
    _arm(core, Goal(id="cnt", condition="reps", say="ten", state={"type": "count", "count": 0}))
    t.assertIsNotNone(core.perceive_boundary("cnt", t=1.0), "the premise: the boundary fired")
    t.assertEqual(core.cache.lookups, [], "the boundary sense must build no cache key")

    # absence (lost coverage) — a fire off the LACK of evidence, zero cache
    core = _core(_CountingVision())
    _arm(core, Goal(id="sev", condition="the baby is visible", say="blind",
                    state={"type": "watch", "miss_cost": "severe", "count": 0}))
    core.last_supported["sev"] = 0.0
    t.assertEqual([e.action for e in core.check_observability(t=100.0, staleness_s=1.0)], ["fire"])
    t.assertEqual(core.cache.lookups, [], "the absence sense must build no cache key")

    # episode boundary — a consolidated answer fired, zero cache
    core = _core(_CountingVision())
    _arm(core, Goal(id="pr", condition="how is the dish coming along", say="done",
                    state={"type": "process", "process_qa": True, "count": 0}))
    ctx = core.executor.ctx("pr")
    ctx.process_log.append("chopped an onion")
    ctx.last_step_at = 0.0
    fires = asyncio.run(core.check_episode_boundary(t=1000.0, inactivity_s=1.0))
    t.assertEqual([e.action for e in fires], ["fire"], "the premise: the episode boundary fired")
    t.assertEqual(core.cache.lookups, [], "the episode sense must build no cache key")


# -- 7 ---------------------------------------------------------------------------------------------
def _ungriddable_frame(t: unittest.TestCase) -> None:
    """``dhash`` returns ``None`` for a frame with no visual signal: ``lookup`` DEFERs, ``store`` no-ops.

    Both halves matter. A DEFER-only fix would still let an un-hashable frame OVERWRITE a live slot
    with a hash it never computed; a no-op store on its own would still let it be served.
    """
    token = "a-token"
    t.assertIsNone(dhash(token), "the premise: a bare token is not griddable")

    clock = _StepClock()
    layer = _layer(clock)
    bad = CacheKey(goal_id="g", frame=token)

    layer.store(bad, _YES, 0.95)
    t.assertEqual(len(layer), 0, "an un-hashable frame filled a slot with no hash behind it")

    good = CacheKey(goal_id="g", frame=_still())
    layer.store(good, _YES, 0.95)
    t.assertEqual(len(layer), 1, "the premise: a griddable frame does fill the slot")
    t.assertEqual(layer.lookup(good).status, "HIT", "the premise: that slot is live")
    t.assertEqual(
        layer.lookup(bad).status, "DEFER",
        "an un-hashable frame was served from a live slot it could not be compared against",
    )
    t.assertEqual(len(layer), 1, "the un-hashable lookup must not evict the live entry either")


# -- 8 ---------------------------------------------------------------------------------------------
def _time_to_live_expired(t: unittest.TestCase) -> None:
    """Past the TTL is a HARD miss and the entry is DELETED; ``ttl_s <= 0`` is the same, immediately.

    The TTL is the only bound on drift the hash cannot see, so ``age >= ttl`` is inclusive and the
    deletion is what stops a dead entry being re-decayed forever.
    """
    clock = _StepClock()
    layer = _layer(clock, ttl_s=10.0)
    key = CacheKey(goal_id="g", frame=_still())
    layer.store(key, _YES, 0.95)

    clock.now = 9.999
    t.assertEqual(layer.lookup(key).status, "HIT", "the premise: inside the TTL it serves")
    clock.now = 10.0  # age == ttl exactly
    t.assertEqual(layer.lookup(key).status, "DEFER", "age == ttl must be a miss, not a serve")
    t.assertEqual(len(layer), 0, "an expired entry must be dropped, not left to decay forever")

    dead_clock = _StepClock()
    dead = _layer(dead_clock, ttl_s=0.0)
    dead.store(key, _YES, 0.95)
    t.assertEqual(len(dead), 1, "the premise: store does not consult the TTL")
    t.assertEqual(dead.lookup(key).status, "DEFER", "ttl_s <= 0 must never serve")
    t.assertEqual(len(dead), 0, "ttl_s <= 0 must drop the entry it refused to serve")


# -- 9 ---------------------------------------------------------------------------------------------
def _distance_beyond_tau(t: unittest.TestCase) -> None:
    """Hamming distance greater than ``tau_bits`` is the frame-diff backstop saying "re-perceive"."""
    clock = _StepClock()
    tau = 4
    layer = _layer(clock, tau_bits=tau)
    key = CacheKey(goal_id="g", frame=_still())
    layer.store(key, _YES, 0.95)

    far_bits, near_bits = dhash(_moved()), dhash(_nudged())
    stored_bits = dhash(_still())
    t.assertGreater(hamming(far_bits, stored_bits), tau, "the premise: the far frame is far")
    t.assertLessEqual(hamming(near_bits, stored_bits), tau, "the premise: the near frame is near")

    t.assertEqual(
        layer.lookup(CacheKey(goal_id="g", frame=_moved())).status, "DEFER",
        "a structurally different frame was served from the slot",
    )
    t.assertEqual(len(layer), 1, "a distance miss must not evict — only the TTL evicts")
    t.assertEqual(
        layer.lookup(CacheKey(goal_id="g", frame=_nudged())).status, "HIT",
        "control: inside the radius it still serves, so the miss above IS the distance",
    )


# -- 10 --------------------------------------------------------------------------------------------
def _below_cascade_threshold(t: unittest.TestCase) -> None:
    """A real HIT whose (decayed) confidence is under the cascade threshold falls through to DEFER.

    The layer answering HIT and the cascade serving it are two different decisions. Age decay is what
    makes the difference visible: the entry is alive and near-matched, and still not good enough.
    """
    clock = _StepClock()
    layer = _layer(clock, ttl_s=10.0, tau_bits=8)
    key = CacheKey(goal_id="g", frame=_still())
    layer.store(key, _YES, 0.95)
    clock.now = 6.0  # decay 0.4 on an exact hash -> confidence 0.4

    direct = layer.lookup(key)
    t.assertEqual(direct.status, "HIT", "the premise: the layer itself still hits")
    t.assertAlmostEqual(direct.confidence, 0.4, places=6)

    t.assertEqual(
        CacheCascade(layers=[layer], threshold=0.9).resolve(key).status, "DEFER",
        "a hit under the threshold was served anyway",
    )
    t.assertEqual(
        CacheCascade(layers=[layer], threshold=0.1).resolve(key).status, "HIT",
        "control: the SAME hit clears a lower bar, so the threshold is what refused it",
    )


# -- 11 --------------------------------------------------------------------------------------------
def _false_serve_audit(t: unittest.TestCase) -> None:
    """The audit bypasses the SERVE, not the lookup: the hit stands, a real call is spent, both compared.

    This is the only one of the twelve where the cache IS consulted and DOES hit. What is bypassed is
    the free ride: on an audited key the runtime re-perceives and the fresh read wins on disagreement.
    So the assertion is inverted — a real judge call on every tick, despite the hits.
    """
    ticks = 8
    vision = _CountingVision()
    core = _core(vision, audit=lambda key: True)
    _watch(core)

    _drive_frames(core, ticks)

    hits = core.metrics.get("cache.hit")
    t.assertGreater(hits, 0, "the premise: the cache was consulted and did hit")
    t.assertEqual(len(core.cache.lookups), ticks, "the audit must not suppress the LOOKUP")
    t.assertEqual(core.metrics.get("cache.audit"), hits, "every hit on an audited key is audited")
    t.assertEqual(
        vision.calls, ticks,
        "an audited hit must still spend a real judge call — that is the whole point of the audit",
    )

    control_vision = _CountingVision()
    control = _core(control_vision, audit=lambda key: False)
    _watch(control)
    _drive_frames(control, ticks)
    t.assertEqual(
        control.metrics.get("cache.hit"), hits,
        "control: the same drive hits the same number of times",
    )
    t.assertEqual(control.metrics.get("cache.audit"), 0)
    t.assertLess(
        control_vision.calls, vision.calls,
        "control: un-audited hits are served free, so the audit really is what spent the calls",
    )


# -- 12 --------------------------------------------------------------------------------------------
def _invalidation_dropped_the_slot(t: unittest.TestCase) -> None:
    """``invalidate`` (goal-change) and ``invalidate_entity`` (assertion-override) drop the slot.

    Both read ``slot[0]``/``slot[1]`` by contract, so each is asserted to be SCOPED — dropping the
    named goal/entity and nothing else. The runtime round trip is asserted too, because
    ``Percept.invalidate_cache`` walks ``cache.layers`` rather than calling the layer directly.
    """
    clock = _StepClock()
    layer = _layer(clock)
    mine = CacheKey(goal_id="ga", frame=_still(), entity_id="e-one")
    yours = CacheKey(goal_id="gb", frame=_still(), entity_id="e-two")
    layer.store(mine, _YES, 0.95)
    layer.store(yours, _YES, 0.95)
    t.assertEqual((layer.lookup(mine).status, layer.lookup(yours).status), ("HIT", "HIT"))

    t.assertEqual(layer.invalidate("ga"), 1, "goal-change invalidation reported the wrong count")
    t.assertEqual(layer.lookup(mine).status, "DEFER", "the invalidated goal was still served")
    t.assertEqual(layer.lookup(yours).status, "HIT", "invalidation must be scoped to its goal")

    t.assertEqual(layer.invalidate_entity("e-two"), 1)
    t.assertEqual(layer.lookup(yours).status, "DEFER", "the corrected entity was still served")
    t.assertEqual(len(layer), 0)

    # the runtime round trip: a warm cache, invalidated through the public hook, re-perceives.
    vision = _CountingVision()
    core = _core(vision)
    goal = _watch(core)
    _drive_judge(core, goal, 4)
    t.assertEqual(vision.calls, 1, "the premise: the cache was warm and serving")
    core.invalidate_cache("g1")
    _drive_judge(core, goal, 1)
    t.assertEqual(vision.calls, 2, "invalidate_cache did not reach the layer through the cascade")


@dataclass(frozen=True)
class _BypassCase:
    """One documented bypass condition: its number, the reason it bypasses, and its proof."""

    number: int
    reason: str
    site: str
    run: Callable[[unittest.TestCase], None]


_BYPASS_CASES: tuple[_BypassCase, ...] = (
    _BypassCase(1, "cache_disabled_builds_a_zero_layer_cascade",
                "agent.py cascade construction / cascade.py resolve", _cache_disabled),
    _BypassCase(2, "confident_direct_judge_short_circuits_before_any_key",
                "agent.py direct-judge branch", _confident_direct_judge),
    _BypassCase(3, "multi_frame_strip_is_not_single_frame_hashable",
                "agent.py judge fence", _multi_frame_strip),
    _BypassCase(4, "audio_evidence_is_not_frame_hashable",
                "agent.py judge fence", _audio_evidence),
    _BypassCase(5, "context_key_the_episode_fingerprint_does_not_cover",
                "agent.py judge fence / episode_memory.py covers", _uncovered_context_key),
    _BypassCase(6, "sense_that_never_builds_a_cache_key",
                "absence of a call site outside the frame-watch route", _sense_that_builds_no_key),
    _BypassCase(7, "frame_that_cannot_be_gridded_has_no_hash",
                "framehash.py lookup and store", _ungriddable_frame),
    _BypassCase(8, "entry_past_its_time_to_live",
                "framehash.py lookup", _time_to_live_expired),
    _BypassCase(9, "hamming_distance_beyond_tau_bits",
                "framehash.py lookup", _distance_beyond_tau),
    _BypassCase(10, "hit_confidence_below_the_cascade_threshold",
                "cascade.py resolve", _below_cascade_threshold),
    _BypassCase(11, "false_serve_audit_selected_this_key",
                "agent.py audit_or_use", _false_serve_audit),
    _BypassCase(12, "invalidation_dropped_the_slot",
                "framehash.py invalidate and invalidate_entity", _invalidation_dropped_the_slot),
)

#: The twelve, written out independently of the table above. Spelling the set a second time is what
#: makes a silently DELETED or RENAMED row fail rather than shrink the matrix unnoticed.
THE_TWELVE_BYPASS_REASONS = frozenset({
    "cache_disabled_builds_a_zero_layer_cascade",
    "confident_direct_judge_short_circuits_before_any_key",
    "multi_frame_strip_is_not_single_frame_hashable",
    "audio_evidence_is_not_frame_hashable",
    "context_key_the_episode_fingerprint_does_not_cover",
    "sense_that_never_builds_a_cache_key",
    "frame_that_cannot_be_gridded_has_no_hash",
    "entry_past_its_time_to_live",
    "hamming_distance_beyond_tau_bits",
    "hit_confidence_below_the_cascade_threshold",
    "false_serve_audit_selected_this_key",
    "invalidation_dropped_the_slot",
})


class TestTheTwelveBypassConditions(unittest.TestCase):
    """All twelve documented cache bypasses, accounted for in one place, each with its own reason."""

    def test_the_table_declares_exactly_the_twelve_documented_conditions(self):
        """The accounting itself: twelve rows, numbered 1..12, no duplicate and no missing reason."""
        self.assertEqual(len(_BYPASS_CASES), 12, "the matrix no longer walks twelve conditions")
        self.assertEqual([case.number for case in _BYPASS_CASES], list(range(1, 13)))
        reasons = [case.reason for case in _BYPASS_CASES]
        self.assertEqual(len(set(reasons)), 12, "two rows share a reason — one bypass is unproved")
        self.assertEqual(
            set(reasons), THE_TWELVE_BYPASS_REASONS,
            "the table and the declared set of twelve reasons disagree",
        )
        self.assertTrue(all(case.site for case in _BYPASS_CASES))

    def test_every_documented_condition_produces_its_own_bypass(self):
        """Walk all twelve. A row that stops bypassing fails HERE, named by its own reason."""
        walked: list[str] = []
        for case in _BYPASS_CASES:
            with self.subTest(condition=case.number, reason=case.reason, site=case.site):
                case.run(self)
                walked.append(case.reason)
        self.assertEqual(
            walked, [case.reason for case in _BYPASS_CASES],
            "12/12 not accounted for: a condition above did not produce its documented bypass",
        )

    def test_the_control_none_of_the_twelve_holds_and_the_cache_is_consulted_and_serves(self):
        """The anti-vacuity guard for the six runtime rows.

        Nine of the twelve assert that something did NOT happen. That is only evidence if the same
        drive, with every bypass condition removed, DOES consult the cache and DOES serve from it.
        """
        vision = _CountingVision()
        core = _core(vision)
        goal = _watch(core)

        _drive_judge(core, goal, 6)

        self.assertEqual(len(core.cache.lookups), 6, "the cache must be consulted on the plain path")
        self.assertEqual(len(core.cache.stores), 1, "the first miss must fill exactly one slot")
        self.assertEqual(vision.calls, 1, "five of six ticks must be served from the cache")
        self.assertEqual(core.metrics.get("cache.hit"), 5)


if __name__ == "__main__":
    unittest.main()
