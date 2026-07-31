"""L1: the entity graph — continuity, provenance, assertion-beats-observation, focus, eviction."""

import unittest

from percept import Percept
from percept.adapters import Box, Observation
from percept.cache.cascade import CacheResult
from percept.entity import EntityGraph
from percept.identity import ContinuityIdentity

A = Box(0.4, 0.4, 0.2, 0.2)
B = Box(0.1, 0.1, 0.2, 0.2)

# Two boxes whose overlap is EXACTLY one half, in binary floating point, with no rounding anywhere:
# the intersection is 0.25 * 0.5 = 0.125 and the union is 0.25 + 0.125 - 0.125 = 0.25, so the ratio
# is 0.125 / 0.25 = 0.5 — every one of those quantities is a dyadic rational and therefore exact.
# The threshold comparison is then a real boundary test, not epsilon luck that a different machine
# could round the other way.
WIDE = Box(0.0, 0.0, 0.5, 0.5)
HALF = Box(0.0, 0.0, 0.25, 0.5)

# One step past the boundary. Kept as a named constant so both directions of the test read as the
# same experiment with the comparison flipped.
ABOVE = 1e-9


def cup(box=A, conf=0.9, **attrs):
    return Observation(label="cup", box=box, confidence=conf, attrs=attrs)


def thing(box, label="thing", conf=0.9):
    return Observation(label=label, box=box, confidence=conf)


def grid_box(i):
    """The i-th cell of a 9-wide grid of 0.05 boxes spaced 0.1 apart — pairwise disjoint, so every
    one of them mints a fresh entity (overlap 0.0 clears no threshold)."""
    row, col = divmod(i, 9)
    return Box(col * 0.1, row * 0.1, 0.05, 0.05)


class SpyCacheLayer:
    """A cache layer that records ``invalidate_entity`` calls and nothing else.

    ``Settings.cache.enabled`` is False by default, so a stock core's cascade is EMPTY — which means
    "the eviction hook swept the cache" is unobservable unless a layer is present to observe it.
    This is that layer. It never HITs, so appending it cannot change any judging path.
    """

    name = "spy"

    def __init__(self):
        self.entity_calls = []
        self.goal_calls = []

    def lookup(self, key):
        return CacheResult.defer()

    def store(self, key, verdict, confidence):
        return None

    def invalidate(self, goal_id=None):
        self.goal_calls.append(goal_id)
        return 0

    def invalidate_entity(self, entity_id, *, goal_id=None):
        self.entity_calls.append((entity_id, goal_id))
        return 0


class TestEntityGraph(unittest.TestCase):
    def setUp(self):
        self.g = EntityGraph(iou_match=0.5, max_entities=8)

    def test_mints_new_entity(self):
        [e] = self.g.ingest([cup()], t=1.0)
        self.assertEqual(e.label, "cup")
        self.assertEqual(len(self.g), 1)

    def test_continuity_same_box_same_id(self):
        [e1] = self.g.ingest([cup()], t=1.0)
        [e2] = self.g.ingest([cup()], t=2.0)  # same box+label -> same entity
        self.assertEqual(e1.id, e2.id)
        self.assertEqual(len(self.g), 1)

    def test_distinct_box_new_id(self):
        [e1] = self.g.ingest([cup(box=A)], t=1.0)
        [e2] = self.g.ingest([cup(box=B)], t=2.0)  # disjoint -> new entity
        self.assertNotEqual(e1.id, e2.id)
        self.assertEqual(len(self.g), 2)

    def test_observed_attrs_carry_provenance(self):
        [e] = self.g.ingest([cup(state="full")], t=1.0)
        self.assertEqual(e.attributes["state"].source, "observed")

    def test_assertion_beats_observation(self):
        [e] = self.g.ingest([cup(contents="coffee")], t=1.0)
        self.g.assert_attribute(e.id, "contents", "tea", t=1.5)
        self.assertEqual(e.attributes["contents"].value, "tea")
        self.assertEqual(e.attributes["contents"].source, "asserted")
        # a later observation must NOT clobber the asserted fact
        self.g.ingest([cup(contents="coffee")], t=2.0)
        self.assertEqual(e.attributes["contents"].value, "tea")
        self.assertEqual(e.asserted(), {"contents": "tea"})

    def test_in_focus_is_largest_in_last_frame(self):
        big = Observation("cup", Box(0.1, 0.1, 0.5, 0.5), 0.9)
        small = Observation("cup", Box(0.8, 0.8, 0.1, 0.1), 0.9)
        self.g.ingest([small, big], t=1.0)
        self.assertEqual(self.g.in_focus().box, big.box)

    def test_in_focus_none_when_no_frame(self):
        self.assertIsNone(self.g.in_focus())

    def test_eviction_respects_max(self):
        g = EntityGraph(iou_match=0.5, max_entities=2)
        g.ingest([Observation("a", Box(0, 0, 0.1, 0.1), 0.9)], t=1.0)
        g.ingest([Observation("b", Box(0.3, 0.3, 0.1, 0.1), 0.9)], t=2.0)
        g.ingest([Observation("c", Box(0.6, 0.6, 0.1, 0.1), 0.9)], t=3.0)
        self.assertEqual(len(g), 2)  # oldest ('a') evicted

    def test_note_mention_binds_to_focus(self):
        self.g.ingest([cup()], t=1.0)
        bound = self.g.note_mention("this is my travel mug", t=1.2)
        self.assertIsNotNone(bound)
        self.assertEqual(self.g.in_focus().attributes["mention"].source, "asserted")


class TestIouThresholdBoundary(unittest.TestCase):
    """An overlap EXACTLY equal to ``iou_match``.

    The shipped side is INCLUSIVE. ``ContinuityIdentity.match`` seeds ``best_iou = self.iou_match``
    and accepts ``iou >= best_iou``, so an observation overlapping a known entity by exactly the
    threshold CONTINUES that entity rather than minting a rival. That matters because the boundary is
    the one place an off-by-epsilon is invisible: a subject sitting right on the threshold would
    otherwise flip between one entity and two from frame to frame, and each flip is a fresh gate key
    — which a user experiences as the same subject notified about twice.
    """

    def test_the_two_boxes_overlap_by_exactly_the_default_threshold(self):
        # Exact, both directions — the ratio is dyadic, so this is equality, not near-equality.
        self.assertEqual(WIDE.iou(HALF), 0.5)
        self.assertEqual(HALF.iou(WIDE), 0.5)
        # ...and 0.5 is the shipped default threshold, so this is the boundary that actually ships.
        self.assertEqual(ContinuityIdentity().iou_match, WIDE.iou(HALF))
        self.assertGreater(0.5 + ABOVE, 0.5)  # the step really is a step

    def test_matcher_accepts_an_overlap_exactly_at_the_threshold(self):
        """The threshold lives on the matcher, so pin it there too — the graph is one caller."""
        entity_like = type("E", (), {"id": "e1", "label": "thing", "box": WIDE})()
        exact = ContinuityIdentity(iou_match=WIDE.iou(HALF))
        just_above = ContinuityIdentity(iou_match=WIDE.iou(HALF) + ABOVE)
        self.assertEqual(exact.match(thing(HALF), [entity_like]), "e1")
        self.assertIsNone(just_above.match(thing(HALF), [entity_like]))

    def test_exactly_at_threshold_continues_the_same_entity(self):
        threshold = WIDE.iou(HALF)
        # The two observations are 1.5s apart, which is longer than the default coast window
        # (coast_s=1.0). That deliberately takes the motion-prediction recovery path out of the
        # experiment, so the only thing under test is the continuity threshold itself.
        for first, second in ((WIDE, HALF), (HALF, WIDE)):  # symmetric: order must not matter
            with self.subTest(first=first, second=second):
                g = EntityGraph(iou_match=threshold, max_entities=8)
                [e1] = g.ingest([thing(first)], t=1.0)
                [e2] = g.ingest([thing(second)], t=2.5)
                self.assertEqual(e1.id, e2.id)
                self.assertEqual(len(g), 1)

    def test_one_step_above_the_threshold_mints_a_second_entity(self):
        g = EntityGraph(iou_match=WIDE.iou(HALF) + ABOVE, max_entities=8)
        [e1] = g.ingest([thing(WIDE)], t=1.0)
        [e2] = g.ingest([thing(HALF)], t=2.5)  # still outside the coast window
        self.assertNotEqual(e1.id, e2.id)
        self.assertEqual(len(g), 2)

    def test_the_boundary_verdict_is_stable_across_repeats(self):
        """Deterministic across runs: same construction, same answer, every time. A graph carries
        mutable per-entity motion state, so build a fresh one each round rather than reusing one."""
        seen = set()
        for _ in range(5):
            g = EntityGraph(iou_match=WIDE.iou(HALF), max_entities=8)
            g.ingest([thing(WIDE)], t=1.0)
            [e] = g.ingest([thing(HALF)], t=2.5)
            seen.add((e.id, len(g)))
        self.assertEqual(seen, {("e1", 1)})

    def test_prediction_recovery_threshold_is_inclusive_on_the_same_side(self):
        """``_predict_match`` compares with the same ``>=``, against its own ``predict_iou``.

        Worth pinning separately because it is the reason raising ``iou_match`` above an observed
        overlap does NOT, on its own, split a track: inside the coast window the recovery path gets a
        second, lower-thresholded say. Here continuity is switched off by construction
        (``iou_match=0.9`` rejects the 0.5 overlap) and only the prediction threshold decides.
        """
        threshold = WIDE.iou(HALF)
        # 1.0s apart is exactly the coast window and is NOT past it (`now - anchor_t > coast_s`),
        # so the track is still coastable; with zero learned velocity the predicted box is the
        # anchor box, and the predicted overlap is the same exact 0.5.
        linked = EntityGraph(iou_match=0.9, max_entities=8, predict_iou=threshold)
        [a1] = linked.ingest([thing(WIDE)], t=1.0)
        [a2] = linked.ingest([thing(HALF)], t=2.0)
        self.assertEqual(a1.id, a2.id, "an exactly-at-threshold prediction must re-link")
        self.assertEqual(len(linked), 1)

        split = EntityGraph(iou_match=0.9, max_entities=8, predict_iou=threshold + ABOVE)
        [b1] = split.ingest([thing(WIDE)], t=1.0)
        [b2] = split.ingest([thing(HALF)], t=2.0)
        self.assertNotEqual(b1.id, b2.id)
        self.assertEqual(len(split), 2)


class TestEvictionSweepsDownstreamState(unittest.TestCase):
    """Eviction at exactly ``max_entities`` must take the entity's derived state with it.

    The graph-side half (the cap holds) is covered by ``test_eviction_respects_max``. This is the
    other half: an evicted entity that leaves its ``(goal_id, entity_id)`` gate key armed and its
    cached verdicts in place is a leak with a user-visible tail — a 24/7 watch whose track ids churn
    accumulates dead edges, and a slot keyed on an id that no longer exists can still be served.
    """

    def _core_with_spy(self):
        core = Percept.create(discover_plugins=False)
        spy = SpyCacheLayer()
        core.cache.layers.append(spy)
        return core, spy

    def test_the_graph_eviction_hook_is_wired_to_the_core(self):
        core = Percept.create(discover_plugins=False)
        hook = core.graph.on_evict
        self.assertIsNotNone(hook, "an unwired graph evicts silently and leaks every derived key")
        # NOTE: `core.graph.on_evict is core._on_entity_evicted` is False — attribute access builds a
        # fresh bound method each time. Identity has to be asserted on the pair it binds.
        self.assertIs(getattr(hook, "__self__", None), core)
        self.assertIs(getattr(hook, "__func__", None), Percept._on_entity_evicted)
        self.assertEqual(core.settings.entity.max_entities, 64)

    def test_eviction_drops_the_dead_entitys_gate_key_and_cache_slots(self):
        core, spy = self._core_with_spy()
        cap = core.settings.entity.max_entities
        goal_id = "zone"

        ids = []
        for i in range(cap):  # fill to exactly the cap — no eviction yet
            [e] = core.graph.ingest([thing(grid_box(i))], t=float(i + 1))
            ids.append(e.id)
        self.assertEqual(len(core.graph), cap)
        self.assertEqual(len(set(ids)), cap, "the grid must be pairwise disjoint")

        for eid in ids:
            core.gate.rearm((goal_id, eid))  # every live entity now owns a gate edge
        self.assertEqual(len(core.gate._armed), cap)
        self.assertEqual(spy.entity_calls, [])

        # One past the cap. The evictee is the least-recently-seen entity, which is the first one.
        [newest] = core.graph.ingest([thing(grid_box(cap))], t=float(cap + 1))
        evicted, survivors = ids[0], ids[1:]

        self.assertEqual(len(core.graph), cap, "the cap must hold")
        self.assertIsNone(core.graph.get(evicted))
        self.assertIsNotNone(core.graph.get(newest.id))

        # -- the cache half: the hook reached every layer, with the dead id and no goal scope.
        self.assertEqual(spy.entity_calls, [(evicted, None)])

        # -- the gate half: zero orphans. Scan the WHOLE key space, not just the key we armed, so a
        # partial sweep that leaves some other tuple shape behind is still caught.
        armed = [tuple(k) for k, _ in core.gate.snapshot()["armed"]]
        orphans = [k for k in armed if len(k) >= 2 and k[1] == evicted]
        self.assertEqual(orphans, [], f"evicted entity left orphan gate keys: {orphans}")

        # -- and nothing else was swept: eviction is surgical, not a reset.
        self.assertEqual(sorted(armed), sorted((goal_id, eid) for eid in survivors))

        # The hook reads the private executor slot rather than the property precisely so it cannot
        # CONSTRUCT an executor for a run that never had one.
        self.assertIsNone(core._executor)

    def test_no_eviction_and_no_sweep_at_exactly_the_cap(self):
        """The boundary in the other direction: the cap-th entity must not evict anything."""
        core, spy = self._core_with_spy()
        cap = core.settings.entity.max_entities
        for i in range(cap):
            [e] = core.graph.ingest([thing(grid_box(i))], t=float(i + 1))
            core.gate.rearm(("zone", e.id))
        self.assertEqual(len(core.graph), cap)
        self.assertEqual(spy.entity_calls, [])
        self.assertEqual(len(core.gate._armed), cap)


if __name__ == "__main__":
    unittest.main()
