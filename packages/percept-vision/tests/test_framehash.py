"""L1: the perception cache (``FrameHashLayer``) on synthetic frames with no models.

Covers near-match (identical -> HIT, structural change -> DEFER), the confidence math
(``min(verdict.conf, sim*age_decay)``: stale decays, low-conf never served confidently), the TTL
hard ceiling, and the invalidation order (goal-change drop, assertion-override drop, TTL backstop).

Frames are synthetic 2D luminance grids / bytes — the layer is pure stdlib, no numpy required.
"""

import unittest

from percept.adapters import Box, Verdict
from percept.cache.framehash import (
    CacheKey,
    FrameHashLayer,
    _default_key_fn,
    _slot,
    dhash,
    hamming,
)
from percept.config import CacheSettings


def smooth_grid(n: int = 10) -> list[list[float]]:
    """A left-to-right luminance ramp (every horizontal-adjacency bit set)."""
    return [[float(x) for x in range(n)] for _ in range(n)]


def reversed_grid(n: int = 10) -> list[list[float]]:
    """The structural opposite of ``smooth_grid`` (max Hamming distance)."""
    return [[float(n - 1 - x) for x in range(n)] for _ in range(n)]


class _Clock:
    """A hand-cranked monotonic clock so TTL/decay tests are deterministic (no real time)."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class TestDHash(unittest.TestCase):
    def test_identical_frame_zero_distance(self):
        g = smooth_grid()
        self.assertEqual(hamming(dhash(g), dhash(g)), 0)

    def test_structural_change_large_distance(self):
        self.assertEqual(hamming(dhash(smooth_grid()), dhash(reversed_grid())), 64)

    def test_bytes_frame_is_hashable(self):
        self.assertIsNotNone(dhash(bytes(range(64))))

    def test_unhashable_frame_returns_none(self):
        # a bare token (string) carries no visual signal -> not griddable.
        self.assertIsNone(dhash("sip-coffee"))


class TestNearMatch(unittest.TestCase):
    def setUp(self):
        self.clock = _Clock(0.0)
        self.cfg = CacheSettings(ttl_s=10.0, tau_bits=6)
        self.layer = FrameHashLayer(self.cfg, clock=self.clock)
        self.v = Verdict(satisfied=True, confidence=0.9, evidence="cached coffee")

    def test_cold_lookup_defers(self):
        r = self.layer.lookup(CacheKey("g1", smooth_grid(), None, "e1"))
        self.assertEqual(r.status, "DEFER")

    def test_identical_frame_hits(self):
        key = CacheKey("g1", smooth_grid(), None, "e1")
        self.layer.store(key, self.v, 0.9)
        r = self.layer.lookup(CacheKey("g1", smooth_grid(), None, "e1"))
        self.assertEqual(r.status, "HIT")
        self.assertIs(r.verdict, self.v)
        self.assertEqual(r.source_layer, "framehash")
        self.assertEqual(r.provenance, "cached")

    def test_structural_change_defers(self):
        self.layer.store(CacheKey("g1", smooth_grid(), None, "e1"), self.v, 0.9)
        r = self.layer.lookup(CacheKey("g1", reversed_grid(), None, "e1"))
        self.assertEqual(r.status, "DEFER")  # beyond tau_bits -> re-perceive

    def test_tau_bits_zero_requires_exact(self):
        cfg = CacheSettings(ttl_s=10.0, tau_bits=0)
        layer = FrameHashLayer(cfg, clock=self.clock)
        g = smooth_grid()
        layer.store(CacheKey("g1", g, None, "e1"), self.v, 0.9)
        # a one-pixel structural nudge that flips >=1 gradient bit must now DEFER.
        nudged = [row[:] for row in g]
        nudged[0][1] = -100.0  # flips the (0,0) adjacency bit
        self.assertEqual(layer.lookup(CacheKey("g1", nudged, None, "e1")).status, "DEFER")


class TestConfidenceMath(unittest.TestCase):
    def setUp(self):
        self.clock = _Clock(0.0)
        self.cfg = CacheSettings(ttl_s=10.0, tau_bits=6)
        self.layer = FrameHashLayer(self.cfg, clock=self.clock)

    def test_fresh_identical_confidence_is_verdict_conf(self):
        key = CacheKey("g1", smooth_grid(), None, "e1")
        self.layer.store(key, Verdict(True, 0.9), 0.9)
        r = self.layer.lookup(CacheKey("g1", smooth_grid(), None, "e1"))
        # sim=1.0 (dist 0), decay=1.0 (age 0) -> min(0.9, 1.0) = 0.9
        self.assertAlmostEqual(r.confidence, 0.9)

    def test_stale_entry_decays_confidence(self):
        self.layer.store(CacheKey("g1", smooth_grid(), None, "e1"), Verdict(True, 0.9), 0.9)
        self.clock.t = 5.0  # half the TTL -> decay 0.5
        r = self.layer.lookup(CacheKey("g1", smooth_grid(), None, "e1"))
        self.assertEqual(r.status, "HIT")
        self.assertAlmostEqual(r.confidence, 0.5)  # min(0.9, 1.0*0.5)

    def test_low_conf_verdict_never_served_confidently(self):
        # a verdict that was itself low-confidence is capped by min() -> can't be laundered to high.
        self.layer.store(CacheKey("g1", smooth_grid(), None, "e1"), Verdict(True, 0.2), 0.2)
        r = self.layer.lookup(CacheKey("g1", smooth_grid(), None, "e1"))
        self.assertEqual(r.status, "HIT")
        self.assertAlmostEqual(r.confidence, 0.2)

    def test_near_but_not_identical_lowers_confidence(self):
        g = smooth_grid()
        self.layer.store(CacheKey("g1", g, None, "e1"), Verdict(True, 0.9), 0.9)
        # flip exactly a couple of gradient bits -> sim < 1 -> confidence drops below 0.9.
        nudged = [row[:] for row in g]
        nudged[0][1] = -100.0
        nudged[1][1] = -100.0
        r = self.layer.lookup(CacheKey("g1", nudged, None, "e1"))
        if r.status == "HIT":  # within tau_bits
            self.assertLess(r.confidence, 0.9)


class TestTTL(unittest.TestCase):
    def test_ttl_is_a_hard_ceiling(self):
        clock = _Clock(0.0)
        layer = FrameHashLayer(CacheSettings(ttl_s=10.0, tau_bits=6), clock=clock)
        layer.store(CacheKey("g1", smooth_grid(), None, "e1"), Verdict(True, 0.99), 0.99)
        clock.t = 10.0  # exactly at the TTL -> always DEFER, even on an identical frame
        self.assertEqual(layer.lookup(CacheKey("g1", smooth_grid(), None, "e1")).status, "DEFER")

    def test_zero_ttl_never_serves(self):
        layer = FrameHashLayer(CacheSettings(ttl_s=0.0, tau_bits=6), clock=_Clock(0.0))
        layer.store(CacheKey("g1", smooth_grid(), None, "e1"), Verdict(True, 0.9), 0.9)
        self.assertEqual(layer.lookup(CacheKey("g1", smooth_grid(), None, "e1")).status, "DEFER")


class TestInvalidation(unittest.TestCase):
    def setUp(self):
        self.clock = _Clock(0.0)
        self.layer = FrameHashLayer(CacheSettings(ttl_s=10.0, tau_bits=6), clock=self.clock)
        self.v = Verdict(True, 0.9)
        # two goals, two entities cached.
        self.layer.store(CacheKey("g1", smooth_grid(), None, "e1"), self.v, 0.9)
        self.layer.store(CacheKey("g2", smooth_grid(), None, "e1"), self.v, 0.9)
        self.layer.store(CacheKey("g1", smooth_grid(), None, "e2"), self.v, 0.9)

    def test_goal_change_drops_only_that_goal(self):
        dropped = self.layer.invalidate("g1")
        self.assertEqual(dropped, 2)  # (g1,e1) and (g1,e2)
        self.assertEqual(self.layer.lookup(CacheKey("g1", smooth_grid(), None, "e1")).status, "DEFER")
        self.assertEqual(self.layer.lookup(CacheKey("g2", smooth_grid(), None, "e1")).status, "HIT")

    def test_invalidate_all_clears_everything(self):
        self.assertEqual(self.layer.invalidate(None), 3)
        self.assertEqual(len(self.layer), 0)

    def test_assertion_override_drops_entity_across_goals(self):
        # a correction binds to an entity -> every cached verdict resting on e1 is suspect.
        dropped = self.layer.invalidate_entity("e1")
        self.assertEqual(dropped, 2)  # (g1,e1) and (g2,e1)
        self.assertEqual(self.layer.lookup(CacheKey("g1", smooth_grid(), None, "e2")).status, "HIT")

    def test_assertion_override_scoped_to_goal(self):
        dropped = self.layer.invalidate_entity("e1", goal_id="g1")
        self.assertEqual(dropped, 1)
        self.assertEqual(self.layer.lookup(CacheKey("g2", smooth_grid(), None, "e1")).status, "HIT")


class TestRoiScoping(unittest.TestCase):
    def test_roi_crops_to_a_subregion(self):
        # two grids that differ only OUTSIDE the ROI should still near-match within it.
        clock = _Clock(0.0)
        layer = FrameHashLayer(CacheSettings(ttl_s=10.0, tau_bits=2), clock=clock)
        base = smooth_grid(12)
        roi = Box(0.0, 0.0, 0.5, 0.5)  # top-left quadrant
        layer.store(CacheKey("g1", base, roi, "e1"), Verdict(True, 0.9), 0.9)
        # perturb the bottom-right (outside the ROI).
        other = [row[:] for row in base]
        for y in range(6, 12):
            for x in range(6, 12):
                other[y][x] = -50.0
        r = layer.lookup(CacheKey("g1", other, roi, "e1"))
        self.assertEqual(r.status, "HIT")  # the watched ROI is unchanged


# ---------------------------------------------------------------------------
# Key composition: the tuple adapter and the slot it lands in.
#
# ``_default_key_fn`` is the compatibility seam. Shipped callers (agent.py) always hand the
# layer a real ``CacheKey``, so every other shape below is reached only by a caller composing
# the cascade itself -- which is exactly why the arity walk and the degradation path are worth
# pinning: nothing in the product exercises them, and a silent mis-adaptation reads as a cache
# that simply never hits.
# ---------------------------------------------------------------------------

# One value per CacheKey field, in field order. Every value is distinct and self-describing so a
# field that lands in the wrong slot names the slot it wrongly landed in.
_FIELD_NAMES = ("goal_id", "frame", "roi", "entity_id", "condition", "context_fingerprint")


def _field_values() -> tuple:
    return (
        "goal-alpha",
        smooth_grid(),
        Box(0.0, 0.0, 0.5, 0.5),
        "entity-bravo",
        "the mug is on the desk",
        "fingerprint-charlie",
    )


class TestDefaultKeyFnArity(unittest.TestCase):
    """``_default_key_fn`` across every tuple arity it accepts."""

    def test_a_cache_key_passes_through_untouched(self):
        ck = CacheKey("g1", smooth_grid(), None, "e1", "cond", "fp")
        # identity, not equality: no copy, no re-coercion, no defaulting.
        self.assertIs(_default_key_fn(ck), ck)

    def test_every_arity_fills_the_fields_in_declared_order(self):
        supplied = _field_values()
        for arity in range(1, 7):
            with self.subTest(arity=arity):
                ck = _default_key_fn(supplied[:arity])
                for index, name in enumerate(_FIELD_NAMES):
                    expected = supplied[index] if index < arity else None
                    self.assertEqual(
                        getattr(ck, name),
                        expected,
                        f"arity {arity}: field {name!r} (position {index}) mis-adapted",
                    )

    def test_a_one_tuple_carries_no_frame_at_all(self):
        ck = _default_key_fn(("goal-alpha",))
        self.assertEqual(ck.goal_id, "goal-alpha")
        self.assertIsNone(ck.frame)
        # ...and a frameless key is un-hashable, so it can never occupy a slot: DEFER forever.
        layer = FrameHashLayer(CacheSettings(ttl_s=10.0, tau_bits=6), clock=_Clock(0.0))
        layer.store(("goal-alpha",), Verdict(True, 0.9), 0.9)
        self.assertEqual(len(layer), 0)  # store is a no-op, not an entry keyed on None
        self.assertEqual(layer.lookup(("goal-alpha",)).status, "DEFER")

    def test_a_six_tuple_equals_the_hand_built_cache_key(self):
        goal, frame, roi, entity, condition, fingerprint = _field_values()
        self.assertEqual(
            _default_key_fn((goal, frame, roi, entity, condition, fingerprint)),
            CacheKey(
                goal_id=goal,
                frame=frame,
                roi=roi,
                entity_id=entity,
                condition=condition,
                context_fingerprint=fingerprint,
            ),
        )

    def test_a_seventh_element_and_beyond_is_silently_ignored(self):
        supplied = _field_values()
        six = _default_key_fn(supplied)
        # An over-long tuple does not raise and does not shift anything: the tail is dropped.
        self.assertEqual(_default_key_fn(supplied + ("extra",)), six)
        self.assertEqual(_default_key_fn(supplied + ("extra", "more", None)), six)

    def test_the_tuple_path_coerces_the_goal_id_with_str(self):
        # a non-string goal id from a composing caller still keys as text.
        self.assertEqual(_default_key_fn((7, smooth_grid())).goal_id, "7")
        self.assertEqual(_default_key_fn((None, smooth_grid())).goal_id, "None")
        # the coercion is on the goal id ONLY -- the frame is passed through raw, because the
        # dHash needs the object, not its repr.
        self.assertEqual(_default_key_fn((7, 5)).frame, 5)

    def test_the_cache_key_path_does_not_coerce_so_the_two_forms_can_split_a_slot(self):
        # Documented asymmetry, not a claim that it is right: the tuple path str()-coerces the
        # goal id and the CacheKey pass-through does not, so an off-contract non-str goal id
        # occupies a DIFFERENT slot depending on which form the caller used.
        layer = FrameHashLayer(CacheSettings(ttl_s=10.0, tau_bits=6), clock=_Clock(0.0))
        grid = smooth_grid()
        layer.store(CacheKey(7, grid), Verdict(True, 0.9), 0.9)  # type: ignore[arg-type]
        self.assertEqual(layer.lookup((7, grid)).status, "DEFER")  # "7" != 7
        self.assertEqual(layer.invalidate("7"), 0)
        self.assertEqual(layer.invalidate(7), 1)  # type: ignore[arg-type]

    def test_an_empty_tuple_is_the_one_shape_that_raises(self):
        # The degradation path catches non-tuples only, so `()` reaches `key[0]`. Pinned as the
        # documented side of the boundary -- no shipped caller can produce it.
        with self.assertRaises(IndexError):
            _default_key_fn(())


class TestDefaultKeyFnNonTupleDegradation(unittest.TestCase):
    """A bare value degrades to ``CacheKey(goal_id=str(key), frame=key)`` -- it never raises.

    The consequence is that the value doubles as BOTH the slot's goal id and the hashed frame.
    """

    def test_a_bare_string_is_both_the_goal_id_and_the_frame(self):
        ck = _default_key_fn("sip-coffee")
        self.assertEqual(ck.goal_id, "sip-coffee")
        self.assertEqual(ck.frame, "sip-coffee")
        self.assertIsNone(ck.roi)
        self.assertIsNone(ck.entity_id)
        self.assertIsNone(ck.condition)
        self.assertIsNone(ck.context_fingerprint)

    def test_a_bare_int_keeps_the_raw_value_as_the_frame(self):
        ck = _default_key_fn(5)
        self.assertEqual(ck.goal_id, "5")  # goal id is the str()
        self.assertEqual(ck.frame, 5)  # the frame is NOT the str()
        self.assertNotIsInstance(ck.frame, str)

    def test_a_bare_bytes_frame_is_keyed_under_its_own_repr(self):
        frame = bytes(range(64))
        ck = _default_key_fn(frame)
        self.assertIs(ck.frame, frame)  # hashable: 64 bytes -> an 8x8 grid
        self.assertEqual(ck.goal_id, str(frame))
        # end-to-end: it really occupies the slot its own repr names.
        layer = FrameHashLayer(CacheSettings(ttl_s=10.0, tau_bits=6), clock=_Clock(0.0))
        layer.store(frame, Verdict(True, 0.9), 0.9)
        self.assertEqual(layer.lookup(frame).status, "HIT")
        self.assertEqual(layer.invalidate(str(frame)), 1)  # dropped by the frame's own repr

    def test_the_degraded_key_forfeits_near_match_because_the_frame_is_the_slot(self):
        # A near-identical frame (1 bit of Hamming distance, well inside tau_bits=6) HITs when
        # the caller supplies a real goal id, and MISSES on the degradation path -- because a
        # changed frame is a changed goal id, i.e. a different slot. Graceful, but the cache
        # stops being a cache.
        frame = bytes(range(64))
        near = bytes([255]) + frame[1:]
        self.assertEqual(hamming(dhash(frame), dhash(near)), 1)

        keyed = FrameHashLayer(CacheSettings(ttl_s=10.0, tau_bits=6), clock=_Clock(0.0))
        keyed.store(CacheKey("g1", frame), Verdict(True, 0.9), 0.9)
        self.assertEqual(keyed.lookup(CacheKey("g1", near)).status, "HIT")

        degraded = FrameHashLayer(CacheSettings(ttl_s=10.0, tau_bits=6), clock=_Clock(0.0))
        degraded.store(frame, Verdict(True, 0.9), 0.9)
        self.assertEqual(degraded.lookup(near).status, "DEFER")
        degraded.store(near, Verdict(True, 0.9), 0.9)
        self.assertEqual(len(degraded), 2)  # one slot per distinct frame

    def test_a_bare_tuple_of_tuples_frame_never_reaches_the_degradation_path(self):
        # `_grid_from_frame` explicitly supports tuple-of-tuples, but a tuple is a KEY here, so
        # such a frame is read as (goal_id, frame, roi, ...): row 0 becomes the goal id and row 1
        # becomes the "frame". Row 1 is a flat tuple of numbers -> not griddable -> dhash None.
        # The shape is inherently ambiguous with the key tuple; the layer resolves it by never
        # caching, rather than by raising.
        grid = tuple(tuple(float(x) for x in range(10)) for _ in range(10))
        ck = _default_key_fn(grid)
        self.assertEqual(ck.goal_id, str(grid[0]))
        self.assertEqual(ck.frame, grid[1])
        self.assertIsNone(dhash(ck.frame, None))
        layer = FrameHashLayer(CacheSettings(ttl_s=10.0, tau_bits=6), clock=_Clock(0.0))
        layer.store(grid, Verdict(True, 0.9), 0.9)
        self.assertEqual(len(layer), 0)
        self.assertEqual(layer.lookup(grid).status, "DEFER")


class TestSlotOrderIsAContract(unittest.TestCase):
    """``invalidate``/``invalidate_entity`` read ``slot[0]``/``slot[1]`` positionally."""

    def _key(self) -> CacheKey:
        # four distinct values, so a reordering cannot pass by coincidence.
        return CacheKey(
            goal_id="alpha",
            frame=smooth_grid(),
            roi=Box(0.0, 0.0, 0.5, 0.5),
            entity_id="bravo",
            condition="charlie",
            context_fingerprint="delta",
        )

    def test_slot_is_goal_entity_condition_fingerprint_positionally(self):
        slot = _slot(self._key())
        self.assertEqual(slot, ("alpha", "bravo", "charlie", "delta"))
        self.assertEqual(len(slot), 4)
        self.assertEqual(slot[0], "alpha")  # invalidate reads this
        self.assertEqual(slot[1], "bravo")  # invalidate_entity reads this

    def test_the_roi_is_deliberately_not_in_the_slot(self):
        ck = self._key()
        moved = CacheKey(
            goal_id=ck.goal_id,
            frame=ck.frame,
            roi=Box(0.5, 0.5, 0.5, 0.5),
            entity_id=ck.entity_id,
            condition=ck.condition,
            context_fingerprint=ck.context_fingerprint,
        )
        self.assertEqual(_slot(moved), _slot(ck))
        layer = FrameHashLayer(CacheSettings(ttl_s=10.0, tau_bits=6), clock=_Clock(0.0))
        layer.store(ck, Verdict(True, 0.9), 0.9)
        layer.store(moved, Verdict(True, 0.9), 0.9)
        self.assertEqual(len(layer), 1)  # the ROI moved -> the newest store wins the slot

    def test_invalidate_matches_position_zero_and_nothing_else(self):
        layer = FrameHashLayer(CacheSettings(ttl_s=10.0, tau_bits=6), clock=_Clock(0.0))
        layer.store(self._key(), Verdict(True, 0.9), 0.9)
        # entity / condition / fingerprint values must NOT be reachable as a goal id.
        for wrong in ("bravo", "charlie", "delta"):
            with self.subTest(goal_id=wrong):
                self.assertEqual(layer.invalidate(wrong), 0)
        self.assertEqual(layer.invalidate("alpha"), 1)

    def test_invalidate_entity_matches_position_one_and_nothing_else(self):
        layer = FrameHashLayer(CacheSettings(ttl_s=10.0, tau_bits=6), clock=_Clock(0.0))
        layer.store(self._key(), Verdict(True, 0.9), 0.9)
        for wrong in ("alpha", "charlie", "delta"):
            with self.subTest(entity_id=wrong):
                self.assertEqual(layer.invalidate_entity(wrong), 0)
        # ...and the goal scoping of invalidate_entity reads position zero for its own filter.
        self.assertEqual(layer.invalidate_entity("bravo", goal_id="charlie"), 0)
        self.assertEqual(layer.invalidate_entity("bravo", goal_id="alpha"), 1)


class TestTupleKeyEndToEnd(unittest.TestCase):
    """An adapted tuple occupies exactly the slot the tuple describes."""

    def setUp(self):
        self.clock = _Clock(0.0)
        self.layer = FrameHashLayer(CacheSettings(ttl_s=10.0, tau_bits=6), clock=self.clock)
        self.verdict = Verdict(satisfied=True, confidence=0.9, evidence="cached coffee")
        self.frame = smooth_grid()
        self.roi = Box(0.0, 0.0, 0.5, 0.5)
        self.key = (
            "goal-alpha",
            self.frame,
            self.roi,
            "entity-bravo",
            "the mug is on the desk",
            "fingerprint-charlie",
        )

    def test_a_six_tuple_store_is_found_by_the_equivalent_cache_key(self):
        self.layer.store(self.key, self.verdict, 0.9)
        self.assertEqual(len(self.layer), 1)
        hit = self.layer.lookup(
            CacheKey(
                goal_id="goal-alpha",
                frame=smooth_grid(),
                roi=self.roi,
                entity_id="entity-bravo",
                condition="the mug is on the desk",
                context_fingerprint="fingerprint-charlie",
            )
        )
        self.assertEqual(hit.status, "HIT")
        self.assertIs(hit.verdict, self.verdict)

    def test_the_fifth_element_really_is_the_condition(self):
        self.layer.store(self.key, self.verdict, 0.9)
        rearmed = self.key[:4] + ("the mug is in the sink",) + self.key[5:]
        self.assertEqual(self.layer.lookup(rearmed).status, "DEFER")
        self.assertEqual(self.layer.lookup(self.key).status, "HIT")  # the original still stands

    def test_the_sixth_element_really_is_the_context_fingerprint(self):
        self.layer.store(self.key, self.verdict, 0.9)
        moved_on = self.key[:5] + ("fingerprint-delta",)
        self.assertEqual(self.layer.lookup(moved_on).status, "DEFER")
        self.assertEqual(self.layer.lookup(self.key).status, "HIT")

    def test_the_fourth_element_is_what_invalidate_entity_drops(self):
        self.layer.store(self.key, self.verdict, 0.9)
        self.assertEqual(self.layer.invalidate_entity("entity-bravo"), 1)
        self.assertEqual(self.layer.lookup(self.key).status, "DEFER")

    def test_the_first_element_is_what_invalidate_drops_after_str_coercion(self):
        numeric = (7,) + self.key[1:]
        self.layer.store(numeric, self.verdict, 0.9)
        self.assertEqual(self.layer.invalidate(7), 0)  # type: ignore[arg-type]
        self.assertEqual(self.layer.invalidate("7"), 1)

    def test_a_four_tuple_shares_the_slot_a_six_tuple_leaves_defaulted(self):
        # the trailing fields default to None on BOTH paths, so the older four-field callers and
        # a six-field caller passing Nones are the same slot -- the documented back-compat claim.
        self.layer.store(self.key[:4], self.verdict, 0.9)
        hit = self.layer.lookup(self.key[:4] + (None, None))
        self.assertEqual(hit.status, "HIT")
        self.assertEqual(len(self.layer), 1)
        # ...and a condition-bearing key is a different slot entirely.
        self.assertEqual(self.layer.lookup(self.key).status, "DEFER")


if __name__ == "__main__":
    unittest.main()
