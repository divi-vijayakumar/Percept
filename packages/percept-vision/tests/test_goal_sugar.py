"""RB-L1SUGAR — ``watch(stream, "a condition")`` string sugar.

L1: offline, FakeVision, no keys. The bar these tests hold is that the sugar is a VERBATIM
pass-through with a deterministic, documented id — never a compilation step. Concretely:

* the condition reaching the judge is byte-for-byte the caller's string (no strip/lower/rewrite);
* the id is a pure function of that string (stable across calls/processes, no counters/clock);
* an explicitly-authored ``Goal`` is untouched by the sugar layer;
* nothing about the firing path changes — a string goal fires exactly where a Goal goal fires.
"""

from __future__ import annotations

import asyncio
import unittest

from percept import FIRE, Goal, Percept, WatchEvent, desugar_goal, desugar_goals, goal_id_for
from percept.adapters import Verdict
from percept.backends.fake import FakeVision
from percept.goalsugar import DIGEST_HEX, SLUG_MAX
from percept.watch import watch

CONDITION = "the milk is boiling over"


def _core(script: dict) -> Percept:
    return Percept.create(vision=FakeVision(script=script), discover_plugins=False)


async def _collect(stream, goals, profile=None, **kw) -> list[WatchEvent]:
    return [ev async for ev in watch(stream, goals, profile, **kw)]


class DesugarPassThroughTests(unittest.TestCase):
    """Pass-through, not compilation: the caller's words survive untouched."""

    def test_condition_is_verbatim(self):
        raw = "  The MILK is boiling over!!  (again)  "
        g = desugar_goal(raw)
        self.assertEqual(g.condition, raw)  # no strip, no case fold, no punctuation fixing
        self.assertEqual(g.say, raw)

    def test_no_attribute_compilation(self):
        """The goal carries one plain-language condition and nothing extracted from it."""
        g = desugar_goal(CONDITION)
        self.assertEqual(g.condition, CONDITION)
        # state is the untouched default watch state — no parsed object/attribute/threshold smuggled in.
        self.assertEqual(g.state, {"type": "watch", "count": 0})

    def test_unicode_and_emoji_survive(self):
        raw = "le chien est sur le canapé 🐕"
        self.assertEqual(desugar_goal(raw).condition, raw)

    def test_goal_instance_is_passed_through_identically(self):
        authored = Goal(id="zone", condition="a person is in the zone", say="alert")
        self.assertIs(desugar_goal(authored), authored)
        self.assertIs(desugar_goals([authored])[0], authored)

    def test_source_is_threaded_onto_desugared_goals(self):
        self.assertEqual(desugar_goal(CONDITION, source="door").source, "door")
        self.assertEqual(desugar_goal(CONDITION).source, "camera")


class DesugarIdTests(unittest.TestCase):
    """The documented id scheme: ``g_<slug>-<h8>``, a pure function of the verbatim condition."""

    def test_id_shape_is_documented(self):
        gid = goal_id_for(CONDITION)
        self.assertTrue(gid.startswith("g_the_milk_is_boiling_over-"), gid)
        slug, h = gid[2:].rsplit("-", 1)
        self.assertEqual(slug, "the_milk_is_boiling_over")
        self.assertEqual(len(h), DIGEST_HEX)
        self.assertTrue(all(c in "0123456789abcdef" for c in h))
        self.assertLessEqual(len(slug), SLUG_MAX)

    def test_id_is_deterministic(self):
        self.assertEqual(goal_id_for(CONDITION), goal_id_for(CONDITION))
        self.assertEqual(desugar_goal(CONDITION).id, desugar_goal(CONDITION).id)
        # no counter: position in the list must not influence the id
        a, b = desugar_goals([CONDITION, "a dog is on the couch"])
        c, d = desugar_goals(["a dog is on the couch", CONDITION])
        self.assertEqual(a.id, d.id)
        self.assertEqual(b.id, c.id)

    def test_slug_collisions_do_not_become_id_collisions(self):
        a, b = goal_id_for("a dog"), goal_id_for("A DOG!")
        self.assertEqual(a.split("-")[0], b.split("-")[0])  # same cosmetic slug
        self.assertNotEqual(a, b)  # different content hash

    def test_distinct_conditions_get_distinct_ids(self):
        ids = {goal_id_for(c) for c in (CONDITION, "a dog is on the couch", "the door is open")}
        self.assertEqual(len(ids), 3)

    def test_long_condition_slug_is_truncated_but_id_still_unique(self):
        long_a = "the " + ("very " * 40) + "big red dog is on the couch"
        long_b = "the " + ("very " * 40) + "big red cat is on the couch"
        ia, ib = goal_id_for(long_a), goal_id_for(long_b)
        self.assertLessEqual(len(ia[2:].rsplit("-", 1)[0]), SLUG_MAX)
        self.assertNotEqual(ia, ib)

    def test_non_ascii_only_condition_falls_back_to_readable_stem(self):
        gid = goal_id_for("🐕🐕🐕")
        self.assertTrue(gid.startswith("g_goal-"))


class DesugarRejectionTests(unittest.TestCase):
    def test_empty_condition_is_loud(self):
        for bad in ("", "   ", "\n\t"):
            with self.assertRaises(ValueError):
                desugar_goal(bad)

    def test_unsupported_type_is_loud(self):
        for bad in (1, None, object(), {"condition": "x"}):
            with self.assertRaises(TypeError):
                desugar_goal(bad)

    def test_goals_argument_rejects_non_iterable(self):
        with self.assertRaises(TypeError):
            desugar_goals(None)
        with self.assertRaises(TypeError):
            desugar_goals(7)


class DesugarGoalsShapeTests(unittest.TestCase):
    def test_bare_string_becomes_one_goal(self):
        out = desugar_goals(CONDITION)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].condition, CONDITION)

    def test_bare_goal_becomes_one_goal(self):
        g = Goal(id="zone", condition="x", say="y")
        self.assertEqual(desugar_goals(g), [g])

    def test_mixed_list_preserves_order(self):
        authored = Goal(id="zone", condition="a person is in the zone", say="alert")
        out = desugar_goals([CONDITION, authored, "the door is open"])
        self.assertEqual([g.condition for g in out],
                         [CONDITION, "a person is in the zone", "the door is open"])

    def test_tuple_and_generator_accepted(self):
        self.assertEqual(len(desugar_goals((CONDITION, "the door is open"))), 2)
        self.assertEqual(len(desugar_goals(c for c in (CONDITION,))), 1)


class WatchStringSugarTests(unittest.TestCase):
    """End-to-end through the frozen ``watch()`` seam — same firing path, no new behavior."""

    def test_watch_accepts_a_bare_string(self):
        core = _core({"boil": Verdict(satisfied=True, confidence=0.95, evidence="milk over the rim")})
        events = asyncio.run(_collect(["boil"], CONDITION, percept=core))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].action, FIRE)
        self.assertEqual(events[0].goal_id, goal_id_for(CONDITION))
        self.assertEqual(events[0].text, CONDITION)  # say mirrors the condition verbatim

    def test_watch_accepts_a_list_of_strings(self):
        core = _core({"boil": Verdict(satisfied=True, confidence=0.95, evidence="over the rim")})
        events = asyncio.run(_collect(["boil"], [CONDITION, "the door is open"], percept=core))
        self.assertEqual({e.goal_id for e in events},
                         {goal_id_for(CONDITION), goal_id_for("the door is open")})

    def test_string_and_equivalent_goal_fire_identically(self):
        script = {"boil": Verdict(satisfied=True, confidence=0.95, evidence="over the rim")}
        from_string = asyncio.run(_collect(["boil"], CONDITION, percept=_core(script)))
        explicit = Goal(id=goal_id_for(CONDITION), condition=CONDITION, say=CONDITION)
        from_goal = asyncio.run(_collect(["boil"], [explicit], percept=_core(script)))
        self.assertEqual(len(from_string), len(from_goal))
        self.assertEqual(
            [(e.goal_id, e.action, e.text) for e in from_string],
            [(e.goal_id, e.action, e.text) for e in from_goal],
        )

    def test_registered_goal_condition_reaches_the_core_verbatim(self):
        core = _core({})
        asyncio.run(_collect([], CONDITION, percept=core))
        armed = core.goals.get(goal_id_for(CONDITION))
        self.assertIsNotNone(armed)
        self.assertEqual(armed.condition, CONDITION)

    def test_watch_source_is_applied_to_string_goals(self):
        core = _core({})
        asyncio.run(_collect([], CONDITION, percept=core, source="kitchen"))
        self.assertEqual(core.goals.get(goal_id_for(CONDITION)).source, "kitchen")

    def test_bad_goal_type_raises_before_any_frame_is_consumed(self):
        consumed = []

        def _stream():
            consumed.append(1)
            yield "boil"

        with self.assertRaises(TypeError):
            asyncio.run(_collect(_stream(), [42], percept=_core({})))
        self.assertEqual(consumed, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
