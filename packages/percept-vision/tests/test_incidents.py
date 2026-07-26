"""Incident producer (L1) — FireEvent -> Incident + snippet assembly from the rolling FrameWindow."""
from __future__ import annotations

import unittest

from percept.events import FrameWindow
from percept.incidents import incident_judgment, make_incident, snippet_from_window
from percept.contracts import Incident, IncidentSnippet, Verdict
from percept.runtime.sinks import incident_payload


class _Fire:
    """Duck-typed stand-in for a FireEvent (goal_id / text / verdict)."""

    def __init__(self, goal_id, text, verdict=None, entity_id=None):
        self.goal_id = goal_id
        self.text = text
        self.verdict = verdict
        self.entity_id = entity_id


class MakeIncidentTests(unittest.TestCase):
    def test_maps_fire_to_incident(self):
        v = Verdict(satisfied=True, confidence=0.9, evidence="person on the floor, not moving")
        inc = make_incident(_Fire("fall", "Possible fall - check on them", v), ts=12.5)
        self.assertEqual(inc.concern_id, "fall")
        self.assertEqual(inc.verdict, v)
        self.assertEqual(inc.message, "Possible fall - check on them")
        self.assertEqual(inc.ts, 12.5)
        self.assertRegex(inc.id, r"^inc-fall-12\.500-[0-9a-f]{8}$")
        self.assertIsNone(inc.label)  # unlabelled until a human / the calibration loop classifies it
        self.assertIsNone(inc.entity_id)  # not entity-scoped -> the back-compatible default

    def test_none_verdict_gets_a_placeholder(self):
        inc = make_incident(_Fire("g", "hi", None), ts=1.0)
        self.assertEqual(inc.verdict.confidence, 0.0)
        self.assertEqual(inc.message, "hi")

    def test_explicit_id_and_snippet_passthrough(self):
        snip = IncidentSnippet(kind="frame", frames=["f"])
        inc = make_incident(_Fire("g", "hi"), ts=1.0, id="X-1", snippet=snip)
        self.assertEqual(inc.id, "X-1")
        self.assertIs(inc.snippet, snip)

    def test_preserves_exact_runtime_local_judgment(self):
        judgment = object()
        fire = _Fire("g", "hi")
        fire.judgment = judgment

        incident = make_incident(fire, ts=1.0)

        self.assertIs(incident_judgment(incident), judgment)
        self.assertNotIn("_judgment", incident_payload(incident))


class IncidentEntityIdTests(unittest.TestCase):
    """The gate key is (goal_id, entity_id); identity must survive the emission boundary."""

    def test_entity_id_carries_from_the_fire(self):
        inc = make_incident(_Fire("caffeine", "stay off caffeine", None, entity_id="cup2"), ts=3.0)
        self.assertEqual(inc.entity_id, "cup2")

    def test_two_subjects_on_one_concern_stay_distinguishable(self):
        # the client-reported failure: interleaved fires from two subjects read as out-of-order
        # divergence because every incident looked identical downstream.
        a = make_incident(_Fire("zone", "left the zone", None, entity_id="person-1"), ts=1.0)
        b = make_incident(_Fire("zone", "left the zone", None, entity_id="person-2"), ts=1.5)
        c = make_incident(_Fire("zone", "left the zone", None, entity_id="person-1"), ts=2.0)
        self.assertEqual([i.entity_id for i in (a, b, c)], ["person-1", "person-2", "person-1"])
        self.assertEqual(a.concern_id, b.concern_id)  # same concern...
        self.assertNotEqual(a.entity_id, b.entity_id)  # ...different subject

    def test_absent_entity_id_defaults_to_none(self):
        class _Legacy:  # a legacy duck-typed fire with NO entity_id attribute at all
            goal_id, text, verdict = "g", "hi", None

        self.assertIsNone(make_incident(_Legacy(), ts=1.0).entity_id)

    def test_explicit_override_wins_and_non_str_is_coerced(self):
        fire = _Fire("g", "hi", None, entity_id="from-fire")
        self.assertEqual(make_incident(fire, ts=1.0, entity_id="explicit").entity_id, "explicit")
        self.assertEqual(make_incident(_Fire("g", "hi", None, entity_id=7), ts=1.0).entity_id, "7")

    def test_positional_construction_is_unchanged(self):
        # entity_id is appended LAST on the dataclass, so every pre-1.3 positional build still means
        # what it meant (a silent field reorder would corrupt existing callers).
        inc = Incident("i", "c", Verdict(satisfied=True, confidence=1.0), 2.0, "real")
        self.assertEqual((inc.label, inc.entity_id), ("real", None))

    def test_end_to_end_through_the_real_gate(self):
        """The gate key IS (goal_id, entity_id) — drive the REAL firing path and prove the entity the
        gate keyed on is the entity the emitted Incident carries."""
        from percept.adapters import Goal
        from percept.agent import Percept
        from percept.backends.fake import FakeVision
        from percept.gate import FIRE

        a = Percept.create(vision=FakeVision(script={}), discover_plugins=False)
        g = Goal(id="caffeine", condition="c", say="stay off caffeine")
        ev = a._fire_concern(g, Verdict(True, 0.95, "yes"), ("caffeine", "cup2"), freshness=1.0, t=0.0)
        self.assertEqual(ev.action, FIRE)
        inc = make_incident(ev, ts=0.0)
        self.assertEqual((inc.concern_id, inc.entity_id), ("caffeine", "cup2"))

    def test_serializes_through_the_webhook_payload(self):
        inc = make_incident(_Fire("g", "hi", None, entity_id="e9"), ts=1.0)
        self.assertEqual(incident_payload(inc)["entity_id"], "e9")
        self.assertIsNone(incident_payload(make_incident(_Fire("g", "hi"), ts=1.0))["entity_id"])


class SnippetFromWindowTests(unittest.TestCase):
    def _window(self, n):
        w = FrameWindow(size=8)
        for t in range(n):
            w.push(float(t), f"f{t}")
        return w

    def test_assembles_strip_from_window(self):
        snip = snippet_from_window(self._window(5), kind="frame")
        self.assertEqual(snip.kind, "frame")
        self.assertEqual(snip.frames, ["f0", "f1", "f2", "f3", "f4"])
        self.assertEqual(snip.t_start, 0.0)
        self.assertEqual(snip.t_end, 4.0)

    def test_pre_s_bounds_by_time(self):
        snip = snippet_from_window(self._window(5), pre_s=2.0)  # t_end=4 -> keep t>=2
        self.assertEqual(snip.frames, ["f2", "f3", "f4"])

    def test_empty_window_is_none(self):
        self.assertIsNone(snippet_from_window(FrameWindow()))


if __name__ == "__main__":
    unittest.main()
