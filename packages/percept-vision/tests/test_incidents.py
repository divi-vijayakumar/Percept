"""Incident producer (L1) — FireEvent -> Incident + snippet assembly from the rolling FrameWindow."""
from __future__ import annotations

import unittest

from percept.events import FrameWindow
from percept.incidents import make_incident, snippet_from_window
from percept_contracts import IncidentSnippet, Verdict


class _Fire:
    """Duck-typed stand-in for a FireEvent (goal_id / text / verdict)."""

    def __init__(self, goal_id, text, verdict=None):
        self.goal_id = goal_id
        self.text = text
        self.verdict = verdict


class MakeIncidentTests(unittest.TestCase):
    def test_maps_fire_to_incident(self):
        v = Verdict(satisfied=True, confidence=0.9, evidence="person on the floor, not moving")
        inc = make_incident(_Fire("fall", "Possible fall - check on them", v), ts=12.5)
        self.assertEqual(inc.concern_id, "fall")
        self.assertEqual(inc.verdict, v)
        self.assertEqual(inc.message, "Possible fall - check on them")
        self.assertEqual(inc.ts, 12.5)
        self.assertEqual(inc.id, "inc-fall-12.500")
        self.assertIsNone(inc.label)  # unlabelled until a human / the calibration loop classifies it

    def test_none_verdict_gets_a_placeholder(self):
        inc = make_incident(_Fire("g", "hi", None), ts=1.0)
        self.assertEqual(inc.verdict.confidence, 0.0)
        self.assertEqual(inc.message, "hi")

    def test_explicit_id_and_snippet_passthrough(self):
        snip = IncidentSnippet(kind="frame", frames=["f"])
        inc = make_incident(_Fire("g", "hi"), ts=1.0, id="X-1", snippet=snip)
        self.assertEqual(inc.id, "X-1")
        self.assertIs(inc.snippet, snip)


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
