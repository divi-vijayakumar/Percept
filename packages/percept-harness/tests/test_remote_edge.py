"""RemoteEdgeReceiver (L1) — the TOPOLOGY-B server shim that consumes a remote on-device edge.

Deterministic stdlib tests (no model, no keys):
  * capability handshake — watchspec() filters a goal's detectors to the edge's advertised skills.
  * boundary consumption — on_boundaries() maps each BoundaryReport event 1:1 to perceive_boundary,
    exactly like GatedWatchLoop's inline count_boundary, but driven by the wire.
"""

from __future__ import annotations

import unittest

from percept.adapters import Goal

from percept_harness.tier0 import RemoteEdgeReceiver


class _Goals:
    def __init__(self, goals):
        self._goals = list(goals)

    def active(self):
        return list(self._goals)


class _FakeCore:
    """Minimal core: only the surface RemoteEdgeReceiver touches (goals.active + perceive_boundary)."""

    def __init__(self, goals=()):
        self.goals = _Goals(goals)
        self.boundary_calls = []

    def perceive_boundary(self, goal_id, *, t):
        self.boundary_calls.append((goal_id, t))
        return None


def _rep_goal(detectors):
    return Goal(
        id="reps", condition="count squats", say="done",
        state={"type": "watch", "detectors": detectors},
    )


class TestRemoteEdgeReceiver(unittest.TestCase):
    def test_watchspec_filters_to_advertised_capabilities(self):
        core = _FakeCore([_rep_goal(["motion-periodicity", "pose-openness"])])
        recv = RemoteEdgeReceiver(core)
        recv.on_capabilities(["motion-periodicity"])  # a glasses-class edge — no pose model
        spec = recv.watchspec(threshold=0.1)
        self.assertEqual(spec.conditions[0].detectors, ["motion-periodicity"])

    def test_no_advertisement_sends_full_selection(self):
        core = _FakeCore([_rep_goal(["motion-periodicity", "pose-openness"])])
        recv = RemoteEdgeReceiver(core)  # never advertised -> capabilities None -> send the full selection
        self.assertIsNone(recv.capabilities)
        spec = recv.watchspec(threshold=0.1)
        self.assertEqual(spec.conditions[0].detectors, ["motion-periodicity", "pose-openness"])

    def test_on_boundaries_maps_each_event_to_perceive_boundary(self):
        core = _FakeCore()
        report = {
            "boundaries": [
                {"goal_id": "reps", "detector": "motion-periodicity", "kind": "peak", "t": 12.3, "value": 0.9},
                {"goal_id": "reps", "detector": "motion-periodicity", "kind": "peak", "t": 14.0, "value": 0.8},
            ]
        }
        RemoteEdgeReceiver(core).on_boundaries(report)
        self.assertEqual(core.boundary_calls, [("reps", 12.3), ("reps", 14.0)])

    def test_on_boundaries_accepts_a_bare_list(self):
        core = _FakeCore()
        RemoteEdgeReceiver(core).on_boundaries(
            [{"goal_id": "g", "detector": "acoustic-onset", "kind": "onset", "t": 4.0, "value": 0.5}]
        )
        self.assertEqual(core.boundary_calls, [("g", 4.0)])

    def test_returns_fire_events_from_perceive_boundary(self):
        core = _FakeCore()
        core.perceive_boundary = lambda goal_id, *, t: {"goal_id": goal_id, "fired_at": t}
        fires = RemoteEdgeReceiver(core).on_boundaries(
            {"boundaries": [{"goal_id": "reps", "detector": "d", "kind": "peak", "t": 1.0, "value": 1.0}]}
        )
        self.assertEqual(fires, [{"goal_id": "reps", "fired_at": 1.0}])


if __name__ == "__main__":
    unittest.main()
