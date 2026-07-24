"""P06-002 — the author's own focused proof (the full 8-test acceptance suite is written separately).

Two things this module pins:

1. ``transition_latch_key`` — every row of the mapping table, including BOTH scene fallbacks and the
   invariant that the fallback is ALWAYS the literal ``"scene"``, never a per-tick value.
2. ``Percept._perceive_transition`` — a scene-scoped transition goal (no identity_policy declared)
   fires across a focus flip in BOTH directions (e1->scene and scene->e1), because the rising-edge
   latch is keyed by the declared identity (default stable "scene"), not the volatile focus.id. And
   the control: WITH a track policy declared, the latch isolates per entity (a flip suppresses).
"""
from __future__ import annotations

import asyncio
import unittest

from percept.adapters import Goal, Verdict
from percept.agent import Percept
from percept.gate import FIRE
from percept.identity_policy import SCENE_IDENTITY, transition_latch_key

FROM = "the light is red"
TO = "the light is green"


class TestTransitionLatchKey(unittest.TestCase):
    def test_none_declared_defaults_to_scene(self):
        self.assertEqual(transition_latch_key(None, "e1"), (SCENE_IDENTITY, "default_scene"))
        # focus present or not, undeclared always scopes to scene.
        self.assertEqual(transition_latch_key(None, None), (SCENE_IDENTITY, "default_scene"))

    def test_declared_scene_is_scene(self):
        self.assertEqual(transition_latch_key("scene", "e1"), (SCENE_IDENTITY, "declared"))
        self.assertEqual(transition_latch_key(SCENE_IDENTITY, None), (SCENE_IDENTITY, "declared"))

    def test_track_policy_with_focus_scopes_to_entity(self):
        self.assertEqual(transition_latch_key("track_iou_label", "e7"), ("e7", "declared"))
        self.assertEqual(
            transition_latch_key("appearance_recent_set", "e7"), ("e7", "declared")
        )

    def test_track_policy_without_focus_falls_back_to_scene(self):
        self.assertEqual(
            transition_latch_key("track_iou_label", None), (SCENE_IDENTITY, "fallback_scene")
        )
        self.assertEqual(
            transition_latch_key("appearance_recent_set", None),
            (SCENE_IDENTITY, "fallback_scene"),
        )

    def test_other_policies_fall_back_to_scene(self):
        # content_key / speaker_label / an unknown name are NOT per-entity track policies: they
        # fail toward the stable scene scope, never a per-tick focus value.
        for policy in ("content_key", "speaker_label", "some_custom_resolver"):
            self.assertEqual(
                transition_latch_key(policy, "e1"), (SCENE_IDENTITY, "fallback_scene")
            )
            self.assertEqual(
                transition_latch_key(policy, None), (SCENE_IDENTITY, "fallback_scene")
            )

    def test_fallback_is_always_the_literal_scene(self):
        # The load-bearing invariant: no branch ever returns focus_id as the fallback.
        for key, source in (
            transition_latch_key("content_key", "e1"),
            transition_latch_key("track_iou_label", None),
            transition_latch_key(None, "e1"),
        ):
            if source in ("default_scene", "fallback_scene", "declared"):
                # every scene branch yields the literal string, never the entity id
                if source != "declared" or key == SCENE_IDENTITY:
                    self.assertEqual(key, "scene")


class _StubVision:
    """(condition, frame_token) -> (satisfied, confidence); detect yields no entity."""

    def __init__(self, script):
        self._script = script

    async def judge(self, condition, frame, *, roi=None, exemplar=None, corrections=None,
                    frames=None, context=None):
        sat, conf = self._script.get((condition, str(frame)), (False, 0.9))
        return Verdict(satisfied=sat, confidence=conf)

    async def detect(self, frame, goals):
        return []

    async def detect_events(self, frames, goals):
        return []


class _Focus:
    """Minimal stand-in for an in-focus Entity: id + box + asserted()."""

    def __init__(self, id):
        self.id = id
        self.box = None

    def asserted(self):
        return {}


def _goal(gid="signal", state_extra=None):
    state = {"type": "transition", "from": FROM, "to": TO, "count": 0}
    if state_extra:
        state.update(state_extra)
    return Goal(id=gid, condition="the signal changed", say="go", state=state)


def _perceive(core, goal, frame, focus):
    return asyncio.run(core._perceive_transition(
        goal, frame, focus, now=1.0, t=1.0, ingest_observed_at=1.0,
        trace_id="t", tracing=False, source="camera",
    ))


def _drive(script, frames):
    """frames: list of (token, focus_or_None). Returns list of FireEvents."""
    core = Percept.create(vision=_StubVision(script), discover_plugins=False)
    g = _goal()
    core.add_goal(g)
    fires = []
    for tok, focus in frames:
        ev = _perceive(core, g, tok, focus)
        if ev is not None and ev.action == FIRE:
            fires.append(ev)
    return fires


class TestScenePerceiveTransition(unittest.TestCase):
    def _arm_then_fire(self, tok):
        return {(FROM, tok): (True, 0.95), (TO, tok): (False, 0.9)}

    def test_scene_goal_fires_when_focus_flips_e1_to_scene(self):
        # arm (from=yes) while entity e1 is in focus, complete (to=yes) after focus drops to scene.
        script = {
            (FROM, "f1"): (True, 0.95), (TO, "f1"): (False, 0.9),   # armed under e1
            (FROM, "f2"): (False, 0.9), (TO, "f2"): (True, 0.95),   # to=yes, focus now None
        }
        fires = _drive(script, [("f1", _Focus("e1")), ("f2", None)])
        self.assertEqual(len(fires), 1)

    def test_scene_goal_fires_when_focus_flips_scene_to_e1(self):
        script = {
            (FROM, "f1"): (True, 0.95), (TO, "f1"): (False, 0.9),   # armed under scene (no focus)
            (FROM, "f2"): (False, 0.9), (TO, "f2"): (True, 0.95),   # to=yes, focus now e1
        }
        fires = _drive(script, [("f1", None), ("f2", _Focus("e1"))])
        self.assertEqual(len(fires), 1)

    def test_track_policy_isolates_per_entity_flip_suppresses(self):
        # WITH a track policy declared, arming under e1 and completing while e2 is in focus latches
        # under DIFFERENT keys — the edge on e2 was never armed, so it must NOT fire.
        script = {
            (FROM, "f1"): (True, 0.95), (TO, "f1"): (False, 0.9),   # arm under e1
            (FROM, "f2"): (False, 0.9), (TO, "f2"): (True, 0.95),   # to=yes under e2 (never armed)
        }
        core = Percept.create(vision=_StubVision(script), discover_plugins=False)
        g = _goal(state_extra={"identity_policy": "track_iou_label"})
        core.add_goal(g)
        ev1 = _perceive(core, g, "f1", _Focus("e1"))
        ev2 = _perceive(core, g, "f2", _Focus("e2"))
        self.assertIsNone(ev1)  # arming tick never fires
        self.assertIsNone(ev2)  # e2 was never armed -> no fire under per-entity isolation

    def test_track_policy_no_focus_does_not_leak_to_scene_latch(self):
        # P06-002 review finding: a per-entity goal must NOT arm/fire the shared "scene" latch during
        # a focus dropout. Both the arm (from=yes) and the completion (to=yes) happen under focus=None:
        # the pre-fix fallback-to-scene would arm then FIRE "scene", conflating entities; the fix SKIPS
        # unfocused per-entity ticks, so nothing fires.
        script = {
            (FROM, "f1"): (True, 0.95), (TO, "f1"): (False, 0.9),   # from=yes, focus None
            (FROM, "f2"): (False, 0.9), (TO, "f2"): (True, 0.95),   # to=yes, focus None
        }
        core = Percept.create(vision=_StubVision(script), discover_plugins=False)
        g = _goal(state_extra={"identity_policy": "track_iou_label"})
        core.add_goal(g)
        self.assertIsNone(_perceive(core, g, "f1", None))
        self.assertIsNone(_perceive(core, g, "f2", None))

    def test_track_policy_fire_defers_across_focus_dropout(self):
        # The complement: a per-entity edge that spans a dropout still fires once the subject returns.
        # arm under e1, a no-focus dropout tick (skipped, latch untouched), complete under e1 -> fires.
        script = {
            (FROM, "f1"): (True, 0.95), (TO, "f1"): (False, 0.9),   # arm under e1
            (FROM, "f2"): (False, 0.9), (TO, "f2"): (False, 0.9),   # dropout (focus None -> skipped)
            (FROM, "f3"): (False, 0.9), (TO, "f3"): (True, 0.95),   # complete under e1
        }
        core = Percept.create(vision=_StubVision(script), discover_plugins=False)
        g = _goal(state_extra={"identity_policy": "track_iou_label"})
        core.add_goal(g)
        fires = []
        for tok, focus in [("f1", _Focus("e1")), ("f2", None), ("f3", _Focus("e1"))]:
            ev = _perceive(core, g, tok, focus)
            if ev is not None and ev.action == FIRE:
                fires.append(ev)
        self.assertEqual(len(fires), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
