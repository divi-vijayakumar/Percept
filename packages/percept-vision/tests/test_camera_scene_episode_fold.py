"""— a camera watch gets a SCENE-scoped episode, and therefore TTL-invariant idempotency.

The hole this closes. ``Percept._document_episode_fold`` needs a content key and
``Percept._claim_episode_fold`` needs a claim slot. The physical-scene profile declares
``identity_strategy='track_iou_label'``, so an ordinary camera concern — one that authored no
``identity_policy`` — resolved to neither and built NO fold at all. Alone among the shipped source
classes it had no episode-owned idempotency: while a condition kept holding, a TTL/LRU eviction of its
gate key re-created that key ARMED and re-reported an occurrence that never stopped.

The fix is a SCENE-keyed subject, not a per-entity one. Per-entity episodes are refused on measured
grounds (scene entity ids churn: 58,101 ids in 82 minutes, p50 track lifetime 0.0s), and the price of
scene keying is stated plainly in ``TestSceneSingletonCannotCountSubjects`` below: a second entity
entering an already-reported open occurrence is idempotently suppressed. Counting subjects is the
gate's per-entity key's job, and this change touches neither that key nor ``percept/gate.py``.

What the suite pins:

1. the defect and the fix — a camera concern resolves to a per-entity policy and now folds
   (``TestCameraConcernFolds``);
2. the occurrence lifecycle end to end through ``perceive_judged`` — open, suppress, close on a
   confident-false, re-fire (``TestSceneOccurrenceLifecycle``);
3. the invariant that motivates it — a gate-key TTL sweep mid-occurrence produces no duplicate, with
   the fold-less control proving the fold is load-bearing (``TestTtlInvariance``);
4. the scope fences — every non-default re-arm mode, every non-watch route, and the content-key /
   claim-slot paths stay on their byte-for-byte gate-only behaviour (``TestScopeFences``);
5. the refusals — no fabricated per-entity key, no out-of-order corruption, no abstention side effects
   (``TestRefusals``).
"""

from __future__ import annotations

import asyncio
import unittest

from percept.adapters import Goal, Verdict
from percept.agent import Percept
from percept.backends.fake import FakeVision
from percept.config import GateSettings, Settings
from percept.contracts import SubjectRef
from percept.identity_policy import SCENE_IDENTITY
from percept.source_profiles_default import PHYSICAL_SCENE, SCREEN_DOCUMENT

_YES = Verdict(True, 0.95, "a person is at the door")
_NO = Verdict(False, 0.95, "the doorway is empty")
_MAYBE = Verdict(True, 0.3, "something, maybe")


def _build(*, source_profile=PHYSICAL_SCENE, state: "dict | None" = None) -> Percept:
    settings = Settings(gate=GateSettings(threshold=0.7, validity_window_s=1000.0))
    a = Percept.create(
        vision=FakeVision(), settings=settings, discover_plugins=False,
        source_profile=source_profile,
    )
    a.add_goal(
        Goal(
            id="g1",
            condition="a person is at the door",
            say="someone is at the door",
            state={"type": "watch", "count": 0, **(state or {})},
        )
    )
    return a


def _goal(a: Percept) -> Goal:
    return a.goals.get("g1")


def _drive(a: Percept, stream, *, sweep_at=None):
    """Replay ``stream`` = [(t, verdict)] through the real judged path; return the actions per tick.

    ``sweep_at`` maps a step index -> the ``t`` at which an UNRELATED gate key is evaluated first,
    forcing the TTL sweep that evicts the (idle) concern key exactly as background traffic would.
    """
    a.vision = FakeVision(script={f"f{i}": v for i, (_, v) in enumerate(stream)})
    actions = []
    for i, (t, _) in enumerate(stream):
        if sweep_at and i in sweep_at:
            a.gate.evaluate(("__housekeeping__", None), Verdict(False, 1.0), now=sweep_at[i])
        events = asyncio.run(a.perceive_judged(f"f{i}", t=t))
        actions.append([e.action for e in events])
    return actions


# --------------------------------------------------------------------------------------------------
# 1 — the defect, and that it is now closed.
# --------------------------------------------------------------------------------------------------


class TestCameraConcernFolds(unittest.TestCase):
    def test_camera_concern_resolves_to_a_per_entity_policy(self):
        # The premise: an undeclared concern on a camera inherits the profile's per-entity strategy,
        # which is neither content_key nor claim_slot -- the two the folds used to require.
        a = _build()
        self.assertEqual(a._resolved_identity_policy(_goal(a)), "track_iou_label")

    def test_it_now_builds_a_fold_keyed_on_the_scene_scope(self):
        a = _build()
        fold = a._document_episode_fold(_goal(a), _YES, 0.0, truth="yes")
        self.assertIsNotNone(fold, "a judged camera concern must not be left without a fold")
        self.assertEqual(fold.match, "new")
        self.assertTrue(fold.should_rearm)
        self.assertTrue(fold.should_evaluate_gate)
        episodes = a.episodes["g1"]._episodes
        self.assertEqual([e.subject_key for e in episodes], [SCENE_IDENTITY])

    def test_the_continuation_policy_comes_from_the_source_profile(self):
        # Not a constant invented at the call site: the profile's declared capability.
        a = _build()
        a._document_episode_fold(_goal(a), _YES, 0.0, truth="yes")
        self.assertEqual(a.episodes["g1"].policy, PHYSICAL_SCENE.default_episode_policy)


# --------------------------------------------------------------------------------------------------
# 2 — the occurrence lifecycle, end to end through the real judged path.
# --------------------------------------------------------------------------------------------------


class TestSceneOccurrenceLifecycle(unittest.TestCase):
    def test_open_suppress_close_refire(self):
        a = _build()
        actions = _drive(a, [(0.0, _YES), (1.0, _YES), (2.0, _NO), (3.0, _YES)])
        self.assertEqual(
            actions, [["fire"], [], [], ["fire"]],
            "one incident per occurrence: the re-statement is suppressed, the SECOND occurrence fires",
        )
        episodes = a.episodes["g1"]._episodes
        self.assertEqual([e.status for e in episodes], ["closed", "open"])
        self.assertEqual({e.subject_key for e in episodes}, {SCENE_IDENTITY})
        self.assertEqual(a.metrics.get("episode.scene.closed"), 1)
        self.assertEqual(a.metrics.get("gate.idempotent_suppressed"), 1)

    def test_a_held_condition_is_reported_exactly_once(self):
        a = _build()
        actions = _drive(a, [(float(t), _YES) for t in range(6)])
        self.assertEqual([act for act in actions if act], [["fire"]])
        self.assertEqual(len(a.episodes["g1"]._episodes), 1, "one unbroken occurrence, one episode")

    def test_without_a_confident_false_boundary_the_occurrence_stays_open(self):
        # An ASK does not end an occurrence, so the episode is not closed and the return to a
        # confident yes is still the SAME occurrence -- not a second incident.
        a = _build()
        actions = _drive(a, [(0.0, _YES), (1.0, _MAYBE), (2.0, _YES)])
        self.assertEqual(actions, [["fire"], ["ask"], []])
        self.assertEqual(len(a.episodes["g1"]._episodes), 1)


# --------------------------------------------------------------------------------------------------
# 3 — THE invariant this exists for: gate-key housekeeping must not be user-visible.
# --------------------------------------------------------------------------------------------------


class TestTtlInvariance(unittest.TestCase):
    _STREAM = [(0.0, _YES), (61.0, _YES), (120.0, _YES)]

    def test_ttl_eviction_mid_occurrence_does_not_refire(self):
        a = _build()
        a.gate.key_ttl_s = 60.0  # a bound the gate already supports; no gate mechanics change
        actions = _drive(a, self._STREAM, sweep_at={1: 61.0})
        self.assertEqual(actions, [["fire"], [], []])
        self.assertEqual(len(a.episodes["g1"]._episodes), 1, "the occurrence never stopped holding")

    def test_without_the_fold_the_evicted_key_does_refire(self):
        # The counter-proof that the fold is load-bearing. A profile-less core resolves no identity
        # policy, so it builds no fold and runs the gate-only path -- and the swept key reads the
        # still-open occurrence as a fresh rising edge.
        a = _build(source_profile=None)
        a.gate.key_ttl_s = 60.0
        self.assertIsNone(a._document_episode_fold(_goal(a), _YES, 0.0, truth="yes"))
        actions = _drive(a, self._STREAM, sweep_at={1: 61.0})
        self.assertEqual(actions, [["fire"], ["fire"], []], "the leak the scene fold closes")


# --------------------------------------------------------------------------------------------------
# 4 — the scope fences. Everything outside them keeps its byte-for-byte gate-only path.
# --------------------------------------------------------------------------------------------------


class TestScopeFences(unittest.TestCase):
    def test_non_default_rearm_modes_get_no_fold(self):
        # Each of these owns its own re-arm rule; a scene fold suppressing the gate ahead of one of
        # them would silently delete its firing.
        for rearm in ("never", "interval", "deadline", "each_event"):
            with self.subTest(rearm=rearm):
                a = _build(state={"rearm": rearm, "cadence_s": 10.0, "deadline_after_s": 10.0})
                self.assertIsNone(a._document_episode_fold(_goal(a), _YES, 0.0, truth="yes"))
                self.assertNotIn("g1", a.episodes)

    def test_a_fire_once_watch_still_fires_once(self):
        a = _build(state={"rearm": "never"})
        actions = _drive(a, [(0.0, _YES), (1.0, _NO), (2.0, _YES)])
        self.assertEqual(actions, [["fire"], [], []], "the never-contract is untouched")

    def test_non_watch_routes_get_no_fold(self):
        # Counting/rhythm/event goals own Percept.episodes for their OWN subjects; a scene subject
        # must never be written into those logs.
        for gtype in ("count", "rhythm", "event", "transition", "process"):
            with self.subTest(type=gtype):
                a = _build(state={"type": gtype, "from": "a", "to": "b", "event_type": "e"})
                self.assertIsNone(a._document_episode_fold(_goal(a), _YES, 0.0, truth="yes"))
                self.assertNotIn("g1", a.episodes)

    def test_a_declared_content_key_concern_still_takes_the_document_path(self):
        # The concern wins over the profile default, and the document path is unchanged: it groups by
        # the judge-emitted content key, not by the scene.
        a = _build(state={"identity_policy": "content_key"})
        ref = SubjectRef(
            scheme="content", namespace="workflow", key="PO-88", resolution="bound",
            scope="durable", strategy="test", confidence=0.9,
        )
        fold = a._document_episode_fold(
            _goal(a), Verdict(True, 0.9, "PO-88 visible", subject=ref), 0.0, truth="yes"
        )
        self.assertEqual(fold.match, "new")
        self.assertEqual(a.episodes["g1"]._episodes[0].subject_key, "PO-88")

    def test_a_screen_document_source_is_unaffected(self):
        a = _build(source_profile=SCREEN_DOCUMENT)
        self.assertEqual(a._resolved_identity_policy(_goal(a)), "content_key")
        # No subject on the verdict -> still "nothing to group", exactly as before.
        self.assertIsNone(a._document_episode_fold(_goal(a), _YES, 0.0, truth="yes"))
        self.assertNotIn("g1", a.episodes)


# --------------------------------------------------------------------------------------------------
# 5 — what the scene fold refuses to do.
# --------------------------------------------------------------------------------------------------


class TestRefusals(unittest.TestCase):
    def test_a_judge_emitted_track_ref_never_becomes_the_episode_key(self):
        # A per-entity key is exactly what this refuses to build -- the churning track handle must not
        # reach the episode log even when the judge volunteers one.
        a = _build()
        ref = SubjectRef(
            scheme="track", namespace="camera", key="trk-90210", resolution="bound",
            scope="session", strategy="test", confidence=0.9,
        )
        a._document_episode_fold(
            _goal(a), Verdict(True, 0.95, "a person", subject=ref), 0.0, truth="yes"
        )
        self.assertEqual(
            [e.subject_key for e in a.episodes["g1"]._episodes], [SCENE_IDENTITY]
        )

    def test_an_abstention_neither_opens_nor_closes_an_occurrence(self):
        a = _build()
        self.assertIsNone(a._document_episode_fold(_goal(a), _MAYBE, 0.0, truth="unknown"))
        self.assertNotIn("g1", a.episodes)
        a._document_episode_fold(_goal(a), _YES, 1.0, truth="yes")
        opened = a.episodes["g1"]._episodes[0].id
        self.assertIsNone(a._document_episode_fold(_goal(a), _MAYBE, 2.0, truth="unknown"))
        self.assertEqual(a.episodes["g1"].get(opened).status, "open")

    def test_a_stale_observation_is_skipped_and_counted(self):
        a = _build()
        a._document_episode_fold(_goal(a), _YES, 10.0, truth="yes")
        episode = a.episodes["g1"]._episodes[0]
        self.assertEqual(episode.last_seen, 10.0)
        before = a.metrics.get("episode.scene.out_of_order_skipped")
        self.assertIsNone(
            a._document_episode_fold(_goal(a), _YES, 2.0, truth="yes"),
            "an out-of-order observation must degrade to gate-only, not govern",
        )
        self.assertEqual(episode.last_seen, 10.0, "a stale read must never rewind last_seen")
        self.assertEqual(a.metrics.get("episode.scene.out_of_order_skipped"), before + 1)
        self.assertEqual(
            a._document_episode_fold(_goal(a), _YES, 15.0, truth="yes").match, "continuation"
        )

    def test_a_stale_confident_false_cannot_close_a_live_occurrence(self):
        a = _build()
        a._document_episode_fold(_goal(a), _YES, 10.0, truth="yes")
        a._document_episode_fold(_goal(a), _NO, 2.0, truth="no")
        self.assertEqual(a.episodes["g1"]._episodes[0].status, "open")
        self.assertEqual(a.metrics.get("episode.scene.closed"), 0)


# --------------------------------------------------------------------------------------------------
# 6 — THE FORFEIT, and the exact line it stops at. A scene singleton cannot count subjects: inside one
# gate key, an already-reported occurrence is reported once. What it must NOT do is take that decision
# away from the gate's per-entity key, which is the thing that actually answers "a second person
# entered". Both halves are asserted here so a later reader sees the boundary, not just the win.
# --------------------------------------------------------------------------------------------------


def _track(key: str) -> SubjectRef:
    return SubjectRef(
        scheme="track", namespace="camera", key=key, resolution="bound",
        scope="session", strategy="test", confidence=0.9,
    )


class TestSceneSingletonCannotCountSubjects(unittest.TestCase):
    def test_a_second_subject_on_the_same_gate_key_is_suppressed(self):
        # THE FORFEIT. Two people inside one unbroken occurrence, both reduced onto the same gate key,
        # are ONE incident. Anyone who later needs per-subject camera incidents changes this, in the
        # fold -- not in the gate.
        a = _build()
        goal, key = _goal(a), ("g1", None)
        opened = a._document_episode_fold(
            goal, Verdict(True, 0.95, "one person", subject=_track("trk-1")), 0.0,
            truth="yes", key=key,
        )
        self.assertEqual(opened.match, "new")
        a.executor._incidents.record(opened.episode_id)  # the occurrence was reported
        arrival = a._document_episode_fold(
            goal, Verdict(True, 0.95, "a second person", subject=_track("trk-2")), 1.0,
            truth="yes", key=key,
        )
        self.assertEqual(arrival.match, "continuation")
        self.assertFalse(
            arrival.should_evaluate_gate,
            "one gate key, one open occurrence, one incident -- the price of scene keying",
        )
        self.assertEqual(len(a.episodes["g1"]._episodes), 1)

    def test_a_different_gate_key_is_handed_back_to_the_gate_untouched(self):
        # THE LINE IT STOPS AT. Focus turnover means a different (goal, entity) key, and whether THAT
        # edge fires stays the gate's decision -- the fold returns None, exactly as it did for a camera
        # concern before this path existed.
        a = _build()
        goal = _goal(a)
        opened = a._document_episode_fold(
            goal, _YES, 0.0, truth="yes", key=("g1", "e1")
        )
        a.executor._incidents.record(opened.episode_id)
        self.assertIsNone(
            a._document_episode_fold(goal, _YES, 1.0, truth="yes", key=("g1", "e2")),
            "a second entity's edge belongs to the gate, not to the scene episode",
        )
        self.assertEqual(a.metrics.get("episode.scene.entity_turnover_skipped"), 1)
        # The occurrence was still ADVANCED -- declining to govern a tick is not pretending it was
        # unobserved -- and the newly governed key is then idempotent in its own right.
        self.assertEqual(a.episodes["g1"]._episodes[0].last_seen, 1.0)
        again = a._document_episode_fold(goal, _YES, 2.0, truth="yes", key=("g1", "e2"))
        self.assertEqual(again.match, "continuation")
        self.assertFalse(again.should_evaluate_gate)

    def test_focus_turnover_still_fires_per_entity_end_to_end(self):
        # The same property through the real judged path: two entities, two incidents, per-entity
        # gate keys intact. This is the behaviour a camera watch had before the fold and still has.
        a = _build()
        goal = _goal(a)
        for i, (t, entity) in enumerate(((0.0, "e1"), (1.0, "e2"))):
            fold = a._document_episode_fold(goal, _YES, t, truth="yes", key=("g1", entity))
            fired = a._fire_concern(goal, _YES, ("g1", entity), t=t, fold=fold)
            self.assertIsNotNone(fired, f"entity {entity} must keep its own rising edge")
            self.assertEqual(fired.action, "fire")
            self.assertEqual(fired.entity_id, entity)
