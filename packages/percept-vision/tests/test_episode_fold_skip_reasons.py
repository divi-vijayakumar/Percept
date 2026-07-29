"""A skipped episode fold must SAY WHICH GUARD SKIPPED IT — and which arm it was even on.

``Percept._document_episode_fold`` and ``Percept._scene_episode_fold`` both answer ``None`` for two
completely different situations: "no episode applies to this kind of concern" (the ordinary, correct
answer for most goals) and "an episode applies but this observation could not be folded into one".
Ten distinct guards across the two methods produce that same ``None``, and from outside they were
indistinguishable — so a run could only be read as "no fold happened", which cannot tell a working
watch from a broken one. Diagnosing it therefore meant reading code and guessing, which has produced
several confidently wrong answers.

What this suite pins:

1. every guard emits its OWN reason, and no two guards share one (``TestEveryGuardNamesItself``);
2. every observation records the identity policy as resolved AT JUDGMENT TIME, which dispatch arm it
   took, and whether the scene fold was ENTERED — so "entered and declined" is never confused with
   "never entered" (``TestTheDispatchArmIsRecorded``);
3. all of it reaches the ``shadow.fold`` trace record, which is the record an operator actually reads
   (``TestTheReasonReachesTheTrace``);
4. the record describes THIS observation, never a stale one from an earlier tick
   (``TestTheReasonIsNeverStale``);
5. the ``fold`` dict in that same record is NOT an answer to any of this — it reads ``unbound`` on
   every tick of every concern by construction, including when the governing fold bound an episode,
   and the two callers are counted apart so the structural ``None`` is visible as structural
   (``TestTheShadowFoldIsNotEvidence``).

All of it is observability. Nothing here asserts a fold DECISION — the decisions are pinned by
``test_camera_scene_episode_fold.py`` and ``test_y12_document_governing.py``, which still pass
unchanged.
"""

from __future__ import annotations

import asyncio
import unittest

from percept.adapters import Goal, Verdict
from percept.agent import (
    EPISODE_FOLD_DISPATCH_ARMS,
    EPISODE_FOLD_SKIP_REASONS,
    Percept,
)
from percept.backends.fake import FakeVision
from percept.config import GateSettings, Settings
from percept.contracts import SubjectRef
from percept.gate import Gate
from percept.identity_policy import SCENE_IDENTITY
from percept.replay import gate_config_of
from percept.source_profiles_default import PHYSICAL_SCENE, SCREEN_DOCUMENT
from percept.tracing import (
    STAGE_SHADOW_FOLD,
    RunTraceManifest,
    TracePolicy,
    TraceRecorder,
)

_YES = Verdict(True, 0.95, "a person is at the door")
_NO = Verdict(False, 0.95, "the doorway is empty")
_MAYBE = Verdict(True, 0.3, "something, maybe")

_METRIC = "episode.fold.skipped"


def _doc_subject(key: str = "PO-88") -> SubjectRef:
    return SubjectRef(
        scheme="content", namespace="workflow", key=key, resolution="bound",
        scope="durable", strategy="test", confidence=0.9,
    )


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


def _recorder():
    return TraceRecorder(
        RunTraceManifest.create(
            run_id="run-fold-skip", source="camera", policy=TracePolicy(level="full"),
            gate_config=gate_config_of(Gate()),
        )
    )


def _shadow_records(rec) -> list:
    return [r for r in rec.records if r.stage == STAGE_SHADOW_FOLD]


# --------------------------------------------------------------------------------------------------
# 1 — every guard names itself, and no two guards answer with the same name.
# --------------------------------------------------------------------------------------------------


class TestEveryGuardNamesItself(unittest.TestCase):
    """One case per ``return None`` in the two methods. Each asserts the reason recorded for the goal,
    the counter that reason bumped, and that the fold really was skipped."""

    def _skipped(self, a: Percept, fold, expected: str) -> None:
        self.assertIsNone(fold, "this case is only meaningful when the fold was skipped")
        self.assertIn(expected, EPISODE_FOLD_SKIP_REASONS, "reason must be a declared one")
        self.assertEqual(a._fold_observation["g1"].skip_reason, expected)
        self.assertEqual(a.metrics.get(f"{_METRIC}.{expected}"), 1)

    # -- the content-key (document/screen) path -----------------------------------------------------

    def test_event_chain_rearm(self):
        a = _build(state={"rearm": "each_event"})
        self._skipped(
            a, a._document_episode_fold(_goal(a), _YES, 0.0, truth="yes"),
            "document.event_chain_rearm",
        )

    def test_identity_policy_is_neither_scene_nor_content_key(self):
        # No source profile and no declared policy: the concern resolves to no identity policy at
        # all, so neither fold applies. The ordinary answer for most goals -- and it now says so.
        settings = Settings(gate=GateSettings(threshold=0.7, validity_window_s=1000.0))
        a = Percept.create(vision=FakeVision(), settings=settings, discover_plugins=False)
        a.add_goal(Goal(id="g1", condition="a person is at the door", say="someone is there"))
        self.assertIsNone(a._resolved_identity_policy(_goal(a)), "the premise of this case")
        self._skipped(
            a, a._document_episode_fold(_goal(a), _YES, 0.0, truth="yes"),
            "document.identity_policy_not_content_key",
        )

    def test_no_content_subject_on_the_verdict(self):
        a = _build(source_profile=SCREEN_DOCUMENT)
        self.assertEqual(a._resolved_identity_policy(_goal(a)), "content_key")
        self._skipped(
            a, a._document_episode_fold(_goal(a), _YES, 0.0, truth="yes"),
            "document.no_content_subject",
        )

    def test_document_out_of_order(self):
        a = _build(source_profile=SCREEN_DOCUMENT)
        seen = Verdict(True, 0.95, "PO-88 is visible", subject=_doc_subject())
        self.assertIsNotNone(a._document_episode_fold(_goal(a), seen, 10.0, truth="yes"))
        self._skipped(
            a, a._document_episode_fold(_goal(a), seen, 2.0, truth="yes"),
            "document.out_of_order",
        )
        # The counter that already existed is untouched by the new one.
        self.assertEqual(a.metrics.get("episode.document.out_of_order_skipped"), 1)

    # -- the scene (camera) path --------------------------------------------------------------------

    def test_scene_rearm_is_not_the_rising_edge(self):
        a = _build(state={"rearm": "never"})
        self._skipped(
            a, a._document_episode_fold(_goal(a), _YES, 0.0, truth="yes"),
            "scene.rearm_not_rising_edge",
        )

    def test_scene_route_is_not_a_frame_watch(self):
        a = _build(state={"type": "rhythm"})
        self._skipped(
            a, a._document_episode_fold(_goal(a), _YES, 0.0, truth="yes"),
            "scene.route_not_frame_watch",
        )

    def test_scene_truth_is_unknown(self):
        a = _build()
        self._skipped(
            a, a._document_episode_fold(_goal(a), _MAYBE, 0.0, truth="unknown"),
            "scene.truth_unknown",
        )

    def test_scene_out_of_order(self):
        a = _build()
        self.assertIsNotNone(a._document_episode_fold(_goal(a), _YES, 10.0, truth="yes"))
        self._skipped(
            a, a._document_episode_fold(_goal(a), _YES, 2.0, truth="yes"),
            "scene.out_of_order",
        )
        self.assertEqual(a.metrics.get("episode.scene.out_of_order_skipped"), 1)

    def test_scene_occurrence_closed_by_a_confident_false(self):
        a = _build()
        self.assertIsNotNone(a._document_episode_fold(_goal(a), _YES, 0.0, truth="yes"))
        self._skipped(
            a, a._document_episode_fold(_goal(a), _NO, 1.0, truth="no"),
            "scene.occurrence_closed",
        )
        self.assertEqual(a.metrics.get("episode.scene.closed"), 1)

    def test_scene_entity_turnover(self):
        a = _build()
        self.assertIsNotNone(
            a._document_episode_fold(_goal(a), _YES, 0.0, truth="yes", key=("g1", "e1"))
        )
        self._skipped(
            a, a._document_episode_fold(_goal(a), _YES, 1.0, truth="yes", key=("g1", "e2")),
            "scene.entity_turnover",
        )
        self.assertEqual(a.metrics.get("episode.scene.entity_turnover_skipped"), 1)

    # -- the discrimination itself ------------------------------------------------------------------

    def test_no_two_guards_share_a_reason(self):
        self.assertEqual(
            len(set(EPISODE_FOLD_SKIP_REASONS)), len(EPISODE_FOLD_SKIP_REASONS),
            "a shared reason would collapse two guards back into one silence",
        )

    def test_every_declared_reason_has_a_case_above(self):
        # A declared reason no case above can produce is as bad as a guard that names nothing: it
        # means the guard moved, or the name is dead. Adding a reason without a case fails here.
        import inspect

        source = inspect.getsource(type(self))
        missing = [r for r in EPISODE_FOLD_SKIP_REASONS if f'"{r}"' not in source]
        self.assertEqual(missing, [], "declared reasons with no case exercising them")


# --------------------------------------------------------------------------------------------------
# 2 — which arm the dispatch took, and whether the scene fold ran at all.
# --------------------------------------------------------------------------------------------------


class TestTheDispatchArmIsRecorded(unittest.TestCase):
    """A skip reason alone still leaves one question open: was the scene fold ENTERED and did it
    decline, or was it never reached? Only the caller's dispatch decides that, and it is the
    difference between "the fold has a bug" and "the fold was never asked". Both are recorded."""

    def _observed(self, a: Percept):
        return a._fold_observation["g1"]

    def test_a_per_entity_policy_enters_the_scene_fold(self):
        a = _build()  # physical_scene -> track_iou_label
        a._document_episode_fold(_goal(a), _YES, 0.0, truth="yes")
        observed = self._observed(a)
        self.assertEqual(observed.identity_policy, "track_iou_label")
        self.assertEqual(observed.dispatch, "scene")
        self.assertTrue(observed.entered_scene_fold)
        self.assertIsNone(observed.skip_reason)

    def test_the_scene_fold_can_be_entered_and_still_decline(self):
        a = _build()
        a._document_episode_fold(_goal(a), _MAYBE, 0.0, truth="unknown")
        observed = self._observed(a)
        self.assertTrue(observed.entered_scene_fold, "it ran; a guard inside it declined")
        self.assertEqual(observed.dispatch, "scene")
        self.assertEqual(observed.skip_reason, "scene.truth_unknown")

    def test_a_content_key_policy_takes_the_content_key_arm(self):
        a = _build(source_profile=SCREEN_DOCUMENT)
        seen = Verdict(True, 0.95, "PO-88 is visible", subject=_doc_subject())
        a._document_episode_fold(_goal(a), seen, 0.0, truth="yes")
        observed = self._observed(a)
        self.assertEqual(observed.identity_policy, "content_key")
        self.assertEqual(observed.dispatch, "content_key")
        self.assertFalse(observed.entered_scene_fold)

    def test_a_policy_in_neither_set_never_enters_the_scene_fold(self):
        # The dispatch routes to the scene fold ONLY for PER_ENTITY_IDENTITY_POLICIES. Any other
        # resolved policy ends the fold on the "none" arm without the scene fold ever running --
        # including the literal scene SCOPE below, which names the scene but is not a resolver.
        a = _build(state={"identity_policy": SCENE_IDENTITY})
        self.assertEqual(a._resolved_identity_policy(_goal(a)), SCENE_IDENTITY)
        self.assertIsNone(a._document_episode_fold(_goal(a), _YES, 0.0, truth="yes"))
        observed = self._observed(a)
        self.assertEqual(observed.identity_policy, SCENE_IDENTITY)
        self.assertEqual(observed.dispatch, "none")
        self.assertFalse(
            observed.entered_scene_fold,
            "the scene fold is never reached by the policy that names the scene scope",
        )
        self.assertEqual(observed.skip_reason, "document.identity_policy_not_content_key")
        self.assertNotIn("g1", a.episodes, "and so no episode is ever opened")

    def test_the_policy_is_resolved_at_judgment_time_not_reused_from_arming(self):
        # The record must reflect what the fold ACTUALLY saw on this tick. Change what the concern
        # declares between two folds and the second record must report the new answer.
        a = _build()
        a._document_episode_fold(_goal(a), _YES, 0.0, truth="yes")
        self.assertEqual(self._observed(a).identity_policy, "track_iou_label")
        _goal(a).state["identity_policy"] = SCENE_IDENTITY
        a._document_episode_fold(_goal(a), _YES, 1.0, truth="yes")
        self.assertEqual(self._observed(a).identity_policy, SCENE_IDENTITY)
        self.assertEqual(self._observed(a).dispatch, "none")

    def test_every_declared_arm_is_reachable(self):
        import inspect

        source = inspect.getsource(type(self))
        missing = [arm for arm in EPISODE_FOLD_DISPATCH_ARMS if f'"{arm}"' not in source]
        self.assertEqual(missing, [], "declared dispatch arms with no case exercising them")

    def test_focus_turnover_is_counted_per_concern(self):
        # Churn is what turns a scene occurrence into a stream of declined ticks, so the count rides
        # the record. A running count, never a set of entity ids -- see the field's comment.
        a = _build()
        for i, entity in enumerate(("e1", "e2", "e3")):
            a._document_episode_fold(
                _goal(a), _YES, float(i), truth="yes", key=("g1", entity)
            )
        self.assertEqual(self._observed(a).scene_key_turnovers, 2)


# --------------------------------------------------------------------------------------------------
# 3 — the reason reaches the record an operator actually reads.
# --------------------------------------------------------------------------------------------------


class TestTheReasonReachesTheTrace(unittest.TestCase):
    def _drive(self, a: Percept, stream):
        a.vision = FakeVision(script={f"f{i}": v for i, (_, v) in enumerate(stream)})
        rec = _recorder()
        a.on_trace = rec.tracer()
        for i, (t, _) in enumerate(stream):
            asyncio.run(a.perceive_judged(f"f{i}", t=t))
        return rec

    def test_a_camera_watch_carries_the_reason_on_the_ticks_it_skips(self):
        a = _build()
        # true, true, false: the second tick folds (a continuation), the third closes the occurrence.
        rec = self._drive(a, [(0.0, _YES), (1.0, _YES), (2.0, _NO)])
        reasons = [r.data.get("fold_skip_reason") for r in _shadow_records(rec)]
        self.assertEqual(len(reasons), 3, "one shadow record per judged tick")
        self.assertEqual(reasons, [None, None, "scene.occurrence_closed"])

    def test_a_concern_with_no_foldable_identity_says_so_every_tick(self):
        settings = Settings(gate=GateSettings(threshold=0.7, validity_window_s=1000.0))
        a = Percept.create(vision=FakeVision(), settings=settings, discover_plugins=False)
        a.add_goal(Goal(id="g1", condition="a person is at the door", say="someone is there"))
        rec = self._drive(a, [(0.0, _YES)])
        self.assertEqual(
            [r.data.get("fold_skip_reason") for r in _shadow_records(rec)],
            ["document.identity_policy_not_content_key"],
        )

    def test_the_field_is_none_when_a_fold_was_built(self):
        a = _build()
        rec = self._drive(a, [(0.0, _YES)])
        record = _shadow_records(rec)[0]
        self.assertIsNone(record.data["fold_skip_reason"])
        self.assertIsNotNone(a.episodes["g1"], "the premise: a fold really was built this tick")

    def test_the_dispatch_and_the_policy_ride_the_same_record(self):
        a = _build()
        record = _shadow_records(self._drive(a, [(0.0, _YES)]))[0]
        self.assertEqual(record.data["fold_identity_policy"], "track_iou_label")
        self.assertEqual(record.data["fold_dispatch"], "scene")
        self.assertIs(record.data["fold_entered_scene"], True)
        self.assertEqual(record.data["fold_scene_key_turnovers"], 0)


# --------------------------------------------------------------------------------------------------
# 4 — the record is about THIS observation.
# --------------------------------------------------------------------------------------------------


class TestTheReasonIsNeverStale(unittest.TestCase):
    def test_a_later_bound_fold_clears_the_earlier_reason(self):
        a = _build()
        self.assertIsNone(a._document_episode_fold(_goal(a), _MAYBE, 0.0, truth="unknown"))
        self.assertEqual(a._fold_observation["g1"].skip_reason, "scene.truth_unknown")
        self.assertIsNotNone(a._document_episode_fold(_goal(a), _YES, 1.0, truth="yes"))
        self.assertIsNone(
            a._fold_observation["g1"].skip_reason,
            "a bound fold must not leave the previous tick's reason readable",
        )

    def test_a_second_skip_overwrites_the_first(self):
        a = _build()
        a._document_episode_fold(_goal(a), _MAYBE, 0.0, truth="unknown")
        a._document_episode_fold(_goal(a), _YES, 1.0, truth="yes")
        a._document_episode_fold(_goal(a), _NO, 2.0, truth="no")
        self.assertEqual(a._fold_observation["g1"].skip_reason, "scene.occurrence_closed")


# --------------------------------------------------------------------------------------------------
# 5 — why these fields were needed: the fold dict answers a different question entirely.
# --------------------------------------------------------------------------------------------------


class TestTheShadowFoldIsNotEvidence(unittest.TestCase):
    def test_the_shadow_fold_reads_unbound_even_when_the_governing_fold_bound(self):
        a = _build()
        a.vision = FakeVision(script={"f0": _YES})
        rec = _recorder()
        a.on_trace = rec.tracer()
        events = asyncio.run(a.perceive_judged("f0", t=0.0))
        self.assertEqual([e.action for e in events], ["fire"])
        # The governing fold bound a real episode this tick...
        self.assertEqual(len(a.episodes["g1"]._episodes), 1)
        record = _shadow_records(rec)[0]
        # ...and the shadow's fold in the SAME record still reads unbound, with no episode id. It
        # binds no continuation decision, so it always does -- for every concern, on every tick.
        self.assertEqual(record.data["fold"]["match"], "unbound")
        self.assertIsNone(record.data["fold"]["episode_id"])
        # Only the governing fields distinguish this tick from one where nothing folded.
        self.assertIsNone(record.data["fold_skip_reason"])
        self.assertIs(record.data["fold_entered_scene"], True)

    def test_the_two_callers_of_the_fold_are_counted_apart(self):
        # behavior.py's unbound branch is reached whenever the decision is None. The shadow supplies
        # None structurally, on every tick; a governing caller supplying None would be a real finding.
        # Counting them together is what makes a guaranteed value look like evidence.
        a = _build()
        a.vision = FakeVision(script={f"f{i}": _YES for i in range(3)})
        rec = _recorder()
        a.on_trace = rec.tracer()
        for i in range(3):
            asyncio.run(a.perceive_judged(f"f{i}", t=float(i)))
        self.assertEqual(a.metrics.get("episode.fold.no_decision.shadow"), 3)
        self.assertEqual(
            a.metrics.get("episode.fold.no_decision.governing"), 0,
            "every governing fold on this run bound a decision",
        )


if __name__ == "__main__":
    unittest.main()
