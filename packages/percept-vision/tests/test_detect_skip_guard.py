"""The admission guard that elides ``vision.detect`` when nothing on the source can consume it.

Every admitted frame spends two serialized provider round trips, ``detect`` then ``judge``, and the
first one exists only to populate the entity graph. The guard skips it for exactly one shape of
source: one where EVERY active goal DECLARES ``identity_policy="scene"``.

Most of what is below is not testing that the guard works — it is testing that it stays narrow.
``test_all_scene_source_elides_the_detect_leg`` is the whole of the positive case; every other test
names one consumer of the entity graph that a wider predicate would silently break, and each was
measured before the guard was written:

* a second undeclared goal anywhere on the source — its watch reducer keys on the focus entity;
* a per-entity transition goal — with no focus it spends no judge call and latches nothing, ever;
* a user correction — with no focus ``assert_attribute`` returns ``None``, a silent no-op;
* two subjects both satisfying — they collapse onto one gate key and the second fire is lost;
* a scene-like SOURCE-PROFILE default — inheriting is not declaring, and the reducer knows it.

A future widening of the predicate has to break one of them.
"""

from __future__ import annotations

import asyncio

from percept.adapters import Goal, Verdict
from percept.agent import _DETECT_SKIP_METRIC, Percept
from percept.backends.fake import FakeVision
from percept.config import GateSettings, Settings
from percept.contracts import Box, Observation
from percept.sourceprofile import SourceProfile

YES = Verdict(True, 0.95, "the condition holds")
NO = Verdict(False, 0.95, "the condition does not hold")


def _settings() -> Settings:
    return Settings(gate=GateSettings(threshold=0.7, validity_window_s=1000.0))


class _CountingVision:
    """Counts the ``detect`` round trips an adapter is actually asked to spend."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.detect_calls = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def detect(self, frame, goals):
        self.detect_calls += 1
        return await self._inner.detect(frame, goals)


def _core(vision, *, source_profile=None) -> Percept:
    return Percept.create(
        vision=vision,
        settings=_settings(),
        discover_plugins=False,
        **({"source_profile": source_profile} if source_profile is not None else {}),
    )


def _drive(core: Percept, frames, *, source: str = "camera", ts=None):
    fired = []

    async def go():
        for index, token in enumerate(frames):
            t = float(index + 1) if ts is None else ts[index]
            for event in await core.perceive_judged(token, source=source, t=t):
                fired.append((event.goal_id, str(event.action), event.entity_id))

    asyncio.run(go())
    return fired


def _skips(core: Percept) -> int:
    return core.metrics.snapshot().get(_DETECT_SKIP_METRIC, 0)


def _scene_goal(goal_id: str = "s1", **overrides) -> Goal:
    state = {"type": "watch", "count": 0, "identity_policy": "scene"}
    state.update(overrides)
    return Goal(
        id=goal_id, condition="the condition holds", say="holds", state=state
    )


def _undeclared_goal(goal_id: str = "u1") -> Goal:
    return Goal(
        id=goal_id,
        condition="the condition holds",
        say="holds",
        state={"type": "watch", "count": 0},
    )


def _scripted_vision() -> _CountingVision:
    return _CountingVision(
        FakeVision(
            script={"y": YES, "n": NO},
            detect_script={"y": [Observation("thing", Box(0.3, 0.3, 0.3, 0.3), 0.95)]},
        )
    )


# -- the one positive case --------------------------------------------------------------------


def test_all_scene_source_elides_the_detect_leg():
    """The whole saving: no goal on the source can read a subject, so the leg is not spent.

    The outcome is the outcome an unguarded run produces — two rising edges on this script, both
    reduced onto the non-entity key — because a scene-declaring goal already ignored the entity
    graph. The graph simply never gets built.
    """
    vision = _scripted_vision()
    core = _core(vision)
    core.add_goal(_scene_goal())

    fired = _drive(core, ["y", "y", "n", "y"])

    assert vision.detect_calls == 0
    assert _skips(core) == 4
    assert fired == [("s1", "fire", None), ("s1", "fire", None)]
    assert len(core.graph) == 0


def test_the_guard_reports_its_saving_rather_than_leaving_it_inferred():
    """One counter per elided round trip, on the frame path and the split path alike."""
    core = _core(_scripted_vision())
    core.add_goal(_scene_goal())
    assert _skips(core) == 0

    _drive(core, ["y", "n"])

    assert core.metrics.snapshot()[_DETECT_SKIP_METRIC] == 2


# -- narrowness: one non-scene goal anywhere on the source is enough ---------------------------


def test_one_undeclared_goal_on_the_source_keeps_the_leg():
    """The broadest consumer. An undeclared goal's watch reducer keys on the focus entity, so the
    scene goal sharing its source cannot license a skip."""
    vision = _scripted_vision()
    core = _core(vision)
    core.add_goal(_scene_goal())
    core.add_goal(_undeclared_goal())

    _drive(core, ["y", "y", "n", "y"])

    assert vision.detect_calls == 4
    assert _skips(core) == 0


def test_an_undeclared_goal_alone_keeps_the_leg_and_its_focus_scoped_edge():
    """Driven alone, the undeclared goal fires ONCE on this script where the scene goal fires
    twice — the difference is the focus-scoped gate key, which only exists because detect ran."""
    vision = _scripted_vision()
    core = _core(vision)
    core.add_goal(_undeclared_goal())

    fired = _drive(core, ["y", "y", "n", "y"])

    assert vision.detect_calls == 4
    assert _skips(core) == 0
    assert fired == [("u1", "fire", "e1")]


def test_a_scene_like_source_profile_default_is_not_a_declaration():
    """Inheriting is not declaring. ``watch_reducer_entity_id`` opts a goal out of focus identity
    only on its OWN ``identity_policy="scene"``; a goal that merely sits on a scene-defaulting
    profile still reduces on the focus entity, so its leg must still be spent."""
    profile = SourceProfile(
        name="scene_defaulting",
        modalities=frozenset({"video"}),
        temporal_unit="frame",
        admission_strategy="frame_delta",
        evidence_strategy="frame",
        identity_strategy="scene",
    )
    vision = _scripted_vision()
    core = _core(vision, source_profile=profile)
    core.add_goal(_undeclared_goal())

    assert core.source_profile.default_identity_policy == "scene"

    _drive(core, ["y", "n"])

    assert vision.detect_calls == 2
    assert _skips(core) == 0


# -- narrowness: the three measured costs of skipping too much ---------------------------------


class _TransitionVision:
    """A judge whose verdict tracks the state the frame token names."""

    def __init__(self) -> None:
        self.detect_calls = 0
        self.judge_calls = 0

    async def judge(self, condition, frame, **kwargs):
        self.judge_calls += 1
        token = str(frame)
        holds = (token == "pending" and "pending" in condition) or (
            token == "approved" and "approved" in condition
        )
        return Verdict(satisfied=holds, confidence=0.95, evidence=condition)

    async def detect(self, frame, goals):
        self.detect_calls += 1
        return [Observation("form", Box(0.3, 0.3, 0.3, 0.3), 0.95)]

    async def detect_events(self, frames, goals):
        return []


def test_a_per_entity_transition_goal_never_skips():
    """Measured cost of getting this wrong: with no focus the goal spends ZERO judge calls and
    never touches its latch, so the transition can never fire again — silently, forever."""
    vision = _TransitionVision()
    core = _core(vision)
    core.add_goal(
        Goal(
            id="t1",
            condition="the form was approved",
            say="approved",
            state={
                "type": "transition",
                "from": "the form is pending",
                "to": "the form is approved",
                "count": 0,
                "identity_policy": "track_iou_label",
            },
        )
    )

    fired = _drive(core, ["pending", "approved", "approved"])

    assert vision.detect_calls == 3
    assert _skips(core) == 0
    assert vision.judge_calls > 0
    assert fired == [("t1", "fire", "e1")]


def test_a_goal_that_can_receive_a_user_correction_never_skips():
    """Measured cost: with nothing in focus ``assert_attribute`` finds no binding target, returns
    ``None`` and drops the correction on the floor — the user's word never reaches the judge."""
    vision = _CountingVision(
        FakeVision(
            script={"c": YES, "c|corrected": NO},
            detect_script={"c": [Observation("liquid", Box(0.3, 0.3, 0.3, 0.3), 0.95)]},
        )
    )
    core = _core(vision)
    core.add_goal(
        Goal(
            id="c1",
            condition="the glass holds juice",
            say="juice",
            state={"type": "watch", "count": 0},
        )
    )

    _drive(core, ["c"])
    bound = core.assert_attribute("content", "water", t=2.0)

    assert bound == "e1"
    assert vision.detect_calls == 1
    assert _skips(core) == 0


def test_two_subjects_keep_their_own_gate_keys():
    """Measured cost: both subjects collapse onto ``(goal, None)`` and the second rising edge is
    suppressed — one incident instead of two, with no error anywhere."""
    vision = _CountingVision(
        FakeVision(
            script={"sA": YES, "sB": YES},
            detect_script={
                "sA": [Observation("person", Box(0.02, 0.02, 0.25, 0.25), 0.95)],
                "sB": [Observation("person", Box(0.70, 0.70, 0.25, 0.25), 0.95)],
            },
        )
    )
    core = _core(vision)
    core.add_goal(_undeclared_goal("g1"))

    fired = _drive(core, ["sA", "sB"])

    assert vision.detect_calls == 2
    assert _skips(core) == 0
    assert fired == [("g1", "fire", "e1"), ("g1", "fire", "e2")]


# -- conservative on ambiguity -----------------------------------------------------------------


def test_a_source_with_no_active_goals_does_not_skip():
    """An empty consumer set is an unknown consumer set, not a licence."""
    vision = _scripted_vision()
    core = _core(vision)
    core.add_goal(_scene_goal(goal_id="on_camera"))

    assert core._detect_output_is_consumable("other-source") is True

    _drive(core, ["y"], source="other-source")

    assert vision.detect_calls == 1
    assert _skips(core) == 0


def test_a_goal_whose_scope_cannot_be_read_does_not_skip():
    """A non-dict / absent ``state`` reads as "declared nothing", which is not "declared scene"."""
    core = _core(_scripted_vision())
    core.add_goal(_scene_goal())

    class _Opaque:
        source = "camera"
        id = "opaque"

        @property
        def state(self):
            raise RuntimeError("goal scope is unreadable")

    assert core._detect_output_is_consumable("camera") is False
    core.goals.active = lambda: [_Opaque()]  # type: ignore[method-assign]
    assert core._detect_output_is_consumable("camera") is True


# -- the split (owner/provider) path ------------------------------------------------------------


def _candidate(core: Percept, *, goal_ids=None, source: str = "camera"):
    from percept.harness.tier0 import gray_frame

    return core.prepare_observation_candidate(
        gray_frame(8, 8, 255),
        source,
        t=1.0,
        expected_goals_revision=core.goals.revision,
        expected_source_generation=core.source_continuity_generation(source),
        goal_ids=goal_ids,
    )


def test_admission_marks_the_all_scene_candidate_and_dispatch_spends_nothing():
    vision = _scripted_vision()
    core = _core(vision)
    core.add_goal(_scene_goal())

    async def drive():
        candidate = _candidate(core)
        assert candidate is not None
        assert candidate.skip_detect is True
        assert core.launch_observation_candidate(candidate)
        completion = await core.dispatch_observation_candidate(candidate)
        assert completion.observations == ()
        assert completion.failure is None
        reduction = core.reduce_observation_completion(completion)
        assert reduction.stale_reason is None
        assert reduction.failure is None
        assert reduction.focus is None
        assert len(core._window("camera").frames()) == 1

    asyncio.run(drive())

    assert vision.detect_calls == 0
    assert _skips(core) == 1


def test_admission_keeps_the_leg_for_an_undeclared_candidate():
    vision = _scripted_vision()
    core = _core(vision)
    core.add_goal(_undeclared_goal())

    async def drive():
        candidate = _candidate(core)
        assert candidate is not None
        assert candidate.skip_detect is False
        assert core.launch_observation_candidate(candidate)
        completion = await core.dispatch_observation_candidate(candidate)
        assert core.reduce_observation_completion(completion).stale_reason is None

    asyncio.run(drive())

    assert vision.detect_calls == 1
    assert _skips(core) == 0


def test_selecting_only_the_scene_goal_does_not_license_a_skip():
    """The predicate is asked of the whole source, never of the admission's selected goals — the
    unselected goal still reads the graph on its own tick."""
    vision = _scripted_vision()
    core = _core(vision)
    core.add_goal(_scene_goal())
    core.add_goal(_undeclared_goal())

    candidate = _candidate(core, goal_ids=("s1",))

    assert candidate is not None
    assert tuple(goal.id for goal in candidate.goals) == ("s1",)
    assert candidate.skip_detect is False


def test_an_externally_built_candidate_still_dispatches():
    """``skip_detect`` defaults to ``False``: only admission can elide a leg."""
    from dataclasses import replace

    vision = _scripted_vision()
    core = _core(vision)
    core.add_goal(_scene_goal())

    async def drive():
        candidate = _candidate(core)
        unguarded = replace(candidate, skip_detect=False)
        assert core.launch_observation_candidate(unguarded)
        completion = await core.dispatch_observation_candidate(unguarded)
        assert completion.failure is None

    asyncio.run(drive())

    assert vision.detect_calls == 1
    assert _skips(core) == 0
