"""Scene scope reaches evidence framing and the correction API.

A concern that declares ``identity_policy="scene"`` has said there is no per-entity subject here.
Two consumers of the entity graph never learned that, because both predate ``identity_policy`` by a
month and neither was revisited when scene scope landed (``_resolve_roi``'s focus branch 2026-06-22,
``assert_attribute`` 2026-06-20, ``identity_policy.py`` 2026-07-21):

* the judge's CROP came from ``in_focus()`` — *"the largest entity in the most recent frame"* — so a
  scene concern was handed whichever object won an election it explicitly opted out of;
* a user CORRECTION bound to that same focus, and returned ``None`` when there was none, which the
  caller could not tell apart from "nothing in focus yet".

Most of what is below is narrowness. The positive cases are
``test_a_scene_concern_is_judged_on_the_whole_frame`` and
``test_a_scene_only_deployment_refuses_a_correction_loudly``; every other test names something a
wider change would break — an undeclared neighbour, the authored ``region_hint`` where-channel, an
inherited source-profile default, an explicit ``entity_id=``, and the transient no-focus case that
must stay a quiet ``None``.
"""

from __future__ import annotations

import asyncio

import pytest

from percept.adapters import Goal, Verdict
from percept.agent import Percept, UnsupportedCorrectionScopeError
from percept.backends.fake import FakeVision
from percept.config import GateSettings, Settings
from percept.contracts import Box, Observation
from percept.sourceprofile import SourceProfile

YES = Verdict(True, 0.95, "the condition holds")
NO = Verdict(False, 0.95, "the condition does not hold")

#: The box the scripted detector reports, and therefore the crop a focus-scoped concern receives.
DETECTED = Box(0.3, 0.3, 0.3, 0.3)


class _RecordingVision:
    """Records the ``roi`` actually handed to each ``judge`` call."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.rois: list[Box | None] = []
        #: ``condition -> roi``, so a two-concern frame is read PER CONCERN rather than by
        #: membership. A membership assertion passes when the right roi appears for the WRONG
        #: concern, which is exactly the framing bug under test.
        self.by_condition: dict[str, Box | None] = {}

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def judge(self, condition, frame, **kwargs):
        roi = kwargs.get("roi")
        self.rois.append(roi)
        self.by_condition[condition] = roi
        return await self._inner.judge(condition, frame, **kwargs)


def _vision() -> _RecordingVision:
    return _RecordingVision(
        FakeVision(
            script={"y": YES, "n": NO},
            detect_script={"y": [Observation("thing", DETECTED, 0.95)]},
        )
    )


def _core(vision, *, source_profile=None) -> Percept:
    return Percept.create(
        vision=vision,
        settings=Settings(gate=GateSettings(threshold=0.7, validity_window_s=1000.0)),
        discover_plugins=False,
        **({"source_profile": source_profile} if source_profile is not None else {}),
    )


def _drive(core: Percept, frames, *, source: str = "camera") -> None:
    async def go():
        for index, token in enumerate(frames):
            await core.perceive_judged(token, source=source, t=float(index + 1))

    asyncio.run(go())


def _scene_goal(goal_id: str = "s1", condition: str = "the scene holds", **state) -> Goal:
    base = {"type": "watch", "count": 0, "identity_policy": "scene"}
    base.update(state)
    return Goal(id=goal_id, condition=condition, say="holds", state=base)


def _undeclared_goal(
    goal_id: str = "u1", condition: str = "the subject holds", **state
) -> Goal:
    base = {"type": "watch", "count": 0}
    base.update(state)
    return Goal(id=goal_id, condition=condition, say="holds", state=base)


# -- framing: the whole frame is the scene concern's evidence -----------------------------------


def test_a_scene_concern_gets_the_whole_frame_while_its_neighbour_keeps_the_crop():
    """The positive case AND its narrowness half, in one frame — deliberately, because they cannot
    be separated without the test passing for the wrong reason.

    A scene concern ALONE on a source proves nothing here: the detect guard elides the observation
    leg for exactly that shape, so focus is ``None`` and the roi would be ``None`` whether or not
    this framing decision exists. Pairing it with an undeclared neighbour forces the leg to run and
    puts a real entity in focus, so the two concerns judge the SAME frame with a live graph and the
    roi each receives is attributable to the framing decision alone.

    Asserted per concern, not by membership: a membership check passes when the right roi reaches
    the wrong concern."""
    vision = _vision()
    core = _core(vision)
    core.add_goal(_scene_goal("s1", condition="the scene holds"))
    core.add_goal(_undeclared_goal("u1", condition="the subject holds"))

    _drive(core, ["y"])

    assert core.graph.in_focus() is not None, "the leg must have run for this test to mean anything"
    assert vision.by_condition == {
        "the scene holds": None,
        "the subject holds": DETECTED,
    }


def test_a_scene_only_source_also_gets_the_whole_frame():
    """The two mechanisms agree. Here the detect guard has already elided the leg, so there is no
    focus to crop to either way — this pins that the framing decision does not contradict the guard
    on the shape they overlap on."""
    vision = _vision()
    core = _core(vision)
    core.add_goal(_scene_goal())

    _drive(core, ["y", "n", "y"])

    assert vision.rois == [None, None, None]


def test_an_authored_region_hint_still_wins_for_a_scene_concern():
    """Narrowness, and the reason this refusal is narrow rather than 'scene means full frame'.

    ``region_hint`` is the NON-IDENTITY where-channel: it says where to look without claiming a
    subject. A scene concern that genuinely wants a crop expresses it here, and it is honoured
    ahead of the identity refusal. Only the focus-derived crop is declined."""
    hint = Box(0.1, 0.2, 0.25, 0.25)
    vision = _vision()
    core = _core(vision)
    core.add_goal(
        _scene_goal("s1", perception={"roi": "hint"}, region_hint=[0.1, 0.2, 0.25, 0.25])
    )

    _drive(core, ["y"])

    assert vision.rois == [hint]


def test_a_scene_like_source_profile_default_still_receives_the_focus_crop():
    """Narrowness, mirroring the detect guard. Inheriting a scene-like profile default is not
    DECLARING scene scope — ``watch_reducer_entity_id`` reduces such a goal on the focus entity, so
    its framing must not change either. The two decisions read the same declaration for the same
    reason."""
    profile = SourceProfile(
        name="scene_defaulting",
        modalities=frozenset({"video"}),
        temporal_unit="frame",
        admission_strategy="frame_delta",
        evidence_strategy="frame",
        identity_strategy="scene",
    )
    vision = _vision()
    core = _core(vision, source_profile=profile)
    core.add_goal(_undeclared_goal())

    assert core.source_profile.default_identity_policy == "scene"

    _drive(core, ["y"])

    assert vision.rois == [DETECTED]


# -- corrections: loud where it can never work, quiet where it has not worked YET ----------------


def test_a_scene_only_deployment_refuses_a_correction_loudly():
    """The positive case. No active concern carries entity scope, so the binding can never succeed —
    not now, not on a later frame. It raises instead of returning the ``None`` that used to drop the
    user's word on the floor."""
    vision = _vision()
    core = _core(vision)
    core.add_goal(_scene_goal())

    _drive(core, ["y"])

    with pytest.raises(UnsupportedCorrectionScopeError) as excinfo:
        core.assert_attribute("content", "water", t=9.0)

    message = str(excinfo.value)
    assert "content" in message and "water" in message
    assert "entity_id" in message, "the message must name the way out"


def test_an_explicit_entity_id_binds_even_on_a_scene_only_deployment():
    """Narrowness. The refusal is about having no subject to INFER, not about forbidding
    corrections. A caller that names the entity has supplied the missing scope itself."""
    vision = _vision()
    core = _core(vision)
    core.add_goal(_scene_goal())
    core.graph.ingest([Observation("thing", DETECTED, 0.95)], "y", "camera", t=1.0)
    entity_id = core.graph.in_focus().id

    assert core.assert_attribute("content", "water", entity_id=entity_id, t=9.0) == entity_id


def test_no_focus_yet_is_still_a_quiet_none_when_scope_exists():
    """Narrowness, and the distinction the whole change rests on. An undeclared concern HAS entity
    scope; an empty graph is a timing condition, not a contradiction, so the next frame may bind.
    That must stay a quiet ``None`` — raising here would turn a normal startup into an error."""
    vision = _vision()
    core = _core(vision)
    core.add_goal(_undeclared_goal())

    assert core.graph.in_focus() is None
    assert core.assert_attribute("content", "water", t=9.0) is None


def test_an_inherited_scene_default_does_not_license_the_refusal():
    """Narrowness, and the corrections-side twin of the framing profile test. The refusal reads the
    DECLARED policy; a concern that merely inherits a scene-like source-profile default still
    reduces on focus identity, so it still HAS entity scope and an empty graph stays a quiet
    ``None``. Widening this to the resolved policy turns a normal startup into an exception."""
    profile = SourceProfile(
        name="scene_defaulting",
        modalities=frozenset({"video"}),
        temporal_unit="frame",
        admission_strategy="frame_delta",
        evidence_strategy="frame",
        identity_strategy="scene",
    )
    vision = _vision()
    core = _core(vision, source_profile=profile)
    core.add_goal(_undeclared_goal())

    assert core.source_profile.default_identity_policy == "scene"
    assert core.graph.in_focus() is None
    assert core.assert_attribute("content", "water", t=9.0) is None


def test_an_undeclared_concern_with_focus_still_binds():
    """Narrowness. The historical path is untouched where scope exists and something is in focus."""
    vision = _vision()
    core = _core(vision)
    core.add_goal(_undeclared_goal())

    _drive(core, ["y"])

    assert core.assert_attribute("content", "water", t=9.0) == "e1"
