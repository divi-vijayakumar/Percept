"""L1: a full spoken turn through ``on_speech_endpoint`` + the proactive ``on_scene_delta``.

Proves **wiring and routing-table logic on fakes**, never model intelligence (standing rule 2):
  * a goal-authoring utterance ("watch my coffee") DEFERS to a scripted ``FakeConductor`` that
    returns a ``create`` goal op -> the agent applies it (the conductor is the only tier that
    writes standing state);
  * a confident question routes LOCALLY and streams a reply via ``FakeSpecialist`` into
    ``FakeTTS`` (push + flush) without ever touching the conductor;
  * a scene delta reaches the conductor so it can ``proactive_speak`` unprompted.

The ``FakeConductor`` here is *scripted* — it returns a pre-built ``Decision``. It asserts
nothing about whether a real model would author that goal; that's L2.
"""

import asyncio
import unittest

from percept.adapters import Decision, Goal
from percept.agent import Percept
from percept.backends.fake import FakeConductor, FakeSpecialist, FakeTTS
from percept.goals import GoalOp


def _build(conductor=None, specialist=None, tts=None) -> Percept:
    return Percept.create(
        conductor=conductor or FakeConductor(),
        specialist=specialist or FakeSpecialist(reply="here you go"),
        tts=tts or FakeTTS(),
        discover_plugins=False,
    )


class TestSpeechEndpointAuthorsGoal(unittest.TestCase):
    def test_watch_coffee_defers_and_conductor_authors_goal(self):
        async def run():
            goal = Goal(id="g-coffee", condition="the user drinks coffee", say="Stay off caffeine.")
            conductor = FakeConductor(
                Decision(route="ignore", goal_ops=[GoalOp(op="create", goal=goal)])
            )
            a = _build(conductor=conductor)

            result = await a.on_speech_endpoint("watch my coffee and tell me if I drink it")

            # Deferred to the conductor (router does not fast-path authoring).
            self.assertEqual(result.source, "conductor")
            # The goal the conductor authored is now active standing state.
            self.assertEqual([g.id for g in a.goals.active()], ["g-coffee"])
            self.assertEqual(a.goals.get("g-coffee").condition, "the user drinks coffee")
            # route=ignore -> no reply spoken.
            self.assertEqual(result.reply, "")

        asyncio.run(run())

    def test_conductor_route_streams_a_reply(self):
        async def run():
            # Conductor defers a non-authoring turn back to a streamed reply.
            conductor = FakeConductor(Decision(route="respond_direct"))
            tts = FakeTTS()
            a = _build(
                conductor=conductor, specialist=FakeSpecialist(reply="the answer is yes"), tts=tts
            )

            # "do the thing" is ambiguous -> router defers -> conductor says respond_direct.
            result = await a.on_speech_endpoint("do the thing")

            self.assertEqual(result.source, "conductor")
            self.assertEqual(result.route, "respond_direct")
            self.assertEqual(result.reply, "the answer is yes")
            self.assertEqual(tts.flushes, 1)
            self.assertEqual("".join(tts.spoken).strip(), "the answer is yes")

        asyncio.run(run())


class TestSpeechEndpointLocalRoute(unittest.TestCase):
    def test_question_routes_locally_and_streams(self):
        async def run():
            # A scripted conductor that would explode if called — proves we never defer here.
            class ExplodingConductor:
                async def run(self, event, utterance, context):
                    raise AssertionError("a confident question must route locally, not defer")

            tts = FakeTTS()
            a = _build(
                conductor=ExplodingConductor(),
                specialist=FakeSpecialist(reply="it is on the table"),
                tts=tts,
            )

            result = await a.on_speech_endpoint("where is my phone?")

            self.assertEqual(result.source, "router")
            self.assertEqual(result.route, "respond_with_vision")
            self.assertEqual(result.reply, "it is on the table")
            self.assertEqual(tts.flushes, 1)
            self.assertTrue(tts.spoken)  # reply streamed to TTS

        asyncio.run(run())

    def test_local_route_authors_no_goal(self):
        async def run():
            a = _build(specialist=FakeSpecialist(reply="ok"))
            await a.on_speech_endpoint("what color is that?")
            self.assertEqual(a.goals.active(), [])  # router never authors

        asyncio.run(run())


class TestProactiveSceneDelta(unittest.TestCase):
    def test_scene_delta_lets_conductor_speak_unprompted(self):
        async def run():
            conductor = FakeConductor(
                Decision(route="proactive_speak", say="Someone just walked in.")
            )
            tts = FakeTTS()
            a = _build(conductor=conductor, tts=tts)

            result = await a.on_scene_delta("person_entered")

            self.assertEqual(result.source, "conductor")
            self.assertEqual(result.route, "proactive_speak")
            self.assertEqual(result.reply, "Someone just walked in.")
            self.assertIn("Someone just walked in.", tts.spoken)
            self.assertEqual(tts.flushes, 1)

        asyncio.run(run())

    def test_scene_delta_can_author_a_goal(self):
        async def run():
            goal = Goal(id="g-door", condition="someone is at the door", say="There's someone here.")
            conductor = FakeConductor(
                Decision(route="ignore", goal_ops=[GoalOp(op="create", goal=goal)])
            )
            a = _build(conductor=conductor)
            await a.on_scene_delta("object_appeared")
            self.assertEqual([g.id for g in a.goals.active()], ["g-door"])

        asyncio.run(run())

    def test_scene_delta_can_stay_silent(self):
        async def run():
            tts = FakeTTS()
            a = _build(conductor=FakeConductor(Decision(route="ignore")), tts=tts)
            result = await a.on_scene_delta("person_entered")
            self.assertEqual(result.reply, "")
            self.assertEqual(tts.spoken, [])

        asyncio.run(run())


class TestTurnMentionFusion(unittest.TestCase):
    def test_endpoint_notes_mention_without_focus_is_safe(self):
        # No in-focus entity -> note_mention is a no-op; the turn must still complete.
        async def run():
            a = _build(specialist=FakeSpecialist(reply="hi"))
            result = await a.on_speech_endpoint("hey there")
            self.assertEqual(result.route, "respond_direct")

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
