"""The voice-transport seam — `VoiceTransport` Protocol + `RuntimeSession.attach_transport`.

Percept never hand-rolls the audio pipe (mic/speaker/AEC/endpointing/barge-in) — a client plugs a real
transport (Pipecat/LiveKit/browser WebRTC/S2S) into the seam and the SDK wires it to cognition. These
tests prove that wiring with `FakeVoiceTransport`: an endpointed utterance becomes a frame-grounded turn
whose reply flows back to the transport; barge-in cancels the turn; the `arm_goal` tool arms a watch
goal; and a speech-to-speech transport's `say` is a no-op. (Rule 2: fakes prove wiring, not model IQ.)
"""
from __future__ import annotations

import asyncio
import unittest

from percept import Percept
from percept.adapters import Decision, Reply
from percept.backends.fake import FakeConductor, FakeVoiceTransport
from percept.contracts import VoiceTransport
from percept.runtime.session import RuntimeSession

FRAME = ("image/jpeg", b"FRAMEBYTES")


def _core(reply: str | None = "I see a desk.") -> Percept:
    """A core whose conductor speaks `reply` on any turn (proactive_speak route), router disabled so the
    turn always defers to the conductor (deterministic)."""
    decision = Decision(route="proactive_speak", say=reply) if reply else Decision(route="ignore")
    core = Percept.create(vision="fake", conductor=FakeConductor(decision), discover_plugins=False)
    core.router = None
    return core


class ProtocolConformanceTests(unittest.TestCase):
    def test_fake_satisfies_voice_transport(self):
        self.assertIsInstance(FakeVoiceTransport(), VoiceTransport)  # runtime_checkable structural check

    def test_start_stop_idempotent_and_stop_never_raises(self):
        async def go():
            t = FakeVoiceTransport()
            await t.start()
            await t.start()
            self.assertTrue(t.started)
            await t.stop()
            await t.stop()
            self.assertTrue(t.stopped)

        asyncio.run(go())


class OrchestrationTests(unittest.TestCase):
    def test_utterance_becomes_frame_grounded_reply(self):
        async def go():
            core = _core("I see a desk.")
            seen: dict = {}
            orig = core.on_speech_endpoint

            async def spy(utterance, *, frame=None, t=None):
                seen["utterance"], seen["frame"] = utterance, frame
                return await orig(utterance, frame=frame, t=t)

            core.on_speech_endpoint = spy

            session = RuntimeSession(core)
            await session._notify_frame(0.0, FRAME)   # the watch's latest frame
            fake = FakeVoiceTransport()
            await session.attach_transport(fake)

            self.assertTrue(fake.started)
            self.assertEqual(fake.current_frame(), FRAME)   # set_frame_provider(latest_frame) wired

            await fake.emit_utterance("what do you see?")

            self.assertEqual(seen["utterance"], "what do you see?")
            self.assertEqual(seen["frame"], FRAME)          # the turn WAS frame-grounded
            self.assertEqual(len(fake.said), 1)
            self.assertIsInstance(fake.said[0], Reply)
            self.assertEqual(fake.said[0].text, "I see a desk.")

        asyncio.run(go())

    def test_empty_reply_is_not_spoken(self):
        async def go():
            session = RuntimeSession(_core(reply=None))  # conductor ignores -> empty reply
            fake = FakeVoiceTransport()
            await session.attach_transport(fake)
            await fake.emit_utterance("hello?")
            self.assertEqual(fake.said, [])   # nothing to say -> no say()

        asyncio.run(go())

    def test_say_records_and_tracks_verbatim(self):
        # say() is the DELIVERY channel (fires + component replies) — it always records, and carries the
        # verbatim flag (the corrected design: say conveys by default, verbatim is a capability request).
        async def go():
            fake = FakeVoiceTransport()
            await fake.say(Reply(text="hi"))
            await fake.say(Reply(text="exact"), verbatim=True)
            self.assertEqual([r.text for r in fake.said], ["hi", "exact"])
            self.assertEqual(fake.spoke_verbatim, [False, True])

        asyncio.run(go())

    def test_audio_only_utterance_uses_injected_stt(self):
        async def go():
            session = RuntimeSession(_core("got it"))

            async def stt(audio):        # a transport that ships raw PCM, no text
                return "transcribed question"

            fake = FakeVoiceTransport()
            await session.attach_transport(fake, stt=stt)
            await fake.emit_utterance(text=None, audio=b"\x00\x01")
            self.assertEqual(len(fake.said), 1)   # audio -> stt -> turn -> reply

        asyncio.run(go())


class BargeInTests(unittest.TestCase):
    def test_barge_in_interrupts_turn_and_transport(self):
        async def go():
            session = RuntimeSession(_core())
            fake = FakeVoiceTransport()
            await session.attach_transport(fake)

            task = asyncio.ensure_future(asyncio.sleep(10))   # a turn in flight
            session._turn_task = task
            await fake.emit_barge_in()
            await asyncio.sleep(0)                             # let the cancellation propagate

            self.assertEqual(fake.interrupts, 1)              # transport playout stopped
            self.assertTrue(task.cancelled())                 # the in-flight turn was cancelled

        asyncio.run(go())


class ArmGoalToolTests(unittest.TestCase):
    def test_arm_goal_tool_arms_a_watch_goal(self):
        async def go():
            core = _core()
            session = RuntimeSession(core)
            fake = FakeVoiceTransport()
            await session.attach_transport(fake)

            self.assertIn("arm_goal", fake.tools)
            rev_before = core.goals.revision
            out = await fake.call_tool("arm_goal", {"condition": "a red mug is visible", "say": "mug!"})

            self.assertIn("watching for", out["output"])
            self.assertTrue(any("mug" in g.id for g in core.goals.active()))
            self.assertNotEqual(core.goals.revision, rev_before)   # registry bumped -> gate re-arms

        asyncio.run(go())

    def test_arm_goal_requires_condition(self):
        async def go():
            session = RuntimeSession(_core())
            fake = FakeVoiceTransport()
            await session.attach_transport(fake)
            out = await fake.call_tool("arm_goal", {"say": "no condition"})
            self.assertIn("error", out)

        asyncio.run(go())

    def test_arm_tool_can_be_disabled(self):
        async def go():
            session = RuntimeSession(_core())
            fake = FakeVoiceTransport()
            await session.attach_transport(fake, arm_tool=False)
            self.assertNotIn("arm_goal", fake.tools)

        asyncio.run(go())


class TeardownTests(unittest.TestCase):
    def test_detach_and_stop_stop_the_transport(self):
        async def go():
            session = RuntimeSession(_core())
            fake = FakeVoiceTransport()
            await session.attach_transport(fake)
            await session.detach_transport()
            self.assertTrue(fake.stopped)
            self.assertIsNone(session._transport)
            await session.detach_transport()   # idempotent

        asyncio.run(go())

    def test_fresh_session_has_no_transport(self):
        session = RuntimeSession(_core())
        self.assertIsNone(session._transport)   # byte-for-byte: nothing attached by default


if __name__ == "__main__":
    unittest.main()
