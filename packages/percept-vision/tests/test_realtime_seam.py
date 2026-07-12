"""The realtime seam-under-contention — RuntimeSession wiring for a held speech-to-speech transport.

Covers the parts a naive `on_utterance`+`say` seam missed (design doc §4): the brain-state PULSE that
keeps the wire's reactive answers coherent, the rich `watch`/`stop_watching`/`whats_armed` command verbs
(raw language → conductor), the free transcript → evidence tee, and FIRE delivery gated by modality +
per-goal delivery policy + the user-speaking signal. All driven by FakeVoiceTransport, fully offline.
"""
from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from percept import Goal, Percept
from percept.adapters import Decision
from percept.backends.fake import FakeConductor, FakeVoiceTransport
from percept.runtime.session import RuntimeSession


def _core() -> Percept:
    core = Percept.create(vision="fake", conductor=FakeConductor(Decision(route="ignore")),
                          discover_plugins=False)
    core.router = None
    return core


def _inc(concern_id: str, message: str) -> SimpleNamespace:
    return SimpleNamespace(concern_id=concern_id, message=message, id="i1", ts=0.0)


async def _attach(core, **kw):
    session = RuntimeSession(core)
    fake = FakeVoiceTransport()
    await session.attach_transport(fake, **kw)
    return session, fake


class StatePulseTests(unittest.TestCase):
    def test_pulses_on_attach_and_on_goal_change(self):
        async def go():
            core = _core()
            session, fake = await _attach(core)
            self.assertEqual(len(fake.state_pulses), 1)          # grounded at connect (force)
            self.assertIn("nothing yet", fake.state_pulses[0])

            await session.arm(Goal(id="coffee", condition="a coffee mug is visible", say="water?"))
            self.assertGreaterEqual(len(fake.state_pulses), 2)   # re-pulsed on arm
            self.assertIn("coffee", fake.state_pulses[-1])       # the wire now knows what's armed

        asyncio.run(go())


class CommandVerbTests(unittest.TestCase):
    def test_watch_routes_raw_language_to_conductor(self):
        async def go():
            core = _core()
            seen = {}
            orig = core.arm_from_language

            async def spy(command, *, t=None):
                seen["cmd"] = command
                return await orig(command, t=t)

            core.arm_from_language = spy   # watch → author-only verb (NOT on_speech_endpoint → no spoken reply)
            session, fake = await _attach(core)
            out = await fake.call_tool("watch", {"command": "tell me the second my baby cries"})
            self.assertEqual(out["output"], "On it.")           # instant ack; never blocks on the conductor
            await asyncio.gather(*session._arm_tasks)           # drain the async arming
            self.assertEqual(seen["cmd"], "tell me the second my baby cries")   # RAW language → conductor
            self.assertIn("output", out)

        asyncio.run(go())

    def test_whats_armed_reads_active_goals(self):
        async def go():
            core = _core()
            session, fake = await _attach(core)
            await session.arm(Goal(id="mug", condition="a mug is visible", say="!"))
            out = await fake.call_tool("whats_armed", {})
            self.assertEqual([g["id"] for g in out["armed"]], ["mug"])

        asyncio.run(go())

    def test_stop_watching_disarms_matching(self):
        async def go():
            core = _core()
            session, fake = await _attach(core)
            await session.arm(Goal(id="caffeine", condition="the user drinks caffeine", say="!"))
            out = await fake.call_tool("stop_watching", {"what": "caffeine"})
            self.assertIn("caffeine", out["output"])
            self.assertEqual(list(core.goals.active()), [])

        asyncio.run(go())


class TranscriptEvidenceTests(unittest.TestCase):
    def test_transcript_tees_to_evidence(self):
        async def go():
            core = _core()
            noted = []
            core.note_mention = lambda text, **kw: noted.append(text)   # spy the evidence hook
            _session, fake = await _attach(core)
            await fake.emit_transcript("that's my daughter Maya")
            self.assertEqual(noted, ["that's my daughter Maya"])        # free evidence, not STT

        asyncio.run(go())


class FireDeliveryTests(unittest.TestCase):
    def test_speak_modality_speaks_the_fire(self):
        async def go():
            core = _core()
            session, fake = await _attach(core)
            await session.arm(Goal(id="mug", condition="mug", say="water?", modality="speak"))
            await session.deliver_fire(_inc("mug", "water?"))
            self.assertEqual([r.text for r in fake.said], ["water?"])

        asyncio.run(go())

    def test_act_modality_hits_the_act_sink_not_voice(self):
        async def go():
            core = _core()
            session = RuntimeSession(core)
            fake = FakeVoiceTransport()
            acts = []
            await session.attach_transport(fake, act_sink=lambda inc: acts.append(inc.message))
            await session.arm(Goal(id="log", condition="dog on couch", say="logged", modality="act"))
            await session.deliver_fire(_inc("log", "logged"))
            self.assertEqual(acts, ["logged"])         # recorded to the act sink
            self.assertEqual(fake.said, [])            # NOT spoken

        asyncio.run(go())

    def test_speak_with_no_transport_falls_back_to_act(self):
        async def go():
            core = _core()
            session = RuntimeSession(core)          # no transport attached
            acts = []
            session._act_sink = lambda inc: acts.append(inc.message)
            await session.arm(Goal(id="mug", condition="mug", say="water?", modality="speak"))
            await session.deliver_fire(_inc("mug", "water?"))
            self.assertEqual(acts, ["water?"])      # assistant-class: notify when no one's listening

        asyncio.run(go())

    def test_gentle_fire_queues_behind_a_turn_then_flushes(self):
        async def go():
            core = _core()
            session, fake = await _attach(core)
            await session.arm(Goal(id="mug", condition="mug", say="water?", modality="speak",
                                   state={"delivery": {"preempt": "queue"}}))
            await fake.emit_user_speaking(True)               # user is mid-sentence
            await session.deliver_fire(_inc("mug", "water?"))
            self.assertEqual(fake.said, [])                  # held, not spoken over the user
            await fake.emit_user_speaking(False)             # turn ends → flush
            self.assertEqual([r.text for r in fake.said], ["water?"])

        asyncio.run(go())

    def test_severe_fire_interrupts_the_turn(self):
        async def go():
            core = _core()
            session, fake = await _attach(core)
            await session.arm(Goal(id="baby", condition="baby cries", say="baby!", modality="speak",
                                   state={"delivery": {"preempt": "interrupt"}}))
            await fake.emit_user_speaking(True)
            await session.deliver_fire(_inc("baby", "baby!"))
            self.assertEqual(fake.interrupts, 1)             # cut into the turn
            self.assertEqual([r.text for r in fake.said], ["baby!"])

        asyncio.run(go())

    def test_stale_queued_fire_is_dropped(self):
        async def go():
            core = _core()
            session, fake = await _attach(core)
            await session.arm(Goal(id="mug", condition="mug", say="water?", modality="speak",
                                   state={"delivery": {"preempt": "queue", "ttl_s": 0.0}}))
            await fake.emit_user_speaking(True)
            await session.deliver_fire(_inc("mug", "water?"))
            await asyncio.sleep(0.01)                        # let ttl_s=0 elapse
            await fake.emit_user_speaking(False)             # flush → stale → dropped
            self.assertEqual(fake.said, [])                 # a late nudge is worse than none

        asyncio.run(go())


if __name__ == "__main__":
    unittest.main()
