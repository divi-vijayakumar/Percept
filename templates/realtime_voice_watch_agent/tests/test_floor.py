"""Floor — the three-signal priority ladder (user > trusted interrupt > wire reply > deferred).

Drives a real :class:`floor.Floor` + :class:`fire_renderer.FireRenderer` over the FakeVoiceEngine
with an injected clock — TTL, coalescing, barge-in, and the clamp-honoring contract are all
deterministic. Intents are the SDK's public :class:`percept.AttentionIntent` (already EFFECTIVE:
the floor trusts ``preempt`` as granted authority — clamping is the SDK policy's job)."""
from __future__ import annotations

import asyncio
import unittest

from percept import AttentionIntent

from fire_renderer import FireRenderer
from floor import Floor
from telemetry import VoiceTelemetry
from tests.fakes import FakeVoiceEngine


def _intent(msg: str = "fire!", *, incident_id: str = "i1", preempt: str = "queue",
            ttl_s: float | None = None, verbatim: bool = False,
            channel: str | None = None) -> AttentionIntent:
    return AttentionIntent(message=msg, goal_id="g", incident_id=incident_id, ts=0.0,
                           preempt=preempt, ttl_s=ttl_s, verbatim=verbatim,
                           preferred_channel=channel)


def _floor():
    engine = FakeVoiceEngine()
    clock = {"t": 0.0}
    tick = lambda: clock["t"]  # noqa: E731
    telemetry = VoiceTelemetry(clock=tick)
    renderer = FireRenderer(engine, telemetry=telemetry, clock=tick)
    floor = Floor(engine, renderer, clock=tick, telemetry=telemetry)
    return engine, floor, renderer, telemetry, clock


class FreeFloorTests(unittest.TestCase):
    def test_free_floor_speaks_immediately(self):
        async def go():
            engine, floor, renderer, _tel, _clock = _floor()
            await floor.submit(_intent("water?"))
            self.assertEqual([r.text for r in engine.spoken], ["water?"])
            self.assertTrue(renderer.fire_playing)   # renderer state until the output-stopped edge

        asyncio.run(go())

    def test_output_stopped_edge_clears_fire_playing(self):
        async def go():
            engine, floor, renderer, _tel, _clock = _floor()
            renderer.bind_engine_edges()
            await floor.submit(_intent())
            self.assertTrue(renderer.fire_playing)
            await engine.emit_output_stopped()
            self.assertFalse(renderer.fire_playing)

        asyncio.run(go())


class QueueBehindTurnTests(unittest.TestCase):
    def test_gentle_fire_queues_behind_a_turn_then_flushes(self):
        async def go():
            engine, floor, _r, _tel, _clock = _floor()
            await floor.set_user_speaking(True)               # user is mid-sentence
            await floor.submit(_intent("water?"))
            self.assertEqual(engine.spoken, [])               # held, not spoken over the user
            await floor.set_user_speaking(False)              # turn ends → flush
            self.assertEqual([r.text for r in engine.spoken], ["water?"])

        asyncio.run(go())

    def test_deferred_fire_waits_for_the_wire_too(self):
        async def go():
            engine, floor, _r, _tel, _clock = _floor()
            await floor.set_wire_speaking(True)               # the engine is mid-reply
            await floor.submit(_intent())
            self.assertEqual(engine.spoken, [])               # wire reply > deferred
            await floor.set_wire_speaking(False)              # playout ends → floor freed → flush
            self.assertEqual(len(engine.spoken), 1)

        asyncio.run(go())

    def test_stale_queued_fire_is_dropped(self):
        async def go():
            engine, floor, _r, tel, clock = _floor()
            await floor.set_user_speaking(True)
            await floor.submit(_intent(ttl_s=0.5))
            clock["t"] = 1.0                                  # went stale while the user talked
            await floor.set_user_speaking(False)
            self.assertEqual(engine.spoken, [])               # a late nudge is worse than none
            self.assertEqual(tel.counters["fires.dropped.stale"], 1)

        asyncio.run(go())

    def test_fresh_queued_fire_survives_its_ttl(self):
        async def go():
            engine, floor, _r, _tel, clock = _floor()
            await floor.set_user_speaking(True)
            await floor.submit(_intent(ttl_s=5.0))
            clock["t"] = 1.0                                  # within TTL
            await floor.set_user_speaking(False)
            self.assertEqual(len(engine.spoken), 1)

        asyncio.run(go())


class CoalescingTests(unittest.TestCase):
    def test_coalescing_preserves_every_incident_id(self):
        async def go():
            engine, floor, _r, _tel, _clock = _floor()
            await floor.set_user_speaking(True)
            await floor.submit(_intent("kettle's boiling.", incident_id="i1"))
            await floor.submit(_intent("the dog is on the couch.", incident_id="i2"))
            await floor.set_user_speaking(False)
            self.assertEqual(len(engine.spoken), 1)           # ONE utterance for the flush
            req = engine.spoken[0]
            self.assertEqual(req.incident_ids, ("i1", "i2"))  # telemetry never loses an incident
            self.assertIn("kettle's boiling.", req.text)
            self.assertIn("the dog is on the couch.", req.text)

        asyncio.run(go())

    def test_verbatim_intents_are_never_coalesced(self):
        async def go():
            engine, floor, _r, _tel, _clock = _floor()
            await floor.set_user_speaking(True)
            await floor.submit(_intent("casual note", incident_id="i1"))
            await floor.submit(_intent("EXACT WORDS", incident_id="i2", verbatim=True))
            await floor.set_user_speaking(False)
            self.assertEqual(len(engine.spoken), 2)           # verbatim means exactly-these-words
            self.assertEqual([r.verbatim for r in engine.spoken], [False, True])
            all_ids = tuple(i for r in engine.spoken for i in r.incident_ids)
            self.assertEqual(sorted(all_ids), ["i1", "i2"])

        asyncio.run(go())


class InterruptTests(unittest.TestCase):
    def test_trusted_interrupt_cuts_into_wire_playout(self):
        async def go():
            engine, floor, _r, _tel, _clock = _floor()
            await floor.set_wire_speaking(True)
            await floor.submit(_intent("baby!", preempt="interrupt"))
            self.assertEqual(engine.cancels, 1)               # cut into the playout
            self.assertEqual([r.text for r in engine.spoken], ["baby!"])

        asyncio.run(go())

    def test_trusted_interrupt_cuts_into_a_user_turn(self):
        async def go():
            engine, floor, _r, _tel, _clock = _floor()
            await floor.set_user_speaking(True)
            await floor.submit(_intent("baby!", preempt="interrupt"))
            self.assertEqual([r.text for r in engine.spoken], ["baby!"])   # severe fire is worth the rudeness

        asyncio.run(go())

    def test_clamped_interrupt_is_honored_as_queue(self):
        # An untrusted author's "interrupt" arrives here ALREADY clamped to "queue" by the SDK policy;
        # the floor must honor the effective value — never re-elevate, never cut in.
        async def go():
            engine, floor, _r, _tel, _clock = _floor()
            await floor.set_user_speaking(True)
            await floor.submit(_intent("clamped", preempt="queue"))
            self.assertEqual(engine.cancels, 0)               # no cut-in
            self.assertEqual(engine.spoken, [])               # held behind the turn
            await floor.set_user_speaking(False)
            self.assertEqual([r.text for r in engine.spoken], ["clamped"])

        asyncio.run(go())


class DropTests(unittest.TestCase):
    def test_drop_intent_speaks_only_into_a_free_floor(self):
        async def go():
            engine, floor, _r, tel, _clock = _floor()
            await floor.set_wire_speaking(True)
            await floor.submit(_intent("now or never", preempt="drop"))
            self.assertEqual(engine.spoken, [])               # busy → dropped, not queued
            self.assertEqual(tel.counters["fires.dropped.floor_busy"], 1)
            await floor.set_wire_speaking(False)
            self.assertEqual(engine.spoken, [])               # dropped means GONE — no late flush
            await floor.submit(_intent("now", preempt="drop"))
            self.assertEqual(len(engine.spoken), 1)           # free floor → spoken

        asyncio.run(go())


class BargeInTests(unittest.TestCase):
    def test_user_speech_during_fire_playback_cancels_playout(self):
        # The top of the ladder: the user outranks a playing fire. The LOCAL VAD stays active during
        # playback (media.py topology) precisely so this edge can arrive mid-playout.
        async def go():
            engine, floor, renderer, tel, _clock = _floor()
            await floor.submit(_intent("long announcement"))
            self.assertTrue(renderer.fire_playing)
            await floor.set_user_speaking(True)               # barge-in
            self.assertEqual(engine.cancels, 1)               # playout stopped
            self.assertFalse(renderer.fire_playing)
            self.assertEqual(tel.counters["playback.cancelled.barge_in"], 1)

        asyncio.run(go())

    def test_user_speech_with_nothing_playing_cancels_nothing(self):
        async def go():
            engine, floor, _r, _tel, _clock = _floor()
            await floor.set_user_speaking(True)
            self.assertEqual(engine.cancels, 0)

        asyncio.run(go())


if __name__ == "__main__":
    unittest.main()
