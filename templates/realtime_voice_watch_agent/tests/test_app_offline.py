"""END-TO-END offline proof — the REAL app wiring (`app.build_agent`), zero keys, zero pipecat.

One test drives the whole loop the live demo runs: a fake :class:`VoiceEngine` +
``Percept.create(mode="offline")`` (announced-scripted backends) + scripted frames through the PUBLIC
runner (``percept.start``) →

  1. a voice `watch` command authors + arms a real goal (conductor path),
  2. a user utterance gets a spoken reply (component-engine archetype),
  3. a scripted FIRE becomes spoken output via bridge → AttentionIntent → floor → engine.speak,
  4. barge-in during fire playback cancels playout.

If this passes, the template's composition is proven; the live browser run only adds the real wire.
"""
from __future__ import annotations

import asyncio
import unittest

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.backends.fake import FakeVision, ScriptedConductor
from percept.edge.tier0 import gray_frame

from app import build_agent
from tests.fakes import FakeVoiceEngine

W, H = 8, 8
STILL = gray_frame(W, H, 0)
MOVED = gray_frame(W, H, 255)   # still<->moved alternation clears the tier-0 motion gate


def _core() -> Percept:
    """Announced-offline core: scripted vision (frame token → verdict) + scripted conductor
    (utterance substring → authored goal). Wiring, not intelligence."""
    hand = Goal(id="hand", condition="the user raises a hand", say="Hand raised!")
    core = Percept.create(
        mode="offline",
        vision=FakeVision(script={
            "hand-up": Verdict(satisfied=True, confidence=0.95, evidence="a hand is raised"),
            "idle": Verdict(satisfied=False, confidence=0.95, evidence="nothing notable"),
        }),
        conductor=ScriptedConductor({"raise my hand": hand}),
        discover_plugins=False,
    )
    core.router = None   # every turn defers to the (scripted) conductor — deterministic
    return core


class OfflineEndToEndTests(unittest.TestCase):
    def test_the_whole_loop_offline(self):
        async def go():
            core = _core()
            # Component archetype: no conversational model of its own, so the bridge answers turns.
            engine = FakeVoiceEngine(speaks_for_itself=False)
            frames: asyncio.Queue = asyncio.Queue()

            async def source():
                while True:
                    item = await frames.get()
                    if item is None:
                        return
                    yield item

            agent = await build_agent(core, engine, source=source())

            # -- 1. voice watch command → conductor authors + arms a REAL goal ------------
            out = await engine.call_tool("watch", {"command": "tell me the second I raise my hand"})
            self.assertEqual(out["output"], "On it.")            # instant ack, never blocks
            for _ in range(200):                                  # arming is async by design
                if any(g["id"] == "hand" for g in agent.session.list_goals()):
                    break
                await asyncio.sleep(0.005)
            else:
                self.fail("the watch command never armed a goal")
            armed = await engine.call_tool("whats_armed", {})
            self.assertEqual([g["id"] for g in armed["armed"]], ["hand"])

            # -- 2. an utterance gets a reply, spoken back through the engine ---------------
            await engine.emit_transcript("what do you see right now?")
            self.assertEqual(len(engine.spoken), 1)              # ask → offline reply → speak
            self.assertTrue(engine.spoken[0].text)
            reply_count = len(engine.spoken)

            # -- 3. a scripted FIRE becomes spoken output via the floor ----------------------
            await frames.put((0.0, STILL, "idle"))               # prime: gate sees no motion
            await frames.put((1.0, MOVED, "hand-up"))            # escalate → judge → FIRE
            t = 2.0
            for _ in range(400):                                 # async judge dispatch: keep the
                await frames.put((t, STILL if int(t) % 2 else MOVED, "idle"))  # loop pumping
                t += 1.0
                if len(engine.spoken) > reply_count:
                    break
                await asyncio.sleep(0.005)
            else:
                self.fail("the fire never reached the engine")
            fire = engine.spoken[-1]
            self.assertIn("Hand raised!", fire.text)             # the goal's say, via the floor
            self.assertEqual(len(fire.incident_ids), 1)          # correlation ID rode along
            self.assertTrue(fire.incident_ids[0])
            self.assertTrue(agent.renderer.fire_playing)         # renderer holds the floor signal
            self.assertEqual(                                    # telemetry admitted the intent
                agent.telemetry.counters["fires.intents"], 1)

            # -- 4. barge-in during fire playback cancels playout -----------------------------
            await agent.floor.set_user_speaking(True)            # the local VAD edge (media topology)
            self.assertEqual(engine.cancels, 1)                  # playout stopped
            self.assertFalse(agent.renderer.fire_playing)
            self.assertEqual(
                agent.telemetry.counters["playback.cancelled.barge_in"], 1)

            await frames.put(None)
            await agent.stop()

        asyncio.run(go())

    def test_a2v_telemetry_correlates_by_incident_id(self):
        async def go():
            core = _core()
            engine = FakeVoiceEngine()
            frames: asyncio.Queue = asyncio.Queue()

            async def source():
                while True:
                    item = await frames.get()
                    if item is None:
                        return
                    yield item

            agent = await build_agent(core, engine, source=source())
            await agent.bridge.arm(Goal(id="mug", condition="a mug", say="water?"))
            from types import SimpleNamespace

            await agent.bridge.on_incident(
                SimpleNamespace(concern_id="mug", message="water?", id="inc-42", ts=0.0))
            self.assertEqual(engine.spoken[-1].incident_ids, ("inc-42",))
            await engine.emit_output_started()                   # audio-out edge closes the window
            rows = [r for r in agent.telemetry.latencies if r[0] == "inc-42"]
            self.assertEqual(len(rows), 1)                       # correlated by incident ID
            self.assertEqual(rows[0][1], "proactive")
            await frames.put(None)
            await agent.stop()

        asyncio.run(go())


if __name__ == "__main__":
    unittest.main()
