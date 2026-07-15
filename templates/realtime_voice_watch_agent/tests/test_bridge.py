"""PerceptBridge — public-seam wiring: tools, pull-only grounding, evidence tee, fire→intent→floor.

Ported semantics from the SDK-era ``test_realtime_seam.py``/``test_voice_transport.py``, re-expressed
against the template's own seam (FakeVoiceEngine + Floor), fully offline."""
from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from percept import Goal, ModeError, Percept
from percept.backends.fake import FakeConductor, FakeVision, ScriptedConductor
from percept.runtime.session import RuntimeSession

from fire_renderer import FireRenderer
from floor import Floor
from percept_bridge import PerceptBridge
from telemetry import VoiceTelemetry
from tests.fakes import FakeVoiceEngine


def _core(script: dict | None = None) -> Percept:
    """A keyless, ANNOUNCED-offline core; ScriptedConductor authors real goals from substrings."""
    core = Percept.create(mode="offline", vision=FakeVision(),
                          conductor=ScriptedConductor(script or {}), discover_plugins=False)
    core.router = None   # always defer to the (scripted) conductor — deterministic turns
    return core


def _wired(core: Percept, *, engine: FakeVoiceEngine | None = None):
    engine = engine or FakeVoiceEngine()
    session = RuntimeSession(core)
    clock = {"t": 0.0}
    telemetry = VoiceTelemetry(clock=lambda: clock["t"])
    renderer = FireRenderer(engine, telemetry=telemetry, clock=lambda: clock["t"])
    floor = Floor(engine, renderer, clock=lambda: clock["t"], telemetry=telemetry)
    bridge = PerceptBridge(core, session, floor, renderer, telemetry=telemetry)
    bridge.attach(engine)
    return engine, session, floor, renderer, bridge


def _inc(concern_id: str, message: str, id: str = "i1") -> SimpleNamespace:
    return SimpleNamespace(concern_id=concern_id, message=message, id=id, ts=0.0)


async def _drain_arm_tasks(session: RuntimeSession, goal_id: str, tries: int = 200) -> None:
    for _ in range(tries):
        if any(g["id"] == goal_id for g in session.list_goals()):
            return
        await asyncio.sleep(0.005)
    raise AssertionError(f"goal {goal_id!r} never armed")


class GuardTests(unittest.TestCase):
    def test_attach_refuses_a_scripted_brain_behind_a_live_voice(self):
        # The demo bug the guard exists for: a LIVE run wired to a conductor that authors nothing.
        core = Percept.create(mode="live", vision=FakeVision(), conductor=FakeConductor(),
                              discover_plugins=False)
        session = RuntimeSession(core)
        telemetry = VoiceTelemetry()
        engine = FakeVoiceEngine()
        renderer = FireRenderer(engine, telemetry=telemetry)
        floor = Floor(engine, renderer, telemetry=telemetry)
        bridge = PerceptBridge(core, session, floor, renderer, telemetry=telemetry)
        with self.assertRaises(ModeError):
            bridge.attach(engine)

    def test_attach_accepts_an_announced_offline_brain(self):
        _wired(_core())   # mode="offline" is announced-scripted → no ModeError


class CommandToolTests(unittest.TestCase):
    def test_watch_acks_instantly_and_routes_raw_language_to_the_conductor(self):
        async def go():
            goal = Goal(id="baby", condition="the baby is crying", say="Baby!")
            core = _core({"baby": goal})
            seen = {}
            orig = core.arm_from_language

            async def spy(command, *, t=None):
                seen["cmd"] = command
                return await orig(command, t=t)

            core.arm_from_language = spy
            engine, session, _f, _r, _b = _wired(core)
            out = await engine.call_tool("watch", {"command": "tell me the second my baby cries"})
            self.assertEqual(out["output"], "On it.")     # instant ack; never blocks on the conductor
            await _drain_arm_tasks(session, "baby")
            self.assertEqual(seen["cmd"], "tell me the second my baby cries")   # RAW language → conductor

        asyncio.run(go())

    def test_watch_requires_a_command(self):
        async def go():
            engine, *_ = _wired(_core())
            out = await engine.call_tool("watch", {})
            self.assertIn("error", out)

        asyncio.run(go())

    def test_whats_armed_reads_the_session_not_core_goals(self):
        async def go():
            core = _core()
            engine, _s, _f, _r, bridge = _wired(core)
            await bridge.arm(Goal(id="mug", condition="a mug is visible", say="!"))
            out = await engine.call_tool("whats_armed", {})
            self.assertEqual([g["id"] for g in out["armed"]], ["mug"])

        asyncio.run(go())

    def test_stop_watching_disarms_matching(self):
        async def go():
            core = _core()
            engine, session, _f, _r, bridge = _wired(core)
            await bridge.arm(Goal(id="caffeine", condition="the user drinks caffeine", say="!"))
            out = await engine.call_tool("stop_watching", {"what": "caffeine"})
            self.assertIn("caffeine", out["output"])
            self.assertEqual(session.list_goals(), [])

        asyncio.run(go())


class GroundingSnapshotTests(unittest.TestCase):
    """Grounding is PULL-only: mutations only INVALIDATE; the percept_state tool rebuilds on pull;
    nothing is ever pushed into the conversation (no response-triggering frames — see the engine
    adapter's grounding tests for the wire-level half of this contract)."""

    def test_pull_rebuilds_after_structured_arm(self):
        async def go():
            engine, _s, _f, _r, bridge = _wired(_core())
            first = await engine.call_tool("percept_state", {})
            self.assertIn("nothing yet", first["state"])
            await bridge.arm(Goal(id="coffee", condition="a coffee mug is visible", say="water?"))
            second = await engine.call_tool("percept_state", {})
            self.assertIn("coffee", second["state"])       # invalidated on arm → rebuilt on pull

        asyncio.run(go())

    def test_pull_rebuilds_after_arm_from_language_the_path_that_bypasses_session_arm(self):
        # THE regression case on_goal_change exists for: the conductor mutates the registry without
        # touching session.arm — a snapshot keyed to arm/disarm would go stale exactly here.
        async def go():
            goal = Goal(id="kettle", condition="the kettle is boiling", say="Kettle!")
            core = _core({"kettle": goal})
            engine, session, _f, _r, _b = _wired(core)
            stale = await engine.call_tool("percept_state", {})
            self.assertNotIn("kettle", stale["state"])
            await core.arm_from_language("tell me when the kettle boils")
            fresh = await engine.call_tool("percept_state", {})
            self.assertIn("kettle", fresh["state"])

        asyncio.run(go())

    def test_snapshot_carries_goal_ids_never_delivery_scripts(self):
        async def go():
            engine, _s, floor, _r, bridge = _wired(_core())
            await bridge.arm(Goal(id="coffee-goal-1", condition="coffee", say="!"))
            await bridge.on_incident(_inc("coffee-goal-1", "Remember to go easy on the coffee!"))
            out = await engine.call_tool("percept_state", {})
            self.assertIn("coffee-goal-1", out["state"])                 # the id grounds the wire
            self.assertNotIn("go easy on the coffee", out["state"])      # the script does not

        asyncio.run(go())

    def test_nothing_is_pushed_no_engine_speak_on_goal_change(self):
        async def go():
            engine, _s, _f, _r, bridge = _wired(_core())
            await bridge.arm(Goal(id="g", condition="c", say="s"))
            self.assertEqual(engine.spoken, [])   # invalidation only — never a pushed turn

        asyncio.run(go())


class TranscriptTests(unittest.TestCase):
    def test_transcript_tees_to_evidence(self):
        async def go():
            core = _core()
            noted = []
            core.note_mention = lambda text, **kw: noted.append(text)   # spy the evidence hook
            engine, *_ = _wired(core)
            await engine.emit_transcript("that's my daughter Maya")
            self.assertEqual(noted, ["that's my daughter Maya"])        # free evidence, not STT

        asyncio.run(go())

    def test_s2s_engine_transcript_gets_no_double_reply(self):
        async def go():
            engine, *_ = _wired(_core())          # speaks_for_itself=True (default)
            await engine.emit_transcript("what do you see?")
            self.assertEqual(engine.spoken, [])   # the S2S model already answered; we must not

        asyncio.run(go())

    def test_component_engine_transcript_becomes_a_spoken_reply(self):
        async def go():
            engine = FakeVoiceEngine(speaks_for_itself=False)
            engine2, *_ = _wired(_core(), engine=engine)
            await engine.emit_transcript("what do you see?")
            self.assertEqual(len(engine.spoken), 1)              # ask → reply → speak
            self.assertTrue(engine.spoken[0].text)               # offline specialist reply ("ok.")

        asyncio.run(go())


class FireDeliveryTests(unittest.TestCase):
    def test_structured_goal_fire_is_spoken_through_the_floor(self):
        async def go():
            engine, _s, _f, _r, bridge = _wired(_core())
            await bridge.arm(Goal(id="mug", condition="mug", say="water?"))
            await bridge.on_incident(_inc("mug", "water?"))
            self.assertEqual([r.text for r in engine.spoken], ["water?"])
            self.assertEqual(engine.spoken[0].incident_ids, ("i1",))   # telemetry correlation handle

        asyncio.run(go())

    def test_structured_interrupt_goal_preempts_playout(self):
        async def go():
            engine, _s, floor, _r, bridge = _wired(_core())
            await bridge.arm(Goal(id="baby", condition="baby cries", say="baby!",
                                  state={"delivery": {"preempt": "interrupt"}}))
            await floor.set_wire_speaking(True)
            await bridge.on_incident(_inc("baby", "baby!"))
            self.assertEqual(engine.cancels, 1)                        # STRUCTURED_TRUSTED may cut in
            self.assertEqual([r.text for r in engine.spoken], ["baby!"])

        asyncio.run(go())

    def test_language_authored_interrupt_is_clamped_to_queue(self):
        # A voice-armed goal is LANGUAGE_AUTHORED; even if the conductor authored preempt=interrupt,
        # attention_intent_for clamps it — and the bridge maps an unknown goal to the least-privileged
        # defaults, so the fire QUEUES behind the wire instead of cutting in.
        async def go():
            goal = Goal(id="kettle", condition="the kettle is boiling", say="Kettle!",
                        state={"delivery": {"preempt": "interrupt"}})
            core = _core({"kettle": goal})
            engine, session, floor, _r, bridge = _wired(core)
            await core.arm_from_language("watch the kettle")           # language path, not session.arm
            await floor.set_wire_speaking(True)
            await bridge.on_incident(_inc("kettle", "Kettle!"))
            self.assertEqual(engine.cancels, 0)                        # clamped: no cut-in
            self.assertEqual(engine.spoken, [])                        # queued behind the playout
            await floor.set_wire_speaking(False)
            self.assertEqual([r.text for r in engine.spoken], ["Kettle!"])

        asyncio.run(go())

    def test_language_authored_ttl_reaches_the_floor_while_interrupt_still_clamps(self):
        # PCPT-029 Part B: the reported seam gap — a language-authored goal (never through session.arm)
        # is now resolved via session.goal_view, so its AUTHORED ttl_s reaches the effective intent,
        # while the policy STILL clamps its requested interrupt to queue by origin.
        async def go():
            goal = Goal(id="kettle", condition="the kettle is boiling", say="Kettle!",
                        state={"delivery": {"preempt": "interrupt", "ttl_s": 30.0}})
            core = _core({"kettle": goal})
            engine, session, floor, _r, bridge = _wired(core)
            await core.arm_from_language("watch the kettle")           # language path, not session.arm
            captured = {}
            orig_submit = floor.submit

            async def spy(intent):
                captured["intent"] = intent
                return await orig_submit(intent)

            floor.submit = spy
            await floor.set_wire_speaking(True)                        # a turn is live
            await bridge.on_incident(_inc("kettle", "Kettle!"))
            intent = captured["intent"]
            self.assertEqual(intent.ttl_s, 30.0)                       # authored TTL now flows through
            self.assertEqual(intent.preempt, "queue")                 # interrupt clamped by origin
            self.assertEqual(engine.cancels, 0)                       # no cut-in

        asyncio.run(go())

    def test_ttl_from_the_structured_goal_reaches_the_floor(self):
        async def go():
            engine, _s, floor, _r, bridge = _wired(_core())
            await bridge.arm(Goal(id="mug", condition="mug", say="water?",
                                  state={"delivery": {"preempt": "queue", "ttl_s": 0.0}}))
            await floor.set_user_speaking(True)
            await bridge.on_incident(_inc("mug", "water?"))
            floor._clock = lambda: 1.0  # advance past the (zero) TTL  # noqa: SLF001 - template-local
            await floor.set_user_speaking(False)
            self.assertEqual(engine.spoken, [])                        # stale → dropped

        asyncio.run(go())

    def test_webhookless_webhook_preference_degrades_to_log_not_floor(self):
        async def go():
            lines = []
            engine = FakeVoiceEngine()
            core = _core()
            session = RuntimeSession(core)
            telemetry = VoiceTelemetry()
            renderer = FireRenderer(engine, telemetry=telemetry, log=lines.append)
            floor = Floor(engine, renderer, telemetry=telemetry)
            bridge = PerceptBridge(core, session, floor, renderer, telemetry=telemetry)
            bridge.attach(engine)
            await bridge.arm(Goal(id="log-me", condition="c", say="logged",
                                  state={"delivery": {"preferred_channel": "webhook"}}))
            await bridge.on_incident(_inc("log-me", "logged"))
            self.assertEqual(engine.spoken, [])                        # never spoke
            self.assertTrue(any("logged" in line for line in lines))   # fell back to log, loudly

        asyncio.run(go())


class DetachTests(unittest.TestCase):
    def test_detach_unsubscribes_the_goal_change_hook(self):
        async def go():
            engine, _s, _f, _r, bridge = _wired(_core())
            await bridge.arm(Goal(id="a", condition="a", say="a"))
            _ = await engine.call_tool("percept_state", {})            # snapshot now clean
            bridge.detach()
            await bridge.arm(Goal(id="b", condition="b", say="b"))     # post-detach mutation
            # The (still-registered) tool serves the cached snapshot: no invalidation after detach.
            out = await engine.call_tool("percept_state", {})
            self.assertNotIn("b", out["state"])

        asyncio.run(go())


if __name__ == "__main__":
    unittest.main()
