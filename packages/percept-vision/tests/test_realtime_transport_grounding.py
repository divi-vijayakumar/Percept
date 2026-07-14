"""GROUNDING is pull, DELIVERY is the only unprompted-audio path — the channel-separation contract.

Regression for the "keeps talking about the armed goals" bug: `sync_state` used to push brain state
into the Gemini Live context as a user-role LLMMessagesAppendFrame with run_llm=False — but pipecat's
GeminiLiveLLMService ignores run_llm and answers EVERY appended turn out loud (send_client_content with
turn_complete=True, plus a forced realtime input on Gemini 3.x). Every arm/disarm pulse became an
unprompted monologue, and the state text (user-role) steered every later noise-turn back to the armed
goals. The fix: sync_state stashes; the model PULLS via a self-registered `percept_state` tool; only
say() may queue a response-triggering frame; and the session's state summary never carries delivery
scripts (incident messages).
"""
from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

try:
    from percept.realtime import PipecatRealtimeTransport
    _HAVE_PIPECAT = True
except Exception:  # noqa: BLE001 - pipecat is an extra ([realtime]); absent in the stdlib lane
    _HAVE_PIPECAT = False


class _RecorderTask:
    """Stands in for the PipelineTask: records every queued frame."""

    def __init__(self) -> None:
        self.frames: list = []

    async def queue_frame(self, frame) -> None:
        self.frames.append(frame)


@unittest.skipUnless(_HAVE_PIPECAT, "needs pipecat (percept-vision[realtime])")
class TestGroundingIsPull(unittest.TestCase):
    def _transport(self) -> "PipecatRealtimeTransport":
        return PipecatRealtimeTransport(object(), gemini_api_key="k")

    def test_sync_state_queues_nothing_even_mid_session(self):
        """The hard contract: grounding must NOT produce a response-triggering frame — ever."""
        t = self._transport()
        t._task = _RecorderTask()  # noqa: SLF001 - simulate a running pipeline
        asyncio.run(t.sync_state("[percept state] watching for → g1: coffee."))
        self.assertEqual(t._task.frames, [])  # noqa: SLF001 - nothing pushed to the wire
        self.assertIn("coffee", t._state_summary)  # noqa: SLF001 - stashed for pull

    def test_percept_state_tool_serves_the_stash(self):
        t = self._transport()
        t._ensure_state_tool()  # noqa: SLF001 - what start() calls
        self.assertIn("percept_state", t._tools)  # noqa: SLF001
        asyncio.run(t.sync_state("[percept state] watching for → g1: coffee."))
        _schema, handler = t._tools["percept_state"]  # noqa: SLF001
        out = asyncio.run(handler({}))
        self.assertIn("coffee", out["state"])

    def test_app_registered_state_tool_wins(self):
        t = self._transport()

        async def mine(_args):
            return {"state": "app-owned"}

        t.register_tool("percept_state", {"properties": {}}, mine)
        t._ensure_state_tool()  # noqa: SLF001 - must not clobber the app's tool
        _schema, handler = t._tools["percept_state"]  # noqa: SLF001
        self.assertEqual(asyncio.run(handler({}))["state"], "app-owned")

    def test_say_still_queues_a_response(self):
        """Delivery stays push: say() is the ONE path allowed to trigger unprompted audio."""
        from pipecat.frames.frames import LLMMessagesAppendFrame

        t = self._transport()
        t._task = _RecorderTask()  # noqa: SLF001
        asyncio.run(t.say(SimpleNamespace(text="the hairdryer is running"), verbatim=True))
        self.assertEqual(len(t._task.frames), 1)  # noqa: SLF001
        frame = t._task.frames[0]  # noqa: SLF001
        self.assertIsInstance(frame, LLMMessagesAppendFrame)
        self.assertTrue(frame.run_llm)
        self.assertIn("hairdryer", frame.messages[0]["content"])

    def test_grounding_contract_rides_the_system_instruction(self):
        self.assertIn("percept_state", PipecatRealtimeTransport._GROUNDING_CONTRACT)  # noqa: SLF001
        self.assertIn("ONLY when the system", PipecatRealtimeTransport._GROUNDING_CONTRACT)  # noqa: SLF001


@unittest.skipUnless(_HAVE_PIPECAT, "needs pipecat (percept-vision[realtime])")
class TestCallbackFanout(unittest.TestCase):
    """The speaking/transcript edges are subscriber LISTS: attach_transport and the app both register,
    in either order — a single slot made that last-write-wins (attach silently killed the app's V2V
    clock and mute-guard, regression)."""

    def _transport(self) -> "PipecatRealtimeTransport":
        return PipecatRealtimeTransport(object(), gemini_api_key="k")

    def test_both_subscribers_get_the_edge_in_order(self):
        t = self._transport()
        got: list = []
        t.on_bot_speaking(lambda s: got.append(("app", s)))
        t.on_bot_speaking(lambda s: got.append(("session", s)))     # attach registering second
        asyncio.run(t._fire(t._on_bot_speaking, True))  # noqa: SLF001
        self.assertEqual(got, [("app", True), ("session", True)])

    def test_one_bad_subscriber_never_silences_its_peers(self):
        t = self._transport()
        got: list = []

        def bad(_s):
            raise RuntimeError("boom")

        t.on_user_speaking(bad)
        t.on_user_speaking(lambda s: got.append(s))
        asyncio.run(t._fire(t._on_user_speaking, False))  # noqa: SLF001
        self.assertEqual(got, [False])

    def test_transcript_synthesizes_one_turn_end_per_burst(self):
        """Gemini Live emits NO UserStarted/StoppedSpeakingFrame — a transcript is the only turn
        boundary. The observer must synthesize on_user_speaking(False) before the transcript, and
        collapse a same-turn burst of transcripts into ONE edge."""
        from percept.realtime.transport import _SeamObserver

        t = self._transport()
        edges: list = []
        texts: list = []
        t.on_user_speaking(lambda s: edges.append(s))
        t.on_transcript(lambda x: texts.append(x))
        obs = _SeamObserver(t)

        class TranscriptionFrame:            # observer dispatches on the type NAME
            def __init__(self, text: str) -> None:
                self.text = text

        async def push(text: str) -> None:
            await obs.on_push_frame(SimpleNamespace(frame=TranscriptionFrame(text)))

        async def burst() -> None:
            await push("Okay, can you watch my caffeine intake?")
            await push("And my hydration goals.")   # same burst → no second edge

        asyncio.run(burst())
        self.assertEqual(edges, [False])                             # one synthesized turn-end
        self.assertEqual(len(texts), 2)                              # both transcripts delivered


class TestSummaryCarriesNoDeliveryScripts(unittest.TestCase):
    """The session's grounding summary must never contain incident MESSAGES (the `say` scripts)."""

    def test_recent_alerts_are_goal_ids_only(self):
        from percept import Percept
        from percept.runtime.session import RuntimeSession

        core = Percept.create(vision="fake", discover_plugins=False)
        session = RuntimeSession(core)
        session.incidents.append(SimpleNamespace(
            concern_id="coffee-goal-1", message="Remember to go easy on the coffee!", id="i1", ts=0.0))
        summary = session._brain_state_summary()  # noqa: SLF001
        self.assertIn("coffee-goal-1", summary)                       # the id grounds the wire
        self.assertNotIn("go easy on the coffee", summary)            # the script does not


if __name__ == "__main__":
    unittest.main()
