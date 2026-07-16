"""PipecatVoiceEngine — GROUNDING is pull, DELIVERY is the only unprompted-audio path.

Ported from the SDK-era ``test_realtime_transport_grounding.py`` (the transport this engine is a
verbatim port of). Regression for the "keeps talking about the armed goals" bug: state used to be
pushed into the Gemini Live context as a user-role LLMMessagesAppendFrame with run_llm=False — but
pipecat's GeminiLiveLLMService ignores run_llm and answers EVERY appended turn out loud, and the
user-role state text steered every later noise-turn back to the armed goals. The fix: state is
STASHED; the model PULLS via the self-registered `percept_state` tool; only speak() may queue a
response-triggering frame.

Needs pipecat (the template's own dependency) — skips cleanly keyless/pipecat-less.
"""
from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

import engine_adapter
from engine_adapter import DEFAULT_MODEL, SpeechRequest

_HAVE_PIPECAT = engine_adapter.HAVE_PIPECAT


class _RecorderTask:
    """Stands in for the PipelineTask: records every queued frame."""

    def __init__(self) -> None:
        self.frames: list = []

    async def queue_frame(self, frame) -> None:
        self.frames.append(frame)


def _engine() -> "engine_adapter.PipecatVoiceEngine":
    return engine_adapter.PipecatVoiceEngine(object(), gemini_api_key="k")


@unittest.skipUnless(_HAVE_PIPECAT, "needs pipecat (template dependency; pip install -e .)")
class TestGroundingIsPull(unittest.TestCase):
    def test_set_state_summary_queues_nothing_even_mid_session(self):
        """The hard contract: grounding must NOT produce a response-triggering frame — ever."""
        e = _engine()
        e._task = _RecorderTask()  # noqa: SLF001 - simulate a running pipeline
        e.set_state_summary("[percept state] watching for → g1: coffee.")
        self.assertEqual(e._task.frames, [])  # noqa: SLF001 - nothing pushed to the wire
        self.assertIn("coffee", e._state_summary)  # noqa: SLF001 - stashed for pull

    def test_percept_state_tool_serves_the_stash(self):
        e = _engine()
        e._ensure_state_tool()  # noqa: SLF001 - what start() calls
        self.assertIn("percept_state", e._tools)  # noqa: SLF001
        e.set_state_summary("[percept state] watching for → g1: coffee.")
        _schema, handler = e._tools["percept_state"]  # noqa: SLF001
        out = asyncio.run(handler({}))
        self.assertIn("coffee", out["state"])

    def test_app_registered_state_tool_wins(self):
        e = _engine()

        async def mine(_args):
            return {"state": "app-owned"}

        e.register_tool("percept_state", {"properties": {}}, mine)
        e._ensure_state_tool()  # noqa: SLF001 - must not clobber the app's tool
        _schema, handler = e._tools["percept_state"]  # noqa: SLF001
        self.assertEqual(asyncio.run(handler({}))["state"], "app-owned")

    def test_speak_still_queues_a_response(self):
        """Delivery stays push: speak() is the ONE path allowed to trigger unprompted audio."""
        from pipecat.frames.frames import LLMMessagesAppendFrame

        e = _engine()
        e._task = _RecorderTask()  # noqa: SLF001
        asyncio.run(e.speak(SpeechRequest(text="the hairdryer is running", verbatim=True)))
        self.assertEqual(len(e._task.frames), 1)  # noqa: SLF001
        frame = e._task.frames[0]  # noqa: SLF001
        self.assertIsInstance(frame, LLMMessagesAppendFrame)
        self.assertTrue(frame.run_llm)
        self.assertIn("hairdryer", frame.messages[0]["content"])
        self.assertIn("VERBATIM", frame.messages[0]["content"])

    def test_empty_speech_request_queues_nothing(self):
        e = _engine()
        e._task = _RecorderTask()  # noqa: SLF001
        asyncio.run(e.speak(SpeechRequest(text="")))
        self.assertEqual(e._task.frames, [])  # noqa: SLF001

    def test_grounding_contract_rides_the_system_instruction(self):
        self.assertIn("percept_state", engine_adapter.PipecatVoiceEngine._GROUNDING_CONTRACT)  # noqa: SLF001
        self.assertIn("ONLY when the system", engine_adapter.PipecatVoiceEngine._GROUNDING_CONTRACT)  # noqa: SLF001

    def test_the_live_model_is_pinned_never_pipecats_silent_default(self):
        # Pipecat's own default (gemini-2.5-flash-native-audio-preview) is a DIFFERENT model class
        # that connects fine and hears nothing. The pin is the verified half-cascade live wire.
        self.assertEqual(DEFAULT_MODEL, "gemini-3.1-flash-live-preview")


@unittest.skipUnless(_HAVE_PIPECAT, "needs pipecat (template dependency; pip install -e .)")
class TestCallbackFanout(unittest.TestCase):
    """The speaking/transcript/output edges are subscriber LISTS: floor, telemetry, and bridge all
    register, in any order — a single slot made that last-write-wins (one registration silently
    killed another's V2V clock and mute-guard; SDK-era regression)."""

    def test_both_subscribers_get_the_edge_in_order(self):
        e = _engine()
        got: list = []
        e.on_output_started(lambda: got.append("telemetry"))
        e.on_output_started(lambda: got.append("floor"))     # second registration is ADDITIVE
        asyncio.run(e._fire(e._on_output_started))  # noqa: SLF001
        self.assertEqual(got, ["telemetry", "floor"])

    def test_one_bad_subscriber_never_silences_its_peers(self):
        e = _engine()
        got: list = []

        def bad(_s):
            raise RuntimeError("boom")

        e.on_user_speaking(bad)
        e.on_user_speaking(lambda s: got.append(s))
        asyncio.run(e._fire(e._on_user_speaking, False))  # noqa: SLF001
        self.assertEqual(got, [False])

    def test_unsubscribe_detaches_exactly_that_callback(self):
        e = _engine()
        got: list = []
        unsub = e.on_user_transcript(lambda t: got.append(("a", t)))
        e.on_user_transcript(lambda t: got.append(("b", t)))
        unsub()
        asyncio.run(e._fire(e._on_transcript, "hi"))  # noqa: SLF001
        self.assertEqual(got, [("b", "hi")])

    def test_transcript_synthesizes_one_turn_end_per_burst(self):
        """Gemini Live emits NO UserStarted/StoppedSpeakingFrame — a transcript is the only turn
        boundary. The observer must synthesize on_user_speaking(False) before the transcript, and
        collapse a same-turn burst of transcripts into ONE edge."""
        from engine_adapter import _SeamObserver

        e = _engine()
        edges: list = []
        texts: list = []
        e.on_user_speaking(lambda s: edges.append(s))
        e.on_user_transcript(lambda x: texts.append(x))
        obs = _SeamObserver(e)

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

    def test_bot_speaking_frames_map_to_output_edges(self):
        from engine_adapter import _SeamObserver

        e = _engine()
        got: list = []
        e.on_output_started(lambda: got.append("start"))
        e.on_output_stopped(lambda: got.append("stop"))
        obs = _SeamObserver(e)

        class BotStartedSpeakingFrame:  # dispatch is on the type NAME
            pass

        class BotStoppedSpeakingFrame:
            pass

        async def drive() -> None:
            await obs.on_push_frame(SimpleNamespace(frame=BotStartedSpeakingFrame()))
            await obs.on_push_frame(SimpleNamespace(frame=BotStoppedSpeakingFrame()))

        asyncio.run(drive())
        self.assertEqual(got, ["start", "stop"])


class TestTemplateLocalTypes(unittest.TestCase):
    """The seam types work without pipecat — keyless CI exercises at least this much."""

    def test_speech_request_defaults(self):
        req = SpeechRequest(text="hi")
        self.assertFalse(req.verbatim)
        self.assertEqual(req.incident_ids, ())
        self.assertIsNone(req.expires_at)

    def test_fake_engine_satisfies_the_protocol(self):
        from engine_adapter import VoiceEngine
        from tests.fakes import FakeVoiceEngine

        self.assertIsInstance(FakeVoiceEngine(), VoiceEngine)   # runtime_checkable structural check


if __name__ == "__main__":
    unittest.main()
