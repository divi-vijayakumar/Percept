"""PipecatRealtimeTransport — a held speech-to-speech voice wire for the :class:`VoiceTransport` seam.

The corrected realtime design (``REALTIME-V2V-DESIGN.md``): Gemini Live is a low-latency **voice wire**
(WAV in, frames in, voice out), grounded by the brain — NOT the assistant. Over Pipecat SmallWebRTC (P2P,
browser AEC, no key). Implements the extended seam: command tools → the conductor; ``sync_state`` grounds
the wire; ``on_user_speaking``/``on_transcript`` flow up; ``say`` speaks fires; frames tee to the watch.

**Audio fix (verified via scratch/watch_demo/test_wav_input.py):** Gemini's OWN auto-VAD handles turns
perfectly on a raw audio stream — so we do NOT put a Silero gate on the transport (that gate is what
silenced the mic in the first live attempt). Audio flows straight to Gemini; Gemini decides turns.

Class names the TRANSPORT vendor (Pipecat, a real adapter choice); the MODEL (``gemini-live``) is runtime
config. Module-level Pipecat imports → only importable via ``percept.realtime`` (the ``[realtime]`` extra).
"""
from __future__ import annotations

import asyncio
import inspect
import os
import sys
import time
from typing import Any, Callable

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import LLMContextFrame, LLMMessagesAppendFrame
from pipecat.observers.base_observer import BaseObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

_USER_STARTED = {"UserStartedSpeakingFrame"}
_USER_STOPPED = {"UserStoppedSpeakingFrame"}
# Pipecat emits Bot{Started,Stopped}SpeakingFrame the moment the model's TTS AUDIO begins/ends playing —
# the honest end of the A2V latency window (measuring to say()/enqueue understates by the TTS first-token time).
_BOT_STARTED = {"BotStartedSpeakingFrame"}
_BOT_STOPPED = {"BotStoppedSpeakingFrame"}
_USER_TEXT = {"TranscriptionFrame"}
_IMAGE = {"InputImageRawFrame", "UserImageRawFrame"}
_IN_RATE, _OUT_RATE = 16000, 24000

# Proven-working model (verified end-to-end in scratch/watch_demo/test_wav_input.py: hears + tool-calls).
# When no model is passed, pipecat silently defaults to gemini-2.5-flash-native-audio-preview — a DIFFERENT
# model class (native-audio, not the half-cascade live wire) that we never verified and that connects fine
# but did not respond to mic audio. Always pin an explicit, verified live model.
_DEFAULT_MODEL = "gemini-3.1-flash-live-preview"

# Set PERCEPT_RT_DEBUG=1 to log where the mic→Gemini audio chain breaks (frame counts + first transcript).
_DEBUG = bool(os.environ.get("PERCEPT_RT_DEBUG"))


class _SeamObserver(BaseObserver):
    """Bridges Pipecat pipeline frames → the VoiceTransport up-channels (user speaking / transcript)."""

    def __init__(self, transport: "PipecatRealtimeTransport") -> None:
        super().__init__()
        self._t = transport
        self._dbg_in = 0            # mic → pipeline input-audio frames seen
        self._dbg_out = 0           # Gemini → pipeline output-audio frames seen
        self._dbg_last = 0.0        # last stderr summary time
        self._dbg_heard = False     # printed the first transcript yet?

    async def on_push_frame(self, data: Any) -> None:
        frame = getattr(data, "frame", None)
        if frame is None:
            return
        name = type(frame).__name__
        if _DEBUG:
            self._debug_tap(name, frame)
        try:
            if name in _USER_STARTED:
                await self._t._fire(self._t._on_user_speaking, True)
            elif name in _USER_STOPPED:
                await self._t._fire(self._t._on_user_speaking, False)
            elif name in _BOT_STARTED:
                await self._t._fire(self._t._on_bot_speaking, True)   # TTS audio-out began — the A2V end
            elif name in _BOT_STOPPED:
                await self._t._fire(self._t._on_bot_speaking, False)
            elif name in _USER_TEXT:
                text = getattr(frame, "text", "") or ""
                if text.strip():
                    await self._t._fire(self._t._on_transcript, text)
        except Exception:  # noqa: BLE001 - an observer must never crash the pipeline
            pass

    def _debug_tap(self, name: str, frame: Any) -> None:
        """PERCEPT_RT_DEBUG diagnostics: prove where the mic→Gemini chain breaks. Never raises."""
        try:
            if "AudioRaw" in name:
                if name.startswith(("Input", "UserAudio")):
                    self._dbg_in += 1
                else:
                    self._dbg_out += 1
            elif name in _USER_TEXT and not self._dbg_heard:
                self._dbg_heard = True
                print(f"[rt-debug] Gemini HEARD mic → transcript: {getattr(frame, 'text', '')!r}",
                      file=sys.stderr, flush=True)
            now = time.monotonic()
            if now - self._dbg_last >= 2.0:
                self._dbg_last = now
                verdict = "MIC AUDIO ARRIVING" if self._dbg_in else "NO MIC AUDIO (browser/track issue, not Gemini)"
                # readiness gate: GeminiLiveLLMService DROPS all mic audio until _ready_for_realtime_input flips
                # true (only after it processes an LLMContextFrame). If this stays False, that's the whole bug.
                ready = getattr(self._t._llm, "_ready_for_realtime_input", "?")
                print(f"[rt-debug] in-audio={self._dbg_in} out-audio={self._dbg_out} heard={self._dbg_heard} "
                      f"gemini_ready={ready} → {verdict}", file=sys.stderr, flush=True)
        except Exception:  # noqa: BLE001
            pass


class _VideoTap(FrameProcessor):
    """Pass-through processor teeing inbound camera frames to the watch (demo: same feed as the wire)."""

    def __init__(self, on_image: Callable[[Any], None] | None) -> None:
        super().__init__()
        self._on_image = on_image

    async def process_frame(self, frame: Any, direction: Any) -> None:
        await super().process_frame(frame, direction)
        if self._on_image is not None and type(frame).__name__ in _IMAGE:
            try:
                self._on_image(frame)
            except Exception:  # noqa: BLE001
                pass
        await self.push_frame(frame, direction)


class PipecatRealtimeTransport:
    """A held Gemini-Live voice wire over Pipecat SmallWebRTC, implementing the extended VoiceTransport.

    Built lazily: ``register_tool``/``on_*``/``set_frame_provider`` accumulate config (called by
    ``attach_transport``), then :meth:`start` constructs + runs the pipeline against the browser
    connection. Gemini's own VAD drives turns — no Silero gate."""

    def __init__(self, webrtc_connection: Any, *, gemini_api_key: str, model: str | None = None,
                 voice: str = "Aoede", system: str = "", on_video_frame: Callable[[Any], None] | None = None) -> None:
        self._conn = webrtc_connection
        self._api_key = gemini_api_key
        self._model = model
        self._voice = voice
        self._system = system
        self._on_video_frame = on_video_frame
        self._tools: dict[str, tuple[dict, Any]] = {}
        self._on_barge_in: Any = None
        self._on_event: Any = None
        self._on_user_speaking: Any = None
        self._on_bot_speaking: Any = None
        self._on_transcript: Any = None
        self._on_utterance: Any = None
        self._frame_provider: Any = None
        self._task: Any = None
        self._runner_task: Any = None
        self._llm: Any = None

    # -- registration (before start; idempotent) -------------------------
    def on_utterance(self, cb: Any) -> None: self._on_utterance = cb          # unused in S2S (Gemini replies)
    def on_barge_in(self, cb: Any) -> None: self._on_barge_in = cb
    def on_event(self, cb: Any) -> None: self._on_event = cb
    def on_user_speaking(self, cb: Any) -> None: self._on_user_speaking = cb
    def on_bot_speaking(self, cb: Any) -> None: self._on_bot_speaking = cb  # audio-out edge (A2V end)
    def on_transcript(self, cb: Any) -> None: self._on_transcript = cb
    def set_frame_provider(self, provider: Any) -> None: self._frame_provider = provider

    def register_tool(self, name: str, schema: dict, handler: Any) -> None:
        self._tools[name] = (schema, handler)

    # -- lifecycle -------------------------------------------------------
    async def start(self) -> None:
        # NO vad_analyzer: Gemini's own auto-VAD drives turns (verified). A Silero gate here silenced the
        # mic in the first live attempt. video_in for the wire's reactive eyes + the watch tee.
        params = TransportParams(audio_in_enabled=True, audio_out_enabled=True, video_in_enabled=True,
                                 audio_in_sample_rate=_IN_RATE, audio_out_sample_rate=_OUT_RATE)
        transport = SmallWebRTCTransport(webrtc_connection=self._conn, params=params)

        fn_schemas = [FunctionSchema(name=n, description=str(s.get("description") or f"call {n}"),
                                     properties=s.get("properties", {}), required=s.get("required", []))
                      for n, (s, _h) in self._tools.items()]
        tools = ToolsSchema(standard_tools=fn_schemas) if fn_schemas else None
        # tools live on the CONTEXT only (not on init) — passing them in both places triggers a mid-session
        # _reconnect() in GeminiLiveLLMService ("tools provided both at init time and in context"). Dispatch
        # still works via register_function below.
        llm = GeminiLiveLLMService(api_key=self._api_key, voice_id=self._voice,
                                   system_instruction=self._system, model=self._model or _DEFAULT_MODEL)
        self._llm = llm
        for n, (_s, handler) in self._tools.items():
            llm.register_function(n, self._wrap_tool(handler))

        # A context aggregator MUST be in the pipeline or function calls can't dispatch
        # ("Function calls are not supported without a context object").
        # realtime_service_mode=True because the model (Gemini Live) owns turn-taking, not local aggregation.
        # Seed an opening user turn so the seeded context below triggers an audible greeting (zero-ambiguity
        # signal that the readiness gate flipped).
        seed = [{"role": "user", "content": "Greet the user in one short sentence."}]
        context = (LLMContext(messages=seed, tools=tools) if tools is not None
                   else LLMContext(messages=seed))
        agg = LLMContextAggregatorPair(context, realtime_service_mode=True)

        procs: list[Any] = [transport.input(), agg.user()]
        if self._on_video_frame is not None:
            procs.append(_VideoTap(self._on_video_frame))
        procs += [llm, transport.output(), agg.assistant()]
        self._task = PipelineTask(Pipeline(procs),
                                  params=PipelineParams(audio_in_sample_rate=_IN_RATE, audio_out_sample_rate=_OUT_RATE),
                                  observers=[_SeamObserver(self)])
        self._runner_task = asyncio.ensure_future(PipelineRunner(handle_sigint=False).run(self._task))

        # CRITICAL — the readiness kick. GeminiLiveLLMService silently DROPS every InputAudioRawFrame until it
        # has processed an LLMContextFrame (_handle_context → _ready_for_realtime_input = True). In this
        # pipeline the only producer of that frame is agg.user(), which emits only on a COMPLETED user turn —
        # but there is no local VAD (removed deliberately) and Gemini Live never emits UserStopped/Started
        # SpeakingFrame. Deadlock: no context → no audio → no turn → no context. Seed it once at start so the
        # gate opens immediately; the (harmless) initial reconnect for tools now happens at t≈0, not mid-call.
        await self._task.queue_frame(LLMContextFrame(context=context))

    async def stop(self) -> None:
        try:
            if self._task is not None:
                await self._task.cancel()
        except Exception:  # noqa: BLE001 - teardown must not raise
            pass

    # -- down-channel ----------------------------------------------------
    async def say(self, reply: Any, *, verbatim: bool = False) -> None:
        """Speak a proactive fire by injecting a turn into the held session (Gemini renders it in-voice)."""
        text = getattr(reply, "text", None)
        if not text or self._task is None:
            return
        instr = (f"Say this to the user now, VERBATIM, exactly, do not paraphrase: {text}" if verbatim
                 else f"Proactively tell the user now, briefly and in your own voice: {text}")
        await self._queue(instr, run_llm=True)

    async def sync_state(self, summary: str) -> None:
        """Ground the wire: push brain-state into context WITHOUT a response (run_llm=False)."""
        if summary:
            await self._queue(summary, run_llm=False)

    async def interrupt(self) -> None:
        return  # Gemini Live handles barge-in natively (server VAD)

    # -- internal --------------------------------------------------------
    async def _queue(self, content: str, *, run_llm: bool) -> None:
        if self._task is None:
            return
        try:
            await self._task.queue_frame(
                LLMMessagesAppendFrame(messages=[{"role": "user", "content": content}], run_llm=run_llm))
        except Exception:  # noqa: BLE001
            pass

    def _wrap_tool(self, handler: Any) -> Any:
        async def _fn(params: Any) -> None:
            try:
                res = handler(dict(getattr(params, "arguments", None) or {}))
                if inspect.isawaitable(res):
                    res = await res
            except Exception as exc:  # noqa: BLE001
                res = {"error": str(exc)}
            await params.result_callback(res or {})
        return _fn

    @staticmethod
    async def _fire(cb: Any, *args: Any) -> None:
        if cb is None:
            return
        res = cb(*args)
        if inspect.isawaitable(res):
            await res
