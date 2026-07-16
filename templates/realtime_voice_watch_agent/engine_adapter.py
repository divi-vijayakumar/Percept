"""PipecatVoiceEngine — a held speech-to-speech voice wire behind the template-local `VoiceEngine` seam.

The corrected realtime design (``REALTIME-V2V-DESIGN.md``): Gemini Live is a low-latency **voice wire**
(WAV in, frames in, voice out), grounded by the brain — NOT the assistant. Over Pipecat SmallWebRTC (P2P,
browser AEC, no key). The bridge registers command tools → the conductor; grounding is served via the
pull-only ``percept_state`` tool; transcripts flow up; ``speak`` renders fires; frames tee to the watch.

**Template-local voice types.** This module deliberately does NOT import the SDK's deprecated
``VoiceTransport``/``UtteranceEvent``/``Reply`` contracts — the application (this template) is the
component that needs engine portability, so it owns the seam: :class:`SpeechRequest` +
:class:`VoiceEngine`. Swap Pipecat/Gemini-Live for LiveKit/OpenAI-Realtime by implementing the Protocol.

**Audio fix (verified via scratch/watch_demo/test_wav_input.py):** Gemini's OWN auto-VAD handles turns
perfectly on a raw audio stream — so we do NOT put a Silero gate on the transport (that gate is what
silenced the mic in the first live attempt). Audio flows straight to Gemini; Gemini decides turns.

Class names the TRANSPORT vendor (Pipecat, a real adapter choice); the MODEL (``gemini-live``) is runtime
config. Module-level Pipecat imports → import this module only on the live path (tests skipUnless).
"""
from __future__ import annotations

import asyncio
import inspect
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

#: Every ``on_*`` registration returns one of these — call it to detach the callback.
Unsubscribe = Callable[[], None]


@dataclass(frozen=True)
class SpeechRequest:
    """One utterance the application asks the engine to render in-voice.

    ``incident_ids`` carries the (possibly coalesced) fired-incident IDs straight into telemetry —
    the A2V clock is correlated by incident ID, not by guessing which audio-out edge belongs to what.
    ``expires_at`` (monotonic) is informational — staleness is the Floor's job (TTL drop-if-stale
    happens BEFORE a request reaches the engine); an engine may use it as a last-resort guard."""

    text: str
    verbatim: bool = False
    incident_ids: tuple[str, ...] = ()
    expires_at: float | None = None


@runtime_checkable
class VoiceEngine(Protocol):
    """The template's minimal voice-engine seam (the app decides everything else).

    An engine owns mic + speaker + AEC + endpointing + its own conversational model (S2S) or none
    (component STT/TTS). ``speaks_for_itself`` (duck-typed attribute, default ``True``) declares the
    archetype: a speech-to-speech engine answers user turns itself; a component engine expects the
    bridge to run ``session.ask`` on each transcript and ``speak`` the reply back."""

    async def speak(self, request: SpeechRequest) -> None: ...
    async def cancel_output(self) -> None: ...
    def on_user_transcript(self, callback: Callable[[str], Any]) -> Unsubscribe: ...
    def on_output_started(self, callback: Callable[[], Any]) -> Unsubscribe: ...
    def on_output_stopped(self, callback: Callable[[], Any]) -> Unsubscribe: ...


# --------------------------------------------------------------------------------------------------
# Pipecat / Gemini Live implementation (the verbatim port of percept/realtime/transport.py internals).
# Heavy vendor imports live inside a guard so `import engine_adapter` (for the types above) stays
# possible without pipecat; constructing PipecatVoiceEngine without it raises loudly.
# --------------------------------------------------------------------------------------------------
try:
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

    HAVE_PIPECAT = True
except Exception:  # noqa: BLE001 - pipecat is the template's live dep; type-only imports stay usable
    HAVE_PIPECAT = False
    BaseObserver = object  # type: ignore[assignment, misc]
    FrameProcessor = object  # type: ignore[assignment, misc]

_USER_STARTED = {"UserStartedSpeakingFrame"}
_USER_STOPPED = {"UserStoppedSpeakingFrame"}
# Pipecat emits Bot{Started,Stopped}SpeakingFrame the moment the model's TTS AUDIO begins/ends playing —
# the honest end of the A2V latency window (measuring to speak()/enqueue understates by the TTS
# first-token time).
_BOT_STARTED = {"BotStartedSpeakingFrame"}
_BOT_STOPPED = {"BotStoppedSpeakingFrame"}
_USER_TEXT = {"TranscriptionFrame"}
_IMAGE = {"InputImageRawFrame", "UserImageRawFrame"}
_AUDIO_IN = {"InputAudioRawFrame"}  # the browser mic frames — AEC'd in-browser before they arrive
_IN_RATE, _OUT_RATE = 16000, 24000

# Proven-working model (verified end-to-end in scratch/watch_demo/test_wav_input.py: hears + tool-calls).
# When no model is passed, pipecat silently defaults to gemini-2.5-flash-native-audio-preview — a DIFFERENT
# model class (native-audio, not the half-cascade live wire) that we never verified and that connects fine
# but did not respond to mic audio. Always pin an explicit, verified live model.
DEFAULT_MODEL = "gemini-3.1-flash-live-preview"

# Set PERCEPT_RT_DEBUG=1 to log where the mic→Gemini audio chain breaks (frame counts + first transcript).
_DEBUG = bool(os.environ.get("PERCEPT_RT_DEBUG"))


class _SeamObserver(BaseObserver):
    """Bridges Pipecat pipeline frames → the VoiceEngine up-channels (user speaking / transcript /
    output edges)."""

    def __init__(self, engine: "PipecatVoiceEngine") -> None:
        super().__init__()
        self._t = engine
        self._last_turn_end = 0.0   # last SYNTHESIZED turn-end edge (dedup a burst of transcripts)
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
                await self._t._fire(self._t._on_output_started)   # TTS audio-out began — the A2V end
            elif name in _BOT_STOPPED:
                await self._t._fire(self._t._on_output_stopped)
            elif name in _USER_TEXT:
                text = getattr(frame, "text", "") or ""
                if text.strip():
                    # SYNTHESIZED turn-end: GeminiLiveLLMService never emits UserStarted/Stopped
                    # SpeakingFrame (pipecat llm.py — "the API exposes an `interrupted` event but no
                    # turn-start/-end"), so on this backend a transcript is the ONLY turn boundary we
                    # ever see. Without this edge, every on_user_speaking consumer is dead code: the
                    # floor's queue-behind-turn never flushes and a V2V clock never starts. A turn's
                    # transcripts can arrive as a burst — collapse edges within 1s into one.
                    now = time.monotonic()
                    if now - self._last_turn_end > 1.0:
                        self._last_turn_end = now
                        await self._t._fire(self._t._on_user_speaking, False)
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


class _AudioTap(FrameProcessor):
    """Pass-through processor teeing inbound MIC frames to the app's audio splitter.

    The exact mirror of :class:`_VideoTap`, one channel over. The browser mic reaches the pipeline as
    ``InputAudioRawFrame`` (already AEC'd in-browser, so the assistant's own voice is gone) — the app
    tees it into its three-way split (see ``media.AudioSplitter``): the conversation engine keeps the
    frame untouched (Gemini still hears the user), the local energy VAD derives ``user_speaking``, and
    the Percept world-audio branch feeds ``core.push_audio``."""

    def __init__(self, on_audio: Callable[[Any], None] | None) -> None:
        super().__init__()
        self._on_audio = on_audio

    async def process_frame(self, frame: Any, direction: Any) -> None:
        await super().process_frame(frame, direction)
        if self._on_audio is not None and type(frame).__name__ in _AUDIO_IN:
            try:
                self._on_audio(frame)
            except Exception:  # noqa: BLE001 - a bad tap consumer must never crash the wire
                pass
        await self.push_frame(frame, direction)


class PipecatVoiceEngine:
    """A held Gemini-Live voice wire over Pipecat SmallWebRTC, implementing :class:`VoiceEngine`.

    Built lazily: ``register_tool``/``on_*`` accumulate config (called by the bridge's ``attach``),
    then :meth:`start` constructs + runs the pipeline against the browser connection. Gemini's own VAD
    drives turns — no Silero gate."""

    #: Speech-to-speech: the engine's model answers user turns itself; the bridge must not run
    #: ``session.ask`` per transcript (transcripts are free evidence, not utterances to answer).
    speaks_for_itself = True

    def __init__(self, webrtc_connection: Any, *, gemini_api_key: str, model: str | None = None,
                 voice: str = "Aoede", system: str = "", on_video_frame: Callable[[Any], None] | None = None,
                 on_audio_frame: Callable[[Any], None] | None = None) -> None:
        if not HAVE_PIPECAT:  # loud, at construction — never a silent no-op engine
            raise ModuleNotFoundError(
                "PipecatVoiceEngine needs pipecat — `pip install -e .` in this template directory "
                "(pulls pipecat-ai[webrtc,google] + the prebuilt browser client)."
            )
        self._conn = webrtc_connection
        self._api_key = gemini_api_key
        self._model = model
        self._voice = voice
        self._system = system
        self._on_video_frame = on_video_frame
        self._on_audio_frame = on_audio_frame   # tee AEC'd mic → the app's three-way audio split
        self._tools: dict[str, tuple[dict, Any]] = {}
        # user-speaking / output-edge / transcript are SUBSCRIBER LISTS, not slots: the floor, the
        # telemetry, and the bridge all register for the same edges, in any order. A single slot made
        # that last-write-wins — one registration silently killed another's handlers (dead V2V clocks,
        # dead mute-guards) in the SDK-era wiring.
        self._on_user_speaking: list[Any] = []
        self._on_output_started: list[Any] = []
        self._on_output_stopped: list[Any] = []
        self._on_transcript: list[Any] = []
        self._task: Any = None
        self._runner_task: Any = None
        self._llm: Any = None
        self._state_summary: str = ""   # last grounding summary — served via the percept_state tool (pull)

    # -- registration (before start; every edge is ADDITIVE — see __init__) --------------------------
    def on_user_transcript(self, callback: Callable[[str], Any]) -> Unsubscribe:
        return self._subscribe(self._on_transcript, callback)

    def on_output_started(self, callback: Callable[[], Any]) -> Unsubscribe:
        """Audio-out actually began (the honest A2V end — see :class:`_SeamObserver`)."""
        return self._subscribe(self._on_output_started, callback)

    def on_output_stopped(self, callback: Callable[[], Any]) -> Unsubscribe:
        return self._subscribe(self._on_output_stopped, callback)

    def on_user_speaking(self, callback: Callable[[bool], Any]) -> Unsubscribe:
        """Engine-native turn edges (incl. the SYNTHESIZED transcript turn-end). These are extra
        signal for the floor's flush — they never replace the app's local energy VAD, which is the
        only ``user_speaking`` source that stays alive during assistant playback (barge-in)."""
        return self._subscribe(self._on_user_speaking, callback)

    @staticmethod
    def _subscribe(subscribers: list, callback: Any) -> Unsubscribe:
        subscribers.append(callback)

        def _unsub() -> None:
            try:
                subscribers.remove(callback)
            except ValueError:
                pass

        return _unsub

    def register_tool(self, name: str, schema: dict, handler: Any) -> None:
        self._tools[name] = (schema, handler)

    def _ensure_state_tool(self) -> None:
        """Self-register the GROUNDING pull tool (idempotent; an app-registered `percept_state` wins).

        On this backend grounding must be PULL (see :meth:`set_state_summary`): the model calls
        ``percept_state`` when it needs to know what the brain is doing, instead of the brain pushing
        state into the conversation as user turns it would answer out loud."""
        if "percept_state" in self._tools:
            return

        async def _percept_state(_args: dict) -> dict:
            return {"state": self._state_summary or "no state yet"}

        self._tools["percept_state"] = (
            {"description": "current Percept brain state — armed watches + recent alert goal-ids. "
                            "Call when asked what you are watching or doing.",
             "properties": {}, "required": []},
            _percept_state)

    # The channel-separation contract, appended to every system instruction: state summaries are
    # metadata (never speech), and a watch's `say` text belongs to DELIVERY — the model must never
    # perform it on its own. Without this, the wire recites armed-goal scripts on every noise-turn.
    _GROUNDING_CONTRACT = (
        "\n\nGROUNDING CONTRACT: call `percept_state` when you need to know what is being watched. "
        "State summaries and tool results are machine metadata — never recite or announce them "
        "unprompted. A watch's `say` text is spoken ONLY when the system explicitly tells you a watch "
        "fired; never say it on your own, and never speak unless the user addressed you or the system "
        "instructed you to."
    )

    # -- lifecycle -------------------------------------------------------
    async def start(self) -> None:
        # NO vad_analyzer: Gemini's own auto-VAD drives turns (verified). A Silero gate here silenced the
        # mic in the first live attempt. video_in for the wire's reactive eyes + the watch tee.
        params = TransportParams(audio_in_enabled=True, audio_out_enabled=True, video_in_enabled=True,
                                 audio_in_sample_rate=_IN_RATE, audio_out_sample_rate=_OUT_RATE)
        transport = SmallWebRTCTransport(webrtc_connection=self._conn, params=params)

        self._ensure_state_tool()   # grounding = pull: the model fetches state; nothing is pushed
        fn_schemas = [FunctionSchema(name=n, description=str(s.get("description") or f"call {n}"),
                                     properties=s.get("properties", {}), required=s.get("required", []))
                      for n, (s, _h) in self._tools.items()]
        tools = ToolsSchema(standard_tools=fn_schemas) if fn_schemas else None
        # tools live on the CONTEXT only (not on init) — passing them in both places triggers a mid-session
        # _reconnect() in GeminiLiveLLMService ("tools provided both at init time and in context"). Dispatch
        # still works via register_function below.
        llm = GeminiLiveLLMService(api_key=self._api_key, voice_id=self._voice,
                                   system_instruction=self._system + self._GROUNDING_CONTRACT,
                                   model=self._model or DEFAULT_MODEL)
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

        procs: list[Any] = [transport.input()]
        if self._on_audio_frame is not None:   # tap raw mic BEFORE aggregation (the three-way split)
            procs.append(_AudioTap(self._on_audio_frame))
        procs.append(agg.user())
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
    async def speak(self, request: SpeechRequest) -> None:
        """Speak a proactive fire by injecting a turn into the held session (Gemini renders it in-voice)."""
        text = request.text
        if not text or self._task is None:
            return
        instr = (f"Say this to the user now, VERBATIM, exactly, do not paraphrase: {text}"
                 if request.verbatim
                 else f"Proactively tell the user now, briefly and in your own voice: {text}")
        await self._queue(instr, run_llm=True)

    def set_state_summary(self, summary: str) -> None:
        """Ground the wire — PULL, not push: stash the summary for the `percept_state` tool.

        The old seam contract said "sync state without triggering a response", and this backend CANNOT
        honor a silent push: pipecat's GeminiLiveLLMService handles LLMMessagesAppendFrame via
        ``_create_single_response`` → ``send_client_content(turn_complete=True)`` (plus a forced
        realtime input on Gemini 3.x) — ``run_llm=False`` is ignored and every push is answered OUT
        LOUD. Worse, the pushed state lands as a user-role turn, so the model reads machine state as
        human speech and recites it back on every later noise-turn (the "keeps talking about the
        armed goals" bug). So nothing is queued: the summary is stashed and the model fetches it via
        the self-registered ``percept_state`` tool when it actually needs grounding."""
        self._state_summary = summary or ""

    async def cancel_output(self) -> None:
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
        """Invoke a callback slot — a single callable OR a subscriber list (fan-out, registration
        order). One subscriber raising must not silence the others."""
        if cb is None:
            return
        cbs = cb if isinstance(cb, (list, tuple)) else (cb,)
        for c in list(cbs):
            try:
                res = c(*args)
                if inspect.isawaitable(res):
                    await res
            except Exception:  # noqa: BLE001 - a bad subscriber must never break its peers
                pass
