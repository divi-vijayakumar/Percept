"""Deepgram STT + TTS backends (L2/L3-only).

Streaming speech-to-text (:class:`DeepgramSTT`, yields :class:`~percept.adapters.STTEvent`s)
and text-to-speech (:class:`DeepgramTTS`, ``push`` / ``flush``) behind the frozen seams. Like
all real backends these are exercised only in L2/L3 — L1 uses ``FakeSTT`` / ``FakeTTS``. The
Deepgram SDK is **lazy-imported inside methods** so the core never imports a vendor SDK at
module top level (invariant 7); a missing SDK or key raises a clear error only on first real
use.
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator

from ..adapters import STTEvent


class DeepgramSTT:
    """Streaming STT: async-iterate :class:`STTEvent`s off an audio stream.

    ``audio`` is an async byte-chunk iterator (mic frames). Interim results are emitted with
    ``is_final=False`` and finalized ones with ``is_final=True`` — the agent endpoints on the
    finals (``on_speech_endpoint``).
    """

    def __init__(self, settings: Any = None) -> None:
        self._settings = settings
        self._api_key = os.environ.get("DEEPGRAM_API_KEY")
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                from deepgram import DeepgramClient  # lazy: vendor SDK
            except ImportError as exc:  # pragma: no cover — L2/L3 only
                raise RuntimeError(
                    "DeepgramSTT needs the 'deepgram-sdk' package (optional extra). "
                    "Install it or use the 'fake' stt for offline runs."
                ) from exc
            if not self._api_key:
                raise RuntimeError(
                    "DeepgramSTT needs DEEPGRAM_API_KEY in the environment."
                )
            self._client = DeepgramClient(self._api_key)
        return self._client

    async def stream(self, audio: Any) -> AsyncIterator[STTEvent]:  # pragma: no cover — L2/L3
        import asyncio
        import time

        client = self._ensure_client()
        connection = client.listen.asyncwebsocket.v("1")

        queue: asyncio.Queue[STTEvent | None] = asyncio.Queue()

        async def _on_transcript(_self: Any, result: Any, **_: Any) -> None:
            alt = result.channel.alternatives[0]
            text = alt.transcript
            if not text:
                return
            await queue.put(
                STTEvent(text=text, is_final=bool(result.is_final), ts=time.monotonic())
            )

        from deepgram import LiveTranscriptionEvents, LiveOptions

        connection.on(LiveTranscriptionEvents.Transcript, _on_transcript)
        await connection.start(LiveOptions(model="nova-2", interim_results=True, smart_format=True))

        async def _pump() -> None:
            async for chunk in audio:
                await connection.send(chunk)
            await connection.finish()
            await queue.put(None)

        pump = asyncio.create_task(_pump())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
        finally:
            pump.cancel()


class DeepgramTTS:
    """Streaming TTS: ``push`` enqueues text, ``flush`` synthesizes + plays the buffered turn.

    Buffering between ``push`` calls lets the specialist stream tokens in while audio drains;
    ``flush`` marks the end of a turn (and clears the buffer for the next one).
    """

    def __init__(self, settings: Any = None) -> None:
        self._settings = settings
        self._api_key = os.environ.get("DEEPGRAM_API_KEY")
        self._client: Any = None
        self._buffer: list[str] = []

    def _ensure_client(self) -> Any:  # pragma: no cover — L2/L3 only
        if self._client is None:
            try:
                from deepgram import DeepgramClient  # lazy: vendor SDK
            except ImportError as exc:
                raise RuntimeError(
                    "DeepgramTTS needs the 'deepgram-sdk' package (optional extra). "
                    "Install it or use the 'fake' tts for offline runs."
                ) from exc
            if not self._api_key:
                raise RuntimeError("DeepgramTTS needs DEEPGRAM_API_KEY in the environment.")
            self._client = DeepgramClient(self._api_key)
        return self._client

    def push(self, text: str) -> None:
        """Buffer a chunk of reply text for the current turn."""
        if text:
            self._buffer.append(text)

    def flush(self) -> None:  # pragma: no cover — L2/L3 only
        """Synthesize + emit the buffered turn, then clear the buffer."""
        if not self._buffer:
            return
        text = "".join(self._buffer)
        self._buffer.clear()
        client = self._ensure_client()
        client.speak.rest.v("1").stream({"text": text}, {"model": "aura-asteria-en"})


__all__ = ["DeepgramSTT", "DeepgramTTS"]
