"""FakeVoiceEngine — the deterministic, vendor-free :class:`engine_adapter.VoiceEngine` test double.

It owns no real audio; a test **drives** it (emit_* / call_tool) and it **records** what the app sent
back (``spoken`` SpeechRequests, ``cancels``, registered ``tools``), so the floor/bridge/app wiring is
fully assertable offline. Proves WIRING, not model intelligence.
"""
from __future__ import annotations

import inspect
from typing import Any

from engine_adapter import SpeechRequest


class FakeVoiceEngine:
    def __init__(self, *, speaks_for_itself: bool = True) -> None:
        self.speaks_for_itself = speaks_for_itself
        self.started = False
        self.stopped = False
        self.spoken: list[SpeechRequest] = []
        self.cancels = 0
        self.tools: dict[str, tuple[dict, Any]] = {}
        self.state_summaries: list[str] = []
        self._on_transcript: list[Any] = []
        self._on_output_started: list[Any] = []
        self._on_output_stopped: list[Any] = []
        self._on_user_speaking: list[Any] = []

    # -- VoiceEngine surface ------------------------------------------------
    async def speak(self, request: SpeechRequest) -> None:
        self.spoken.append(request)

    async def cancel_output(self) -> None:
        self.cancels += 1

    def on_user_transcript(self, callback: Any):
        return self._subscribe(self._on_transcript, callback)

    def on_output_started(self, callback: Any):
        return self._subscribe(self._on_output_started, callback)

    def on_output_stopped(self, callback: Any):
        return self._subscribe(self._on_output_stopped, callback)

    def on_user_speaking(self, callback: Any):
        return self._subscribe(self._on_user_speaking, callback)

    def register_tool(self, name: str, schema: dict, handler: Any) -> None:
        self.tools[name] = (schema, handler)

    def set_state_summary(self, summary: str) -> None:
        self.state_summaries.append(summary)

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    # -- test drivers ------------------------------------------------------------
    async def emit_transcript(self, text: str) -> None:
        await self._fire(self._on_transcript, text)

    async def emit_output_started(self) -> None:
        await self._fire(self._on_output_started)

    async def emit_output_stopped(self) -> None:
        await self._fire(self._on_output_stopped)

    async def emit_user_speaking(self, speaking: bool) -> None:
        await self._fire(self._on_user_speaking, speaking)

    async def call_tool(self, name: str, args: dict) -> Any:
        _schema, handler = self.tools[name]
        res = handler(args)
        return await res if inspect.isawaitable(res) else res

    # -- internals --------------------------------------------------------------------
    @staticmethod
    def _subscribe(subscribers: list, callback: Any):
        subscribers.append(callback)

        def _unsub() -> None:
            try:
                subscribers.remove(callback)
            except ValueError:
                pass

        return _unsub

    @staticmethod
    async def _fire(subscribers: list, *args: Any) -> None:
        for cb in list(subscribers):
            res = cb(*args)
            if inspect.isawaitable(res):
                await res
