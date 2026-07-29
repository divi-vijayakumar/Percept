"""percept.realtime — MOVED (v0.6 SDK voice diet).

The held speech-to-speech voice wire (Pipecat + serverless P2P WebRTC driving Gemini Live) and its
signaling boilerplate no longer ship inside the SDK. The perception package must not own the voice stack.
The voice embodiment now lives app-side: wire a ``VoiceEngine`` to Percept through the SDK's public
seams (``RuntimeSession`` / ``percept.start``), arbitrating delivery with a ``PerceptBridge``/``Floor``
pattern of your own. See ``docs/sources/voice-and-watchdog.md`` for the seam and a worked sketch.

The vendor-neutral ``VoiceTransport``/``UtteranceEvent``/``Reply`` contract remains (deprecated) in
``percept.contracts`` for this release. Importing ``percept.realtime`` raises so the move is loud, not a
silent missing module.
"""

raise ImportError(
    "percept.realtime moved app-side (v0.6 SDK voice diet) — wire a voice engine app-side via "
    "RuntimeSession/percept.start, arbitrating with your own PerceptBridge/Floor pattern. See "
    "docs/sources/voice-and-watchdog.md. The VoiceTransport contract lives (deprecated) in "
    "percept.contracts."
)
