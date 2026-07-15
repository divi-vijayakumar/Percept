"""percept.realtime — MOVED (v0.6 SDK voice diet).

The held speech-to-speech voice wire (Pipecat + serverless P2P WebRTC driving Gemini Live) and its
signaling boilerplate no longer ship inside the SDK. The perception package must not own the voice stack.
The full working embodiment now lives, app-side, in ``templates/realtime_voice_watch_agent`` — copy that
template (``cd templates/realtime_voice_watch_agent && pip install -e .``) and wire a ``VoiceEngine`` to
Percept through its ``PerceptBridge``/``Floor`` and the SDK's public seams.

The vendor-neutral ``VoiceTransport``/``UtteranceEvent``/``Reply`` contract remains (deprecated) in
``percept.contracts`` for this release. Importing ``percept.realtime`` raises so the move is loud, not a
silent missing module.
"""

raise ImportError(
    "percept.realtime moved to templates/realtime_voice_watch_agent (v0.6 SDK voice diet) — copy the "
    "template and wire a VoiceEngine via its PerceptBridge/Floor. The VoiceTransport contract lives "
    "(deprecated) in percept.contracts."
)
