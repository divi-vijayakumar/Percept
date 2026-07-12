"""percept.realtime — a held speech-to-speech voice wire for the VoiceTransport seam.

    from percept.realtime import PipecatRealtimeTransport, serve_realtime

Gemini Live over Pipecat serverless P2P WebRTC (browser AEC, no key), grounded by the brain via the
extended VoiceTransport seam. NOT a framework — you keep ``percept.run``/``attach_transport`` visible;
this ships the transport + the signaling boilerplate. Needs ``pip install "percept-vision[realtime]"``.
See ``scratch/watch_demo/REALTIME-V2V-DESIGN.md`` and ``examples/realtime_watch_agent.py``.
"""
from percept.realtime.server import build_realtime_app, serve_realtime
from percept.realtime.transport import PipecatRealtimeTransport

__all__ = ["PipecatRealtimeTransport", "serve_realtime", "build_realtime_app"]
