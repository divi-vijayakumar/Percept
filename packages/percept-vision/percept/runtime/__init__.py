"""percept.runtime — the live worker that owns the camera→incidents lifecycle.

    from percept.runtime import run
    async for incident in run(core, source="rtsp://cam/stream", on_incident=notify):
        ...

The worker code is pure stdlib (ships with `percept-vision`); the camera source needs the `[runtime]`
extra (`pip install "percept-vision[runtime]"` → opencv-python-headless). A pre-decoded/iterable source
works with no extra (tests, offline replay).
"""
from percept.runtime._vad import EnergyVad, endpoint_energy_sequence, frame_energy
from percept.runtime.session import RuntimeSession, start
from percept.runtime.sinks import BoundedDeadLetterQueue, FileDeadLetterQueue, WebhookSink, incident_payload, webhook
from percept.runtime.worker import RuntimeHealthEvent, RuntimeStats, run

__all__ = [
    "run",
    "RuntimeStats",
    "RuntimeHealthEvent",
    "RuntimeSession",
    "start",
    "EnergyVad",
    "frame_energy",
    "endpoint_energy_sequence",
    "WebhookSink",
    "FileDeadLetterQueue",
    "BoundedDeadLetterQueue",
    "incident_payload",
    "webhook",
]
