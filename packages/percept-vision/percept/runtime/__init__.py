"""percept.runtime — the live worker that owns the camera→incidents lifecycle.

    from percept.runtime import run
    async for incident in run(core, source="rtsp://cam/stream", on_incident=notify):
        ...

The worker code is pure stdlib (ships with `percept-vision`); the camera source needs the `[runtime]`
extra (`pip install "percept-vision[runtime]"` → opencv-python-headless). A pre-decoded/iterable source
works with no extra (tests, offline replay).
"""
from percept.events import AudioBoundary, AudioPushResult, AudioUnit, AudioWindow
from percept.runtime._vad import EnergyVad, endpoint_energy_sequence, frame_energy
from percept.runtime.envelope import (
    SOURCE_ENVELOPE_VERSION,
    SourceCoverage,
    SourceCoverageEffect,
    SourceEnvelope,
    SourceFormat,
    SourceInspection,
    SourceTracker,
)
from percept.runtime.audio import pcm_energy
from percept.interest import (
    InterestBackpressureError,
    InterestCallbackUnsupportedError,
    InterestRunError,
    InterestSubscription,
    LiveInterestAdmission,
    LiveInterestEvent,
)
from percept.runtime.session import RuntimeSession, start
from percept.runtime.sinks import BoundedDeadLetterQueue, FileDeadLetterQueue, WebhookSink, incident_payload, webhook
from percept.runtime.worker import (
    EffectiveConfig,
    EffectiveFps,
    EffectiveTimebase,
    RuntimeHealthEvent,
    RuntimeStats,
    run,
)

__all__ = [
    "run",
    "RuntimeStats",
    "RuntimeHealthEvent",
    "EffectiveConfig",
    "EffectiveFps",
    "EffectiveTimebase",
    "RuntimeSession",
    "start",
    "EnergyVad",
    "AudioBoundary",
    "AudioUnit",
    "AudioPushResult",
    "AudioWindow",
    "frame_energy",
    "pcm_energy",
    "endpoint_energy_sequence",
    "SOURCE_ENVELOPE_VERSION",
    "SourceCoverage",
    "SourceCoverageEffect",
    "SourceEnvelope",
    "SourceFormat",
    "SourceInspection",
    "SourceTracker",
    "LiveInterestAdmission",
    "LiveInterestEvent",
    "InterestSubscription",
    "InterestBackpressureError",
    "InterestCallbackUnsupportedError",
    "InterestRunError",
    "WebhookSink",
    "FileDeadLetterQueue",
    "BoundedDeadLetterQueue",
    "incident_payload",
    "webhook",
]
