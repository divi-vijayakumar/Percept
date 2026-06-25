"""percept.runtime — the live worker that owns the camera→incidents lifecycle.

    from percept.runtime import run
    async for incident in run(core, source="rtsp://cam/stream", on_incident=notify):
        ...

The worker code is pure stdlib (ships with `percept-vision`); the camera source needs the `[runtime]`
extra (`pip install "percept-vision[runtime]"` → opencv-python-headless). A pre-decoded/iterable source
works with no extra (tests, offline replay).
"""
from percept.runtime.worker import run

__all__ = ["run"]
