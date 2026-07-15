"""Media plumbing — SmallWebRTC signaling, the video tee, and the THREE-WAY audio split.

The signaling half is deliberately NOT a framework: it exposes the fiddly WebRTC signaling routes
(``/start`` → ``/sessions/{id}/api/offer`` POST+PATCH, mirroring Pipecat's own runner so ICE trickling
works) and mounts the prebuilt browser client — and calls back into YOUR ``on_connection
(webrtc_connection)`` per browser, where the composition root stays visible. The pipeline boot is
DEFERRED off the callback so the WebRTC answer returns immediately (the ICE-stability fix).

Audio topology (three independent signals — the corrected design)::

    AEC'd microphone
        ├── conversation-engine audio          (stays live; engine-native endpointing/barge-in)
        ├── local energy VAD → user_speaking   (STAYS ACTIVE during assistant playback — barge-in detection)
        └── Percept world-audio tee            (suppressed/tagged ONLY this branch during local playout)

    engine output edges → wire_speaking
    renderer state      → fire_playing

Engine speaking-edges give ``wire_speaking`` only — they never detect the user. ``user_speaking``
comes from the local floor VAD below and must never be derived from the gated Percept energy value
(that would make barge-in during playback impossible: the one moment the gate is closed is the one
moment barge-in matters).
"""
from __future__ import annotations

import asyncio
import io
import time
import uuid
from typing import Any, Awaitable, Callable

import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import RedirectResponse
from PIL import Image  # not cv2: opencv + pyav (pipecat) both bundle libavdevice and clash on macOS

from pipecat.transports.smallwebrtc.request_handler import (
    IceCandidate,
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)
from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI

from percept.edge.tier0 import Frame

GATE_W, GATE_H = 48, 36  # tier-0 motion-gate downsample (mirrors percept.runtime.source)


# --------------------------------------------------------------------------------------------------
# Video tee: pipecat camera frames → a Percept iterable source of (t, gate_frame, judge_frame).
# --------------------------------------------------------------------------------------------------
def _to_rgb(image_frame: Any):
    """Pipecat InputImageRawFrame -> RGB ndarray (or None if the pixel format is unhandled)."""
    try:
        w, h = image_frame.size
        fmt = (image_frame.format or "RGB").upper()
        buf = np.frombuffer(image_frame.image, dtype=np.uint8)
        if fmt.startswith("RGBA") and buf.size >= w * h * 4:
            return buf[: w * h * 4].reshape(h, w, 4)[..., :3]
        if fmt.startswith("RGB") and buf.size >= w * h * 3:
            return buf[: w * h * 3].reshape(h, w, 3)
    except Exception:  # noqa: BLE001
        pass
    return None


class VideoTee:
    """Tee inbound browser camera frames to the watch: wire :meth:`on_video_frame` into the engine's
    video tap and hand :meth:`source` to ``percept.start(source=...)``. The watch is a sampler —
    dropping a frame under load is fine (bounded queue, drop-newest)."""

    def __init__(self, *, maxsize: int = 8, clock: Callable[[], float] = time.monotonic) -> None:
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._clock = clock

    def on_video_frame(self, image_frame: Any) -> None:
        rgb = _to_rgb(image_frame)
        if rgb is None:
            return
        img = Image.fromarray(rgb, "RGB")
        gray = np.asarray(img.convert("L").resize((GATE_W, GATE_H)))   # tier-0 motion-gate downsample
        gate_frame = Frame(w=GATE_W, h=GATE_H, pixels=tuple(int(p) for p in gray.flatten()))
        bio = io.BytesIO()
        img.save(bio, format="JPEG", quality=70)                        # judge frame the VLM reads
        item = (self._clock(), gate_frame, ("image/jpeg", bio.getvalue()))
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            pass  # watch is a sampler — dropping a frame under load is fine

    async def source(self):
        """The async-iterable Percept source. Ends when :meth:`close` is called."""
        while True:
            item = await self._queue.get()
            if item is None:
                break
            yield item

    async def close(self) -> None:
        await self._queue.put(None)


# --------------------------------------------------------------------------------------------------
# Three-way audio split (see the module docstring's topology).
# --------------------------------------------------------------------------------------------------
class EnergyVad:
    """A tiny template-local energy endpointer: RMS threshold + hangover → speaking edges.

    Deliberately local (not the SDK's VAD): the floor's ``user_speaking`` is APPLICATION policy and
    must keep working with zero SDK voice surface. Returns the new state on edge, else ``None``."""

    def __init__(self, *, threshold: float = 0.015, hangover_frames: int = 8) -> None:
        self._threshold = threshold
        self._hangover = hangover_frames
        self._quiet = 0
        self.speaking = False

    def push(self, energy: float) -> bool | None:
        if energy >= self._threshold:
            self._quiet = 0
            if not self.speaking:
                self.speaking = True
                return True
            return None
        if self.speaking:
            self._quiet += 1
            if self._quiet >= self._hangover:
                self.speaking = False
                return False
        return None


def _pcm16_energy(data: bytes) -> float:
    """Normalized RMS of a 16-bit little-endian PCM chunk (0.0–1.0)."""
    if not data:
        return 0.0
    samples = np.frombuffer(data, dtype=np.int16)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((samples.astype(np.float64) / 32768.0) ** 2)))


class AudioSplitter:
    """The mic fan-out. Wire :meth:`on_audio_frame` into the engine's ``on_audio_frame`` tap
    (branch 1, the conversation engine, is the pipeline pass-through — the tap never consumes).

    - branch 2: local energy VAD → ``on_user_speaking(bool)`` edges. **STAYS ACTIVE during assistant
      playback** — this is the barge-in detector; gating it would blind the floor exactly when the
      assistant is talking over the user.
    - branch 3: Percept world-audio → ``push_world_audio(chunk, t)`` (→ ``core.push_audio``).
      Suppressed ONLY while ``suppressed()`` is true (local playout: ``wire_speaking or
      fire_playing``) so the browser-AEC residue of the assistant's own voice never becomes
      "world audio" evidence. Never derive ``user_speaking`` from this gated branch.

    Callbacks are late-bound via :meth:`bind` because the floor (their consumer) is constructed after
    the engine (this tap's producer) in the composition root."""

    def __init__(self, *, vad: EnergyVad | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._vad = vad or EnergyVad()
        self._clock = clock
        self._on_user_speaking: Callable[[bool], Any] | None = None
        self._push_world: Callable[[Any, float], Any] | None = None
        self._suppressed: Callable[[], bool] = lambda: False

    def bind(self, *, on_user_speaking: Callable[[bool], Any] | None = None,
             push_world_audio: Callable[[Any, float], Any] | None = None,
             suppressed: Callable[[], bool] | None = None) -> None:
        if on_user_speaking is not None:
            self._on_user_speaking = on_user_speaking
        if push_world_audio is not None:
            self._push_world = push_world_audio
        if suppressed is not None:
            self._suppressed = suppressed

    def on_audio_frame(self, frame: Any) -> None:
        """One AEC'd mic chunk from the engine tap (pipecat ``InputAudioRawFrame``). Never raises."""
        try:
            data = getattr(frame, "audio", None) or b""
            t = self._clock()
            # branch 2 — the LOCAL VAD: unconditional, playback or not (barge-in detection).
            edge = self._vad.push(_pcm16_energy(data))
            if edge is not None and self._on_user_speaking is not None:
                res = self._on_user_speaking(edge)
                if asyncio.iscoroutine(res):
                    asyncio.ensure_future(res)
            # branch 3 — Percept world audio: gated during local playout, and ONLY this branch.
            if self._push_world is not None and not self._suppressed():
                self._push_world(data, t)
        except Exception:  # noqa: BLE001 - a bad tap consumer must never crash the wire
            pass


# --------------------------------------------------------------------------------------------------
# SmallWebRTC signaling (verbatim port of the SDK-era serve_realtime).
# --------------------------------------------------------------------------------------------------
def build_app(on_connection: Callable[[Any], Awaitable[None]]) -> FastAPI:
    """A FastAPI app that runs ``on_connection(webrtc_connection)`` (deferred) per browser connection."""
    app = FastAPI()
    handler = SmallWebRTCRequestHandler()
    sessions: dict[str, Any] = {}

    async def _cb(connection: Any) -> None:
        asyncio.ensure_future(on_connection(connection))   # deferred → answer returns immediately (ICE)

    @app.post("/api/offer")
    async def offer(body: dict) -> Any:
        return await handler.handle_web_request(SmallWebRTCRequest.from_dict(body), _cb)

    @app.patch("/api/offer")
    async def ice(request: SmallWebRTCPatchRequest) -> Any:
        await handler.handle_patch_request(request)
        return {"status": "success"}

    @app.post("/start")
    async def start_agent(request: Request) -> Any:
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            body = {}
        sid = str(uuid.uuid4())
        sessions[sid] = body.get("body", {})
        out: dict[str, Any] = {"sessionId": sid}
        if body.get("enableDefaultIceServers"):
            out["iceConfig"] = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        return out

    @app.api_route("/sessions/{sid}/{path:path}", methods=["GET", "POST", "PATCH"])
    async def proxy(sid: str, path: str, request: Request) -> Any:
        if sid not in sessions:
            return Response(content="unknown session", status_code=404)
        if path.endswith("api/offer"):
            data = await request.json()
            if request.method == "POST":
                req = SmallWebRTCRequest(sdp=data["sdp"], type=data["type"], pc_id=data.get("pc_id"),
                                         restart_pc=data.get("restart_pc"),
                                         request_data=data.get("request_data") or data.get("requestData") or sessions[sid])
                return await handler.handle_web_request(req, _cb)
            if request.method == "PATCH":
                req = SmallWebRTCPatchRequest(pc_id=data["pc_id"],
                                              candidates=[IceCandidate(**c) for c in data.get("candidates", [])])
                await handler.handle_patch_request(req)
                return {"status": "success"}
        return Response(status_code=200)

    @app.get("/client", include_in_schema=False)
    @app.get("/client/", include_in_schema=False)
    async def _client_redirect() -> Any:
        return RedirectResponse(url="/")

    app.mount("/", SmallWebRTCPrebuiltUI, name="client")   # bundle posts to /api/offer relative to "/"
    return app


def serve(on_connection: Callable[[Any], Awaitable[None]], *,
          host: str = "127.0.0.1", port: int = 8930) -> None:
    """Blocking: serve the browser client + WebRTC signaling; call ``on_connection`` per browser."""
    print(f"realtime voice watch agent → http://{host}:{port}  (allow camera + mic)")
    uvicorn.run(build_app(on_connection), host=host, port=port, log_level="warning")
