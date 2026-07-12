"""serve_realtime — the SDK owns the SmallWebRTC signaling boilerplate; the app owns the cognition.

This is deliberately NOT a framework: it exposes the fiddly WebRTC signaling routes (``/start`` →
``/sessions/{id}/api/offer`` POST+PATCH, mirroring Pipecat's own runner so ICE trickling works) and mounts
the prebuilt browser client — and calls back into YOUR ``on_connection(webrtc_connection)`` per browser,
where ``percept.run``/``attach_transport`` stay visible. The pipeline boot is DEFERRED off the callback so
the WebRTC answer returns immediately (the ICE-stability fix).

Only importable with the ``[realtime]`` extra.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any, Awaitable, Callable

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import RedirectResponse

from pipecat.transports.smallwebrtc.request_handler import (
    IceCandidate,
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)
from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI


def build_realtime_app(on_connection: Callable[[Any], Awaitable[None]]) -> FastAPI:
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


def serve_realtime(on_connection: Callable[[Any], Awaitable[None]], *,
                   host: str = "127.0.0.1", port: int = 8930) -> None:
    """Blocking: serve the browser client + WebRTC signaling; call ``on_connection`` per browser."""
    print(f"percept realtime → http://{host}:{port}  (allow camera + mic)")
    uvicorn.run(build_realtime_app(on_connection), host=host, port=port, log_level="warning")
