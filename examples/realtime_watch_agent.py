"""A realtime voice watch agent — Percept is the brain; Gemini Live is a voice wire.

    pip install "percept-vision[realtime]"
    export GEMINI_API_KEY=...
    python examples/realtime_watch_agent.py     # → open http://127.0.0.1:8930, allow camera + mic

You talk (Gemini Live, realtime, browser AEC). "Watch for when I drink coffee" → Gemini calls the `watch`
tool → Percept's CONDUCTOR authors the watchspec + arms it. The watch runs (percept.run) and its fires are
spoken back through the same voice. Everything below is EXPLICIT — no framework hides run()/attach_transport.

Note: for a single-machine demo the watch reads the BROWSER's camera (tee'd through the transport). In a
product the watch feed is a PERSISTENT device (glasses/edge cam) so it runs 24/7 independent of the voice
session; then a fire while you're disconnected degrades to the act sink (assistant-class).
"""
import asyncio
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

import percept
from percept.adapters import Goal
from percept.edge.tier0 import Frame
from percept.runtime.session import RuntimeSession
from percept.realtime import PipecatRealtimeTransport, serve_realtime

# --- config -------------------------------------------------------------
for _p in [Path.cwd(), *Path.cwd().parents]:
    if (_p / ".env").exists():
        for _l in (_p / ".env").read_text().splitlines():
            if _l.strip() and "=" in _l and not _l.startswith("#"):
                _k, _, _v = _l.partition("="); os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
        break
KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_AISTUDIO_API_KEY")
MODEL = os.environ.get("PERCEPT_GEMINI_LIVE_MODEL")  # None → transport pins the verified gemini-3.x-flash-live-preview
# Division of labor only — do NOT reimplement the conductor's standing-vs-one-shot classifier here; that
# lives in _CONDUCTOR_SYSTEM and already declines to author for one-off questions. The wire just needs to
# know: it can see (answer live), and anything to track over time goes to `watch` verbatim.
SYSTEM = ("You are Percept, a warm ambient assistant wired to a vision watch agent. You can SEE the live "
          "camera, so answer questions about what's on screen right now YOURSELF, briefly. When the user "
          "wants something tracked or wants to be alerted about it LATER, call the `watch` tool with their "
          "request VERBATIM and let the watch agent decide and run it — it speaks its own alerts. Keep "
          "replies short.")
GATE_W, GATE_H = 48, 36


def _ensure_nltk_punkt():
    """Pipecat splits the bot's text into sentences with NLTK's ``punkt_tab`` tokenizer; a fresh
    ``pip install`` never fetches that data, and a missing resource throws mid-turn — surfacing as a
    bogus Gemini 'Connection error' + reconnection loop. Fetch it once (routing TLS through certifi,
    since macOS' framework Python ships no system CA bundle)."""
    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab/english/")
        return
    except LookupError:
        pass
    try:
        import ssl
        import certifi
        ssl._create_default_https_context = lambda *a, **k: ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001 - best effort; plain download may still work
        pass
    nltk.download("punkt_tab")


def _to_rgb(image_frame):
    try:
        w, h = image_frame.size
        buf = np.frombuffer(image_frame.image, dtype=np.uint8)
        fmt = (image_frame.format or "RGB").upper()
        if fmt.startswith("RGBA") and buf.size >= w * h * 4:
            return buf[:w * h * 4].reshape(h, w, 4)[..., :3]
        if fmt.startswith("RGB") and buf.size >= w * h * 3:
            return buf[:w * h * 3].reshape(h, w, 3)
    except Exception:
        pass
    return None


async def on_connection(connection):
    """Per browser: ONE watch agent + the voice wire, explicitly wired (this is the app, not the SDK)."""
    core = percept.Percept.create(vision="gemini", discover_plugins=False)   # the brain (judge = gemini vision)
    frame_q: asyncio.Queue = asyncio.Queue(maxsize=8)
    session = RuntimeSession(core)

    tee = {"calls": 0, "rgb_none": 0, "put": 0, "full": 0}

    def on_video_frame(image_frame):   # tee browser frames → the watch (single-machine demo)
        tee["calls"] += 1
        rgb = _to_rgb(image_frame)
        if rgb is None:
            tee["rgb_none"] += 1
            if tee["rgb_none"] % 30 == 1:
                print(f"[tee] _to_rgb returned None (type={type(image_frame).__name__} "
                      f"fmt={getattr(image_frame, 'format', '?')} size={getattr(image_frame, 'size', '?')}) "
                      f"— frame NOT tee'd", file=sys.stderr, flush=True)
            return
        img = Image.fromarray(rgb, "RGB")
        gray = np.asarray(img.convert("L").resize((GATE_W, GATE_H)))
        gate = Frame(w=GATE_W, h=GATE_H, pixels=tuple(int(p) for p in gray.flatten()))
        bio = io.BytesIO(); img.save(bio, format="JPEG", quality=70)
        try:
            frame_q.put_nowait((time.monotonic(), gate, ("image/jpeg", bio.getvalue())))
            tee["put"] += 1
        except asyncio.QueueFull:
            tee["full"] += 1
        if (tee["put"] + tee["full"]) % 30 == 1:
            print(f"[tee] browser video → watch: calls={tee['calls']} put={tee['put']} "
                  f"queuefull={tee['full']} rgb_none={tee['rgb_none']}", file=sys.stderr, flush=True)

    transport = PipecatRealtimeTransport(connection, gemini_api_key=KEY or "", model=MODEL,
                                         voice="Aoede", system=SYSTEM, on_video_frame=on_video_frame)

    async def source():   # the watch's feed = the tee'd browser frames
        n = 0
        while True:
            item = await frame_q.get()
            if item is None:
                break
            n += 1
            if n % 30 == 1:
                print(f"[source] watch consumed {n} frames from the queue", file=sys.stderr, flush=True)
            yield item

    frame_seen = [0]

    def on_trace(ev):   # the SDK's per-frame decision trace — why each frame stayed silent or fired
        stage = getattr(ev, "stage", "?")
        if stage == "source.frame":
            frame_seen[0] += 1
            if frame_seen[0] % 30:   # throttle the per-frame heartbeat; the judge/gate stages are sparse
                return
            print(f"[trace] source.frame        (feed alive: {frame_seen[0]} frames judged-eligible)",
                  file=sys.stderr, flush=True)
            return
        data = {k: v for k, v in (getattr(ev, "data", {}) or {}).items()}
        line = f"[trace] {stage:18} goal={getattr(ev, 'goal_id', None)!r} {data}"
        print(line[:400], file=sys.stderr, flush=True)

    async def run_watch():
        try:
            async for inc in percept.run(core, source=source(),
                                          on_incident=session.deliver_fire,       # fires → spoken via the wire
                                          on_frame=session._notify_frame,
                                          on_trace=on_trace,                      # full decision trace → stderr
                                          # CRITICAL: goals are armed at RUNTIME by the `watch` voice tool. Without
                                          # this, the gate compiles its spec once at t0 (empty) and never re-arms —
                                          # runtime-authored goals stay inert and NOTHING EVER FIRES. (session.start()
                                          # sets this for you; this hand-rolled run() must pass it explicitly.)
                                          regate_on_goal_change=True):
                print(f"[incident] 🔔 FIRED concern={getattr(inc, 'concern_id', None)!r} "
                      f"t={getattr(inc, 't', None)}", file=sys.stderr, flush=True)
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001
            print(f"[percept] watch ended: {exc!r}", file=sys.stderr)

    async def log_watchspecs():   # surface the AUTHORED watchspec the moment a goal arms (visibility)
        seen: set = set()
        while True:
            for g in list(core.goals.active()):
                if g.id not in seen:
                    seen.add(g.id)
                    print(f"[watchspec] ✅ armed id={g.id!r} condition={getattr(g, 'condition', '')!r} "
                          f"say={getattr(g, 'say', '')!r} state={dict(getattr(g, 'state', {}) or {})}",
                          file=sys.stderr, flush=True)
            await asyncio.sleep(1.0)

    watch_task = asyncio.ensure_future(run_watch())
    asyncio.ensure_future(log_watchspecs())
    await session.attach_transport(transport)   # tools (watch→conductor) + state pulse + fire delivery
    print("[percept] call up: Gemini Live wire attached; watch armed (say 'watch for …' to add goals)")

    @connection.event_handler("closed")
    async def _closed(_c):
        watch_task.cancel()
        await frame_q.put(None)
        await session.stop()


if __name__ == "__main__":
    if not KEY:
        sys.exit("set GEMINI_API_KEY (or GOOGLE_API_KEY)")
    _ensure_nltk_punkt()
    serve_realtime(on_connection, host="127.0.0.1", port=8930)
