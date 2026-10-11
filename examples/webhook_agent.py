"""The integration recipe: watch a camera, POST each decision to a webhook, ack only on success.

Two conditions show the two kinds: an *event* ("a person falls down": reported once, then rearmed)
and a *state* ("the door is left open": ``met`` when it becomes true, ``cleared`` when it stops).
Decisions are read with ``ack="manual"`` and acknowledged only after the webhook answered 2xx, so a
failed POST is never lost; a background task logs ``monitor.health()``.

Live: ``WEBHOOK_URL=https://... python examples/webhook_agent.py --camera rtsp://<camera>`` with
``pip install "percept-vision[video,gemini]"`` and ``GEMINI_API_KEY`` set. Offline (no camera, key or
network): ``python examples/webhook_agent.py --offline``. It pushes a drawn scene, answers with
``FakeJudge``, posts to a local webhook (whose first answer is a 503, to show the retry) and checks
that every decision arrived.
"""

import argparse, asyncio, dataclasses, json, logging, os, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
import percept
from percept import Condition, MemoryStore, Monitor, PushSource, SqliteStore, every
from percept.judges import FakeJudge

log = logging.getLogger("webhook_agent")
FALL = Condition("person_fell", "a person falls down", kind="event")     # verified, reported once
DOOR = Condition("door_open", "the door is left open", emit_cleared=True)  # held; met, then cleared


def post(url: str, body: bytes) -> int:
    """One blocking POST; raises on a non-2xx answer or a network error."""
    request = urllib.request.Request(url, body, {"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status


async def deliver(url: str, decision: percept.Decision, tries: int = 5) -> bool:
    """POST ``decision.to_dict()`` with exponential backoff. Percept does not retry for you: an
    unacknowledged decision is only redelivered by the next ``watch()`` or after a restart."""
    body = json.dumps(decision.to_dict()).encode()
    for attempt in range(tries):
        try:
            if 200 <= await asyncio.to_thread(post, url, body) < 300:    # never block the event loop
                return True
        except OSError as exc:                       # urllib's HTTPError and URLError are OSErrors
            log.warning("POST %s failed (%s); retrying", decision.decision_id, exc)
        await asyncio.sleep(min(30.0, 0.25 * 2 ** attempt))
    return False


async def log_health(monitor: Monitor) -> None:
    """health() is an async iterator over a bounded ring; it ends when the monitor closes."""
    async for h in monitor.health():
        log.warning("health %s on %s: %s (x%d)", h.kind, h.stream_id, h.detail, h.count)


async def run(monitor: Monitor, source, url: str) -> None:
    health = asyncio.create_task(log_health(monitor))    # start it alongside watch()
    async with monitor:
        async for delivery in monitor.watch(source, ack="manual"):
            d = delivery.decision
            if await deliver(url, d):
                await delivery.ack()           # only now does it leave the outbox
                print(f"{d.type:<8} {d.condition_id:<12} attempt={delivery.attempt}  {d.reason}")
            else:
                # Left unacknowledged: kept in the store, delivered again by the next watch() or
                # after a restart, with the same decision_id. Past max_unacked (100) unacked
                # decisions, delivery for this stream pauses (health store_lag).
                log.error("gave up on %s for now", d.decision_id)
    await health


# ----------------------------------------------------------------- offline demo


def scene(t: float) -> np.ndarray:
    """A drawn room: the door opens at 1 s and closes at 4 s; a person walks in at 1.5 s and
    falls at 2.5 s."""
    y, x = np.mgrid[0:120, 0:160]
    frame = np.dstack([((x * 7 + y * 13) % 200 + 20).astype(np.uint8)] * 3)
    frame[20:110, 120:150] = (60, 220, 240) if 1.0 <= t < 4.0 else (30, 60, 90)   # door: lit gap or shut (BGR)
    if 1.5 <= t < 2.5:
        frame[40:110, 40:55] = (200, 120, 40)     # a person standing, in blue
    elif t >= 2.5:
        frame[80:112, 10:110] = (200, 120, 40)    # the same person lying down
    return frame


def judge_script(request):
    """Stands in for a vision model: it looks at the same images a real judge would get."""
    answers = {}
    for item in request.items:
        if item.condition.id == "door_open":      # a state: judged on the newest frame
            img = request.images[-1]
            lit = ((img[..., 1] > 200) & (img[..., 2] > 200) & (img[..., 0] < 100)).any()
            answers["door_open"] = ("yes", 0.93, "the door stands open") if lit else ("no", 0.9, "the door is shut")
        else:                                     # an event: judged on a short clip
            lying = False
            for img in request.images:
                ys, xs = np.nonzero((img[..., 0] > 180) & (img[..., 2] < 80))
                lying |= len(xs) > 0 and np.ptp(xs) > np.ptp(ys)
            answers["person_fell"] = ("yes", 0.9, "a person is lying on the floor") if lying \
                else ("no", 0.9, "nobody is on the floor")
    return answers


def local_webhook() -> tuple[str, list]:
    """A stdlib HTTP server standing in for your webhook. Its first answer is a 503."""
    received: list = []
    calls = []

    class Hook(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(body["decision_id"])
            if len(calls) > 1:                    # the first call fails, as a real endpoint may
                received.append(body)
            self.send_response(200 if len(calls) > 1 else 503)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Hook)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_port}/decisions", received


async def offline() -> None:
    url, received = local_webhook()
    # Demo settings, not production ones: the door is looked at every 0.5 s (by default a state is
    # looked at on a scene change plus every 300 s) and the budget allows 600 attempts a minute
    # (the default, 6 a stream, is about 5 requests a minute when no request retries).
    # MemoryStore logs once that it is not crash-safe; that warning is expected here.
    door = dataclasses.replace(DOOR, trigger=every(0.5))
    monitor = Monitor(FakeJudge(judge_script), [FALL, door], store=MemoryStore(), max_requests_per_min=600)
    camera = PushSource("hallway")

    async def play() -> None:                     # 7 s at 10 frames per second, then the clip ends
        start = time.monotonic()
        while (t := time.monotonic() - start) < 7.0:
            camera.push(scene(t))
            await asyncio.sleep(0.1)
        camera.close()

    player = asyncio.create_task(play())
    await run(monitor, camera, url)
    await player
    got = [(d["type"], d["condition_id"]) for d in received]
    assert got.count(("met", "person_fell")) == 1, got
    assert ("met", "door_open") in got and ("cleared", "door_open") in got, got
    print(f"webhook received {len(received)} decisions, each acknowledged after a 2xx")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--offline", action="store_true", help="run on a drawn scene with FakeJudge")
    parser.add_argument("--camera", help="a camera URI such as rtsp://...")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.offline:
        asyncio.run(offline())
    else:   # production: a real judge (needs the gemini extra), the durable default store
        if not args.camera or "WEBHOOK_URL" not in os.environ:
            parser.error("live mode needs --camera and WEBHOOK_URL (or run with --offline)")
        judge = percept.judges.gemini("gemini-2.5-flash")   # reads GEMINI_API_KEY
        monitor = Monitor(judge, [FALL, DOOR], store=SqliteStore("percept.db"))
        asyncio.run(run(monitor, percept.open_source(args.camera, stream_id="hallway"), os.environ["WEBHOOK_URL"]))
