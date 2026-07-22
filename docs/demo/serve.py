#!/usr/bin/env python3
"""Percept demo harness — run the REAL runtime behind the two demo pages.

This is an INTERNAL product-test harness (dogfood the wheel before a client wheel ships), not a
shipped API. It wraps the frozen ``watch()``/``run()`` seam — the exact path ``percept.watch_cli
--trace`` uses — so the browser pages render REAL WatchEvents and a REAL ``.ptrace``, never mocked data.

    python docs/demo/serve.py                 # http://localhost:8788  (keyless: bundled fixture, real runtime)
    PERCEPT_VISION_MODEL=... OPENROUTER_API_KEY=...  python docs/demo/serve.py   # real judge over uploads

Endpoints
    GET  /                         -> demo index (links to both pages)
    GET  /demo-01-video-watch.html -> the MP4 upload demo (real client)
    GET  /demo-02-screen-watch.html-> the screen-share demo (real client)
    POST /run?goal=..&backend=..&fps=..&source_name=..   body = raw video bytes
                                   -> runs the REAL runtime over the upload, returns the real trace JSON
    POST /run?mode=fixture         -> runs the REAL runtime over the bundled fixture (keyless, no spend)
    GET  /trace/<run_id>.ptrace    -> download the real .ptrace written for that run

No judge key present -> uploads are refused with a clear message and the fixture path is offered, so the
harness is honest about when it is and is not exercising a real model.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
PORT = int(os.environ.get("PERCEPT_DEMO_PORT", "8788"))
RUNS: dict[str, dict] = {}  # run_id -> {"trace": <path>, "json": <dict>}

# --- make percept-vision importable whether or not it is pip-installed -------------------------
_PKG = HERE.parent.parent / "packages" / "percept-vision"
if _PKG.exists() and str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))


def _load_dotenv() -> None:
    """Best-effort: pull only API KEYS from a repo .env (same rule as percept.watch_cli) — never the
    PERCEPT_* config, so a stale PERCEPT_VISION_MODEL cannot hijack the UI-chosen provider. Values are
    loaded into this process only; they are never printed or sent to the browser."""
    for env in (Path.cwd() / ".env", _PKG / ".env", _PKG.parent.parent / ".env"):
        if env.exists():
            for line in env.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = (s.strip() for s in line.split("=", 1))
                    if k.endswith("_API_KEY") or k.endswith("_KEY"):
                        os.environ.setdefault(k, v)


def _have_key() -> str | None:
    # names this repo's .env actually uses: OPENROUTER_API_KEY, ANTHROPIC_API_KEY, GOOGLE_AISTUDIO_API_KEY
    for k in ("OPENROUTER_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
              "GOOGLE_AISTUDIO_API_KEY", "ANTHROPIC_API_KEY"):
        if os.environ.get(k):
            return k
    return None


_DEFAULT_MODEL = {  # same per-provider defaults percept.watch_cli forces
    "openrouter": "qwen/qwen3-vl-32b-instruct",
    "gemini": "gemini-3.1-flash-lite",
    "claude": "claude-sonnet-5",
}


def _apply_good_defaults() -> None:
    """The H0-winning tier-0 defaults percept.watch_cli sets BEFORE importing percept. Without
    block_max (the localized 'eye') the gate admits almost nothing on real content — the exact silent
    failure a client hit. Set at import, before any percept import in the handlers."""
    os.environ["PERCEPT_SALIENCE_STAT"] = "block_max"
    os.environ.setdefault("PERCEPT_BENCH_BLOCKMAX_THRESHOLD", "0.20")
    os.environ.setdefault("PERCEPT_GATE_REFRACTORY_S", "15")
    os.environ.setdefault("PERCEPT_EGO_COMPENSATE", "false")
    os.environ.setdefault("PERCEPT_NO_BANNER", "1")


_load_dotenv()          # load keys from .env before anything reads them
_apply_good_defaults()  # block_max eye + thresholds before the first `import percept`


def _recorder(core, goals, *, source: str, driver: str):
    """Same recorder the real CLI builds for --trace: full retention + the live Gate's knobs."""
    from percept.replay import gate_config_of
    from percept.tracing import RunTraceManifest, TracePolicy, TraceRecorder

    manifest = RunTraceManifest.create(
        source=source, driver=driver, goals=goals,
        policy=TracePolicy(level="full", max_duration_s=None),
        gate_config=gate_config_of(core.gate),
    )
    return TraceRecorder(manifest)


async def _run_fixture(goal_text: str, trace_path: Path) -> dict:
    """Keyless: replay the bundled fixture through the REAL watch() seam. No spend, no pretence."""
    from percept.offline import load_fixture, offline_core, offline_frames, offline_goals
    from percept.watch import watch

    fixture = load_fixture(None)
    core = offline_core(fixture)
    goals = offline_goals(fixture)
    rec = _recorder(core, goals, source="fixture", driver="watch")
    core.on_trace = rec.tracer()
    events = []
    async for ev in watch(offline_frames(fixture), goals, percept=core, source="fixture"):
        events.append({"t": getattr(ev, "t", None), "goal_id": getattr(ev, "goal_id", None),
                       "action": getattr(ev, "action", None), "text": getattr(ev, "text", None)})
    rec.close()
    rec.write_json(str(trace_path))
    return {"mode": "fixture", "events": events, "note":
            "Bundled fixture through the real runtime — scripted verdicts, proves WIRING not intelligence."}


async def _run_real(video_path: Path, goal_text: str, backend: str, fps: float, trace_path: Path) -> dict:
    """Real judge over the uploaded video — the exact path percept.watch_cli --trace runs."""
    import logging
    logging.disable(logging.WARNING)
    import percept
    from percept import Goal, Percept

    goals = [Goal(id="g0", condition=goal_text, say=goal_text)]
    os.environ["PERCEPT_VISION_BACKEND"] = backend
    os.environ["PERCEPT_VISION_MODEL"] = _DEFAULT_MODEL.get(backend, "")
    core = Percept.create(vision=backend)
    for g in goals:
        core.add_goal(g)
    rec = _recorder(core, goals, source="upload", driver="run")
    events = []
    stream = percept.run(core, source=str(video_path), fps=fps,
                         on_trace=rec.tracer(), on_health=rec.health_sink())
    async for inc in stream:
        events.append({
            "t": None,
            "goal_id": getattr(inc, "concern_id", getattr(inc, "goal_id", None)),
            "action": getattr(getattr(inc, "verdict", None), "satisfied", None) and "fire" or "fire",
            "text": getattr(inc, "message", None) or getattr(inc, "say", None),
            "confidence": getattr(getattr(inc, "verdict", None), "confidence", None),
            "entity_id": getattr(inc, "entity_id", None),
        })
    rec.close()
    rec.write_json(str(trace_path))
    return {"mode": "real", "backend": backend, "events": events}


def _do_run(params: dict, body: bytes) -> dict:
    """Blocking entry: run one real run to completion, return {run_id, manifest, summary, records, events}."""
    run_id = f"run_{int(time.time()*1000)}"
    trace_path = Path(tempfile.gettempdir()) / f"percept-demo-{run_id}.ptrace"
    goal = (params.get("goal", ["the target condition is met"])[0]).strip() or "the target condition is met"
    mode = params.get("mode", [""])[0]

    if mode == "fixture" or not body:
        result = asyncio.run(_run_fixture(goal, trace_path))
    else:
        key = _have_key()
        if not key:
            return {"error": "no_key",
                    "message": "No judge API key is set (OPENROUTER_API_KEY / GEMINI_API_KEY / "
                               "ANTHROPIC_API_KEY). Set one to run a real judge over your video, or use "
                               "the keyless fixture button (real runtime, scripted verdicts)."}
        backend = (params.get("backend", ["openrouter"])[0]).strip() or "openrouter"
        fps = float((params.get("fps", ["1.0"])[0]) or 1.0)
        suffix = Path(params.get("source_name", ["upload.mp4"])[0]).suffix or ".mp4"
        vid = Path(tempfile.gettempdir()) / f"percept-demo-{run_id}{suffix}"
        vid.write_bytes(body)
        try:
            result = asyncio.run(_run_real(vid, goal, backend, fps, trace_path))
        finally:
            try: vid.unlink()
            except OSError: pass

    trace_json = json.loads(trace_path.read_text())
    RUNS[run_id] = {"trace": str(trace_path)}
    return {"run_id": run_id, "goal": goal,
            "manifest": trace_json.get("manifest"), "summary": trace_json.get("summary"),
            "records": trace_json.get("records", []), "events": result.get("events", []),
            "mode": result.get("mode"), "note": result.get("note"), "backend": result.get("backend")}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._send(200, INDEX, "text/html; charset=utf-8")
        if path.startswith("/trace/") and path.endswith(".ptrace"):
            run_id = path[len("/trace/"):-len(".ptrace")]
            run = RUNS.get(run_id)
            if not run:
                return self._send(404, json.dumps({"error": "unknown run"}))
            return self._send(200, Path(run["trace"]).read_bytes(), "application/json")
        # static files from docs/demo/
        fname = path.lstrip("/")
        f = HERE / fname
        if f.suffix in (".html", ".md", ".css", ".js") and f.is_file() and HERE in f.resolve().parents:
            ctype = {"html": "text/html; charset=utf-8", "md": "text/markdown; charset=utf-8",
                     "css": "text/css", "js": "text/javascript"}[f.suffix.lstrip(".")]
            return self._send(200, f.read_bytes(), ctype)
        return self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/run":
            return self._send(404, json.dumps({"error": "not found"}))
        params = parse_qs(urlparse(self.path).query)
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        try:
            result = _do_run(params, body)
            code = 400 if result.get("error") else 200
            return self._send(code, json.dumps(result))
        except Exception as e:
            traceback.print_exc()
            return self._send(500, json.dumps({"error": "run_failed", "message": str(e)}))


INDEX = """<!doctype html><meta charset=utf-8><title>Percept demos</title>
<style>body{font:15px/1.6 -apple-system,system-ui,sans-serif;max-width:640px;margin:60px auto;padding:0 20px;color:#0a0a0a}
h1{font-family:Arial Narrow,sans-serif;letter-spacing:.02em}code{background:#f4f4f0;padding:2px 5px;font-size:13px}
a{display:block;border:1px solid #0a0a0a;padding:14px 16px;margin:12px 0;color:#0a0a0a;text-decoration:none}a:hover{background:#faf7ee}
.k{font:11px ui-monospace,monospace;color:#8a8a8a;margin-top:24px}</style>
<h1>Percept demo harness</h1>
<p>Real <code>watch()</code> runtime behind both pages. Uploads run a real judge when an API key is set;
otherwise the keyless fixture runs the real runtime with scripted verdicts.</p>
<a href="/demo-01-video-watch.html">Demo 01 · MP4 video watch →</a>
<a href="/demo-02-screen-watch.html">Demo 02 · Screen-share watch →</a>
<p class=k>judge key: %s</p>""" % ("SET — real judge available" if _have_key() else
                                   "UNSET — fixture only (set OPENROUTER_API_KEY / GEMINI_API_KEY / ANTHROPIC_API_KEY)")


def main():
    try:
        import percept  # noqa: F401
    except Exception as e:
        print(f"  ! cannot import percept-vision ({e}). Run from the repo, or pip install the package.")
        return 1
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    key = _have_key()
    print("─" * 68)
    print(f"  PERCEPT demo harness  ·  http://localhost:{PORT}")
    print(f"  judge key: {key or 'UNSET (keyless fixture only — real runtime, scripted verdicts)'}")
    print(f"  real judge over uploads: {'ENABLED' if key else 'set an API key to enable'}")
    print("─" * 68)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
