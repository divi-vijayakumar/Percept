#!/usr/bin/env python3
"""Percept — live watch demo.

Point it at a webcam (or a video file / RTSP URL), give it goals in plain English,
and it prints a notification the moment each one becomes true — once per episode.

    # webcam, two goals
    percept-watch \
        --goal "someone raises their hand" \
        --goal "someone is holding a phone"

    # a video file
    percept-watch --source path/to/clip.mp4 \
        --goal "someone is drinking from a cup"

Needs an OpenRouter key (cheap, fast Qwen judge):  export OPENROUTER_API_KEY=sk-or-...
(or use --vision gemini / claude with that provider's key.)

This bakes in the H0-winning defaults so it just works:
  * block_max localized salience  — catches low-motion actions the global gate misses
  * a per-episode time-refractory  — one notification per event, not a stream of duplicates

TRACING (card Z5). ``--trace out.json`` records the run's per-frame decision trace — the artifact
``percept-replay`` consumes and ``percept-doctor --visualize-from`` renders. The manifest carries the
Gate's knobs (``percept.replay.gate_config_of``), which is what makes the replay a replay of THIS run
rather than of today's library defaults::

    percept-watch --offline --trace t.json     # keyless: replays the bundled fixture
    percept-replay t.json                      # -> 0 divergences, exit 0

``--offline`` is the keyless path: no API key, no camera, a bundled deterministic fixture
(``percept.offline``). It proves the WIRING end to end — it is a replay of a human-authored file,
never a perception result.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path


_DEFAULT_MODEL = {  # a sensible default model per provider (the demo forces these unless --model is given)
    "openrouter": "qwen/qwen3-vl-32b-instruct",
    "gemini": "gemini-3.1-flash-lite",
    "claude": "claude-sonnet-5",
}


def _load_dotenv() -> None:
    """Best-effort: pull only API KEYS from a repo .env — never the demo-controlled PERCEPT_* config
    (a stale PERCEPT_VISION_MODEL in .env would otherwise hijack the chosen provider)."""
    for env in (Path.cwd() / ".env", Path(__file__).resolve().parents[1] / ".env"):
        if env.exists():
            for line in env.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = (s.strip() for s in line.split("=", 1))
                    if k.endswith("_API_KEY") or k.endswith("_KEY"):
                        os.environ.setdefault(k, v)


# --- the H0-winning defaults (set BEFORE importing percept so settings pick them up) ---
def _apply_good_defaults(refractory_s: float, vision: str, model: str | None, egomotion: bool) -> None:
    os.environ["PERCEPT_SALIENCE_STAT"] = "block_max"        # localized recall (the eye)
    os.environ.setdefault("PERCEPT_BENCH_BLOCKMAX_THRESHOLD", "0.20")
    os.environ["PERCEPT_GATE_REFRACTORY_S"] = str(refractory_s)   # one notification per episode
    # EGOMOTION (glasses / body-cam / any MOVING camera): subtract the global head/camera shift before
    # measuring motion. Without this, a moving camera makes the whole frame "move" every tick and the gate
    # escalates on ~every frame (measured 98% on real first-person footage; ego-comp drops it to ~30%).
    os.environ["PERCEPT_EGO_COMPENSATE"] = "true" if egomotion else "false"
    os.environ["PERCEPT_VISION_BACKEND"] = vision                 # force the chosen provider…
    os.environ["PERCEPT_VISION_MODEL"] = model or _DEFAULT_MODEL.get(vision, "")  # …and its model
    # Quiet the "scripted backends" banner: it's about the (unused) voice/conductor seam — the vision
    # judge here is REAL, and goals are code-armed, so the banner is just noise for this demo.
    os.environ.setdefault("PERCEPT_NO_BANNER", "1")


def _recorder(core, goals, *, source: str, driver: str):
    """Build the TraceRecorder for ``--trace``: full retention + the Gate's knobs in the manifest.

    The gate_config is the load-bearing part. ``percept.replay`` rebuilds a Gate from EXACTLY these
    numbers; a trace without them replays against library defaults and says so
    (``ReplayReport.gate_config_recorded is False``) — honest, but not a replay of this run.
    """
    from percept.replay import gate_config_of
    from percept.tracing import RunTraceManifest, TracePolicy, TraceRecorder

    manifest = RunTraceManifest.create(
        source=source,
        driver=driver,
        goals=goals,
        policy=TracePolicy(level="full", max_duration_s=None),
        gate_config=gate_config_of(core.gate),
    )
    return TraceRecorder(manifest)


def _write_trace(rec, path: str) -> None:
    rec.close()
    written = rec.write_json(path)
    lost = rec.lost_records
    print(f"\n  trace: {written}"
          + (f"   ({lost} record(s) LOST to a retention bound)" if lost else "")
          + f"\n  replay it:  percept-replay {written}")


async def _run_offline(args) -> int:
    """The keyless path: replay the bundled fixture through the frozen watch() seam.

    No API key, no camera, no network — and no pretence that it is perception. It exists so the
    ``--trace`` -> ``percept-replay`` round trip is reachable by anyone who ran ``pip install``.
    """
    # NB: deliberately no `logging.disable()` here. The live demo path quiets the whole root logger
    # for cosmetics; doing that offline would be a global, process-wide side effect from a function
    # that is also the acceptance harness for the trace round trip.
    from percept.offline import load_fixture, offline_core, offline_frames, offline_goals
    from percept.watch import watch

    fixture = load_fixture(args.fixture)
    core = offline_core(fixture)
    goals = offline_goals(fixture)
    rec = _recorder(core, goals, source="fixture", driver="watch") if args.trace else None
    if rec is not None:
        core.on_trace = rec.tracer()

    print("─" * 68)
    print(f"  PERCEPT · OFFLINE — replaying bundled fixture {fixture.fixture_id!r}")
    print("  (scripted verdicts from a human-authored file — proves WIRING, not intelligence)")
    for g in goals:
        print(f"    • {g.condition}")
    print("─" * 68)

    fired = 0
    async for ev in watch(offline_frames(fixture), goals, percept=core, source="fixture"):
        fired += 1
        print(f"  🔔  [{ev.t:6.1f}]  {ev.text or ev.goal_id}"
              f"   ·   goal={ev.goal_id}   ·   {ev.action}")
    print(f"\n  {fired} notification(s).")
    if rec is not None:
        _write_trace(rec, args.trace)
    return 0


async def _run(args) -> int:
    import logging
    logging.disable(logging.WARNING)  # clean demo: gate out info/warning globally (only ERROR+ shows)
    import percept
    from percept import Goal

    goals = [
        Goal(id=f"g{i}", condition=c, say=(args.say[i] if i < len(args.say) else c.capitalize()))
        for i, c in enumerate(args.goal)
    ]
    ego = os.environ.get("PERCEPT_EGO_COMPENSATE") == "true"
    print("─" * 68)
    print(f"  PERCEPT · watching {args.source}" + ("   [EGOMOTION: head-motion compensated]" if ego else ""))
    print(f"  judge: {os.environ.get('PERCEPT_VISION_MODEL')}  ·  eye: block_max"
          f"{' + ego-comp' if ego else ''}  ·  refractory: {args.refractory}s")
    for g in goals:
        print(f"    • {g.condition}")
    print("─" * 68)
    print("  (waiting for the first rising edge — Ctrl-C to stop)\n")

    # --trace: build the core HERE (run() would build the identical one internally) so the manifest
    # can carry the live Gate's knobs. Without a core in hand there is nothing to read them off.
    rec = None
    if args.trace:
        from percept import Percept

        core = Percept.create(vision=args.vision)
        for g in goals:
            core.add_goal(g)
        rec = _recorder(core, goals, source="camera", driver="run")
        stream = percept.run(core, source=args.source, fps=args.fps,
                             on_trace=rec.tracer(), on_health=rec.health_sink())
    else:
        stream = percept.run(source=args.source, goals=goals, vision=args.vision, fps=args.fps)

    fired = 0
    t0 = time.monotonic()
    try:
        async for inc in stream:
            fired += 1
            gid = getattr(inc, "concern_id", getattr(inc, "goal_id", "?"))
            msg = getattr(inc, "message", None) or getattr(inc, "say", "") or gid
            conf = getattr(getattr(inc, "verdict", None), "confidence", None)
            elapsed = time.monotonic() - t0
            tag = f" ({conf:.0%})" if isinstance(conf, (int, float)) else ""
            eid = getattr(inc, "entity_id", None)  # D4: which subject fired (None = not entity-scoped)
            ent = f"   ·   entity={eid}" if eid else ""
            print(f"  🔔  [{elapsed:6.1f}s]  {msg}{tag}   ·   goal={gid}{ent}")
    except KeyboardInterrupt:
        print("\n  stopped.")
    finally:
        # A Ctrl-C'd run is exactly the run you want the trace of. Write it either way.
        if rec is not None:
            _write_trace(rec, args.trace)
    print(f"\n  {fired} notification(s).")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Percept live watch demo")
    p.add_argument("--source", default="webcam:0", help="webcam:0 | a video path | rtsp://…  (default webcam:0)")
    p.add_argument("--goal", action="append", default=[], help="a plain-English condition (repeatable)")
    p.add_argument("--say", action="append", default=[], help="optional friendly message per goal (in order)")
    p.add_argument("--vision", default="openrouter", help="openrouter (qwen) | gemini | claude  (default openrouter)")
    p.add_argument("--model", default=None, help="override the judge model id (else a good per-provider default)")
    p.add_argument("--fps", type=float, default=1.0, help="frames per second to pull (default 1.0)")
    p.add_argument("--refractory", type=float, default=15.0, help="seconds to suppress repeat notifications (default 15)")
    p.add_argument("--egomotion", action="store_true",
                   help="MOVING camera (glasses / body-cam): subtract head motion so the gate doesn't escalate every frame")
    p.add_argument("--trace", default=None, metavar="PATH",
                   help="record this run's decision trace to PATH (JSON). Replay it with "
                        "`percept-replay PATH`; render it with `percept-doctor --visualize-from PATH`.")
    p.add_argument("--offline", action="store_true",
                   help="keyless: replay the bundled deterministic fixture instead of a live source "
                        "(no API key, no camera). Proves WIRING, not intelligence.")
    p.add_argument("--fixture", default=None,
                   help="which bundled fixture --offline replays (default: kettle-boils)")
    args = p.parse_args(argv)

    if args.offline:
        # Offline owns its own wiring: no provider env forcing, no credential check, no camera.
        try:
            return asyncio.run(_run_offline(args))
        except KeyboardInterrupt:
            return 0

    if not args.goal:
        args.goal = ["someone raises their hand", "someone is holding a phone"]
        print("  (no --goal given; using demo goals)\n")

    _load_dotenv()
    _apply_good_defaults(args.refractory, args.vision, args.model, args.egomotion)

    if args.vision == "openrouter" and not os.environ.get("OPENROUTER_API_KEY"):
        print("  ✗ set OPENROUTER_API_KEY (export OPENROUTER_API_KEY=sk-or-…) or pass --vision gemini/claude", file=sys.stderr)
        return 2
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
