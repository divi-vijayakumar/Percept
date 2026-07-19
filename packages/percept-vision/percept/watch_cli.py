#!/usr/bin/env python3
"""Percept — live watch demo (hackathon-ready).

Point it at a webcam (or a video file / RTSP URL), give it goals in plain English,
and it prints a notification the moment each one becomes true — once per episode.

    # webcam, two goals
    python examples/hackathon_watch.py \
        --goal "someone raises their hand" \
        --goal "someone is holding a phone"

    # a video file
    python examples/hackathon_watch.py --source path/to/clip.mp4 \
        --goal "someone is drinking from a cup"

Needs an OpenRouter key (cheap, fast Qwen judge):  export OPENROUTER_API_KEY=sk-or-...
(or use --vision gemini / claude with that provider's key.)

This bakes in the H0-winning defaults so it just works:
  * block_max localized salience  — catches low-motion actions the global gate misses
  * a per-episode time-refractory  — one notification per event, not a stream of duplicates
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

    fired = 0
    t0 = time.monotonic()
    try:
        async for inc in percept.run(source=args.source, goals=goals, vision=args.vision, fps=args.fps):
            fired += 1
            gid = getattr(inc, "concern_id", getattr(inc, "goal_id", "?"))
            msg = getattr(inc, "message", None) or getattr(inc, "say", "") or gid
            conf = getattr(getattr(inc, "verdict", None), "confidence", None)
            elapsed = time.monotonic() - t0
            tag = f" ({conf:.0%})" if isinstance(conf, (int, float)) else ""
            print(f"  🔔  [{elapsed:6.1f}s]  {msg}{tag}   ·   goal={gid}")
    except KeyboardInterrupt:
        print("\n  stopped.")
    print(f"\n  {fired} notification(s).")
    return 0


def main() -> int:
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
    args = p.parse_args()

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
