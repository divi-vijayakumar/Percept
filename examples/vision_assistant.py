#!/usr/bin/env python3
"""A personal vision assistant on Percept — for smart glasses / a body-cam.

Two things a "where are my things" assistant needs, both built on `percept.run`:

  1. LOCATION MEMORY — "where are my keys?"  Watch for each item; every time it's seen, remember WHEN and
     WHERE (the judge's own description — "keys on the kitchen counter" — is the location answer).

  2. LEAVE-WITHOUT-X REMINDER — "remind me to grab my handbag."  Watch a "leaving" cue and each item; when
     you head out and an item hasn't been seen recently, fire a reminder.

Run it:
    export OPENROUTER_API_KEY=sk-or-...
    python examples/vision_assistant.py --egomotion            # glasses/body-cam
    python examples/vision_assistant.py --source clip.mp4      # test on a recording

This is a STARTING POINT — the memory + leave-logic below is plain Python you own; Percept supplies the
detections. See docs/how-to/integrate-into-your-app.md.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
from pathlib import Path

# ---- what the assistant cares about (edit these) ----
ITEMS = ["keys", "handbag", "wallet", "phone"]           # things to keep track of
LEAVING_CUE = "the person is at the front door about to go out (jacket or shoes on, reaching for the door)"
FRESH_S = 45.0            # an item "with you" if seen within this many seconds of leaving
# -----------------------------------------------------


def _load_key() -> None:
    for env in (Path.cwd() / ".env", Path(__file__).resolve().parents[1] / ".env"):
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    os.environ.setdefault("OPENROUTER_API_KEY", line.split("=", 1)[1].strip())


def _config(egomotion: bool) -> None:
    os.environ["PERCEPT_SALIENCE_STAT"] = "block_max"
    os.environ["PERCEPT_EGO_COMPENSATE"] = "true" if egomotion else "false"
    os.environ["PERCEPT_GATE_REFRACTORY_S"] = "8"      # short: we WANT periodic re-sightings to refresh location
    os.environ["PERCEPT_VISION_BACKEND"] = "openrouter"
    os.environ.setdefault("PERCEPT_VISION_MODEL", "qwen/qwen3-vl-32b-instruct")
    os.environ.setdefault("PERCEPT_NO_BANNER", "1")


class Assistant:
    """Your app state: last-seen location per item + the leave-without-X reminder logic."""

    def __init__(self) -> None:
        self.last_seen: dict[str, dict] = {}     # item -> {"t": monotonic, "where": evidence string}

    # --- 1. LOCATION MEMORY ---
    def note_seen(self, item: str, t: float, where: str) -> None:
        self.last_seen[item] = {"t": t, "where": where or "(no description)"}
        print(f"  👁  saw your {item}: {where}")

    def where_is(self, item: str) -> str:
        r = self.last_seen.get(item)
        if not r:
            return f"I haven't spotted your {item} yet."
        ago = int(time.monotonic() - r["t"])
        return f"Your {item} — last seen ~{ago}s ago: {r['where']}"

    # --- 2. LEAVE-WITHOUT-X ---
    def on_leaving(self, t: float) -> None:
        forgotten = [it for it in ITEMS if t - self.last_seen.get(it, {}).get("t", 0.0) > FRESH_S]
        if forgotten:
            print(f"\n  🔔 HEADING OUT — you may have forgotten: {', '.join(forgotten)}!\n")
        else:
            print("\n  ✅ Heading out — looks like you have everything.\n")


async def run(source: str, egomotion: bool) -> None:
    logging.disable(logging.WARNING)
    import percept
    from percept import Goal

    a = Assistant()
    goals = [Goal(id=it, condition=f"my {it} is clearly visible", say=it) for it in ITEMS]
    goals.append(Goal(id="__leaving__", condition=LEAVING_CUE, say="leaving"))

    print("─" * 68)
    print(f"  VISION ASSISTANT · {source}" + ("  [egomotion]" if egomotion else ""))
    print(f"  tracking: {', '.join(ITEMS)}   ·   reminder on: leaving")
    print("─" * 68 + "\n")

    async for inc in percept.run(source=source, goals=goals, vision="openrouter", fps=1.0):
        if inc.concern_id == "__leaving__":
            a.on_leaving(inc.ts)
        else:
            a.note_seen(inc.concern_id, inc.ts, getattr(inc.verdict, "evidence", "") if inc.verdict else "")

    # after the stream ends (a recording): answer the classic query
    print("\n" + "─" * 68)
    for it in ITEMS:
        print("  " + a.where_is(it))


def main() -> int:
    p = argparse.ArgumentParser(description="Personal vision assistant on Percept")
    p.add_argument("--source", default="webcam:0", help="webcam:0 | a video file | rtsp://…")
    p.add_argument("--egomotion", action="store_true", help="moving camera (glasses/body-cam)")
    args = p.parse_args()
    _load_key()
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("set OPENROUTER_API_KEY=sk-or-…"); return 2
    _config(args.egomotion)
    try:
        asyncio.run(run(args.source, args.egomotion))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
