"""Trust scorecard — the firing contract over the human-approved golden gate cases (spec §9).

This is the L2 measurement that matters most: does the system **refuse to guess**? For each
human-approved ``gate_case`` we replay the scene's frames through the real vision judge + the
three-state gate and compare the predicted action (fire / silent / ask) to the human label. The
headline is **false-fire** (predicted FIRE when the truth is SILENT) — the trust gate, target 0.

Gated: ``PERCEPT_LIVE=1`` + a vision key runs it live; offline it runs the fake backend so the
harness itself is exercised (the numbers are only meaningful live). Frames per scene are judged
concurrently (a scene fires iff ANY frame yields a confident-satisfied verdict — armed gate,
freshness=1.0 — so order doesn't change "did it fire").

It scores the **approved** subset only (human ground truth) and prints, in big letters, that the
current fixtures are private/YouTube-sourced and NOT a publishable benchmark.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from percept import registry
from percept.config import load_settings
from percept.eval import find_golden_root, load_golden
from percept.gate import classify

from ._dotenv import load_dotenv
from .frames import frames_for_scene

VERDICTS = ("fire", "silent", "ask")


def _predict(verdicts: list, threshold: float, fresh_min: float = 0.5) -> str:
    """Collapse a scene's per-frame verdicts to one action (armed gate, fresh facts)."""
    truths = [classify(v, 1.0, threshold, fresh_min) for v in verdicts]
    if "yes" in truths:
        return "fire"  # a rising edge on a fresh, armed gate
    if "unknown" in truths:
        return "ask"  # ambiguous / low-confidence -> never a confident-wrong nag
    return "silent"


async def _judge_scene(vision, condition, frames, sem):
    async def one(frame):
        async with sem:
            return await vision.judge(condition, frame)

    return await asyncio.gather(*(one(f) for _, f in frames))


async def run(live: bool, concurrency: int = 8) -> dict:
    settings = load_settings()
    root = find_golden_root()
    if root is None:
        raise SystemExit("no golden fixtures found")
    golden = load_golden(root)
    cases = [
        g
        for g in golden.gate_cases
        if g.get("review_status") == "approved"
        and (root / "scenes" / g["scene_id"] / "clip.json").exists()
    ]
    backend = "claude" if live else "fake"
    vision = registry.make_vision(backend, settings)
    sem = asyncio.Semaphore(concurrency)
    threshold = settings.gate.threshold

    rows, calls = [], 0
    for g in cases:
        frames = frames_for_scene(root / "scenes" / g["scene_id"])
        verdicts = await _judge_scene(vision, g["goal"]["condition"], frames, sem)
        calls += len(verdicts)
        rows.append(
            {
                "scene_id": g["scene_id"],
                "split": g.get("split"),
                "condition": g["goal"]["condition"],
                "expected": g["expected_verdict"],
                "predicted": _predict(verdicts, threshold),
                "frames": len(verdicts),
            }
        )
    return {"backend": backend, "threshold": threshold, "calls": calls, "rows": rows}


def metrics(report: dict) -> dict:
    rows = report["rows"]
    n = len(rows)
    conf = {e: {p: 0 for p in VERDICTS} for e in VERDICTS}
    for r in rows:
        conf[r["expected"]][r["predicted"]] += 1
    exp = {e: sum(conf[e].values()) for e in VERDICTS}
    false_fire = conf["silent"]["fire"]  # the trust number
    agreement = sum(conf[v][v] for v in VERDICTS)
    return {
        "n": n,
        "false_fire": false_fire,
        "silent_cases": exp["silent"],
        "agreement": agreement,
        "fire_recall": (conf["fire"]["fire"], exp["fire"]),
        "silent_specificity": (conf["silent"]["silent"], exp["silent"]),
        "ask_accuracy": (conf["ask"]["ask"], exp["ask"]),
        "confusion": conf,
    }


def render(report: dict, m: dict) -> str:
    c = m["confusion"]
    pct = lambda a, b: f"{100*a/b:.0f}%" if b else "n/a"  # noqa: E731
    lines = [
        "",
        "=" * 64,
        "  PERCEPT TRUST SCORECARD",
        f"  backend={report['backend']}  gate_threshold={report['threshold']}  "
        f"cases={m['n']}  vision_calls={report['calls']}",
        "  ⚠ DEV FIXTURES — private, YouTube-sourced, NOT a publishable benchmark (spec §9.1).",
        "=" * 64,
        "",
        "  THE TRUST GATE",
        f"    false-fire ............ {m['false_fire']} / {m['silent_cases']} silent   "
        f"(target 0 — predicted FIRE when truth is SILENT)",
        "",
        "  AGREEMENT WITH HUMAN LABELS",
        f"    exact verdict match ... {m['agreement']} / {m['n']}  ({pct(m['agreement'], m['n'])})",
        f"    fire recall ........... {m['fire_recall'][0]} / {m['fire_recall'][1]}",
        f"    silent specificity .... {m['silent_specificity'][0]} / {m['silent_specificity'][1]}",
        f"    ask accuracy .......... {m['ask_accuracy'][0]} / {m['ask_accuracy'][1]}",
        "",
        "  CONFUSION  (rows=human truth, cols=predicted)",
        "                fire   silent   ask",
    ]
    for e in VERDICTS:
        lines.append(
            f"    {e:<8} {c[e]['fire']:>6} {c[e]['silent']:>8} {c[e]['ask']:>6}"
        )
    lines += ["", "=" * 64, ""]
    return "\n".join(lines)


def main() -> int:
    load_dotenv()
    live = os.environ.get("PERCEPT_LIVE") == "1"
    if not live:
        print("scorecard: PERCEPT_LIVE!=1 — running the FAKE backend (harness check only; "
              "numbers are not meaningful). Set PERCEPT_LIVE=1 + a key for the real scorecard.")
    report = asyncio.run(run(live))
    m = metrics(report)
    out = Path(os.environ.get("PERCEPT_SCORECARD_OUT", "scorecard.json"))
    out.write_text(json.dumps({"report": report, "metrics": m}, indent=2, default=str))
    print(render(report, m))
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI entry
    raise SystemExit(main())
