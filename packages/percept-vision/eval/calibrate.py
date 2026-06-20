"""Calibration loop — the gate stops being magic (spec §10.3).

The trust scorecard exposed an uncalibrated gate (false-fire > 0; the ASK valve never engaged
because the model is overconfident at the default threshold). This runner:

1. **captures** raw per-frame ``(satisfied, confidence)`` for every approved gate case in ONE live
   pass (then all sweeping is offline + free);
2. **sweeps** the gate threshold on the **train** split, picking the value that **zeros false-fire
   while maximizing fire-recall** (when no threshold zeros it, the model's confidence isn't
   discriminative — an honest, important finding);
3. **validates** the chosen threshold on dev + test (never tuned on test) and prints before/after.

It writes a calibration artifact (the sweep + the chosen threshold) — the auditable record the spec
asks for. Gated by ``PERCEPT_LIVE`` for the capture; the sweep runs on whatever capture exists.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from percept import registry
from percept.adapters import Verdict
from percept.config import load_settings
from percept.eval import find_golden_root, load_golden

from ._dotenv import load_dotenv
from .frames import frames_for_scene

VERDICTS = ("fire", "silent", "ask")
GRID = [round(0.50 + 0.02 * i, 2) for i in range(26)]  # 0.50 .. 1.00
FRESH_MIN = 0.5


def _predict(verdicts: list[tuple[bool, float]], threshold: float) -> str:
    """Per-scene action from captured (satisfied, confidence) pairs at a given threshold."""
    truths = []
    for satisfied, conf in verdicts:
        if conf < threshold:
            truths.append("unknown")
        else:
            truths.append("yes" if satisfied else "no")
    if "yes" in truths:
        return "fire"
    if "unknown" in truths:
        return "ask"
    return "silent"


async def _capture(live: bool, concurrency: int = 8) -> list[dict]:
    settings = load_settings()
    root = find_golden_root()
    golden = load_golden(root)
    cases = [
        g
        for g in golden.gate_cases
        if g.get("review_status") == "approved"
        and (root / "scenes" / g["scene_id"] / "clip.json").exists()
    ]
    vision = registry.make_vision("claude" if live else "fake", settings)
    sem = asyncio.Semaphore(concurrency)

    async def judge(condition, frame):
        async with sem:
            v: Verdict = await vision.judge(condition, frame)
            return [bool(v.satisfied), float(v.confidence)]

    out = []
    for g in cases:
        frames = frames_for_scene(root / "scenes" / g["scene_id"])
        verdicts = await asyncio.gather(*(judge(g["goal"]["condition"], f) for _, f in frames))
        out.append(
            {
                "scene_id": g["scene_id"],
                "split": g.get("split"),
                "expected": g["expected_verdict"],
                "verdicts": verdicts,
            }
        )
    return out


def _evaluate(cases: list[dict], threshold: float) -> dict:
    conf = {e: {p: 0 for p in VERDICTS} for e in VERDICTS}
    for c in cases:
        conf[c["expected"]][_predict(c["verdicts"], threshold)] += 1
    exp = {e: sum(conf[e].values()) for e in VERDICTS}
    return {
        "false_fire": conf["silent"]["fire"],
        "fire_recall": (conf["fire"]["fire"], exp["fire"]),
        "ask_acc": (conf["ask"]["ask"], exp["ask"]),
        "agreement": sum(conf[v][v] for v in VERDICTS),
        "n": sum(exp.values()),
        "confusion": conf,
    }


def _choose(train: list[dict]) -> tuple[float, list[dict]]:
    """Pick the threshold that zeros false-fire while maximizing fire-recall on train."""
    sweep = []
    for thr in GRID:
        m = _evaluate(train, thr)
        sweep.append({"threshold": thr, **{k: m[k] for k in ("false_fire", "fire_recall", "ask_acc")}})
    zero = [s for s in sweep if s["false_fire"] == 0]
    pool = zero or sweep
    # maximize fire-recall, then prefer the LOWER threshold (keeps more confident fires)
    best = max(pool, key=lambda s: (s["fire_recall"][0], -s["threshold"]))
    return best["threshold"], sweep


def _fmt(m: dict) -> str:
    fr, ar = m["fire_recall"], m["ask_acc"]
    return (
        f"false-fire={m['false_fire']}  fire-recall={fr[0]}/{fr[1]}  "
        f"ask={ar[0]}/{ar[1]}  agree={m['agreement']}/{m['n']}"
    )


def main() -> int:
    load_dotenv()
    live = os.environ.get("PERCEPT_LIVE") == "1"
    root = find_golden_root()
    cap_path = Path(os.environ.get("PERCEPT_CALIB_CAPTURE", root / "calibration_capture.json"))

    if live or not cap_path.exists():
        if not live:
            print("calibrate: no capture and PERCEPT_LIVE!=1 — set PERCEPT_LIVE=1 + a key to capture.")
        cases = asyncio.run(_capture(live))
        cap_path.write_text(json.dumps(cases))
        print(f"captured {len(cases)} cases -> {cap_path}")
    else:
        cases = json.loads(cap_path.read_text())
        print(f"loaded {len(cases)} cases from {cap_path} (offline sweep)")

    by = {s: [c for c in cases if c["split"] == s] for s in ("train", "dev", "test")}
    default_thr = load_settings().gate.threshold
    chosen, sweep = _choose(by["train"])

    lines = ["", "=" * 70, "  GATE CALIBRATION (spec §10.3) — sweep on TRAIN, validate on dev/test", "=" * 70, ""]
    lines.append("  TRAIN SWEEP (threshold -> false-fire / fire-recall / ask):")
    for s in sweep:
        mark = "  <- chosen" if s["threshold"] == chosen else ""
        lines.append(
            f"    {s['threshold']:.2f}   ff={s['false_fire']}  "
            f"recall={s['fire_recall'][0]}/{s['fire_recall'][1]}  "
            f"ask={s['ask_acc'][0]}/{s['ask_acc'][1]}{mark}"
        )
    lines += ["", f"  CHOSEN THRESHOLD: {chosen}  (was {default_thr})", ""]
    for split in ("train", "dev", "test"):
        if not by[split]:
            continue
        before, after = _evaluate(by[split], default_thr), _evaluate(by[split], chosen)
        lines.append(f"  {split.upper():5}  before({default_thr}): {_fmt(before)}")
        lines.append(f"         after({chosen}):  {_fmt(after)}")
    overall_after = _evaluate(cases, chosen)
    c = overall_after["confusion"]
    lines += ["", "  CONFUSION @ chosen (rows=truth, cols=pred)   fire  silent  ask"]
    for e in VERDICTS:
        lines.append(f"    {e:<8}                               {c[e]['fire']:>4} {c[e]['silent']:>7} {c[e]['ask']:>5}")
    lines += ["", "=" * 70, ""]
    print("\n".join(lines))

    artifact = {
        "default_threshold": default_thr,
        "chosen_threshold": chosen,
        "train_sweep": sweep,
        "by_split": {s: _evaluate(by[s], chosen) for s in by if by[s]},
        "recommendation": f"PERCEPT_GATE_THRESHOLD={chosen}",
    }
    art_path = Path(os.environ.get("PERCEPT_CALIB_OUT", "calibration.json"))
    art_path.write_text(json.dumps(artifact, indent=2, default=str))
    print(f"  wrote {art_path}  —  recommended PERCEPT_GATE_THRESHOLD={chosen}")
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI entry
    raise SystemExit(main())
