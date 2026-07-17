"""Phase-B EPISODE-LEVEL quality run — executes the FROZEN protocol in QUALITY_GATE_PROTOCOL.md.

ONE for-record measurement. Stock gemini-3.1-flash-lite + default prompt/threshold — NO tuning.
Frozen predicted-event rule: one FIRE at the FIRST frame where the judge returns satisfied=True per
scene×condition. Scored at the EVENT level via golden_eval.score (Viability §5a matching). The
recall/precision is recorded AS-IS against the locked 0.90 bars; the stopping rule (protocol §Stopping)
governs what happens next — this script never tunes or re-runs to chase the bar.
"""
from __future__ import annotations

import asyncio
import base64
import datetime as _dt
import hashlib
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "packages" / "percept-vision"))
sys.path.insert(0, str(_REPO / "eval" / "viability"))

import yaml  # noqa: E402

from percept.backends.gemini import GeminiVision  # noqa: E402
from percept.config import load_settings  # noqa: E402
from percept.contracts import JudgmentEvent, JudgmentTiming  # noqa: E402
from percept.harness.golden_eval import score  # noqa: E402
from golden_scenes import load_real_clip_episodes  # noqa: E402

CONFIG_PATH = _REPO / "eval" / "viability" / "reference_config.yaml"
SCENES_DIR = _REPO / "eval" / "fixtures-golden" / "scenes"
RESULTS_ROOT = _REPO / "eval" / "viability" / "results" / "phase_b_episode"


def _load_key():
    for line in (_REPO / ".env").read_text().splitlines():
        if line.startswith("GOOGLE_AISTUDIO_API_KEY="):
            v = line.split("=", 1)[1].strip().strip('"').strip("'")
            if v:
                for k in ("GOOGLE_AISTUDIO_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
                    os.environ.setdefault(k, v)
                return v
    return None


def _frames(scene_id):
    clip = json.loads((SCENES_DIR / scene_id / "clip.json").read_text())
    return [(float(t), base64.b64decode(b64)) for t, b64 in clip.get("frames", [])]


def _elig_window(gte):
    gt = gte.ground_truth
    return float(gt.decision_eligible_ts), float(gte.episode_end_ts), float(getattr(gte, "allowed_lateness_s", 0.0) or 0.0)


async def run():
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    if cfg.get("bars_locked") is not True:
        raise SystemExit("REFUSING: bars_locked is not true.")
    if not _load_key():
        raise SystemExit("REFUSING: no GOOGLE_AISTUDIO_API_KEY.")
    os.environ["PERCEPT_VISION_MODEL"] = cfg["reference_config"]["provider_model_id"]
    os.environ["PERCEPT_VISION_BACKEND"] = "gemini"
    judge = GeminiVision(load_settings())
    model = str(getattr(judge, "_model", "?"))
    if not model.startswith("gemini"):
        raise SystemExit(f"model resolution failed: {model!r}")

    nb = cfg["numeric_bars"]
    rmin = float(list(nb["recall_min"].values())[0]); pmin = float(list(nb["precision_min"].values())[0])

    eps = load_real_clip_episodes()  # default: approved + pinned (strictest / cleanest labels)
    episodes = []; predictions = []; per = []; in_tok = out_tok = calls = 0
    for i, ep in enumerate(eps):
        episodes.append(ep.episode)
        elig, end, late = _elig_window(ep.episode)
        fire_t = None
        for t, jpeg in _frames(ep.scene_id):   # FROZEN rule: time order, first satisfied=True
            v = await judge.judge(ep.condition, ("image/jpeg", jpeg)); calls += 1
            u = v.usage or {}; in_tok += int(u.get("input_tokens", 0)); out_tok += int(u.get("output_tokens", 0))
            if bool(v.satisfied):
                fire_t = t; break
        if fire_t is not None:
            predictions.append(JudgmentEvent(
                key=ep.scene_id, evidence_seq=i, action="fire",
                timing=JudgmentTiming(ingest_observed_ts=fire_t, admitted_ts=fire_t, judge_started_ts=fire_t,
                                      judge_completed_ts=fire_t, decision_emitted_ts=fire_t)))
        in_window = fire_t is not None and (elig <= fire_t <= end + late)
        per.append({"scene_id": ep.scene_id, "condition": ep.condition[:70],
                    "fired_at": fire_t, "window": [round(elig, 2), round(end + late, 2)],
                    "in_window": in_window})

    res = score(episodes, predictions)
    recall, precision = res.recall, res.precision
    passed = (recall >= rmin and precision >= pmin)

    return {
        "phase": "B-episode", "protocol": "QUALITY_GATE_PROTOCOL.md (frozen, pre-registered)",
        "model": model, "n_episodes": len(episodes), "n_predictions": len(predictions), "n_calls": calls,
        "recall": recall, "precision": precision, "recall_min": rmin, "precision_min": pmin,
        "matches": len(episodes) - res.misses, "misses": res.misses,
        "false_fires": res.false_fires, "verdict": "PASS" if passed else "REVIEW/NO-GO",
        "stopping_rule": ("G2 quality GO — emit launch_profile reflex (G3 GREEN, econ PASS)" if passed else
                          "G2 quality REVIEW/NO-GO. Per protocol: NO tune-on-eval-set. Founder decision: "
                          "accept NO-GO / ONE pre-registered held-out attempt / committed bar change."),
        "power_caveat": f"n_episodes={len(episodes)} (§5a single-reviewer) — wide CIs; not a rigorous gate alone.",
        "tokens": {"in": in_tok, "out": out_tok}, "per_episode": per,
    }


def _write(res):
    now = _dt.datetime.now(_dt.timezone.utc); run_id = now.strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_ROOT / run_id; out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    (out / "run_stamp.json").write_text(json.dumps({
        "run_id": run_id, "timestamp_utc": now.isoformat(), "phase": "B-episode",
        "protocol_sha256": hashlib.sha256((_REPO / "eval" / "viability" / "QUALITY_GATE_PROTOCOL.md").read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "model": res["model"], "n_episodes": res["n_episodes"]}, indent=2))
    md = f"""# Phase-B Episode-Level Quality — For-Record (frozen protocol)

**Model:** `{res['model']}` · protocol `QUALITY_GATE_PROTOCOL.md` (pre-registered) · run `{run_id}`
**n_episodes = {res['n_episodes']}** (approved+pinned §5a) · predictions {res['n_predictions']} · calls {res['n_calls']}

## Event-level result (as-is, no tuning)
- **recall {round(res['recall'], 3)}** (min {res['recall_min']}) · **precision {round(res['precision'], 3)}** (min {res['precision_min']})
- matches {res['matches']} · misses {res['misses']} · false-fires {res['false_fires']}
- **VERDICT: {res['verdict']}**

## Stopping rule (frozen, applied)
{res['stopping_rule']}

## Power caveat
{res['power_caveat']}

## Per-episode
| scene | fired_at | eligible window | in-window |
|---|---|---|---|
""" + "\n".join(f"| {p['scene_id']} | {p['fired_at']} | {p['window']} | {p['in_window']} |" for p in res["per_episode"]) + "\n"
    (out / "report.md").write_text(md)
    return out


if __name__ == "__main__":
    res = asyncio.run(run())
    out = _write(res)
    print(f"Episode-level run → {out}")
    print(f"  n_episodes={res['n_episodes']} predictions={res['n_predictions']} calls={res['n_calls']}")
    print(f"  recall={round(res['recall'],3)} precision={round(res['precision'],3)} (bar {res['recall_min']}/{res['precision_min']})")
    print(f"  matches={res['matches']} misses={res['misses']} false_fires={res['false_fires']}")
    print(f"  VERDICT: {res['verdict']}")
