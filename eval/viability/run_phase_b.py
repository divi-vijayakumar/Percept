"""Phase-B for-record viability run — REAL Gemini (gemini-3.1-flash-lite, generateContent).

Real judge over REAL golden-clip frames (eval/fixtures-golden), scored against the reviewer-approved
rising-edge ground truth. Produces the numbers Phase-A could not: real recall/precision, real judge
latency (G3-A floor), and real $/stream-hour (from returned token usage). Emits the launch_profile.

Honesty: quality runs ONLY on real imagery (golden clips), never the synthetic architecture fixtures.
The §5a golden set is small + single-reviewer — sample sizes are reported, not hidden. Real long-soak
(P-STAB-1 24h) is NOT run against the provider (architecture-validated in Phase-A); noted as such.

Run: python3 eval/viability/run_phase_b.py   (needs GOOGLE_AISTUDIO_API_KEY + PERCEPT_VISION_MODEL=gemini-*)
"""
from __future__ import annotations

import asyncio
import base64
import datetime as _dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "packages" / "percept-vision"))
sys.path.insert(0, str(_REPO / "eval" / "viability"))

import yaml  # noqa: E402

from percept.backends.gemini import GeminiVision  # noqa: E402
from percept.config import load_settings  # noqa: E402
# (golden_scenes loader not needed: quality labels come from clip.json `intervals` directly)

CONFIG_PATH = _REPO / "eval" / "viability" / "reference_config.yaml"
SCENES_DIR = _REPO / "eval" / "fixtures-golden" / "scenes"
RESULTS_ROOT = _REPO / "eval" / "viability" / "results" / "phase_b"

# gemini-3.1-flash-lite published rates (USD / 1M tokens) — ASSUMPTION, labeled in the report.
# If pricing differs, only the $ number scales; token counts are exact from usage_metadata.
PRICE_IN_PER_M = 0.10
PRICE_OUT_PER_M = 0.40


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _pctile(xs, q):
    if not xs:
        return 0.0
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round((q / 100.0) * (len(s) - 1)))))
    return s[i]


def _preflight():
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    if cfg.get("bars_locked") is not True:
        raise SystemExit("REFUSING: bars_locked is not true.")
    if not os.environ.get("GOOGLE_AISTUDIO_API_KEY") and not _dotenv_key():
        raise SystemExit("REFUSING: no GOOGLE_AISTUDIO_API_KEY (Phase-B needs the provider).")
    return cfg


def _dotenv_key():
    for line in (_REPO / ".env").read_text().splitlines():
        if line.startswith("GOOGLE_AISTUDIO_API_KEY="):
            v = line.split("=", 1)[1].strip().strip('"').strip("'")
            if v:
                os.environ["GOOGLE_AISTUDIO_API_KEY"] = v
                # the google-genai client also honors GEMINI_API_KEY / GOOGLE_API_KEY
                os.environ.setdefault("GEMINI_API_KEY", v)
                os.environ.setdefault("GOOGLE_API_KEY", v)
                return v
    return None


def _labeled_samples():
    """Yield (scene_id, condition, t, jpeg_bytes, label) using clip.json's `intervals` — the labeled
    true-spans per condition (the CORRECT per-frame ground truth): a frame is positive iff its time
    falls inside any true-span for that condition, else negative. Empty intervals => all-negative."""
    for scene_dir in sorted(SCENES_DIR.iterdir()):
        clip_path = scene_dir / "clip.json"
        if not clip_path.exists():
            continue
        clip = json.loads(clip_path.read_text())
        conditions = clip.get("conditions") or []
        intervals = clip.get("intervals") or {}
        frames = clip.get("frames") or []
        for ci, cond in enumerate(conditions):
            spans = intervals.get(str(ci), [])
            for entry in frames:
                t, b64 = float(entry[0]), entry[1]
                label = any(float(a) <= t <= float(b) for a, b in spans)
                yield scene_dir.name, cond, t, base64.b64decode(b64), label


async def _judge_one(judge, sem, scene_id, condition, t, jpeg, label):
    async with sem:
        t0 = time.monotonic()
        try:
            v = await judge.judge(condition, ("image/jpeg", jpeg))
            dt = time.monotonic() - t0
            return {"scene_id": scene_id, "t": t, "label": label,
                    "pred": bool(v.satisfied), "latency_s": dt, "usage": dict(v.usage or {})}
        except Exception as e:
            return {"scene_id": scene_id, "t": t, "label": label, "error": f"{type(e).__name__}: {str(e)[:80]}"}


async def run():
    cfg = _preflight()
    _dotenv_key()
    nb = cfg["numeric_bars"]
    recall_min = float(list(nb["recall_min"].values())[0]) if isinstance(nb["recall_min"], dict) else float(nb["recall_min"])
    precision_min = float(list(nb["precision_min"].values())[0]) if isinstance(nb["precision_min"], dict) else float(nb["precision_min"])
    cost_ceiling = float(nb["cost_ceiling_usd_per_stream_hour"])
    lp = cfg["load_profile"]
    E = float(lp["admitted_episodes_per_stream_minute"]); F = int(lp["fan_out_judge_calls_per_episode"])
    vid_judgments_per_stream_hour = E * 60.0 * F

    # Force the LOCKED provider model into env BEFORE load_settings (which reads only os.environ);
    # else Settings.vision_model defaults to claude-opus-4-8 and the Gemini adapter silently falls
    # back to GEMINI_DEFAULT. Verified below via judge._model — refuse to run on the wrong model.
    locked_model = cfg["reference_config"]["provider_model_id"]
    os.environ["PERCEPT_VISION_MODEL"] = locked_model
    os.environ["PERCEPT_VISION_BACKEND"] = "gemini"

    settings = load_settings()
    judge = GeminiVision(settings)
    actual_model = str(getattr(judge, "_model", "?"))
    if not actual_model.startswith("gemini"):
        raise SystemExit(f"model resolution failed: judge._model={actual_model!r} (expected a gemini id)")

    samples = list(_labeled_samples())
    sem = asyncio.Semaphore(6)
    results = await asyncio.gather(*[
        _judge_one(judge, sem, sid, cond, t, jpeg, label) for (sid, cond, t, jpeg, label) in samples
    ])

    tp = fp = fn = tn = 0
    latencies = []; in_tok = out_tok = 0; errors = []
    per = {}
    for r in results:
        if "error" in r:
            errors.append(f"{r['scene_id']}@{r['t']}: {r['error']}"); continue
        latencies.append(r["latency_s"])
        u = r["usage"]; in_tok += int(u.get("input_tokens", 0)); out_tok += int(u.get("output_tokens", 0))
        label, pred = r["label"], r["pred"]
        cell = "tp" if (label and pred) else "fn" if label else "fp" if pred else "tn"
        tp += cell == "tp"; fn += cell == "fn"; fp += cell == "fp"; tn += cell == "tn"
        d = per.setdefault(r["scene_id"], {"tp": 0, "fp": 0, "fn": 0, "tn": 0}); d[cell] += 1
    per_scene = [{"scene_id": k, **v} for k, v in sorted(per.items())]

    n_pos = tp + fn; n_neg = tn + fp
    recall = tp / n_pos if n_pos else None
    precision = tp / (tp + fp) if (tp + fp) else None
    p50 = _pctile(latencies, 50); p95 = _pctile(latencies, 95); p99 = _pctile(latencies, 99)
    n_calls = len(latencies)
    avg_in = in_tok / n_calls if n_calls else 0; avg_out = out_tok / n_calls if n_calls else 0
    cost_per_call = (avg_in * PRICE_IN_PER_M + avg_out * PRICE_OUT_PER_M) / 1e6
    usd_per_stream_hour = cost_per_call * vid_judgments_per_stream_hour

    band = "GREEN" if p95 < 2.0 else ("YELLOW" if p95 < 3.0 else "RED")
    g2_pass = (recall is not None and precision is not None and recall >= recall_min and precision >= precision_min)
    econ_pass = usd_per_stream_hour <= cost_ceiling
    # launch_profile (Viability §9): a deadline_class GO requires G1 AND G2 (utility AND economics) to
    # pass; G3's band only picks reflex-vs-deliberative AMONG workloads that already pass. If G2 quality
    # misses the bar, NO clean profile is emitted no matter how fast the latency is.
    if not (g2_pass and econ_pass):
        deadline_class = "UNDETERMINED — G2 quality below bar (resolve before emitting a profile)"
    elif band == "GREEN":
        deadline_class = "reflex"
    elif band == "YELLOW":
        deadline_class = "bounded (reflex experimental)"
    else:
        deadline_class = "deliberative"

    return {
        "phase": "B", "provider": "gemini generateContent", "model": actual_model,
        "quality": {"recall": recall, "precision": precision, "n_positive": n_pos, "n_negative": n_neg,
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "recall_min": recall_min, "precision_min": precision_min, "pass": g2_pass,
                    "caveat": "frame-level labels from clip.json true-span `intervals` (per-condition); §5a single-reviewer set"},
        "latency_g3a": {"p50_s": round(p50, 3), "p95_s": round(p95, 3), "p99_s": round(p99, 3),
                        "n_calls": n_calls, "band": band,
                        "note": "real judge latency under light concurrency (sem=6); full under-load G3-A/G3-B need the worker pipeline. Clean single-call floor was 0.728s at probe."},
        "econ": {"avg_input_tokens": round(avg_in, 1), "avg_output_tokens": round(avg_out, 1),
                 "cost_per_call_usd": round(cost_per_call, 6),
                 "vid_judgments_per_stream_hour": vid_judgments_per_stream_hour,
                 "usd_per_stream_hour": round(usd_per_stream_hour, 4), "cost_ceiling": cost_ceiling,
                 "pass": econ_pass, "price_assumption_per_m": {"in": PRICE_IN_PER_M, "out": PRICE_OUT_PER_M},
                 "note": "$ scales with the assumed flash-lite price; token counts are exact from usage_metadata"},
        "launch_profile_candidate": {"deadline_class": deadline_class, "provider": "gemini", "model": actual_model,
                                     "g1": "PASS (Phase-A architecture)", "g2_quality": "PASS" if g2_pass else "REVIEW",
                                     "g3_latency": band, "econ": "PASS" if econ_pass else "REVIEW"},
        "per_scene": per_scene, "errors": errors,
        "not_run": ["real 24h soak against provider (architecture-validated in Phase-A; ops decision)",
                    "full under-load G3-A/G3-B through the worker pipeline (single-judge floor measured here)"],
    }


def _write(res):
    now = _dt.datetime.now(_dt.timezone.utc); run_id = now.strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS_ROOT / run_id; out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    (out / "run_stamp.json").write_text(json.dumps({
        "run_id": run_id, "timestamp_utc": now.isoformat(), "phase": "B",
        "git_sha": "e0eb3df57482b93fc78202b992a48c23ea829c31 (+ local Phase-A/B commits)",
        "config_sha256": _sha256(CONFIG_PATH), "model": res["model"],
        "substrate": "REAL gemini-3.1-flash-lite over real golden-clip frames"}, indent=2))
    q = res["quality"]; l = res["latency_g3a"]; e = res["econ"]; lp = res["launch_profile_candidate"]
    md = f"""# Percept Viability — Phase-B For-Record Run (REAL Gemini)

**Model:** `{res['model']}` (generateContent) · git `e0eb3df` (+local) · run `{run_id}`

> Quality is measured on REAL golden-clip frames (§5a secondary set) scored against reviewer-approved
> rising edges. **The §5a set is single-reviewer and small — sample sizes are reported.** Real 24h soak
> was NOT run against the provider (architecture-validated in Phase-A).

## G2 — Quality (real recall/precision)
- **recall {q['recall']}** (min {q['recall_min']}) · **precision {q['precision']}** (min {q['precision_min']}) → **{'PASS' if q['pass'] else 'REVIEW'}**
- positives n={q['n_positive']} · negatives n={q['n_negative']} · TP={q['tp']} FP={q['fp']} FN={q['fn']} TN={q['tn']}
- caveat: {q['caveat']}

## G3 — Latency (real single-judge floor)
- p50 **{l['p50_s']}s** · **p95 {l['p95_s']}s** · p99 {l['p99_s']}s · n={l['n_calls']} → band **{l['band']}**
- {l['note']}

## G2 — Economics (real tokens)
- avg {e['avg_input_tokens']} in / {e['avg_output_tokens']} out tokens/call → **${e['cost_per_call_usd']}/call**
- at {e['vid_judgments_per_stream_hour']} judgments/stream-hour → **${e['usd_per_stream_hour']}/stream-hour** (ceiling ${e['cost_ceiling']}) → **{'PASS' if e['pass'] else 'REVIEW'}**
- {e['note']} (assumed ${e['price_assumption_per_m']['in']}/1M in, ${e['price_assumption_per_m']['out']}/1M out)

## Emitted launch_profile (candidate)
```yaml
launch_profile:
  deadline_class: {lp['deadline_class']}
  provider: gemini    model: {lp['model']}
  g1_boundedness: {lp['g1']}
  g2_quality: {lp['g2_quality']}    g3_latency: {lp['g3_latency']}    economics: {lp['econ']}
  supported_camera: static
```

## Not run (stated, not faked)
""" + "\n".join(f"- {x}" for x in res["not_run"]) + ("\n\n## Errors\n" + "\n".join(f"- {x}" for x in res["errors"]) if res["errors"] else "\n")
    (out / "report.md").write_text(md)
    return out


if __name__ == "__main__":
    res = asyncio.run(run())
    out = _write(res)
    print(f"Phase-B run complete → {out}")
    q = res["quality"]; l = res["latency_g3a"]; e = res["econ"]
    print(f"  G2 quality: recall={q['recall']} precision={q['precision']} (n_pos={q['n_positive']}, n_neg={q['n_negative']})")
    print(f"  G3 latency: p95={l['p95_s']}s → {l['band']} (n={l['n_calls']} real calls)")
    print(f"  Econ: ${e['cost_per_call_usd']}/call → ${e['usd_per_stream_hour']}/stream-hour (ceiling ${e['cost_ceiling']})")
    print(f"  launch_profile.deadline_class = {res['launch_profile_candidate']['deadline_class']}")
    if res["errors"]:
        print(f"  ⚠ {len(res['errors'])} call errors (see report)")
