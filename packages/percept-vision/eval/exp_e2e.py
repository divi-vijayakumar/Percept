"""E2E benchmark — full Percept (stateful gate) vs a raw-VLM-in-a-loop baseline (spec §9 Tier-1).

The model A/B measured the *eyes* (per-frame VLM). This measures the *system*: it replays each
Perception Test clip's frames IN ORDER and, holding **perception constant** (one VLM judgment per
frame, shared by both systems), compares two firing policies against the ground-truth action onsets:

* **baseline** — raw VLM in a loop: fire on EVERY confident-yes frame (no memory).
* **percept** — the real three-state ``Gate``: rising-edge + refractory -> fire ONCE per onset,
  silent while it stays true, re-arm on confident-false.

Both are aligned to the onsets with the same replay-and-align scorer (``eval.replay``). The delta —
same recall, far fewer fires, higher precision, near-zero count error — is what the cognition layer
is worth (it turns a spammy loop into one precise notification per event). Perception cost is equal
by construction (both judge every frame); the call-cost win is the edge/cache, measured separately.

Live: PERCEPT_LIVE=1 + a key. Model via PERCEPT_E2E_MODEL (default gemini-2.5-flash, the A/B winner).
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path

from percept.adapters import Verdict
from percept.config import load_settings
from percept.gate import FIRE, Gate, classify

from ._dotenv import load_dotenv
from .exp_perception_test import DATA, SPECIFIC, condition, extract_jpeg, _find_video
from .replay import Edge, Fire, score

NUM_SYS = ('Answer if the statement is true of the image. Return STRICT JSON '
           '{"satisfied":true|false,"confidence":0..1}.')


def _json(t):
    s, e = t.find("{"), t.rfind("}")
    try:
        return json.loads(t[s:e + 1])
    except Exception:
        return {}


def gemini_client():
    key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
           or os.environ.get("GOOGLE_AISTUDIO_API_KEY"))
    from google import genai
    return genai.Client(api_key=key)


def gemini_judge(client, model, cond, frame):
    from google.genai import types
    media, raw = frame
    r = client.models.generate_content(
        model=model,
        contents=[types.Part.from_bytes(data=raw, mime_type=media), f"{NUM_SYS}\nStatement: \"{cond}\". JSON."],
    )
    d = _json(r.text or "")
    return bool(d.get("satisfied")), float(d.get("confidence", 0.5))


def sample_indices(num_frames, fps, cap=30):
    step = max(1, round(fps))  # ~1 fps
    idx = list(range(0, num_frames, step))
    if len(idx) > cap:  # thin uniformly to cap
        idx = idx[:: max(1, len(idx) // cap)][:cap]
    return idx


async def main():
    load_dotenv()
    s = load_settings(); thr = s.gate.threshold
    model = os.environ.get("PERCEPT_E2E_MODEL", "gemini-2.5-flash")
    omega = float(os.environ.get("PERCEPT_EVAL_OMEGA", 2.0))
    ann = json.load(open(DATA / "sample.json"))
    client = gemini_client()
    sem = asyncio.Semaphore(6)
    loop = asyncio.get_event_loop()

    # build (video, label) goals over specific actions; edges = segment onsets (seconds)
    goals = []
    for vid, v in ann.items():
        if _find_video(vid) is None:
            continue
        fps = v["metadata"]["frame_rate"]; nframes = v["metadata"]["num_frames"]
        by_label = {}
        for seg in v.get("action_localisation", []):
            if seg["label"] in SPECIFIC:
                by_label.setdefault(seg["label"], []).append(seg["frame_ids"][0] / fps)
        for label, onsets in by_label.items():
            goals.append({"vid": vid, "label": label, "fps": fps, "nframes": nframes, "onsets": onsets})

    print(f"E2E: {len(goals)} (video,action) goals across {len(ann)} clips · model={model} · omega={omega}s")

    base_fires, perc_fires, edges = [], [], []
    for g in goals:
        gid = f"{g['vid']}:{g['label']}"
        cond = condition(g["label"])
        idxs = sample_indices(g["nframes"], g["fps"])

        async def judge_idx(i):
            frame = extract_jpeg(_find_video(g["vid"]), i)
            if frame is None:
                return None
            ts = i / g["fps"]
            try:
                async with sem:
                    sat, cf = await loop.run_in_executor(None, gemini_judge, client, model, cond, frame)
            except Exception:
                return None
            return ts, sat, cf

        verdicts = [r for r in await asyncio.gather(*(judge_idx(i) for i in idxs)) if r]
        verdicts.sort(key=lambda x: x[0])
        # edges
        for t in g["onsets"]:
            edges.append(Edge(gid, t))
        # baseline: fire on every confident-yes frame
        gate = Gate(threshold=thr)
        for ts, sat, cf in verdicts:
            v = Verdict(satisfied=sat, confidence=cf)
            if classify(v, 1.0, thr, gate.fresh_min) == "yes":
                base_fires.append(Fire(gid, ts, cf))
            # percept: through the stateful gate (rising-edge + refractory)
            if gate.evaluate((gid,), v, freshness=1.0) == FIRE:
                perc_fires.append(Fire(gid, ts, cf))

    b = score(base_fires, edges, omega)
    p = score(perc_fires, edges, omega)
    print("\n" + "=" * 76)
    print("  E2E — FULL PERCEPT (gate) vs RAW-VLM-IN-A-LOOP  (same perception, same clips)")
    print(f"  {len(edges)} ground-truth action onsets · omega={omega}s · perception held constant")
    print("=" * 76)
    hdr = f"  {'system':<10}{'fires':>7}{'TP':>5}{'FP':>5}{'FN':>5}{'precision':>11}{'recall':>9}{'PAUC':>7}{'count-err':>11}"
    print(hdr)
    for name, m in [("baseline", b), ("percept", p)]:
        print(f"  {name:<10}{m.n_fires:>7}{m.tp:>5}{m.fp:>5}{m.fn:>5}"
              f"{m.precision:>11.2f}{m.recall:>9.2f}{m.pauc:>7.2f}{m.count_off_by_one:>11}")
    print("=" * 76)
    print(f"  Percept fires {b.n_fires}->{p.n_fires} ({100*(1-p.n_fires/max(1,b.n_fires)):.0f}% fewer) at "
          f"recall {p.recall:.2f} vs {b.recall:.2f}; precision {p.precision:.2f} vs {b.precision:.2f}; "
          f"count-err {p.count_off_by_one} vs {b.count_off_by_one}")
    Path("e2e_result.json").write_text(json.dumps({"baseline": b.as_dict(), "percept": p.as_dict()}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    if os.environ.get("PERCEPT_LIVE") != "1":
        print("set PERCEPT_LIVE=1 + key."); raise SystemExit(0)
    asyncio.run(main())
