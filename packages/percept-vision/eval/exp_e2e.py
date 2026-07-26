"""E2E benchmark: full Percept (stateful gate) versus a raw-VLM-in-a-loop baseline.

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
from percept.events import EventLog
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


def interval_score(fires, seg_by_goal) -> dict:
    """Score for STATE-watch goals: a fire inside an action segment [onset, offset] is correct.

    Each segment is satisfied by at most one fire (the earliest inside it); extra fires in a segment
    and fires outside every segment are false-fires; a segment with no fire is a miss.
    """
    by_goal: dict = {}
    for f in fires:
        by_goal.setdefault(f.goal_id, []).append(f)
    tp = fp = fn = count_err = 0
    for gid, segs in seg_by_goal.items():
        gfires = sorted(by_goal.get(gid, []), key=lambda f: f.ts)
        used: set = set()
        for onset, offset in segs:
            hit = None
            for i, f in enumerate(gfires):
                if i not in used and onset <= f.ts <= offset:
                    hit = i
                    break
            if hit is None:
                fn += 1
            else:
                used.add(hit); tp += 1
        fp += len(gfires) - len(used)
        count_err += abs(len(gfires) - len(segs))
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    return {"n_fires": len(fires), "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "count_err": count_err}


async def main():
    load_dotenv()
    s = load_settings(); thr = s.gate.threshold
    model = os.environ.get("PERCEPT_E2E_MODEL", "gemini-2.5-flash")
    omega = float(os.environ.get("PERCEPT_EVAL_OMEGA", 2.0))
    ann = json.load(open(DATA / "sample.json"))
    client = gemini_client()
    sem = asyncio.Semaphore(6)
    loop = asyncio.get_event_loop()

    merge_gap = float(os.environ.get("PERCEPT_EVENT_MERGE_GAP_S", 3.0))
    cap_path = Path("e2e_verdicts.json")

    # build (video, label) goals; each segment is an interval [onset, offset] in seconds.
    goals = []
    for vid, v in ann.items():
        if _find_video(vid) is None:
            continue
        fps = v["metadata"]["frame_rate"]; nframes = v["metadata"]["num_frames"]
        by_label = {}
        for seg in v.get("action_localisation", []):
            if seg["label"] in SPECIFIC:
                f0, f1 = seg["frame_ids"]
                by_label.setdefault(seg["label"], []).append([f0 / fps, f1 / fps])
        for label, segs in by_label.items():
            goals.append({"gid": f"{vid}:{label}", "vid": vid, "label": label,
                          "fps": fps, "nframes": nframes, "segments": segs})

    # ---- capture per-frame verdicts once (or reuse the saved capture for free re-scoring) ----
    if cap_path.exists() and os.environ.get("PERCEPT_E2E_RECAPTURE") != "1":
        caps = json.loads(cap_path.read_text())
        print(f"loaded {len(caps)} goals' verdicts from {cap_path} (offline re-score)")
    else:
        caps = {}
        for g in goals:
            cond = condition(g["label"])
            idxs = sample_indices(g["nframes"], g["fps"])

            async def judge_idx(i, g=g, cond=cond):
                frame = extract_jpeg(_find_video(g["vid"]), i)
                if frame is None:
                    return None
                try:
                    async with sem:
                        sat, cf = await loop.run_in_executor(None, gemini_judge, client, model, cond, frame)
                except Exception:
                    return None
                return [i / g["fps"], sat, cf]

            vs = [r for r in await asyncio.gather(*(judge_idx(i) for i in idxs)) if r]
            vs.sort(key=lambda x: x[0])
            caps[g["gid"]] = {"label": g["label"], "segments": g["segments"], "verdicts": vs}
        cap_path.write_text(json.dumps(caps))
        print(f"captured + saved {len(caps)} goals' verdicts -> {cap_path}")

    print(f"E2E: {len(caps)} (video,action) goals · model={model} · merge_gap={merge_gap}s · "
          f"omega={omega}s (onset ref)")

    # ---- three firing policies over the SAME verdicts ----
    base_fires, perc_fires, event_fires, edges, seg_by_goal = [], [], [], [], {}
    for gid, cap in caps.items():
        seg_by_goal[gid] = cap["segments"]
        for onset, _off in cap["segments"]:
            edges.append(Edge(gid, onset))
        gate = Gate(threshold=thr)
        elog = EventLog(merge_gap_s=merge_gap)
        for ts, sat, cf in cap["verdicts"]:
            v = Verdict(satisfied=sat, confidence=cf)
            if classify(v, 1.0, thr, gate.fresh_min) == "yes":
                base_fires.append(Fire(gid, ts, cf))
                _ev, is_new = elog.observe(cap["label"], ts, confidence=cf)
                if is_new:
                    event_fires.append(Fire(gid, ts, cf))
            if gate.evaluate((gid,), v, freshness=1.0) == FIRE:
                perc_fires.append(Fire(gid, ts, cf))

    print("\n" + "=" * 82)
    print("  E2E — PERCEPT vs RAW-VLM-IN-A-LOOP   (same perception, clips, model)")
    print(f"  {len(edges)} action segments · INTERVAL scoring (a fire inside [onset,offset] = correct")
    print("  for a state-watch goal) · onset@2s shown for reference")
    print("=" * 82)
    print(f"  {'system':<14}{'fires':>7}{'TP':>5}{'FP':>5}{'FN':>5}{'precision':>11}{'recall':>9}{'cnt-err':>9}{'  onset-prec':>12}")
    out = {}
    for name, fires in [("baseline(loop)", base_fires), ("percept-watch", perc_fires), ("percept-event", event_fires)]:
        iv = interval_score(fires, seg_by_goal)
        onset = score(fires, edges, omega)
        out[name] = {"interval": iv, "onset_precision": onset.precision}
        print(f"  {name:<14}{iv['n_fires']:>7}{iv['tp']:>5}{iv['fp']:>5}{iv['fn']:>5}"
              f"{iv['precision']:>11.2f}{iv['recall']:>9.2f}{iv['count_err']:>9}{onset.precision:>12.2f}")
    print("=" * 82)
    biv, eiv = out["baseline(loop)"]["interval"], out["percept-event"]["interval"]
    print(f"  event-track vs raw loop (interval): fires {biv['n_fires']}->{eiv['n_fires']}, "
          f"precision {biv['precision']:.2f}->{eiv['precision']:.2f} "
          f"({eiv['precision']/max(0.01,biv['precision']):.1f}x), recall {biv['recall']:.2f}->{eiv['recall']:.2f}, "
          f"count-err {biv['count_err']}->{eiv['count_err']}")
    Path("e2e_result.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":  # pragma: no cover
    if os.environ.get("PERCEPT_LIVE") != "1":
        print("set PERCEPT_LIVE=1 + key."); raise SystemExit(0)
    asyncio.run(main())
