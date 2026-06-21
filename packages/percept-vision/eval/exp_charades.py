"""E2E on CHARADES — the spec's temporal-localization axis, exhaustively labeled (the clean version).

Charades fixes what Perception Test couldn't: discrete daily activities ("drinking from a cup",
"holding a broom") with EXHAUSTIVE start/end intervals (consensus annotation). So a fire inside an
interval is genuinely correct and a fire outside is genuinely a false-fire. Same harness as
``exp_e2e`` — full Percept (stateful gate + event dedup) vs a raw-VLM-in-a-loop baseline, perception
held constant — but on properly-shaped data. Perception = Haiku (reliable; the gate-vs-loop delta is
model-agnostic). Live: PERCEPT_LIVE=1 + ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path

import cv2

from percept.adapters import Verdict
from percept.config import load_settings
from percept.events import EventLog
from percept.gate import FIRE, Gate, classify

from ._dotenv import load_dotenv
from .exp_e2e import interval_score
from .exp_perception_test import extract_jpeg
from .replay import Edge, Fire, score

ROOT = Path(__file__).resolve().parents[3]
VID_DIR = ROOT / "eval/bench-staging/charades/raw/Charades_v1_480"
SEL = ROOT / "eval/bench-staging/charades/raw/selection.json"
NUM_SYS = ('Answer if the statement is true of the image. Return STRICT JSON '
           '{"satisfied":true|false,"confidence":0..1}.')


def _json(t):
    s, e = t.find("{"), t.rfind("}")
    try:
        return json.loads(t[s:e + 1])
    except Exception:
        return {}


async def haiku_judge(client, model, cond, frame):
    media, raw = frame
    img = {"type": "image", "source": {"type": "base64", "media_type": media, "data": base64.b64encode(raw).decode()}}
    r = await client.messages.create(model=model, max_tokens=120, system=[{"type": "text", "text": NUM_SYS}],
        messages=[{"role": "user", "content": [img, {"type": "text", "text": f'Statement: "{cond}". JSON.'}]}])
    d = _json("".join(getattr(b, "text", "") for b in r.content))
    return bool(d.get("satisfied")), float(d.get("confidence", 0.5))


def video_fps(vid):
    cap = cv2.VideoCapture(str(VID_DIR / f"{vid}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return fps, n


def condition(desc):
    return "someone is " + (desc[0].lower() + desc[1:])


async def main():
    load_dotenv()
    s = load_settings(); thr = s.gate.threshold
    model = os.environ.get("PERCEPT_E2E_MODEL", "claude-haiku-4-5-20251001")
    omega = float(os.environ.get("PERCEPT_EVAL_OMEGA", 2.0))
    merge_gap = float(os.environ.get("PERCEPT_EVENT_MERGE_GAP_S", 3.0))
    cap_path = ROOT / "charades_verdicts.json"
    sel = json.load(open(SEL))

    # goals: (video, action-class) -> intervals; sample frames at ~1 fps by time
    goals = []
    for v in sel:
        vid = v["id"]; fps, nframes = video_fps(vid)
        if nframes == 0:
            continue
        by_desc = {}
        for desc, st, en in v["actions"]:
            by_desc.setdefault(desc, []).append([st, en])
        for desc, segs in by_desc.items():
            goals.append({"gid": f"{vid}:{desc}", "vid": vid, "desc": desc, "fps": fps,
                          "nframes": nframes, "length": v["length"], "segments": segs})

    if cap_path.exists() and os.environ.get("PERCEPT_E2E_RECAPTURE") != "1":
        caps = json.loads(cap_path.read_text())
        print(f"loaded {len(caps)} goals' verdicts (offline re-score)")
    else:
        import anthropic
        client = anthropic.AsyncAnthropic(timeout=90.0, max_retries=3); sem = asyncio.Semaphore(6)
        caps = {}
        for g in goals:
            cond = condition(g["desc"])
            ts_list = [t for t in range(0, int(g["length"]) + 1) if round(t * g["fps"]) < g["nframes"]]

            async def judge_t(t, g=g, cond=cond):
                frame = extract_jpeg(VID_DIR / f"{g['vid']}.mp4", round(t * g["fps"]))
                if frame is None:
                    return None
                try:
                    async with sem:
                        sat, cf = await haiku_judge(client, model, cond, frame)
                except Exception:
                    return None
                return [float(t), sat, cf]

            vs = [r for r in await asyncio.gather(*(judge_t(t) for t in ts_list)) if r]
            vs.sort(key=lambda x: x[0])
            caps[g["gid"]] = {"desc": g["desc"], "segments": g["segments"], "verdicts": vs}
        cap_path.write_text(json.dumps(caps))
        print(f"captured {len(caps)} goals -> {cap_path}")

    n_seg = sum(len(c["segments"]) for c in caps.values())
    print(f"CHARADES e2e: {len(caps)} (video,action) goals · {n_seg} labeled intervals · model={model}")

    base_fires, perc_fires, event_fires, edges, seg_by_goal = [], [], [], [], {}
    for gid, c in caps.items():
        seg_by_goal[gid] = c["segments"]
        for onset, _off in c["segments"]:
            edges.append(Edge(gid, onset))
        gate = Gate(threshold=thr); elog = EventLog(merge_gap_s=merge_gap)
        for ts, sat, cf in c["verdicts"]:
            v = Verdict(satisfied=sat, confidence=cf)
            if classify(v, 1.0, thr, gate.fresh_min) == "yes":
                base_fires.append(Fire(gid, ts, cf))
                _ev, is_new = elog.observe(c["desc"], ts, confidence=cf)
                if is_new:
                    event_fires.append(Fire(gid, ts, cf))
            if gate.evaluate((gid,), v, freshness=1.0) == FIRE:
                perc_fires.append(Fire(gid, ts, cf))

    print("\n" + "=" * 84)
    print("  E2E on CHARADES (exhaustive labels) — PERCEPT vs RAW-VLM-IN-A-LOOP")
    print(f"  {len(edges)} labeled action intervals · INTERVAL scoring (fire inside [start,end] = correct)")
    print("=" * 84)
    print(f"  {'system':<14}{'fires':>7}{'TP':>5}{'FP':>5}{'FN':>5}{'precision':>11}{'recall':>9}{'cnt-err':>9}{'  onset-prec':>12}")
    out = {}
    for name, fires in [("baseline(loop)", base_fires), ("percept-watch", perc_fires), ("percept-event", event_fires)]:
        iv = interval_score(fires, seg_by_goal)
        on = score(fires, edges, omega)
        out[name] = {"interval": iv, "onset_precision": on.precision, "pauc": on.pauc}
        print(f"  {name:<14}{iv['n_fires']:>7}{iv['tp']:>5}{iv['fp']:>5}{iv['fn']:>5}"
              f"{iv['precision']:>11.2f}{iv['recall']:>9.2f}{iv['count_err']:>9}{on.precision:>12.2f}")
    print("=" * 84)
    biv, eiv = out["baseline(loop)"]["interval"], out["percept-event"]["interval"]
    print(f"  event-track vs raw loop: fires {biv['n_fires']}->{eiv['n_fires']}, "
          f"precision {biv['precision']:.2f}->{eiv['precision']:.2f} "
          f"({eiv['precision']/max(0.01,biv['precision']):.1f}x), recall {biv['recall']:.2f}->{eiv['recall']:.2f}, "
          f"count-err {biv['count_err']}->{eiv['count_err']}")
    Path("charades_result.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":  # pragma: no cover
    if os.environ.get("PERCEPT_LIVE") != "1":
        print("set PERCEPT_LIVE=1 + key."); raise SystemExit(0)
    asyncio.run(main())
