"""Legacy Charades dense/core replay diagnostic — do not run live for publishable benchmarking.

TML's Charades task: say "start" when an action begins and "stop" when it ends; graded on **mIoU** of
the predicted [start,stop] span vs the ground-truth interval. That is the metric directly comparable
to TML's reported **32.4** (vs **0** for GPT-realtime-2.0). Perception is held constant (Gemini 2.5
Flash, one judgment per frame, shared by all three systems). That dense capture is expensive and is no
longer the publishable Percept benchmark path. Three span-builders over the SAME verdicts:

  * gemini-loop  — legacy dense VLM loop: every contiguous run of confident-yes frames is a span (no memory)
  * percept-gate — the stateful Gate: rising-edge "start" -> confident-false re-arm "stop" = one span/episode
  * percept-event— Gate + EventLog merge_gap: merge yes-runs within the gap into one clean span (the dedup track)

mIoU = mean over ground-truth intervals of the best temporal IoU with any predicted span (unmatched GT = 0),
reported x100 to sit on TML's scale. Retain this script for offline/historical analysis only. The
publishable Charades adapter should run edge/E2E candidate calls only and model dense call counts
without sending media through a raw loop.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from percept.adapters import Verdict
from percept.config import load_settings
from percept.gate import FIRE, Gate, classify

from ._dotenv import load_dotenv
from .exp_charades import SEL, VID_DIR, condition, video_fps
from .exp_e2e import gemini_client, gemini_judge
from .exp_perception_test import extract_jpeg

DT = 1.0  # one sample per second (1 fps), so a single yes-frame covers ~1s
TML_MIOU = 32.4  # native interaction model; GPT-realtime-2.0 = 0


def _iou(a, b) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def _yes_runs(verdicts, thr, fresh_min):
    """Raw VLM loop: maximal contiguous runs of confident-yes frames -> spans."""
    spans, start, last = [], None, None
    for ts, sat, cf in verdicts:
        yes = classify(Verdict(satisfied=sat, confidence=cf), 1.0, thr, fresh_min) == "yes"
        if yes:
            if start is None:
                start = ts
            last = ts
        elif start is not None:
            spans.append([start, last + DT]); start = None
    if start is not None:
        spans.append([start, last + DT])
    return spans


def _gate_spans(gid, verdicts, thr):
    """Real stateful Gate: span = rising-edge FIRE -> next confident-false (re-arm)."""
    gate = Gate(threshold=thr); spans, cur, last = [], None, None
    for ts, sat, cf in verdicts:
        v = Verdict(satisfied=sat, confidence=cf); last = ts
        decision = gate.evaluate((gid,), v, freshness=1.0)
        cls = classify(v, 1.0, thr, gate.fresh_min)
        if decision == FIRE:
            cur = ts
        elif cls == "no" and cur is not None:
            spans.append([cur, ts]); cur = None
    if cur is not None:
        spans.append([cur, last + DT])
    return spans


def _merge(spans, gap):
    """Event-dedup track: merge yes-runs separated by < gap into one clean span."""
    out = []
    for s in spans:
        if out and s[0] - out[-1][1] <= gap:
            out[-1][1] = s[1]
        else:
            out.append(list(s))
    return out


def miou(spans_by_goal, gt_by_goal):
    ious, n_spans = [], 0
    for gid, gts in gt_by_goal.items():
        spans = spans_by_goal.get(gid, []); n_spans += len(spans)
        for g in gts:
            ious.append(max((_iou(g, s) for s in spans), default=0.0))
    return (sum(ious) / len(ious) if ious else 0.0), n_spans


async def main():
    load_dotenv()
    s = load_settings(); thr = s.gate.threshold; fresh_min = Gate(threshold=thr).fresh_min
    model = os.environ.get("PERCEPT_GEMINI_MODEL", "gemini-2.5-flash")
    merge_gap = float(os.environ.get("PERCEPT_EVENT_MERGE_GAP_S", 3.0))
    cap_path = Path("charades_gemini_verdicts.json")
    sel = json.load(open(SEL))

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

    # Resume-friendly capture: load whatever is already cached and only judge the rest, checkpointing after
    # every goal so progress is visible (the file grows) and a kill/crash is resumable. To force a full
    # recapture, set PERCEPT_RECAPTURE=1 or delete cap_path.
    recapture = os.environ.get("PERCEPT_RECAPTURE") == "1"
    caps = {} if (recapture or not cap_path.exists()) else json.loads(cap_path.read_text())
    todo = [g for g in goals if g["gid"] not in caps]
    if not todo:
        print(f"loaded {len(caps)} goals' Gemini verdicts (offline re-score)")
    else:
        if caps:
            print(f"resuming: {len(caps)} cached, {len(todo)} to capture", flush=True)
        client = gemini_client(); sem = asyncio.Semaphore(4); loop = asyncio.get_event_loop()
        for i, g in enumerate(todo, 1):
            cond = condition(g["desc"])
            ts_list = [t for t in range(0, int(g["length"]) + 1) if round(t * g["fps"]) < g["nframes"]]

            async def judge_t(t, g=g, cond=cond):
                frame = extract_jpeg(VID_DIR / f"{g['vid']}.mp4", round(t * g["fps"]))
                if frame is None:
                    return None
                try:
                    async with sem:
                        sat, cf = await loop.run_in_executor(None, gemini_judge, client, model, cond, frame)
                except Exception:
                    return None
                return [float(t), sat, cf]

            vs = [r for r in await asyncio.gather(*(judge_t(t) for t in ts_list)) if r]
            vs.sort(key=lambda x: x[0])
            caps[g["gid"]] = {"desc": g["desc"], "segments": g["segments"], "verdicts": vs}
            cap_path.write_text(json.dumps(caps))  # checkpoint per goal -> visible + resumable
            print(f"  captured {i}/{len(todo)} {g['gid']} ({len(caps)} cached)", flush=True)
        print(f"captured {len(caps)} goals (Gemini {model}) -> {cap_path}")

    gt = {gid: c["segments"] for gid, c in caps.items()}
    n_gt = sum(len(v) for v in gt.values())
    loop_spans, gate_spans, event_spans = {}, {}, {}
    for gid, c in caps.items():
        runs = _yes_runs(c["verdicts"], thr, fresh_min)
        loop_spans[gid] = runs
        gate_spans[gid] = _gate_spans(gid, c["verdicts"], thr)
        event_spans[gid] = _merge(runs, merge_gap)

    print("\n" + "=" * 78)
    print("  CHARADES mIoU — GEMINI FLASH perception · Percept vs raw-Gemini-loop vs TML")
    print(f"  {len(caps)} (video,action) goals · {n_gt} ground-truth intervals · model={model}")
    print("=" * 78)
    print(f"  {'system':<22}{'spans':>7}{'mIoU x100':>12}")
    out = {}
    for name, spans in [("gemini-loop (raw)", loop_spans), ("percept-gate", gate_spans),
                        ("percept-event (dedup)", event_spans)]:
        m, ns = miou(spans, gt)
        out[name] = {"miou": m, "n_spans": ns}
        print(f"  {name:<22}{ns:>7}{m * 100:>12.1f}")
    print(f"  {'TML (native model)':<22}{'—':>7}{TML_MIOU:>12.1f}")
    print(f"  {'GPT-realtime-2.0':<22}{'—':>7}{0.0:>12.1f}")
    print("=" * 78)
    lm = out["gemini-loop (raw)"]["miou"] * 100; em = out["percept-event (dedup)"]["miou"] * 100
    print(f"  cognition delta: gemini-loop {lm:.1f} -> percept-event {em:.1f} mIoU x100  "
          f"({em - lm:+.1f}); TML bar {TML_MIOU}")
    Path("charades_miou_result.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":  # pragma: no cover
    if os.environ.get("PERCEPT_LIVE") != "1":
        print("set PERCEPT_LIVE=1 + a Gemini key."); raise SystemExit(0)
    asyncio.run(main())
