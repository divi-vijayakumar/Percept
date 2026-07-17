"""Charades persistent-state G2 quality — the pre-registered N=20 run (QUALITY_GATE_PROTOCOL Addendum 2).

The heavyweight `charades-e2e` benchmark harness HUNG on a live socket send with no timeout (full-core
path). This runner applies the IDENTICAL pre-registered rule via the reliable direct-judge path:
- same data + labels: eval/bench-staging/charades/raw/{selection.json, Charades_v1_480/<id>.mp4}
- same condition text: "someone is <action>" (track's _condition, idempotent)
- same tolerance: a fire at t HITs a segment iff onset <= t <= end + WINDOW_PAD_S (1.0s)
- the REAL gate: percept.gate.Gate with require_false_first=True (the landed fix), driven per goal in
  time order (faithful rising-edge behaviour, not a reimplementation)
- N=20, live gemini-3.1-flash-lite, hard per-call timeout so no hung request stalls the run.
ONE frozen run; recall/precision reported as-is vs the 0.90 bar; this is the launch_profile number.
"""
from __future__ import annotations

import asyncio, base64, datetime as _dt, json, os, sys, time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "packages" / "percept-vision"))
import cv2  # noqa: E402
import yaml  # noqa: E402
from percept.backends.gemini import GeminiVision  # noqa: E402
from percept.config import load_settings  # noqa: E402
from percept.gate import Gate, FIRE  # noqa: E402

RAW = _REPO / "eval" / "bench-staging" / "charades" / "raw"
VID_DIR = RAW / "Charades_v1_480"
SEL = RAW / "selection.json"
CONFIG = _REPO / "eval" / "viability" / "reference_config.yaml"
RESULTS = _REPO / "eval" / "viability" / "results" / "charades"
N = 20
FPS = 1.0
WINDOW_PAD_S = 1.0
MAXDIM = 768
CALL_TIMEOUT_S = 25.0   # hard cap so a hung socket send can't stall the run
SEM = 8


def _load_key():
    for line in (_REPO / ".env").read_text().splitlines():
        if line.startswith("GOOGLE_AISTUDIO_API_KEY="):
            v = line.split("=", 1)[1].strip().strip('"').strip("'")
            for k in ("GOOGLE_AISTUDIO_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
                os.environ.setdefault(k, v)
            return v


def _condition(desc: str) -> str:
    text = (desc or "").strip() or "the annotated action"
    low = text.lower()
    for p in ("someone is ", "a person is ", "the person is ", "person is ", "a person "):
        if low.startswith(p):
            rest = text[len(p):].strip()
            return "someone is " + (rest[0].lower() + rest[1:] if rest else "doing the annotated action")
    return "someone is " + text[0].lower() + text[1:]


def _rows(n):
    rows = json.loads(SEL.read_text())
    return [r for r in rows if (VID_DIR / f"{r['id']}.mp4").exists()][:n]


def _goals(row):
    by_desc = {}
    for desc, start, end in row.get("actions", []):
        by_desc.setdefault(str(desc), []).append((float(start), float(end)))
    return [{"goal_id": f"charades-{row['id']}-{i}", "condition": _condition(d), "desc": d, "segments": segs}
            for i, (d, segs) in enumerate(sorted(by_desc.items()))]


def _frames(vid_id):
    cap = cv2.VideoCapture(str(VID_DIR / f"{vid_id}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    step = max(1, round(fps / FPS)); out = []; i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            h, w = fr.shape[:2]; s = min(1.0, MAXDIM / max(h, w))
            if s < 1.0:
                fr = cv2.resize(fr, (int(w * s), int(h * s)))
            ok2, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok2:
                out.append((round(i / fps, 2), buf.tobytes()))
        i += 1
    cap.release()
    return out


def _score(goals, fires_by_goal):
    tp = fp = fn = 0; lat = []
    for g in goals:
        gf = sorted(fires_by_goal.get(g["goal_id"], []))
        used = set()
        for t in gf:
            for si, (s, e) in enumerate(g["segments"]):
                if si not in used and s <= t <= e + WINDOW_PAD_S:
                    used.add(si); tp += 1; lat.append(t - s); break
        fp += len(gf) - len(used)
        fn += len(g["segments"]) - len(used)
    recall = tp / (tp + fn) if (tp + fn) else None
    precision = tp / (tp + fp) if (tp + fp) else None
    return tp, fp, fn, recall, precision, lat


async def run():
    cfg = yaml.safe_load(CONFIG.read_text())
    assert cfg.get("bars_locked") is True, "bars not locked"
    _load_key()
    os.environ["PERCEPT_VISION_MODEL"] = cfg["reference_config"]["provider_model_id"]
    os.environ["PERCEPT_VISION_BACKEND"] = "gemini"
    os.environ["PERCEPT_GATE_REQUIRE_FALSE_FIRST"] = "1"
    settings = load_settings()
    judge = GeminiVision(settings)
    model = str(getattr(judge, "_model", "?"))
    assert model.startswith("gemini"), model
    threshold = settings.gate.threshold

    rows = _rows(N)
    # build the full (video, goal, frame) work list
    work = []          # (row_id, goal, t, jpeg)
    all_goals = []
    frames_cache = {}
    for row in rows:
        gs = _goals(row)
        if not gs:
            continue
        frames = _frames(row["id"]); frames_cache[row["id"]] = frames
        for g in gs:
            all_goals.append(g)
            for t, jpg in frames:
                work.append((row["id"], g, t, jpg))
    print(f"videos={len(rows)} goals={len(all_goals)} judge_calls={len(work)}", flush=True)

    sem = asyncio.Semaphore(SEM)
    errors = 0

    async def judge_one(row_id, g, t, jpg):
        nonlocal errors
        async with sem:
            try:
                v = await asyncio.wait_for(judge.judge(g["condition"], ("image/jpeg", jpg)), CALL_TIMEOUT_S)
                return (g["goal_id"], t, bool(v.satisfied), float(v.confidence))
            except Exception:
                errors += 1
                return (g["goal_id"], t, None, 0.0)

    t0 = time.monotonic()
    results = await asyncio.gather(*[judge_one(*w) for w in work])
    elapsed = time.monotonic() - t0

    # group verdicts by goal, then drive the REAL gate per goal IN TIME ORDER -> fires
    verd = {}
    for goal_id, t, sat, conf in results:
        verd.setdefault(goal_id, []).append((t, sat, conf))
    gate = Gate(threshold=threshold, require_false_first=True)
    from percept.adapters import Verdict
    fires_by_goal = {}
    for g in all_goals:
        gid = g["goal_id"]
        for t, sat, conf in sorted(verd.get(gid, [])):
            if sat is None:
                continue  # errored/timed-out call: no verdict (does not fire)
            action = gate.evaluate(gid, Verdict(satisfied=sat, confidence=conf, evidence=""), now=t)
            if action == FIRE:
                fires_by_goal.setdefault(gid, []).append(t)

    # per-goal forensic trace: for each goal, segments (hit? + did the judge say yes IN-segment?),
    # all fires, false fires (fires in no segment window), and the judge yes-fraction over the clip.
    per_goal = []
    for g in all_goals:
        gid = g["goal_id"]; segs = g["segments"]
        vs = sorted(verd.get(gid, [])); fires = sorted(fires_by_goal.get(gid, []))
        seg_info = []
        for (s, e) in segs:
            hit = next((round(t, 1) for t in fires if s <= t <= e + WINDOW_PAD_S), None)
            yes_in = [round(t, 1) for (t, sat, _c) in vs if sat and s <= t <= e + WINDOW_PAD_S]
            seg_info.append({"seg": [round(s, 1), round(e, 1)], "hit_at": hit, "judge_yes_in_seg": yes_in})
        fp_fires = [round(t, 1) for t in fires if not any(s <= t <= e + WINDOW_PAD_S for (s, e) in segs)]
        yes_n = sum(1 for (_t, sat, _c) in vs if sat)
        per_goal.append({"goal_id": gid, "video": gid.split("-")[1], "condition": g["condition"],
                         "desc": g["desc"], "n_frames": len(vs), "judge_yes_frac": round(yes_n / len(vs), 2) if vs else 0.0,
                         "segments": seg_info, "fires": [round(t, 1) for t in fires], "false_fires": fp_fires})

    tp, fp, fn, recall, precision, lat = _score(all_goals, fires_by_goal)
    rmin = float(list(cfg["numeric_bars"]["recall_min"].values())[0])
    pmin = float(list(cfg["numeric_bars"]["precision_min"].values())[0])
    g2_pass = (recall is not None and precision is not None and recall >= rmin and precision >= pmin)
    band = "GREEN"  # G3 latency established GREEN (p95 ~1.2s) in Phase-B
    if g2_pass:
        deadline_class = "reflex"
    else:
        deadline_class = "UNDETERMINED — G2 quality below bar"

    return {
        "phase": "B-charades", "substrate": "Charades persistent-state actions (N=%d)" % len(rows),
        "model": model, "gate": "require_false_first=True", "tolerance_s": WINDOW_PAD_S, "fps": FPS,
        "videos": len(rows), "goals": len(all_goals), "judge_calls": len(work), "call_errors": errors,
        "elapsed_s": round(elapsed, 1),
        "tp": tp, "fp": fp, "fn": fn, "recall": recall, "precision": precision,
        "recall_min": rmin, "precision_min": pmin, "g2_pass": g2_pass,
        "median_latency_from_onset_s": round(sorted(lat)[len(lat)//2], 2) if lat else None,
        "launch_profile": {"deadline_class": deadline_class, "provider": "gemini", "model": model,
                           "g1": "PASS (architecture)", "g3_latency": band, "econ": "PASS", "g2_quality": "PASS" if g2_pass else "REVIEW"},
        "note": "Direct-judge runner applying the pre-registered charades rule (harness hung on a live "
                "socket send). Real percept.gate.Gate(require_false_first=True); tolerance [onset, end+1.0s].",
        "per_goal": per_goal,
    }


def _write(res):
    now = _dt.datetime.now(_dt.timezone.utc); rid = now.strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS / rid; out.mkdir(parents=True, exist_ok=True)
    per_goal = res.pop("per_goal", [])
    (out / "per_goal.json").write_text(json.dumps(per_goal, indent=1))
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    lp = res["launch_profile"]
    md = f"""# Charades persistent-state G2 quality — pre-registered N={res['videos']} (frozen)

**Model:** `{res['model']}` · gate `require_false_first=True` · tolerance `[onset, end+{res['tolerance_s']}s]` · {res['fps']} fps · run `{rid}`
**Calls:** {res['judge_calls']} ({res['call_errors']} errored/timed-out) in {res['elapsed_s']}s

## Result (as-is)
- **recall {round(res['recall'],3) if res['recall'] is not None else None}** (min {res['recall_min']}) · **precision {round(res['precision'],3) if res['precision'] is not None else None}** (min {res['precision_min']})
- tp {res['tp']} · fp {res['fp']} · fn {res['fn']} · median latency-from-onset {res['median_latency_from_onset_s']}s
- **G2 quality: {'PASS ✅' if res['g2_pass'] else 'REVIEW/NO-GO ⚠️'}**

## Emitted launch_profile
```yaml
launch_profile:
  deadline_class: {lp['deadline_class']}
  provider: gemini    model: {lp['model']}
  g1_boundedness: {lp['g1']}   g3_latency: {lp['g3_latency']}   economics: {lp['econ']}   g2_quality: {lp['g2_quality']}
  supported_camera: static
```

_{res['note']}_
"""
    (out / "report.md").write_text(md)
    return out


if __name__ == "__main__":
    res = asyncio.run(run())
    out = _write(res)
    print(f"Charades quality → {out}")
    print(f"  recall={round(res['recall'],3) if res['recall'] is not None else None} precision={round(res['precision'],3) if res['precision'] is not None else None} (bar {res['recall_min']}/{res['precision_min']})")
    print(f"  tp={res['tp']} fp={res['fp']} fn={res['fn']} errors={res['call_errors']}")
    print(f"  launch_profile.deadline_class = {res['launch_profile']['deadline_class']}")
