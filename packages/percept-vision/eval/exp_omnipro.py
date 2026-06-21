"""OmniPro (visual-only slice) Online-Mode — Percept vs Gemini-loop, scored by OmniPro's OWN metric.

OmniPro (arXiv 2605.18577) is a proactive STREAMING benchmark: the model must autonomously decide WHEN
to speak. Online Mode = greedy 1-to-1 emit<->trigger matching within +/-3s -> time_F1 (+ content/joint).
That timing metric IS the gate's job: fire ONCE at each event vs the loop's fire-every-frame spam.

HONEST SCOPE: OmniPro is omni-modal and ~84% audio-dependent. Percept is vision-only, so we run ONLY the
`audio_dependency=none` (visually-determinable) slice — a 20-sample subset across 8 tasks. We are sitting
out the audio/speech/AV axes by construction. Perception = Gemini 2.5 Flash (not Gemini-3-Flash); scoring
imports OmniPro's official `metrics.online.scorer` (greedy ±3s match, MIT-licensed) so the metric is THEIR
implementation, not ours. Directional vs the published leaderboard, but the SCORER is faithful.

Two emit policies over the same per-frame Gemini detections:
  * gemini-loop — raw: emit at EVERY frame where the event is visible (no memory)
  * percept     — rising-edge + refractory (~tolerance): emit ONCE per event onset (the gate)
Live: PERCEPT_LIVE=1 + a Gemini key.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import cv2

from ._dotenv import load_dotenv
from .exp_e2e import gemini_client
from .exp_perception_test import extract_jpeg

ROOT = Path(__file__).resolve().parents[3]
SUB = ROOT / "eval/bench-staging/omnipro/raw/visual_subset.json"
VID = ROOT / "eval/bench-staging/omnipro/raw/videos"
CODE = ROOT / "eval/bench-staging/omnipro/code"
STEP = float(os.environ.get("PERCEPT_OMNIPRO_STEP_S", 2.0))
REFRACTORY_S = float(os.environ.get("PERCEPT_OMNIPRO_REFRACTORY_S", 3.0))
TOL = float(os.environ.get("PERCEPT_OMNIPRO_TOL", 3.0))
sys.path.insert(0, str(CODE))  # import OmniPro's official scorer

DET_SYS = ('You watch a streaming video to proactively detect an event. Given ONE frame and an event '
           'description, decide if that event is visibly occurring in THIS frame. Reply STRICT JSON '
           '{"happening":true|false,"note":"<one short clause>"}.')


def _json(t):
    s, e = t.find("{"), t.rfind("}")
    try:
        return json.loads(t[s:e + 1])
    except Exception:
        return {}


def gemini_detect(client, model, event, question, frame):
    from google.genai import types
    media, raw = frame
    r = client.models.generate_content(model=model, contents=[
        types.Part.from_bytes(data=raw, mime_type=media),
        f'{DET_SYS}\nEvent: "{event}"\nTask context: "{question}"\nJSON only.'])
    d = _json(r.text or "")
    return bool(d.get("happening")), str(d.get("note", "")).strip()


def fps_of(vid):
    cap = cv2.VideoCapture(str(VID / f"{vid}.mp4")); f = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0); cap.release()
    return f, n


def emits_loop(seq):
    return [{"t_sec": t, "raw": note} for t, on, note in seq if on]


def emits_percept(seq, refractory):
    out, last, prev = [], None, False
    for t, on, note in seq:
        if on and not prev and (last is None or t - last >= refractory):
            out.append({"t_sec": t, "raw": note}); last = t
        prev = on
    return out


async def main():
    load_dotenv()
    model = os.environ.get("PERCEPT_GEMINI_MODEL", "gemini-2.5-flash")
    sub = json.load(open(SUB))
    sub = [x for x in sub if (VID / f"{x['video_id']}.mp4").exists()]
    cap_path = ROOT / "omnipro_signals.json"

    if cap_path.exists() and os.environ.get("PERCEPT_RECAPTURE") != "1":
        caps = json.loads(cap_path.read_text())
        print(f"loaded {len(caps)} samples' detections (offline re-score)")
    else:
        client = gemini_client(); sem = asyncio.Semaphore(4); loop = asyncio.get_event_loop()
        caps = {}
        for x in sub:
            vid = x["video_id"]; fps, nfr = fps_of(vid)
            if nfr == 0:
                continue
            event = x.get("target") or x["question"]; q = x["question"]
            ts_list = [t for t in range(0, int(float(x["duration"])) + 1, int(STEP)) if round(t * fps) < nfr]

            async def det_t(t, vid=vid, fps=fps, event=event, q=q):
                frame = extract_jpeg(VID / f"{vid}.mp4", round(t * fps))
                if frame is None:
                    return None
                try:
                    async with sem:
                        on, note = await loop.run_in_executor(None, gemini_detect, client, model, event, q, frame)
                except Exception:
                    return None
                return [float(t), on, note]

            seq = [r for r in await asyncio.gather(*(det_t(t) for t in ts_list)) if r]
            seq.sort(key=lambda r: r[0])
            caps[x["id"]] = {"task": x["task"], "question": x["question"], "gt": x["_gt"], "seq": seq}
            print(f"  {x['id'][:42]:<44} task={x['task']:<22} frames={len(seq)} fires={sum(s[1] for s in seq)}")
        cap_path.write_text(json.dumps(caps))
        print(f"captured {len(caps)} samples (Gemini {model}) -> {cap_path}")

    # ---- score with OmniPro's OWN scorer ----
    from metrics.online.scorer import evaluate_sample, aggregate

    def score(policy):
        per = []
        for sid, c in caps.items():
            preds = emits_loop(c["seq"]) if policy == "loop" else emits_percept(c["seq"], REFRACTORY_S)
            per.append(evaluate_sample(
                {"id": sid, "task": c["task"], "question": c["question"],
                 "predictions": preds, "ground_truth": c["gt"]}, tolerance=TOL))
        return aggregate(per)

    agg_loop, agg_perc = score("loop"), score("percept")

    print("\n" + "=" * 86)
    print("  OmniPro Online-Mode (visual-only slice) — Percept vs Gemini-loop · OmniPro's OWN scorer")
    print(f"  {len(caps)} samples · {sum(len(c['gt']) for c in caps.values())} GT triggers · tol=±{TOL:.0f}s · model={model}")
    print("=" * 86)
    o_l, o_p = agg_loop["overall"], agg_perc["overall"]
    print(f"  {'system':<16}{'emits(TP/FP)':>14}{'time_P':>9}{'time_R':>9}{'time_F1':>9}{'joint_F1':>10}")
    for nm, o in [("gemini-loop", o_l), ("percept", o_p)]:
        emits = f"{o['tp_time']}/{o['fp']}"
        print(f"  {nm:<16}{emits:>14}{o['time_precision']:>9.3f}{o['time_recall']:>9.3f}"
              f"{o['time_f1']:>9.3f}{o['joint_f1']:>10.3f}")
    print("=" * 86)
    print("  OmniPro Online F1 leaderboard (full benchmark, omni-modal): MiniCPM-o 4.5 20.9 · "
          "MMDuet2 11.3 · LiveStar 3.6")
    print(f"  time_F1 x100: gemini-loop {o_l['time_f1']*100:.1f} -> percept {o_p['time_f1']*100:.1f} "
          f"(+{(o_p['time_f1']-o_l['time_f1'])*100:.1f}); the gate cuts FP {o_l['fp']}->{o_p['fp']}")
    print("  per-task time_F1 (loop -> percept):")
    for t in sorted(set(agg_loop) - {"overall"}):
        print(f"    {t:<26} {agg_loop[t]['time_f1']*100:>5.1f} -> {agg_perc[t]['time_f1']*100:.1f}")
    Path("omnipro_result.json").write_text(json.dumps({"loop": agg_loop, "percept": agg_perc}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    if os.environ.get("PERCEPT_LIVE") != "1":
        print("set PERCEPT_LIVE=1 + a Gemini key."); raise SystemExit(0)
    asyncio.run(main())
