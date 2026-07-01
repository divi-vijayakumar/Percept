"""ProactiveVideoQA (EGO subset) PAUC — core-replay diagnostic, not full Percept E2E.

ProactiveVideoQA: a question is posed at t=0; the model watches a streamed first-person video and must
proactively answer at the moment the answer becomes visible. Graded by **PAUC** — for each answer turn,
each in-window model response is judged (3-point: mostly/partial/incorrect -> 0..2, CUMULATIVE) against
the ground truth, the quality-over-time curve is integrated (omega timeliness weight), normalized, and
averaged. This is a FAITHFUL port of the official pauc/judge_llm_scores_openai.py math (omega=0.5,
max_score=2, start_score=0.5 -> a no-response turn scores 0.25 = TML's reported 25.0 baseline). TML's
native model = 33.5 @omega=0.5.

DIFFERENCES vs the official eval (so this is DIRECTIONAL, not an identical TML head-to-head): (1) the
judge is Claude Haiku, not GPT-4.1 (and is independent of the Gemini perceiver, to avoid self-grading);
(2) a 15-question single-turn EGO subset, not the full 326. Perception held constant at Gemini Flash.
Three response policies over the SAME per-frame narrations:
  * gemini-loop  — raw: emit the narration at EVERY frame where an answering step is visible (no memory)
  * percept-gate — emit once on each rising edge (first visible frame after a gap)
  * percept-event— rising edge + merge within merge_gap (the dedup track) -> one clean answer per episode
PAUC ignores out-of-window spam, so we also report INTERRUPTIONS (total responses emitted) — the user-
facing cost the metric doesn't price.

This runner is useful only for offline/historical ``core-replay`` analysis: it measures whether Percept
reduces repeated answers given identical backend observations. It does not measure E2E call suppression
because Gemini still narrates every sampled frame. Do not run it live for publishable benchmarking. The
publishable product lane should be ``edge-e2e``: edge wakes only candidate answer-evidence moments, the
backend answers candidates, and core emits sparse answers.
Live: PERCEPT_LIVE=1 + a Gemini key + ANTHROPIC_API_KEY (judge).
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import cv2

from ._dotenv import load_dotenv
from .exp_e2e import gemini_client
from .exp_perception_test import extract_jpeg

ROOT = Path(__file__).resolve().parents[3]
SUB = ROOT / "eval/bench-staging/proactiveqa/raw/EGO_subset.json"
VID = ROOT / "eval/bench-staging/proactiveqa/raw/EGO_videos"
STEP = float(os.environ.get("PERCEPT_PVQA_STEP_S", 5.0))
TML_PAUC = 33.5  # native model @omega=0.5; no-response baseline = 25.0

NARR_SYS = ("You watch a first-person video to proactively answer a question. Given ONE frame and the "
            "question, decide if a step that answers the question is visibly happening in THIS frame. "
            'Reply STRICT JSON {"happening":true|false,"step":"<one short sentence describing it, or empty>"}.')
JUDGE_SYS = (
    "You are an evaluator for a video question answering system. Your task is to rate whether the "
    "predicted answer covers the key points of the ground truth answer. Use the following scale:\n"
    "- 3: Mostly covered; the predicted answer covers all key information in the ground truth, though "
    "it may have minor inaccuracies or rephrases.\n"
    "- 2: Partially covered; some correct information, but significant inaccuracies or missing key points.\n"
    "- 1: Incorrect; most information missing or not relevant. Reply with ONLY the digit 1, 2, or 3.")


def _json(t):
    s, e = t.find("{"), t.rfind("}")
    try:
        return json.loads(t[s:e + 1])
    except Exception:
        return {}


def gemini_narrate(client, model, question, frame):
    from google.genai import types
    media, raw = frame
    r = client.models.generate_content(model=model, contents=[
        types.Part.from_bytes(data=raw, mime_type=media),
        f"{NARR_SYS}\nQuestion: {question}\nJSON only."])
    d = _json(r.text or "")
    return (str(d.get("step", "")).strip() if d.get("happening") else "")


def fps_of(name):
    cap = cv2.VideoCapture(str(VID / name)); f = cap.get(cv2.CAP_PROP_FPS) or 24.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0); cap.release()
    return f, n


# ---- faithful PAUC port (pauc/judge_llm_scores_openai.py) ----
def area_under_line_ratio(points, max_x, max_y, omega=0.5, start_score=0.5):
    if not points:
        return 0.0
    points = sorted(points, key=lambda p: p[0])
    points = [(x * (1 - omega), y) for x, y in points]
    points.append([max_x, points[-1][1]])
    prev_y, prev_x, area = start_score, 0.0, 0.0
    max_area = max_x * max_y
    for x1, y1 in points:
        area += (x1 - prev_x) * prev_y
        prev_x, prev_y = x1, y1
    return area / max_area if max_area else 0.0


def pauc(turns, omega=0.5, max_score=2, start_score=0.5):
    aucs = []
    for scores, span in turns:
        if not scores:
            aucs.append(start_score / max_score); continue
        pts = [[t - span[0], v] for t, v in scores.items()]
        aucs.append(area_under_line_ratio(pts, span[1] - span[0], max_score, omega, start_score))
    return sum(aucs) / len(aucs) if aucs else 0.0


def policies(narr, merge_gap):
    """narr: [[t, step_or_empty]] sorted -> dict of system -> [[t, content]]."""
    loop = [[t, s] for t, s in narr if s]
    gate, event, prev_on, last_emit = [], [], False, None
    for t, s in narr:
        on = bool(s)
        if on and not prev_on:               # rising edge
            gate.append([t, s])
            if last_emit is None or t - last_emit > merge_gap:
                event.append([t, s])
            last_emit = t
        elif on:
            last_emit = t
        prev_on = on
    return {"gemini-loop (raw)": loop, "percept-gate": gate, "percept-event (dedup)": event}


async def main():
    load_dotenv()
    merge_gap = float(os.environ.get("PERCEPT_EVENT_MERGE_GAP_S", 3.0))
    gmodel = os.environ.get("PERCEPT_GEMINI_MODEL", "gemini-2.5-flash")
    jmodel = os.environ.get("PERCEPT_JUDGE_MODEL", "claude-haiku-4-5-20251001")
    sub = json.load(open(SUB))
    sub = [x for x in sub if (VID / x["video"]).exists()]
    cap_path = ROOT / "proactiveqa_narrations.json"

    # Resume-friendly capture: load whatever is already cached and only narrate the rest, checkpointing
    # after every question so progress is visible (the file grows) and a kill/crash is resumable. To force
    # a full recapture, set PERCEPT_RECAPTURE=1 or delete cap_path.
    recapture = os.environ.get("PERCEPT_RECAPTURE") == "1"
    caps = {} if (recapture or not cap_path.exists()) else json.loads(cap_path.read_text())
    todo = [x for x in sub if x["question_id"] not in caps]
    if not todo:
        print(f"loaded {len(caps)} questions' narrations (offline re-score)")
    else:
        if caps:
            print(f"resuming: {len(caps)} cached, {len(todo)} to capture", flush=True)
        client = gemini_client(); sem = asyncio.Semaphore(4); loop = asyncio.get_event_loop()
        for i, x in enumerate(todo, 1):
            q = x["conversation"][0]["content"]; fps, nfr = fps_of(x["video"])
            if nfr == 0:
                continue
            ts_list = [t for t in range(0, int(x["duration"]) + 1, int(STEP)) if round(t * fps) < nfr]

            async def narrate_t(t, x=x, q=q, fps=fps):
                frame = extract_jpeg(VID / x["video"], round(t * fps))
                if frame is None:
                    return None
                try:
                    async with sem:
                        step = await loop.run_in_executor(None, gemini_narrate, client, gmodel, q, frame)
                except Exception:
                    return None
                return [float(t), step]

            narr = [r for r in await asyncio.gather(*(narrate_t(t) for t in ts_list)) if r]
            narr.sort(key=lambda r: r[0])
            caps[x["question_id"]] = {"question": q, "answer": x["answer"], "narr": narr}
            cap_path.write_text(json.dumps(caps))  # checkpoint per question -> visible + resumable
            print(f"  captured {i}/{len(todo)} {x['question_id']} ({len(caps)} cached)", flush=True)
        print(f"captured {len(caps)} questions (Gemini {gmodel}) -> {cap_path}")

    # ---- judge (Claude Haiku, independent of Gemini) with an in-run cache ----
    import anthropic
    jclient = anthropic.AsyncAnthropic(timeout=90.0, max_retries=3); jsem = asyncio.Semaphore(6)
    jcache: dict = {}

    async def judge(question, gt, pred):
        key = (question, gt, pred)
        if key in jcache:
            return jcache[key]
        msg = f"Question: {question}\nGround Truth Answer: {gt}\nPredicted Answer: {pred}"
        try:
            async with jsem:
                r = await jclient.messages.create(model=jmodel, max_tokens=5,
                    system=[{"type": "text", "text": JUDGE_SYS}],
                    messages=[{"role": "user", "content": msg}])
            txt = "".join(getattr(b, "text", "") for b in r.content).strip()
            score = max(0, min(2, int(txt[0]) - 1)) if txt and txt[0] in "123" else 0
        except Exception:
            score = 0
        jcache[key] = score
        return score

    sys_names = ["gemini-loop (raw)", "percept-gate", "percept-event (dedup)"]
    sys_turns: dict = {n: [] for n in sys_names}
    sys_interrupts: dict = {n: 0 for n in sys_names}

    for qid, c in caps.items():
        pol = policies(c["narr"], merge_gap)
        for name in sys_names:
            resp = pol[name]; sys_interrupts[name] += len(resp)
            for turn in c["answer"]:
                span = turn["reply_timespan"]; gt = turn["content"]
                inw = sorted([(t, s) for t, s in resp if span[0] <= t <= span[1]])
                scores, cum = {}, []
                for t, s in inw:
                    cum.append(s)
                    scores[t] = await judge(c["question"], gt, " ".join(cum))
                sys_turns[name].append((scores, span))

    n_turns = len(sys_turns[sys_names[0]])
    print("\n" + "=" * 80)
    print("  ProactiveVideoQA EGO subset — PAUC · GEMINI FLASH perception · judge=Claude Haiku")
    print(f"  {len(caps)} questions · {n_turns} answer turns · step={STEP:.0f}s · TML native=33.5 @w0.5")
    print("=" * 80)
    print(f"  {'system':<24}{'PAUC@0':>9}{'PAUC@0.5':>10}{'PAUC@1':>9}{'  interrupts(spam)':>19}")
    out = {}
    for name in sys_names:
        p0 = pauc(sys_turns[name], 0.0) * 100
        p5 = pauc(sys_turns[name], 0.5) * 100
        p1 = pauc(sys_turns[name], 1.0) * 100
        out[name] = {"pauc@0": p0, "pauc@0.5": p5, "pauc@1": p1, "interrupts": sys_interrupts[name]}
        print(f"  {name:<24}{p0:>9.1f}{p5:>10.1f}{p1:>9.1f}{sys_interrupts[name]:>19}")
    print(f"  {'TML (native model)':<24}{'—':>9}{TML_PAUC:>10.1f}{'—':>9}{'—':>19}")
    print(f"  {'no-response baseline':<24}{25.0:>9.1f}{25.0:>10.1f}{25.0:>9.1f}{0:>19}")
    print("=" * 80)
    lp = out["gemini-loop (raw)"]["pauc@0.5"]; ep = out["percept-event (dedup)"]["pauc@0.5"]
    li = out["gemini-loop (raw)"]["interrupts"]; ei = out["percept-event (dedup)"]["interrupts"]
    print(f"  @0.5: gemini-loop {lp:.1f} ({li} interruptions) vs percept-event {ep:.1f} ({ei}); TML 33.5")
    print(f"  PAUC prices content-in-window, NOT spam: Percept cuts interruptions {li}->{ei} "
          f"({li/max(1,ei):.0f}x) at ~similar PAUC.")
    Path("proactiveqa_result.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":  # pragma: no cover
    if os.environ.get("PERCEPT_LIVE") != "1":
        print("set PERCEPT_LIVE=1 + Gemini key + ANTHROPIC_API_KEY."); raise SystemExit(0)
    asyncio.run(main())
