"""Prevalence read on Perception Test (DeepMind, CC-BY) — clean, licensed, third-person action data.

Auto-derives gate-cases from the temporal action-localisation annotations: a frame at the MIDPOINT
of an action segment is a FIRE case for that action's condition; a frame from a video where the
action is absent is a SILENT case. Then judges each frame BOTH ways — the production numeric judge
AND the categorical {yes/no/unsure} judge — and reports false-fire / recall / ask on each, so we see
the prevalence of the overconfidence + the categorical fix on clean data (vs the 25 YouTube cases).

Needs the sample split downloaded to eval/bench-staging/perception_test/raw/ (sample.json +
sample_videos.zip extracted) and cv2 for frame extraction. Live (PERCEPT_LIVE=1 + key).
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import random
from pathlib import Path

import cv2  # frame extraction (opencv-python-headless)

from percept.config import load_settings
from percept.gate import classify

from ._dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("PERCEPT_PT_DIR", ROOT / "eval/bench-staging/perception_test/raw"))
VERDICTS = ("fire", "silent", "ask")

_CAT_SYSTEM = (
    "You are a perception module answering ONE narrow question about an image. Answer with exactly "
    "one category: YES the statement is clearly true; NO clearly false; UNSURE a genuine judgment "
    "call you cannot determine from this view. Prefer UNSURE when ambiguous/subjective. "
    'Return STRICT JSON: {"answer":"yes"|"no"|"unsure"}.'
)


def condition(label: str) -> str:
    return "someone is " + (label[0].lower() + label[1:] if label else label)


def _find_video(video_id: str) -> Path | None:
    for p in DATA.rglob(f"{video_id}.mp4"):
        return p
    return None


def extract_jpeg(video_path: Path, frame_id: int, max_dim: int = 768) -> tuple[str, bytes] | None:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_id))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    h, w = frame.shape[:2]
    m = max(h, w)
    if m > max_dim:  # downscale — full-res payloads time out, and the model resizes anyway
        s = max_dim / m
        frame = cv2.resize(frame, (int(w * s), int(h * s)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return ("image/jpeg", buf.tobytes()) if ok else None


# Specific, visually-distinct actions where a "this action is absent" negative is genuinely negative.
# Excludes Something-Something generic primitives ("moving objects around", "putting something into
# something", "holding something") that are near-universally true of any manipulation frame.
SPECIFIC = {
    "Clapping hands", "Cutting something with something", "Tearing something into pieces",
    "Pouring something from something into something", "Opening something",
    "Mixing something with something", "Hitting something against/with something",
}
USE_SPECIFIC = os.environ.get("PERCEPT_PT_SPECIFIC") == "1"


def build_cases(ann: dict, max_per_label_neg: int = 4) -> list[dict]:
    """FIRE = midpoint frame of each action segment; SILENT = a frame from a video lacking that label."""
    rng = random.Random(0)
    keep = (lambda lbl: lbl in SPECIFIC) if USE_SPECIFIC else (lambda lbl: True)
    present = {vid: {s["label"] for s in v.get("action_localisation", [])} for vid, v in ann.items()}
    cases = []
    # positives
    for vid, v in ann.items():
        for s in v.get("action_localisation", []):
            if not keep(s["label"]):
                continue
            f0, f1 = s["frame_ids"]
            cases.append({"video": vid, "frame": (f0 + f1) // 2, "label": s["label"], "expected": "fire"})
    # negatives: for each label, a midpoint frame from a video that does NOT contain it
    labels = {s["label"] for v in ann.values() for s in v.get("action_localisation", []) if keep(s["label"])}
    for label in labels:
        others = [vid for vid in ann if label not in present[vid]]
        rng.shuffle(others)
        for vid in others[:max_per_label_neg]:
            n = ann[vid]["metadata"]["num_frames"]
            cases.append({"video": vid, "frame": n // 2, "label": label, "expected": "silent"})
    return cases


async def _numeric(client, model, cond, frame):
    import base64 as b64
    media, raw = frame
    img = {"type": "image", "source": {"type": "base64", "media_type": media, "data": b64.b64encode(raw).decode()}}
    sys = ('Answer if the statement is true of the image. Return STRICT JSON '
           '{"satisfied": true|false, "confidence": 0..1}.')
    r = await client.messages.create(model=model, max_tokens=120, system=[{"type": "text", "text": sys}],
        messages=[{"role": "user", "content": [img, {"type": "text", "text": f'Statement: "{cond}". JSON.'}]}])
    t = "".join(getattr(b, "text", "") for b in r.content); s, e = t.find("{"), t.rfind("}")
    try:
        d = json.loads(t[s:e + 1]); return bool(d.get("satisfied")), float(d.get("confidence", 0.5))
    except Exception:
        return False, 0.5


async def _categorical(client, model, cond, frame):
    media, raw = frame
    img = {"type": "image", "source": {"type": "base64", "media_type": media, "data": base64.b64encode(raw).decode()}}
    r = await client.messages.create(model=model, max_tokens=80, system=[{"type": "text", "text": _CAT_SYSTEM}],
        messages=[{"role": "user", "content": [img, {"type": "text", "text": f'Statement: "{cond}". JSON.'}]}])
    t = "".join(getattr(b, "text", "") for b in r.content); s, e = t.find("{"), t.rfind("}")
    try:
        a = str(json.loads(t[s:e + 1]).get("answer", "")).lower()
    except Exception:
        a = ""
    return a if a in ("yes", "no", "unsure") else "unsure"


def _num_action(sat, conf, thr):
    t = classify(type("V", (), {"satisfied": sat, "confidence": conf})(), 1.0, thr, 0.5)
    return {"yes": "fire", "no": "silent", "unknown": "ask"}[t]


def _cat_action(ans):
    return {"yes": "fire", "no": "silent", "unsure": "ask"}[ans]


def _metrics(rows, key):
    conf = {e: {p: 0 for p in VERDICTS} for e in VERDICTS}
    for r in rows:
        conf[r["expected"]][r[key]] += 1
    exp = {e: sum(conf[e].values()) for e in VERDICTS}
    ff = conf["silent"]["fire"]
    return {"false_fire": ff, "silent_n": exp["silent"], "recall": (conf["fire"]["fire"], exp["fire"]),
            "ask": sum(conf[e]["ask"] for e in VERDICTS), "n": sum(exp.values()), "conf": conf}


def _fmt(m):
    r = m["recall"]
    return (f"false-fire={m['false_fire']}/{m['silent_n']}  recall={r[0]}/{r[1]}  "
            f"ask-total={m['ask']}/{m['n']}")


async def main_async() -> int:
    import anthropic
    load_dotenv()
    settings = load_settings(); model = settings.models.vision_model; thr = settings.gate.threshold
    ann = json.load(open(DATA / "sample.json"))
    cases = build_cases(ann)
    # keep only cases whose video file is present
    cases = [c for c in cases if _find_video(c["video"])]
    print(f"Perception Test sample: {len(ann)} videos, {len(cases)} auto gate-cases "
          f"({sum(c['expected']=='fire' for c in cases)} fire / {sum(c['expected']=='silent' for c in cases)} silent)")
    client = anthropic.AsyncAnthropic(timeout=90.0, max_retries=3); sem = asyncio.Semaphore(6)

    async def judge_one(c):
        frame = extract_jpeg(_find_video(c["video"]), c["frame"])
        if frame is None:
            return None
        cond = condition(c["label"])
        try:
            async with sem:
                sat, cf = await _numeric(client, model, cond, frame)
                cat = await _categorical(client, model, cond, frame)
        except Exception:
            return None  # one flaky call must not abort the batch
        c = dict(c); c["numeric"] = _num_action(sat, cf, thr); c["categorical"] = _cat_action(cat); c["conf"] = cf
        return c

    rows = [r for r in await asyncio.gather(*(judge_one(c) for c in cases)) if r]
    num, cat = _metrics(rows, "numeric"), _metrics(rows, "categorical")
    print("\n" + "=" * 64)
    print("  PREVALENCE on PERCEPTION TEST (DeepMind, CC-BY, third-person, CLEAN actions)")
    print("=" * 64)
    print(f"  NUMERIC judge:     {_fmt(num)}")
    print(f"  CATEGORICAL judge: {_fmt(cat)}")
    print(f"\n  confidence distribution (numeric): "
          f"min={min(r['conf'] for r in rows):.2f} mean={sum(r['conf'] for r in rows)/len(rows):.2f} "
          f"<0.7={sum(r['conf']<0.7 for r in rows)}/{len(rows)}")
    print("=" * 64)
    Path("perception_test_prevalence.json").write_text(json.dumps(
        {"numeric": _fmt(num), "categorical": _fmt(cat), "n": len(rows)}, indent=2))
    return 0


def main() -> int:
    if not (DATA / "sample.json").exists():
        print(f"missing {DATA}/sample.json — download the Perception Test sample first."); return 1
    if not os.environ.get("PERCEPT_LIVE") == "1":
        print("set PERCEPT_LIVE=1 + a key to run the live prevalence read."); return 0
    return asyncio.run(main_async())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
