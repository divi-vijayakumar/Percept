"""A/B the underlying vision model — does overconfidence / the dead-ASK valve shift with the model?

Runs the SAME Perception Test specific-action cases through several models (Claude Haiku, Claude
Opus, and Gemini if a key is present), each with the numeric AND categorical judge, and compares
false-fire / recall / ask / the confidence distribution. Tests whether the ~0.9-constant confidence
is a Haiku artifact or universal across model strength + vendor.

Live: PERCEPT_LIVE=1 + ANTHROPIC_API_KEY (Haiku+Opus); + GEMINI_API_KEY for the Gemini column.
Run: PERCEPT_LIVE=1 PYTHONPATH=packages/percept-vision <py3.12> -m eval.exp_model_ab
"""

from __future__ import annotations

import asyncio
import base64
import json
import os

from percept.config import load_settings
from percept.gate import classify

from ._dotenv import load_dotenv
from .exp_perception_test import DATA, SPECIFIC, condition, extract_jpeg, _find_video

VERDICTS = ("fire", "silent", "ask")

NUM_SYS = ('Answer if the statement is true of the image. Return STRICT JSON '
           '{"satisfied":true|false,"confidence":0..1,"evidence":"<short>"}.')
CAT_SYS = ('Answer one category: YES clearly true, NO clearly false, UNSURE a genuine judgment call '
           'you cannot determine. Prefer UNSURE if ambiguous/subjective. Return STRICT JSON '
           '{"answer":"yes"|"no"|"unsure","evidence":"<short>"}.')


def _json(t):
    s, e = t.find("{"), t.rfind("}")
    try:
        return json.loads(t[s:e + 1])
    except Exception:
        return {}


def specific_cases(ann, max_neg=4):
    import random
    rng = random.Random(0)
    present = {v: {s["label"] for s in a.get("action_localisation", [])} for v, a in ann.items()}
    cases = []
    for vid, a in ann.items():
        for s in a.get("action_localisation", []):
            if s["label"] in SPECIFIC:
                f0, f1 = s["frame_ids"]
                cases.append({"video": vid, "frame": (f0 + f1) // 2, "label": s["label"], "expected": "fire"})
    labels = {s["label"] for a in ann.values() for s in a.get("action_localisation", []) if s["label"] in SPECIFIC}
    for label in labels:
        others = [v for v in ann if label not in present[v]]
        rng.shuffle(others)
        for v in others[:max_neg]:
            cases.append({"video": v, "frame": ann[v]["metadata"]["num_frames"] // 2,
                          "label": label, "expected": "silent"})
    return cases


# ---- provider judges: return (numeric_dict, categorical_dict) ----
async def _anthropic_judge(client, model, cond, frame):
    media, raw = frame  # extract_jpeg yields (media_type, bytes)
    img = {"type": "image", "source": {"type": "base64", "media_type": media,
           "data": base64.b64encode(raw).decode()}}

    async def call(sysmsg):
        r = await client.messages.create(model=model, max_tokens=200,
            system=[{"type": "text", "text": sysmsg}],
            messages=[{"role": "user", "content": [img, {"type": "text", "text": f'Statement: "{cond}". JSON.'}]}])
        return _json("".join(getattr(b, "text", "") for b in r.content))

    return await call(NUM_SYS), await call(CAT_SYS)


def _gemini_call(client, model, cond, frame, sysmsg):
    from google.genai import types
    media, raw = frame
    r = client.models.generate_content(
        model=model,
        contents=[types.Part.from_bytes(data=raw, mime_type=media),
                  f"{sysmsg}\nStatement: \"{cond}\". JSON only."],
    )
    return _json(r.text or "")


async def _gemini_judge(client, model, cond, jpeg):
    loop = asyncio.get_event_loop()
    n = await loop.run_in_executor(None, _gemini_call, client, model, cond, jpeg, NUM_SYS)
    c = await loop.run_in_executor(None, _gemini_call, client, model, cond, jpeg, CAT_SYS)
    return n, c


def num_action(n, thr):
    v = type("V", (), {"satisfied": bool(n.get("satisfied")), "confidence": float(n.get("confidence", 0.5))})()
    return {"yes": "fire", "no": "silent", "unknown": "ask"}[classify(v, 1.0, thr, 0.5)]


def metrics(rows, key):
    conf = {e: {p: 0 for p in VERDICTS} for e in VERDICTS}
    for r in rows:
        conf[r["expected"]][r[key]] += 1
    exp = {e: sum(conf[e].values()) for e in VERDICTS}
    return {"ff": conf["silent"]["fire"], "sil": exp["silent"],
            "rec": (conf["fire"]["fire"], exp["fire"]),
            "ask": sum(conf[e]["ask"] for e in VERDICTS), "n": sum(exp.values())}


async def run_model(name, provider, model, cases, thr):
    confs = []  # numeric confidence values
    rows = []
    if provider == "anthropic":
        import anthropic
        client = anthropic.AsyncAnthropic(timeout=120.0, max_retries=3)
        sem = asyncio.Semaphore(4)

        async def one(c):
            jpeg = extract_jpeg(_find_video(c["video"]), c["frame"])
            if jpeg is None:
                return None
            try:
                async with sem:
                    n, cat = await _anthropic_judge(client, model, condition(c["label"]), jpeg)
            except Exception:
                return None
            return _row(c, n, cat, thr)

        results = await asyncio.gather(*(one(c) for c in cases))
    else:  # gemini (sync SDK; run sequentially-ish via executor)
        from google import genai
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        results = []
        for c in cases:
            jpeg = extract_jpeg(_find_video(c["video"]), c["frame"])
            if jpeg is None:
                results.append(None); continue
            try:
                n, cat = await _gemini_judge(client, model, condition(c["label"]), jpeg)
                results.append(_row(c, n, cat, thr))
            except Exception:
                results.append(None)
    rows = [r for r in results if r]
    confs = [r["num_conf"] for r in rows]
    return name, rows, confs


def _row(c, n, cat, thr):
    na = num_action(n, thr)
    ca = {"yes": "fire", "no": "silent", "unsure": "ask"}.get(str(cat.get("answer", "")).lower(), "ask")
    return {**c, "numeric": na, "categorical": ca, "num_conf": float(n.get("confidence", 0.5)),
            "num_ev": n.get("evidence", ""), "cat_ev": cat.get("evidence", "")}


async def main():
    load_dotenv()
    thr = load_settings().gate.threshold
    ann = json.load(open(DATA / "sample.json"))
    cases = [c for c in specific_cases(ann) if _find_video(c["video"])]
    models = [("haiku", "anthropic", "claude-haiku-4-5-20251001"),
              ("opus", "anthropic", "claude-opus-4-8")]
    if os.environ.get("GEMINI_API_KEY"):
        try:
            import google.genai  # noqa: F401
            models.append(("gemini", "gemini", os.environ.get("PERCEPT_GEMINI_MODEL", "gemini-2.5-flash")))
        except ImportError:
            print("GEMINI_API_KEY set but google-genai SDK not installed — skipping Gemini.")
    print(f"A/B over {len(cases)} cases × {len(models)} models: {[m[0] for m in models]}")

    out = {}
    for name, provider, model in models:
        print(f"  judging {name} ({model})...")
        nm, rows, confs = await run_model(name, provider, model, cases, thr)
        out[nm] = rows

    print("\n" + "=" * 78)
    print("  MODEL A/B — Perception Test specific actions (same cases)")
    print("=" * 78)
    print(f"  {'model':<8} {'NUMERIC false-fire / recall / ask':<38} {'confidence min/mean/<0.7':<24}")
    for name in out:
        rows = out[name]
        m = metrics(rows, "numeric"); mc = metrics(rows, "categorical")
        confs = [r["num_conf"] for r in rows]
        cd = f"{min(confs):.2f} / {sum(confs)/len(confs):.2f} / {sum(c<0.7 for c in confs)}/{len(confs)}" if confs else "—"
        print(f"  {name:<8} ff={m['ff']}/{m['sil']}  rec={m['rec'][0]}/{m['rec'][1]}  ask={m['ask']}/{m['n']}"
              f"   |  cat: ff={mc['ff']}/{mc['sil']} ask={mc['ask']}/{mc['n']}   |  {cd}")
    print("=" * 78)
    os.makedirs(os.path.join(os.getcwd(), "scratch"), exist_ok=True) if os.path.isdir("scratch") else None
    open("model_ab.json", "w").write(json.dumps(out, indent=2, default=str))
    print("  wrote model_ab.json (per-case per-model — for a multi-model inspector)")


if __name__ == "__main__":  # pragma: no cover
    if os.environ.get("PERCEPT_LIVE") != "1":
        print("set PERCEPT_LIVE=1 + key."); raise SystemExit(0)
    asyncio.run(main())
