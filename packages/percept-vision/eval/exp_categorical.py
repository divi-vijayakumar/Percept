"""Experiment: a categorical-ambiguity judge to revive the dead ASK valve (see plan).

The numeric-confidence judge pins ~0.9 on everything, so the gate's ``conf < threshold -> ASK``
path is structurally dead (0.2% of frames ever dip below 0.70). Hypothesis: an LLM expresses
*categorical* ambiguity ("genuinely can't tell") far better than a calibrated number. This runner
asks the model for ``{yes, no, unsure}`` per frame and maps ``unsure -> ASK``, then evaluates a few
per-scene aggregation rules against the human gate labels — train to pick, dev/test to report.

Live capture (``PERCEPT_LIVE=1`` + key), offline re-evaluation from the saved capture. Reuses the
golden loader + frame extractor; does NOT touch the production AnthropicVision (experiment first).
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from collections import Counter
from pathlib import Path

from percept.config import load_settings
from percept.eval import find_golden_root, load_golden

from ._dotenv import load_dotenv
from .frames import frames_for_scene

VERDICTS = ("fire", "silent", "ask")

_CAT_SYSTEM = (
    "You are a perception module answering ONE narrow question about a cropped image. Answer with "
    "exactly one category: YES = the statement is clearly true in the image; NO = clearly false; "
    "UNSURE = a genuine judgment call you cannot determine from this view (reasonable people would "
    "disagree) — NOT merely low resolution. Prefer UNSURE over guessing when the situation is "
    'ambiguous or subjective. Return STRICT JSON: {"answer":"yes"|"no"|"unsure","evidence":"<short>"}.'
)


def _image_block(frame) -> dict:
    media_type, raw = frame  # frames_for_scene yields (media_type, bytes)
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": base64.b64encode(raw).decode()},
    }


def _parse(text: str) -> str:
    s, e = text.find("{"), text.rfind("}")
    try:
        ans = str(json.loads(text[s : e + 1]).get("answer", "")).lower().strip()
    except Exception:
        ans = ""
    return ans if ans in ("yes", "no", "unsure") else "unsure"


async def _capture(concurrency: int = 8) -> list[dict]:
    import anthropic  # lazy (L2 boundary)

    settings = load_settings()
    model = settings.models.vision_model
    client = anthropic.AsyncAnthropic()
    root = find_golden_root()
    golden = load_golden(root)
    cases = [
        g
        for g in golden.gate_cases
        if g.get("review_status") == "approved"
        and (root / "scenes" / g["scene_id"] / "clip.json").exists()
    ]
    sem = asyncio.Semaphore(concurrency)

    async def judge(condition, frame) -> str:
        async with sem:
            resp = await client.messages.create(
                model=model,
                max_tokens=200,
                system=[{"type": "text", "text": _CAT_SYSTEM, "cache_control": {"type": "ephemeral"}}],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            _image_block(frame),
                            {"type": "text", "text": f'Statement: "{condition}". Answer JSON.'},
                        ],
                    }
                ],
            )
            return _parse("".join(getattr(b, "text", "") for b in resp.content))

    out = []
    for g in cases:
        frames = frames_for_scene(root / "scenes" / g["scene_id"])
        answers = await asyncio.gather(*(judge(g["goal"]["condition"], f) for _, f in frames))
        out.append({"scene_id": g["scene_id"], "split": g.get("split"),
                    "expected": g["expected_verdict"], "answers": list(answers)})
    return out


# -- per-scene aggregation rules: counts {yes,no,unsure} -> action ----------
def rule_unsure_first(c):
    if c["unsure"]:
        return "ask"
    return "fire" if c["yes"] else "silent"


def rule_yes_first(c):
    if c["yes"]:
        return "fire"
    return "ask" if c["unsure"] else "silent"


def rule_plurality(c):
    top = max(("yes", "no", "unsure"), key=lambda k: c[k])
    return {"yes": "fire", "no": "silent", "unsure": "ask"}[top]


RULES = {"unsure_first": rule_unsure_first, "yes_first": rule_yes_first, "plurality": rule_plurality}


def _counts(answers) -> dict:
    c = Counter(answers)
    return {"yes": c["yes"], "no": c["no"], "unsure": c["unsure"]}


def _evaluate(cases, rule) -> dict:
    conf = {e: {p: 0 for p in VERDICTS} for e in VERDICTS}
    for c in cases:
        conf[c["expected"]][rule(_counts(c["answers"]))] += 1
    exp = {e: sum(conf[e].values()) for e in VERDICTS}
    return {
        "false_fire": conf["silent"]["fire"],
        "fire_recall": (conf["fire"]["fire"], exp["fire"]),
        "ask_acc": (conf["ask"]["ask"], exp["ask"]),
        "agreement": sum(conf[v][v] for v in VERDICTS),
        "n": sum(exp.values()),
        "confusion": conf,
    }


def _fmt(m):
    fr, ar = m["fire_recall"], m["ask_acc"]
    return (f"false-fire={m['false_fire']}  fire-recall={fr[0]}/{fr[1]}  "
            f"ask={ar[0]}/{ar[1]}  agree={m['agreement']}/{m['n']}")


def main() -> int:
    load_dotenv()
    root = find_golden_root()
    cap_path = Path(os.environ.get("PERCEPT_CAT_CAPTURE", root / "categorical_capture.json"))
    live = os.environ.get("PERCEPT_LIVE") == "1"

    if live or not cap_path.exists():
        if not live:
            print("exp-categorical: no capture and PERCEPT_LIVE!=1 — set PERCEPT_LIVE=1 + a key.")
            return 0
        cases = asyncio.run(_capture())
        cap_path.write_text(json.dumps(cases))
        print(f"captured {len(cases)} cases (categorical) -> {cap_path}")
    else:
        cases = json.loads(cap_path.read_text())
        print(f"loaded {len(cases)} cases from {cap_path}")

    by = {s: [c for c in cases if c["split"] == s] for s in ("train", "dev", "test")}
    print("\n" + "=" * 70)
    print("  CATEGORICAL-AMBIGUITY JUDGE  (yes/no/unsure -> fire/silent/ASK)")
    print("  baseline numeric judge: ask-accuracy 0/5, false-fire 3/10 (whole eval)")
    print("=" * 70)
    # pick the rule: zero false-fire if possible, then best agreement, on TRAIN
    scored = {name: _evaluate(by["train"], r) for name, r in RULES.items()}
    for name, m in scored.items():
        print(f"  rule {name:<12} TRAIN: {_fmt(m)}")
    best = min(scored, key=lambda n: (scored[n]["false_fire"], -scored[n]["agreement"]))
    print(f"\n  CHOSEN rule (zero-false-fire then best-agreement on train): {best}\n")

    for split in ("train", "dev", "test"):
        if by[split]:
            print(f"  {split.upper():5}  {_fmt(_evaluate(by[split], RULES[best]))}")
    overall = _evaluate(cases, RULES[best])
    c = overall["confusion"]
    print("\n  CONFUSION @ chosen (rows=truth, cols=pred)   fire  silent  ask")
    for e in VERDICTS:
        print(f"    {e:<8}                               {c[e]['fire']:>4} {c[e]['silent']:>7} {c[e]['ask']:>5}")
    print("\n" + "=" * 70)

    out = Path(os.environ.get("PERCEPT_CAT_OUT", "exp_categorical.json"))
    out.write_text(json.dumps({"rule": best, "by_split": {s: _evaluate(by[s], RULES[best]) for s in by if by[s]}}, indent=2, default=str))
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
