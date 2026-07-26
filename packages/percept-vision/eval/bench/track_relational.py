"""RELATIONAL track — gate-decision quality on the golden gate-cases (the ASK-discipline set).

The question this track answers: does the three-state gate *refuse to guess* without paying it in
missed fires? We replay the approved golden gate-cases through the real vision judge once, then derive
two configs from the *same* per-frame verdicts (no extra calls):

- ``dense-modeled`` — naive diagnostic: fire if any frame says yes, else silent. Never ASK.
- ``core-replay`` — the actual gate (:func:`eval.scorecard._predict`): yes->fire / unknown->ask / silent.

P/R/F1 are strict over ``expected_verdict in {fire,silent}`` (ask/silent predictions count as not-fire);
``expected=="ask"`` cases are excluded from P/R. This is a core replay diagnostic, not product E2E.

``score(live=False)`` returns a clearly-fake offline stub (no API, no key) so the benchmark card still renders.
"""

from __future__ import annotations

import asyncio

from percept.gate import classify

from ..frames import frames_for_scene
from ..scorecard import _predict
from .types import TrackResult

TRACK = "relational"


def _judge_all(vision, condition, frames) -> list:
    """Judge a scene's frames concurrently (the judge is async) — one verdict per frame."""
    async def _go():
        sem = asyncio.Semaphore(6)

        async def one(f):
            async with sem:
                return await vision.judge(condition, f)

        return await asyncio.gather(*(one(f) for f in frames))

    return asyncio.run(_go())

_METRICS = ("precision", "recall", "f1", "ask_rate", "auroc")
_FMTS = {m: "{:.2f}" for m in _METRICS}


def _pr_f1(preds: list[str], expected: list[str]) -> tuple[float, float, float | None]:
    """Strict P/R/F1 over expected in {fire,silent}; ask/silent preds are NOT-fire; ask-expected excluded."""
    tp = fp = fn = 0
    for pred, exp in zip(preds, expected):
        if exp == "ask":
            continue  # excluded from P/R
        pred_fire = pred == "fire"
        if pred_fire and exp == "fire":
            tp += 1
        elif pred_fire and exp == "silent":
            fp += 1
        elif not pred_fire and exp == "fire":
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else None
    return precision, recall, f1


def _auroc(scores: list[float], positive: list[bool]) -> float | None:
    """Rank-based AUROC (Mann-Whitney) of per-case max-confidence vs (expected=="fire"). Pure python."""
    n_pos = sum(positive)
    n_neg = len(positive) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None  # degenerate
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):  # average ties
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # 1-based average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sum_ranks_pos = sum(r for r, p in zip(ranks, positive) if p)
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _live(n: int) -> TrackResult:
    from percept import registry
    from percept.config import load_settings
    from percept.eval import find_golden_root, load_golden

    settings = load_settings()
    thr = settings.gate.threshold
    root = find_golden_root()
    if root is None:
        raise SystemExit("no golden fixtures found")
    golden = load_golden(root)
    cases = [
        g
        for g in golden.gate_cases
        if g.get("review_status") == "approved"
        and (root / "scenes" / g["scene_id"] / "clip.json").exists()
    ]
    cases.sort(key=lambda g: (g.get("split") != "dev", g["scene_id"]))  # prefer dev split
    cases = cases[:n]

    backend = settings.models.vision if settings.models.vision != "fake" else "gemini"
    vision = registry.make_vision(backend, settings)

    raw_preds, core_preds, expected, scores = [], [], [], []
    calls = 0
    for g in cases:
        frames = frames_for_scene(root / "scenes" / g["scene_id"])
        if len(frames) > 6:  # cap at 6 via even subsample to bound cost
            step = len(frames) / 6
            frames = [frames[int(k * step)] for k in range(6)]
        condition = g["goal"]["condition"]
        verdicts = _judge_all(vision, condition, [frame for _, frame in frames])  # awaited (judge is async)
        calls += len(verdicts)
        raw_preds.append("fire" if any(classify(v, 1.0, thr, 0.5) == "yes" for v in verdicts) else "silent")
        core_preds.append(_predict(verdicts, thr))
        expected.append(g["expected_verdict"])
        scores.append(max((v.confidence for v in verdicts), default=0.0))

    n_cases = len(cases)
    rawP, rawR, rawF = _pr_f1(raw_preds, expected)
    coreP, coreR, coreF = _pr_f1(core_preds, expected)
    ask_rate = (sum(1 for p in core_preds if p == "ask") / n_cases) if n_cases else 0.0
    positive = [e == "fire" for e in expected]
    auroc = _auroc(scores, positive)

    cells: dict = {}
    measured: list[str] = []
    for cfg, (P, R, F) in (("dense-modeled", (rawP, rawR, rawF)), ("core-replay", (coreP, coreR, coreF))):
        for metric, val in (("precision", P), ("recall", R), ("f1", F)):
            cells[(cfg, metric)] = val
            measured.append(f"{cfg}:{metric}")
    for metric, val in (("ask_rate", ask_rate), ("auroc", auroc)):
        for cfg in ("dense-modeled", "core-replay"):
            cells[(cfg, metric)] = val
            measured.append(f"{cfg}:{metric}")

    return _result(cells, _FMTS, rawP, rawR, rawF, coreP, coreR, coreF, ask_rate, auroc,
                   measured, ["dense-modeled:precision", "dense-modeled:recall", "dense-modeled:f1"], calls, notes="live")


def _result(cells, fmts, rawP, rawR, rawF, coreP, coreR, coreF, ask_rate, auroc,
            measured, modeled, calls, notes):
    headline = {
        "relational P/R/F1 raw->core": (
            f"{rawP:.2f}/{rawR:.2f}/{(rawF if rawF is not None else 0.0):.2f} -> "
            f"{coreP:.2f}/{coreR:.2f}/{(coreF if coreF is not None else 0.0):.2f}"
        ),
        "relational ASK-rate (core)": f"{ask_rate:.2f}",
        "relational AUROC (core)": f"{(auroc if auroc is not None else 0.0):.2f}",
    }
    return TrackResult(
        track=TRACK, cells=cells, fmts=fmts, headline=headline,
        measured=measured, modeled=modeled, calls=calls, notes=notes,
        diagnostic=True,
        excluded=["relational: core replay diagnostic, not product E2E"],
    )


def _offline() -> TrackResult:
    """Clearly-fake PLACEHOLDER cells — no API, no key, no model call."""
    rawP, rawR, rawF = 0.55, 0.70, 0.62
    coreP, coreR, coreF = 0.72, 0.68, 0.70
    ask_rate, auroc = 0.25, 0.55
    cells: dict = {}
    measured: list[str] = []
    for cfg, (P, R, F) in (("dense-modeled", (rawP, rawR, rawF)), ("core-replay", (coreP, coreR, coreF))):
        for metric, val in (("precision", P), ("recall", R), ("f1", F)):
            cells[(cfg, metric)] = val
            measured.append(f"{cfg}:{metric}")
    for cfg in ("dense-modeled", "core-replay"):
        cells[(cfg, "ask_rate")] = ask_rate
        cells[(cfg, "auroc")] = auroc
        measured.append(f"{cfg}:ask_rate")
        measured.append(f"{cfg}:auroc")
    return _result(cells, _FMTS, rawP, rawR, rawF, coreP, coreR, coreF, ask_rate, auroc,
                   measured, ["dense-modeled:precision", "dense-modeled:recall", "dense-modeled:f1"], calls=0, notes="offline stub")


def score(*, live: bool, n: int) -> TrackResult:
    return _live(n) if live else _offline()
