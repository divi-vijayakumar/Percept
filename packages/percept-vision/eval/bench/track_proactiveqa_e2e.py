"""ProactiveVideoQA edge/E2E benchmark.

This is the product-shaped lane for ProactiveVideoQA:

1. The user's question becomes a standing Percept watch goal.
2. The edge wakes only candidate answer-evidence moments.
3. A Gemini-backed vision adapter answers candidates or returns "not enough evidence".
4. The core emits sparse rising-edge answers; scoring compares those answers to reply windows.

The dense baseline is modeled only. No dense media-bearing VLM loop is executed.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from percept import Goal, Percept
from percept.adapters import Goal as ContractGoal, Verdict
from percept.config import load_settings
from percept.edge.tier0 import MotionGate, compile_watchspec
from percept.gate import FIRE
from percept.runtime.source import open_source

from .types import TrackResult

TRACK = "proactiveqa-e2e"
ROOT = Path(__file__).resolve().parents[4]
RAW = ROOT / "eval" / "bench-staging" / "proactiveqa" / "raw"
SUBSET = RAW / "EGO_subset.json"
VIDEO_DIR = RAW / "EGO_videos"
FPS = 0.2  # 5s cadence, matching the legacy PVQA diagnostic runner.
PVQA_WINDOW_S = 15.0  # clip window handed to the judge: a few 5s-spaced frames -> temporal evidence
TML_PAUC = 33.5

PVQA_SYSTEM = (
    "You are the perception backend for a proactive video assistant. A user asked a standing "
    "question at the start of the video. Looking at this short clip (a few frames in TIME ORDER, oldest "
    "first) and how the scene changes across them, decide whether there is now enough visual evidence to "
    "answer. Reply STRICT JSON "
    '{"sufficient": true|false, "answer": "<short answer or empty>", "confidence": 0..1}. '
    "If the clip does not contain enough evidence, set sufficient=false."
)


@dataclass
class AnswerTurn:
    content: str
    span: tuple[float, float]


@dataclass
class QuestionRun:
    question_id: str
    video: str
    question: str
    turns: list[AnswerTurn]
    frames: int = 0
    escalations: list[float] = field(default_factory=list)
    answers: list[tuple[float, str, float]] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    judge_calls: int = 0
    error: str | None = None


class PVQAVision:
    """Gemini-backed answer-evidence vision adapter."""

    def __init__(self, *, model: str | None = None) -> None:
        from google import genai

        import os

        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_AISTUDIO_API_KEY")
        self.client = genai.Client(api_key=key)
        self.model = model or os.environ.get("PERCEPT_VISION_MODEL") or os.environ.get("PERCEPT_GEMINI_MODEL") or "gemini-3-flash-preview"
        self.calls = 0
        self.last_verdict: dict[str, Any] | None = None

    async def detect(self, frame: Any, goals: list[ContractGoal]) -> list:
        return []

    async def detect_events(self, frames: list, goals: list[ContractGoal]) -> list:
        return []

    async def judge(self, condition: str, frame: Any, **kwargs: Any) -> Verdict:
        from google.genai import types

        self.calls += 1
        # Use the temporal clip when perceive_judged supplied one (goal authored window_s); the strip is
        # the earlier frames, current frame last (oldest->newest), matching the GeminiVision ordering.
        strip = [f for f in (kwargs.get("frames") or []) if f is not frame]
        clip = [*strip, frame]
        parts = [types.Part.from_bytes(data=raw, mime_type=media)
                 for media, raw in (_frame_bytes(f) for f in clip)]
        prompt = f"{PVQA_SYSTEM}\nStanding question/evidence condition: {condition}\n{len(clip)} frames, JSON only."
        resp = await self.client.aio.models.generate_content(
            model=self.model,
            contents=[*parts, prompt],
            config=types.GenerateContentConfig(max_output_tokens=256),
        )
        data = _json(resp.text or "")
        sufficient = bool(data.get("sufficient"))
        confidence = float(data.get("confidence", 0.5) or 0.5)
        answer = str(data.get("answer", "") or "").strip()
        if sufficient and not answer:
            sufficient = False
        self.last_verdict = {
            "condition": condition,
            "satisfied": sufficient,
            "confidence": confidence,
            "evidence": answer or "not enough evidence",
        }
        return Verdict(satisfied=sufficient, confidence=confidence, evidence=answer or "not enough evidence")


def _frame_bytes(frame: Any) -> tuple[str, bytes]:
    if isinstance(frame, tuple) and len(frame) == 2 and isinstance(frame[1], (bytes, bytearray)):
        return str(frame[0]), bytes(frame[1])
    if isinstance(frame, (bytes, bytearray)):
        return "image/jpeg", bytes(frame)
    if isinstance(frame, dict) and frame.get("type") == "image":
        src = frame.get("source", {})
        return src.get("media_type", "image/jpeg"), base64.b64decode(src.get("data", ""))
    raise TypeError(f"unsupported frame shape {type(frame).__name__}")


def _json(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return {}


def _question(row: dict[str, Any]) -> str:
    if row.get("question"):
        return str(row["question"])
    conv = row.get("conversation") or []
    if conv and conv[0].get("content"):
        return str(conv[0]["content"])
    return "What should I say next?"


def _turns(row: dict[str, Any]) -> list[AnswerTurn]:
    out: list[AnswerTurn] = []
    for item in row.get("answer", []):
        span = item.get("reply_timespan") or item.get("related_timespan")
        if not span or len(span) != 2:
            continue
        out.append(AnswerTurn(content=str(item.get("content", "")), span=(float(span[0]), float(span[1]))))
    return out


def _load_rows(n: int) -> list[dict[str, Any]]:
    if not SUBSET.exists():
        return []
    rows = json.loads(SUBSET.read_text())
    return [row for row in rows[:n] if (VIDEO_DIR / row["video"]).exists()]


def _token_score(pred: str, gt: str) -> int:
    pred_tokens = set(re.findall(r"[a-z0-9]+", pred.lower()))
    gt_tokens = set(re.findall(r"[a-z0-9]+", gt.lower()))
    if not gt_tokens or not pred_tokens:
        return 0
    overlap = len(pred_tokens & gt_tokens) / len(gt_tokens)
    if overlap >= 0.65:
        return 2
    if overlap >= 0.30:
        return 1
    return 0


def _area_under_line_ratio(points: list[list[float]], max_x: float, max_y: float, omega: float = 0.5, start_score: float = 0.5) -> float:
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


def _pauc(turn_scores: list[tuple[dict[float, int], tuple[float, float]]], omega: float = 0.5) -> float:
    aucs: list[float] = []
    for scores, span in turn_scores:
        if not scores:
            aucs.append(0.25)
            continue
        pts = [[t - span[0], v] for t, v in scores.items()]
        aucs.append(_area_under_line_ratio(pts, span[1] - span[0], 2.0, omega, 0.5))
    return sum(aucs) / len(aucs) if aucs else 0.0


def _score_answers(runs: list[QuestionRun]) -> dict[str, Any]:
    turn_scores: list[tuple[dict[float, int], tuple[float, float]]] = []
    answered = duplicates = 0
    latencies: list[float] = []
    for run in runs:
        seen_norm: set[str] = set()
        for t, answer, _conf in run.answers:
            norm = " ".join(re.findall(r"[a-z0-9]+", answer.lower()))
            if norm in seen_norm:
                duplicates += 1
            seen_norm.add(norm)
        for turn in run.turns:
            scores: dict[float, int] = {}
            for t, answer, _conf in run.answers:
                if turn.span[0] <= t <= turn.span[1]:
                    scores[t] = _token_score(answer, turn.content)
            if scores:
                answered += 1
                latencies.append(min(scores) - turn.span[0])
            turn_scores.append((scores, turn.span))
    latencies.sort()
    return {
        "pauc@0": _pauc(turn_scores, 0.0) * 100,
        "pauc@0.5": _pauc(turn_scores, 0.5) * 100,
        "pauc@1": _pauc(turn_scores, 1.0) * 100,
        "unresolved_rate": 1.0 - (answered / len(turn_scores)) if turn_scores else None,
        "duplicate_rate": duplicates / sum(len(r.answers) for r in runs) if sum(len(r.answers) for r in runs) else 0.0,
        "first_answer_latency_p50": latencies[len(latencies) // 2] if latencies else None,
        "turns": len(turn_scores),
    }


def _candidate_metrics(runs: list[QuestionRun]) -> dict[str, float | int | None]:
    turns = hits = candidates = candidate_hits = 0
    for run in runs:
        for t in run.escalations:
            candidates += 1
            if any(turn.span[0] <= t <= turn.span[1] for turn in run.turns):
                candidate_hits += 1
        for turn in run.turns:
            turns += 1
            if any(turn.span[0] <= t <= turn.span[1] for t in run.escalations):
                hits += 1
    return {
        "candidate_recall": hits / turns if turns else None,
        "candidate_precision_proxy": candidate_hits / candidates if candidates else None,
    }


def _silent_reason(verdict: dict[str, Any] | None, threshold: float) -> str:
    if verdict is None:
        return "silent: no_verdict_recorded"
    if not verdict.get("satisfied"):
        return "silent: insufficient_answer_evidence"
    if float(verdict.get("confidence") or 0.0) < threshold:
        return "silent: low_confidence"
    return "silent: suppressed_or_refractory"


async def _score_one(row: dict[str, Any]) -> QuestionRun:
    question = _question(row)
    run = QuestionRun(
        question_id=str(row["question_id"]),
        video=str(row["video"]),
        question=question,
        turns=_turns(row),
    )
    settings = load_settings()
    vision = PVQAVision()
    core = Percept.create(vision=vision, settings=settings, discover_plugins=False)
    core.add_goal(
        Goal(
            id=f"pvqa-{run.question_id}",
            condition=f"there is enough visual evidence to answer: {question}",
            say="answer ready",
            # author a temporal clip so perceive_judged hands the judge a few frames of context (not one
            # frame at 0.2fps) — the single-frame starvation behind 0 answers in n1-fresh.
            state={"perception": {"temporal": "window", "window_s": PVQA_WINDOW_S}},
        )
    )
    spec = compile_watchspec(list(core.goals.active()), threshold=settings.edge.motion_gate)
    gate = MotionGate(spec, settings=settings)

    try:
        async for t, gate_frame, judge_frame in open_source(str(VIDEO_DIR / run.video), fps=FPS):
            run.frames += 1
            decision = gate.evaluate(gate_frame, ts=float(t))
            if not decision.escalate:
                continue
            run.escalations.append(float(t))
            vision.last_verdict = None
            events = await core.perceive_judged(judge_frame, "proactiveqa", t=float(t))
            action = "silent"
            for event in events:
                if event.action == FIRE and event.verdict is not None:
                    action = FIRE
                    run.answers.append((float(t), str(event.verdict.evidence or ""), float(event.verdict.confidence)))
                else:
                    action = getattr(event, "action", "ask")
            verdict = vision.last_verdict
            run.trace.append({
                "ts": round(float(t), 3),
                "edge": {
                    "decision": "escalate",
                    "motion": round(float(decision.signal.motion), 4),
                    "structural": round(float(decision.signal.structural), 4),
                    "semantic": round(float(decision.signal.semantic), 4),
                    "threshold": round(float(decision.threshold), 4),
                },
                "goal_id": f"pvqa-{run.question_id}",
                "goal": f"there is enough visual evidence to answer: {question}",
                "gemini_verdict": {
                    "satisfied": bool(verdict.get("satisfied")) if verdict else None,
                    "confidence": round(float(verdict.get("confidence") or 0.0), 3) if verdict else None,
                    "evidence": str(verdict.get("evidence", "") if verdict else "")[:240],
                },
                "core_gate_action": action,
                "reason": f"{action}: emitted" if action != "silent" else _silent_reason(verdict, settings.gate.threshold),
            })
    except Exception as exc:
        run.error = f"{type(exc).__name__}: {exc}"
    finally:
        run.judge_calls = vision.calls
    return run


async def _score_all(rows: list[dict[str, Any]]) -> list[QuestionRun]:
    runs: list[QuestionRun] = []
    for row in rows:
        runs.append(await _score_one(row))
    return runs


def _unavailable(reason: str) -> TrackResult:
    return TrackResult(
        track=TRACK,
        cells={
            ("e2e", "pauc@0.5_proxy"): None,
            ("e2e", "interruptions"): None,
            ("edge", "candidate_recall"): None,
            ("e2e", "call_reduction"): None,
            ("dense-modeled", "dense_frame_calls"): None,
        },
        headline={"ProactiveQA E2E": "n/a"},
        excluded=[f"{TRACK}: {reason}"],
        notes=reason,
    )


def score(*, live: bool, n: int) -> TrackResult:
    if not live:
        return _unavailable("live backend required; offline fake cannot assert answer quality")
    rows = _load_rows(n)
    if not rows:
        return _unavailable(f"no ProactiveQA subset found at {SUBSET}")

    runs = asyncio.run(_score_all(rows))
    failed = [r for r in runs if r.error]
    observed = len(runs) - len(failed)
    answer = _score_answers(runs)
    candidates = _candidate_metrics(runs)
    dense_calls = sum(r.frames for r in runs)
    e2e_calls = sum(r.judge_calls for r in runs)
    call_reduction = 1.0 - (e2e_calls / dense_calls) if dense_calls else None
    minutes = sum((r.frames / FPS) / 60.0 for r in runs)
    calls_per_min = e2e_calls / minutes if minutes else None
    interruptions = sum(len(r.answers) for r in runs)

    cells = {
        ("e2e", "pauc@0_proxy"): answer["pauc@0"],
        ("e2e", "pauc@0.5_proxy"): answer["pauc@0.5"],
        ("e2e", "pauc@1_proxy"): answer["pauc@1"],
        ("e2e", "interruptions"): interruptions,
        ("e2e", "duplicate_rate"): answer["duplicate_rate"],
        ("e2e", "first_answer_latency_p50"): answer["first_answer_latency_p50"],
        ("e2e", "unresolved_rate"): answer["unresolved_rate"],
        ("edge", "candidate_recall"): candidates["candidate_recall"],
        ("edge", "candidate_precision_proxy"): candidates["candidate_precision_proxy"],
        ("e2e", "candidate_calls"): e2e_calls,
        ("e2e", "calls_per_min"): calls_per_min,
        ("e2e", "call_reduction"): call_reduction,
        ("dense-modeled", "dense_frame_calls"): dense_calls,
    }
    per_question = [
        {
            "question_id": r.question_id,
            "video": r.video,
            "frames": r.frames,
            "turns": len(r.turns),
            "escalations": len(r.escalations),
            "judge_calls": r.judge_calls,
            "answers": [{"t": round(t, 3), "text": text[:160], "confidence": round(conf, 3)} for t, text, conf in r.answers],
            "trace": r.trace,
            "error": r.error,
        }
        for r in runs
    ]
    return TrackResult(
        track=TRACK,
        cells=cells,
        fmts={
            "pauc@0_proxy": "{:.1f}",
            "pauc@0.5_proxy": "{:.1f}",
            "pauc@1_proxy": "{:.1f}",
            "interruptions": "{:.0f}",
            "duplicate_rate": "{:.2f}",
            "first_answer_latency_p50": "{:.2f}",
            "unresolved_rate": "{:.2f}",
            "candidate_recall": "{:.2f}",
            "candidate_precision_proxy": "{:.2f}",
            "candidate_calls": "{:.0f}",
            "calls_per_min": "{:.2f}",
            "call_reduction": "{:.2f}",
            "dense_frame_calls": "{:.0f}",
        },
        headline={
            "ProactiveQA PAUC@0.5 proxy / calls": f"{answer['pauc@0.5']:.1f} / {e2e_calls} (N={len(runs)})",
            "ProactiveQA calls dense-modeled -> E2E": f"{dense_calls} -> {e2e_calls}",
        },
        measured=[
            "e2e:pauc@0_proxy",
            "e2e:pauc@0.5_proxy",
            "e2e:pauc@1_proxy",
            "e2e:interruptions",
            "e2e:duplicate_rate",
            "e2e:first_answer_latency_p50",
            "e2e:unresolved_rate",
            "edge:candidate_recall",
            "edge:candidate_precision_proxy",
            "e2e:candidate_calls",
            "e2e:calls_per_min",
            "e2e:call_reduction",
        ],
        modeled=["dense-modeled:dense_frame_calls"],
        excluded=["proactiveqa-e2e: PAUC uses token-overlap proxy; independent answer judge not wired"],
        reference={"pauc@0.5_proxy": {"value": TML_PAUC, "by": "TML ProactiveVideoQA PAUC@0.5 reference"}},
        calls=e2e_calls,
        coverage=[{
            "dataset": "proactiveqa/ego-subset",
            "expected": len(rows),
            "observed": observed,
            "failed": len(failed),
            "skipped": 0,
            "notes": "question rows through edge-gated Percept answer-evidence path",
        }],
        notes=f"subset={SUBSET}; fps={FPS}; PAUC score is token-overlap proxy until independent judge is wired; per_question={per_question}",
    )
