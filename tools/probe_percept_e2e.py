#!/usr/bin/env python3
"""Probe the Percept E2E benchmark shape with N=1 per dataset family.

This is a wiring and metric-shape probe, not a benchmark result. It verifies:

- runtime source emits separate gate frames and judge frames,
- the tier-0 gate wakes exactly one candidate moment,
- Percept calls ``core.perceive_judged`` only on that candidate,
- fire/ask/silent results flow into the metric fields the benchmark spec expects.

Default mode is deterministic fake vision. ``--live-gemini`` spends one Gemini candidate
judgment per E2E scenario through the same runtime path.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import zlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for rel in ("packages/percept-vision", "packages/percept-harness", "packages/percept-contracts"):
    p = str(ROOT / rel)
    if p not in sys.path:
        sys.path.insert(0, p)

from percept import Goal, Percept  # noqa: E402
from percept.adapters import Verdict  # noqa: E402
from percept.backends.fake import FakeVision  # noqa: E402
from percept.config import load_settings  # noqa: E402
from percept.edge.tier0 import gray_frame  # noqa: E402
from percept.runtime import RuntimeHealthEvent, RuntimeStats, run  # noqa: E402


STILL = gray_frame(8, 8, 0)
MOVED = gray_frame(8, 8, 255)


def load_dotenv(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def png_solid_rgb(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """Tiny valid PNG writer, avoiding a PIL/cv2 dependency in the probe."""
    scanline = b"\x00" + bytes(rgb) * width
    raw = scanline * height

    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(kind + data) & 0xFFFFFFFF
        return len(data).to_bytes(4, "big") + kind + data + crc.to_bytes(4, "big")

    ihdr = (
        width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + b"\x08\x02\x00\x00\x00"
    )
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


class CountingVision:
    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls = 0

    async def judge(self, *args: Any, **kwargs: Any) -> Verdict:
        self.calls += 1
        return await self.inner.judge(*args, **kwargs)

    async def detect(self, *args: Any, **kwargs: Any) -> list:
        return await self.inner.detect(*args, **kwargs)

    async def detect_events(self, *args: Any, **kwargs: Any) -> list:
        return await self.inner.detect_events(*args, **kwargs)


def _vision(live: bool, token: str) -> tuple[CountingVision, Any, str]:
    if live:
        from percept.backends.gemini import GeminiVision

        settings = load_settings()
        frame = ("image/png", png_solid_rgb(96, 96, (255, 0, 0)))
        return CountingVision(GeminiVision(settings=settings)), frame, settings.models.vision_model
    script = {
        token: Verdict(satisfied=True, confidence=0.95, evidence=f"probe scripted yes: {token}"),
        "clear": Verdict(satisfied=False, confidence=0.95, evidence="probe scripted no"),
    }
    return CountingVision(FakeVision(script=script)), token, "fake"


async def _run_watch_probe(*, name: str, goal: Goal, live: bool) -> dict[str, Any]:
    vision, event_frame, model = _vision(live, name)
    core = Percept.create(vision=vision, discover_plugins=False)
    core.add_goal(goal)
    stats = RuntimeStats()
    health: list[RuntimeHealthEvent] = []
    source = [
        (0.0, STILL, "clear" if not live else ("image/png", png_solid_rgb(96, 96, (0, 0, 0)))),
        (1.0, MOVED, event_frame),
        (2.0, MOVED, event_frame),
    ]
    incidents = [
        incident
        async for incident in run(
            core,
            source=source,
            stats=stats,
            on_health=health.append,
            snippet_pre_s=2.0,
            snippet_size=3,
        )
    ]
    fire_ts = [i.ts for i in incidents]
    return {
        "probe": name,
        "model": model,
        "runtime_stats": asdict(stats),
        "vision_judge_calls": vision.calls,
        "health_kinds": [e.kind for e in health],
        "fire_count": len(incidents),
        "fire_ts": fire_ts,
        "actions": ["fire" for _ in incidents],
        "evidence": [getattr(i.verdict, "evidence", "") for i in incidents],
    }


def _latency(fire_ts: list[float], onset: float) -> float | None:
    valid = [t for t in fire_ts if t >= onset]
    return None if not valid else min(valid) - onset


def _miou(pred: tuple[float, float] | None, gt: tuple[float, float]) -> float:
    if pred is None:
        return 0.0
    inter = max(0.0, min(pred[1], gt[1]) - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) - min(pred[0], gt[0])
    return 0.0 if union <= 0 else inter / union


async def run_probe(*, live: bool) -> dict[str, Any]:
    load_dotenv()
    settings = load_settings()
    live_model = os.environ.get("PERCEPT_GEMINI_LIVE_MODEL")
    meta = {
        "probe_only": True,
        "sample_n_per_family": 1,
        "backend": "gemini" if live else "fake",
        "generate_content_model": settings.models.vision_model if live else "fake",
        "live_api_model_recorded": live_model,
        "live_api_note": (
            "PERCEPT_GEMINI_LIVE_MODEL is bidiGenerateContent-only; current GeminiVision uses generateContent."
            if live
            else "fake mode does not call a provider"
        ),
    }

    pt = await _run_watch_probe(
        name="pt_action_n1",
        live=live,
        goal=Goal(
            id="pt-open-door",
            condition="a red square is visible in the image",
            say="The action is happening.",
        ),
    )
    pt_latency = _latency(pt["fire_ts"], 1.0)
    pt_metrics = {
        "dataset_shape": "Perception Test action/sound label converted to one Percept goal",
        "hypothesis": "Edge wakes candidate moments; E2E notifies near event onset while spending fewer heavy calls than dense.",
        "edge_event_recall": 1.0 if pt["runtime_stats"]["escalations"] > 0 else 0.0,
        "edge_precision_proxy": pt["runtime_stats"]["escalations"] / max(1, pt["runtime_stats"]["frames"]),
        "notification_recall": 1.0 if pt["fire_count"] > 0 else 0.0,
        "notification_latency_s": pt_latency,
        "dense_modeled_calls": pt["runtime_stats"]["frames"],
        "heavy_model_calls": pt["vision_judge_calls"],
    }

    charades = await _run_watch_probe(
        name="charades_n1",
        live=live,
        goal=Goal(
            id="charades-holding-object",
            condition="a red square is visible in the image",
            say="The Charades action is active.",
        ),
    )
    pred_span = (charades["fire_ts"][0], 2.0) if charades["fire_ts"] else None
    charades_metrics = {
        "dataset_shape": "One Charades action interval converted to one Percept watch goal",
        "hypothesis": "Percept emits the action interval from sparse candidate judgments, with dense calls modeled.",
        "gt_span_s": [1.0, 2.0],
        "pred_span_s": list(pred_span) if pred_span else None,
        "e2e_miou": _miou(pred_span, (1.0, 2.0)),
        "event_recall": 1.0 if charades["fire_count"] > 0 else 0.0,
        "dense_modeled_calls": charades["runtime_stats"]["frames"],
        "heavy_model_calls": charades["vision_judge_calls"],
    }

    pvqa = await _run_watch_probe(
        name="proactiveqa_n1",
        live=live,
        goal=Goal(
            id="pvqa-answer",
            condition="the image contains enough evidence to answer the user's question: what color is the square?",
            say="The square is red.",
        ),
    )
    pvqa_latency = _latency(pvqa["fire_ts"], 1.0)
    pvqa_metrics = {
        "dataset_shape": "One standing question, answer evidence appears during the stream",
        "hypothesis": "Percept waits until answer evidence appears, then answers once without polling dense VLM.",
        "answer_window_s": [1.0, 2.0],
        "answer_recall": 1.0 if pvqa["fire_count"] > 0 else 0.0,
        "first_answer_latency_s": pvqa_latency,
        "p_pauc_proxy": 100.0 if pvqa_latency is not None and 0.0 <= pvqa_latency <= 1.0 else 0.0,
        "duplicate_count": max(0, pvqa["fire_count"] - 1),
        "dense_modeled_calls": pvqa["runtime_stats"]["frames"],
        "heavy_model_calls": pvqa["vision_judge_calls"],
    }

    repcount_metrics = {
        "dataset_shape": "One RepCount-A action contract; edge/core attempts count, no raw VLM baseline",
        "hypothesis": "The brain owns start/stop/unit semantics; edge/core count reps and report unresolved failures.",
        "ground_truth_count": 1,
        "predicted_count": 1,
        "obo": 1.0,
        "absolute_mae": 0,
        "normalized_mae": 0.0,
        "unresolved": False,
        "heavy_model_calls": 0,
        "note": "shape probe only; this row does not exercise MediaPipe or a real RepCount-A video",
    }

    cue_speak_metrics = {
        "dataset_shape": "Derived from PT/PVQA E2E outputs; no separate video pass",
        "hypothesis": "Score whether the emitted cue is semantically correct and non-duplicative.",
        "cue_recall": pt_metrics["notification_recall"],
        "cue_precision": 1.0 if pt["fire_count"] == 1 else 0.0,
        "fires_per_episode": pt["fire_count"],
        "duplicate_count": max(0, pt["fire_count"] - 1),
        "source": "perception-test",
    }

    time_speak_metrics = {
        "dataset_shape": "Derived from PT/PVQA E2E outputs; no separate video pass",
        "hypothesis": "Score whether the first useful notification lands inside the event-time window.",
        "timing_hit": pt_latency is not None and 0.0 <= pt_latency <= 1.0,
        "latency_s": pt_latency,
        "early_fire": bool(pt["fire_ts"] and min(pt["fire_ts"]) < 1.0),
        "late_fire": pt_latency is None or pt_latency > 1.0,
        "p_pauc_proxy": 100.0 if pt_latency is not None and 0.0 <= pt_latency <= 1.0 else 0.0,
        "source": "perception-test",
    }

    return {
        "meta": meta,
        "families": {
            "perception_test_action_sound": {"raw_probe": pt, "metrics": pt_metrics},
            "charades_temporal_notification": {"raw_probe": charades, "metrics": charades_metrics},
            "repcount_a": {"metrics": repcount_metrics},
            "proactive_videoqa": {"raw_probe": pvqa, "metrics": pvqa_metrics},
            "cuespeak": {"metrics": cue_speak_metrics},
            "timespeak": {"metrics": time_speak_metrics},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-gemini", action="store_true", help="spend one Gemini candidate call per E2E scenario")
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    args = parser.parse_args()
    out = asyncio.run(run_probe(live=args.live_gemini))
    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
        return
    print("=== Percept E2E N=1 shape probe ===")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
