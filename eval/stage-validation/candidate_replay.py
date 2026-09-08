"""Offline qualification for local candidate generators (zero provider spend).

The same detector is replayed over a labeled event clip and a long source.  The
artifact reports candidate recall/latency, local CPU cost, calls per source-hour,
and projected provider spend.  It does not score semantic precision: candidates
are intentionally high-recall proposals; the VLM confirmation owns the final
collision decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from percept.harness.candidates import FixedCameraCollisionCandidate


DEFAULT_CLIP = Path(
    "/private/tmp/traffic-review-corrected/"
    "Traffic_Data_collision_single_shot_122p433_129p733.mp4"
)
DEFAULT_FULL = Path("/Users/dhivyavijayakumar/Downloads/Traffic_Data.mp4")
DEFAULT_ONSET_S = 2.2666666667
DEFAULT_CALL_COST_USD = 0.0009274


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def replay(
    path: Path,
    *,
    fps: float,
    detector: FixedCameraCollisionCandidate,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"unable to open {path}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 0.0:
        raise RuntimeError(f"source reports invalid fps: {source_fps}")
    step = max(1, int(round(source_fps / fps)))
    triggers: list[dict] = []
    elapsed_ms: list[float] = []
    decoded_frames = 0
    segment_frames = 0
    sampled_frames = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frame_index = decoded_frames
        decoded_frames += 1
        if frame_index < start_frame:
            continue
        if end_frame is not None and frame_index >= end_frame:
            break
        segment_frames += 1
        if (frame_index - start_frame) % step:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        started = time.perf_counter()
        score = detector(rgb)
        elapsed_ms.append((time.perf_counter() - started) * 1000.0)
        sampled_frames += 1
        if score > 0.0:
            receipt = detector.last_receipt
            triggers.append(
                {
                    "t": (frame_index - start_frame) / source_fps,
                    "source_frame": frame_index,
                    "score": score,
                    "local_motion": receipt.local_motion,
                    "local_shock": receipt.local_shock,
                    "recent_motion": receipt.recent_motion,
                    "global_change_fraction": receipt.global_change_fraction,
                }
            )
    cap.release()
    duration_s = segment_frames / source_fps
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "source_fps": source_fps,
        "sample_fps_requested": fps,
        "sample_fps_effective": source_fps / step,
        "source_frame_interval_half_open": [start_frame, end_frame],
        "source_frames": segment_frames,
        "sampled_frames": sampled_frames,
        "duration_s": duration_s,
        "triggers": triggers,
        "trigger_count": len(triggers),
        "triggers_per_source_hour": 0.0 if duration_s <= 0 else len(triggers) * 3600.0 / duration_s,
        "local_stage_ms": {
            "p50": statistics.median(elapsed_ms) if elapsed_ms else None,
            "p95": _percentile(elapsed_ms, 0.95),
            "max": max(elapsed_ms) if elapsed_ms else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--full", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--onset-s", type=float, default=DEFAULT_ONSET_S)
    parser.add_argument("--original-start-frame", type=int, default=3673)
    parser.add_argument("--original-end-frame", type=int, default=3892)
    parser.add_argument("--region", default="0.10,0.20,0.84,0.65")
    parser.add_argument("--call-cost-usd", type=float, default=DEFAULT_CALL_COST_USD)
    parser.add_argument("--cost-ceiling-usd-per-hour", type=float, default=0.10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    region = tuple(float(value) for value in args.region.split(","))
    if len(region) != 4:
        raise SystemExit("--region requires x,y,width,height")

    config = {
        "region": region,
        "grid": (6, 4),
        "min_local_motion": 0.15,
        "min_local_shock": 0.08,
        "min_recent_motion": 0.012,
        "recent_frames": 8,
        "warmup_frames": 20,
        "cut_pixel_delta": 48,
        "min_global_change_fraction": 0.05,
        "max_global_change_fraction": 0.58,
        "rearm_motion": 0.025,
        "rearm_frames": 3,
        "cooldown_frames": 8,
    }
    clip = replay(
        args.clip,
        fps=args.fps,
        detector=FixedCameraCollisionCandidate(**config),
    )
    original_slice = replay(
        args.full,
        fps=args.fps,
        detector=FixedCameraCollisionCandidate(**config),
        start_frame=args.original_start_frame,
        end_frame=args.original_end_frame,
    )
    long_source = replay(
        args.full,
        fps=args.fps,
        detector=FixedCameraCollisionCandidate(**config),
    )
    post_onset = [row for row in clip["triggers"] if row["t"] >= args.onset_s]
    first = post_onset[0] if post_onset else None
    original_post_onset = [
        row for row in original_slice["triggers"] if row["t"] >= args.onset_s
    ]
    original_first = original_post_onset[0] if original_post_onset else None
    projected_cost = long_source["triggers_per_source_hour"] * args.call_cost_usd
    artifact = {
        "schema": "percept.candidate-replay/1",
        "detector": FixedCameraCollisionCandidate.name,
        "config": config,
        "ground_truth": {"onset_s": args.onset_s, "positive_interval": [args.onset_s, clip["duration_s"]]},
        "event_clip": clip,
        "original_source_slice": original_slice,
        "long_source": long_source,
        "metrics": {
            "candidate_recall": 1.0 if first is not None else 0.0,
            "candidate_latency_s": None if first is None else first["t"] - args.onset_s,
            "original_slice_candidate_recall": 1.0 if original_first is not None else 0.0,
            "original_slice_candidate_latency_s": (
                None if original_first is None else original_first["t"] - args.onset_s
            ),
            "transport_invariance_pass": (
                first is not None
                and original_first is not None
                and abs(first["t"] - original_first["t"]) <= 1.0 / clip["sample_fps_effective"]
            ),
            "candidate_calls_per_source_hour": long_source["triggers_per_source_hour"],
            "projected_provider_cost_usd_per_source_hour": projected_cost,
            "provider_cost_ceiling_usd_per_source_hour": args.cost_ceiling_usd_per_hour,
            "economy_pass": projected_cost <= args.cost_ceiling_usd_per_hour,
            "semantic_precision": None,
            "semantic_precision_note": "requires the downstream VLM confirmation; candidates are proposals",
        },
    }
    rendered = json.dumps(artifact, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
