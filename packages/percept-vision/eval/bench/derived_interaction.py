"""Derived CueSpeak/TimeSpeak scoring over saved E2E benchmark artifacts.

These helpers intentionally do not read video or call models. They consume the JSON artifacts already
written by ``tools/percept_benchmark.py run`` and answer the interaction question on top of the same
E2E pass:

* CueSpeak: did Percept say/fire once inside the labelled cue/answer window?
* TimeSpeak: did it speak soon enough after the useful window opened?

The artifact formats are still early, so the parser is conservative and best-effort. A missing artifact
returns an empty derivation and the caller can fall back to deterministic sim-core regression.
"""

from __future__ import annotations

import ast
import json
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
RUNS = ROOT / "eval" / "local-benchmark" / "runs"
CHARADES_SELECTION = ROOT / "eval" / "bench-staging" / "charades" / "raw" / "selection.json"


@dataclass(frozen=True)
class DerivedCue:
    source: str
    label: str
    window: tuple[float, float]
    fires: tuple[float, ...]
    asks: int = 0

    @property
    def valid_fires(self) -> tuple[float, ...]:
        lo, hi = self.window
        return tuple(t for t in self.fires if lo <= t <= hi)

    @property
    def first_valid_latency(self) -> float | None:
        valid = self.valid_fires
        return None if not valid else min(valid) - self.window[0]

    def fires_before(self) -> int:
        return sum(1 for t in self.fires if t < self.window[0])

    def fires_after(self, *, omega_s: float) -> int:
        return sum(1 for t in self.fires if t > self.window[0] + omega_s)


def _read_result_path(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    if not data.get("coverage"):
        return None
    data["_artifact"] = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    return data


def _current_run_result(track: str) -> dict[str, Any] | None:
    raw = os.environ.get("PERCEPT_BENCH_RUN_DIR")
    if not raw:
        return None
    run_dir = Path(raw)
    path = run_dir / "results" / f"{track}.json"
    if not path.exists():
        return None
    return _read_result_path(path)


def _latest_result(track: str) -> dict[str, Any] | None:
    current = _current_run_result(track)
    if current is not None:
        return current
    if not RUNS.exists():
        return None
    paths = [p for p in RUNS.glob(f"*/results/{track}.json") if p.is_file()]
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for path in paths:
        data = _read_result_path(path)
        if data is not None and not data.get("excluded"):
            return data
    return None


def _literal_after(notes: str, marker: str) -> Any:
    idx = notes.find(marker)
    if idx < 0:
        return None
    chunk = notes[idx + len(marker):].strip()
    # Stop at the next top-level-ish field marker when present. Current notes put the payload last,
    # but this keeps the parser tolerant of future suffixes.
    for stop in ("; metric=", "; PAUC", "; fps=", "; per_video=", "; per_question=", "; per_event="):
        if stop.strip() and stop in chunk and not marker.startswith(stop.strip()):
            chunk = chunk.split(stop, 1)[0].strip()
    try:
        return ast.literal_eval(chunk)
    except Exception:
        # Last resort: trim to the final bracketed literal.
        end = max(chunk.rfind("]"), chunk.rfind("}"))
        if end >= 0:
            try:
                return ast.literal_eval(chunk[: end + 1])
            except Exception:
                return None
    return None


def _charades_goal_map(video_id: str) -> dict[str, tuple[str, tuple[float, float]]]:
    if not CHARADES_SELECTION.exists():
        return {}
    try:
        rows = json.loads(CHARADES_SELECTION.read_text())
    except Exception:
        return {}
    row = next((r for r in rows if r.get("id") == video_id), None)
    if row is None:
        return {}
    by_desc: dict[str, list[tuple[float, float]]] = {}
    for desc, start, end in row.get("actions", []):
        by_desc.setdefault(str(desc), []).append((float(start), float(end)))
    out = {}
    for idx, (desc, segs) in enumerate(sorted(by_desc.items())):
        # The N=1 smoke has one segment per desc. Keep the first if future data has repeats; repeated
        # segments should become separate DerivedCue rows when the artifact carries structured intervals.
        out[f"charades-{video_id}-{idx}"] = (desc, segs[0])
    return out


def cues_from_latest_e2e() -> tuple[list[DerivedCue], list[str], int]:
    """Return derived cue rows, artifact refs, and candidate call count from latest E2E artifacts."""

    cues: list[DerivedCue] = []
    refs: list[str] = []
    calls = 0

    pt = _latest_result("pt-action-e2e")
    if pt:
        refs.append(pt["_artifact"])
        calls += int(pt.get("calls") or 0)
        rows = _literal_after(str(pt.get("notes", "")), "per_event=") or []
        for row in rows:
            window = row.get("window") or []
            if len(window) != 2:
                continue
            fires = tuple(float(t) for t in row.get("fires", []))
            cues.append(DerivedCue("pt-action-e2e", str(row.get("label") or row.get("goal")), (float(window[0]), float(window[1])), fires))

    charades = _latest_result("charades-e2e")
    if charades:
        refs.append(charades["_artifact"])
        calls += int(charades.get("calls") or 0)
        rows = _literal_after(str(charades.get("notes", "")), "per_video=") or []
        for row in rows:
            goals = _charades_goal_map(str(row.get("video_id")))
            fires_by_goal: dict[str, list[float]] = {}
            for fire in row.get("fires", []):
                fires_by_goal.setdefault(str(fire.get("goal_id")), []).append(float(fire.get("t", 0.0)))
            for gid, (label, window) in goals.items():
                cues.append(DerivedCue("charades-e2e", label, window, tuple(fires_by_goal.get(gid, []))))

    pvqa = _latest_result("proactiveqa-e2e")
    if pvqa:
        refs.append(pvqa["_artifact"])
        calls += int(pvqa.get("calls") or 0)
        rows = _literal_after(str(pvqa.get("notes", "")), "per_question=") or []
        # Current PVQA artifacts do not include per-turn windows in notes, only the number of turns and
        # emitted answers. Record a coarse row so Cue/TimeSpeak can surface "no answer emitted" rather
        # than silently ignoring the run.
        for row in rows:
            answers = tuple(float(a.get("t", 0.0)) for a in row.get("answers", []))
            turns = int(row.get("turns") or 0)
            for i in range(turns):
                cues.append(DerivedCue("proactiveqa-e2e", f"{row.get('question_id')} answer turn {i + 1}", (0.0, 0.0), answers))

    return cues, refs, calls


def cue_metrics(cues: list[DerivedCue]) -> dict[str, float | int | None]:
    if not cues:
        return {"precision": None, "recall": None, "fires_per_episode": None, "duplicate_rate": None, "unresolved_rate": None}
    all_fires = sum(len(c.fires) for c in cues)
    valid = sum(len(c.valid_fires) for c in cues)
    hit = sum(1 for c in cues if c.valid_fires)
    duplicates = sum(max(0, len(c.valid_fires) - 1) for c in cues)
    unresolved = sum(1 for c in cues if not c.valid_fires and c.asks > 0)
    return {
        "precision": valid / all_fires if all_fires else 0.0,
        "recall": hit / len(cues),
        "fires_per_episode": all_fires / len(cues),
        "duplicate_rate": duplicates / all_fires if all_fires else 0.0,
        "unresolved_rate": unresolved / len(cues),
    }


def time_metrics(cues: list[DerivedCue], *, omega_s: float = 2.0) -> dict[str, float | int | None]:
    if not cues:
        return {"timing_accuracy": None, "p_pauc": None, "latency_p50": None, "early_fire_rate": None, "late_fire_rate": None}
    latencies = [c.first_valid_latency for c in cues]
    hits = [x for x in latencies if x is not None and 0.0 <= x <= omega_s]
    p_pauc = sum(max(0.0, 1.0 - (x / omega_s)) for x in hits) * 100.0 / len(cues)
    all_hit_latencies = [x for x in latencies if x is not None]
    return {
        "timing_accuracy": len(hits) / len(cues),
        "p_pauc": p_pauc,
        "latency_p50": median(all_hit_latencies) if all_hit_latencies else None,
        "early_fire_rate": sum(1 for c in cues if c.fires_before()) / len(cues),
        "late_fire_rate": sum(1 for c in cues if c.fires_after(omega_s=omega_s) and not c.first_valid_latency) / len(cues),
    }


def summarize_sources(cues: list[DerivedCue]) -> dict[str, int]:
    out: dict[str, int] = {}
    for cue in cues:
        out[cue.source] = out.get(cue.source, 0) + 1
    return out
