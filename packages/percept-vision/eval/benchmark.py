"""Consumer-facing benchmark: per-concern precision / recall from a labelled-clip manifest.

The L2 scorer (:mod:`eval.replay`) aligns *fires* to *rising edges* and rolls up one global
scorecard. A consumer shipping their own concerns wants the same trust numbers **per concern**:
"for *this* watch condition, what fraction of my fires were right (precision), and what fraction
of the moments it should have fired did it catch (recall)?" — plus the two operational rates the
gate exposes, **false-fire** (the spec's trust gate) and **refuse/ask**.

This module adds that consumer path on top of the existing engine — it does not rebuild it. The
alignment + metric math is still :func:`eval.replay.align`; here we just *partition by concern*
and add per-concern precision / recall / false-fire / refuse rates.

Two ways to feed it, both **offline, no keys**:

* **replay a manifest** (:func:`run_manifest`) — the manifest lists labelled clips with scripted
  frame tokens and the expected rising edges per concern; this drives a real :class:`Percept` on
  the deterministic fake backend, records its fires/asks, and scores them. This is the faithful
  path: the gate, refractory, and rising-edge logic all run.
* **score recorded fires** (:func:`per_concern_report`) — if a consumer already replayed their
  clips live and captured ``(concern, observed_fires, observed_asks, expected_edges)``, hand those
  in directly. The report is identical; only the source of the fires differs.

The benchmark input format is documented in ``docs/eval.md`` and on :class:`Manifest`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .replay import Edge, Fire, align


# ---------------------------------------------------------------------------
# Per-concern scoring (pure; no models, no IO)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConcernObservation:
    """Everything the scorer needs for one concern: the labels and what the system did.

    * ``expected_edges`` — ground-truth rising edges (the moments the concern *should* fire).
    * ``fires`` — the FIRE actions the system actually emitted for this concern.
    * ``n_asks`` — the count of ASK (refuse-to-guess) actions for this concern; the refuse rate's
      numerator. Asks are neither fires nor misses — they are the gate declining to guess.
    """

    concern_id: str
    expected_edges: list[Edge] = field(default_factory=list)
    fires: list[Fire] = field(default_factory=list)
    n_asks: int = 0


@dataclass
class ConcernReport:
    """Per-concern precision / recall / false-fire / refuse, from one alignment."""

    concern_id: str
    omega: float
    tp: int
    fn: int
    fp: int
    n_edges: int
    n_fires: int
    n_asks: int
    precision: float
    recall: float
    false_fire: int
    false_fire_rate: float  # FP / n_fires (the fraction of fires that were wrong)
    refuse_rate: float  # asks / (fires + asks): how often the gate declined to guess

    def as_dict(self) -> dict:
        return asdict(self)


def score_concern(obs: ConcernObservation, omega: float) -> ConcernReport:
    """Align one concern's fires to its expected edges and roll up the per-concern numbers.

    Reuses :func:`eval.replay.align` (the same timing discipline as the global scorer: a fire only
    satisfies an edge inside ``[edge, edge+omega]`` and each edge consumes at most one fire). The
    only thing added on top is the per-concern packaging plus the two rate denominators.
    """
    a = align(obs.fires, obs.expected_edges, omega)
    n_fires = len(obs.fires)
    n_asks = obs.n_asks
    return ConcernReport(
        concern_id=obs.concern_id,
        omega=omega,
        tp=a.tp,
        fn=a.fn,
        fp=a.fp,
        n_edges=len(obs.expected_edges),
        n_fires=n_fires,
        n_asks=n_asks,
        precision=a.precision,
        recall=a.recall,
        false_fire=a.false_fire,
        false_fire_rate=(a.fp / n_fires) if n_fires else 0.0,
        refuse_rate=(n_asks / (n_fires + n_asks)) if (n_fires + n_asks) else 0.0,
    )


def per_concern_report(observations: list[ConcernObservation], omega: float) -> dict:
    """Score every concern and roll the per-concern rows up into one report dict.

    The roll-up ``overall`` micro-averages (sums TP/FP/FN across concerns, then divides) so a
    concern with more edges weighs more — the same convention as the global scorecard. ``false_fire``
    is the absolute total, so a CI gate can assert ``overall.false_fire == 0`` exactly like
    ``eval.run``'s existing gate.
    """
    per = [score_concern(o, omega) for o in observations]
    tp = sum(r.tp for r in per)
    fp = sum(r.fp for r in per)
    fn = sum(r.fn for r in per)
    n_fires = sum(r.n_fires for r in per)
    n_asks = sum(r.n_asks for r in per)
    overall = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_edges": sum(r.n_edges for r in per),
        "n_fires": n_fires,
        "n_asks": n_asks,
        "precision": (tp / (tp + fp)) if (tp + fp) else 1.0,
        "recall": (tp / (tp + fn)) if (tp + fn) else 1.0,
        "false_fire": fp,
        "false_fire_rate": (fp / n_fires) if n_fires else 0.0,
        "refuse_rate": (n_asks / (n_fires + n_asks)) if (n_fires + n_asks) else 0.0,
    }
    return {
        "omega": omega,
        "concerns": {r.concern_id: r.as_dict() for r in per},
        "overall": overall,
    }


# ---------------------------------------------------------------------------
# Manifest (the documented consumer input format)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ManifestClip:
    """One labelled clip: a frame timeline + the per-concern ground truth on it.

    * ``clip_id`` — a stable id for the clip (used in fire ids so concerns can share frame tokens).
    * ``frames`` — ``[[t, frame_token], ...]``; replayed in order. With the fake backend
      ``frame_token`` is a scripted string (see ``percept.backends.fake``); with a real backend it
      would be a decoded frame, but the manifest format is identical.
    * ``expected`` — ``{concern_id: [t, ...]}``: the rising-edge timestamps each concern should fire
      at on this clip. A concern absent from ``expected`` (or with ``[]``) is a *no-fire* clip for
      that concern — any fire there is a false-fire.

    Note on the *fake* backend: it scripts verdicts by **frame token, not by condition**, so on a
    shared frame every concern sees the same verdict. For faithful offline examples script one
    concern per clip (or give each concern its own no-fire tokens). A real backend judges each
    concern's condition independently against the same frame, so multi-concern clips are exact.
    """

    clip_id: str
    frames: list[tuple[float, Any]]
    expected: dict[str, list[float]] = field(default_factory=dict)


@dataclass(frozen=True)
class Manifest:
    """A benchmark manifest: the concerns under test + the labelled clips.

    JSON shape (``docs/eval.md`` is the spec)::

        {
          "omega": 2.0,
          "concerns": [
            {"id": "coffee", "condition": "the user is drinking coffee", "say": "..."}
          ],
          "clips": [
            {
              "clip_id": "c1",
              "frames": [[0.0, "empty"], [1.0, "sip-coffee"]],
              "expected": {"coffee": [1.0]}
            }
          ]
        }

    Every ``expected`` key and every concern in a clip must be declared in ``concerns`` (so a typo'd
    concern id fails loudly rather than silently scoring nothing).
    """

    concerns: list[dict]
    clips: list[ManifestClip]
    omega: float = 2.0

    @property
    def concern_ids(self) -> list[str]:
        return [c["id"] for c in self.concerns]

    @staticmethod
    def from_dict(data: dict) -> "Manifest":
        concerns = list(data.get("concerns", []))
        for c in concerns:
            if "id" not in c or "condition" not in c:
                raise ValueError(f"each concern needs 'id' and 'condition'; got {c!r}")
        declared = {c["id"] for c in concerns}
        clips: list[ManifestClip] = []
        for raw in data.get("clips", []):
            clip_id = raw["clip_id"]
            frames = [(float(t), frame) for t, frame in raw.get("frames", [])]
            expected = {k: [float(t) for t in v] for k, v in raw.get("expected", {}).items()}
            unknown = set(expected) - declared
            if unknown:
                raise ValueError(
                    f"clip {clip_id!r} references undeclared concern(s) {sorted(unknown)}; "
                    f"declared concerns are {sorted(declared)}"
                )
            clips.append(ManifestClip(clip_id=clip_id, frames=frames, expected=expected))
        omega = float(data.get("omega", 2.0))
        return Manifest(concerns=concerns, clips=clips, omega=omega)

    @staticmethod
    def load(path: str | Path) -> "Manifest":
        return Manifest.from_dict(json.loads(Path(path).read_text()))


def edges_from_manifest(manifest: Manifest) -> dict[str, list[Edge]]:
    """Collect the ground-truth edges per concern across all clips.

    Edge ids are namespaced by ``clip_id`` so the same condition firing on two clips stays
    distinct during alignment (``"<clip>::<concern>"``), mirroring ``eval.run.edges_from_labels``.
    """
    out: dict[str, list[Edge]] = {c: [] for c in manifest.concern_ids}
    for clip in manifest.clips:
        for concern_id, times in clip.expected.items():
            for t in times:
                out.setdefault(concern_id, []).append(Edge(_goal_id(clip.clip_id, concern_id), t))
    return out


def _goal_id(clip_id: str, concern_id: str) -> str:
    return f"{clip_id}::{concern_id}"


# ---------------------------------------------------------------------------
# Replaying a manifest through a real Percept on the fake backend (offline)
# ---------------------------------------------------------------------------


def run_manifest(manifest: Manifest, *, vision: Any = None, settings: Any = None) -> dict:
    """Replay every clip in ``manifest`` through a :class:`Percept` and emit the per-concern report.

    Offline by default: with no ``vision`` backend supplied this builds a ``Percept`` on the
    deterministic ``FakeVision`` (no keys, no network). The clip's ``frame`` tokens drive the
    scripted verdicts; FIRE actions are recorded as fires, ASK actions as refuses. Each clip is
    replayed on a *fresh* agent so refractory/entity state never leaks across clips.

    Returns the same dict as :func:`per_concern_report` (``omega`` / ``concerns`` / ``overall``).
    """
    import asyncio

    from percept.adapters import Goal
    from percept.agent import Percept

    if vision is None:
        from percept.backends.fake import FakeVision

        vision = FakeVision()

    edges = edges_from_manifest(manifest)
    fires_by_concern: dict[str, list[Fire]] = {c: [] for c in manifest.concern_ids}
    asks_by_concern: dict[str, int] = {c: 0 for c in manifest.concern_ids}

    async def replay() -> None:
        for clip in manifest.clips:
            kwargs: dict[str, Any] = {"vision": vision, "discover_plugins": False}
            if settings is not None:
                kwargs["settings"] = settings
            agent = Percept.create(**kwargs)
            # one goal per concern, per clip, with a clip-namespaced id so fires map back cleanly.
            for c in manifest.concerns:
                agent.add_goal(
                    Goal(
                        id=_goal_id(clip.clip_id, c["id"]),
                        condition=c["condition"],
                        say=c.get("say", c["condition"]),
                    )
                )
            for ts, frame in clip.frames:
                events = await agent.perceive_judged(frame, "camera", t=ts)
                for ev in events:
                    concern_id = _concern_of(ev.goal_id, clip.clip_id)
                    if concern_id is None:
                        continue
                    if ev.action == "fire":
                        conf = ev.verdict.confidence if ev.verdict is not None else 1.0
                        fires_by_concern.setdefault(concern_id, []).append(
                            Fire(goal_id=ev.goal_id, ts=ts, conf=conf)
                        )
                    elif ev.action == "ask":
                        asks_by_concern[concern_id] = asks_by_concern.get(concern_id, 0) + 1

    asyncio.run(replay())

    observations = [
        ConcernObservation(
            concern_id=c,
            expected_edges=edges.get(c, []),
            fires=fires_by_concern.get(c, []),
            n_asks=asks_by_concern.get(c, 0),
        )
        for c in manifest.concern_ids
    ]
    return per_concern_report(observations, manifest.omega)


def _concern_of(goal_id: str, clip_id: str) -> str | None:
    prefix = f"{clip_id}::"
    return goal_id[len(prefix):] if goal_id.startswith(prefix) else None


# ---------------------------------------------------------------------------
# CI regression gate (consumer-wired; analogous to eval.run's false_fire == 0)
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkGate:
    """Per-concern P/R thresholds a consumer can wire into CI.

    The defaults encode the spec's trust posture: ``max_false_fire == 0`` (never confidently-wrong
    fire — the same hard gate as ``eval.run``) and no precision/recall floor unless the consumer
    sets one. ``min_precision`` / ``min_recall`` apply per concern; set them to add a quality floor
    (e.g. "every concern must recall >= 0.8").
    """

    max_false_fire: int = 0
    min_precision: float | None = None
    min_recall: float | None = None


def check_benchmark_gate(report: dict, gate: BenchmarkGate) -> tuple[bool, list[str]]:
    """Evaluate ``gate`` against a per-concern report. Returns ``(passed, reasons)``.

    ``false_fire`` is checked on the overall total (the absolute trust gate); precision/recall
    floors are checked **per concern** so a single weak concern fails the build with a precise
    reason, not a diluted average.
    """
    reasons: list[str] = []
    overall = report["overall"]
    if overall["false_fire"] > gate.max_false_fire:
        reasons.append(
            f"false-fire == {overall['false_fire']} > max {gate.max_false_fire} (trust gate)"
        )
    for concern_id, r in report["concerns"].items():
        if gate.min_precision is not None and r["precision"] < gate.min_precision:
            reasons.append(
                f"concern {concern_id!r}: precision {r['precision']:.3f} < min {gate.min_precision}"
            )
        if gate.min_recall is not None and r["recall"] < gate.min_recall:
            reasons.append(
                f"concern {concern_id!r}: recall {r['recall']:.3f} < min {gate.min_recall}"
            )
    return (not reasons), reasons


# ---------------------------------------------------------------------------
# Pretty-print + CLI
# ---------------------------------------------------------------------------


def format_report(report: dict) -> str:
    lines = [
        "",
        "=" * 78,
        "  PER-CONCERN BENCHMARK  (precision / recall / false-fire / refuse)",
        "=" * 78,
        "",
        f"  omega (reaction window): {report['omega']}s",
        "",
        f"  {'concern':<22} {'P':>6} {'R':>6} {'FF':>4} {'refuse':>7}   {'edges':>5} {'fires':>5} {'asks':>4}",
        f"  {'-'*22} {'-'*6} {'-'*6} {'-'*4} {'-'*7}   {'-'*5} {'-'*5} {'-'*4}",
    ]
    for cid, r in report["concerns"].items():
        lines.append(
            f"  {cid:<22} {r['precision']:>6.3f} {r['recall']:>6.3f} {r['false_fire']:>4} "
            f"{r['refuse_rate']:>7.3f}   {r['n_edges']:>5} {r['n_fires']:>5} {r['n_asks']:>4}"
        )
    o = report["overall"]
    lines += [
        f"  {'-'*22} {'-'*6} {'-'*6} {'-'*4} {'-'*7}   {'-'*5} {'-'*5} {'-'*4}",
        f"  {'OVERALL':<22} {o['precision']:>6.3f} {o['recall']:>6.3f} {o['false_fire']:>4} "
        f"{o['refuse_rate']:>7.3f}   {o['n_edges']:>5} {o['n_fires']:>5} {o['n_asks']:>4}",
        "",
        "=" * 78,
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover — CLI entry / IO
    import argparse
    import os

    parser = argparse.ArgumentParser(
        prog="python -m eval.benchmark",
        description="Run a labelled-clip benchmark manifest and emit a per-concern P/R report "
        "(offline; fake backend; no keys).",
    )
    parser.add_argument("manifest", help="path to the benchmark manifest JSON")
    parser.add_argument(
        "--out",
        default=os.environ.get("PERCEPT_BENCHMARK_OUT", "benchmark.json"),
        help="where to write the JSON report (default: benchmark.json)",
    )
    parser.add_argument(
        "--min-precision", type=float, default=None, help="per-concern precision floor for the gate"
    )
    parser.add_argument(
        "--min-recall", type=float, default=None, help="per-concern recall floor for the gate"
    )
    parser.add_argument(
        "--max-false-fire", type=int, default=0, help="overall false-fire ceiling (default 0)"
    )
    args = parser.parse_args(argv)

    manifest = Manifest.load(args.manifest)
    report = run_manifest(manifest)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(format_report(report))
    print(f"  wrote {args.out}")

    gate = BenchmarkGate(
        max_false_fire=args.max_false_fire,
        min_precision=args.min_precision,
        min_recall=args.min_recall,
    )
    passed, reasons = check_benchmark_gate(report, gate)
    if not passed:
        print("BENCHMARK GATE FAILED: " + "; ".join(reasons))
        return 1
    print("BENCHMARK GATE PASSED")
    return 0


__all__ = [
    "ConcernObservation",
    "ConcernReport",
    "Manifest",
    "ManifestClip",
    "BenchmarkGate",
    "score_concern",
    "per_concern_report",
    "edges_from_manifest",
    "run_manifest",
    "check_benchmark_gate",
    "format_report",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
