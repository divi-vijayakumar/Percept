"""Replay-and-align runner: orchestrate the scorer and wire the CI gates.

Two modes:

* **L2 live** (``PERCEPT_LIVE=1`` + a vision key): replay each golden clip through a real
  ``Percept``, record fires, align to the ground-truth rising edges, roll up the scorecard, and
  enforce the hard CI gates. The Makefile drives this via ``python -m eval.run``.
* **offline self-check** (no key): the live replay is skipped (clean L2 skip), and the runner
  instead verifies the gate machinery on a tiny synthetic fixture so ``results.json`` is always
  emitted and the gate logic is exercised without a model.

**CI gates (hard fail):**

* ``false_fire == 0`` on the test split (precision = 1; the trust gate);
* ``false_serve <= bound`` (the cache's analog of false-fire). The bound is
  ``PERCEPT_CACHE_FALSE_SERVE_MAX`` (absolute count, default 0).

Frames are an opaque object; this runner replays whatever ``frames_for_scene`` yields. With no
real frame extractor wired (the golden corpus tracks labels, not pixels), the live path replays
the label-derived edge timeline so the *alignment* is exercised; real pixel replay is a drop-in
once a frame source is attached.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from percept.adapters import Goal
from percept.config import load_settings
from percept.eval import find_golden_root, load_golden

from ._dotenv import load_dotenv
from .replay import Edge, Fire, ReplayMetrics, score


# ---------------------------------------------------------------------------
# Replaying a clip through a Percept (records fires; the model/cache decide)
# ---------------------------------------------------------------------------


def replay_clip(
    agent: Any,
    goals: list[Goal],
    frames: Iterable[tuple[float, Any]],
    *,
    source: str = "camera",
) -> list[Fire]:
    """Replay a ``(ts, frame)`` sequence through ``agent`` and record every fire.

    Production cadence is the caller's job (the ``frames`` iterable carries the timestamps). Each
    ``perceive_judged`` returns gate actions; we record the ``FIRE`` ones as :class:`Fire`s, with
    cost/tokens stamped from the agent's metrics delta when the backend reports them (0 for fakes).
    """
    import asyncio

    for g in goals:
        agent.add_goal(g)

    async def run() -> list[Fire]:
        fires: list[Fire] = []
        for ts, frame in frames:
            events = await agent.perceive_judged(frame, source, t=ts)
            for ev in events:
                if ev.action == "fire":
                    conf = ev.verdict.confidence if ev.verdict is not None else 1.0
                    fires.append(Fire(goal_id=ev.goal_id, ts=ts, conf=conf))
        return fires

    return asyncio.run(run())


def edges_from_labels(labels: list[dict], goal_id_for: Callable[[dict, str], str]) -> list[Edge]:
    """Extract ground-truth rising edges from golden ``labels.jsonl`` rows.

    Each label row carries ``rising_edges: [{condition, t, ...}]``. ``goal_id_for(row, condition)``
    maps a (scene, condition) pair to the goal id used in the replay so alignment can match them.
    """
    edges: list[Edge] = []
    for row in labels:
        for re in row.get("rising_edges", []):
            cond = re.get("condition", "")
            t = re.get("t")
            if t is None:
                continue
            edges.append(Edge(goal_id=goal_id_for(row, cond), t=float(t)))
    return edges


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


@dataclass
class GateReport:
    false_fire: int
    false_serve: int
    false_serve_bound: int
    passed: bool
    reasons: list[str]


def check_gates(metrics: ReplayMetrics, false_serve: int, false_serve_bound: int) -> GateReport:
    """Enforce the hard CI gates: false-fire == 0 and false-serve <= bound."""
    reasons: list[str] = []
    if metrics.n_edges <= 0:
        reasons.append("no labelled rising edges were evaluated")
    if metrics.n_edges > 0 and metrics.n_fires <= 0:
        reasons.append(f"no fires observed for {metrics.n_edges} labelled rising edges")
    if metrics.false_fire != 0:
        reasons.append(f"false-fire == {metrics.false_fire} (gate requires 0)")
    if false_serve > false_serve_bound:
        reasons.append(
            f"false-serve == {false_serve} > bound {false_serve_bound}"
        )
    return GateReport(
        false_fire=metrics.false_fire,
        false_serve=false_serve,
        false_serve_bound=false_serve_bound,
        passed=not reasons,
        reasons=reasons,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

_OMEGA_DEFAULT = 2.0  # seconds; the reaction window. Env: PERCEPT_EVAL_OMEGA.


def _omega() -> float:
    raw = os.environ.get("PERCEPT_EVAL_OMEGA")
    try:
        return float(raw) if raw is not None else _OMEGA_DEFAULT
    except ValueError:
        return _OMEGA_DEFAULT


def _false_serve_bound() -> int:
    raw = os.environ.get("PERCEPT_CACHE_FALSE_SERVE_MAX")
    try:
        return int(raw) if raw is not None else 0
    except ValueError:
        return 0


def _selfcheck_metrics() -> tuple[ReplayMetrics, int]:
    """A tiny synthetic fixture so the gate machinery runs with no key (offline self-check).

    One edge, one in-window fire -> a clean scorecard (false-fire 0); false-serve 0. This is NOT
    an intelligence test — it only proves the *scorer + gate wiring* execute end-to-end offline.
    """
    omega = _omega()
    edges = [Edge("g1", 10.0)]
    fires = [Fire("g1", 10.5, conf=0.9, cost=0.2, tokens=128)]
    return score(fires, edges, omega), 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()  # pick up a local .env (real env still wins)
    settings = load_settings()
    live = os.environ.get("PERCEPT_LIVE") == "1"
    out_path = Path(os.environ.get("PERCEPT_EVAL_OUT", "results.json"))

    if not live:
        print(
            "L2 STATUS: SKIPPED — set PERCEPT_LIVE=1 (+ provider key and budget) for live qualification. "
            "Running the offline gate self-check instead."
        )
        metrics, false_serve = _selfcheck_metrics()
        mode = "selfcheck"
        status = "SKIPPED"
    else:
        from percept.config import missing_credential

        backend = settings.models.vision or "gemini"
        blockers: list[str] = []
        missing = missing_credential(backend)
        if missing:
            blockers.append(f"{backend} credential missing (set one of: {', '.join(missing)})")
        from eval.bench.budget import live_budget_blockers

        configured_model = os.environ.get("PERCEPT_VISION_MODEL")
        if not configured_model:
            blockers.append("PERCEPT_VISION_MODEL must explicitly pin the budgeted provider model")
        blockers.extend(
            live_budget_blockers(
                provider=backend,
                model=configured_model,
                raw=os.environ.get("PERCEPT_BENCH_BUDGET_PLAN"),
            )
        )
        if backend == "openrouter":
            blockers.append(
                "eval.run does not attach the OpenRouter request-budget controller; "
                "use the pinned pt-action-e2e benchmark runner"
            )
        root = find_golden_root()
        if root is None:
            blockers.append("golden fixture labels are missing")
        else:
            media = [
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}
            ]
            if not media:
                blockers.append("golden fixture media is missing; labels alone cannot qualify live perception")
        if blockers:
            report = {"status": "BLOCKED", "mode": "live", "blockers": blockers}
            out_path.write_text(json.dumps(report, indent=2))
            print("L2 STATUS: BLOCKED", file=sys.stderr)
            for blocker in blockers:
                print(f"  - {blocker}", file=sys.stderr)
            return 2
        assert root is not None
        golden = load_golden(root)
        # NOTE: a real frame extractor is wired here in the live build; until then the live path
        # scores the held-out test split's label-derived edges to exercise alignment + gates.
        test_scenes = set(golden.scenes("test"))
        test_labels = [r for r in golden.labels if r.get("scene_id") in test_scenes]
        edges = edges_from_labels(test_labels, lambda row, cond: f'{row["scene_id"]}::{cond}')
        # Without a live frame source, we have no fires to record yet -> an empty fire list is a
        # recall floor, never a false-fire. (Replace with replay_clip once frames are attached.)
        metrics = score([], edges, _omega())
        false_serve = 0
        mode = "live"
        status = "PASS"

    bound = _false_serve_bound()
    gates = check_gates(metrics, false_serve, bound)

    report = {
        "status": status if gates.passed else "FAILED",
        "mode": mode,
        "omega": metrics.omega,
        "metrics": metrics.as_dict(),
        "gates": {
            "false_fire": gates.false_fire,
            "false_serve": gates.false_serve,
            "false_serve_bound": gates.false_serve_bound,
            "passed": gates.passed,
            "reasons": gates.reasons,
        },
        "settings": {
            "cache_enabled": settings.cache.enabled,
            "cache_ttl_s": settings.cache.ttl_s,
            "cache_tau_bits": settings.cache.tau_bits,
            "cache_audit_rate": settings.cache.audit_rate,
        },
    }
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    if not gates.passed:
        print("L2 STATUS: FAILED — " + "; ".join(gates.reasons), file=sys.stderr)
        return 1
    if live:
        print("L2 STATUS: PASS")
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI entry
    raise SystemExit(main())
