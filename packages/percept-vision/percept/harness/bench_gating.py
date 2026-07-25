"""bench_gating — the Phase-6 'cost graph vs ungated' demo (deterministic, runnable).

Replays a synthetic frame sequence (mostly still, occasional motion) through the wired
WatchSpec↓/Tier0Signal↑ loop and prints **vision calls gated vs ungated + the % reduction**.

The point is *our gating economics*, not model intelligence: a still scene spends no cloud calls,
motion escalates. Backends are the offline fakes (deterministic, no keys). Run::

    PYTHONPATH=packages/percept-vision \
        python3 -m percept.harness.bench_gating
"""

from __future__ import annotations

import asyncio

from percept.adapters import Goal
from percept.agent import Percept

from .tier0 import Frame, GatedWatchLoop, gray_frame

# A small grid is plenty for a coarse motion gate (and keeps the bench instant + deterministic).
GRID_W, GRID_H = 8, 8
# How many "motion" pixels a moving frame flips, and to what value (drives the motion score high).
MOTION_PIXELS = 40
MOTION_VALUE = 255
# A frame index is a "motion" frame iff it is in this set (occasional motion over a still scene).
MOTION_FRAMES = frozenset({5, 6, 14, 23, 24, 38})
TOTAL_FRAMES = 48


def _still() -> Frame:
    """A dark, uniform still scene (the baseline the camera mostly sees)."""
    return gray_frame(GRID_W, GRID_H, fill=0)


def _motion() -> Frame:
    """A frame with a bright patch flipped on — a large, deterministic change from still."""
    pixels = [0] * (GRID_W * GRID_H)
    for i in range(min(MOTION_PIXELS, len(pixels))):
        pixels[i] = MOTION_VALUE
    return Frame(w=GRID_W, h=GRID_H, pixels=tuple(pixels))


def synthetic_stream() -> list[Frame]:
    """A deterministic mostly-still sequence with occasional motion bursts."""
    return [_motion() if i in MOTION_FRAMES else _still() for i in range(TOTAL_FRAMES)]


def _demo_goals() -> list[Goal]:
    return [
        Goal(id="g_coffee", condition="a coffee cup is on the desk", say="that looks like coffee"),
        Goal(id="g_person", condition="a person entered the frame", say="someone's here"),
    ]


async def _run() -> dict:
    # Explicit mode is part of the benchmark contract. This gate proves deterministic
    # programmable-Watch execution and must never inherit a caller's live/default mode.
    core = Percept.create(mode="offline", discover_plugins=False)
    goals = _demo_goals()
    for g in goals:
        core.add_goal(g)

    loop = GatedWatchLoop(core, goals)
    report = await loop.run(synthetic_stream(), clock_start=0.0, dt=1.0)
    return {
        "spec_conditions": len(loop.spec.conditions),
        "threshold": loop.gate.threshold(),
        "report": report,
    }


def run_benchmark() -> dict:
    info = asyncio.run(_run())
    report = info["report"]
    ungated = report.ungated_calls
    gated = report.vision_calls
    pct = report.reduction * 100.0

    print("=== Phase 6 tier-0 gating — cost graph vs ungated ===")
    print(f"WatchSpec conditions compiled : {info['spec_conditions']}")
    print(f"effective motion threshold    : {info['threshold']:.3f}")
    print(f"frames replayed               : {report.frames}")
    print(f"escalations (salience cleared): {report.escalations}")
    print(f"vision calls (gated)          : {gated}")
    print(f"vision calls (ungated)        : {ungated}")
    print(f"reduction                     : {pct:.1f}%  ({ungated} -> {gated})")
    print(f"Tier0Signals reported up      : {len(report.signals)}")
    return info


def main() -> int:
    info = run_benchmark()
    report = info["report"]
    failures: list[str] = []
    if report.frames <= 0:
        failures.append("no frames executed")
    if report.ungated_calls <= 0:
        failures.append("ungated reference made no calls")
    if report.vision_calls >= report.ungated_calls:
        failures.append(
            f"no measurable call reduction ({report.ungated_calls} -> {report.vision_calls})"
        )
    if report.escalations <= 0 or not report.signals:
        failures.append("the deterministic stream produced no salience escalation")
    if failures:
        print("CALL-REDUCTION DIAGNOSTIC: FAIL — " + "; ".join(failures))
        return 1
    print(
        "CALL-REDUCTION DIAGNOSTIC: PASS — deterministic programmable Watches executed offline; "
        f"vision calls reduced {report.ungated_calls} -> {report.vision_calls}. "
        "This is not a matched-quality/same-archetype comparison and does not qualify launch "
        "quality, provider pricing, or Cloud cost."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - runnable demo entrypoint
    raise SystemExit(main())
