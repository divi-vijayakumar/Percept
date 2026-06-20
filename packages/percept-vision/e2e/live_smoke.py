"""L3-lite live smoke — the real backends, end to end, on a couple of calls.

This is the smallest honest "does the real system work" check: it builds a real ``Percept`` with
the Claude backends and runs two real calls —

  1. **vision**: judge a real golden frame against a condition -> a well-formed ``Verdict``;
  2. **conductor**: turn an utterance into a ``Decision`` that authors a goal.

It is **gated**: with no ``PERCEPT_LIVE=1`` + ``ANTHROPIC_API_KEY`` (loaded from ``.env``) it prints
a clean skip and exits 0, so it is safe in CI and offline. Kept tiny on purpose — a few calls, not
the full corpus scorecard (that is ``eval.run``). Run via ``make e2e``.
"""

from __future__ import annotations

import asyncio
import os

from eval._dotenv import load_dotenv
from eval.frames import first_frame
from percept.eval import find_golden_root, load_golden
from percept.adapters import Verdict
from percept.agent import Percept


def _skip(reason: str) -> int:
    print(f"e2e live smoke SKIPPED: {reason}")
    return 0


def _pick_scene_and_condition():
    """An approved golden scene that has pixels on disk + a gate-case condition to judge."""
    root = find_golden_root()
    if root is None:
        return None
    golden = load_golden(root)
    approved = {
        gc["scene_id"]: gc
        for gc in golden.gate_cases
        if gc.get("review_status") == "approved"
    }
    for sid, gc in approved.items():
        frame = first_frame(root / "scenes" / sid)
        if frame is not None:
            return root, sid, gc, frame
    return None


async def _run() -> int:
    a = Percept.create(vision="claude", conductor="claude", discover_plugins=False)

    # 1) vision — judge a real frame
    picked = _pick_scene_and_condition()
    if picked is None:
        print("  [vision] no golden pixels on disk — skipping the vision call")
    else:
        _root, sid, gc, (ts, frame) = picked
        condition = gc["goal"]["condition"]
        print(f"  [vision] scene={sid} t={ts:.2f}s  condition={condition!r} "
              f"(golden expects: {gc['expected_verdict']})")
        verdict: Verdict = await a.vision.judge(condition, frame)
        assert isinstance(verdict.satisfied, bool), "satisfied must be bool"
        assert 0.0 <= verdict.confidence <= 1.0, "confidence must be in [0,1]"
        print(f"  [vision] -> satisfied={verdict.satisfied} conf={verdict.confidence:.2f} "
              f"evidence={verdict.evidence!r}")

    # 2) conductor — author a goal from an utterance
    utterance = "watch my coffee and nudge me if I start drinking it"
    print(f"  [conductor] utterance={utterance!r}")
    decision = await a.conductor.run("speech_endpoint", utterance, {})
    print(f"  [conductor] -> route={decision.route} goal_ops={len(decision.goal_ops)}")
    assert decision.goal_ops or decision.route in {
        "task_with_tools",
        "proactive_speak",
        "respond_direct",
        "respond_with_vision",
    }, "conductor should route or author a goal"

    print("e2e live smoke PASSED")
    return 0


def main() -> int:
    load_dotenv()
    if os.environ.get("PERCEPT_LIVE") != "1":
        return _skip("set PERCEPT_LIVE=1 to run live")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return _skip("ANTHROPIC_API_KEY not set (paste it into .env)")
    try:
        import anthropic  # noqa: F401
    except ModuleNotFoundError:
        return _skip("anthropic SDK not installed (pip install anthropic)")
    return asyncio.run(_run())


if __name__ == "__main__":  # pragma: no cover — CLI entry
    raise SystemExit(main())
