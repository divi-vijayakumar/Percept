"""A runnable, fully-offline "watch X" agent — no keys, no camera, no network.

Pattern: state a goal in plain language, feed frames, and let the three-state gate fire
ONCE on the false->true rising edge. The fake vision backend is *scripted, not intelligent*:
it maps a string "frame" token to a pre-written Verdict, so you control perception by hand
and the cognition layer (gate + rising-edge logic) does the real work.

Run it (no install needed — point PYTHONPATH at the package):

    PYTHONPATH=packages/percept-vision python3 examples/watch_agent.py

Or, after `pip install percept-vision`:

    python3 examples/watch_agent.py
"""

import asyncio

import percept
from percept.adapters import Goal, Verdict
from percept.backends.fake import FakeVision


async def main() -> None:
    # FakeVision maps each string "frame" token to a scripted Verdict (no model, no keys).
    # satisfied=False keeps the gate silent; satisfied=True with confidence >= 0.7 can fire.
    vision = FakeVision(
        script={
            "empty-hands": Verdict(satisfied=False, confidence=0.95, evidence="no cup"),
            "holding-cup": Verdict(satisfied=False, confidence=0.90, evidence="cup, not sipping"),
            "drinking":    Verdict(satisfied=True,  confidence=0.92, evidence="sipping from cup"),
        }
    )

    # discover_plugins=False skips third-party backend lookup — pure offline.
    agent = percept.Percept.create(mode="offline", vision=vision, discover_plugins=False)

    agent.add_goal(
        Goal(
            id="caffeine",
            condition="the user is drinking coffee",
            say="Heads up — stepping back from caffeine?",
        )
    )

    # A scripted frame sequence (token, timestamp). The condition becomes true at t=3.0.
    sequence = [
        ("empty-hands", 1.0),
        ("holding-cup", 2.0),
        ("drinking",    3.0),  # rising edge: false -> true  => exactly one FIRE
        ("drinking",    4.0),  # still true => refractory holds it SILENT (no double-fire)
        ("empty-hands", 5.0),  # confident-false re-arms the edge
        ("drinking",    6.0),  # rising edge again => FIRE
    ]

    print("frame          t    -> events")
    print("-------------------------------")
    for token, t in sequence:
        events = await agent.perceive_judged(token, t=t)
        if not events:
            print(f"{token:<14} {t:<4} -> (silent)")
            continue
        for ev in events:  # ev is a FireEvent(goal_id, action, text, ...)
            print(f"{token:<14} {t:<4} -> {ev.action.upper():4}  [{ev.goal_id}]  {ev.text}")


if __name__ == "__main__":
    asyncio.run(main())
