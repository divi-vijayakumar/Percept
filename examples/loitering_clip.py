"""A "judge me on a CLIP, not one frame" concern — the B1 temporal-judging path, fully offline.

Some conditions can't be read from a single still: "someone is *loitering*" looks identical to
"someone is standing there" in one frame — what makes it loitering is *dwelling over time*. So the
concern DECLARES it needs a clip via ``window_s`` (seconds), and the brain assembles a time-bounded
temporal strip from its rolling frame window and feeds it to the judge (``Vision.judge(frames=...)``).

Two ways to reach the temporal judge, both shown below:
  1. A standing concern that declared ``window_s`` — the watch loop assembles the clip per tick.
  2. The public ``judge_clip(condition, frames)`` seam — judge a clip you already hold, directly.

FakeVision is *scripted, not intelligent* (standing rule 2): a single "loiter" frame reads
inconclusive, but once a multi-frame strip reaches ``judge`` it returns the "dwelling" clip verdict —
proving the clip actually flows into perception. No keys, no camera, no network.

Run it:

    PYTHONPATH=packages/percept-vision python3 examples/loitering_clip.py
"""

import asyncio

import percept
from percept.adapters import Concern, Goal, Subject, Verdict
from percept.behavior import perception_for_concern
from percept.backends.fake import FakeVision


async def main() -> None:
    # Scripted: one "loiter" frame is inconclusive; a strip (clip) reads as dwelling -> satisfied.
    vision = FakeVision(
        script={
            "empty":        Verdict(satisfied=False, confidence=0.95, evidence="nobody there"),
            "loiter":       Verdict(satisfied=False, confidence=0.80, evidence="someone standing — one frame, can't tell"),
            "loiter|strip": Verdict(satisfied=True,  confidence=0.92, evidence="same person, dwelling across the clip -> loitering"),
        }
    )
    agent = percept.Percept.create(vision=vision, discover_plugins=False)

    # The concern DECLARES it needs a 4-second clip. `wants_clip` reflects the declaration, and
    # `perception_for_concern` maps it to the perception sub-dict the watch loop consumes.
    concern = Concern(
        id="loiter",
        identity=Subject(kind="entity", ref="person at the door"),
        condition="someone is loitering by the door",
        say="Someone's been lingering by the door for a while.",
        window_s=4.0,  # <-- "judge me on a 4s clip, not one frame"
    )
    print(f"concern.wants_clip = {concern.wants_clip}   (window_s={concern.window_s}s)")

    # Register it as a watch goal carrying the clip declaration in its perception state.
    state = {"type": "watch", "count": 0, **perception_for_concern(concern)}
    agent.add_goal(Goal(id=concern.id, condition=concern.condition, say=concern.say, state=state))

    print("\n[1] Standing concern — the watch loop assembles the clip per tick:")
    print("frame    t    -> events")
    print("-----------------------")
    sequence = [("empty", 0.0), ("loiter", 1.0), ("loiter", 2.0), ("loiter", 3.0), ("loiter", 4.0)]
    for token, t in sequence:
        events = await agent.perceive_judged(token, t=t)
        if not events:
            print(f"{token:<8} {t:<4} -> (silent)")
            continue
        for ev in events:
            print(f"{token:<8} {t:<4} -> {ev.action.upper():4} [{ev.goal_id}] {ev.text}")

    # [2] The public judge_clip seam: judge a clip you already hold (oldest -> newest), directly.
    print("\n[2] Direct judge_clip over a held clip:")
    clip = ["loiter", "loiter", "loiter", "loiter"]
    verdict = await agent.judge_clip(concern.condition, clip)
    print(f"judge_clip(...) -> satisfied={verdict.satisfied} conf={verdict.confidence} :: {verdict.evidence}")


if __name__ == "__main__":
    asyncio.run(main())
