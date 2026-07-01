"""Scenario: help me set up Claude Code, by watching my screenshare.

A CONTENT event — the opposite of the fall. A screen changes *content* (text appears, a command runs)
with **~zero pixel motion**, so the motion gate would never escalate and the assistant would silently
under-sample. The fix is the per-goal temporal CADENCE: `gate_on="fixed"` looks every `min_judge_interval`
seconds *regardless of motion*.

Runnable offline (FakeVision scripts the screen; the gate frames are a STILL screen to prove the cadence
escalates with no motion). Go live by passing an iterable screen source:
    source=my_screen_frames()  (install `percept-vision[screen]`; see docs/scenarios/screenshare.md)

    python examples/screenshare_assistant.py
"""
import asyncio

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.backends.fake import FakeVision
from percept.edge.tier0 import gray_frame
from percept.runtime import run


def show_tip(incident):
    print(f"[TIP] {incident.message}")
    print(f"      (saw: {incident.verdict.evidence})\n")


def build_agent() -> Percept:
    vision = FakeVision(
        script={
            "normal": Verdict(satisfied=False, confidence=0.95, evidence="terminal looks fine"),
            "missing-key": Verdict(
                satisfied=True, confidence=0.9,
                evidence="the terminal shows an authentication error - ANTHROPIC_API_KEY is unset",
            ),
        }
    )
    agent = Percept.create(vision=vision, discover_plugins=False)
    agent.add_goal(Goal(
        id="cc-setup",
        condition="the user is stuck on a Claude Code setup step - a visible error, a missing API key, a wrong command",
        say="Looks like ANTHROPIC_API_KEY isn't set - set it in your shell or .env (export ANTHROPIC_API_KEY=<your_key>)",
        state={
            "perception": {"temporal": "strip", "n_frames": 3},
            # the screen has ~0 pixel motion -> the motion gate never escalates -> look on a FIXED cadence
            "cadence": {"min_judge_interval": 0.5, "gate_on": "fixed"},
        },
    ))
    return agent


SCREEN = gray_frame(8, 8, 120)  # a STILL screen — constant pixels, zero motion (the whole point)
# the motion gate alone would NEVER escalate on this; gate_on="fixed" looks every 0.5s anyway
STREAM = [
    (0.0, SCREEN, "normal"),
    (0.5, SCREEN, "normal"),
    (1.0, SCREEN, "missing-key"),  # fixed cadence escalates with no motion -> "missing-key" -> FIRE -> tip
    (1.5, SCREEN, "missing-key"),  # refractory holds (already alerted)
]


async def main():
    print("watching the screenshare (fixed cadence, no motion needed)...\n")
    async for _ in run(build_agent(), source=STREAM, on_incident=show_tip):
        pass
    print("stream ended.")


if __name__ == "__main__":
    asyncio.run(main())
