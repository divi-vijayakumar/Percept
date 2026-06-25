"""Scenario: send a notification when someone falls.

A PHYSICAL event — motion-gated, high-consequence (a missed fall >> a false alarm), and judged on a
temporal STRIP so the VLM sees the *descent*, not just "a person on the floor" (which is also yoga).

Runnable offline (FakeVision scripts the fall). Go live by swapping two lines:
    vision="gemini"  (pip install "percept-vision[gemini]" + GEMINI_API_KEY)
    source="rtsp://door-cam/stream"  (pip install "percept-vision[runtime]")

    python examples/fall_notification.py

⚠️ Assistant-class, not a safety device: Percept *alerts a human who stays responsible*. It is not a
certified medical fall-detector and must not be the sole safety mechanism.
"""
import asyncio

from percept import Goal, Percept
from percept.adapters import Verdict
from percept.backends.fake import FakeVision
from percept.runtime import run
from percept_harness.tier0 import gray_frame


def notify(incident):
    """The notification sink — your SMS / Slack / webhook goes here. The incident carries the alert
    message, the VLM's evidence, and a pre-roll snippet (the clip) for the caregiver to verify."""
    print(f"[ALERT] {incident.message}")
    print(f"        why: {incident.verdict.evidence}  (confidence {incident.verdict.confidence:.2f})")
    snip = incident.snippet
    print(f"        clip: {len(snip.frames)} frames  t={snip.t_start:.1f}-{snip.t_end:.1f}s\n")


def build_agent() -> Percept:
    vision = FakeVision(  # live: Percept.create(vision="gemini", ...)
        script={
            "upright": Verdict(satisfied=False, confidence=0.95, evidence="person standing"),
            "falling": Verdict(satisfied=False, confidence=0.40, evidence="motion blur, mid-fall"),
            "on-floor": Verdict(satisfied=True, confidence=0.92, evidence="person on the floor, not moving"),
        }
    )
    agent = Percept.create(vision=vision, discover_plugins=False)
    agent.add_goal(Goal(
        id="fall",
        condition="a person has fallen and is on the floor",
        say="Possible fall - check on them",
        state={
            "perception": {"temporal": "strip", "n_frames": 4},  # see the DESCENT, not just "on floor"
            "miss_cost": "severe",                                # bias toward alerting
        },
    ))
    return agent


STILL, MOVED = gray_frame(8, 8, 0), gray_frame(8, 8, 255)
# a scripted stream: standing -> the fall (motion spike) -> settled on the floor (the rising edge)
STREAM = [
    (0.0, STILL, "upright"),
    (1.0, MOVED, "falling"),    # the fall: big motion -> escalate -> ambiguous (0.40, satisfied=False) -> silent
    (2.0, STILL, "on-floor"),   # settled: big motion -> escalate -> "on-floor" (satisfied) -> FIRE -> alert
    (3.0, STILL, "on-floor"),   # still down: no motion -> no cloud call (refractory holds anyway)
]


async def main():
    print("watching for a fall...\n")
    async for _ in run(build_agent(), source=STREAM, on_incident=notify):
        pass
    print("stream ended.")


if __name__ == "__main__":
    asyncio.run(main())
