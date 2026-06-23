"""A minimal Percept "watch" agent — offline, no keys.

Watches a stream for a plain-language condition and FIRES once on the rising edge
(false -> true). Refuses to guess: a low-confidence read is an ASK, not a fire.

Run it (offline, FAKE backend, deterministic):
    pip install percept-vision
    python template_watch_agent.py

To make it real, swap the FAKE vision for a VLM and feed real frames:
    pip install "percept-vision[gemini]"   # + export GEMINI_API_KEY=...
    # then replace `build_agent()` body with:
    #     return percept.Percept.create(vision="gemini", discover_plugins=False)
    # and replace `scripted_frames()` with a camera/video source that yields
    # image bytes (e.g. cv2.imencode(".jpg", bgr)[1].tobytes()).
"""

import asyncio

import percept
from percept.adapters import Goal, Verdict
from percept.backends.fake import FakeVision

# --- WHAT to watch for (plain language) -------------------------------------
GOAL = Goal(
    id="kettle",
    condition="the kettle has finished boiling",
    say="The kettle's done — go make your tea.",
)


def build_agent() -> "percept.Percept":
    """Build the agent. Offline: a FakeVision scripts each frame token to a Verdict.

    The FAKE backend is *scripted, not intelligent* — it maps a string frame token
    to a pre-written Verdict so the cognition layer (gate, rising edge) does the real
    work while you control perception by hand. Swap `vision=` for "gemini"/"claude"
    (+ a key) to perceive the real world; everything below the seam is identical.
    """
    vision = FakeVision(
        script={
            # token              -> what the "eyes" report for this frame
            "rolling-boil": Verdict(satisfied=False, confidence=0.95, evidence="still boiling"),
            "switched-off": Verdict(satisfied=True, confidence=0.95, evidence="steam stopped"),
            "blurry": Verdict(satisfied=True, confidence=0.30, evidence="cannot tell — blurry"),
        }
    )
    agent = percept.Percept.create(vision=vision, discover_plugins=False)
    agent.add_goal(GOAL)
    return agent


def scripted_frames():
    """Yield (frame, t) pairs. Offline, a frame is just a string token the FAKE
    backend understands. With a real backend, a frame is image bytes from a camera.
    """
    yield ("rolling-boil", 1.0)  # condition false   -> SILENT
    yield ("rolling-boil", 2.0)  # still false        -> SILENT
    yield ("blurry", 3.0)        # low confidence     -> ASK (refuses to guess)
    yield ("switched-off", 4.0)  # false -> true      -> FIRE (rising edge)
    yield ("switched-off", 5.0)  # still true         -> SILENT (refractory: no nag)


def handle(ev) -> None:
    """Do something with a gate event. `ev.action` is "fire" or "ask"."""
    print(f"[{ev.action.upper():4}] {ev.goal_id}: {ev.text}")


async def main() -> None:
    agent = build_agent()
    for frame, t in scripted_frames():
        # Per frame: judge every active goal, run the verdict through the three-state
        # gate, and return the rising-edge events (FIRE / ASK). SILENT yields nothing.
        for ev in await agent.perceive_judged(frame, t=t):
            handle(ev)


if __name__ == "__main__":
    asyncio.run(main())
