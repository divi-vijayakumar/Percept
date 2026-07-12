"""A live Percept "watch" agent — `percept.run` owns the capture loop.

Watches a real stream for a plain-language condition and FIRES once on the rising
edge (false -> true). `percept.run` opens the source, runs the tier-0 motion gate
(a still scene spends no cloud calls), judges each frame, and yields an `Incident`
on each rising-edge fire. You do NOT write a capture loop.

Run it (needs a perceiving backend + a key):
    pip install "percept-vision[gemini]"   # + runtime/opencv extras
    export GEMINI_API_KEY=...               # (or GOOGLE_API_KEY)
    python template_watch_agent_live.py

To point it at a different feed, change `SOURCE` below:
    "webcam:0"          # a local camera (index 0)
    "rtsp://…"          # an IP camera / RTSP stream
    "/path/to/clip.mp4" # a video file
Swap `vision="gemini"` for "claude" (+ ANTHROPIC_API_KEY) to use a different VLM;
everything above the perception seam (gate, rising edge, memory) is identical.
"""

import asyncio

import percept
from percept.adapters import Goal

# --- WHERE to watch --------------------------------------------------------
SOURCE = "webcam:0"  # "webcam:0" | "rtsp://…" | "/path/to/video.mp4"

# --- WHAT to watch for (plain language) ------------------------------------
GOAL = Goal(
    id="kettle",
    condition="the kettle has finished boiling",
    say="The kettle's done — go make your tea.",
)


def on_incident(incident) -> None:
    """The notification sink — SMS / Slack / webhook goes here.

    Called once per rising-edge fire with a durable `Incident` record.
    """
    print(f"[FIRE] {incident.concern_id}: {incident.message}")


def on_ask(event) -> None:
    """Optional refusal sink: low-confidence reads ASK instead of guessing."""
    print(f"[ASK ] {event.goal_id}: {event.text}")


async def main() -> None:
    # vision="gemini" | "claude" (+ its API key). discover_plugins=False skips plugin lookup.
    agent = percept.Percept.create(vision="gemini", discover_plugins=False)
    agent.add_goal(GOAL)

    # `run` opens SOURCE, gates + judges, and yields an Incident on each fire.
    async for incident in percept.run(
        agent,
        source=SOURCE,
        on_incident=on_incident,
        on_ask=on_ask,
    ):
        # The async-for body also sees every fired Incident (same object as on_incident).
        print(incident.concern_id, incident.message)


if __name__ == "__main__":
    asyncio.run(main())
