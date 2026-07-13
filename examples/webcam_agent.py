"""A real-backend "watch X" agent: webcam -> Gemini vision -> speaks/notifies on the rising edge.

The batteries-included path: `percept.run` OWNS the camera → tier-0 gate → judge → incident lifecycle.
You bring a goal and a sink; Percept brings the capture loop, the motion gate (only spends a cloud
call when something moves), the three-state gate (fires once on the rising edge, ASKs when unsure),
and a durable Incident (with a pre-roll snippet) per fire.

    ┌─ Needs a camera + key. This is a correct, copy-pasteable template. ─┐

Requirements:

    pip install "percept-vision[gemini,runtime]"   # gemini judge + opencv camera capture
    export GEMINI_API_KEY=...                        # or GOOGLE_API_KEY

Then:

    python3 examples/webcam_agent.py

Source URIs `percept.run` accepts: "webcam:0" / "webcam://0" (a local camera), an "rtsp://..." URL,
or a video file path. No manual cv2 loop, no frame encoding — the runtime does it.

Add sound with two words: `audio_source="mic:0"` (needs the [mic] extra) turns this into a fused
audio+video watch — an off-camera sound can wake the gate, and a `modality="fused"` goal hears it.
"""

import asyncio
import os

import percept
from percept.adapters import Goal


async def main() -> None:
    # Real-backend example: degrade to a one-line instruction (never a stack trace) when the key is
    # absent. For a fully offline run with no keys, see examples/watch_agent.py (fake backend).
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        print('Set GEMINI_API_KEY (or GOOGLE_API_KEY) to run the live webcam agent, '
              'or try the offline examples/watch_agent.py (no keys).')
        return

    # Code-armed watch: goals are armed below via add_goal(), there is no voice/on_utterance, so the
    # default conductor stays SCRIPTED — this constructs with a loud "language goal-authoring is off"
    # log (not a crash). Needs GEMINI_API_KEY for the real vision judge; keyless? use mode="offline".
    agent = percept.Percept.create(vision="gemini", discover_plugins=False)
    agent.add_goal(
        Goal(
            id="caffeine",
            condition="the user is drinking coffee",
            say="Heads up — stepping back from caffeine?",
        )
    )

    print("Watching webcam:0 ... (Ctrl-C to stop)")
    # percept.run owns capture + gate + judge; it YIELDS an Incident on each rising-edge fire. The
    # motion gate means a still scene spends no cloud calls; the three-state gate means one fire per
    # rising edge (no nagging) and an ASK instead of a confident-wrong alert.
    try:
        async for incident in percept.run(agent, source="webcam:0", on_incident=None):
            print(f"[FIRE] {incident.concern_id}: {incident.message}")
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    asyncio.run(main())
