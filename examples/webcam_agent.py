"""A real-backend "watch X" agent: cv2 webcam -> Gemini vision -> fires on the rising edge.

This is the SAME agent as watch_agent.py with one substitution — the eyes. Everything above
the perception seam (the goal, the three-state gate, the rising-edge firing) is identical;
only `vision="fake"` becomes `vision="gemini"` and the frames become real JPEG bytes.

    ┌─ DO NOT run this without a camera + key. It is a correct, copy-pasteable template. ─┐

Requirements:

    pip install "percept-vision[gemini]" opencv-python
    export GEMINI_API_KEY=...        # or GOOGLE_API_KEY

Then:

    python3 examples/webcam_agent.py

Frame format for the gemini/claude backends: image bytes, or a (media_type, bytes) pair.
We encode each BGR webcam frame to JPEG and pass ("image/jpeg", <bytes>) — the explicit,
unambiguous form. (The fake backend, by contrast, takes a plain string token.)

Cost / latency note: each judged frame is one ~1s cloud VLM call per active goal. Sample
slowly (e.g. ~1 fps, as below) rather than judging every camera frame — the gate fires once
on the rising edge regardless of how densely you sample.
"""

import asyncio

import cv2  # pip install opencv-python

import percept
from percept.adapters import Goal


def encode_frame(bgr) -> tuple[str, bytes]:
    """Encode an OpenCV BGR frame to the (media_type, bytes) pair the VLM backends accept."""
    ok, buf = cv2.imencode(".jpg", bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed to encode the frame")
    return ("image/jpeg", buf.tobytes())


async def main() -> None:
    # Resolve the real backend BY NAME via the registry. Needs percept-vision[gemini] +
    # GEMINI_API_KEY (or GOOGLE_API_KEY). discover_plugins=False keeps resolution to built-ins.
    agent = percept.Percept.create(vision="gemini", discover_plugins=False)

    agent.add_goal(
        Goal(
            id="caffeine",
            condition="the user is drinking coffee",
            say="Heads up — stepping back from caffeine?",
        )
    )

    cap = cv2.VideoCapture(0)  # default webcam
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (device 0). Is a camera connected?")

    print("Watching... (Ctrl-C to stop)")
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                # Transient read failure — skip this tick rather than crash.
                await asyncio.sleep(1.0)
                continue

            frame = encode_frame(bgr)  # ("image/jpeg", bytes)

            # YOU feed frames — Percept has no built-in capture. The cognition layer below this
            # call is identical to the offline example; only the eyes (and frame type) changed.
            events = await agent.perceive_judged(frame, source="camera")
            for ev in events:  # FireEvent(goal_id, action, text, ...); action is "fire" or "ask"
                print(f"[{ev.action.upper()}] {ev.goal_id}: {ev.text}")

            # Sample ~1 fps: each judged frame is a cloud VLM call. The rising-edge gate means
            # slower sampling does NOT cause missed or duplicate fires for a sustained condition.
            await asyncio.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()


if __name__ == "__main__":
    asyncio.run(main())
