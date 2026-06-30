# Scenario: notify me when someone falls

A **physical** event — the template for "alert me when X happens in the room."

> ⚠️ **Assistant-class, not a safety device.** Percept *alerts a human who stays responsible.* It is
> **not** a certified medical fall-detector and must not be the sole safety mechanism (a pendant /
> medical device is). Set this expectation with anyone deploying it.

Full runnable demo: [`examples/fall_notification.py`](../../examples/fall_notification.py).

```python
agent.add_goal(Goal(
    id="fall",
    condition="a person has fallen and is on the floor",
    say="Possible fall - check on them",
    state={
        "perception": {"temporal": "strip", "n_frames": 4},  # see the DESCENT, not just "on the floor"
        "miss_cost": "severe",                                # a missed fall >> a false alarm -> bias to alert
    },
))
async for incident in run(agent, source="rtsp://room-cam/stream", on_incident=notify):
    ...
```

Why each knob:
- **`temporal="strip"`** — a single frame of "person on the floor" is also yoga or a push-up. The strip
  shows the *fall* (the rapid descent), which is the distinguishing signal.
- **`miss_cost="severe"`** — biases the three-state gate: on an ambiguous mid-fall read it **asks**
  ("can you confirm?") rather than staying silent. (The demo shows this ASK, then a FIRE on "on-floor".)
- **motion gate** (default) — a fall is high-motion, so the cheap gate escalates reliably.

Recall caveat: a **fast fall** the gate catches well; a **slow slump** (low motion) risks being skipped —
tune the gate threshold for the deployment, and see [motion-and-strip](../explanation/motion-and-strip.md).
For better cost/recall, register a fall **edge skill** (pose-centroid vertical velocity) — framework
supported, not yet shipped.
