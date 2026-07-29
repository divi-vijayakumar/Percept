---
title: Scenario: tell me when a package is on the porch
updated: 2026-07-20
applies_to: percept-vision 1.0.0
status: stable
---
# Scenario: tell me when a package is on the porch

A **physical** event — the template for "alert me when X happens in the room/frame."

Here's how to set it up:

```python
agent.add_goal(Goal(
    id="porch-package",
    condition="a package has been left on the porch",
    say="A package was dropped off",
    state={
        "perception": {"temporal": "strip", "n_frames": 3},  # see the DROP-OFF, not just "box present"
        "miss_cost": "moderate",
    },
))
async for incident in run(agent, source="rtsp://porch-cam/stream", on_incident=notify):
    ...
```

Why each knob:
- **`temporal="strip"`** — a single frame of "box near the door" is ambiguous (was it always there?). The
  strip shows the *drop-off* (someone approaching and leaving), which is the distinguishing signal.
- **`miss_cost="moderate"`** — a missed delivery notice is an inconvenience, not an emergency; the default
  gate balance is fine (no need to bias toward asking on every ambiguous read).
- **motion gate** (default) — a delivery is a discrete, high-motion event, so the cheap gate escalates
  reliably.

Recall caveat: **fast drop-and-go** the gate catches well; a package that's **already there** when the
stream starts won't trigger a rising edge (percept fires on *becoming true*, not on steady state) — pair
with a periodic still-frame check if "is anything on the porch right now" matters more than "did a
delivery just happen."

> **Assistant-class, not a guarantee.** The runtime *alerts a human who stays responsible*; it doesn't
> replace a security system, and a missed or delayed notification is not a safety failure.
