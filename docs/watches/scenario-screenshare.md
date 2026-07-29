---
title: Scenario: help me by watching my screenshare
updated: 2026-07-16
applies_to: percept-vision 1.0.0
status: stable
---
# Scenario: help me by watching my screenshare

A **content** event — the template for "watch my screen and help / tell me when X." The opposite of a
physical event: the screen changes *content* with **~zero pixel motion**, so the motion gate would never
escalate. The fix is a **fixed cadence**.

Here's how to set it up:

```python
agent.add_goal(Goal(
    id="cc-setup",
    condition="the user is stuck on a Claude Code setup step - a visible error, a missing API key, a wrong command",
    say="Looks like ANTHROPIC_API_KEY isn't set - set it in your shell or .env (export ANTHROPIC_API_KEY=<your_key>)",
    state={
        "perception": {"temporal": "strip", "n_frames": 3},
        "cadence": {"min_judge_interval": 0.5, "gate_on": "fixed"},  # screen content != pixel motion
    },
))
async for incident in run(agent, source=my_screen_frames(), on_incident=show_tip):
    ...
```

Why `gate_on="fixed"`: the screenshare is nearly all still pixels, so `motion ≈ 0` every frame — the
motion gate would **never** spend a cloud call, and the assistant would see nothing. `gate_on="fixed"`
makes the gate escalate **every `min_judge_interval` seconds regardless of motion** (the demo fires on a
constant, zero-motion screen). See [motion-and-strip](../admission/how-admission-works.md) for why.

**Live screen source.** `pip install "percept-vision[screen]"` installs the screen-grab dependency
(`mss`), but the runtime intentionally keeps the capture adapter thin: capture the display, downsample
to a gray `Frame` for the gate + JPEG for the judge, and yield `(t, gate_frame, judge_frame)` (the same
shape the iterable source uses). Wire that iterable into `source=` until the built-in `screen://` adapter
lands.
