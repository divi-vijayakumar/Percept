# Scenario: help me by watching my screenshare

A **content** event — the template for "watch my screen and help / tell me when X." The opposite of a
physical event: the screen changes *content* with **~zero pixel motion**, so the motion gate would never
escalate. The fix is a **fixed cadence**.

Full runnable demo: [`examples/screenshare_assistant.py`](../../examples/screenshare_assistant.py).

```python
agent.add_goal(Goal(
    id="cc-setup",
    condition="the user is stuck on a Claude Code setup step - a visible error, a missing API key, a wrong command",
    say="Looks like ANTHROPIC_API_KEY isn't set - run:  export ANTHROPIC_API_KEY=sk-...",
    state={
        "perception": {"temporal": "strip", "n_frames": 3},
        "cadence": {"min_judge_interval": 0.5, "gate_on": "fixed"},  # screen content != pixel motion
    },
))
async for incident in run(agent, source="screen://0", on_incident=show_tip):
    ...
```

Why `gate_on="fixed"`: the screenshare is nearly all still pixels, so `motion ≈ 0` every frame — the
motion gate would **never** spend a cloud call, and the assistant would see nothing. `gate_on="fixed"`
makes the gate escalate **every `min_judge_interval` seconds regardless of motion** (the demo fires on a
constant, zero-motion screen). See [motion-and-strip](../concepts/motion-and-strip.md) for why.

**Live screen source.** `source="screen://0"` needs a screen grabber. v1 ships the camera/RTSP source
(`[runtime]`); a screen source is a thin adapter — capture the display with `mss`, downsample to a gray
`Frame` for the gate + JPEG for the judge, and yield `(t, gate_frame, judge_frame)` (the same shape the
iterable source uses). Wire it as the `source=` iterable until the `[screen]` extra lands.
