# How the gate works: motion (t0 vs t1), the strip, and cadence

Three separate decisions drive *when* and *how* the brain looks. Keeping them distinct is the whole game.

## 1. The motion score — `t0` vs `t1`
The tier-0 gate is **mean absolute pixel difference**, normalized to 0–1 (no model, deterministic):

```
motion = Σ |t1[i] − t0[i]|  /  (n_pixels × 255)
```

On a 2×2 luma patch (real frames downsample to ~48×36 — same math):

| case | t0 → t1 | motion | decision |
|---|---|---|---|
| **identical** (still scene) | `[100,100,100,100]` → `[100,100,100,100]` | **0.00** | skip — no cloud call |
| **big change** (someone walked in) | `[0,0,0,0]` → `[255,255,255,255]` | **1.00** | escalate |
| **partial** (something moved in part) | `…` → one region jumps | ~**0.15** | threshold decides |
| **subtle** (slow slump, a screen scroll) | `[100,…]` → `[105,…]` | ~**0.005** | skip ← the recall trap |

Identical frames cancel to 0; any real change accumulates; **the threshold is the dial** between
"worth a cloud call" and "still". The subtle case is why low-pixel-motion events get silently skipped —
see §3.

## 2. The strip — frames *after* (and before) the motion
Every frame is cheaply buffered into a rolling `FrameWindow`. When the gate escalates, the brain doesn't
just judge that one frame — with `PerceptionSpec.temporal="strip"` it judges the **lead-up**:

```
… t-2  t-1   t0    t1 …      every frame → buffer + gate
                   ▲
   gate: t0→t1 motion HIGH  → escalate at t1
        strip = the last n_frames = [t-2, t-1, t0, t1]   (frames BEFORE + at the motion)
        vision.judge(condition, frames=strip) → Verdict
```

The strip is what lets the VLM see the **transition**: a *fall* (the descent, not just "on the floor"),
a *screenshare* (what scrolled past). The **motion score decides whether to look; the strip decides how
much history the VLM sees when it does** — two separate knobs.

## 3. Cadence — *how often* to look, and *on what signal*
The motion gate is right for **physical** events. But a **screenshare** changes content with ~0 pixel
motion (the subtle case in §1) → the motion gate never escalates → fast content is missed. So a goal also
carries a **cadence**, distinct from the strip:

```python
# physical event — motion-gated (default)
state = {"perception": {"temporal": "strip", "n_frames": 4}}

# content event — look every 0.5s regardless of motion
state = {"cadence": {"min_judge_interval": 0.5, "gate_on": "fixed"}}
```

`gate_on="fixed"` makes the gate escalate on a timer even with zero motion (the floor for content events);
`cadence_s` also rate-limits a motion goal (a ceiling). The lesson: **if the gate's signal doesn't match
the event type, the strip never even gets assembled — you have to clear the gate first.**

See [the runtime guide](../runtime.md) for using these on a live `run(source) → incidents` loop.
