# Hackathon cookbook — a personal vision assistant (smart glasses)

Build a "where are my things" assistant on Percept: it watches your first-person view and can answer
**"where are my keys?"** and remind you **"you're leaving without your handbag."**

Runnable starting point: **[`examples/vision_assistant.py`](../examples/vision_assistant.py)** — the code
below is lifted from it and is verified on real first-person footage.

```bash
pip install "percept-vision[runtime,openrouter]"
export OPENROUTER_API_KEY=sk-or-...
python examples/vision_assistant.py --egomotion            # glasses / body-cam
python examples/vision_assistant.py --source clip.mp4      # test on a recording first
```

> **Glasses = a moving camera → always use `--egomotion`** (sets `PERCEPT_EGO_COMPENSATE=true`). Without it,
> head motion makes the gate escalate on ~98% of frames; with it, ~30%. Measured, not guessed.

---

## How Percept fits (and what you build)

Percept gives you: **detections** — "is `<my keys>` visible right now?" — once per sighting, cheap (it only
calls the model when the scene changes), with a **plain-English description** of what it saw. On glasses it
subtracts head motion first.

You build the two thin layers on top — both are ~10 lines of plain Python:
1. **Memory** — remember *when* and *where* each item was last seen.
2. **Leave logic** — on a "leaving" cue, check which items weren't seen recently.

---

## Pattern 1 — "Where are my keys?" (location memory)

**The key insight:** you don't need coordinates. **The judge's `evidence` string *is* the location** — in
plain language. Verified on real first-person (Ego4D-style) footage:

```
👁  a pot or pan: A black frying pan is clearly visible on the stovetop, left portion of the image.
👁  a bowl:       A dark-colored bowl is visible on the countertop near the center of the image.
```

So `"where's my <item>?"` = the last `evidence` you recorded for it.

```python
import asyncio, percept, time
from percept import Goal

ITEMS = ["keys", "handbag", "wallet", "phone"]
last_seen = {}   # item -> {"t": monotonic_seconds, "where": evidence string}

async def watch():
    goals = [Goal(id=it, condition=f"my {it} is clearly visible", say=it) for it in ITEMS]
    async for inc in percept.run(source="webcam:0", goals=goals, vision="openrouter", fps=1.0):
        last_seen[inc.concern_id] = {"t": inc.ts, "where": inc.verdict.evidence}

def where_is(item):
    r = last_seen.get(item)
    if not r:
        return f"Haven't spotted your {item} yet."
    return f"Your {item} — last seen ~{int(time.monotonic()-r['t'])}s ago: {r['where']}"
```

**Tuning for location memory:** set a *short* refractory so re-sightings refresh the location —
`export PERCEPT_GATE_REFRACTORY_S=8` (the default 15 is for one-alert-per-event; here you want updates). Each
time an item comes back into view it re-fires and updates `last_seen`.

---

## Pattern 2 — "You're leaving without your handbag" (leave-without-X)

Two watches + your logic: a **leaving cue** and each **item**. When "leaving" fires, flag any item not seen
in the last ~45s (i.e. not with you as you head out).

```python
LEAVING = "the person is at the front door about to go out (jacket or shoes on, reaching for the door)"
FRESH_S = 45.0

async def assistant():
    goals  = [Goal(id=it, condition=f"my {it} is clearly visible", say=it) for it in ITEMS]
    goals += [Goal(id="__leaving__", condition=LEAVING, say="leaving")]
    async for inc in percept.run(source="webcam:0", goals=goals, vision="openrouter", fps=1.0):
        if inc.concern_id == "__leaving__":
            forgot = [it for it in ITEMS if inc.ts - last_seen.get(it, {}).get("t", 0) > FRESH_S]
            if forgot:
                notify(f"Heading out — don't forget: {', '.join(forgot)}!")
        else:
            last_seen[inc.concern_id] = {"t": inc.ts, "where": inc.verdict.evidence}
```

That's the whole reminder. `notify(...)` is yours — a phone push, a TTS line, a Slack message.

---

## The `Incident` you get on each fire

| field | use |
|---|---|
| `inc.concern_id` | which item (the `Goal.id`) |
| `inc.verdict.evidence` | **the location description** — "keys on the counter near the sink" |
| `inc.verdict.confidence` | 0–1; gate on `>= 0.8` if you want fewer, surer notes |
| `inc.ts` | timestamp (monotonic seconds) — for "how long ago" + the freshness check |

---

## Honest limits (know these before you demo)

- **Presence, not instance-tracking.** Percept detects *"a handbag is visible"*, not *"YOUR specific
  handbag vs a stranger's."* For a personal glasses assistant (one user, their things) that's usually fine —
  but two identical mugs on a table are indistinguishable.
- **The "leaving" cue is a judged condition** — it's only as good as the goal wording. Tune it to your setup
  (`"...at the front door with keys in hand"`, `"...putting on a coat by the door"`). Test on a recording of
  yourself leaving before trusting it live.
- **Location = last *seen*, not live GPS.** If you set the keys down and the glasses looked away, the answer
  is the last time they were in view. Good enough for "I think they're on the kitchen counter."
- **Latency ~1–2s** per judged frame (Qwen at 1 fps). Reactions trail the moment slightly — fine for this.

## Tuning cheatsheet

| want | do |
|---|---|
| glasses / moving camera | `--egomotion` (always, for smart glasses) |
| location keeps refreshing | `PERCEPT_GATE_REFRACTORY_S=8` (short) |
| fewer, surer detections | filter on `inc.verdict.confidence >= 0.8` |
| catch fast moments | `--fps 2` |
| specific, distinct items | one goal per item, worded specifically ("my brown leather handbag") — few and distinct keeps accuracy up |

More: **[docs/integration.md](integration.md)** (full `run()` API) · **[authoring goals](how-to/authoring-goals-and-watch-plans.md)**.
