# Percept runtime — integration guide

Watch a live video stream for a **plain-English condition** and get a notification the moment it becomes
true — once per event. Point it at a webcam, a video file, or an RTSP camera; give it goals in words; handle
the notifications in your app.

```
your source ──▶ fovea (cheap motion/salience gate) ──▶ VLM judge ──▶ rising-edge gate ──▶ 🔔 notification
                 spends a model call only when             "is <goal> true?"     once per episode
                 something worth judging happens
```

> **Before you design around it**, read
> [Scope and persistence (v1.4)](reference/v1-4-scope-and-persistence.md). Three things clients have
> read too generously: v1.4 covers **screen documents with stable content keys** and **conversational
> claim slots** only — biometric re-identification, growth prediction and robotics control are
> explicitly excluded; **episodes are session-only** and do not survive a restart, which blocks any
> 24/7 claim; and delivery is **provisional incident + superseding revision**, not a wide settle
> window.

---

## 1. Install

> ⚠️ **Not on PyPI yet.** The published `percept-vision==0.6.0` is an older research-preview and does **not**
> include this runtime (`watch()`, the refractory, block_max defaults). Install from the repo:

```bash
# clone, then editable-install the package with the runtime + judge extras
git clone https://github.com/divi-vijayakumar/Percept.git
cd Percept
pip install -e 'packages/percept-vision[runtime,openrouter]'
#                                │        └ the judge (Qwen via OpenRouter — cheap, fast, hosted)
#                                └ capture (OpenCV) for webcam / video / RTSP
```

<sub>Once the branch is pushed / a release is cut, a direct install will work:
`pip install 'git+https://github.com/divi-vijayakumar/Percept.git@BRANCH#subdirectory=packages/percept-vision'`
(or a fresh `pip install percept-vision` after the next PyPI release).</sub>

A judge API key (pick one provider):

```bash
export OPENROUTER_API_KEY=sk-or-...     # Qwen (recommended: cheap + fast).  extras: [openrouter]
# or:  export GEMINI_API_KEY=...        # then vision="gemini"               extras: [gemini]
# or:  export ANTHROPIC_API_KEY=...     # then vision="claude"               extras: [claude]
```

## 2. 30-second smoke test

```bash
percept-watch --source webcam:0 --goal "someone raises their hand"
```
If you see a `🔔` when you raise your hand, you're good. Now wire it into your app.

---

## 3. The one function you need: `run(...)`

```python
import asyncio, percept
from percept import Goal

async def main():
    goals = [
        Goal(id="hand",  condition="someone raises their hand", say="Hand raised!"),
        Goal(id="phone", condition="someone is holding a phone", say="Phone spotted"),
    ]
    async for incident in percept.run(
        source="webcam:0",     # or "path/to/clip.mp4" or "rtsp://user:pass@host/stream"
        goals=goals,
        vision="openrouter",   # "gemini" | "claude"
        fps=1.0,               # frames pulled per second (1–2 is plenty; higher = faster + costlier)
    ):
        # incident fires ONCE when a goal first becomes true (rising edge)
        print(incident.concern_id, "→", incident.message, incident.verdict.confidence)
        # … do your thing: POST a webhook, push a notification, update UI …

asyncio.run(main())
```

`run(...)` is an **async generator** — every `incident` it yields is a real notification. That's the whole
integration surface.

### A goal

```python
Goal(id="unique_id", condition="a plain-English thing to watch for", say="the message you want back")
```
- `id` — your handle for it (shows up as `incident.concern_id`).
- `condition` — what the judge is asked, verbatim-ish. **Be specific and falsifiable** (see §7).
- `say` — the human message; comes back as `incident.message`.

### The notification you get (`Incident`)

| field | what it is |
|---|---|
| `incident.concern_id` | the `Goal.id` that fired |
| `incident.message` | your `Goal.say` text |
| `incident.verdict.confidence` | 0–1 model confidence |
| `incident.verdict.evidence` | one line of *why* (great for UI / debugging) |
| `incident.ts` | timestamp (monotonic seconds) |
| `incident.id` | stable event id (use it to dedupe on your side if you fan out) |
| `incident.entity_id` | which SUBJECT fired (the gate key's entity), or `None` if the fire was not entity-scoped — use it to tell two subjects on the same goal apart instead of reading their interleaved fires as out-of-order |

---

## 4. Two ways to consume it

**A — iterate (simplest):**
```python
async for incident in percept.run(source=src, goals=goals, vision="openrouter"):
    await handle(incident)
```

**B — callback (fire-and-forget into your handler):**
```python
def on_fire(incident):
    requests.post(SLACK_WEBHOOK, json={"text": incident.message})

async for _ in percept.run(source=src, goals=goals, vision="openrouter", on_incident=on_fire):
    pass   # still drive the loop; on_incident is called for each fire
```

**Run it alongside a sync app (Flask/FastAPI/thread):**
```python
import threading, asyncio
def watch_forever():
    asyncio.run(main())          # main() drives percept.run(...)
threading.Thread(target=watch_forever, daemon=True).start()
```

---

## 5. Sources

| `source=` | opens |
|---|---|
| `"webcam:0"` / `"webcam"` | default camera (try `webcam:1` if `0` is dark) |
| `"clip.mp4"` / any file path | a video file (great for testing without a camera) |
| `"rtsp://user:pass@host:554/stream"` | an IP camera (add `source_rtsp_tcp=True`, the default) |

Flaky network camera? `run(..., source_reconnects=5, source_reconnect_backoff_s=1.0)`.

---

## 6. Config that matters (sensible defaults + the few knobs worth touching)

Set these as env vars **before** `import percept`, or leave them — the demo bakes in the good ones:

```bash
export PERCEPT_SALIENCE_STAT=block_max     # ✅ recommended for a FIXED camera + small/localized actions
                                           #    (holding, sitting). "mean" misses low-motion events.
export PERCEPT_GATE_REFRACTORY_S=15        # ✅ suppress repeat notifications of the SAME goal for N seconds
                                           #    (0 = notify on every confirmation = noisy). 8–30 is typical.
export PERCEPT_VISION_MODEL=qwen/qwen3-vl-32b-instruct   # the judge model id for your provider
```

At the call site: `fps` (1–2 default; raise for fast events), `vision` (provider), and pass a fresh
`Goal.state` if you want per-goal overrides.

---

## 6b. Moving cameras — glasses, body-cam, phone (egomotion) ⚠️

If your camera **moves** (smart glasses, body-cam, handheld), a plain motion gate escalates on *every*
frame — the whole scene shifts every time your head does. Measured on real first-person footage: `block_max`
alone calls the VLM on **98%** of frames. The fix is **ego-motion compensation** — subtract the global
camera shift before measuring motion:

```bash
export PERCEPT_EGO_COMPENSATE=true     # ← subtract head/camera motion
export PERCEPT_SALIENCE_STAT=block_max
```
That drops escalation to **~30%** and object detection works (verified: containers/hands/people read
correctly on head-mounted frames). The CLI has a flag for it:

```bash
percept-watch --egomotion \
    --goal "my keys are visible" --goal "a person is nearby"
```

**Honest limits on moving cameras:**
- Ego-comp makes the *gate* economical, but detection is still frame-by-frame VLM — on very fast/blurry head
  motion, frames can be too smeared to judge. Raising `--fps` helps catch a clean frame.
- **Detecting** an object (keys visible) is solid. **Tracking a specific instance** across occlusion/glances
  (that's *my* keys, still the same ones) is the entity layer — weaker, and not battle-tested on egomotion.
- All of this was validated on the ProactiveQA EGO set, not tuned per your rig — treat the thresholds as a
  starting point and adjust `--refractory` / `--fps` on your own footage.

## 7. Write good goals (this is what makes it accurate)

- **Specific & falsifiable:** `"someone is holding a red mug"` ✓ — not `"something is happening"` ✗.
- **Name the observable, not the intent:** `"a pot is boiling over"` ✓ — not `"cooking is going wrong"` ✗.
- **Few and mutually-distinct:** 2–4 goals that don't overlap. Ten similar goals on one scene makes the
  judge confuse them and tanks precision.
- **Match the camera:** fixed mount + small actions → keep `block_max` on.

Full guidance: `docs/how-to/authoring-goals-and-watch-plans.md`.

---

## 8. Recipes

**Notify Slack / a webhook:**
```python
async for inc in percept.run(source=src, goals=goals, vision="openrouter"):
    requests.post(WEBHOOK, json={"text": f"{inc.message} ({inc.verdict.confidence:.0%})"})
```

**Only act on high-confidence fires:**
```python
async for inc in percept.run(...):
    if inc.verdict.confidence >= 0.8:
        alert(inc.message)
```

**Test on a recorded clip first (no camera needed):**
```python
async for inc in percept.run(source="my_test.mp4", goals=goals, vision="openrouter", fps=1.0):
    print(inc.ts, inc.concern_id, inc.verdict.evidence)
```

---

## 9. Cost & latency (set expectations)

- **Latency:** a fire trails the action by ~1 frame interval + one judge call (~1–2 s with Qwen at 1 fps).
- **Cost:** the fovea spends a model call **only when salience clears the gate** — not every frame — so a
  quiet scene is nearly free. Cost scales with *activity*, not runtime. Qwen keeps it cheap.
- **Tuning:** higher `fps` = faster detection + more cost; a longer refractory = fewer duplicate alerts.

## 10. Troubleshooting

| symptom | fix |
|---|---|
| `0 notifications`, judge shows the wrong model | your provider/model mismatch — set `PERCEPT_VISION_MODEL` to a valid id for `vision=` (OpenRouter needs `qwen/…`, not a Gemini id) |
| `Insufficient credits` / `402` | top up OpenRouter, or switch `--vision gemini/claude` |
| webcam black / no frames | wrong index — try `webcam:1`; on macOS grant camera permission to your terminal |
| a stream of duplicate alerts | raise `PERCEPT_GATE_REFRACTORY_S` |
| misses low-motion actions | set `PERCEPT_SALIENCE_STAT=block_max` |
| `needs the 'openai' SDK` | `pip install 'percept-vision[openrouter]'` |
| loud `SCRIPTED BACKENDS` banner | harmless (it's about the unused *voice* seam); set `PERCEPT_NO_BANNER=1` |

---

**Reference demo:** [`examples/vision_assistant.py`](../examples/vision_assistant.py) — a complete, runnable
integration you can copy from.
