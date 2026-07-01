# Percept examples

Two runnable "watch X" agents. Both state a goal in plain language and fire **once on the
rising edge** (false → true) of the condition — the same cognition layer; only the eyes differ.

| file | backend | needs | run it? |
|---|---|---|---|
| [`watch_agent.py`](watch_agent.py) | `fake` (offline, scripted) | nothing — pure stdlib | **yes** |
| [`webcam_agent.py`](webcam_agent.py) | `gemini` (real VLM) | `[gemini]` extra + key + camera | template only |
| [`loitering_clip.py`](loitering_clip.py) | `fake` (offline, scripted) | nothing — pure stdlib | **yes** |
| [`web-demo/`](web-demo/) | static browser demo | browser only | **yes** |

---

## 1. `watch_agent.py` — offline, no keys

A scripted frame sequence drives a goal through the three-state gate. No model, no camera,
no network. Run from the repo root with `PYTHONPATH` pointed at the package:

```bash
PYTHONPATH=packages/percept-vision python3 examples/watch_agent.py
```

Or, after installing the SDK:

```bash
pip install percept-vision
python3 examples/watch_agent.py
```

Expected output:

```
frame          t    -> events
-------------------------------
empty-hands    1.0  -> (silent)
holding-cup    2.0  -> (silent)
drinking       3.0  -> FIRE  [caffeine]  Heads up — stepping back from caffeine?
drinking       4.0  -> (silent)
empty-hands    5.0  -> (silent)
drinking       6.0  -> FIRE  [caffeine]  Heads up — stepping back from caffeine?
```

What to notice:

- It fires **once** at `t=3.0` (the false → true rising edge), not on every "drinking" frame.
- `t=4.0` is still true → the **refractory** holds it silent (no double-fire).
- `t=5.0` is confidently false → that **re-arms** the edge, so `t=6.0` fires again.

---

## 2. `webcam_agent.py` — real Gemini backend (template)

The identical agent with one substitution: `vision="fake"` → `vision="gemini"`, and the
frames become real JPEG bytes from your webcam. **Do not run it without a camera + key** — it
is a correct, copy-pasteable template.

```bash
pip install "percept-vision[gemini]" opencv-python
export GEMINI_API_KEY=...        # or GOOGLE_API_KEY
python3 examples/webcam_agent.py
```

Key points:

- **You feed frames.** Percept has no built-in capture; the example's `cv2.VideoCapture(0)`
  loop is yours to own. The cognition layer below `perceive_judged(...)` is unchanged.
- **Frame format** for `gemini`/`claude` is image bytes or a `(media_type, bytes)` pair. The
  example encodes each BGR frame with `cv2.imencode(".jpg", bgr)` and passes
  `("image/jpeg", buf.tobytes())`.
- **Sample slowly** (~1 fps). Each judged frame is one ~1s cloud VLM call per active goal; the
  rising-edge gate means slower sampling won't cause missed or duplicate fires.

---

## 3. `loitering_clip.py` — judge a CLIP, not one frame (offline)

Some conditions can't be read from a single still: *loitering* looks identical to *standing there*
in one frame — what makes it loitering is **dwelling over time**. The concern declares it needs a
clip via `window_s` (seconds); the brain assembles a time-bounded temporal strip from its rolling
window and feeds it to the judge. The public `judge_clip(condition, frames)` seam judges a clip you
already hold, directly.

```bash
PYTHONPATH=packages/percept-vision python3 examples/loitering_clip.py
```

What to notice:

- The concern sets `window_s=4.0` — `concern.wants_clip` reflects it, and `perception_for_concern`
  maps it to the `goal.state["perception"]` the watch loop consumes (a single-frame concern is
  unchanged — `window_s` defaults to `None`).
- A single `"loiter"` frame reads inconclusive; once a multi-frame **strip** reaches `judge`, the
  scripted backend returns the "dwelling" clip verdict — proof the clip flowed into perception.

---

## 4. `benchmark/` — measure per-concern precision/recall + a CI gate (offline)

Evaluate Percept on *your* concerns: a manifest of labelled clips → per-concern precision, recall,
false-fire, and refuse rates. Runs fully offline on the `fake` backend (no keys).

```bash
# the report (also: make eval-benchmark)
PYTHONPATH=packages/percept-vision python3 -m percept.bench examples/benchmark/manifest.json

# the copy-pasteable CI regression gate (exits non-zero on a regression)
PYTHONPATH=packages/percept-vision python3 examples/benchmark/ci_gate.py
```

- [`benchmark/manifest.json`](benchmark/manifest.json) — the input format (concerns + labelled clips).
- [`benchmark/ci_gate.py`](benchmark/ci_gate.py) — wire a per-concern P/R threshold into CI.

Full guide + the manifest spec: [`docs/how-to/eval.md`](../docs/how-to/eval.md).

---

## 5. `web-demo/` — hosted-looking OSS demo shell

Open [`web-demo/index.html`](web-demo/index.html) in a browser. It is a static replay of the Percept
loop: edge gate, perception read, cognition gate, incident snippet, and scorecard panel.

---

See [`AGENTS.md`](../AGENTS.md) for the full integration cheat-sheet, and
[`docs/tutorials/quickstart.md`](../docs/tutorials/quickstart.md) /
[`docs/explanation/concepts.md`](../docs/explanation/concepts.md) for the deeper tour.
