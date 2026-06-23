# Percept examples

Two runnable "watch X" agents. Both state a goal in plain language and fire **once on the
rising edge** (false → true) of the condition — the same cognition layer; only the eyes differ.

| file | backend | needs | run it? |
|---|---|---|---|
| [`watch_agent.py`](watch_agent.py) | `fake` (offline, scripted) | nothing — pure stdlib | **yes** |
| [`webcam_agent.py`](webcam_agent.py) | `gemini` (real VLM) | `[gemini]` extra + key + camera | template only |

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

See [`AGENTS.md`](../AGENTS.md) for the full integration cheat-sheet, and
[`docs/quickstart.md`](../docs/quickstart.md) / [`docs/concepts.md`](../docs/concepts.md) for the
deeper tour.
