# Percept examples

## `percept-watch` — live watch demo (no code)

Point Percept at a camera (or a video / RTSP stream), give it goals in plain English, and it prints a
notification the moment each becomes true — **once per episode** (a per-episode refractory dedupes repeats).

```bash
# 1. install the capture + judge extras, and a judge key (cheap, fast Qwen via OpenRouter)
pip install "percept-vision[runtime,openrouter]"
export OPENROUTER_API_KEY=sk-or-...

# 2. webcam, your own goals
percept-watch \
    --goal "someone raises their hand" \
    --goal "someone is holding a phone"

# or a video file
percept-watch --source clip.mp4 \
    --goal "someone is drinking from a cup"
```

**Flags:** `--source` (`webcam:0` default · a file path · `rtsp://…`) · `--goal` (repeatable) ·
`--vision` (`openrouter`/`gemini`/`claude`) · `--model` · `--fps` (default 1.0) · `--refractory` (seconds,
default 15) · `--egomotion` (moving camera).

**What it bakes in (the H0-winning config):**
- `block_max` localized salience — catches low-motion actions (holding, sitting) a global motion gate misses.
- a per-episode **time-refractory** — one notification per event, not a stream of duplicates.

## `vision_assistant.py` — a personal vision assistant

A "where are my things" assistant for a first-person view (smart glasses / body-cam): it remembers where
items were last seen and reminds you when you head out without one. The memory and leave-logic are plain
Python you own; Percept supplies the detections.

```bash
python examples/vision_assistant.py --egomotion            # glasses/body-cam
python examples/vision_assistant.py --source clip.mp4      # test on a recording
```

## `byo_admission_detector.py` — bring your own model in front of the judge

Offline, deterministic, **no keys and no model**. Shows the two supported ways to put your own cheap
model (OCR, CLIP, a classifier) in front of Percept's expensive judge — `register_detector` for timed
occurrences, `percept.run(..., semantic_adapter=...)` for admission — and states precisely what neither
does: they decide *when to look closely*; they do not deliver extracted facts (`invoice_number = …`) to
the judge. See [`docs/admission/detectors.md`](../docs/admission/detectors.md#admission-is-not-extraction--what-register_detector-does-and-does-not-do).

```bash
python examples/byo_admission_detector.py
```

### Dependencies
- **Judge:** an API key for your provider (`OPENROUTER_API_KEY`, or Gemini/Claude with `--vision`).
- **Webcam / video / RTSP:** the capture extra — `pip install 'percept-vision[runtime]'` (OpenCV + numpy).

### Tips
- Test your webcam index first: if `webcam:0` is dark, try `webcam:1`.
- Keep goals **specific and falsifiable** ("someone is holding a red mug", not "something happens") and to a
  **few, mutually-distinct** conditions — that's what keeps precision high (see
  `docs/watches/writing-goals.md`).
- 1 fps is plenty for most watches and keeps cost/latency low; raise `--fps` only for fast events.

---
Full integration guide: [`docs/watches/creating-a-watch.md`](../docs/watches/creating-a-watch.md).
