# Percept examples

## `hackathon_watch.py` — live watch demo

Point Percept at a camera (or a video / RTSP stream), give it goals in plain English, and it prints a
notification the moment each becomes true — **once per episode** (a per-episode refractory dedupes repeats).

```bash
# 1. a judge key (cheap, fast Qwen via OpenRouter)
export OPENROUTER_API_KEY=sk-or-...

# 2. webcam, your own goals
python examples/hackathon_watch.py \
    --goal "someone raises their hand" \
    --goal "someone is holding a phone"

# or a video file
python examples/hackathon_watch.py --source clip.mp4 \
    --goal "someone is drinking from a cup"
```

**Flags:** `--source` (`webcam:0` default · a file path · `rtsp://…`) · `--goal` (repeatable) ·
`--vision` (`openrouter`/`gemini`/`claude`) · `--model` · `--fps` (default 1.0) · `--refractory` (seconds,
default 15).

**What it bakes in (the H0-winning config):**
- `block_max` localized salience — catches low-motion actions (holding, sitting) a global motion gate misses.
- a per-episode **time-refractory** — one notification per event, not a stream of duplicates.

### Dependencies
- **Judge:** an API key for your provider (`OPENROUTER_API_KEY`, or Gemini/Claude with `--vision`).
- **Webcam / video:** the capture extra — `pip install 'percept-vision[cv]'` (OpenCV). A `.wav`/file path
  needs nothing extra beyond that.

### At the hackathon
- Test your webcam index first: if `webcam:0` is dark, try `webcam:1`.
- Keep goals **specific and falsifiable** ("someone is holding a red mug", not "something happens") and to a
  **few, mutually-distinct** conditions — that's what keeps precision high (see
  `docs/how-to/authoring-goals-and-watch-plans.md`).
- 1 fps is plenty for most watches and keeps cost/latency low; raise `--fps` only for fast events.
