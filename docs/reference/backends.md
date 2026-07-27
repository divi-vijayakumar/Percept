# Backends

Percept reasons over a `Verdict` stream; it never imports a vendor SDK directly. Each **seam**
(vision, conductor, specialist, stt, tts) resolves a backend **by name** through the registry. Swap
a backend by changing one string — the cognition layer (gate, entity graph, escalation) is unchanged.

```
vision · specialist · conductor · stt · tts   ← the five cognition seams (+ extractor)
       │ resolved by name via percept.registry
       ▼
   "fake" / "scripted"  ← built in, no deps, offline
   "gemini"             ← needs the [gemini] extra + a key
   "claude"             ← needs the [claude] extra + a key
   "openrouter"         ← vision; needs the [openrouter] extra + a key
   "deepgram"           ← stt/tts, needs the [deepgram] extra + a key
```

---

## Install + extras

The **core is pure stdlib** — `pip install percept-vision` runs offline with the `fake` backends,
no keys, no heavy deps. Real backends are optional extras:

```bash
pip install percept-vision                 # core only — fakes, offline
pip install "percept-vision[gemini]"       # Google Gemini vision/conductor/specialist
pip install "percept-vision[claude]"       # Anthropic Claude vision/conductor/specialist
pip install "percept-vision[openrouter]"   # OpenRouter vision (OpenAI-compatible; many models)
pip install "percept-vision[deepgram]"     # Deepgram STT + TTS (voice)
pip install "percept-vision[audio]"        # numpy — audio/acoustic processing
pip install "percept-vision[runtime]"      # camera/video/RTSP runtime worker
pip install "percept-vision[bench]"        # benchmark + scorecard tooling
pip install "percept-vision[edge]"         # Python tier-0 gate + detector registry
pip install "percept-vision[all]"          # everything above
```

| extra        | pulls in            | enables                                     |
|--------------|---------------------|---------------------------------------------|
| `[gemini]`   | `google-genai`      | `vision="gemini"`, `conductor`, `specialist`|
| `[claude]`   | `anthropic`         | `vision="claude"`, `conductor`, `specialist`|
| `[openrouter]` | `openai>=1.40`    | `vision="openrouter"` (OpenAI-compatible)   |
| `[deepgram]` | `deepgram-sdk`      | `stt="deepgram"`, `tts="deepgram"`          |
| `[ocr]`      | `pytesseract`, `Pillow` | the real `extractor="ocr"` (also needs the `tesseract` binary on PATH) |
| `[audio]`    | `numpy`             | audio/acoustic signal processing            |
| `[harness]`  | `numpy`             | the bundled `percept.harness` tier-0 gate + detector registry |
| `[runtime]`  | `opencv-python-headless`, `numpy` | camera/video/RTSP runtime worker |
| `[bench]`    | `numpy`, `opencv-python-headless` | `percept-bench` scorecards |
| `[edge]`     | `numpy`             | public `percept.edge` / `percept.harness` tier-0 gate + detector registry |
| `[pose]`     | `mmpose`, `mmdet`, `mmcv`, `mmengine`, `opencv-python-headless` | opt-in pose detector dependencies |
| `[otel]`     | `opentelemetry-api`, `opentelemetry-sdk` | OpenTelemetry export |

There is **no `[screen]` extra** and **no built-in screen capture** in 1.0: `screen://` is not a capture
URI and opening one raises. Grab the display yourself and push the frames in as an iterable `source=` —
see [screenshare](../how-to/screenshare.md).

---

## API keys

Each real backend reads its key from the environment (no key is ever passed in code):

| backend      | env var(s)                                                          |
|--------------|--------------------------------------------------------------------|
| `gemini`     | `GEMINI_API_KEY` **or** `GOOGLE_API_KEY` **or** `GOOGLE_AISTUDIO_API_KEY` (any one) |
| `claude`     | `ANTHROPIC_API_KEY`                                                |
| `openrouter` | `OPENROUTER_API_KEY` **or** `OPENROUTER_KEY`                       |
| `deepgram`   | `DEEPGRAM_API_KEY`                                                 |

```bash
export GEMINI_API_KEY=...      # or GOOGLE_API_KEY / GOOGLE_AISTUDIO_API_KEY
export ANTHROPIC_API_KEY=...
export OPENROUTER_API_KEY=...  # or OPENROUTER_KEY
```

The `fake` backends need **no key** — that's what makes the offline path work.

---

## Selecting backends

### In code — by name

Pass a backend **name** to `Percept.create(...)`; the registry resolves it:

```python
import percept

agent = percept.Percept.create(
    vision="gemini",          # the eyes — the high-frequency perception calls
    conductor="claude",       # authors goals from natural language
    specialist="claude",      # streams spoken replies
    stt="deepgram",           # speech-in
    tts="deepgram",           # speech-out
    discover_plugins=False,
)
```

A seam you omit takes the config/env selection (`PERCEPT_*_BACKEND`), else the mode: `mode="offline"`
selects scripted backends; a bare LIVE run with vision unset raises `ModeError` (no silent fake). You
can also pass an **adapter instance** instead of a name — it is used **verbatim** (Percept never wraps
or substitutes your object; it lands on `core.vision` as-is) — handy for tests or custom wiring:

```python
from percept.backends.fake import FakeVision
from percept.adapters import Verdict

agent = percept.Percept.create(
    mode="offline",
    vision=FakeVision(script={"frame-1": Verdict(satisfied=True, confidence=0.9)}),
    discover_plugins=False,
)
```

### By environment — no code change

Every seam has a `PERCEPT_*_BACKEND` env var. A seam **not explicitly passed** to `create()` reads
its var, so you can flip backends without touching code:

| seam       | env var                       | default |
|------------|-------------------------------|---------|
| vision     | `PERCEPT_VISION_BACKEND`      | `fake`  |
| conductor  | `PERCEPT_CONDUCTOR_BACKEND`   | `fake`  |
| specialist | `PERCEPT_SPECIALIST_BACKEND`  | `fake`  |
| stt        | `PERCEPT_STT_BACKEND`         | `fake`  |
| tts        | `PERCEPT_TTS_BACKEND`         | `fake`  |

```bash
export PERCEPT_VISION_BACKEND=gemini
export PERCEPT_CONDUCTOR_BACKEND=claude
python my_agent.py            # create() with no vision= now uses gemini
```

The model id within a backend is also overridable:

| seam       | model env var             | default            |
|------------|---------------------------|--------------------|
| vision     | `PERCEPT_VISION_MODEL`     | `claude-opus-4-8`  |
| conductor  | `PERCEPT_CONDUCTOR_MODEL`  | `claude-haiku-4-5` |
| specialist | `PERCEPT_SPECIALIST_MODEL` | `claude-haiku-4-5` |

For Gemini, set the model to a `gemini-*` id (e.g. `PERCEPT_VISION_MODEL=gemini-3.5-flash`); the
Gemini backend falls back to its default `gemini-3.5-flash` when the configured model id isn't a
Gemini one. The Gemini-Live backend (`vision="gemini-live"`) defaults to
`gemini-3.1-flash-live-preview`. OpenRouter defaults to `qwen/qwen3-vl-32b-instruct` and respects any
explicit OpenRouter-routable id you set.

### By identity — `resolve_vision(name, model=...)`

The provider **name** is identity; the **model id** is configuration. `resolve_vision` keeps the two
separate so a volatile model id never gets mistaken for an adapter name:

```python
from percept.provider import resolve_vision

vision = resolve_vision("openrouter", model="qwen/qwen3-vl-32b-instruct")
agent = percept.Percept.create(vision=vision)
```

A model id is **never** an adapter name: `vision="qwen/qwen3-vl-32b-instruct"` does not fall through to
some "qwen" backend — you name the provider (`openrouter`) and pass the model id as `model=`.

---

## Available backends per seam

| seam       | built-in names                                                         |
|------------|------------------------------------------------------------------------|
| vision     | `fake`, `scripted`, `fixture`, `claude`, `gemini`, `gemini-live`, `openrouter` |
| specialist | `fake`, `scripted`, `claude`, `gemini`                                  |
| conductor  | `fake`, `scripted`, `scripted-conductor`, `claude`, `gemini`            |
| stt        | `fake`, `scripted`, `deepgram`                                          |
| tts        | `fake`, `scripted`, `deepgram`                                          |
| extractor  | `fake`, `scripted`, `ocr-scripted`, `ocr`, `fields-scripted`, `fields`, … |

`fake` and `scripted` are the same deterministic offline doubles under two names; `fixture` replays a
recorded verdict stream. Query the live list at runtime rather than trusting this table:

```python
import percept
percept.registry.available("vision")   # ['claude', 'fake', 'fixture', 'gemini', 'gemini-live', 'openrouter', 'scripted']
percept.registry.available("stt")      # ['deepgram', 'fake', 'scripted']
agent.health()                         # {'status': 'ok', 'backends': {...}, ...}
```

If you select a real backend whose extra isn't installed, resolution raises a **clear, actionable
error** telling you which extra to install — it never fails silently.

---

## Custom / third-party backends

A backend is any object implementing the relevant component `Protocol` (`Vision`, `Conductor`,
`Specialist`, `STT`, `TTS`). These Protocols are canonically defined in `percept.contracts` and
re-exported from `percept.adapters` — both import paths work (`from percept.contracts import Vision`
and `from percept.adapters import Vision` are the same object). The `Vision` contract is just:

```python
async def judge(self, condition: str, frame, *, roi=None, exemplar=None, corrections=None,
                frames=None, audio=None, context=None) -> Verdict: ...
async def detect(self, frame, goals) -> list[Observation]: ...
async def detect_events(self, frames, goals) -> list: ...
```

Register your factory under a seam + name:

```python
from percept import registry
registry.register("vision", "myvlm", lambda settings: MyVision(settings))

agent = percept.Percept.create(vision="myvlm")
```

Third-party packages can auto-register by advertising a `percept.backends` entry point; Percept
discovers them when `discover_plugins=True` (the default). A backend that lacks a modality (e.g. a
vision model with no audio) degrades gracefully — the returned `Verdict` shape is unchanged, so the
gate downstream is identical.

The **Vision** seam is documented in full — every escalation kwarg, the `Verdict`/`Observation`
invariants, fail-closed injection, and a runnable conformance suite you can execute against your own
adapter — in [The Vision adapter interface](vision-adapter.md):

```bash
python -m percept.conformance myvlm.vision:MyVision
```

For the step-by-step **bring-your-own-backend recipe**, see
[Custom judge](../how-to/custom-judge.md); for the full extension-point contract (`judge`/`detect`/
`detect_events`, invariants, conformance), see [The Vision adapter interface](vision-adapter.md).

---

## Lane discipline (why fakes matter)

The `fake` backends are **deterministic and scripted** — they prove your wiring and the cognition
logic without asserting any model intelligence. This split is deliberate:

- **L1 (offline):** `fake` backends, no keys, fast, reproducible. The default for tests and the
  quickstart.
- **L2/L3 (live):** real backends, keys present. The only place model behavior is exercised.

So a green offline test means *your integration is correct*; testing whether Percept actually
*perceives* the world is a live-backend job. See [quickstart.md](../tutorials/quickstart.md) for both paths.
