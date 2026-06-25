# Backends

Percept reasons over a `Verdict` stream; it never imports a vendor SDK directly. Each **seam**
(vision, conductor, specialist, stt, tts) resolves a backend **by name** through the registry. Swap
a backend by changing one string — the cognition layer (gate, entity graph, escalation) is unchanged.

```
vision · specialist · conductor · stt · tts   ← the five seams
       │ resolved by name via percept.registry
       ▼
   "fake"        ← built in, no deps, offline
   "gemini"      ← needs the [gemini] extra + a key
   "claude"      ← needs the [claude] extra + a key
   "deepgram"    ← stt/tts, needs the [deepgram] extra + a key
```

---

## Install + extras

The **core is pure stdlib** — `pip install percept-vision` runs offline with the `fake` backends,
no keys, no heavy deps. Real backends are optional extras:

```bash
pip install percept-vision                 # core only — fakes, offline
pip install "percept-vision[gemini]"       # Google Gemini vision/conductor/specialist
pip install "percept-vision[claude]"       # Anthropic Claude vision/conductor/specialist
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
| `[deepgram]` | `deepgram-sdk`      | `stt="deepgram"`, `tts="deepgram"`          |
| `[audio]`    | `numpy`             | audio/acoustic signal processing            |
| `[runtime]`  | `opencv-python-headless`, `percept-harness` | camera/video/RTSP runtime worker |
| `[screen]`   | `mss`               | screen-capture dependency for custom iterable sources |
| `[bench]`    | `numpy`, `opencv-python-headless`, `percept-harness` | `percept-bench` scorecards |
| `[edge]`     | `percept-harness`   | public `percept.edge` tier-0 gate + detector registry |
| `[pose]`     | `percept-harness[pose]` | opt-in pose detector dependencies |
| `[otel]`     | `opentelemetry-api`, `opentelemetry-sdk` | OpenTelemetry export |

---

## API keys

Each real backend reads its key from the environment (no key is ever passed in code):

| backend    | env var(s)                              |
|------------|-----------------------------------------|
| `gemini`   | `GEMINI_API_KEY` **or** `GOOGLE_API_KEY` |
| `claude`   | `ANTHROPIC_API_KEY`                     |
| `deepgram` | `DEEPGRAM_API_KEY`                      |

```bash
export GEMINI_API_KEY=...      # or GOOGLE_API_KEY
export ANTHROPIC_API_KEY=...
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

Any seam you omit falls back to the config/env selection (default: `fake`). You can also pass an
**adapter instance** instead of a name — handy for tests or custom wiring:

```python
from percept.backends.fake import FakeVision
from percept.adapters import Verdict

agent = percept.Percept.create(
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

For Gemini, set the model to a `gemini-*` id (e.g. `PERCEPT_VISION_MODEL=gemini-2.5-flash`); the
Gemini backend falls back to `gemini-2.5-flash` when the configured model id isn't a Gemini one.

---

## Available backends per seam

| seam       | built-in names              |
|------------|-----------------------------|
| vision     | `fake`, `claude`, `gemini`  |
| specialist | `fake`, `claude`, `gemini`  |
| conductor  | `fake`, `claude`, `gemini`  |
| stt        | `fake`, `deepgram`          |
| tts        | `fake`, `deepgram`          |

Query at runtime:

```python
import percept
percept.registry.available("vision")   # ['claude', 'fake', 'gemini']
percept.registry.available("stt")      # ['deepgram', 'fake']
agent.health()                         # {'status': 'ok', 'backends': {...}, ...}
```

If you select a real backend whose extra isn't installed, resolution raises a **clear, actionable
error** telling you which extra to install — it never fails silently.

---

## Custom / third-party backends

A backend is any object implementing the relevant `Protocol` in `percept.adapters` (`Vision`,
`Conductor`, `Specialist`, `STT`, `TTS`). The `Vision` contract is just:

```python
async def judge(self, condition: str, frame, *, roi=None, corrections=None,
                frames=None, audio=None) -> Verdict: ...
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

---

## Lane discipline (why fakes matter)

The `fake` backends are **deterministic and scripted** — they prove your wiring and the cognition
logic without asserting any model intelligence. This split is deliberate:

- **L1 (offline):** `fake` backends, no keys, fast, reproducible. The default for tests and the
  quickstart.
- **L2/L3 (live):** real backends, keys present. The only place model behavior is exercised.

So a green offline test means *your integration is correct*; testing whether Percept actually
*perceives* the world is a live-backend job. See [quickstart.md](quickstart.md) for both paths.
