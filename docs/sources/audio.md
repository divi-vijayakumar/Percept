---
title: Audio modality — the acoustic-onset → judge wake path
updated: 2026-07-23
applies_to: percept-vision 1.0.0
status: stable
---
# Audio modality — the acoustic-onset → judge wake path

Percept has a first-class **audio** modality, end to end: a cheap on-device detector watches the mic
energy, proposes *where* a sound started, and only then does the brain escalate to a multimodal
`judge(...)` call carrying a **typed audio clip**. Sound is treated exactly like motion — a Tier-0
signal that *gates* the expensive cloud read, never replaces it.

```
   mic energy stream                    rising-edge boundary               escalate
   ───────────────►  acoustic-onset  ─────────────────────────►  brain  ──────────────►
   (already flowing  (CATEGORY_AUDIO   "an onset at t=…"          (gate)   judge(condition,
    for STT)          Tier-0 detector)                                      frame, audio=AudioFrame)
                                                                                │
                                                                                ▼
                                                                   Verdict{satisfied, …}  → FIRE
```

Two pieces make this a *modality* and not a one-off:

1. **`AudioFrame`** — a typed, vendor-neutral, zero-dependency audio signal exposed from `percept`,
   the typed form of the (still-accepted) opaque `judge(audio=…)` payload.
2. **the audio sense category** — `acoustic-onset` is marked `CATEGORY_AUDIO`, so audio detectors are
   discoverable *as a class* (`detectors_in_category("audio")`), the same way motion ones group.

---

## 1. The typed signal — `AudioFrame`

```python
from percept import AudioFrame

clip = AudioFrame(
    pcm=raw_pcm_bytes,     # raw little-endian PCM samples
    sample_rate=16000,     # the STT mic stream rate
    channels=1,            # mono
    sample_format="s16le", # signed 16-bit (default); also "f32le" | "u8"
    t_start=12.0,          # optional clip bounds (seconds) for provenance/freshness
    t_end=14.0,
)
```

| field           | type                          | meaning                                                  |
|-----------------|-------------------------------|----------------------------------------------------------|
| `pcm`           | `bytes`                       | raw little-endian PCM samples                            |
| `sample_rate`   | `int` (default `16000`)       | samples/second per channel                               |
| `channels`      | `int` (default `1`)           | interleaved channel count (1 = mono)                    |
| `sample_format` | `"s16le" \| "f32le" \| "u8"`  | PCM sample layout (default `s16le`)                     |
| `encoded`       | `bytes \| None`               | optional pre-encoded blob (e.g. a WAV/Opus container)   |
| `mime`          | `str \| None`                 | the `encoded` blob's mime type                          |
| `t_start`/`t_end` | `float \| None`             | optional clip bounds in seconds                         |

It is a **frozen** dataclass (stdlib only) and ships in the committed JSON Schema
(`percept/contracts/schema/AudioFrame.schema.json`), so non-Python consumers can validate the wire
shape too.

`AudioFrame.as_pair()` renders it to the legacy `(mime, bytes)` shape the backends already read, so
the typed object flows through the existing path unchanged. Prefer the pre-`encoded` blob when present;
otherwise the raw `pcm` is tagged `audio/L16;rate=…;channels=…`:

```python
clip.as_pair()                       # -> ("audio/L16;rate=16000;channels=1", raw_pcm_bytes)
AudioFrame(encoded=wav, mime="audio/wav").as_pair()   # -> ("audio/wav", wav)
clip.duration_s                      # -> 2.0  (from t_start/t_end; None if unbounded)
```

`judge(audio=…)` stays **typed-but-flexible**: it accepts an `AudioFrame`, raw `bytes`, or a
`(mime, bytes)` pair. A backend that can hear (Gemini) consumes the clip; one that can't (Claude)
degrades honestly to the frame — the returned `Verdict` shape is identical either way, so the gate
downstream never changes.

---

## 2. The audio sense category

`acoustic-onset` is the first-class **audio** Tier-0 detector. Its manifest carries
`category="audio"`, so a whole modality is discoverable at once:

```python
from percept.edge import (
    CATEGORY_AUDIO, detectors_in_category, manifest_of, make_detector,
)

detectors_in_category(CATEGORY_AUDIO)        # -> ["acoustic-onset"]  (audio senses as a class)
manifest_of("acoustic-onset").category       # -> "audio"
```

The JS edge twin mirrors it byte-for-byte (`detectorsInCategory("audio")`, `CATEGORY_AUDIO`), so the
category rides the capability handshake unchanged.

`acoustic-onset` reads the **energy** channel of the mic stream (already flowing for STT) and emits an
`"onset"` boundary on each rising spike, with hysteresis (fire above `high`, re-arm only after falling
below `low`) so a held tone is a single onset, not a stutter. Pure stdlib DSP — no model.

---

## 3. The wake path, end to end

```python
import asyncio
from percept import AudioFrame
from percept.edge import CATEGORY_AUDIO, DetectorBank, detectors_in_category

# 1. compose the audio detector(s) the brain selected for this concern
bank = DetectorBank(detectors_in_category(CATEGORY_AUDIO))   # -> the acoustic-onset bank

# 2. feed the cheap energy signal at native fps; the detector proposes WHERE a sound started
energy_stream = [(0.0, 0.02), (0.1, 0.05), (0.2, 0.6), (0.3, 0.7), (0.4, 0.05)]  # a spike at t=0.2
onsets = []
for t, energy in energy_stream:
    onsets += bank.push(t, energy=energy)
# onsets == [Boundary(detector="acoustic-onset", kind="onset", t=0.2, value=0.6)]

# 3. on an onset boundary -> escalate: ONE multimodal judge with the typed audio clip
async def wake(condition, frame, pcm_bytes, onset_t):
    from percept.backends.fake import FakeVision   # swap for GeminiVision to actually hear
    vision = FakeVision(script={
        # the fake supports a "{frame}|audio" key, proving the clip reaches judge (wiring, not magic)
        f"{frame}|audio": __import__("percept").Verdict(True, 0.92, "scripted: hiss heard"),
    })
    clip = AudioFrame(pcm=pcm_bytes, sample_rate=16000, channels=1,
                      t_start=onset_t - 1.0, t_end=onset_t + 1.0)
    return await vision.judge(condition, frame, audio=clip)

if onsets:
    verdict = asyncio.run(wake("the kettle is boiling", "kettle", b"\x10\x20\x30", onsets[0].t))
    # verdict.satisfied is True -> the gate sees a confident-yes on a rising edge -> FIRE
```

In a wired agent you don't drive `judge` by hand: `Percept.push_audio(...)` feeds a rolling
`AudioWindow`, and a concern whose `PerceptionSpec` sets `modality="audio"` (or `"fused"`) makes the
agent assemble the last `audio_s` seconds into one clip passed to `judge(audio=…)` on escalation. The
acoustic-onset boundary is what tells the gate *when* that escalation is worth a cloud call.

> The boundary that keeps it honest: the edge proposes *where* a sound happened (cheap, native fps);
> the VLM confirms *what it means* (expensive) only at those boundaries. Audio gates the read; it
> never becomes a second, ungated judge.

See also: [`registration.md`](../admission/detectors.md) (register your own detector + category),
[`concepts.md`](../resources/how-it-works.md) (the gate + the one perception seam).
