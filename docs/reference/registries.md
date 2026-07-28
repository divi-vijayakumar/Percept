---
title: Registries and adapters
updated: 2026-07-28
applies_to: percept-vision 1.0.0
status: stable
generated_by: tools/build_registries_doc.py
---

# Registries and adapters

**This page is generated** by `tools/build_registries_doc.py` from the live registries. A backend or detector appears here by *being registered*, not by someone remembering to document it.

Percept's core never imports a vendor SDK. Every external capability resolves through a
**name → factory** registry, which is what makes a backend swappable by string and testable
with a double. There are three kinds of registry, and they answer different questions.

| registry | answers | add to it with |
|---|---|---|
| **Backend seams** | *who performs this external capability?* | `percept.registry.register(seam, name, factory)` or the `percept.backends` entry point |
| **Detectors** | *what cheap local measurement can we take?* | `percept.harness.detectors.register(name, factory, manifest)` |
| **Profiles / packs** | *what configuration applies to this world?* | `DEFAULT_*_REGISTRY.register(obj)` per surface |

---

## 1. Backend seams

**7 seams.** Each maps a name to a factory that takes `settings` and returns
an adapter. Selection is by string (`Percept.create(vision="gemini")`), by adapter
instance, or by env (`PERCEPT_VISION_BACKEND`).

### `vision`

the general judge — decides *is `<goal>` true?* for an admitted frame

Registered in a stock build (7): `claude`, `fake`, `fixture`, `gemini`, `gemini-live`, `openrouter`, `scripted`

### `specialist`

a narrow, authorized judge bound to a verified static context

Registered in a stock build (4): `claude`, `fake`, `gemini`, `scripted`

### `conductor`

turns a plain-language goal into a runnable plan (goal compilation)

Registered in a stock build (5): `claude`, `fake`, `gemini`, `scripted`, `scripted-conductor`

### `stt`

speech to text, for voice-armed watches and fused audio judging

Registered in a stock build (3): `deepgram`, `fake`, `scripted`

### `tts`

text to speech, for spoken escalation

Registered in a stock build (3): `deepgram`, `fake`, `scripted`

### `transport`

the realtime voice/session transport seam

Registered in a stock build (2): `fake`, `scripted`

### `extractor`

pulls structured fields off a frame (OCR, form fields)

Registered in a stock build (7): `fake`, `fields`, `fields-scripted`, `ocr`, `ocr-fields-scripted`, `ocr-scripted`, `scripted`

**Naming.** `scripted` is the canonical keyless double (`fake` is its historical alias);
you select it **out loud** via `mode="offline"` and never fall onto it by omission. A live
backend with a missing key prints one actionable line, never a stack trace.

**Registering your own:**

```python
from percept.registry import register

def my_vision_factory(settings):
    return MyVisionAdapter(model=settings.models.vision)

register("vision", "my-backend", my_vision_factory)
core = percept.Percept.create(vision="my-backend")
```

Or ship it as a plugin — declare an entry point in the **`percept.backends`** group and it
is discovered on import, no call required:

```toml
[project.entry-points."percept.backends"]
my-backend = "my_pkg.adapters:register"
```

Your adapter must pass the runnable conformance suite before it is trusted — see
[Vision adapter interface](vision-adapter.md) and [Custom judge](../how-to/custom-judge.md).

---

## 2. Detectors

**4 registered.** Detectors are the cheap, local, model-free measurements a
plan composes by name, each declaring a `DetectorManifest` of what it senses and when to use it.

**For the how-to — register / introspect / select into a `WatchSpec` / teardown, built-ins,
and the conformance bar — see [Registration](registration.md).** Below is the generated,
can't-drift inventory of what is registered in a stock build:

| name | category | senses | output | when to use |
|---|---|---|---|---|
| `acoustic-onset` | audio | audio energy over time (the mic stream already flowing for STT) | an 'onset' boundary each time energy spikes up (a sound starting) | audio-cued events — a kettle/boil hiss rising, a timer beep, a rep grunt, a knock |
| `motion-periodicity` | motion | motion magnitude over time (frame-diff) | a 'peak' boundary per motion oscillation (≈ one rep), with no per-frame VLM call | counting repetitive movements — reps, strokes, paces; any 'count the N-th time' on a periodic motion |
| `pose-channels` | frame | body pose over time as a DECOMPOSED channel basis (wrist-elevation, ankle-spread, hip-height, torso-height) — the eye measures, it does not decide reps | per-frame channel measurements (via measure()) or one 'sample' boundary per channel | the rhythm sense — counting any repetitive action (reps, strokes, steps) where the brain (percept.rhythm) selects the dominant channel and counts cycles causally with no VLM call |
| `pose-openness` | frame | body pose over time (RTMO/COCO keypoints: ankle-spread + wrist-elevation) | one 'peak' boundary per rep (the open phase of the openness waveform) | counting FAST whole-body reps where motion-diff over-counts and the VLM aliases — jumping jacks, squats, push-ups; any rep with a clean open↔close body shape |

> `_REGISTRY[name]` is a **`(factory, manifest)` tuple**, not a bare factory. Unpack it.

---

## 3. Configuration registries

Six ordered configuration surfaces resolve into one runtime (ADR-008). Four of them carry a
process-wide default registry you can register named objects into and then select by name.

| surface | default registry | what it decides |
|---|---|---|
| `SourceProfile` | `percept.sourceprofile.DEFAULT_SOURCE_REGISTRY` | what KIND of world this is (fixed camera, egomotion, screen) |
| `WatchProfile` | `percept.watchprofile.DEFAULT_REGISTRY` | cadence, gate, salience for one watch |
| `ExecutionPolicy` | `percept.execution_policy.DEFAULT_EXECUTION_POLICY_REGISTRY` | deployment-wide policy (e.g. strict admission, cadence ceilings) |
| `DomainPack` | `percept.domain_pack.DEFAULT_DOMAIN_PACK_REGISTRY` | a bundle of goals + detectors + tunable values for a domain |

These ship **empty** — they are extension points, not a preset catalog. Register your own:

```python
from percept.watchprofile import DEFAULT_REGISTRY, WatchProfile

DEFAULT_REGISTRY.register(WatchProfile(name="kitchen", environment="fixed_camera", ...))
core = percept.Percept.configure(watch_profile="kitchen")
```

Ownership is enforced at resolve time: a surface may only write tunables it owns, and a
write from the wrong surface is **refused, not silently applied**. The full ownership map is
the `owner` column in [Tunables](tunables.md).

---

## Related

- [Tunables](tunables.md) — all 157 knobs, and which surface owns each.
- [Configuration reference](configuration.md) — the `Settings` tree and resolver precedence.
- [Backends](backends.md) — the seams in prose, extras, and selection.
- [Vision adapter interface](vision-adapter.md) — the extension contract + conformance suite.
- [Registration](registration.md) — detector / concern / enricher registration in detail.
