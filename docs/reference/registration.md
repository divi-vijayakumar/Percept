---
title: Registration
updated: 2026-07-20
applies_to: percept-vision 1.0.0
status: stable
---
# Registration

Percept has two registries you extend by **registering**, never by editing the gate or adding a
per-task `if`: the **detector registry** (cheap, high-fps edge signal skills the brain composes by name)
and the **concern registry** (standing units of consequence-weighted attention the brain watches).
Both are process-global singletons keyed by name/id, and both expose the same lifecycle shape:

```
register  →  introspect / select into a WatchSpec  →  teardown
```

These names are a **public, semver-stable API**. Within a contract MAJOR (`CONTRACT_VERSION`, see
`percept.CONTRACT_VERSION`) the signatures and the `DetectorManifest` shape only change additively —
new optional args, new functions beside the old ones. A breaking change to any name below is a MAJOR
bump. Import them through `percept.edge` (or `percept.edge.detectors` / `percept.edge.concerns` /
`percept.edge.concerns`). The implementation currently delegates to the bundled `percept.harness`
package.

> **`register_detector` is the supported bring-your-own path.** If you are here looking for "how do I
> plug my own OCR / classifier / model into Percept", read [Admission is not
> extraction](#admission-is-not-extraction-what-register_detector-does-and-does-not-do) first — it states
> exactly what that buys you and what it does not.

---

## Detector registry — cheap edge skills

A **detector** consumes a scalar signal stream (`motion` / `energy`) or a raw `frame` and emits **timed
boundaries** (`peak` / `onset` / `threshold`): the edge proposes *where* a rep / onset / crossing
happened at native fps, with no per-frame VLM call. Adding a detector is a deliberate, calibratable
event — the brain *composes* a registered detector into a goal; it never *writes* one.

### Register

Two interchangeable entrypoints (use whichever fits):

```python
from percept.edge import register, register_detector, DetectorManifest

# Ergonomic façade — derives the manifest from keyword strings:
register_detector(
    "tempo-tap",
    TempoTap,                       # a zero-arg factory (the class, if its __init__ defaults suffice)
    senses="audio energy over time",
    when_to_use="count metronome taps",
    output="one 'onset' boundary per tap",
)

# Explicit form — pass a positional DetectorManifest:
register(
    "tempo-tap",
    TempoTap,
    DetectorManifest(
        name="tempo-tap",
        senses="audio energy over time",
        when_to_use="count metronome taps",
        output="one 'onset' boundary per tap",
    ),
)
```

`register_detector` delegates to `register`, so a detector added either way is identical in the
registry. A detector instance is stateful and exposes `push(t, value) -> [Boundary]`, `reset()`, and a
`signal` attribute (`"motion"` | `"energy"` | `"frame"`) naming the channel it reads. Re-registering the
same name replaces the prior entry (last-writer-wins) — how a product overrides a built-in with a
calibrated variant.

### Introspect

```python
registered()            # -> sorted detector names (the edge's advertised capability vocabulary)
list_manifests()        # -> every DetectorManifest (what the conductor reads to compose detectors)
manifest_of("tempo-tap")# -> the one manifest
make_detector("tempo-tap")  # -> a fresh, stateful instance (KeyError if unknown)
```

### Select into a WatchSpec

A goal selects detectors by name through `goal.state["detectors"]`. The selection passes through the
**firewall** (`coerce_authored_detectors`): an unknown name is *dropped*, so a goal can never activate a
skill that isn't registered. The brain then intersects the selection with the connected edge's advertised
capabilities (the **capability handshake**) so it never sends a skill the edge can't run:

```python
from percept.edge import compile_watchspec
spec = compile_watchspec(goals, threshold=0.5, edge_capabilities={"motion-periodicity"})
# each WatchCondition.detectors holds the goal's selected, validated, capability-filtered skills
```

`DetectorBank` constructs one instance per selected name and runs the goal's detectors over the stream,
routing each its signal channel. The default selection is **empty** — an un-annotated goal selects no
detectors, so existing behavior is byte-for-byte unchanged.

### Teardown

```python
unregister("tempo-tap")   # -> True if it was registered, False if absent (idempotent)
clear_detectors()         # -> empties the registry, including the built-ins
```

Teardown affects **future** selection only: a `DetectorBank` already constructed keeps the detector
objects it built. `clear_detectors()` is mainly a test-isolation affordance — production wiring registers
at import and leaves the registry intact for the process lifetime; re-import to restore the built-ins.

### Built-ins

| name                | signal   | counts                                                   |
|---------------------|----------|----------------------------------------------------------|
| `motion-periodicity`| `motion` | reps/strokes/paces — one peak per motion oscillation     |
| `acoustic-onset`    | `energy` | audio-cued onsets — a hiss rising, a beep, a knock       |
| `pose-openness`     | `frame`  | fast whole-body reps (RTMO by default; lazy + opt-in `[pose]`) |

The JS edge twin (`percept-edge/src/skills.js`) mirrors this surface
(`registerSkill` / `makeDetector` / `registered`); the manifest + `Boundary` field names match
byte-for-byte so a manifest rides the wire unchanged.

---

## Concern registry — standing concerns

A **concern** (`percept.Concern`) is a specific standing thing-to-watch — one per id, carrying
its own plain-language `condition`, `consequence`, and `tracking`. Concerns were previously composed only
per-call (the conductor authors goals; `compile_watchspec` turns each watch goal into a `WatchCondition`
on the fly). The concern registry adds the missing **registration lifecycle**, mirroring the detector
registry and running **alongside** the goal path — it does not touch `compile_watchspec` or its
byte-for-byte behavior.

> Detector vs concern: a *detector* is a cheap reusable signal skill (one impl, many goals); a *concern*
> is a specific standing thing-to-watch (one per id, with its own condition + consequence). They share
> the register/teardown shape but not the keyspace.

### Register

```python
from percept.edge import register_concern
from percept import Concern
from percept.adapters import Subject

register_concern(Concern(
    id="milk",
    identity=Subject(kind="entity", ref="pot of milk"),
    condition="milk is boiling over",
    say="your milk is about to boil over",
))
```

`register_concern` keys on `Concern.id` (last-writer-wins, so a refinement updates a standing concern in
place) and returns the concern for chaining. It also accepts a `Goal` and runs it through
`Concern.from_goal` (the compat shim), so a goal-shaped caller can register without first building a full
Concern.

### Introspect

```python
registered_concerns()      # -> all concerns, ordered by id
registered_concern_ids()   # -> the ids (sorted)
get_concern("milk")        # -> the one concern (KeyError if absent)
```

### Select into a WatchSpec

Registered concerns compile DOWN to a `WatchSpec`, one `WatchCondition` per concern (in id order):

```python
from percept.edge import compile_registered_watchspec
spec = compile_registered_watchspec(threshold=0.5)              # all registered concerns
spec = compile_registered_watchspec(threshold=0.5, concerns=[c])# an explicit subset, no registration
spec = compile_registered_watchspec(threshold=0.5, edge_capabilities={"motion-periodicity"})
```

Each `WatchCondition` gets `goal_id <- concern.id`, `anchor_text <- concern.condition`, and `detectors`
selected via `detectors_for_concern` — an OPTIONAL `detectors` list attached to the concern, validated
through the same registry firewall and capability handshake as the goal path. A concern with no
annotation selects no detectors (byte-for-byte parity with the goal default). Use `concern_to_condition`
to compile one concern directly.

### Teardown

```python
unregister_concern("milk")  # -> True if it was registered, False if absent (idempotent)
clear_concerns()            # -> empties the concern registry
```

Teardown affects future compilation only: a `WatchSpec` already compiled keeps the conditions it
captured.

---

## Admission is not extraction — what `register_detector` does and does not do

`register_detector` is **the** supported bring-your-own path, and it is a bring-your-own **admission**
path. Say that precisely, because the loose version ("plug in your own OCR") walks you into a wall.

Your model can be anything — pytesseract, PaddleOCR, a CLIP embedding, a small on-device VLM. It reaches
Percept through one of two seams:

| you want | the seam | what it produces |
|---|---|---|
| "spend the cloud judge on THIS frame" | `percept.run(..., semantic_adapter=f)` / `structural_adapter=f`, with the goal gated on that channel | a `0..1` score in `Tier0Signal.semantic` / `.structural`; escalation when it clears the goal's threshold |
| "count the occurrences, don't judge at all" | `register_detector(name, MyDetector)` selected by a goal | timed `Boundary` events fed to `perceive_boundary` — the goal fires per its `accumulate.fire_when` with **no vision call** |

Both answer **WHEN to look closely.** Neither answers **WHAT it says.**

A detector boundary is `document_changed at t=8.2`. It is *not* `invoice_number = "INV-2291"`, and it is
*not* `speaker = "buyer"`. There is no channel in the watch loop that carries your model's extracted
facts into the semantic judge: `Percept.perceive_judged` takes a frame, not a fact bundle. If you need
facts in front of the model, call the judge yourself — `Vision.judge` accepts `context=`:

```python
verdict = await vision.judge(
    "the invoice total changed",
    frame,
    context={"invoice_number": "INV-2291", "total": "4,120.00"},   # read as authoritative
)
```

In practice most workloads do not need this. A modern VLM judge reads small on-screen text directly and
scores it correctly; what the SDK was missing was not extraction but **admission** — a way to say "the
page changed, look now" on a stream whose pixels barely move. That is exactly what the two seams above
are for.

### Runnable example

[`examples/byo_admission_detector.py`](../../examples/byo_admission_detector.py) — offline, no keys, no
model. It registers a `document-changed` detector over a cheap "ink density" read (stand in your real
OCR), drives the same score through `semantic_adapter`, and prints what each seam produced:

```
$ python examples/byo_admission_detector.py
seam 1 — frames admitted to the judge by the semantic adapter: [2.0, 4.0]
seam 2 — detector boundaries (no vision call spent):          [2.0, 4.0]
neither carries an extracted fact; pass context={...} to judge yourself.
```

The detector protocol it implements is the whole contract: a `signal` attribute
(`"motion"` | `"energy"` | `"frame"`), `push(t, value) -> [Boundary]`, and `reset()`.

### Check it before you ship it — `percept.conformance`

Both seams above have a runnable conformance suite, the same one the Vision adapter has:

```bash
python -m percept.conformance mypkg.ocr:DocumentChanged --role detector
python -m percept.conformance mypkg.ocr:ink_density     --role semantic
```

or in your own test suite:

```python
from percept.conformance import check_detector, check_tier0_adapter

def test_my_detector_is_well_formed():
    report = check_detector(DocumentChanged, manifest=MY_MANIFEST)
    assert report.ok, report.format()

def test_my_admission_adapter_is_well_formed():
    report = check_tier0_adapter(ink_density, role="semantic", frames=load_my_frames())
    assert report.ok, report.format()
```

`check_detector` verifies the manifest is well-formed, the call signature is the one `DetectorBank`
uses, boundaries are real attributed `Boundary` objects, the detector is deterministic on a fixed
input, `reset()` really restores initial state, and the name is selectable end-to-end into a
`WatchSpec` through the real firewall. `check_tier0_adapter` verifies the score is a bounded `0..1`
float, the adapter survives a degenerate frame (1×1, uniform black/white), and a real `MotionGate`
carrying it runs a stream without crashing and reports the value on the matching `Tier0Signal`
channel. A frame-backed detector's push checks **skip** (loudly, marked `PARTIALLY CHECKED`) unless
you pass `samples=` from your own stream — there is no universal probe frame.

> **This proves your plug-in is WELL-FORMED. It is NOT a measure of your model's quality.** No check
> asserts a correct answer: a conformant detector is free to be *wrong*, it is not free to be *shaped*
> wrong. **A green conformance report does not mean "my OCR model is good."** That question — does my
> detector actually help? — is `percept eval`'s (`percept-eval`): a paired A/B on **your** fixtures
> against **your** pre-registered bars.

### Removed: the enricher registry

`register_enricher` / `enrich_signal` / `enricher_for` (`percept.edge.enrichers`) were **removed** in
0.7.x. They were documented here as "add an OCR specialist without editing core", but they had **zero
callers**: nothing in the runtime ever ran a registered enricher, so registering one changed nothing
about any watch. Rather than keep an architecture-shaped seam that silently does nothing, it is gone.
`percept.Enricher` (the protocol) survives as a deprecated name for one contract MAJOR and is scheduled
for removal. Use the two seams above for admission, and `Vision.judge(context=...)` for facts.

---

## Stability guarantee

| surface | guarantee |
|---------|-----------|
| `register`, `register_detector`, `make_detector`, `registered`, `list_manifests`, `manifest_of`, `unregister`, `clear_detectors` | semver-stable; additive-only within a contract MAJOR |
| `DetectorManifest` (`name`, `senses`, `when_to_use`, `output`), `Boundary` (`detector`, `kind`, `t`, `value`) | frozen shapes; new fields are additive + optional |
| `percept.conformance.check_detector`, `check_tier0_adapter`, `CONFORMANCE_VERSION` | semver-stable; a new REQUIRED check is a MAJOR bump of `CONFORMANCE_VERSION` (an adapter that passed may now fail), a new ADVISORY check is MINOR |
| `register_concern`, `registered_concerns`, `registered_concern_ids`, `get_concern`, `unregister_concern`, `clear_concerns`, `detectors_for_concern`, `concern_to_condition`, `compile_registered_watchspec` | semver-stable; additive-only within a contract MAJOR |
| `percept.run(..., structural_adapter=, semantic_adapter=)` (the tier-0 admission seam), `Vision.judge(context=...)` | semver-stable; additive-only within a contract MAJOR |
| `register_enricher`, `enricher_for`, `enrich_signal`, `Enrichment`, `NullEnricher`, `FakeEnricher` | **REMOVED** in 0.7.x (zero callers — the seam did nothing); `Enricher` protocol deprecated, removed next MAJOR |
| `Concern` and the wire DTOs (`WatchSpec`, `WatchCondition`) | live behind the frozen `percept` / contract membrane — a breaking change is a MAJOR bump, flagged not silent |
