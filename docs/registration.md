# Registration

Percept has three registries you extend by **registering**, never by editing the gate or adding a
per-task `if`: the **detector registry** (cheap, high-fps edge signal skills the brain composes by name),
the **concern registry** (standing units of consequence-weighted attention the brain watches), and the
**enricher registry** (specialist MODELS that turn a cheap signal into structured facts for the judge).
All three are process-global singletons keyed by name/id, and all expose the same lifecycle shape:

```
register  →  introspect / select into a WatchSpec  →  teardown
```

These names are a **public, semver-stable API**. Within a contract MAJOR (`CONTRACT_VERSION`, see
`percept.CONTRACT_VERSION`) the signatures and the `DetectorManifest` shape only change additively —
new optional args, new functions beside the old ones. A breaking change to any name below is a MAJOR
bump. Import them through `percept.edge` (or `percept.edge.detectors` / `percept.edge.concerns` /
`percept.edge.enrichers`). The implementation currently delegates to the internal `percept-harness`
package.

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
| `pose-openness`     | `frame`  | fast whole-body reps (BlazePose; lazy + opt-in `[pose]`) |

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

## Enricher registry — specialist models (signal → judge context)

An **enricher** (`percept.Enricher`, beside `Specialist`) is a narrow specialist MODEL that
turns a cheap Tier-0 signal into **structured facts** the general VLM judge can't cheaply produce — an
LPR model for a `plate-presence` cue, a fire/smoke classifier for a hot-surface cue, a face-id model for
a known-person cue. The registry routes a *signal/detector name* to the specialist it hands off to, runs
it, and feeds the structured output into `Vision.judge(context=...)`:

```
signal name  →  Enricher.enrich(signal, frame) -> {facts}  →  Vision.judge(context={facts})
```

> Detector vs enricher: a *detector* is a cheap stdlib/edge signal skill that emits boundaries (it
> *finds* a cue); an *enricher* is a heavier specialist model that *reads out structured facts* about a
> cue. The detector says "a plate-shaped thing is here"; the enricher (LPR) says `{"plate": "7XYZ123"}`.

Adding a specialist is the WHOLE integration cost — **no change to Percept core, the judge, or the
contract**:

```python
from percept.edge import register_enricher

register_enricher("plate-presence", LprEnricher())   # an LPR model for the plate cue
register_enricher("smoke-cue", SmokeClassifier())     # a second specialist — one more call, nothing else
```

`register_enricher` keys on the signal name (last-writer-wins, so a product swaps a stub for a calibrated
model in place) and returns the enricher for chaining. Any class with an `enrich(signal, frame) -> dict`
method satisfies the protocol — no base class to inherit.

### Run it (the driver)

```python
from percept.edge import enrich_signal

enr = enrich_signal("plate-presence", signal=cue, frame=frame)  # -> an Enrichment
verdict = await vision.judge(condition, frame, context=enr.context)   # facts injected into the judge
```

`enrich_signal` resolves the signal name, runs the specialist, and returns an `Enrichment(context,
verdict)`:

- `context` — the structured facts to pass as `Vision.judge(context=...)` (`{}` when no specialist is
  registered or it returned nothing — the byte-for-byte un-enriched path).
- `verdict` — present iff the enricher SHORT-CIRCUITED (returned a `{"verdict": Verdict}` key): the brain
  uses that verdict directly and skips the VLM judge call. `enr.is_noop` is the cheap "nothing fired" test.

A specialist that raises (a flaky model, a bad payload) degrades to a no-op — an enrichment failure never
breaks the judge path. F1 ships only the SEAM plus fakes (`NullEnricher` / `FakeEnricher`); no real model.

### Introspect / teardown

```python
enricher_for("plate-presence")   # -> the registered Enricher, or None
registered_enrichers()           # -> the signal names (sorted)
unregister_enricher("plate-presence")  # -> True if registered, False if absent (idempotent)
clear_enrichers()                # -> empties the registry (no built-ins to restore)
```

---

## Stability guarantee

| surface | guarantee |
|---------|-----------|
| `register`, `register_detector`, `make_detector`, `registered`, `list_manifests`, `manifest_of`, `unregister`, `clear_detectors` | semver-stable; additive-only within a contract MAJOR |
| `DetectorManifest` (`name`, `senses`, `when_to_use`, `output`), `Boundary` (`detector`, `kind`, `t`, `value`) | frozen shapes; new fields are additive + optional |
| `register_concern`, `registered_concerns`, `registered_concern_ids`, `get_concern`, `unregister_concern`, `clear_concerns`, `detectors_for_concern`, `concern_to_condition`, `compile_registered_watchspec` | semver-stable; additive-only within a contract MAJOR |
| `register_enricher`, `enricher_for`, `registered_enrichers`, `unregister_enricher`, `clear_enrichers`, `enrich_signal`, `Enrichment` (`context`, `verdict`), `Enricher` protocol (`enrich(signal, frame) -> dict`) | semver-stable; additive-only within a contract MAJOR |
| `Concern` and the wire DTOs (`WatchSpec`, `WatchCondition`) | live behind the frozen `percept` / contract membrane (§0) — a breaking change is a MAJOR bump, flagged not silent |
