### Changed — the wake-fusion contract (contract MINOR; one migration)

The cheap-evidence wake fusion was one incoherent contract violated five ways. It now says one thing:
**fuse over the MEASURED channels, with `weights` as per-channel priors defaulting to `1.0`; let each
physical measurement populate exactly one channel; make the fused score a first-class `gate_on` /
`AdmissionCause` rather than an override of every other channel; and derive `top_channel` from the same
per-channel contribution terms the score summed.** Both runtimes changed together
(`percept/harness/detectors.py`, `percept/harness/tier0.py`, `percept-edge/src/{skills,edge,contract}.js`).

- **`wake_score` fuses the signals, not the weights.** It iterated `policy.weights`, so any measured
  channel with no explicit weight was deleted from the fusion and `signals.get(channel, 0.0)` made "this
  sensor is not reporting" identical to "this sensor sees nothing". Measured: `{'presence': 0.9,
  'motion': 0.8}` scored 0.9 with no policy but **0.8** under `weights={'motion': 1.0}` — adding a weight
  scored *below* the max it replaced — and exactly **0.0** under `weights={'presenc': 1.0}`, one typo
  producing total deafness with every signal present. Weights are now priors defaulting to `1.0` over the
  measured channels, which makes the weighted form a strict generalization: adding a weight can never
  score below `max(signals)`. Silence a channel by weighting it `0.0`, explicitly. A channel named in
  `weights` but absent from `signals` contributes nothing in either direction.
- **`WakeDecision.top_channel` names a channel that contributed.** It was the argmax of the raw signals,
  computed with no reference to the weights, then returned beside a score computed *from* them — so
  `weights={'motion': 1.0}` on `{'presence': 0.9, 'motion': 0.4}` reported `top_channel='presence'` for a
  score that was all motion. Score and top channel now come from one `_fuse` walk over one population.
  `top_channel` is `None` when every contribution is `0.0`, including when a `floor` lifted the score
  alone. Ties still take the first channel.
- **One measurement, one channel.** `MotionGate.evaluate` built seven keys from four measurements:
  `change` aliased `motion`, `semantic_relevance` aliased `semantic`, `audio_event` aliased `audio`. The
  noisy-OR assumes independent cues, so an alias fused one reading with itself — `weights={'motion': 0.5,
  'change': 0.5}` scored **0.51** on a single reading of 0.6 against **0.30** for the same reading on one
  channel. The three aliases are gone and the producer is the named, documented
  `percept.harness.tier0.cheap_signals` (twin: `cheapSignals` in `skills.js`). `audio` is now **omitted**
  rather than zeroed when no audio source is open. `audio_event` names an acoustic-event class with zero
  registered detectors, so it is absent until one exists; any other channel (`presence`, `pose`, …) is a
  caller overlay through the gate's `signals=` argument.
- **A malformed policy is refused, not degraded.** `coerce_wake_policy` read `weights` by exact key, so
  `'weigths'` yielded `{}`; a stringified number vanished and a `bool` passed as `1.0`. All three landed
  in the legacy `max(signals)` branch *wearing a configured threshold*, so nothing distinguished
  "calibrated" from "running the default". Unknown top-level keys, non-string channel names, bools,
  strings and non-finite numbers now raise the new `UnsupportedWakePolicyError`, naming the field. This
  coercer is deliberately the strict member of the `_coerce_*` family — the reason is on the exception
  class. An empty `weights` remains equal to no policy: a table of all-default priors *is* `max(signals)`.

### Changed — `gate_on: "fused"` (the one migration)

`gate_on` and `AdmissionCause` gain **`fused`**. Previously the `elif fused is not None` branch sat above
the `structural` and `semantic` branches, so **any** non-`None` `wake_policy` replaced both for every goal
in the spec — and the admission was recorded as the goal's own `gate_on`, so a receipt could name
`semantic` for a frame whose measured semantic value was `0.0`. Fusion is now a channel a goal opts in to;
`fixed`, `structural`, `semantic` and `motion` each reach their own channel unconditionally. A `fused`
admission reports cause `fused` and `detector_id = "wake_policy.noisy_or"`; a `fused` goal with no policy
wired scores `0.0` rather than falling back to motion.

> **Migration.** A deployment that relied on a `wake_policy` overriding `gate_on` must now set
> `state["cadence"]["gate_on"] = "fused"` on the goals that should fuse. Goals left on another `gate_on`
> keep their own channel with a policy wired. Nothing else changes: a deployment with no `wake_policy` is
> byte-for-byte unaffected, and `fused` is additive to every vocabulary that carries it
> (`percept.contracts.GateOn`, `percept.behavior._VALID_GATE_ON`, `interest._LIVE_CAUSES`,
> `duration_routes`, the tunables registry, and `GATE_ON` in `percept-edge/src/contract.js`).

### Fixed — two JS-only defects the parity gate could not see

- `SkillBank.push` routed every channel that was not exactly `"energy"` or `"frame"` to frame-diff
  **motion**, so a skill declaring `signal="thermal"` — or a typo like `"enrgy"` — emitted boundaries the
  brain counted as its declared sense. The channel vocabulary is now closed (`motion` / `energy` /
  `frame`), matching Python `DetectorBank`; an unrecognised channel gets no input.
- `AcousticOnset` shipped `high=0.09 / low=0.0225` against Python's `0.001 / 0.00025`, and accepted an
  inverted band. **This changes edge wake behaviour**: `0.09` energy is an RMS amplitude of 0.3, and the
  loudest chunk in the whole 987 s annotated audio corpus reached `0.0277`, so the JS edge's shipped gate
  could wake **zero** times on material it exists to hear. It now ships the measured Python bars and
  refuses `low > high` at construction (`low == high` stays permitted).

### Changed — internal

`percept.harness.tier0._coerce_cadence` and `_GATE_ON_VOCAB` were byte-identical duplicates of
`percept.behavior.coerce_authored_cadence` and `_VALID_GATE_ON`, held equal by a drift test. They are now
imports of those names, so a channel cannot be added to one authoring surface and not the other; the drift
test asserts object identity. `tier0` imports `StreamTimeGuard` from `percept._streamtime` rather than
through the `percept.clock` harness seam (ADR-001 §5). The edge parity gate grew from 47 to **79** checks:
the wake fusion had no coverage at all, and the acoustic-onset vectors were blind to the shipped bar
(which is how the two runtimes shipped different defaults while the gate reported MATCH).
