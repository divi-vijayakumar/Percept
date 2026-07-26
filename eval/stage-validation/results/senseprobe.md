# SenseProbe — adapter coverage over `percept`'s senses

**Runtime under test:** `percept` 1.0.0, the launch wheel at
`/private/tmp/percept-launch/oss-final-venv-bba572a/lib/python3.12/site-packages/percept/`.
Python 3.12.8, macOS-26.5-arm64. The repo's `packages/percept-vision` was never on `PYTHONPATH`.

**Adapter:** `tools/senseprobe/probe.py`. Lives in `tools/` because unifying the senses inside
`percept/` would break public API on a qualified 1.0.0 wheel to buy the runtime nothing. Nothing
under `percept/` imports it (verified by grep over the candidate source).

## Coverage

All 8 registry entries construct, accept at least one realistic feed, and read back. Nothing was
faked, stubbed past a failure, or left out.

| key | class (module) | constructs | feed works | read works | real signature the adapter had to accept |
|---|---|---|---|---|---|
| `rhythm` | `RhythmTracker` (`percept.rhythm`) | yes | yes | yes | `observe(t: float, channels: dict[str, float]) -> Rep \| None` / `windowed_count(now: float, window_s: float) -> int` |
| `entity` | `EntityGraph` (`percept.entity`) | yes | yes | yes | `ingest(observations: list[Observation], frame=None, source='camera', *, t=None) -> list[Entity]` / `active() -> list[Entity]` |
| `span` | `SpanAssociator` (`percept.individuation`) | yes | yes | yes | `add(cand: EventCandidate) -> tuple[EventCandidate, bool]` / `count(kind: str \| None = None) -> int` |
| `sequencer` | `EvidenceSequencer` (`percept.sequencing`) | yes | yes | yes | `admit(key: Hashable, *, episode_id=None) -> int` / `latest_admitted(key: Hashable) -> int` |
| `reducer` | `SerializedDecisionReducer` (`percept.sequencing`) | yes (gate supplied) | yes (`admit` only) | yes | `__init__(gate: Gate, sequencer=None, *, policy=None, clock=monotonic)`; `admit(key) -> int` sync, **`async submit(...) -> GateDecision`** / `last_applied(key) -> int` |
| `continuity_id` | `ContinuityIdentity` (`percept.identity`) | yes | yes | n/a — pure | `match(obs: Observation, known: list, *, signature=None) -> str \| None` |
| `appearance_id` | `AppearanceIdentity` (`percept.identity`) | yes | yes | n/a — pure | same as `continuity_id`; appearance path needs `signature=` **and** `known` entities carrying `.appearance` |
| `extractor` | `ScriptedOCR` (`percept.extractors.ocr`) | yes | yes | n/a — pure | `extract(condition: str, frame: Any, roi: Box \| None = None) -> Enrichment` |

Nothing could not be probed. No sense needed a heavy fixture: the whole registry runs stdlib-only,
offline, with no model, no camera, and no network.

## Adaptations, and why each was necessary

**`reducer` needs a `Gate`.** It is the only sense with a hard constructor dependency. The registry
factory defaults it to `Gate()`; a caller measuring a specific gate policy passes
`probe("reducer", gate=my_gate)`. A default gate is honest here because what the reducer is under
test *for* is sequence ordering, not the threshold the gate applies.

**`extractor` names a conformer, not the seam.** `percept.extraction.EvidenceExtractor` is a
`typing.Protocol` — `EvidenceExtractor()` raises `TypeError: Protocols cannot be instantiated`. The
registry therefore points at `ScriptedOCR`, the deterministic offline conformer that ships in
`percept.extractors.ocr`. `TesseractOCR` is the other shipped conformer and needs a real
`tesseract` binary, so it is not the default. Construct with
`probe("extractor", script={frame_key: {"text": ...}})`; an unscripted frame returns an empty
`Enrichment` rather than raising (the seam's no-raise-on-a-miss contract).

**`rhythm`'s read takes arguments.** `windowed_count(now, window_s)` has no all-time form, so the
spec binds `read_args=(0.0, float("inf"))` — an unbounded window counts every confirmed rep on the
selected channel independent of the clock. `probe.read(now, window_s)` overrides it.

**Keyed reads replay the last fed key.** `latest_admitted(key)` and `last_applied(key)` are per-key,
so a bare `read()` on `sequencer`/`reducer` replays the first positional argument of the most recent
`feed()`. Before any feed — including immediately after `reset()` — that raises a `TypeError` naming
the requirement, and does **not** increment `errors` (the sense was never called). Guessing a key
would have reported "no admissions yet" about a key the caller never named, because both senses
answer `0` for an unseen key.

**No sense has a reset verb.** Every `SenseSpec.reset` is `None` and `SenseProbe.reset()` re-runs the
factory. `EvidenceSequencer.forget(key)` / `SerializedDecisionReducer.forget(key)` are per-key and
`EntityGraph.restore(dumps)` is a snapshot load; using either as a reset would leave residue a
validation run could not see.

## Interface surprises other agents must know

1. **`SerializedDecisionReducer.submit` is a coroutine.** `admit` is synchronous, `submit` is not.
   The synchronous `probe.feed()` therefore drives `admit` only, and `admit` **applies nothing** —
   after 5 `feed("k")` calls `read()` is still `0`. To move `last_applied` you must
   `await probe.sense.submit(key, seq, verdict)`. Confirmed end to end: `submit("k", 5, Verdict(True, 0.95))`
   returns `action='fire'` and `last_applied` becomes `5`; a later `submit("k", 2, ...)` returns
   `GateDecision(dropped=True, drop_reason='superseded_by_newer_admission', action='silent')`.
   `.sense` exists on the probe for exactly this.

2. **`SpanAssociator.add` does no type checking, and fails late.** `add("a-plain-string")` on a fresh
   associator **succeeds** — it returns `('a-plain-string', True)` and `count()` becomes 1, because
   the first candidate is appended without any attribute access. Only the *second* `add` reaches
   `_same_event` and raises `AttributeError: 'str' object has no attribute 'kind'`. Unlike
   `InterestIntervalEngine.observe`, which type-checks its sample and fails closed, this sense will
   quietly ingest garbage for one call. A harness feeding malformed candidates will see the error one
   unit later than it occurred.

3. **`EntityGraph.ingest` takes a LIST per frame**, not one `Observation`. Feeding a bare
   `Observation` is a shape error, not a convenience the graph absorbs.

4. **The two identity matchers are pure, and their "state" is the caller's.** `match(obs, known)`
   holds nothing between calls; the `known` list *is* the memory, and the harness owns it. Entities
   in `known` are duck-typed — only `.id`, `.label`, `.box` and (for appearance) `.appearance` are
   read, so a test double needs no `Entity` construction. `AppearanceIdentity` silently degrades to
   pure continuity when `signature=None`, so an appearance test that forgets the keyword measures
   continuity and still passes.

5. **`percept.identity.Signature` is `typing.Any`**, not a class — an appearance embedding is any
   sequence of floats (`cosine` is plain stdlib). Do not try to construct one.

6. **`reset()` clears the counters as well as the sense**, so a per-clip `stats` read is never
   contaminated by the previous clip. Snapshot `stats` before resetting if you need cumulative
   totals.

7. **Errors are counted and re-raised, never swallowed.** A failed `feed` increments `errors`, leaves
   `units_fed` unchanged, and still adds its wall time (excluding it would flatter a sense that
   raises).

## Verification performed

Driven through the adapter on the wheel, all passing:

- every registry key: construct → N realistic feeds → `read()` → `stats` assertions
  (`units_fed` matches the feed count, `reads == 1`, `errors == 0`, `wall_time_s > 0`) → `reset()` →
  counters zeroed → post-reset read returns the empty state (`0` / `[]` / `None`);
- `feed` error path: `errors == 1`, `units_fed` unchanged, exception propagates;
- pure senses: `read()` is `None` and `readable` is `False`, so a `None` read is never mistaken for
  state;
- keyed read: replays the last fed key and honours an explicit one;
- keyed read before any feed: raises with a message that says what to do, `errors` stays `0`;
- `reducer.submit` awaited via `.sense`, including the stale-supersession drop;
- `probe("nope")` raises `KeyError` listing the registered senses.
