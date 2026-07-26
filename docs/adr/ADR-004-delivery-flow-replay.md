# ADR-004: Public delivery flow & normative replay definition

- **Status:** Proposed
- **Depends on:** ADR-001 (the Gate/judgment contract — `Verdict -> GateDecision/GateAction ->
  JudgmentEvent`, unchanged by this ADR); `percept.watch` (the frozen public entrypoint,
  `packages/percept-vision/percept/watch.py`).
- **Related:** ADR-002 (Runtime health & coverage — health is a separate, parallel channel; this
  ADR does not fold health into `WatchEvent` delivery, consistent with ADR-002 §3. **Note ADR-002
  is still `proposed`: its `WatchHealth` DTO is UNBUILT — zero occurrences under `percept/`. The
  shipped channel is `run(..., on_health=...)` carrying
  `percept.runtime.worker.RuntimeHealthEvent` (`worker.py:141-147`)**); ADR-003
  (Evidence artifacts — `evidence.frame_ref` resolution is referenced, not redefined, here).
- **Source:** `percept/watch.py`, `percept/gate.py`, `percept/behavior.py`, `percept/agent.py`,
  `percept/contracts/__init__.py` (`WatchEvent`/`WatchAction`), `percept/runtime/worker.py`
  (contrasting `run()` delivery model), `percept/backends/fake.py` (`FakeVision`/`FakeJudge`, the
  existing deterministic-judge precedent).

## Context

ADR-003 named this ADR as the future home of "what `Incident` durably carries forward from a
`JudgmentEvent`." That framing assumes the fuller `Verdict -> GateDecision -> JudgmentEvent ->
Incident -> DeliveryAttempt[]` chain ADR-001/002/003 describe is the thing shipping today. It is
not, in full. `JudgmentEvent` and `Incident` both exist as real dataclasses
(`percept/contracts/__init__.py:834` and `:1228`), but `DeliveryAttempt` does not exist anywhere in
the contracts module — there is no class by that name in this codebase today. What is actually
shipped and frozen as the public seam (per its own module docstring, `watch.py:1-23`) is narrower:

```python
watch(stream, goals, profile=None, *, percept=None, source="camera") -> AsyncIterator[WatchEvent]
```

`WATCH_VERSION = "1.0.0"` (`watch.py:55`) versions this function's signature and streaming
semantics; `CONTRACT_VERSION` in `percept/contracts/__init__.py:33` (currently `"1.6.0"` — grew
additively through `WatchAction`/`WatchEvent`, `SubjectRef`, the claim-mutation channel, and the
judgment envelope) separately versions the DTO wire shapes it returns. This ADR pins
the delivery contract for what is actually implemented — `watch()` and `WatchEvent` — rather than
inventing behavior for the `Incident`/`DeliveryAttempt` chain ADR-003 gestured at. Reconciling
`WatchEvent` with that fuller chain is explicitly deferred (see Open Questions).

Two questions have been asked repeatedly by early integrators and have no written answer yet:

1. **What does a caller actually get, and when, and what happens if their code is slow?** — the
   public delivery flow from a judged verdict to a `WatchEvent` in their `async for` loop.
2. **If I record a stream and play it back, do I get "the same thing"?** — and critically, what
   "the same thing" can honestly mean when a live VLM sits in the loop.

Percept has two independent implementations that turn a judged verdict into an outward
notification, and this ADR must be honest that they are not the same:

- `percept.watch()` (`watch.py`) — the frozen public contract. A bare async generator with **no
  internal queue**: `async for frame in _aiter(stream): now = core.clock(); fired = await
  core.perceive_judged(frame, source=source, t=now); for ev in fired: yield
  _as_watch_event(ev, t=now)` (`watch.py:193-197`). It is deliberately "a thin wrapper, not a new
  firing path... zero judge/gate logic" (`watch.py:5-6`).
- `percept.runtime.worker.run()` (`worker.py:856`) — a separate, richer, sink-callback runtime
  (`on_incident`/`on_ask`/`on_frame`/`on_health`) with a bounded capture queue
  (`source_queue_size=64` default, `worker.py:868`), drop-oldest frame loss under load
  (`stats.frames_dropped`, `worker.py:113,1714`), and an async single-flight judge scheduler
  (`_JudgeScheduler`, `worker.py:150`) that keeps a slow VLM call from stalling capture. It also
  wraps `on_incident` in an `IdempotentSink` bound to a persistent `IdempotencyLedger` keyed by
  `incident.id`, so "a redelivery of an event_id already delivered to terminal-OK is deduped"
  (`worker.py:1160-1171`).

`watch()` has none of `run()`'s queueing, drop-on-overflow, or idempotency ledger. That is not an
oversight to paper over — it is the actual, current shape of the frozen public contract, and this
ADR states plainly what that does and does not guarantee.

## Decision

### 1. Public delivery flow: verdict → gate action → `WatchEvent` → consumer

**The path.** `core.perceive_judged(frame, ...)` (`agent.py:1023-1248`) judges every active `watch` goal
and, per goal, computes `base_key = (goal.id, focus.id if focus else None)` (`agent.py:1135`) then
calls `self._fire_concern(...)`, which is `BehaviorExecutor.fire_judged` (`behavior.py:320-378`,
aliased per `behavior.py:3-10`). `fire_judged` classifies the verdict
(`classify(verdict, freshness, threshold, fresh_min)`, `gate.py:186-198`) and calls
`self.gate.evaluate(key, verdict, freshness=freshness, hold_s=..., now=t, ...)` — "the only firing
path" (`gate.py:202`; `evaluate` returns the `.action` of `Gate.decide`, `gate.py:321-342`) —
which returns exactly one of `FIRE` / `ASK` / `SILENT`
(`GateAction = Literal["fire","ask","silent"]`, `gate.py:76`). `fire_judged` then maps that:

- `FIRE` → optionally gated further by an authored `fire_when(n)` nth-distinct-occurrence rule
  (`behavior.py:342-345`); then, if `is_hard_quadrant(consequence_of(goal))`
  (`concern.py:50,71`), the **outward** action is escalated to `ASK` (`Fired(goal.id, ASK,
  escalate_text(goal), ...)`, `behavior.py:346-348`) even though the gate's internal edge state
  was already consumed as a fire one line above; otherwise `Fired(goal.id, FIRE, goal.say, ...)`
  (`behavior.py:349`).
- `ASK` → `Fired(goal.id, ASK, ask_text(goal), ...)` (`behavior.py:350-351`).
- `SILENT` → `None` — no `Fired` object is produced (`behavior.py:349,377`).

Back in `perceive_judged`, a non-`None` `Fired` becomes a `FireEvent` (`agent.py:1214-1218`,
`FireEvent` defined `agent.py:42-50`: `goal_id`, `action: GateAction` (`"fire"|"ask"`), `text`,
`entity_id`, `verdict`) and is appended to that call's `events: list[FireEvent]`
(`agent.py:1096,1219`). `watch()`'s loop then maps each returned `FireEvent` to a `WatchEvent`
1:1 via `_as_watch_event` (`watch.py:93-101`), copying `goal_id`, `action`, `text`, `entity_id`,
`verdict`, and stamping `t` from the **same** `now = core.clock()` value used to call
`perceive_judged` (`watch.py:194-197`) — `WatchEvent.t` is frame-admission time, not a delivery
timestamp. `WatchAction` (`contracts/__init__.py:1129`) is declared to "mirror
`percept.gate.GateAction` value-for-value" but is kept as an independent `Literal`, not imported,
so the contracts package stays dependency-free of the cognition core (`contracts/__init__.py:
1130-1132`).

**`"silent"` is in the type but never emitted.** `WatchEvent`'s own docstring is explicit:
`"silent"` exists "for forward/observability completeness"; "today's `watch()` wraps the existing
internal gate wiring byte-for-byte, and that wiring only ever surfaces FIRE/ASK to callers (a
silent tick simply emits nothing)" (`contracts/__init__.py:1141-1144`). A caller cannot distinguish
"condition false this tick," "condition true but not a rising edge," or "goal not yet evaluated"
from `watch()`'s stream alone — none of those produce an event, and **`watch()` has no second
channel at all**: its only output is the `WatchEvent` iterator.

What actually exists for a consumer that needs more than the fire/ask stream is
`percept.runtime.worker.run()`'s `on_health` sink, which receives a
`RuntimeHealthEvent(kind, ts, data)` (`worker.py:141-147`). Be precise about what that does and
does not give you:

- It is **not per-tick liveness.** It is event-driven: `runtime_started` / `runtime_ended`,
  `judge_dispatch`, `gate_suppressed` (`worker.py:563`), `judge_degraded` (`worker.py:539-558`),
  `audio_stale`, `timebase_skew` (`worker.py:520-537`), `escalation`, `runtime_error`,
  `state_saved`. A quiet-but-healthy run is silent between them. The per-frame hook is `on_frame`
  (`worker.py:864, 1716-1718`), not `on_health`.
- It **now closes every run with a coverage figure**: `RuntimeHealthEvent("runtime_ended")`
  carries the run's admission coverage instead of the empty payload it used to carry, and a run
  that saw frames but admitted none of them emits
  `RuntimeHealthEvent("zero_admission_coverage")` immediately (`worker.py:1866-1885`, documented at
  `worker.py:1020-1023`). Before X4 this channel emitted no coverage signal whatsoever, so an
  integrator debugging "nothing ever fired" got nothing from it.
- ADR-002's `WatchHealth` — the frozen staged vector with `state`/`reasons`, latest-state retention
  and transition dedup (ADR-002 §1) — **does not exist in the tree.** It is a proposed design, not
  a channel anything can subscribe to today. Do not route an integrator to it.

The non-folding boundary ADR-002 §3 draws for health still holds, restated from the delivery side:
health does not become a `WatchEvent`. The correction here is only about *which* health surface an
integrator can actually reach — `run(on_health=...)`/`RuntimeHealthEvent`, not `WatchHealth` — and
that reaching it means using `run()` rather than `watch()`.

**At-most-once-per-episode.** The gate's rising edge is the mechanism, and it is keyed per
`(goal_id, entity_or_episode_id)` (`gate.py:13-16`) — a new entity gets a fresh edge; the same
entity does not double-fire until it goes confident-`false` and re-arms. Mechanically, inside
`Gate.evaluate`:

- A `FIRE` decision sets `self._armed[key] = False` and stamps `self._last_fire[key] = now`
  (`gate.py:444-448`; the dwell-satisfied fire does the same at `gate.py:467-471`) — this happens
  **before** `fire_judged`'s hard-quadrant escalation check, so the edge is consumed identically
  whether the outward `WatchAction` the caller receives ends up being `"fire"` or the escalated
  `"ask"`. The guarantee is anchored to edge consumption, not to the literal string in
  `WatchEvent.action`.
- A confident-`false` re-arms it: `self._armed[key] = True` (`gate.py:414-417`).
- `refractory_s` (L3, off by default — `None`/`0` = "byte-for-byte the historical gate," `gate.py:
  17-20, 365-375`) is an additional time-based guard: a fire landing within `refractory_s` seconds of
  that key's last **real** fire is suppressed even if a spurious confident-`false` blip re-armed
  the edge in between (`gate.py:434-443`). It is set gate-wide from
  `cfg.gate.refractory_s or None` at construction (`agent.py:504-507`) — off unless a caller's
  `Settings`/profile turns it on.
- `require_false_first` (opt-in, default off) and `hold_s` (dwell, opt-in via a goal's program)
  are two further, off-by-default authored variants on the same edge (`gate.py:360-383,406-433,
  449-472`) — neither changes the default at-most-once semantics described above.

So: **the guarantee is "at most one edge-consuming outward event per `(goal, entity)` per
continuous true-episode,"** not "at most one event per goal ever" (multiple entities under one
goal each get their own edge) and not "exactly one" (a `SILENT` tick or an aborted judge produces
zero).

**Fail-closed judging never fires.** If the judge call raises or returns something malformed
(`_JudgeDegraded`), `perceive_judged` never reads `.confidence`/`.satisfied` off it and degrades
that goal's tick to an ASK-class abstention. It preserves the gate's armed/disarmed bit and
last-fire/refractory memory, but clears pending positive-dwell and last-supported evidence: a
failed read cannot prove continuity. No-hold gate state is unchanged
(`agent.py:1995-2007,2561-2577`). A degraded judge call cannot produce a `FIRE`.

**Backpressure on a slow consumer.** `watch()` has no internal queue (contrast `run()`'s bounded
`source_queue_size`, `worker.py:868,1328`, and drop-oldest `frames_dropped`, `worker.py:113,1714`).
Because the loop `await`s `perceive_judged` inline and only then `yield`s, the async generator
protocol itself provides backpressure: the next `stream` item is not pulled (`_aiter`'s
`async for item in stream`, `watch.py:104-111`) until the consumer resumes the generator by
requesting the next `WatchEvent`. Concretely:

- **Guaranteed:** no `WatchEvent` that `perceive_judged` has already produced is ever dropped by
  `watch()` itself — the `for ev in fired: yield _as_watch_event(ev, t=now)` loop
  (`watch.py:196-197`) yields every element unconditionally. Delivery order matches frame-admission
  order (no concurrency inside `watch()` to reorder it).
- **Best-effort / NOT guaranteed:** frame throughput and freshness while the consumer is slow.
  `watch()` makes no promise about `stream`'s own buffering — if `stream` wraps a live, free-running
  source, frames may be lost **upstream of `watch()`**, outside its control, while the consumer is
  stalled. This is a materially different guarantee from `run()`, which owns an explicit bounded
  capture queue and counts what it drops. A caller needing bounded-loss, counted frame drop under a
  slow consumer should use `run()`, not `watch()`, today.

**Other things this ADR states are NOT guaranteed — research preview, said plainly:**

- No idempotency ledger for `watch()`. Unlike `run()`'s `IdempotentSink`/`IdempotencyLedger`
  (`worker.py:1160-1171`), a `WatchEvent` carries no `id` and there is no persisted
  delivery-attempt record. The only de-dup mechanism is the in-memory gate edge state, itself
  bounded by `Gate`'s TTL/LRU eviction (`_DEFAULT_MAX_KEYS=10_000`,
  `_DEFAULT_KEY_TTL_S=3600.0`, `gate.py`) and **not persisted** by `watch()` itself. A plain-goal
  core can explicitly use `Percept.snapshot()`/`restore()` to round-trip armed edge state, but
  `watch()` never calls them. A core with an applied compiled Watch refuses snapshot/save with
  `CompiledWatchSnapshotUnsupportedError` because its runtime bindings are session-local.
  Furthermore, `Gate.restore()` deliberately drops pending positive dwell: the pre-restart clock
  cannot prove continuous support in the new process. Across an unmanaged restart, a still-true
  plain goal can re-fire; a duration Watch must begin its pending interval again in a fresh
  supported session.
- No wall-clock delivery-latency SLA — how quickly the consumer's `async for` loop resumes after a
  `yield` is entirely a function of the consumer's own code, not bounded by `watch()`.

**`profile` IS resolved — and it changes the numbers §1 is stated against.** (Correction: an earlier
draft of this ADR listed profile resolution under "NOT guaranteed," describing an older `watch.py`
before profile resolution was wired.
That was false as shipped and is retracted here.) Whenever `watch()` builds its own core — i.e.
whenever `percept=` is NOT passed — `_build_core(profile)` (`watch.py:77-90`) branches on the
argument's type:

- A real `percept.watchprofile.WatchProfile` **is resolved for real** by
  `percept.profile_wiring.resolve_watch_profile(profile)` (imported at `watch.py:47`, applied at
  `watch.py:83-89`), and the result is applied at CORE CONSTRUCTION time — never as a mid-run
  mutation: `Percept.create(settings=resolved.settings, sense_policy=resolved.sense_policy,
  plan_mode=resolved.plan_mode)`. Concretely, `resolve_watch_profile` seeds from `load_settings()`
  and overrides (`profile_wiring.py:61-83`):
  - `gate.require_false_first` and `gate.refractory_s` — **both are §1 edge semantics.** Passing a
    profile can therefore turn on the start-anchored rule and the time-refractory that the
    at-most-once-per-episode bullets above describe as "off by default." A caller who read those
    bullets and then passed a profile is running different gate semantics than they read about.
  - `edge.salience_stat` and the `edge.motion_gate*` thresholds (which frames are looked at at all).
  - `sense_policy` (cadence) and `plan_mode`, threaded into `Percept.create` directly.
  Everything a `WatchProfile` has no opinion on — including `gate.threshold` — is left at the
  `load_settings()` value; `Gate` itself is constructed from `cfg.gate` at `agent.py:504-511`, and
  `fresh_min` is a `Gate` constructor default (`gate.py:207`), not a profile-settable knob.
- A bare `Settings` is used directly, and any other object's `.settings` attribute is read if
  present (`_settings_from_profile`, `watch.py:58-74`). Neither of those shapes carries a
  `sense_policy`/`plan_mode` override — only a real `WatchProfile` does.
- `None` (the default) resolves to `None`, so `Percept.create` falls back to `load_settings()` —
  today's documented default behavior, byte-for-byte unchanged.

Two things about `profile` that genuinely are NOT guaranteed:

- **`percept=` silently wins over `profile=`.** `watch()` uses a caller-supplied core as-is
  (`core = percept if percept is not None else _build_core(profile)`, `watch.py:191`), so a
  `profile` passed alongside an explicit `percept=` is ignored, with no error and no warning.
- **No mid-run reprofiling.** A `WatchProfile` is resolved once, at core construction; nothing in
  `watch()` re-resolves or re-applies it while the stream is running.

### 2. Normative replay definition: behavioural equivalence under a fixed judge, not bit-equality

**The claim this ADR pins:** replaying a recorded stream through `watch()` and getting "the same
result" can only ever mean **behavioural equivalence of the deterministic layer, under a FIXED
judge** — never bit-equality of output against a live, hosted vision model. Conflating the two is
the single most likely way this contract gets misread, so the two halves are separated explicitly
below.

**What is deterministic, and why.** Given (a) the same ordered frame sequence, (b) the same set of
goals, and (c) the same sequence of `now` values, the entire path from a `Verdict` to a
`WatchEvent` is a pure function of its inputs:

- `classify()` is pure arithmetic — `verdict.confidence < threshold`, `freshness < fresh_min`,
  `verdict.satisfied` (`gate.py:186-198`). No randomness, no I/O.
- `Gate.evaluate`'s armed-bit rising edge, `require_false_first`, dwell, and refractory logic are
  all pure state transitions over `(key, truth, now)` (`gate.py:405-472`) — deterministic given the
  same key and the same `now` sequence.
- `BehaviorExecutor.fire_judged`'s mapping from `GateAction` to `Fired`/`None`, including the
  hard-quadrant `FIRE`→`ASK` escalation, is a pure function of the goal's authored config and the
  gate's return value (`behavior.py:320-378`).
- `watch()`'s own loop and `_as_watch_event` mapping add no branching on anything but the above
  (`watch.py:193-197,93-101`).

**"The same `now` sequence" is load-bearing, not a footnote.** `core.clock` defaults to
`time.monotonic` (`clk = clock or time.monotonic`, `agent.py:481`) — wall-clock, and therefore
**not** reproducible run-to-run by default. `watch()` reads `now = core.clock()` fresh once per
frame (`watch.py:194`) and that same value feeds both `perceive_judged`'s `t` and the resulting
`WatchEvent.t`, and is what refractory/dwell timing compares against. Byte-identical replay of gate
timing requires the caller to inject a deterministic clock (`Percept.create(clock=...)`) — e.g. one
driven from each recorded frame's own timestamp — not the default. `watch()` does nothing to make
this automatic today; that is a real limitation, stated here rather than assumed away.

**What is NOT deterministic, and cannot be made so by this ADR.** The judge call itself —
`self.vision.judge(...)`, reached via `self._judge(...)` inside `perceive_judged` (`agent.py:1151-1154`,
`agent.py:1561`), dispatched to whichever
backend `Percept.create(vision=...)` resolved (e.g. `percept/backends/anthropic.py`,
`percept/backends/gemini.py` — real network calls to a hosted model) — is not reproducible.
Sampling/temperature, server-side model-version drift, provider-side safety/routing changes, and
network/timeout variance (which itself can flip the fail-closed degrade path of §1) can all cause
the identical frame, replayed twice against a **live** backend, to return a different
`Verdict.satisfied`/`confidence`. That changes `classify()`'s truth, which changes the `GateAction`,
which changes whether/which `WatchEvent` is emitted — regardless of how carefully the clock and
frame sequence are pinned. No amount of gate-side determinism fixes this, because the
non-determinism enters one layer below the gate, at the judge.

**The normative contract, stated precisely:** two replays of the same recorded stream against the
same goals are defined to match if and only if they are run against a **fixed judge** — a `Vision`
implementation that returns the identical `Verdict` for the identical `(condition, frame, roi,
exemplar, corrections, frames, audio, context)` input every time — under a pinned clock. This
codebase already has the working precedent for a fixed judge: `FakeVision`/`FakeJudge`
(`percept/backends/fake.py`), whose own module docstring states the exact property this ADR needs:
"Deliberately deterministic, offline, key-free... **scripted, never random**"
(`backends/fake.py:1-10`); `FakeVision.judge` "looks the frame token up in a dict... it does not
'understand' anything" (`backends/fake.py:7-9,28-37`). Under such a fixed judge plus a pinned
clock, `watch()`'s output is a pure function of its inputs (per the deterministic layer above), and
"replay produces the same result" is a claim this ADR can actually stand behind.

**What this explicitly is NOT:** a claim that replaying a recorded stream against a **live**
`vision="claude"`/`vision="gemini"` backend reproduces a prior run's `WatchEvent` sequence. It does
not, and no future implementation of "replay" should imply otherwise by omission. This is the same
honesty discipline ADR-003 §3 already applies to evidence resolution (typed failures, never a
silent empty result standing in for "don't know") — applied here to determinism: never claim a
guarantee this system cannot deliver.

## Consequences

- A consumer of `watch()` gets a precise, written answer to "what will I receive, at most how
  often, and what happens if my code is slow" — §1 above — instead of having to read `gate.py` and
  `behavior.py` themselves to find out.
- The gap between `watch()` (thin, unqueued, no idempotency ledger) and `run()` (queued,
  drop-counted, idempotent-sink-wrapped) is now a documented, deliberate difference in guarantees,
  not an undocumented inconsistency a caller discovers by reading two files and wondering which one
  is "real."
- Anyone building a replay/regression-testing tool for Percept now has a normative target: pin the
  clock, pin the judge (a `FakeVision`/`FakeJudge`-style fixed `Vision`), and compare `WatchEvent`
  sequences — rather than the unachievable target of bit-matching a live model.
- This ADR does **not** resolve the ADR-003 forward reference to `Incident`/`DeliveryAttempt` — see
  Open Questions. Any future work that folds `watch()` into that fuller chain must not weaken the
  guarantees pinned in §1 (in particular: `watch()` staying a thin, unqueued wrapper is a stated
  design choice, `watch.py:5-6`, not an oversight to "fix" by silently adding a queue).

## Open questions — explicit API-scope decisions

1. **Is there a first-class "recorded-judge" replay backend, or is `FakeVision`/`FakeJudge`
   (hand-scripted) the only sanctioned fixed judge?** I read `percept/backends/` in full for this
   ADR (`anthropic.py`, `anthropic_text.py`, `deepgram.py`, `fake.py`, `fake_judge.py`, `gemini.py`,
   `gemini_live.py`, `openrouter.py`) and found no "VCR"/cassette-style `Vision` implementation that
   replays previously-captured **live** verdicts frame-for-frame. The pattern for a fixed judge
   exists (`FakeVision`/`FakeJudge`, hand-scripted); a build-a-fixture-from-a-prior-live-run tool
   does not, as far as this ADR's source reading found. Whether to build one, and its shape, is not
   decided here — it is real, undecided work this ADR surfaces rather than invents.
2. **Byte-identical `WatchEvent` field equality, or a looser "same actions, same order" bar?**
   `WatchEvent.t` is stamped from `core.clock()` per frame (`watch.py:194`) and will only match
   across replays if the clock is pinned identically. Recommended default: field-value equality
   given an explicitly pinned clock (the stronger, more useful bar for regression testing) — but
   this is not decided in code or in any PRD text this ADR's author read, and is called out rather
   than assumed.
3. **Should `watch()` eventually gain an opt-in bounded queue / drop-counting mode** (closer to
   `run()`'s guarantees) **without becoming `run()`,** or should a caller who needs those guarantees
   simply be told to use `run()` instead? Not resolved here — `watch()`'s current thin-wrapper shape
   is stated as deliberate (`watch.py:5-6`) and this ADR does not recommend changing it, but the
   question of whether that remains true under broader production use requires an explicit
   API-scope decision, not an engineering default to slip in quietly.
4. **Reconciling `WatchEvent` with the `Incident`/`DeliveryAttempt` chain ADR-003 forward-referenced.**
   `Incident` exists (`contracts/__init__.py:1228`) and is what `run()` yields; `DeliveryAttempt` does
   not exist as a type anywhere in `percept/contracts/__init__.py` today. Whether `watch()` should
   ever emit `Incident`/gain a `DeliveryAttempt` ledger, or whether that chain is scoped to `run()`
   only and `watch()` stays the deliberately thinner contract, is unresolved and left as an explicit
   API-scope decision — inventing an answer here would be exactly the kind of premature contract-freezing
   ADR-003 §"the contract is frozen" warns against.

---

Decision required on: all four open questions above, plus whether this ADR's "fixed judge,
pinned clock" normative-replay bar should become a documented, first-class testing mode (e.g. a
`percept.testing`/`replay` helper) or remain an implication callers must assemble themselves from
`Percept.create(vision=FakeVision(...), clock=...)`.
