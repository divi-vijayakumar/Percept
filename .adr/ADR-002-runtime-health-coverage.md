# ADR-002 — Runtime health & coverage control path

- **Status:** proposed
- **Depends on:** ADR-001 (the Gate/judgment contract, §1–10, unchanged by this ADR)
- **Referenced by:** PRD Revision B §6, §6.1, §14 gate 1 (delivery object) and gate 6 (concurrency model)

## Context

A watch is not a single pass/fail thing. It is a pipeline — source → tier-0 gate → judge →
delivery — and each stage can fail independently:

1. A live source drops below the FPS the plan requires, or goes dark entirely.
2. Frames arrive fine, but nothing in them lands in a relevant region of interest — the tier-0
   gate never escalates.
3. The gate escalates and evidence is admitted, but the judge is timing out, circuit-broken, or
   unreachable.
4. The judge is healthy and returning verdicts, but the sink the app registered to receive them is
   failing.

These are four different failures with four different owners: the camera operator, the plan
author, the judge/provider, and the app integrator, respectively. A single scalar — "watch health:
73%" or "watch health: degraded" — collapses all four into one number and destroys the information
an operator needs to act. A dashboard that only ever says "degraded" teaches operators to ignore
health entirely, because it never tells them where to look. **Health must be a staged vector, not
a scalar**, so each coordinate maps to one failure domain and one owner.

Two pieces of prior art already exist in this codebase and this ADR must reconcile with both:

- `percept/runtime/worker.py` already defines `RuntimeHealthEvent` (`kind: str`, `ts: float | None`,
  `data: dict | None`) and threads an `on_health` callback through the runtime loop. It fires
  synchronously, inline, through the same `_call_sink` helper used for `on_frame` / `on_ask` /
  `on_incident` delivery — i.e. today, health delivery and judgment/incident delivery share one call
  path and one failure mode (a slow or raising `on_health` handler can back-pressure the loop that
  is also trying to deliver incidents). This is flat event *telemetry* (a log of things that
  happened), not staged *state* (the current condition of the watch). It is a reasonable building
  block for the reason-code stream this ADR specifies, but it does not by itself satisfy "cannot be
  starved by judgment delivery, cannot starve incidents" — that requires the independent transport
  in §5 below, which is new work this ADR authorizes.
- `percept/observability.py` already defines a process-level `health()` payload
  (`status` / `version` / `backends` / `metrics` snapshot) for a `/health` liveness endpoint, and a
  `Funnel` (`seen → gated → judged → refused → fired`) of per-concern, per-camera counters. The
  `/health` payload is process liveness, not per-watch coverage, and is out of scope here. The
  funnel counters are a natural input to `source_coverage` / `evidence_coverage` arithmetic (a later
  a runtime-implementation concern rather than this ADR's), but they are aggregate counters, not the staged,
  per-watch, reason-coded state object this ADR defines.

This ADR specifies the target shape. It does not implement it, and it does not mandate how
`RuntimeHealthEvent` migrates — that belongs to the runtime implementation.

## Decision

### 1. `WatchHealth` — a frozen, staged vector

```python
@dataclass(frozen=True)
class WatchHealth:
    watch_id: str
    plan_revision: int
    sequence: int
    state: Literal["healthy", "degraded", "failed"]
    since_ts: float
    source_coverage: float       # 0.0-1.0: is the source delivering frames at the rate the plan needs?
    evidence_coverage: float     # 0.0-1.0: of frames the gate escalated, how many produced admitted evidence?
    judge_availability: float    # 0.0-1.0: fraction of judge calls completing without timeout/circuit-open
    reasons: tuple[str, ...]
```

`state` is the coarse rollup for a status page; the three coverage floats are the vector an
operator drills into; `reasons` names *why* — never silent, never inferred by the reader. `sequence`
makes transitions orderable and idempotent for a subscriber that reconnects. `plan_revision` ties a
health reading to the plan version it is measuring, so a health snapshot taken against a stale plan
is never silently compared against a live one.

Each of the three coverage floats is independently addressable and independently ownable:

| Field                | Answers                                            | Owner              |
|-----------------------|-----------------------------------------------------|--------------------|
| `source_coverage`     | Is the camera/feed alive and fast enough?           | Camera operator    |
| `evidence_coverage`   | Is anything worth judging actually reaching the gate?| Plan author        |
| `judge_availability`  | Is the judge answering?                              | Judge/provider     |

A failed sink is deliberately **not** a fourth coordinate here (see "Delivery health" below) —
folding it in would make `WatchHealth` responsible for a failure domain it cannot diagnose or fix.

### 2. Reason-code vocabulary

`reasons` is drawn from a closed, versioned vocabulary — never free text, so an operator's alerting
rules can pattern-match on it and a dashboard can group by it:

- `SOURCE_DISCONNECTED` — the source is unreachable (stream closed, camera offline).
- `SOURCE_STALLED` — the source is connected but no new frames are arriving.
- `SOURCE_FPS_BELOW_PLAN` — frames are arriving slower than the plan's minimum cadence.
- `SOURCE_OBSTRUCTED` — frames are arriving at rate but the content looks structurally wrong
  (e.g. a covered/blacked-out lens) — distinct from `SOURCE_STALLED` because the fix is physical,
  not a reconnect.
- `EVIDENCE_BUFFER_OVERFLOW` — the gate is escalating faster than evidence can be captured/retained;
  frames are being dropped before they reach a judge.
- `JUDGE_TIMEOUT` — judge calls are exceeding their deadline.
- `JUDGE_CIRCUIT_OPEN` — the judge circuit breaker has tripped; calls are being short-circuited.
- `PLAN_UNOBSERVABLE` — the plan asks for something the wired source/goal combination cannot in
  principle answer (e.g. a ROI outside frame bounds) — a configuration failure, not a runtime one.

Each code maps to exactly one of the three coverage coordinates (or to `state` directly for
`PLAN_UNOBSERVABLE`, which cannot be scored on a 0–1 coverage axis — it's a binary "this plan cannot
be observed as configured"). This mapping is normative and MUST be documented alongside the enum in
code, not left to be inferred from the name.

### 3. Why health is additive, not a contract change

`WatchHealth` adds **nothing** to `GateDecision.action` and **no field** to `JudgmentEvent`. This is
a hard boundary, not a style preference: ADR-001 fixes the cardinal firing contract
(`Verdict → GateDecision → JudgmentEvent → Incident → DeliveryAttempt[]`) and states no field is
renamed, no type is added, no invariant relaxed. Folding health into `JudgmentEvent` — e.g. a
`health_at_judgment` field — would relax that invariant (ADR-001 invariant 3) by making the judgment
record carry state that mutates independently of the judgment itself, and would conflate "was this
specific observation trustworthy" (a health question, answerable only in hindsight as coverage
degrades) with "what did the judge decide" (a judgment question, frozen at
`decision_emitted_ts`). A `JudgmentEvent` must remain fully interpretable in isolation; it must not
require a join against a mutable health stream to know what it means.

Health is a parallel, independently-evolving signal about the *pipeline's condition*, not a
property of any one judgment. It answers "should I trust what this watch has been reporting for
the last 30 seconds", not "was this particular fire correct."

**Delivery health is not folded in either.** A failed sink (an app's webhook endpoint down, its
callback raising) is `DeliveryAttempt` territory — a different failure domain (the app's own
infrastructure, not the observation pipeline) with a different owner (the app integrator, not the
camera operator, plan author, or judge provider) and its own retry/backoff semantics already
defined by the delivery contract. `WatchHealth` describes whether the watch *has something worth
delivering and can produce it*; `DeliveryAttempt` describes whether *what was produced got where it
was going*. Merging them would make an app's flaky webhook indistinguishable from a dead camera.

### 4. API surface

```python
current: WatchHealth = w.health()
unsubscribe = w.on_health(cb, replay_current=True)
```

- `w.health()` — a synchronous pull of the current staged vector. Always available, always O(1)
  (one slot per watch — see Transport, below).
- `w.on_health(cb, replay_current=True)` — a push subscription for *transitions* (state changes,
  coverage crossing a threshold, a reason code appearing/clearing), with `replay_current=True`
  immediately delivering the current `WatchHealth` to the new subscriber before any future
  transition.

`replay_current` is not a convenience default — it is load-bearing. A transition-only callback is
useless to a late subscriber: a dashboard that attaches to a watch five minutes after it started
degrading would otherwise sit blank and "healthy-looking" (no transition has fired *since it
subscribed*) while the watch has in fact been failed the entire time. Without replay-on-subscribe,
every consumer would be forced to also call `w.health()` on startup and hand-merge it with the
transition stream to avoid this gap — logic every caller would have to reinvent, and would
sometimes get wrong (a race between the `health()` snapshot and the first transition callback).
Building the replay into `on_health` closes that race once, centrally.

### 5. Transport

Health is delivered over an **independent bounded queue**, one per watch, with two properties:

- **Latest-state retention**: the queue holds at most one pending `WatchHealth` per watch — a new
  reading overwrites a pending, undelivered one rather than queuing behind it. This is correct
  *because* health is current state, not a log (see §6.1) — a subscriber that is behind never needs
  the intermediate readings, only the latest.
- **Transition dedup**: identical consecutive `WatchHealth` values (same `state`, same `reasons`,
  coverage unchanged within the implementation's declared epsilon) do not re-fire the callback. A
  flapping coverage float must not turn into a callback storm.

This queue MUST be independent of the judgment/incident work queue: health delivery cannot be
starved by a backlog of judgment/incident delivery (a slow app-side sink must never delay an
operator finding out the camera is dark), and, symmetrically, a burst of health transitions must
never starve or delay incident delivery. The transport MAY reuse the same sink-callback interface
worker.py already has (`on_health(event)`, matching the existing shape used for
`RuntimeHealthEvent`) but MUST NOT share the underlying work queue judgment/incident delivery uses.
Concretely: today's `_call_sink_with_policy` path threads `on_health` through the *same* async call
stack as `on_frame`/`on_ask`/`on_incident` inside `worker.py`'s single loop iteration — that
coupling is exactly what this ADR requires be removed when this is implemented; the interface can
stay familiar, the isolation cannot be optional.

## §6.1 — Why this is not the deferred event bus

The PRD defers a general event bus (per-subscriber backpressure, replay-from-offset, retained
history) as out of scope for v0.1. `WatchHealth` delivery must not be read as that bus arriving
through the side door, because the two are structurally different:

**Health is STATE. Judgments are a LOG.**

- A `WatchHealth` reading *supersedes* its predecessor. The moment a new reading exists, the old one
  has no further meaning — nothing downstream needs both the value from ten seconds ago and the
  value from now, only the latter. Retaining and replaying "the latest" is therefore correct *and*
  bounded by construction: exactly one slot per watch, full stop. There is no scenario in which
  retention grows — a watch that runs for a week retains the same one `WatchHealth` slot it retained
  after its first second.
- A `JudgmentEvent` is an immutable, append-only record with an idempotency key and a delivery
  ledger (`DeliveryAttempt[]`). Its value is exactly that it does *not* get superseded — each one is
  a distinct fact that happened at a distinct time, and a consumer that connects late is understood
  to need the ones it missed, not just the newest. Supporting that means retained history, offset
  management, and per-subscriber backpressure — each subscriber can be arbitrarily far behind,
  which is unbounded by default and needs its own deliberate design (a future ADR/PRD §9 row), not
  something this ADR can authorize as a side effect of health delivery.

Retaining one health snapshot per watch is a fixed-size cache, not a bus. Building `WatchHealth`
delivery on "keep the latest, replay it on subscribe, dedup identical transitions" does not create
any of the primitives (offsets, per-subscriber cursors, retained history) that would make it the
deferred bus in disguise — which is exactly why this ADR can proceed independently of that
deferral, and exactly why that independence needs to be stated explicitly here rather than left for
a reader to assume.

## Consequences

- Operators get a diagnosis, not just an alarm: `reasons` plus the relevant coverage coordinate
  points at one owner and one fix.
- `JudgmentEvent` and `GateDecision` stay exactly as ADR-001 defined them — no migration, no
  versioning concern for existing consumers of the judgment contract.
- `DeliveryAttempt` remains the single place delivery failures are diagnosed; `WatchHealth` never
  needs to reason about an app's webhook infrastructure.
- Implementation must retire (or clearly demote to an internal detail behind) the current
  `worker.py` coupling where `on_health` shares a call stack with `on_frame`/`on_ask`/`on_incident`
  delivery — that requires separate delivery-path implementation; it is not free.
- A future health-consuming dashboard/CLI can rely on `w.health()` being cheap and always fresh, and
  on `on_health(..., replay_current=True)` never showing a false "all quiet" for a watch that has
  been failed the whole time the subscriber was away.

## Open question for sign-off

**Should `WatchHealth.reasons` be a closed enum enforced at the type level (a `Literal[...]` /
`StrEnum` of exactly the eight codes above), or an open `tuple[str, ...]` where the eight codes are
a documented-but-unenforced convention, leaving room for a provider/backend to emit its own
provider-specific reason codes alongside the standard ones?**

**Recommended default: closed enum for v0.1**, with provider-specific detail carried in a future
`data: dict` sidecar (mirroring the existing `RuntimeHealthEvent.data` shape) rather than in
`reasons` itself. An open string vocabulary from day one means every consumer's pattern-matching
logic (alerting rules, dashboards) is written against strings that could silently drift per
provider, defeating the entire point of a reason-code vocabulary — namely, that `reasons` values are
things a consumer can safely branch on. A closed enum can still be widened in a later minor version;
loosening a closed set is a compatible change, tightening an open one is not.

---

Decision required on: the closed-vs-open `reasons` question above, plus whether this ADR's
staged-vector shape should also govern PRD §14 gate 6 (canonical concurrency model for `on_health`
delivery — sync iterator, async iterator, or background worker), since the transport section here
assumes a background-worker-style bounded queue.
