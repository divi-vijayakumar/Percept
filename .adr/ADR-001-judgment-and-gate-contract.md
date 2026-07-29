# ADR-001 — The judgment and gate contract

- **Status:** **accepted as a decision; partially implemented.** Accepted means the runtime is bound
  by this contract as the direction of record — it does *not* mean every section below describes
  shipped code. §8 (per-key evidence sequencing) is decided but **not wired into the production
  path**, and §5's named clock seam is **harness-only** today. Each such gap is called out inline in
  its own section under an *Implementation status* note. Read those notes before citing a section as
  a description of what the runtime does.
- **Referenced by:** ADR-002 (runtime health & coverage), ADR-003 (evidence artifacts & retention),
  ADR-004 (delivery flow & replay), and ~30 citation sites across `percept/`

> **On the section numbering.** Sections §1–§10 below are cited directly from the shipped source
> (`percept/gate.py`, `percept/sequencing.py`, `percept/runtime/`, `percept/contracts/`). The
> numbering is normative and stable: if you renumber a section, the citations stop resolving. New
> material is added as a new section, never by inserting one.

## Context

Percept's entire reason for existing is to interrupt a human being — and to do it rarely, correctly,
and on time. Everything else in the system is machinery in service of that one act. That makes the
decision to notify the single most consequential thing the runtime does, and it is why the decision
path is a contract rather than an implementation detail.

Three properties have to hold simultaneously, and each of them is hostile to the other two:

1. **One firing path.** If any component other than the gate can decide to notify, the guarantees
   below are unenforceable — you cannot reason about notification frequency when there are two
   places that notify.
2. **Concurrency without reordering.** A worker pool judging the same subject concurrently can
   apply an older result after a newer one and silently corrupt the gate's per-key state.
3. **Bounded resources that fail loudly.** A 24/7 watch process must not grow without bound, and a
   drop forced by a bound must never be invisible — a silently dropped notification is
   indistinguishable, from the outside, from "nothing happened."

This ADR fixes the contract that makes all three hold at once.

## §1 — The judgment lifecycle

One pass through the judge funnel is `Verdict → GateDecision → JudgmentEvent`.

- **`Verdict`** — the backend's answer. The sole contract a vision adapter must satisfy.
- **`GateDecision`** — what the gate decided to do about that verdict, and why. *Shipped as
  `percept.gate.GateOutcome`; see the implementation-status note below.*
- **`JudgmentEvent`** — the frozen, immutable record of the pass, frozen at
  `timing.decision_emitted_ts`.

Nothing on a `JudgmentEvent` may be mutated after construction, and **no delivery timestamp is ever
added to it**. Delivery attempts are recorded in a separate ledger precisely because that ledger
records attempts made *against* an already-frozen event. Conflating the two would make the event's
meaning depend on how many times someone tried to deliver it.

Diagnostic and dead-letter payloads carry no secrets, no raw media, and no incident payload.

> **Implementation status — the shipped symbol is `GateOutcome`, not `GateDecision`.** The gate's
> own return type is `percept.gate.GateOutcome` (`percept/gate.py`, `Gate.decide`); `Gate.evaluate`
> returns `outcome.action` alone. The name in the lifecycle above was never updated. It differs on
> purpose: two *other*, unrelated types in this package already own the name `GateDecision` —
> `percept.sequencing.GateDecision` (a reducer submission result: an action plus sequencing/drop
> metadata, see §8) and `percept.harness.tier0.GateDecision` (the tier-0 edge decision,
> escalate-or-not, which happens strictly before any of this). Of the three, `GateOutcome` is the one
> exported from the top-level `percept` namespace. Read `Verdict → GateOutcome → JudgmentEvent`.

## §2 — The three-state decision model

`FIRE` / `ASK` / `SILENT`.

- `FIRE` — confident yes, fresh, on a rising edge. Delivered.
- `ASK` — low confidence or stale support: the refusal to guess, rather than a confident-wrong nag.
  Delivered.
- `SILENT` — nothing to say. **Internal-only**, never delivered.

`SILENT` is a value of the decision model, not an event anyone receives. Code switching on
`event.action` needs exactly two branches. See
[the consequence gate reference](../docs/watches/the-gate.md) for the normative reason-code
vocabulary.

## §3 — The rising edge

A **key** is `(goal_id, entity_or_episode_id)` and carries one bit — armed or spent. A confident yes
on an armed key fires and spends the edge; a confident no re-arms it; a confident yes on a spent edge
is `SILENT`.

This is what makes "tell me when the cup is empty" fire once per cup rather than once per frame,
while a genuinely new cup still fires.

## §4 — Dwell and refractory

Two opt-in guards sit on top of §3, both off by default: `require_false_first` (only a witnessed
false→true edge fires) and `refractory_s` (a fire within N seconds of the same key's last real fire
is withheld, even if a spurious confident-false blip re-armed the edge).

Both **suppress a fire the baseline gate would have emitted**, so both are observable per §9 — they
raise `gate_suppressed` rather than vanishing. `already_fired` / `dwell_pending` / `not_satisfied`
are deliberately *not* suppressions: they are the base contract, and reporting them would be
per-frame noise.

## §5 — Two time domains, never conflated

Every judgment carries timing in two strictly separated domains:

- **MONOTONIC (production, local, trusted)** — `ingest_observed_ts`, `admitted_ts`,
  `judge_started_ts`, `judge_completed_ts`, `decision_emitted_ts`. **All** latency arithmetic —
  dwell and refractory windows, funnel latency, p50/p95/p99 — reads only these five, sourced from a
  monotonic time source, never from a wall clock that NTP can adjust backwards mid-window.
- **SOURCE (optional, untrusted)** — `source_media_ts`, the source/fixture/RTSP presentation
  timestamp. Carried for correlation, never used for latency math.

**Ground truth is not a time domain.** The production runtime cannot know when an event "really"
began, or when a human could first have judged it confidently. Those are labels an annotator or a
fixture generator supplies after the fact, for evaluation only. `GroundTruth` is therefore a wholly
separate type that a live pipeline never constructs — the separation is structural, not a
convention someone has to remember.

> **Implementation status — the named clock seam is harness-only.** `percept/clock.py` (`Clock`,
> `WallClock`, `FakeClock`) is imported *only* by harness modules: `percept/harness/stream_gen.py`,
> `load_driver.py`, `metrics.py`, `soak_runner.py`. The production cognition path does **not** import
> it: `percept/agent.py`, `percept/gate.py` and `percept/runtime/worker.py` each default their
> injectable time callable to bare `time.monotonic`. This is *behaviourally* equivalent —
> `clock.WallClock` **is** `time.monotonic`, and every one of those call sites is already an
> injectable `Callable[[], float]` parameter, so tests can and do pass a `FakeClock` in — but the
> claim that production latency math flows through the module `percept/clock.py` is not true today.
> What holds is the weaker, real invariant: production latency math reads monotonic seconds through
> an injectable callable. Adopting `clock.WallClock` as the default in those three modules would be
> a no-op rename that makes the seam literal; it has not been done.

## §6 — Delivery is at-least-once, with idempotency

Delivery retries. A stable `event_id` is reused across every attempt for the same event, and the
delivery path keeps a bounded, restart-surviving ledger keyed on it, so a consumer that receives a
duplicate can recognise it as one.

At-least-once is the deliberate choice: for a system whose failure mode is *missing* the thing you
asked to be told about, a duplicate notification is a far cheaper defect than a lost one.

## §7 — The judge/backend contract

The backend contract is `Verdict` and nothing else. A backend fails in exactly two ways — it raises
(timeout, provider error), or it returns something that is not a well-formed `Verdict` (malformed,
partial, truncated JSON). Both are normalized at a single boundary; see §10 invariant 4.

## §8 — Per-key evidence sequencing

Single-flight judging gives ordering safety for free. A worker pool does not: an older job admitted
first can finish *after* a newer job admitted second, and the older result would be applied last,
clobbering the newer one's gate state.

**`evidence_seq < last_applied_sequence` is not a sufficient staleness test.** Picture seq 3 in
flight when seq 5 is admitted and applied first. Job 3 still looks "new" relative to a
`last_applied` that has not caught up to 5, and would be applied anyway.

The correct gate is **`sequence < latest_admitted_sequence`**, with the sequence reserved at job
**dispatch** time, not completion. A result is stale the instant a newer job has been *dispatched*
for that key — whether or not that newer job has itself completed. Results are then applied one at a
time per key, and a result that lost the race is dropped as a traced stale completion (§9) with the
underlying gate never touched.

> **Implementation status — HALF wired; the sequence is reserved, the rule is not enforced.** The
> two halves of `percept/sequencing.py` have diverged and this note pins the split.
>
> **Reserved (live).** `EvidenceSequencer` IS in the production path as of the implementation:
> `Percept._admit_evidence` (`percept/agent.py`) calls `sequencer.admit(key)` at judge-DISPATCH time
> so the `JudgmentEvent` this core emits carries a real, dispatch-ordered `evidence_seq` instead of a
> placeholder — which is what lets it correlate with the `GateDecision` a future worker-pool reducer
> would produce for the same key. The map is LRU-bounded like the gate's own key maps.
>
> **Not enforced.** `SerializedDecisionReducer` / `ResultPolicy` still have
> **zero production importers** — nothing under `percept/` constructs them, and the only other import of the module is
> `percept/harness/metrics.py`, which pulls in the `percept.sequencing.GateDecision` dataclass purely
> as a counter input for `dropped`/`drop_reason` accounting. So the stale-completion RULE
> (`sequence < latest_admitted_sequence` ⇒ drop) is decided but unexecuted; today nothing depends on
> it, because the live path never races (see below).
>
> `tests/test_adr001_status_honesty.py` pins this exact split, so the ADR cannot drift back to
> claiming either "fully implemented" or "entirely severed" without a test failing.
>
> The shipped runtime does not implement this rule — it **sidesteps the race** rather than resolving
> it. `percept/runtime/worker.py` runs the judge scheduler **single-flight with one latest-pending**:
> at most one judgment in flight per source, and a newly admitted frame *replaces* the queued
> candidate (latest wins, traced as `judge_superseded`) instead of queueing behind it. With never
> more than one judge job in flight for a key, two results can never complete out of order, so no
> sequence number is needed. Staleness across a goals-revision change is handled by a different
> mechanism entirely: `expected_goals_revision` travels with the admitted frame and aborts a stale
> in-flight or pending judgment (`judge_revision_dropped`).
>
> §8 is therefore the contract a real **worker pool** must satisfy on the day one is introduced — a
> decision held in advance, with the primitives pre-built and tested. It is not a description of
> current behaviour. Any client reading §8 to explain what percept 0.7.x does at run time is reading
> the wrong section; read the single-flight scheduler in `percept/runtime/worker.py`.

## §9 — Bounded, and never silent

Every unbounded structure in a 24/7 process is a leak with a long fuse. Gate keys, episode logs,
in-flight and pending scheduler slots, delivery attempts, and the dead-letter queue are **all**
bounded — by count, by TTL, or both — and a bare `Gate()` with no configuration is bounded too,
because the default is what actually ships in most deployments.

The bound is only half the rule. **A drop forced by a bound is always reported with a `reason`, never
a silent counter bump.** This covers gate-key eviction, episode-window drops, superseded and
revision-dropped pending work, terminal delivery failures, and suppressed fires (§4). A drop the
operator cannot see is worse than the resource exhaustion the bound prevents: it looks exactly like
a quiet, working system.

## §10 — Invariants

1. **One firing path.** `percept.gate.Gate` is the only component that may decide to notify.
2. **Frozen events.** A `JudgmentEvent` is immutable after construction (§1).
3. **Monotonic-only latency math.** No wall clock feeds a window or a percentile (§5).
4. **Fail closed at the judge boundary.** A judge call that raises, or returns a malformed
   `Verdict`, is normalized at one place and degrades *that tick* to `ASK`/`SILENT` — **never to
   `FIRE`, never to a crash** — and the watch continues to the next frame. A well-formed `Verdict`
   passes through byte-for-byte unchanged. This exists because the code downstream dereferences
   `.confidence` / `.satisfied` with no guard of its own: an unvalidated malformed verdict is not a
   crash, it is a *confident wrong notification*, which is the worst output this system can produce.
5. **Bounded and loud.** Every bound reports its drops with a reason (§9).
6. **`SILENT` never escapes.** No delivered event carries `action == "silent"` (§2).

## Consequences

**Accepted costs.** At-least-once delivery (§6) pushes deduplication onto consumers that care.
Reserving sequences at dispatch (§8) would mean a newer job that later fails still marks an older
in-flight result stale — that tick lost rather than applied out of order, which is the trade we
want *if and when* a worker pool ships; today the shipped single-flight scheduler pays a different
price for the same guarantee, dropping the superseded pending candidate (traced) so only the
freshest frame is judged. Degrading to `ASK` on judge failure (§4, §10.4) means provider flakiness
surfaces to the user as questions rather than silence.

**What this buys.** The gate's per-key state is correct today because judging is single-flight, and
§8 pre-commits the rule that keeps it correct under a future worker pool. Latency numbers are
trustworthy because they cannot be perturbed by clock adjustment. A long-running watch has a
bounded footprint, and every bound it hits is visible. And the worst-case output of a broken
backend is a question, not a confident lie.
