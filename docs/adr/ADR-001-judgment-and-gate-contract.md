# ADR-001 — The judgment and gate contract

- **Status:** accepted
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
- **`GateDecision`** — what the gate decided to do about that verdict, and why.
- **`JudgmentEvent`** — the frozen, immutable record of the pass, frozen at
  `timing.decision_emitted_ts`.

Nothing on a `JudgmentEvent` may be mutated after construction, and **no delivery timestamp is ever
added to it**. Delivery attempts are recorded in a separate ledger precisely because that ledger
records attempts made *against* an already-frozen event. Conflating the two would make the event's
meaning depend on how many times someone tried to deliver it.

Diagnostic and dead-letter payloads carry no secrets, no raw media, and no incident payload.

## §2 — The three-state decision model

`FIRE` / `ASK` / `SILENT`.

- `FIRE` — confident yes, fresh, on a rising edge. Delivered.
- `ASK` — low confidence or stale support: the refusal to guess, rather than a confident-wrong nag.
  Delivered.
- `SILENT` — nothing to say. **Internal-only**, never delivered.

`SILENT` is a value of the decision model, not an event anyone receives. Code switching on
`event.action` needs exactly two branches. See
[the consequence gate reference](../reference/consequence-gate.md) for the normative reason-code
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
  dwell and refractory windows, funnel latency, p50/p95/p99 — reads only these five, sourced from the
  injectable monotonic clock seam (`percept/clock.py`), never from a wall clock that NTP can adjust
  backwards mid-window.
- **SOURCE (optional, untrusted)** — `source_media_ts`, the source/fixture/RTSP presentation
  timestamp. Carried for correlation, never used for latency math.

**Ground truth is not a time domain.** The production runtime cannot know when an event "really"
began, or when a human could first have judged it confidently. Those are labels an annotator or a
fixture generator supplies after the fact, for evaluation only. `GroundTruth` is therefore a wholly
separate type that a live pipeline never constructs — the separation is structural, not a
convention someone has to remember.

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
Reserving sequences at dispatch (§8) means a newer job that later fails still marks an older
in-flight result stale — that tick is lost rather than applied out of order, which is the trade we
want. Degrading to `ASK` on judge failure (§4, §10.4) means provider flakiness surfaces to the user
as questions rather than silence.

**What this buys.** The gate's per-key state is correct under a worker pool. Latency numbers are
trustworthy because they cannot be perturbed by clock adjustment. A long-running watch has a
bounded footprint, and every bound it hits is visible. And the worst-case output of a broken
backend is a question, not a confident lie.
