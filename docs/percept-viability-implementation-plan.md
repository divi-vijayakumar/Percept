# Percept — PRE-MVP Viability Implementation Plan

> **Phase authority:** `CURRENT_PHASE.md` (law) → `Percept-Viability-Probe-Spec.md` v0.2 → Accepted ADRs (`ADR-001`) → `Percept-OSS-v0.1-PRD.md` (**post-GO only**) → Roadmap/Strategy (rationale, never build authority).
> **Confirmed reads before writing:** `CURRENT_PHASE.md` and `ADR-001-v0.1-public-contracts.md`.
> **Current phase:** PRE-MVP VIABILITY. No OSS-product work begins until the spike returns GO / CONDITIONAL-GO and a typed `launch_profile` is locked.

This plan drives a **two-tier dispatch board** (`.dispatch-board/board.json`):

- **ACTIVE (pullable now)** — derived *only* from the Viability Spec + M0 + ADR-001. The maximum buildable without your numbers.
- **GATED BACKLOG (not pullable)** — derived from the v0.1 PRD + Roadmap. Every card carries an explicit unlock `gate` and starts `blocked: true`.

---

## 0. Guardrails (from CURRENT_PHASE.md)

**✅ Allowed now:** the four M0 fixes; the monotonic production-timestamp contract + separate evaluation ground-truth annotation; instrumentation (funnel + timestamps + queue/memory/error/delivery metrics); the probe harness (`stream_gen`/`load_driver`/`fake_judge`/`metrics_collector`/`golden_eval`/`soak_runner`); the neutral archetype golden fixtures; `evidence_seq` + serialized decision reducer (P-ALLOC-1); the latency/quality/cost/soak/delivery probes **scaffolded**; `P-JUDGE-SAFE-1`; `P-WRAP-1` **only if** the realtime wrapper ships.

**⛔ Not allowed yet:** full public-contract migration beyond what a probe needs · broad NL WatchPlan compiler · voice authoring / Pipecat / LiveKit / WebRTC · distributed/multi-region JudgePool · entity re-ID / cross-camera identity · specialist plugin ecosystem beyond one probe example · predictive/RL firing · hosted control plane · launch demos / polished public schemas · **any patient-monitoring / clinical scenario**.

### Flags — ACTIVE cards that graze the NOT-ALLOWED list (called out, not silently included)

| Card | Why it's still allowed now | The flag |
|---|---|---|
| **A2 · AUDIO_PUMP_DEADLOCK** | Named M0 correctness fix (Viability §1, CURRENT_PHASE). It repairs the runtime pump so latency numbers are trustworthy. | It touches the audio path; **audio/fused watches are a v0.1 non-goal**. Scope is deadlock-fix only — do **not** build audio admission/VAD or fused watches here (that is H3). |
| **A9 · Probe-scoped bounded-state** | "What a probe needs" for P-BOUND-1/P-ALLOC-1/P-DELIVERY-1 is allowed. | It is a **subset** of ADR-001 §9. The **full** §9 inventory is release-blocking `R-BOUND-1` and lives in the GATED backlog (B-BOUND, gate: GO). Bound only the structures the G1/delivery probes actually exercise. |
| **A16 · Neutral archetype fixtures** | The core gating set (state/transition/brief-action/persistence/absence/repeated-episode) is visual-only and matches v0.1 scope. | **Exclude** the supplemental audio-visual + specialist-verification archetypes (Viability §5) — they are NON-gating and pull in audio/specialist modalities that are NOT allowed now. |
| **A29 · P-WRAP-1 scaffold** | Allowed *only if* the Gemini-Live realtime wrapper is a launch candidate. | The wrapper is a **probe candidate, not a committed backend**. Kept CONDITIONAL: scaffold only; the **run** is double-gated (`bars-locked` + `wrapper-in-launch-profile`). Fail-closed safety for all adapters is covered separately by P-JUDGE-SAFE-1. |

No ACTIVE card requires patient-monitoring/clinical content, a semantic NL compiler, distributed pools, entity re-ID, or a hosted control plane. Those appear only in the GATED backlog behind GO / horizon gates.

---

## 1. Why the gates are shaped this way

- **`launch_profile=<reflex|bounded|deliberative>` is the load-bearing gate.** A GO is *not* one outcome — it is one of three, and they build different things (different hero SLA, different provider, Reflex language on/off). Gating v0.1 build cards on the specific profile value stops the board from pulling Reflex work after a YELLOW/RED result that said "ship deliberative."
- **`bars-locked` splits *scaffold* from *run*.** Building the harness and wiring each probe needs no numbers → ACTIVE now. **Running a probe for record** needs the §2 numeric bars committed-before-inspection → GATED. This is the maximum the team can do without you.
- **`horizon>=H2/H3/H4/H5`** keeps H2–H5 visible but inert — nothing is lost, nothing is prematurely pullable.

**Gate vocabulary used on the board:**
`bars-locked` · `GO` · `launch_profile=reflex` · `launch_profile=bounded` · `launch_profile=deliberative` · `wrapper-in-launch-profile` · `horizon>=H2` · `horizon>=H3` · `horizon>=H4` · `horizon>=H5`.

A card is pullable **iff** `tier: active` (equivalently `blocked: false` and `gate: null`). Nothing in the GATED backlog is pullable until its gate condition is met.

---

## 2. ACTIVE tier — ready now (Viability Spec + M0 + ADR-001)

Every ACTIVE card is scaffold/build-of-harness/fix work that needs no locked numbers. Source docs are tagged per card.

### 2.1 M0 correctness fixes — *Viability §1; CURRENT_PHASE "Allowed now"*
A mixed clock corrupts the very latency numbers the spike depends on, so these land first.

| Card | Fix | Landing seam (grounded in repo) |
|---|---|---|
| **A1** | `MIXED_CLOCK` — one monotonic clock across source→gate→judge→sink; all §3 timestamps derive from it. | `percept/clock.py` (`WallClock`/`FakeClock` seam already exists), `agent.py`, `scheduler.py` |
| **A2** | `AUDIO_PUMP_DEADLOCK` — clear the audio-pump deadlock. *(flag: deadlock-fix only; no audio watches)* | `percept/runtime/audio.py` |
| **A3** | `GATE_STATE_LEAK` — finite gate-key bound (default `max_keys=None`/`key_ttl_s=None` → set + overload behavior). | `percept/gate.py:57` |
| **A4** | `WAKE_POLICY_NONE` — remove the `wake_policy=None` silent-fail path. | `percept/harness/detectors.py`, `percept/harness/tier0.py`, `percept/runtime/worker.py` *(corrected 2026-07-17: not scheduler.py/behavior.py — those have no wake_policy code)* |

### 2.2 Timestamp contract + instrumentation — *Viability §3; ADR-001 §5*
- **A5 · Production `JudgmentTiming` (two time domains).** `source_media_ts` (optional/untrusted) + `ingest_observed_ts` (local monotonic) + `admitted_ts`/`judge_started_ts`/`judge_completed_ts`/`decision_emitted_ts`. All latency math uses the monotonic clock; event frozen at `decision_emitted_ts`. *(ADR-001 §5; delivery stamps are NOT on the event — see A21/B-SINK.)*
- **A6 · Evaluation ground-truth record (separate type).** `ground_truth{event_start_ts, decision_eligible_ts, annotation_version, annotator_or_fixture_id}` — nullable/absent on live events; carries the written eligibility rubric. *(Viability §3b, §5a; ADR-001 §5.)*
- **A7 · Funnel + metric series.** `seen→gated→judged→refused→fired` + queue depth + oldest-pending age + memory/error/delivery series, emitted as structured records. *(Viability §2/§6; CURRENT_PHASE instrumentation.)*
- **A8 · `evidence_seq` + serialized reducer + `result_policy`.** Monotonic per-key `evidence_seq`; reducer serialized per `(source_id, plan_id, episode_or_entity_key)`; gate on `latest_admitted_sequence`; `result_policy{ordering: latest_only|episode_ordered, stale_after_ms, apply_after_supersession}`. Stale completions traced-and-dropped, never mutate the gate. *(ADR-001 §8; P-ALLOC-1.)*
- **A9 · Probe-scoped bounded-state.** Finite bound + overload behavior for the structures the G1/delivery probes exercise: judge in-flight `J`, judge pending `P`, episode buffer (s+bytes), gate keys, delivery queue, delivery attempts. *(Subset of ADR-001 §9 — full inventory is B-BOUND/`R-BOUND-1`, gate: GO.)*

### 2.3 Probe harness scaffolding — *Viability §6; CURRENT_PHASE "Allowed now"*
- **A10 · `stream_gen`** — replay fixtures + synthetic, fan-out to N streams, injectable clock.
- **A11 · `load_driver`** — N×M ramp + steady-state; exposes the full `N/M/E/F` load profile knobs.
- **A12 · `fake_judge`** — deterministic verdict + configurable latency distribution + error/timeout injection (the architecture-isolation lever, §4).
- **A13 · `metrics_collector`** — all six timestamps + derived p50/p95/p99, queue depth + oldest-pending age, stale-completion drops, fan-out per admission (`F`), judgments-per-hour, `$/stream-hour`, RSS + clock skew.
- **A14 · `golden_eval`** — labeled fixtures → recall/precision + golden `JudgmentEvent` timeline; event-level matching rule (§5a).
- **A15 · `soak_runner`** — long-run driver, hourly snapshots, hour-N vs hour-1 comparison.

### 2.4 Fixtures + config manifest — *Viability §5, §5a, §2*
- **A16 · Neutral archetype golden fixtures.** Core gating set: state / transition / brief-action / persistence / absence / repeated-episode; `decision_eligible_ts` generated **by construction** (synthetic primary substrate). *(Excludes supplemental audio-visual + specialist archetypes — flag above.)*
- **A17 · §2 reference-config + numeric-bars manifest (empty, ready-to-fill).** A committed template capturing: exact SKU/CPU/GPU/RAM/OS/Python/commit/lockfile-hash/provider-model-ID/API-version/region/RTT + `N/M/E/F` + `$/stream-hour` ceiling + **naïve-VLM baseline sampling FPS** + `recall_min`/`precision_min` + queue-age/memory/latency-drift/starvation tolerances. **Filling and committing this is the human decision that flips `bars-locked`** — it is the boundary of what the team can do without you.

### 2.5 Probe scaffolds (scaffold-only — NOT runs-for-record) — *Viability §7*
Each wires the probe against the harness and asserts its pass/fail logic on fixtures/fakes. **None produces a for-record number** (that is the `bars-locked` run in the GATED backlog).

| Card | Probe | Gate | Notes |
|---|---|---|---|
| **A18** | P-BOUND-1 (G1 overload/backlog) | — | Phase A fake-judge first |
| **A19** | P-ALLOC-1 (G1 stale-completion safety) | — | builds on A8 |
| **A20** | P-STAB-1 (G1 soak) scaffold | — | uses A15 |
| **A21** | P-DELIVERY-1 (bounded delivery + idempotency) | — | `DeliveryAttempt` ledger, `event_id` dedup (ADR-001 §6) |
| **A22** | P-QUAL-1 (G2 judgment quality) | — | event-level scoring (§5a) |
| **A23** | P-JUDGE-SAFE-1 (all adapters fail-closed) | — | **release-blocking safety**; malformed/partial/timeout → ASK/SILENT/provider-failure |
| **A24** | P-ECON-1 (G2 gate economy) | — | vs naïve-VLM baseline; needs A17 filled to *run* |
| **A25** | P-LAT-BASE-1 (single-call baseline) | — | diagnostic floor only |
| **A26** | P-LAT-RUNTIME-1 (G3-A `admitted→decision_emitted`) | — | |
| **A27** | P-LAT-ACTIONABLE-1 (G3-B `decision_eligible→decision_emitted`) | — | needs A6 ground-truth |
| **A28** | P-LAT-LOAD-1 (G3 under N/M/E/F load) | — | dual gate A+B |
| **A29** | P-WRAP-1 (realtime wrapper) scaffold | *conditional* | **flag**: only if wrapper is a launch candidate; run double-gated |

---

## 3. GATED BACKLOG — blocked, not pullable

### 3.1 Probe RUNS for record — **gate: `bars-locked`** — *Viability §7–§10*
Same probes as A18–A29, but the **for-record run** against the locked reference config. Unlock only when the §2 bars (A17) are committed **before results are inspected** (CURRENT_PHASE run-validity gate).

`GB-RUN-BOUND` P-BOUND-1 · `GB-RUN-ALLOC` P-ALLOC-1 · `GB-RUN-STAB` P-STAB-1 · `GB-RUN-DELIVERY` P-DELIVERY-1 · `GB-RUN-QUAL` P-QUAL-1 · `GB-RUN-SAFE` P-JUDGE-SAFE-1 · `GB-RUN-ECON` P-ECON-1 · `GB-RUN-LAT-BASE` P-LAT-BASE-1 · `GB-RUN-LAT-RT` P-LAT-RUNTIME-1 · `GB-RUN-LAT-ACT` P-LAT-ACTIONABLE-1 · `GB-RUN-LAT-LOAD` P-LAT-LOAD-1.
`GB-RUN-WRAP` P-WRAP-1 — **gate: `bars-locked` + `wrapper-in-launch-profile`**.
`GB-MEMO` — published benchmark report + go/no-go memo that **emits the typed `launch_profile`** (Viability §9/§10) — **gate: `bars-locked`** (deps: all runs).

### 3.2 v0.1 PRD build — profile-agnostic core — **gate: `GO`** — *PRD §2/§5; ADR-001*
Preserves the R-* IDs verbatim.

| Card | Scope | R-* | Source |
|---|---|---|---|
| `GB-CMD` | structured `WatchPlan` (Choice A; deterministic field validation; `condition` → judge; `unobservable`=capability mismatch only) | R-CMD-4, R-CMD-5 | PRD §2, ADR-001 §7 |
| `GB-LIFE` | plan lifecycle: `plan_id`/`revision`/`status`; DRAFT→ACTIVE→PAUSED…; stale-revision pinning; snapshot/restore (no secrets/raw media) | R-LIFE-1..4 | PRD §2 |
| `GB-FEED` | file · webcam · RTSP (static camera only); two time domains | R-FEED-1..3 | PRD §2, ADR-001 §5 |
| `GB-VID` | visual-change + ROI + fixed-cadence admission; VID region; recall-preservation metric | R-VID-1..4 | PRD §2 |
| `GB-JUDGE` | one generalist VLM (frozen `Vision` seam) + deterministic fake; bounded pool + `result_policy`; all adapters fail-closed | R-JUDGE-1, R-JUDGE-3, R-JUDGE-4a, R-SEQ-1..3 | PRD §2, ADR-001 §8 |
| `GB-COG` | bounded episode buffer (pre/active/post-roll, strip/snapshot); no entity re-ID | R-COG-2 | PRD §2 |
| `GB-GATE` | one gate; rising-edge + refractory; `Verdict→GateDecision→JudgmentEvent`; SILENT internal-only; structured ASK | R-GATE-1..5 | PRD §2, ADR-001 §1–4 |
| `GB-SINK` | webhook + log sinks; bounded delivery queue + `DeliveryAttempt` ledger + at-least-once + dead-letter | R-SINK-1..3 | PRD §2, ADR-001 §6 |
| `GB-RT` | Test/offline (no-key fakes) + Online (refuse silent fake-live); funnel + production timestamps + metrics | R-RT-1..3 | PRD §2 |
| `GB-EVAL` | golden-timeline replay + eval CLI; event-level scoring | R-EVAL-1..2 | PRD §2, Viability §5a |
| `GB-BOUND` | **full** ADR-001 §9 bounded-state inventory (boundedness by construction) | R-BOUND-1 | PRD §2, ADR-001 §9 |
| `GB-DEMO` | one neutral hero scenario skeleton + §3 behavioral matrix (package/porch-arrival or equiv.) | — | PRD §3 |
| `GB-DOD` | definition-of-done conformance gates (P-BOUND-1/P-ALLOC-1/P-DELIVERY-1/P-JUDGE-SAFE-1 release-blocking) + OSS packaging (Apache-2.0, ≥5 external quickstart testers) | — | PRD §5 |
| `GB-COMMERCIAL` | one signed paying design partner on the Percept viability profile *(business milestone, non-engineering)* | — | PRD §6 |

### 3.3 v0.1 PRD build — profile-branched (mutually exclusive) — *PRD §1/§5*
Exactly one branch unlocks, per the memo's emitted profile.

| Card | Scope | Gate |
|---|---|---|
| `GB-BR-REFLEX` | Reflex hero + Reflex latency **claim**; dual-gate SLA (G3-A **and** G3-B p95 < 2s); Reflex language ON | **`GO` + `launch_profile=reflex`** |
| `GB-BR-BOUNDED` | default/deliberative build; Reflex **experimental** only; re-run P-LAT-LOAD as models improve | **`GO` + `launch_profile=bounded`** |
| `GB-BR-DELIB` | deliberative hero SLA + deliberative provider; **no Reflex language**; **re-run G2 economics with the actual multi-frame/multi-pass judge** | **`GO` + `launch_profile=deliberative`** |

### 3.4 Roadmap horizons — parked, inert — *Roadmap H2–H5*
| Card | Scope | Gate |
|---|---|---|
| `GB-H2` | NL semantic compiler (ambiguous/under-specified/unsupported/**unobservable**/executable assessment) + author lint + plan libraries + **Deliberative lane** | `horizon>=H2` |
| `GB-H3` | audio + fused watches (VAD alongside VID) + voice authoring + voice-native realtime-judge wrappers (after P-WRAP-1) + TTS + command/watch arbitration | `horizon>=H3` |
| `GB-H4` | entity re-ID + cross-episode continuity + specialist/`Enricher` seam + **Shape-2 Procedure runtime** (incl. `P-BURST-1`) | `horizon>=H4` |
| `GB-H5` | hosted control plane: fleet deploy/upgrade, secure creds, distributed pools + autoscaling, model routing, cross-runtime SLOs, RBAC/audit, managed edge | `horizon>=H5` |

---

## 4. Sequencing (ACTIVE only; fail-fast per Viability §8)
1. Instrumentation + timestamp taxonomy (**A5–A8**) + M0 fixes (**A1–A4**).
2. Harness (**A10–A15**) + fixtures (**A16**) + bars/config manifest scaffold (**A17**).
3. Probe scaffolds in Viability §8 order: **A18** (P-BOUND Phase A, cheapest killer) → **A25** (LAT-BASE) → **A22/A23** (QUAL + SAFE) → **A24** (ECON) → **A29** (WRAP, cond.) → **A26/A27/A28** (dual-gate latency) + **A19** (ALLOC) → **A21** (DELIVERY) → **A20** (STAB scaffold).
4. **Stop.** Runs-for-record and everything downstream are gated. The board hands the ASK back to you: **fill and commit A17's bars → `bars-locked` → the runs unlock → the memo emits `launch_profile` → exactly one PRD branch unlocks.**

## 5. Definition of "phase done" (from CURRENT_PHASE.md)
The spike is done when the Viability §9 go/no-go memo (`GB-MEMO`) is published with numeric evidence against the locked reference config **and emits a typed `launch_profile`**. On GO: update Current phase, repoint execution authority #4 to the v0.1 PRD, and feed the `launch_profile` in as build scope — which is precisely what the §3.2/§3.3 gates encode.
