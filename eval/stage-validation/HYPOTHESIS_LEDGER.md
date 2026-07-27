# Overnight debug loop — hypothesis ledger

**Goal:** get the OSS launch ready. Fix recall, precision, latency, and above all cognition.
**Budget:** $50 Gemini. A reflex N=100 costs **$1.43** (measured), a 10-case subset ~$0.14.
**Method:** Opus formulates a hypothesis from data + code; a fast cheap model tests it and tries to
kill it; the result is recorded here whether it survives or dies. One issue at a time.

**Rule that governs this loop:** a hypothesis is only closed by evidence, never by plausibility.
Three diagnoses were already lost today to reasoning ahead of a measurement — they are recorded in
the graveyard below so they are not re-proposed.

---

## Ground truth (established, do not re-litigate)

Live N=100, both modes, Gemini 3.1 Flash-Lite direct, ordered-image transport, 0 errors:

| mode | recall | precision | incidents | ASK | FIRE | in tok/call | out tok/call | cost |
|---|---|---|---|---|---|---|---|---|
| deliberate | **0.684** | 0.486 | 2,338 | 8.3% | 25.7% | 5,575 | 77.5 | $9.69 |
| reflex | **0.137** | 0.392 | 186 | 85.5% | 2.4% | **878** | 5.7 | $1.43 |

- Provider latency p50 **1.365 s** / p95 **1.473 s** — the 1.5 s budget is MET.
- Per-judgment pipeline (ingest→Incident) p50 1.667 s.
- `action→Incident` figures are NOT usable: computed against whichever Incident the matcher paired,
  and ASK timestamps were never stamped on the media clock.
- Corpus: 100 cases, 329 GT occurrences. RepCount alone is 203 of them (62%).
- Only 3 of 8 behavior routes are drivable: `worker_can_drive` accepts
  `{FRAME_WATCH, FRAME_TRANSITION, FRAME_PROCESS}`. `count`, `rhythm`, `event`, `scheduler` are not.
- All 100 goals compile to one byte-identical artifact (`Goal.state = {'type':'watch','count':0}`).
- The live run used a **scripted conductor** — it authored nothing. Every goal was hand-armed with one
  fixed plan shape.

---

## OPEN — work these in order

### H6 (NOW HIGHEST) — cutting reflex to 3 frames buys the headroom the deadline needs
**Follows from H5 being CONFIRMED and the deadline being unraisable.**
Arithmetic (measured legs): non-provider overhead is 0.241 s. A 1.5 s pool deadline yields a
worst-case ingest→Incident of 1.741 s (OK); **1.8 s yields 2.041 s and BREAKS the 2.0 s product
gate**. Measured ingest→Incident p95 was already 2.001 s WITH the 1.5 s deadline. So the deadline
cannot be raised — the provider must get faster.
Probe evidence: 5 frames ≈ 1.19 s median; 3 frames ≈ 0.73–0.91 s. That is ~0.3–0.45 s of headroom,
taking p95 from 1.473 s to roughly 1.0–1.1 s and ending the expiries.
**The real question is a trade, and it is empirical:** fewer frames means fewer ASKs (recall UP)
but less temporal context per judgment (recall DOWN). Nobody has measured which dominates. The
reflex commit deliberately held frames at 5 on the reasoning that halving temporal resolution would
surface as lost recall — that reasoning was sound but untested, and it was made before we knew the
deadline was expiring 84% of judgments.
**Test (live, ~$1.43):** reflex N=100 with 3 frames, against the existing 5-frame reflex baseline
(recall 0.137 / ASK 85.5% / FIRE 2.4%). Report recall, precision, ASK rate, deadline expiries and
provider p95.

### ~~H5~~ CONFIRMED — the reflex deadline has no headroom
**Status: CONFIRMED. Fix is H6, not a deadline change.**
Measured: deliberate's implied real provider calls = 6,419 = **100.0%** of judged. Reflex = **987 =
15.7%**. So 5,282 judged units never received a provider answer — and reflex logged **1,010
`judge_deadline_expired`** against deliberate's **zero**. Unanswered units fail closed to UNKNOWN →
confidence 0.0 → **ASK**. 5,282 unanswered ≈ 5,357 ASKs observed.
**Why:** the reflex pool deadline is 1.5 s; measured provider latency is p95 **1.473 s**. That is
**27 ms of headroom**. Queuing and scheduling under concurrency consume it, so the pool times out
before the provider replies. Deliberate's 30 s deadline expires never.
**This is the cause of the ASK storm, the 2.4% fire rate, and the 0.137 recall.** Not the terse
format, not the frame count, not cognition.
**Next step:** decide the fix rather than tune it. Candidates, in order of honesty:
  (a) raise the reflex pool deadline so p95 provider + scheduling fits with real margin, and re-measure
      whether the product's 2 s action→Incident gate still holds;
  (b) reduce provider latency further (fewer/smaller frames) to buy headroom under the existing 1.5 s;
  (c) accept that a 1.5 s pool deadline is infeasible against a 1.473 s p95 provider and say so.
**Verified 2026-07-27 by direct measurement:** with a fake judge sleeping 1.6 s against a 1.5 s
deadline, reflex produced 0 provider calls, 5 deadline + 5 cancelled receipts and 5 ASK verdicts;
deliberate (30 s) produced 5 provider calls and 5 FIRE. The pool behaves exactly as designed.
Two accounting facts settled: `funnel.judged` counts **dispatch attempts**, not completed provider
calls (`agent.py:4159`, before `pool.judge()`), so the 878-tokens-per-"judged" figure was a
meaningless denominator; and the 1,010-vs-5,282 gap is granularity — `judge_deadline_expired` is a
BATCH-level health event while deadline receipts are per-unit. **No silent drop path exists.**
Raising the deadline is arithmetically ruled out (see H6).

### ~~H1~~ REFUTED — reflex sends ~1 frame where deliberate sends 5
**Evidence:** input tokens per call are 878 (reflex) vs 5,575 (deliberate). Deliberate's figure matches
five 768 px frames (~1,100 tok each) + prompt. Reflex's is roughly one frame's worth. Both arms are
supposed to send the identical temporal strip; the reflex commit was meant to touch only `max_tokens`
and the reply format.
**Why it matters:** a temporal condition judged on one frame has no change signal. This alone could
explain the recall collapse, and it would make the whole reflex-vs-deliberate comparison mean
something different.
**Where to look:** two separate strip-assembly sites, `percept/backends/gemini.py:391` and `:573`.
Determine which path each arm takes and whether `terse`/`reflex`/`structured` gates frame count.
**REFUTED 2026-07-27.** Measured directly by capturing assembled parts with a stubbed client: reflex
sends **5** images, deliberate **5**, reflex+event_chain **5**. Frame assembly at `gemini.py:391-400`
does not read `reflex` or `terse`. The refutation was productive — it forced the token gap to be a
denominator artifact, which produced H5 above.
*(Note: the tester's own explanation — that `max_tokens` reservation inflates input tokens — is wrong;
an output cap cannot change input token count. The real explanation is H5.)*

### H2 — a majority of judgments bypass `GeminiVision.judge`
**Evidence:** a replay spy wrapping `GeminiVision.judge` captured 56 replies against 224 judged units.
~75% of traffic takes another dispatch path.
**Why it matters:** every per-call claim so far rests on a quarter of the traffic.
**Kill test (offline, free):** instrument at the `JudgePool` boundary instead and count; identify the
other path by name.

### H3 — the run is not reproducible
**Evidence:** same case, same build, same config → 16 incidents vs 50 on replay, at entirely
different timestamps.
**Why it matters:** until the variance is characterised, no single-run number is a measurement.
**Kill test (~$0.43):** run the SAME 10 cases 3× in reflex and report per-case incident-count
variance. Separate provider non-determinism from runtime non-determinism by also running 3× against
a scripted judge (free) — if variance persists with a fixed judge, it is the runtime.

### H4 — cognition: 71% of GT events are scored against a primitive that cannot represent them
**Evidence:** RepCount = 203 of 329 GT occurrences, recall 0.099, running as a boolean presence Watch.
`RhythmTracker` measures MAE 0.00 / 20-of-20 exact offline but is unreachable — `EXTERNAL_RHYTHM` is
not in `_FRAME_ROUTES`.
**Fix shape:** (a) make sense routes drivable, (b) something must select `type="rhythm"`. (b) is the
conductor's job and is next-release architecture; (a) is mechanical.
**Do NOT start H4 until H1–H3 are closed** — otherwise an improvement cannot be attributed.

---

## GRAVEYARD — refuted, do not re-propose

| # | Hypothesis | Killed by |
|---|---|---|
| G1 | The 9.5 s / 62 s latency is a matcher artifact pairing the wrong Incident | First-fire-after-onset is itself 47 s late on proactiveqa |
| G2 | The 320-token cap truncates event-chain JSON, losing `phase` → normalizer abstains → ASK | Mean output is 5.7 tok (reflex) / 77.5 (deliberate), far below the cap |
| G3 | Terse replies omit `confidence`, so the 0.7 gate can never clear | A replay captured 56 replies; confidence present on **every** one, values 0.8–1.0 |
| G4 | Replay pace lag / frame drops cause the pre-ingest delay | pace_lag 0.00 s median / 0.35 s max, zero decoder drops, 6 fps throughout |
| G5 | Episode logic holds Incidents pending close | `decision → Incident observed` p50 0.000 s, max 0.003 s |
| G6 | Reflex sends ~1 frame vs deliberate's 5 | Measured: 5 images in both modes; `gemini.py:391-400` does not gate on reflex/terse |
| G7 | `max_tokens` reservation inflates input tokens | An output cap cannot change input token count — arithmetic error |

---

## Budget ledger

| date | what | cost | running total |
|---|---|---|---|
| 2026-07-27 | live N=100 × 2 modes (baseline, already spent) | $11.12 | — |
| | | | **$50.00 remaining for this loop** |

Record every provider spend here. Stop and report if the remaining budget falls below $5.

---

## Loop protocol

1. Read this ledger. Pick the highest-priority OPEN hypothesis.
2. **Prefer a free offline kill test.** Only spend when a live run is the only way to discriminate.
3. Dispatch a fast model to execute the test and try to KILL the hypothesis. Its job is refutation,
   not confirmation.
4. Record the outcome here — CONFIRMED, REFUTED (→ graveyard), or INCONCLUSIVE with what would settle it.
5. If CONFIRMED and the fix is contained: fix it, run the full suite (baseline **4,866 passed**), commit
   locally on `codex/launch-bounded-judges`. **Never push to remote.**
6. Re-measure the affected metric. Update "Ground truth" only with a measured number.
7. Repeat.

**Launch-ready means:** both release blockers closed (done), reflex not quality-destructive, recall and
precision measured and stable across repeated runs, latency attributable, and every claim in the launch
matrix backed by an exact-wheel measurement. Nothing is pushed or published without explicit approval.
