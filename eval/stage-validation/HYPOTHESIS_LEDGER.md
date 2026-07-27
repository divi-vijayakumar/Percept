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

## ✅ RESOLVED — the reflex regression is CLOSED (`23de596`)

The reflex bound landed as a 3-frame evidence cap on the existing route policy.
**Recall 0.137 → 0.626 · precision 0.392 → 0.417 · ASK 85.5% → 31.4%.**
Suite 4,875 passed / 4 skipped / 3 xfailed / 463 subtests (baseline 4,866/441 — delta is exactly the
9 new tests). Deliberate byte-for-byte unchanged, verified on the whole request dict. `git
diff --check` clean. **The candidate `codex/launch-bounded-judges` is mergeable — merge/push/tag/
publish needs explicit human approval and must not happen autonomously.**

Total diagnostic spend: **$3.71 of $50**. Every decisive step except the frame curve was offline.

---

## OPEN — work these in order

### H6 — **CONFIRMED, FULL CORPUS N=100.** Cutting reflex 5→3 frames raises recall 4.6×

| run | recall | precision | ASK | FIRE | incidents |
|---|---|---|---|---|---|
| reflex 5-frame (baseline) | 0.137 | 0.392 | 85.5% | 2.4% | 186 |
| **reflex 3-frame** | **0.626** | **0.417** | 31.4% | 15.9% | 1,698 |
| deliberate 5-frame | 0.684 | 0.486 | 8.3% | 25.7% | 2,338 |

3-frame reflex reaches **92% of deliberate's recall** inside the 1.5 s deadline at a fraction of the
token cost. Precision improved too. **Every dataset improved:**
charades 0.700→1.000 · perception_test 0.050→0.800 · repcount 0.099→**0.650** ·
proactiveqa 0.300→0.650 · omnipro 0.061→0.379.

**This partially refutes H4.** I predicted RepCount would stay poor because a boolean presence Watch
structurally cannot represent 10 reps. It went 0.099 → 0.650. The structural argument still holds in
principle — a boolean watch has one rising edge per continuous true-period — but it was clearly NOT
the dominant limiter; deadline expiry was. H4's priority drops accordingly and its framing needs
rewriting before any conductor/rhythm work is justified on these grounds.

**Open caveat:** 1,698 incidents for 329 GT occurrences is still 5.2× over-delivery (deliberate is
7.1×). Over-firing is unfixed.

### H7 — **RESOLVED. 3 frames is a genuine interior optimum.**
Frame curve, identical 40 cases, live Gemini direct:

| frames | recall | precision | ASK | FIRE | incidents |
|---|---|---|---|---|---|
| 2 | 0.449 | **0.354** | 30.4% | 8.9% | 285 |
| **3** | **0.653** | 0.291 | 25.3% | 14.2% | 547 |
| 4 | 0.314 | 0.312 | 69.0% | 6.2% | 192 |
| 5 (shipped) | 0.093 | 0.175 | 83.8% | 2.3% | 63 |

Fewer is NOT monotonically better: 2 frames has a HIGHER ASK rate than 3 and loses a fifth of the
recall — the temporal-signal trade finally appears there. So 3 is bracketed on both sides and is a
defensible constant rather than the one value that happened to be tried first.
**Note honestly:** precision peaks at 2 (0.354), not 3 (0.291). This is a RECALL-optimal choice, not a
dominant one. Anyone optimising for precision should revisit it.
**Status:** landing now as a reflex-only frame bound in the shared latency policy.

### ~~H2~~ REFUTED — a majority of judgments bypass `GeminiVision.judge`
*(Note: first launch died instantly on an argparse error — the flag is `--limit`, not `--n`. No budget
spent, but the jobs sat dead for a full 25-minute interval. Verify a background job is ALIVE and
producing output, not merely launched.)*
Only 5 and 3 have ever been measured. Landing 3 as a shipped default without checking its neighbours
would be an unvalidated constant. Stratified n=40 runs at **2** and **4** frames are running to
bracket it. If 2 ≈ 3, prefer fewer (cheaper, faster). If 4 > 3, the optimum is between 3 and 5.

### ~~H6~~ (original statement) — cutting reflex to 3 frames buys the headroom the deadline needs
**Like-for-like, 20 proactiveqa cases in BOTH runs:**

| run | recall | precision | ASK | FIRE | SILENT | incidents |
|---|---|---|---|---|---|---|
| 5-frame baseline | 0.300 | 0.299 | 86.8% | 1.4% | 11.8% | 87 |
| **3-frame (H6)** | **0.650** | **0.328** | 36.5% | 9.5% | 54.1% | 795 |

**Recall more than doubled and precision did not regress.** The feared trade — less temporal context
costing recall — did NOT materialise: the deadline expiries were costing far more than frames 4 and 5
were contributing. For scale, 3-frame reflex (0.650) approaches deliberate's full-corpus 0.684 at a
fraction of the token cost.
**Honest caveat:** incidents went 87 → 795 (9x). 795 fires for 20 GT occurrences is still ~40x
over-delivery, and precision only held rather than improved. The over-firing problem is untouched by
this fix and remains open.
**Status:** awaiting the remaining 72 cases before this is a corpus result rather than a slice.

### ~~H6~~ (original statement) — cutting reflex to 3 frames buys the headroom the deadline needs
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

**REFUTED 2026-07-27.** No hidden dispatch path exists. A code trace found only intentional
non-adapter outcomes — direct-judge confident hit (`agent.py:4254`), cache hit (`agent.py:5983`),
cache-audit degradation (`agent.py:6002`), stale-verdict discard (`agent.py:4269`).
**Both explanations the tester offered were checked against the run data and do NOT apply:** the run
was SINGLE-route (only `timestamped-images-primary`, 220 receipts — so no "another route answered
first"), and there were ZERO cache/direct_judge counters (so no cache hits). The 56-of-224 gap is
therefore deadline attrition — i.e. H5 — and H2 is a duplicate of it, not a separate defect.
*(Caveat: the fast-judge/slow-judge counter experiment was specified but not actually run; this is
refuted by code trace plus run data, not by that measurement.)*

### H3 — the run is not reproducible — **STILL OPEN (a first attempt was DEGENERATE)**
**Evidence:** same case, same build, same config → 16 incidents vs 50 on replay, at entirely
different timestamps.
**Why it matters:** until the variance is characterised, no single-run number is a measurement.

**Attempt 1 (2026-07-27) — INCONCLUSIVE, not a refutation.** 3 charades cases × 5 repeats with a
seeded fake judge returned identical results — but **every run produced ZERO incidents**
(`[0,0,0,0,0]`, gate actions 100% ASK). The tester reported "REFUTED / runtime is deterministic",
and its own output conceded timestamps were "empty, but would be deterministic if non-empty".
**A run that fires nothing is trivially reproducible.** The fire path — where the 16-vs-50 variance
actually appeared — was never exercised. This shows only that a no-fire configuration is stable.

**What would actually settle it:** a fake judge that FIRES — returns `satisfied=True` with confidence
above the 0.7 gate threshold on a known subset of frames — run 5× on cases that produce a non-zero
incident count, comparing incident COUNT and TIMESTAMPS across repeats. If those are identical, the
runtime is deterministic and the variance is provider-side (expected, not a defect). Also worth
checking whether the fake path involves any wall-clock deadline, since a real timing dependency makes
bit-identical output impossible by construction.
**Cost: free (offline).**

**Attempt 2 (2026-07-27) — ALSO INCOMPLETE.** The tester reported a *provisional* verdict with
if/then branches while its harness run was still going; an agent's background work dies when the
agent stops, so no matrix result exists. Two things it DID establish and that are worth keeping:
- **The previous zero-incident result is explained:** the default `FakeVision` script returns
  `Verdict(satisfied=False, confidence=0.2)` for unknown frames, and the gate needs `satisfied=True`
  AND `confidence > 0.7` to FIRE — so it could only ever ASK. Any future determinism test must assert
  a non-zero incident count before comparing anything.
- **`FakeJudge` itself is deterministic** — identical fire sequence across seeded runs, using
  `random.Random(seed)` with no global state.
- Its named suspect, `judge_pool.py:636`, is **the opposite of a hazard**: the code deliberately
  resolves `asyncio.wait`'s non-deterministic result SET "in declared route priority order", i.e. an
  existing determinism guard.

**H3 is DEPRIORITISED.** Two attempts, no result, and it is not on the launch critical path — the
16-vs-50 observation is adequately explained by provider non-determinism, which is expected. Revisit
only if a reproducibility claim is needed for a published benchmark.

### R1 — residuals disclosed in `23de596` (highest remaining)
Both were recorded in the commit rather than buried, and both are contained:
**(a) un-timestamped `event_chain` skew.** On that path a reflex request carries only "the evidence
spans X seconds", where X still describes the FULL captured window while the frames now cover its
tail — so reported occurrence times can be attributed early. Left deliberately, because the prior
behaviour there was deadline expiry to UNKNOWN (zero occurrences), and slightly-shifted beats none.
**Fix shape:** a `structured`/`event_chain` carve-out in `_judge_max_frames`, mirroring how
`_judge_max_tokens` already treats structured replies. Contained, offline-testable.
**(b) claim-mode evidence trim — DECIDED 2026-07-27: document, do not exempt.**
Reachability: `claim_mode` (set when `identity_policy == "claim_slot"`, `agent.py:2769`) and
`event_chain` (`rearm == "each_event"`, `agent.py:2771`) are INDEPENDENT conditions, so they can
co-occur, and `_judge_max_frames(*, reflex)` bounds regardless — it takes no `claim_mode` argument.
So the exposure is real, not theoretical.

**But it is narrow and largely self-limiting.** The bound drops the OLDEST 2 of 5 frames — roughly
the first 0.8 s of a 2.0 s window — and keeps the newest. A claim correction is a STATE CHANGE: if
it occurred, the retained newer frames show it. To be missed it would have to be visible ONLY in the
dropped prefix and NOT afterwards, i.e. transient — which is not a durable claim correction.

**Decision:** exempting claim mode would return those watches to the 5-frame deadline-expiry regime
(recall 0.137, ASK 85.5%) to protect against a case that is both narrow and self-limiting. Not worth
it. The reasoning is already written into `_judge_max_frames`'s docstring, which explains why this
carve-out is deliberately narrower than `_terse_reply`/`_judge_max_tokens` — those trim the REPLY
(and the claim ledger reads corrections off the reply) whereas this trims only the oldest evidence,
leaving the mutation channel untouched.
**Revisit if:** a claim watch is ever run in reflex mode on material where corrections are transient.

### R2 — **DIAGNOSED. Re-scoped: the judge's per-window boolean is near-chance on 3 of 5 datasets.**

**Mechanism (source-verified).** The rising-edge gate contributes no suppression on the path that
actually fires. `_fire_clip_occurrences` (`agent.py:1901`) routes any verdict with a non-empty
`events` list to `behavior.py:1093`, which builds `should_rearm=True` for EVERY judge-reported event
unconditionally; `behavior.py:1197-1200` then calls `Gate.rearm`, and `gate.py:474-480` sets
`_armed=True` **and pops `_last_fire`**. So `decide` can never return `already_fired` and the
refractory check can never find a last-fire. **This is documented as intentional** — the docstring
says an explicit rearm is a fresh episode boundary. The open question is whether *every* reported
event deserves that treatment, not whether `rearm` is wrong.
*(Caveat: the tester reported `gate.fire == event_chain_opened` with exact per-case equality on
300/300 runs. Re-counting independently I get 121/300 equal and different totals (3,085 fires vs
4,769 chain counters). The SOURCE chain verifies exactly; the per-case statistic does not reproduce
under my counting and should be re-derived before being quoted.)*

**Refractory is not the lever — measured, not argued.** Simulated offline across the real timestamps:
precision 0.435 → 0.435 → 0.421 → 0.401 → 0.376 as R goes 0 → 1.5 → 3 → 5 → 15 s, while recall falls
0.626 → 0.565 → 0.483 → 0.380 → 0.243. **Recall is paid 1:1 and precision buys nothing.**

**THE FINDING THAT MATTERS — chance-corrected precision.** Against fires placed uniformly over media
time (expected = GT coverage fraction):

| dataset | precision | chance | lift |
|---|---|---|---|
| charades | 0.915 | 0.852 | **1.07** |
| repcount | 0.848 | 0.811 | **1.04** |
| perception_test | 0.204 | 0.199 | **1.02** |
| proactiveqa | 0.328 | 0.183 | 1.79 |
| omnipro | 0.115 | 0.069 | 1.67 |
| POOLED | 0.435 | 0.345 | **1.26** |

On charades, repcount and perception_test the fires are **statistically indistinguishable from random
timestamps**. The headline precision 0.417 is largely a COVERAGE artifact — charades ground truth
covers 85.1% of its media, repcount 80.7%. And these are not boundary misses: proactiveqa's median
false fire is **106.6 s** from the nearest GT span.

**One explanation for both over- and under-firing.** `incidents ≈ judgments × p(fire|judgment)`, and
judgments-per-media-second is a RUNTIME CONSTANT (0.68–0.79 across all five datasets,
`corr(media_end_s, judged) = +0.993`). With the gate suppressing nothing, incidents track judgments,
not content: `corr(GT_occurrences, incidents) = **−0.07**`. Direction is set purely by
judgments-per-GT-occurrence — proactiveqa gets **246.5** chances per occurrence (39.8× over),
repcount gets **2.1** (1.03×, at parity, which also caps its recall at 0.650). The apparent
"under-firing" at 5 frames was H5 deadline attrition, already closed — at 3 frames everything
over-fires except repcount.

**Proposed fix — only one part is free, and it is a volume fix, not a quality fix.**
Making `should_rearm` conditional and letting `refractory_s` survive an episode re-arm gives, measured:
full burst-collapse 1,698 → 710 incidents (−58%) but precision 0.435 → **0.414** and recall 0.626 →
**0.465**. The only free operating point is a ~1.0 s merge: −17% incidents, precision unchanged,
recall 0.626 → 0.623. **17% of incidents are pure zero-cost duplicates; everything beyond ~1.0 s buys
nothing and costs recall.**

**R2 RE-SCOPED.** No suppression policy reaches the real gap. A per-window decision with lift 1.26
caps ANY function of those decisions at ~0.44 precision. The remaining levers change what is judged
or what is asked: stage-1 admission is currently precision-neutral-to-negative (0.165 → **0.139**
across admission), and condition wording spans a 0.00–0.75 fire rate across 20 proactiveqa cases that
are otherwise identical (same 360.2 s length, same judgment volume, exactly 1 GT occurrence each).
**Next test should be a wording/admission A/B on a fixed slice — not another gate change.**

**Data caveats.** `incident_ts` is capped at 200/case; 3 cases hit it (deliberate proactiveqa lost
537 — its numbers are under-counted). omnipro's 66 GT occurrences are zero-width instants widened by
a ±1.0 s tolerance, so its 0.115 measures timing tolerance, not detection, and should not be pooled
with interval datasets.

### ~~R2 (original framing)~~ — over-delivery, untouched by anything so far
1,698 incidents against 329 GT occurrences = **5.2×** (deliberate is 7.1×). Precision 0.417 means
~58% of fires land outside a ground-truth window. No fix tonight addressed this; it is the largest
remaining quality gap and it is NOT a latency problem.

### ~~H4~~ — premise WEAKENED, needs rewriting before any conductor work
**Original claim:** 71% of GT events are scored against a primitive that cannot represent them, so
RepCount is structurally capped as a boolean presence Watch.
**What actually happened:** RepCount went **0.099 → 0.650 on a frame-count change alone**. The
structural argument still holds in principle — a boolean watch has one rising edge per continuous
true-period — but it was demonstrably NOT the dominant limiter; deadline expiry was.
**Do not justify conductor/rhythm work on these grounds without re-measuring.** The honest open
question is now: with the deadline fixed, how much headroom remains that a rhythm primitive would
actually recover? That is measurable and unmeasured.
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
| G8 | Over-delivery is caused by interval fragmentation at `interest.py:1165-1180` | Wrong layer — that mints INTEREST intervals; incidents are minted by `behavior.fire_clip_events`. My strongest candidate, refuted |
| G9 | A non-zero `gate.refractory_s` would fix precision | Measured: precision 0.435→0.376 while recall 0.626→0.243. Recall paid 1:1, precision buys nothing |
| G10 | Early attribution (R1a skew) inflates false fires | Precision peaks exactly at δ=0 and falls symmetrically either side |
| G11 | Raising the confidence threshold would help | Outside-GT fires are MORE confident than inside-GT on deliberate (0.981 vs 0.968); sweeping 0.7→1.0 moves precision 0.433→0.440 reflex, 0.405→0.378 deliberate |

---

## Budget ledger

| date | what | cost | running total |
|---|---|---|---|
| 2026-07-27 | live N=100 × 2 modes (baseline, already spent) | $11.12 | — |
| 2026-07-27 | H6 reflex N=100 @ 3 frames | ~$1.43 | $48.57 left |
| 2026-07-27 | H7 bracket 2 + 4 frames, n=40 each | ~$1.14 | **~$47.43 left** |
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
