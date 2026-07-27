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
