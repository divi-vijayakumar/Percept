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

## 🛑 LOOP STOPPED 2026-07-27 — everything autonomous is done

**Launch-readiness: the candidate is ready for a human merge decision.** Everything that could be
fixed without a founder decision has been. What remains needs prioritisation, not more debugging.

| item | state |
|---|---|
| Evidence-substitution blocker | closed (earlier session) |
| Deaf audio gate blocker | closed (earlier session) |
| Reflex regression | **closed** — `23de596`, recall 0.137 → 0.626 |
| R1a event_chain skew | **closed** — `ecc10e4` |
| R1b claim-mode trim | **decided** — `8ad2271`, document not exempt |
| R2 over-delivery | **diagnosed + re-scoped** — `00d63ec` |
| Claim scope | **verified honest** — pyproject says no accuracy/latency claims, `4 - Beta` |
| Suite | 4,883 passed / 4 skipped / 3 xfailed / 475 subtests |
| Merge / push / tag / publish | **BLOCKED — needs explicit human approval** |

**Total provider spend: $3.71 of $50.** The frame curve was the only thing worth paying for.

### What needs YOUR decision, not more autonomous work
1. **Merge the candidate.** It is clean and mergeable. Nothing has been pushed or tagged.
2. **R2's real lever is wording, not machinery.** A per-window judge decision with lift 1.26 over
   chance caps ANY downstream policy at ~0.44 precision. Condition wording spans a 0.00–0.75 fire
   rate across 20 otherwise-identical proactiveqa cases. The next experiment is a wording/admission
   A/B on a fixed slice — that is a product question about how goals are phrased, not a bug.
3. **The free 1.0 s incident merge** (−17% notifications, precision and recall unchanged) is
   available but NOT landed: it changes gate semantics for a volume win, which is your call.
4. **H4 / the conductor.** Its original justification is gone — RepCount improved 0.099 → 0.650 on a
   frame change alone. Whether a rhythm primitive still earns its build is now an open, measurable
   question rather than a settled one.

### The most useful artifact here is the graveyard
13 refuted hypotheses, five of them mine, each killed by a measurement rather than an argument. The
pattern that held all night: plausible mechanism, wrong layer. Check reachability with `file:line`
before building anything.

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

### ✅ CLOSED — incident merge narrowed 1.0 s → 0.7 s (the founder's rep-counting objection was right)

The 1.0 s window was picked against the wrong distribution. Its own docstring justified itself with
"observed rep spacing (~2.3s)" and claimed rep counting was off this seam via `fire_episode`. **Both
were false.** Measured rep-to-rep gaps, n=183 over the 20 repcount cases: **min 0.79 s**, p10 1.13,
p50 1.87, max 12.76. And `fire_episode` is not landed — repcount goals reach this seam *today* via
`type="watch"` → `fire_clip_events`. So a 1.0 s window sat on top of real occurrences and deleted
them one-for-one.

Judgment strides (the artifacts the window exists to collapse): 0.333 / 0.5 / 0.667 / **0.833** s.
0.7 s catches the first three and clears the 0.79 s floor with 0.09 s margin. **The 0.833 s stride is
deliberately left uncaught** — it sits *above* the rep floor, so no window removes it without also
deleting real reps. That makes ~0.75 s a hard ceiling, now written into the docstring with the full
percentile table. Volume: reflex −12.4%, deliberate −26.5% (was −14.0/−30.5 at 1.0 s); stride purity
of merged incidents *rises* to 87/81%. Two tests pin the constant under the floor.

### ✅ CLOSED — `EXTERNAL_RHYTHM` is drivable; 4 of 8 routes still are not

`RhythmDrive` resolves a rhythm goal's *measuring* eyes and folds channels into the existing
`core.perceive_rhythm`. `worker_can_drive()` admits it; `needs_watch_condition()` deliberately does
not — **a rhythm goal compiles to no `WatchCondition` because it is measured, not admitted.**

The load-bearing decision: **an Incident for a rhythm goal is a count MILESTONE, never a rep.** The
count is state, updated every confirmed rep and free to read; an Incident is minted only via the
existing `Goal.state.fire_when` predicate over `{n, t}`. No `fire_when` → zero incidents.
One-incident-per-rep is exactly the every-judgment-mints-an-incident pathology. Counting runs at
native frame rate off `RhythmTracker`'s causal reversal detection — a test asserts `judge.calls == 0`
while the count is real. Zero new state vocabulary, so plans stay hashable.

Still undriven: `count`, `event`, `remind`, `scheduler`. `count` → `EXTERNAL_BOUNDARY` is closest but
needs a boundary-detector selection policy that does not exist. **The conductor routes 7 corpus goals
to `count`, and those plans will not arm** — that is the top remaining functional gap.

### ✅ CLOSED — Haiku conductor: primitive selection 4/100 → 91/100

`PlanConductor.author(goal_text)`, per goal, once, at submission; nothing on the reflex path.
`claude-haiku-4-5`, cached system block, 320 tokens, temp 0. Reply is untrusted — through the shipped
coercion firewalls, then `_guard_authoring_namespace` refuses any path outside `Goal.state.`.

| kind | hit/total | acc |
|---|---|---|
| **overall** | **91/100** | **0.910** |
| duration | 39/40 | 0.975 |
| count | 27/28 | 0.964 |
| transition | 21/28 | 0.750 |
| state | 4/4 | 1.000 |

**All 20 rep-counting goals select `rhythm`.** Two temp-0 runs both scored 91 and differed on 1/100.
The 7 residual `transition` misses are dominated by proactiveqa items the corpus labels `transition`
wholesale but that read as ongoing watches — that is label coarseness; it was left unfitted rather
than tuned away. Pinned offline via a content-addressed frozen cache (73 decisions, zero provider
calls in CI); the optimizer guardrail is untouched and a test asserts it still denies `Goal.state.*`
to the optimizer. **Authoring and tuning stay separate acts.**

### 🚨 EVIDENCE AUDIT — four of six "wins" were never in the binary that produced their number

Adversarial audit of the six claims. **Two governing facts:**

1. **Build provenance.** Every live measurement in `results/` except five smokes carries
   `build_commit = 2d5097d` (or `083b972`). Verified by `git grep` at that commit: it contains
   **none** of `_REFLEX_MAX_FRAMES`, `_bounded_strip`, `INCIDENT_MERGE_WINDOW_S`, `PlanConductor`,
   `RhythmDrive`. **Four of six wins were never present in the binary that produced their headline.**
2. **The harness levers are not the shipped levers.** `live_run.py:141` sets frames via the WatchPlan;
   `:259` monkeypatches `GenerateContentConfig`. Its own docstring says "WITHOUT touching `percept/`".
   These reproduce the *effect* on old code; they do not exercise the shipped code.

| # | claim | verdict |
|---|---|---|
| 1 | reflex 3-frame, recall 0.137→0.626 | **DIRECTIONAL** — shipped mechanism ≠ measured mechanism |
| 2 | `block_max` default, recall 0.31→0.59 | **UNPROVEN** (recall) / DIRECTIONAL (admission) |
| 3 | 0.7 s merge, "no real occurrences deleted" | **RETRACTED** — refuted by its own commit message |
| 4 | `media_resolution=LOW` | tokens **PROVEN**; cost/ASK/expiries DIRECTIONAL |
| 5 | conductor 91/100 | **UNPROVEN as a product claim** |
| 6 | rhythm route drivable | **RETRACTED** |

#### 🔴 The conductor makes things WORSE in the only controlled test available

`goal_kind` is a **hardcoded per-dataset constant** (`build_corpus.py:117/158/291/374`); only omnipro's 8
vary. The task scored is effectively "given a sentence from dataset D, predict D's stipulated kind" — a
classifier knowing only the dataset scores 100/100. **The honest majority-class baseline is 40/100, not
4/100.**

Worse is what happens *after* selection. One real-frame test per primitive on shipped build `1c542ff`:

| authored primitive | ~goals | result |
|---|---|---|
| `watch` | 44 | works — 7 incidents, precision 1.0 |
| `transition` | 28 | arms, 37 calls, **19 decisive verdicts, 0 incidents**, recall 0.0 |
| `count` | 8 | **`WatchPlan did not arm: status=rejected`** |
| `rhythm` | 20 | **crashes** — pose runtime unresolved, then `'tuple' has no attribute 'ndim'` |

**≈56 of 100 goals route to a primitive that cannot deliver an Incident today.** And the control is
decisive: `smokeBp` runs `repcount::stu6_15::0` with the conductor OFF → **recall 0.75, 11 incidents**.
Conductor ON → crash. **Selection accuracy is not delivery.**

`FRAME_TRANSITION` producing 0 incidents from 19 decisive verdicts is a **second broken route**,
independent of rhythm, and was not previously known.

#### 🔴 `salience_stat` default is SPLIT

`config.py:198/295` say `block_max`; **`contracts/__init__.py:1287` says
`WatchCondition.salience_stat: SalienceStat = "mean"`.** A client constructing a `WatchCondition`
directly gets the legacy path. Resolve before claiming block_max is "the default".

The original 0.31→0.59 evidence (`eval/viability/LADDER_LEDGER.md`, 2026-07-17) fails the bar on four
counts: different judge/transport (`qwen3-vl-32b` via OpenRouter), single run per cell (stated in its
own header), n=20 one dataset, and a **volume confound** — judge calls 1,731→3,425 (1.98×) while recall
went 1.93×. That is more shots on goal, not a better eye. The defensible half is the admission-stage
`edge_recall` 0.698→0.995, which is deterministic in frames/statistic/cutoff.

**The running consolidated comparison cannot test it:** `live_run.py:242` writes `block_max` into the
WatchPlan `explicit` dict for **both** arms, overriding the spec's `PERCEPT_SALIENCE_STAT=mean`
baseline. The spec's own "an arm that cannot prove its configuration must abort" rule is not enforced
for #2.

#### Claim 1: the shipped bound has never had its recall measured

`agent.py:5683` with `sampling="uniform", n_frames=3, window_s=2.0` spans the **full 2.0 s**. The
shipped `_bounded_strip` keeps the newest of a 5-frame strip, spanning **~0.7–1.0 s**. Same image count
— so the latency mechanism transfers — but **different temporal evidence**, and H7's own curve shows
recall is sharply sensitive to that axis. The shipped path also carries residual R1a; the measured path
does not.

#### Claim 4: cost is not the deterministic metric it was filed as

Tokens/call ARE deterministic — HIGH 3,579/3,574, LOW 1,092/1,100 across three runs. But cost =
tokens × **calls completed**, and calls completed is the load-dependent deadline race: the 40-case pair
gives **−52.9%**, the matched 18-case replicate pair gives **−29.7%**. A 23-point spread on the same
comparison. ASK% and expiry *directions* are solid (non-overlapping replicate ranges); the specific
−68.6% / −72.8% are not.

#### Claim 3: the volume numbers rest on under-counted data

`incident_ts` is capped at 200/case and three cases hit it — deliberate proactiveqa lost 537
timestamps, so **−26.5% is computed over admittedly under-counted data**. And `5bf26d0` / `f5f1cf8`,
eleven minutes apart, **flatly contradict each other** on whether repcount reps reach this seam. The
second is right, which means the safety argument in the first was wrong when it shipped.

#### Process: the run in flight is a build-vs-build delta, not a six-lever ablation

Arm A `2d5097d` vs arm B `1c542ff` differ by **eight commits**, including `7e17378` and `1c542ff` which
are not among the six wins. Claim 2 is pinned identically in both arms and contributes nothing.

### 🚨 LONG-HORIZON RUN — 82 min, 73 goals, one core. Two launch-relevant holes.

Source: 97 corpus videos concatenated to **81.6 min**, 640×360 @15 fps, container-vs-parts drift
**0.0 s**. 73 distinct goals armed simultaneously, 315 GT occurrences. Replay wall 5,008.8 s,
**realtime ratio 1.023**, 24,484 frames, **0 decoder drops**, pace lag p50 0.04 s. The run was
genuinely real-time. Spend $1.535.

#### ✅ NOT A HOLE — zero episodes on a camera source is correct behaviour (my earlier entry was wrong)

**Retracted.** The run's agent reported that `_document_episode_fold` returns `None` whenever
`rearm == 'each_event'`, concluded "no watch goal can ever open an episode", and I recorded that.
**Both the diagnosis and my repetition of it were wrong.** Checked against the source:

1. **The corpus contains ZERO `each_event` goals.** The guard it blamed could not have fired on a
   single one of the 73 goals. `grep -c each_event corpus.json` → 0.
2. **There are TWO gates, and the second is the operative one.** After the `each_event` check both
   folds require an identity policy: `!= "content_key"` → `None` (document fold), `!= "claim_slot"` →
   `None` (claim fold).
3. **The run armed `PHYSICAL_SCENE`, whose `identity_strategy` is `track_iou_label`**
   (`source_profiles_default.py:117`) — neither `content_key` nor `claim_slot`. So both folds
   correctly returned `None` for every goal.

**This is designed behaviour, not a defect.** On a physical camera scene identity is resolved by IoU
track labels — the `EntityGraph` — not by document content-keys or claim slots. Episode folds belong
to *document* sources (`content_key`, `source_profiles_default.py:183`) and claim sources. A camera
run opening zero episodes is exactly right.

**The real lesson is about the test, not the code:** an 82-minute *camera* run cannot exercise episode
memory at all. To measure episodes we need a **document/screen source**, which also lines up with the
standing note that screen content needs a different eye. Episode machinery itself was independently
shown healthy — driven through the public `count_event` API it opened 4,088 episodes, 292
continuations, 1,168 `window_expired` drops, and the live curve flattens exactly at media ≈3,600 s
where `Settings.episodes.window_s` bites. **The window works correctly on the first test that ever
crossed it.** That result stands.

Open product question this does surface: if episode folds only ever engage for `content_key` /
`claim_slot` sources, the episode epic does not apply to camera video at all. Worth confirming that is
the intended scope before the launch claims otherwise.

#### 🔴 HOLE 2 — judge allocation is perfectly fair, and that is the problem

Every one of the 73 goals judged **exactly 60 times**. Gini 0.0, zero starved. Fairness is
*structural*: a batch judges all admitted goals sequentially (`agent.py:4090`), so nobody is skipped —
**everybody waits.** Over 4,897 s the scheduler launched only **60 batches**; **16,571 of 16,631
admissions (99.6%) were superseded.**

**onset→incident p50 79.8 s, max 94.0 s, against a 2 s product gate. The batch period IS the
latency.** This is the multi-goal scaling wall, and it is not a tuning problem.

Worse, at the shipped single-goal 1.9 s decision deadline **a 73-goal batch is cancelled wholesale**
(`percept/runtime/worker.py:1120`). The run only produced anything because the deadline was scaled
73× to 138.7 s and that scaling was declared. A deadline sized for one goal cancelling the entire
batch is a defect in its own right.

#### Entity graph: the bound HOLDS; what it buys is churn

`max_entities=64` saturates at media 15 s and stays saturated for **99.7%** of the run. 20,184 ingest
batches / 102,030 observations → 58,101 tracks minted, 58,037 evicted, 64 live at every sample.
Lifetime p50 **0.0 s**, touches per track p50 1. 35,460 fragmentations, 7,934 re-ids past the 1.0 s
coast horizon. **Zero label switches, zero resurrections of an evicted id** — eviction is clean and
monotonic (`percept/entity.py:202`). At this observation density the graph is a **64-slot LRU, not a
memory**. Caveat that limits the claim: observations came from a harness-local motion-blob detector at
frame rate, so the tracks are moving *regions*, not semantic entities. The association/eviction logic
measured is the shipped `EntityGraph`.

#### Metering defects

- **`funnel.refused` double-counts.** 7,480 = exactly 2 × `gate.ask` (3,740), and exceeds the 4,380
  judged ticks — no one-per-tick counter can produce that. Sites: `agent.py:4278` (degraded) and
  `agent.py:4321` (`truth == "unknown"`).
- **`stats.admission_by_cause` is in frame×goal units** (sums to 1,214,063 = 73 × 16,631) while
  `escalations` is in frame units. Two different denominators under one report.
- The primary ledger itself reconciles **exactly**: 24,484 frames = emitted = `edge.decision` =
  16,631 admitted + 7,853 suppressed. Admission residual **0**. `judge.started` = `judge.finished` =
  `gate.classified` = 4,380.

#### ✅ Memory is clean

334 samples. RSS 219.4 MB start → 272.5 MB peak → 132.5 MB end. Least-squares slope **−43.3 MB/h** —
net negative drift. No leak with 73 goals, 58k entity mints and 4k episodes over 82 minutes.

#### Defects in the harness itself, self-caught

- First builder used `scale=640:-2`, giving parts different heights; `-c copy` concat produced a file
  with correct duration and timestamps whose **pixels froze after the first part**. Two live runs
  (~$0.41) were spent before anyone checked the pixels. Fixed with fixed-geometry letterboxing and a
  mandatory `verify_content()` gate (`build_long_run.py:118`) that refuses to publish a manifest
  unless sampled frames actually differ.
- Per-goal incident counts read `inc.goal_id`; the contract field is `concern_id`
  (`contracts/__init__.py:1394`). **This run's per-goal incident column is unusable; the totals are
  not.**
- Provider latency was NOT measured on a quiet host — another agent ran concurrently. Same
  orchestration error as the resolution arm.

### 🚨 METHODOLOGY DEFECT — recall is not reproducible run-to-run; it tracks machine load

**This is the most consequential finding in the ledger and it retroactively limits every
single-run recall comparison we have made.**

Same 18 short-media cases, same settings, same build, three HIGH runs:

| arm | recall | ASK% | provider p50 | deadline expiries |
|---|---|---|---|---|
| HIGH (03:33, idle machine) | 0.797 | 26.2 | 1.143 s | 65 |
| HIGH replicate 1 | 0.469 | 53.6 | 1.355 s | 129 |
| HIGH replicate 2 | 0.531 | 37.1 | 1.237 s | 95 |
| LOW replicate 1 | **0.844** | 8.8 | 1.065 s | 12 |
| LOW replicate 2 | **0.844** | 9.9 | 1.013 s | 20 |

**HIGH swings 0.33 recall run-to-run.** It tracks provider p50 against the 1.5 s deadline: a busier
machine → slower calls → more deadline expiries → more ASKs → fewer decisive verdicts → lower recall.
LOW returns 0.844 twice because it sits ~0.3 s further from the deadline.

**Run-to-run noise (0.33) is ~20× the pooled LOW-vs-HIGH difference (+0.017).** Any recall claim built
on a single cross-run pair is uninterpretable — including the +0.115 recorded above.

**Cause was self-inflicted:** a long-horizon run (PID 60011) was executing throughout the replicate
session, while the 03:33 baseline had the machine to itself. Latency-sensitive measurement was
scheduled concurrently with a heavy job. **Machine quiescence is now a precondition for any timing or
recall arm, not a nicety.**

**What this means beyond resolution:** we have not been measuring model quality. We have been
measuring a **deadline race**. Recall at the 1.5 s reflex deadline is a function of provider latency
versus that deadline, and provider latency is not under our control. Effects smaller than ~0.33 recall
require replicates. The reflex-frame-bound result (0.137 → 0.626, Δ0.49) exceeds the noise band and
survives; nothing smaller does without repetition.

### 💰 `media_resolution=LOW` — CORRECTED at n=40: the cost win holds, the recall win does not

Superseding the n=15 entry below. Pooled over the full 40 matched cases:

| metric | HIGH | LOW | delta | verdict |
|---|---|---|---|---|
| input tok/call | 3,568 | 1,100 | **−69.2%** | deterministic, solid |
| cost | $2.599 | $1.225 | **−52.9%** | deterministic, solid |
| ASK % | 25.3 | 8.0 | **−68.6%** | solid |
| deadline expiries | 670 | 182 | **−72.8%** | solid |
| provider p50 | 1.123 s | 1.047 s | −6.8% | solid |
| recall | 0.653 | 0.669 | +0.017 | **inside the noise band — no claim** |
| precision | 0.340 | 0.316 | −0.024 | **inside the noise band — no claim** |
| timely recall @2 s | 0.186 | 0.271 | +0.085 | directional |

**Cases better/same/worse: 4 / 29 / 7** — more cases got *worse* than better. Pooled recall rises only
because gains land in repcount, which carries 52 of 118 occurrences.

**The n=15 fear was wrong, and in the opposite direction.** LOW does not trade short-media recall for
long-media recall: short 0.797 → 0.844, long 0.481 → 0.463. The prior arm's "+0.115, 0 cases worse"
was an artifact of comparing across runs at different machine loads.

**Adopt LOW for cost and deadline robustness, not for recall.** What it actually buys is ~0.3 s of
headroom against the 1.5 s deadline, which is why its recall is *stable* while HIGH's collapses under
load. Note `media_resolution` is **not implemented in `backends/gemini.py`** — it exists only as a
harness env override, so shipping it as a default is real work, not a config edit.

### 💰 `media_resolution=LOW` (SUPERSEDED — n=15, single-run pair) — the biggest single win found, and it is threefold

**`max_dimension` is strictly dominated and must NOT be touched.** Gemini bills a **fixed ~1,100
image tokens per frame regardless of pixel dimensions** — 768, 640, 512, 384 and 256 px all produce
an identical 3,434-token 3-frame prompt. Resizing the JPEG shrinks the upload and nothing else: same
price, same latency, less detail. (5.7× byte range, flat TTFT.)

**Gemini's own `media_resolution` knob is the real lever, and we never set it.** Measured, 15 matched
cases:

| metric | HIGH (today) | LOW | delta |
|---|---|---|---|
| recall | 0.462 | **0.577** | **+0.115** |
| precision | 0.175 | 0.164 | −0.011 |
| timely recall @2 s | 0.115 | **0.192** | +0.077 |
| ASK % | 27.0 | **7.8** | **−71%** |
| input tok/call | 3,560 | **1,100** | **−69.1%** |
| cost | $1.917 | **$0.922** | **−51.9%** |
| provider p50 | 1.145 s | 1.055 s | −7.9% |
| deadline expiries | 548 | **142** | **−74%** |

**Cases better 2, same 13, worse 0.**

**Same mechanism as the frame-count fix.** LOW's ~8% latency edge pushes calls back under the 1.5 s
deadline, so decisive verdicts rise 1,949 → 2,740 and ASKs collapse. That is why the realised token
saving is 57.6% rather than 69% — the pipeline completes **37% MORE calls** and still costs half as
much.

**Confidence is split, and the split matters:**
- **The cost claim is solid** — −69.1% per call is deterministic provider tokenisation, not statistics.
- **The recall claim is directional only** — n=15 (budget cap, not 40), and those 15 are all
  `omnipro` + `proactiveqa`, the LONG-media cells where deadline pressure is highest and LOW
  therefore helps most. The gain rests on 2 cases / +3 occurrences of 26. Zero regressions, but the
  remaining 25 (short media, ~$0.30) are needed before quoting +0.115.

**My stated baseline precision of 0.291 does not reproduce.** Under `live_run`'s own `quality()` it is
**0.340**; recall 0.653 / ASK 25.3% / FIRE 14.2% match exactly. Comparing against 0.291 would have
invented a ~5-point precision gain. Use 0.340.

### ⚠️ LATENCY — CORRECTION: the honest end-to-end is ~2.55 s, not 1.667 s

**The judge only samples the scene once per pipeline-cycle, and that wait was never counted.**
The scheduler is single-flight with one latest-wins pending slot (`worker.py:731-812`), so:

| run | tier-0 admitted | judged | **superseded** | discarded before any judge |
|---|---|---|---|---|
| reflex 5-frame | 67,775 | 6,269 | 45,564 | **87.9%** (verified independently) |
| deliberate | 67,775 | 6,419 | 46,449 | 87.9% |
| reflex 3-frame | 67,775 | 8,112 | 44,401 | 84.5% |

Judge sampling period: **1.771 s** (5-frame) / **1.389 s** (3-frame). An action onset therefore waits
an additional expected HALF-CYCLE — +0.885 s / +0.695 s — before a judge even looks.
**Honest onset→Incident ≈ 2.55 s (5-frame), ≈ 2.20 s (3-frame).** Every "1.667 s" figure earlier in
this ledger is the pipeline leg only and understates the product-visible latency.

**And the multiplier is the important part.** The sampling period IS the pipeline latency
(1.771 ≈ 0.167 + 1.433 + 0.072 + slack), so **every second cut from the pipeline is worth ~1.5 s
end-to-end.** Measured, not modelled — cutting 5→3 frames removed 0.178 s of prefill and bought
**0.35 s** end-to-end:

| frames | provider p50 | pipeline p50 | judge cycle | ≈ onset→Incident |
|---|---|---|---|---|
| 2 | 1.321 | 1.666 | 1.489 | 2.41 s |
| **3** | **1.255** | **1.502** | **1.389** | **2.20 s** |
| 4 | 1.356 | 1.666 | 1.662 | 2.50 s |
| 5 (was shipped) | 1.433 | 1.667 | 1.771 | 2.55 s |

**`ingest→admission` is mislabelled — it is queue wait, not gating.** On the async dispatch path used
by every run, `admitted_ts` is stamped when the batch LAUNCHES (`agent.py:3242`), so the 0.167 s is a
candidate waiting for the single-flight judge. Real pre-dispatch compute is **≤ 8.7 ms, ever**
(n=4,769). Offline fit: **overhead ≈ 0.046 + 1.63/fps** — ~46 ms is a genuine compute floor, the rest
is rendezvous with the capture tick and is recoverable ONLY by raising capture fps.
**Raising fps does not raise provider spend** (the scheduler already discards 88%): 0.69 calls/s at
6 fps vs 0.68 at 12 fps. 6→12 fps buys ~0.19 s of ~2.55 s (~7%) for decode CPU. Note capture fps
comes from the sensor plan (`worker.py:4066`, cap `_BRIEF_CAPTURE_CAP_FPS = 8.0` at `worker.py:2494`)
— a profile change, not a code change.

### LATENCY — prefill decomposition (unchanged, and it is the only lever)

**TTFT tracks the NUMBER OF IMAGE PARTS, not their tokens or bytes** (interleaved, Gemini direct):

| images | prompt tok | TTFT |
|---|---|---|
| 0 (text only) | 356 | **0.489 s** ← irreducible per-request floor |
| 1 | — | 0.744 s |
| 2 | — | 0.841 s |
| 3 (production) | 3,656 | 1.225 s |

~0.25 s per image part on a ~0.49 s floor. So:
- **Frame count is the only real latency lever, and it is spent** — 3 is the measured optimum; 2 costs recall 0.653 → 0.449.
- **Caching is dead** (G12) and **resolution is dead for latency** (G13).
- Sub-1.0 s would have to come from **overlapping the call with capture**, not from shrinking the request.

**But resolution is a large COST lever nobody has exploited.** We never set Gemini's `media_resolution`,
so we pay ~1,101 tokens/image; `MEDIA_RESOLUTION_LOW` is ~264 — a **69% cut in input billing on every
judge call**. A parallel probe saw identical `satisfied`/`action_present` across 5 samples at LOW but
correctly called 5 samples "not evidence". **A 40-case quality run is what would make it evidence,
and it is in flight.**

### DIRECT JUDGE / CACHE — both structurally unreachable BY DESIGN. Zero counters was correct.

**Direct judge** is a client-supplied RULE registry, not a model (`direct_judge.py:154-190`). It never
engaged for three independent sufficient reasons: it defaults to `None` and is constructor-only —
**no env var, no settings knob** (`agent.py:1155/1179/1262`); the harness never passes it
(`live_run.py:470-478`); and **no `DirectRule` implementation ships anywhere** in `percept/`,
`examples/`, `templates/` or `docs/`. The corpus goals also lack the namespace it routes on.
**Correctness risk if forced open:** the confident path substitutes the verdict wholesale and the Gate
treats it as a real judge read — a wrong rule produces a real **FIRE**, not an ASK. Verify cadence is
stakes-derived (`{"low":0,"medium":20,"high":5}`) and **no corpus goal sets `stakes`** → medium →
**19 of every 20 confident answers are never verified**. Against a pipeline whose per-window lift is
1.26× over chance, enable only for a deterministic oracle, pinned `stakes="high"`.

**Cache** is disabled by default (`config.py:120`) AND bypassed regardless: `agent.py:5910` skips the
cache when `frames or audio or process_mode or claim_mode or event_mode` — and the runs used a
temporal strip (`frames`) with `rearm="each_event"` (`event_mode`), so **both** were truthy on every
judgment. It is a single-frame dHash cache (`ttl_s=8.0`) for a near-static scene re-asked the same
question; this corpus is continuous motion over temporal strips, which is exactly what the bypass
exists for. **Risk if forced:** `audit_rate=0.05` and the auditor only exists if `audit=` is passed —
`live_run` passes nothing, so a false-serve returns unchecked (`agent.py:5982-5983`).

**Neither can take a single judgment off the provider.**

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
| G12 | Context caching can cut the 1.43 s provider call | Cacheable prefix is **100 tokens** (images are built FIRST, question appended LAST, so only the system instruction precedes them); Gemini's cache floor is **1024**. And the impossible best case — caching system + 2 of 3 images, 63% of prefill — moved TTFT **−0.7%** (0.940 vs 0.946 s, n=8 interleaved) |
| G13 | Lower image RESOLUTION buys latency (my hypothesis) | `media_resolution=LOW` cuts the prompt **−69%** (3656→1148 tok) for **−2.4% TTFT**; cutting upload bytes 4× at identical pixel dims gave −0.2%. Prefill token count is only weakly coupled to TTFT on this model |
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
