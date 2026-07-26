# Stage-4 COGNITION + DISPOSITION — validation results

**Suite:** SV-4.1 / SV-4.5 / SV-4.6 / SV-4.7 / SV-4.8 (the exactness tests — no pre-registered bars)
**Runtime under test:** `percept 1.0.0` @ `/private/tmp/percept-launch/oss-final-venv-bba572a/lib/python3.12/site-packages/percept` (the launch wheel; the harness refuses to run against a source tree)
**Harness:** `eval/stage-validation/sv4_cognition.py` · machine-readable: `eval/stage-validation/results/sv4_cognition.json`
**Method:** every scenario is driven through the real public runtime path (`percept.run`) with scripted, network-free judges. No provider calls. Runs are deterministic — two consecutive full runs produced byte-identical check outcomes and defect sets.
**Wall time:** ~13 s for the full suite.

```
SV-4.1   PASS    9/9   checks
SV-4.5   FAIL    7/8   checks
SV-4.6   PASS   11/11  checks
SV-4.7   FAIL   13/16  checks   <-- 1 RELEASE BLOCKER
SV-4.8   FAIL   11/14  checks
```

---

## Headline verdict on evidence integrity (SV-4.7)

**The evidence media on the DEFAULT path is real. There is no fabrication, no re-ordering, no
substitution — on `snippet_kind="frame"`, which is the default.** Across three scenarios (baseline
multi-occurrence, an occurrence spanning an 11-tick admission gap, and a cache-served fire), **56
evidence frames over 12 delivered Incidents content-hashed exactly to frames the runtime was
actually fed**, each snippet aligning to a *contiguous ordered window* of the fed stream ending at
or before its Incident's timestamp. A negative control confirms the detector is not vacuous:
withholding one real frame from the ground truth immediately produced three unmatched evidence
frames.

**But there is one release blocker, on the opt-in `snippet_kind="clip"` path — the path that exists
precisely so evidence can LEAVE the process** (a single bytes payload for webhook/storage sinks).
On that path a decoded frame is silently replaced by a **truncated ASCII `repr`**. That is
substituted evidence, and it is unrecoverable.

| defect | severity | one line |
|---|---|---|
| **SV-4.7-D1** | **BLOCKER** | `snippet_kind="clip"` replaces a decoded (ndarray) frame with a truncated `repr` string and still labels it `application/x-percept-frame-bundle+json`. Silent. Irreversible. |
| SV-4.7-D3 | high | With the cache enabled, a fire's newest evidence frame can be one the judge never saw (dHash near-match serve). Media is real; the verdict↔media binding is not. |
| SV-4.7-D2 | medium | A cache-served Incident carries `provenance="observed"` — indistinguishable on the wire from a freshly-judged one. |
| SV-4.5-D1 | high | The perception cache's slot omits the ROI, so a moved ROI over a dHash-colliding region is served the old region's verdict — a stale entry satisfying a CHANGED Watch config. |
| SV-4.8-D1 | high | The default async judge scheduler never bumps `funnel.dropped`; 37 dropped units read as 0 in the funnel. |
| SV-4.8-D2 | high | One ASK is counted twice in `funnel.refused`, so `judged == fired + refused + silent` cannot hold on any run containing an ASK. |

**Does the drop ledger reconcile?** The **primary, per-unit ledger reconciles EXACTLY — zero
unaccounted units at every boundary, in all three scenarios**, and every dropped unit carries a
health receipt. The **funnel counters do not** (D1, D2). Two ledgers that must agree do not, and
the one that goes quiet is the funnel — the single definition `percept doctor`, `percept eval` and
the visualizer derive their numbers from.

**Is memory flat?** **Yes.** Over 3,000 frames × 16 KiB of media, in both the byte-bounded and
count-only configurations, the steady-state Python heap drifts **+0.51% / +0.23%** across the
second half (slope +5.4 / +4.1 bytes per frame, i.e. +0.02 MB projected over the whole run) and
peaks at **0.86 MB / 1.43 MB**. RSS is flat at 60.2 MB from frame ~350 onward.

---

## SV-4.1 — Gate decision + state — **PASS** (9/9)

Driven through `percept.run` over a 14-tick source alternating confident-false / confident-true,
with an alternating tier-0 motion signal so every tick after the prime escalates.

| check | measured |
|---|---|
| rising edge fires exactly once per occurrence | fired at t = 1, 3, 5, 7, 9, 11, 13 — one fire per false→true edge, zero extras |
| a HELD-true occurrence fires exactly once | 1 incident; gate reasons `[rising_edge, already_fired ×6]` |
| repeated occurrences produce **DISTINCT** Incidents | 7 incidents → 7 distinct ids, 7 distinct `judgment_event_id`s, 7 distinct `evidence_seq`s, 7 distinct newest-evidence content hashes |
| no field-for-field duplicate | 7/7 distinct on (id, ts, concern, evidence_seq, judgment link, verdict, confidence) |
| refractory honoured (`refractory_s=10.0`) | fired at t = 1 and t = 11 only |
| every suppressed fire carries a receipt | 5 `gate_suppressed` health events at t = 3, 5, 7, 9, 13, all `reason="refractory"` |
| re-arm after the window elapses | t = 11 is exactly 10 s after the last real fire and fires; t = 13 (2 s later) is suppressed again |
| closed reason vocabulary | 33 gate decisions, every reason ∈ `GATE_REASONS` |
| reason ⇒ action consistency | total agreement with `REASON_ACTION` |

Notable: the refractory window is anchored to the *last real fire*, not to the suppression, so a
suppressed fire does not extend the window — visible in the t = 11 fire landing at exactly
`last_fire + refractory_s`.

---

## SV-4.5 — Cache correctness — **FAIL** (7/8)

Cache enabled via `Settings(cache=CacheSettings(enabled=True, audit_rate=0.0, ttl_s=…))`. Default
in the shipped runtime is `enabled=False`.

**Passing:**

| check | measured |
|---|---|
| hit on an unchanged frame | 3 escalations → **1** provider call, 2 `cache.hit` |
| miss on structurally distinct frames | 5 escalations → 5 provider calls, 0 hits (min pairwise dHash Hamming **23** vs `tau_bits=6`, so no accidental collision in the fixture set) |
| goal-change invalidation | 1 entry → 0 after `invalidate_cache("zone")` |
| **the "same goal id, new condition" vector is closed at the front door** | `add_goal(same id)` **raises**; `GoalOp(op="update", fields={"condition": …})` **raises** (only `say` is a mutable field) — the runtime does not permit the config change the cache would have to notice |
| an accepted goal mutation still invalidates | `GoalOp(update, {"say": …})` → 1 entry → 0 |
| TTL is a hard ceiling | age 0 s → `HIT`; age 6 s with `ttl_s=5.0` → `DEFER` |
| assertion-override invalidation | 1 entry → 0 after `invalidate_cache_entity(...)` |

**Failing — the pass bar was "zero stale satisfactions of a CHANGED Watch config":**

Five config mutations were probed against the live `FrameHashLayer` of a live core, keyed exactly
as `percept/agent.py:5934` keys it (`CacheKey(goal_id, frame, roi, entity_id, condition)`):

| mutation | unchanged-config lookup | changed-config lookup | stale satisfaction? |
|---|---|---|---|
| condition changed | HIT | DEFER | no |
| goal id changed | HIT | DEFER | no |
| entity scope changed | HIT | DEFER | no |
| ROI moved to a structurally different region | HIT | DEFER | no |
| **ROI moved to a dHash-colliding region** | HIT | **HIT** | **YES** |

### DEFECT SV-4.5-D1 — high — stale entry satisfies a moved ROI

**Where:** `percept/cache/framehash.py:195` (the slot is `(goal_id, entity_id, condition)`) together
with `percept/cache/framehash.py:213` (the ROI is folded into the *hash*, not the *key*).

The layer assumes an ROI change changes the dHash. dHash is brightness-invariant by construction —
it emits one bit per horizontal adjacency comparison — so two genuinely different regions with
identical gradient structure hash identically. Moving a Watch's ROI (`goal.state["region_hint"]` →
the `PerceptionSpec` roi) is then served the previous region's verdict.

**Reproduction** (measured `dhash(top) == dhash(bottom) == 18446744073709551615`, Hamming **0** ≤
`tau_bits=6`):

```python
frame = _offset_ramp()          # top half = ramp(0..99); bottom half = the SAME ramp + 120
layer.store(CacheKey("zone", frame, Box(0, 0,   1, 0.5), None, "cond A"), YES, 0.95)
layer.lookup(CacheKey("zone", frame, Box(0, 0.5, 1, 0.5), None, "cond A")).status
# -> 'HIT'   (the bottom region is served the TOP region's verdict)
```

**Scope / mitigations already present:** the cache is `enabled=False` by default, and
`audit_rate=0.05` re-perceives 5% of hits. Neither closes the hole. The fix is to put the ROI in
the entry slot (or hash the ROI box into the key), not to widen the audit.

---

## SV-4.6 — Media buffers — **PASS** (11/11)

Two long synthetic legs through `percept.run`: **3,000 frames × 16,384 bytes** of media each,
`snippet_size=48`, `snippet_pre_s=5.0`, one true occurrence every 200 frames (15 incidents per leg).

> **Measurement hygiene.** An earlier revision of this leg reported **+941 bytes/frame** of growth.
> That was the *harness's own* accumulation — a retained list of 2,985 `RuntimeHealthEvent`s and a
> 3,000-entry hash→index map. The harness was rewritten to allocate O(1) per frame (frames carry
> their own index in their first four bytes; health events are counted, not retained; incidents are
> consumed and dropped). The curve below is the runtime's. This is also evidence the measurement is
> sensitive enough to see a leak of that size.

### Leg 1 — byte budget binding (`snippet_max_bytes=256_000`, i.e. 48 × 16 KiB = 768 KiB of media against a 250 KiB ceiling)

| metric | value |
|---|---|
| steady-state drift, second half | **+0.51%** |
| slope (linear fit, r² = 0.994) | **+5.4 bytes/frame** → +0.02 MB projected over 3,000 frames |
| mean steady-state heap | 0.81 MB |
| peak heap | 0.86 MB |
| max pre-roll held | **15 frames / 245,760 bytes** (bounds: 48 frames, 256,000 bytes + the newest frame, which is never evicted) |
| byte-budget evictions | 2,985, **every one announced** — `stats.preroll_frames_dropped == preroll_capped` health events == 2,985 |
| max snippet delivered | 15 frames / 245,760 bytes |
| pre-admission window retained | 15/15 incidents carry pre-fire frames, 0 without |

### Leg 2 — byte budget disabled (`snippet_max_bytes=0`, count bound only)

| metric | value |
|---|---|
| steady-state drift, second half | **+0.23%** |
| slope (r² = 0.991) | **+4.1 bytes/frame** → +0.01 MB projected |
| mean steady-state heap | 1.39 MB |
| peak heap | 1.43 MB |
| max snippet delivered | 21 frames / 344,064 bytes (ceiling 48 × 16,384 = 786,432) |
| byte evictions | 0 (as expected — the bound is off) |
| pre-admission window retained | 15/15 incidents |

### The measured curve

`heap` = `tracemalloc.get_traced_memory()[0]` (live Python allocation — the sensitive signal).
`RSS` = `resource.getrusage(RUSAGE_SELF).ru_maxrss`, a **process-wide monotone high-water mark**, so
leg 2 necessarily starts where leg 1 ended; it is reported for context, not as the pass criterion.

| frame | leg 1 heap (MB) | leg 1 RSS (MB) | leg 2 heap (MB) | leg 2 RSS (MB) |
|---:|---:|---:|---:|---:|
| 50 | 0.358 | 59.2 | 0.899 | 61.8 |
| 350 | 0.684 | 60.0 | 1.259 | 62.5 |
| 650 | 0.721 | 60.2 | 1.296 | 63.4 |
| 950 | 0.770 | 60.4 | 1.345 | 64.2 |
| 1250 | 0.782 | 60.5 | 1.357 | 64.2 |
| 1550 | 0.809 | 60.5 | 1.384 | 64.2 |
| 1850 | 0.811 | 60.6 | 1.386 | 64.3 |
| 2150 | 0.813 | 60.6 | 1.387 | 64.3 |
| 2450 | 0.815 | 61.6 | 1.388 | 64.3 |
| 2750 | 0.816 | 61.8 | 1.389 | 64.4 |

**Shape: flat, not growing.** Both heap curves rise during the first ~350 frames (the pre-roll ring
filling, plus warmup) and then plateau. From frame 1,550 to 2,750 leg 1 moves 0.809 → 0.816 MB
(+0.9%) and leg 2 moves 1.384 → 1.389 MB (+0.4%); the residual few bytes/frame are the `Metrics`
counter dict and the gate's touched-stamp map, both bounded by cardinality, not by frame count. The
RSS high-water mark moves 59.2 → 64.4 MB across **both** legs combined — 5.2 MB over 6,000 frames
carrying 98 MB of media through the runtime, and it is a high-water mark, so it can only be an
upper bound on real growth. Full 60-point curves per leg are in the JSON under
`measurements.legs.*.curve`.

---

## SV-4.7 — EVIDENCE INTEGRITY — **FAIL** (13/16) — 1 RELEASE BLOCKER

### What passed

| scenario | check | measured |
|---|---|---|
| A baseline | every evidence frame is real fed media | **42 frames / 8 incidents, 0 unmatched** by sha-256 |
| A baseline | contiguous, ordered, never postdating the fire | every snippet aligns exactly to a contiguous run of the fed stream, `t_end <= incident.ts` |
| A baseline | no empty evidence | 0 incidents without a snippet |
| A baseline | newest evidence frame is a frame the judge saw | 0 offenders |
| A′ **negative control** | the detector is not vacuous | withholding one real frame from ground truth ⇒ 3 unmatched frames flagged |
| B admission gap | evidence across a gap is real | 14 frames observed, **3 admitted** (11-tick gap); 8 evidence frames, 0 unmatched |
| B admission gap | un-admitted frames retained as honest pre-roll | pre-roll spans more frames than were ever admitted — real captured media covers the gap |
| C cache hit | the adversarial state was reached | 2 escalations, **1** provider call, 1 `cache.hit`, 2 incidents |
| C cache hit | evidence media is real | 6 evidence frames, 0 not in the fed stream, 0 misaligned |
| D revisions | no dangling supersession links | 0 dangling (no revisions produced in these scenarios) |
| D links | `judgment_event_id` injective **within a run** | baseline 8/8, admission-gap 2/2, cache-hit 2/2 — one judgment record per Incident |
| E resolver | `EvidenceStore` round-trips by content hash and is tamper-evident | sha match ✓; `EvidenceCorrupt` raised on a byte tamper ✓ |
| F1 clip | a `bytes` frame is byte-recoverable from the clip bundle | 4/4 entries recover the exact bytes — *but as a Python `repr` string, not the documented base64 `data` field* |

### DEFECT SV-4.7-D1 — **RELEASE BLOCKER** — substituted evidence on the clip path

**Where:** `percept/incidents.py:165-179` (`_encode_frame_bundle`), reached from
`percept/incidents.py:203` (`snippet_from_window(kind="clip")`), reached from
`percept/runtime/worker.py:5353` on every fire when the runtime is started with
`snippet_kind="clip"`.

`_encode_frame_bundle` base64-encodes a frame **only if it is a literal `(mime, bytes)` 2-tuple**.
Anything else falls into the `else` branch and is stored as `{"repr": repr(frame)}`. A decoded
frame from the shipped camera source is a **numpy ndarray** (`percept/runtime/source.py:219`,
`cap.read()`), and numpy's `repr` is **truncated with `...` above 1,000 elements**.

Measured: a 64×64×3 frame — **12,288 bytes of pixels** — becomes a **1,034-character truncated
string**. The delivered `IncidentSnippet` still advertises
`mime="application/x-percept-frame-bundle+json"` and the bundle still has the documented
`{"version":1,"t_start","t_end","frames":[…]}` shape. Nothing distinguishes it from a real frame
bundle. There is **no health event, no error, no counter** — it is silent, and it is
unrecoverable.

**Reproduction (unit):**

```python
import numpy as np, json
from percept.incidents import _encode_frame_bundle
a = (np.arange(64*64*3, dtype=np.uint8) % 251).reshape(64, 64, 3)   # 12288 pixel bytes
mime, media = _encode_frame_bundle([a], t_start=0.0, t_end=1.0)
e = json.loads(media)["frames"][0]
assert list(e) == ["repr"] and "..." in e["repr"]     # 1034 chars, pixels gone
```

**Reproduction (end-to-end, through the real public path):**

```python
core = Percept.create(vision=<scripted judge>, discover_plugins=False)
core.add_goal(Goal(id="zone", condition="c", say="Alert"))
async for inc in percept.run(core, source=[(t, gate_frame, ndarray_frame), ...],
                             snippet_kind="clip"):
    json.loads(inc.snippet.media)["frames"][0]   # -> {'repr': 'array([[[  0,   1, ...'}
```

**Why this is the blocker and not a nit:** `snippet_kind="frame"` (the default) keeps the frame
objects themselves, so in-process consumers are fine. `snippet_kind="clip"` exists specifically to
produce *one portable bytes payload for webhook/storage sinks without requiring ffmpeg* — i.e. it
is the mode chosen when evidence is about to leave the machine. On that mode, with the frame type
the shipped camera source actually produces, the evidence that leaves is a fabricated ASCII
stand-in.

**Suggested fix direction (not applied):** `_encode_frame_bundle` should encode any buffer-protocol
/ `tobytes()`-capable frame as base64 `data` with its shape/dtype, and **raise or emit a health
event** for a frame it cannot encode rather than falling back to `repr`. A bound-driven loss of
evidence must never be silent (ADR-001 §9); this one is worse than a bound-driven loss because the
payload still claims to be the media.

### DEFECT SV-4.7-D3 — high — verdict/evidence mis-binding under a cache near-match

**Where:** `percept/cache/framehash.py:218-232` (the near-match serve), reached from
`percept/agent.py:5937` and `percept/agent.py:2799`.

The layer serves on a dHash **near**-match (Hamming ≤ `tau_bits`, default 6), and dHash is
brightness-invariant. A frame with *different pixels* but the same gradient structure is therefore
served an earlier frame's verdict. The delivered Incident then pairs a verdict computed on frame X
with evidence media ending at frame **Y ≠ X**. The media is real captured media — nothing is
synthesized — but the verdict-to-evidence binding is broken and nothing on the wire says so.

**Reproduction:**

```python
p_judged = bytes(((x*7) % 100) for x in range(256))
p_near   = bytes(v + 120 for v in p_judged)          # different pixels
hamming(dhash(p_judged), dhash(p_near)) == 0          # <= tau_bits = 6
# cache enabled, audit_rate=0.0
# t=1: judge p_judged -> FIRE ;  t=2: re-arm via the documented Gate.rearm seam
# t=3: feed p_near     -> cache HIT -> FIRE
# incident.snippet.frames[-1] == p_near, which the judge was NEVER shown.
```

**Scope:** `PERCEPT_CACHE_ENABLED` is **off** by default; `audit_rate` (default 0.05) re-perceives
5% of hits, so 95% of near-match serves are unaudited.

### DEFECT SV-4.7-D2 — medium — a cache-served Incident is unmarked

**Where:** `percept/incidents.py:99` (provenance is read straight off the fire) vs
`percept/cache/framehash.py:226-232` (`CacheResult(provenance="cached")`).

Measured: the Incident delivered off a cache HIT carries `provenance="observed"`, byte-identical to
a freshly-judged fire, with the earlier verdict's `evidence` text. The layer's own `"cached"`
provenance never reaches the wire Incident. A downstream auditor cannot separate "the model looked
at this" from "we re-served an earlier answer".

---

## SV-4.8 — Drop-rate reconciliation — **FAIL** (11/14)

Three scenarios: a **clean** run (including one low-confidence ASK), an **overload** run (20 ms
judge under a 40-frame source at `fps=100`, forcing supersession drops), and a **cache**-enabled
run.

### The full ledger

| boundary | clean | overload | cache | balances |
|---|---|---|---|---|
| **source**: observed = accepted + excluded | 14 = 14 + 0 | 40 = 40 + 0 | 8 = 8 + 0 | ✅ |
| queue drops (`frames_dropped`) | 0 | 0 | 0 | — |
| **admission**: frames = admitted + not-admitted | 14 = 13 + 1 | 40 = 39 + 1 | 8 = 7 + 1 | ✅ |
| admitted partition (signal/fixed/coast/audio) | 13 = 13+0+0+0 | 39 = 39+0+0+0 | 7 = 7+0+0+0 | ✅ |
| `funnel.seen` / `funnel.gated` agree with stats | 14 / 13 | 40 / 39 | 8 / 7 | ✅ |
| **judge dispatch**: admitted = judge_calls + all named drops | 13 = 13 + 0 | **39 = 2 + 37** | 7 = 7 + 0 | ✅ |
| &nbsp;&nbsp;`judge_superseded` | 0 | **37** | 0 | |
| &nbsp;&nbsp;`judge_revision_dropped` / `_deadline_dropped` / `_deadline_expired` / `_source_discontinuity_dropped` | 0 | 0 | 0 | |
| **unaccounted at judge dispatch** | **0** | **0** | **0** | ✅ |
| drop receipts on `on_health` | 0 / 0 | **37 / 37** | 0 / 0 | ✅ |
| **judge execution**: judge_calls = provider calls + cache hits | 13 = 13 + 0 | 2 = 2 + 0 | **7 = 1 + 6** | ✅ |
| **gate**: judged = one typed decision receipt per unit | 13 = 6 fire + 1 ask + 6 silent | 2 = 1 + 0 + 1 | 7 = 0 + 0 + 7 | ✅ |
| **unaccounted at the gate** | **0** | **0** | **0** | ✅ |
| every SILENT carries a typed reason | 6/6 (`not_satisfied` ×6) | 1/1 (`already_fired`) | 7/7 (`not_satisfied` ×7) | ✅ |
| **delivery**: fires = incidents + sink errors = yielded | 6 = 6 + 0 = 6 | 1 = 1 + 0 = 1 | 0 = 0 + 0 = 0 | ✅ |
| — *mirror ledger* — | | | | |
| `funnel.dropped` vs total drops | 0 vs 0 | **0 vs 37** | 0 vs 0 | ❌ |
| `funnel.refused` vs gate ASK decisions | **2 vs 1** | 0 vs 0 | 0 vs 0 | ❌ |
| `funnel.judged` = `funnel.fired` + `funnel.refused` + silent | **13 ≠ 6+2+6 = 14** | 2 = 1+0+1 | 7 = 0+0+7 | ❌ |

**Reconciliation verdict:** the per-unit ledger — `RuntimeStats` + the `gate.classified` trace
receipts + `on_health` drop receipts — **reconciles exactly, with zero unaccounted units, in all
three scenarios**, including the run where 95% of admitted frames were dropped under overload. The
funnel mirror does not.

### DEFECT SV-4.8-D1 — high — `funnel.dropped` is dead on the default dispatch path

**Where:** `percept/runtime/worker.py:1417` (`class _JudgeScheduler`) — grep the whole class body
(lines 1417–2239) and there is **no `funnel` reference at all**. Its predecessor
`_LegacyJudgeScheduler` calls `funnel.dropped(...)` at `worker.py:793`, `:838`, `:1172`, `:1196`,
`:1230`, `:1303`. The active class is aliased `_ProviderScheduler = _JudgeScheduler`
(`worker.py:2068`) and instantiated for the visual watch path at `worker.py:3616` — i.e. this is
what the **default** `judge_dispatch="async"` runs.

Measured: the overload scenario genuinely dropped **37** admitted units. `stats.judge_superseded ==
37`, `on_health` received **37** `judge_superseded` events — so the drop is *not* fully silent —
but `core.metrics.get("funnel.dropped") == 0`.

**Reproduction:**

```python
core = Percept.create(vision=<scripted judge with a 20 ms delay>, discover_plugins=False)
core.add_goal(Goal(id="zone", condition="c", say="Alert"))
stats = RuntimeStats()
async for _ in percept.run(core, source=<40 frames, alternating motion>, stats=stats, fps=100.0):
    pass
stats.judge_superseded                      # 37
core.metrics.get("funnel.dropped")          # 0   <- should be 37
```

### DEFECT SV-4.8-D2 — high — one ASK is counted twice in `funnel.refused`

**Where:** `percept/agent.py:3777` (bumped when `classify(...)` returns `"unknown"`) **and**
`percept/agent.py:3828` (bumped again when the executor emits the resulting non-FIRE event for the
same unit).

Measured on the clean scenario, which contained exactly one low-confidence verdict
(`satisfied=True, confidence=0.20`):

```
gate.classified ASK decisions  = 1
stats.asks (delivered)         = 1
funnel.refused                 = 2      <- 2x
funnel.judged = 13   vs   funnel.fired 6 + funnel.refused 2 + silent 6 = 14
```

The funnel over-reports refusals by exactly the number of delivered ASKs. That is also the refusal
rate a deployment tunes `fresh_min` and the confidence threshold against.

---

## Reproducing

```
/private/tmp/percept-launch/oss-final-venv-bba572a/bin/python eval/stage-validation/sv4_cognition.py
/private/tmp/percept-launch/oss-final-venv-bba572a/bin/python eval/stage-validation/sv4_cognition.py --only SV-4.7
```

Exit code is non-zero if any test is not PASS. The harness refuses to run against a source
checkout (`assert_runtime_under_test`). All numbers above are reproduced verbatim in
`sv4_cognition.json`.

## Scope notes / what this suite does NOT cover

- **Fake judges only.** Per the brief, no provider or network calls. These tests validate the
  runtime's cognition and disposition machinery, never model quality.
- **Revised/superseded Incidents** (SV-4.7 D) were checked for link *integrity* (no dangling
  supersession, injective judgment links) but no scenario here produced an actual revision — the
  episode policies that emit `Revision` were not exercised, so "evidence for a revised incident"
  is verified structurally, not end-to-end. That gap should be closed by a scenario that drives a
  real `EpisodePolicy` correction.
- **Three checks call a live object directly** rather than steering the outer loop into the state
  (labelled `LAYER PROBE` in the source): the `FrameHashLayer` config-mutation matrix, the TTL
  ceiling, and `_encode_frame_bundle`. Each uses the live object on a live `Percept` core with the
  exact key/argument shapes `percept/agent.py` builds, and the clip defect is additionally
  confirmed end-to-end through `percept.run`.
- **Cache-served fires** required forcing a re-arm through the documented public `Gate.rearm` seam
  (the post-correction path), because the ordinary re-arm route — a confident-false read —
  overwrites the very cache slot the scenario needs to survive.
