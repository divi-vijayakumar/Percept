# SV-3.x — Judge pool boundedness / overload validation

**Component under test:** `percept/backends/judge_pool.py` (`JudgePool`)
**Runtime:** `percept 1.0.0` @ `/private/tmp/percept-launch/oss-final-venv-bba572a/lib/python3.12/site-packages/percept/backends/judge_pool.py`, CPython 3.12.8
**Harness:** `eval/stage-validation/sv3_pool.py` → `eval/stage-validation/results/sv3_pool.json`
**Provider calls:** none. Every judge is a locally constructed double with an injected `asyncio.sleep` delay. No credentials, no network, zero spend.
**Reproducibility:** the suite was run 4× back-to-back. Every check produced the identical verdict every time; the one failure below is deterministic, not flaky. All figures below are from the canonical run recorded in `sv3_pool.json`; latency and peak-depth figures move by a few ms / a few units between runs, counts and verdicts do not.

## Verdict

| Test | Result | Checks |
|---|---|---|
| SV-3.3 Overload allocation | **PASS** | 18/18 |
| SV-3.4 Deadline authority | **FAIL** | 17/18 |
| SV-3.5 Reflex route rejection | **PASS** | 11/11 |
| SV-3.6 Specialist/audio parity | **PASS** | 30/30 |
| **Overall** | **FAIL (1 defect)** | 76/77 |

The central claim — *admission-time rejection plus fail-closed shedding with a typed receipt, not unbounded queueing* — **holds**. In-flight occupancy was never exceeded by even one call, in any regime, on any evidence shape. The single failure is a distinct correctness issue (post-decision object aliasing) found by SV-3.4's mutation trap.

---

## SV-3.3 — Overload allocation — **PASS**

Two routes on one pool: `route_a` (`max_inflight=4`), `route_b` (`max_inflight=2`), each a fake judge with a fixed 20 ms provider delay. `deadline_s=0.25`, `hedge_after_s=0.0`, `receipt_limit=256`. Served capacity is therefore ~200 decisions/s on the preferred route.

Load was offered **open-loop** — arrivals fire on a fixed schedule regardless of drain. A closed loop (N workers looping) cannot demonstrate unbounded queueing because its outstanding count is pinned to N by construction, so it would have been an unfalsifiable test.

### Measured

| Metric | Sustained (open-loop) | Burst (10 × 800, 15 ms apart) |
|---|---|---|
| Units offered / completed | 4000 / 4000 | 8000 / 8000 |
| Wall time | 2.267 s | 0.208 s |
| Offered rate | 1 765 decisions/s | 38 384 decisions/s |
| Served capacity | ~200 decisions/s | ~200 decisions/s |
| **Peak in-flight `route_a`** (cap 4) | **4** | **4** |
| **Peak in-flight `route_b`** (cap 2) | **2** | **2** |
| Peak pool-wide in-flight (cap 6) | 6 | 6 |
| **Peak queue depth** (outstanding decisions) | **8** | **1 604** |
| Peak live `_call_route` tasks | 6 | 6 |
| Outstanding quartile means | `[7.94, 8.00, 8.00, 7.98]` | residual/burst `[800, 804, 800, 804, 800, 804, 800, 804, 800, 804]` |
| Growth slope | `+4.1e-05` / tick | `+0.121` / burst |
| **Shed count** (`capacity` receipts) | **7 334** | **15 970** |
| Fully-shed decisions | 3 556 (88.9 %) | 7 980 (99.75 %) |
| **In-flight leak at shutdown** | **0 / 0 / 0** | **0 / 0 / 0** |
| Exceptions raised to caller | 0 | 0 |

Leak triple = (live provider calls / `pool.snapshot()` inflight / live `JudgePool._call_route` tasks), measured after a 300 ms settle.

### Findings

- **`max_inflight` was never exceeded — not once, in either regime.** This is measured *at the adapter* (an enter/exit counter with a `finally`, so it counts occupancy including calls killed by cancellation), not read back from the pool's own bookkeeping. The pool's counters agreed with the adapter's exactly.
- **Excess sheds with a typed `capacity` receipt.** 7 334 sustained / 15 970 burst, each a `JudgeRouteReceipt(outcome="capacity")` naming the route. The shed path is admission-time: `_call_route` fails `_reserve()` and returns immediately (`judge_pool.py:430-431`).
- **Queue depth is bounded and flat.** Sustained outstanding held at ~8 decisions across a 4 000-unit run at ~9× overload, with a slope of `+4.1e-05`/tick — indistinguishable from zero. Peak outstanding was 8 out of 4 000 offered.
- **Zero in-flight leak at shutdown**, and all 4 000 / 8 000 decisions terminated. Nothing hung.
- **Receipt memory is bounded.** The retained ring held exactly 256 entries (`receipt_limit`) while 8 000 receipts were emitted. Accounting in this suite therefore goes through the `on_receipt` callback, not `pool.receipts()`.
- **Nothing was fabricated.** Every outcome was `served` or `shed`; a shed decision returns the synthesised UNKNOWN (`satisfied=False, confidence=0.0`), never a guess.

### Sustained vs burst — they differ, and the difference matters

The invariants are identical in both regimes, but the **service ratio is not**:

- **Sustained (1 765/s):** 11.1 % of decisions served, 88.9 % shed.
- **Burst (38 384/s):** **0.25 % served, 99.75 % shed.**

Under burst arrival, 800 decisions land in the same event-loop turn. The 6 available slots are taken by the first arrivals; the other ~794 fail `_reserve()` and shed in the same turn. Because the shed path is O(1) and the served path costs a full 20 ms provider call, a burst is overwhelmingly resolved by shedding. The pool completed 38 384 decisions/s in the burst regime — that number is almost entirely the cost of *refusing*, which is the honest reading: **the pool is very fast at saying no, and that is what protects it.**

Burst residual was flat at ~one burst (800 → 804 across ten bursts, slope `+0.121`/burst). If work accumulated, residual would have climbed toward 8 000; it did not move. Peak outstanding (1 604) is ~2 bursts, i.e. the driver's own offered depth, not amplification by the pool.

**Operational consequence:** for a real fast-packet stream, the pool will not fall over and will not queue — but it will also serve only a small fraction of packets, and the caller sees UNKNOWN for the rest. Boundedness is bought with shedding. Sizing `max_inflight` (and upstream admission) is what determines *which* packets get judged; the pool does not prioritise, it takes whoever reserves first.

### Architectural notes (not defects)

- **The pool holds no queue of its own.** `judge()` is a per-decision call that immediately fans out one `asyncio.Task` per eligible route; a task that cannot reserve a slot returns `capacity` at once. "Queue depth" is therefore the caller-side outstanding-decision count — the only place work can pile up. This is the correct design for the claim being made, but it means **backpressure is the caller's responsibility**: the pool will happily accept 8 000 concurrent `judge()` calls and refuse 7 980 of them.
- **`max_parallel` is a per-decision fan-out bound, not a pool-wide one.** The semaphore is constructed fresh inside every `judge()` call (`judge_pool.py:568`), so it cannot bound concurrency across decisions. The only pool-wide bound is `sum(JudgeCapability.max_inflight)`. This is easy to misread from the constructor signature; a deployment that sets `max_parallel=2` expecting a global cap will get `sum(max_inflight)` instead.

---

## SV-3.4 — Deadline authority — **FAIL** (17/18)

`deadline_s=0.30` throughout.

### A — outer deadline beats the declared SLO — PASS

Route declares `latency_slo_s=0.05`; the adapter actually takes 1.5 s.

- `judge()` returned in **0.3004 s** (deadline 0.30 s).
- Returned the UNKNOWN-shaped fail-closed verdict (`satisfied=False, confidence=0.0`, evidence `deadline expired`).
- Receipts: `{deadline: 1, cancelled: 1}`.
- The declared 0.05 s SLO did **not** truncate the call — the pool ran the route for the full 0.3004 s. Confirms the docstring at `judge_pool.py:63-65`: the SLO is a routing ceiling, not a timeout.
- The provider call was genuinely cancelled (`cancellations=1`, adapter live count back to 0).

### B — a cancellation-hostile late reply cannot resurrect the decision — PASS

The adapter swallows `CancelledError` and returns `satisfied=True, confidence=0.99` **after** expiry — deliberately the most attractive possible late reply.

- Adapter confirmed hostile: `cancel_seen=1`, `late_returns=1`.
- `judge()` still returned at **0.3026 s** with the UNKNOWN verdict.
- The returned verdict object was **byte-identical before and after** the late reply landed.
- The late reply injected **no new receipt** (`{cancelled:1, capacity:1, deadline:1}` immediately before → identical after; the `capacity` entry is this test's own occupancy probe).
- **No `decisive` receipt was ever recorded** for the hostile route.
- The route kept occupying its bounded slot (`inflight=1`) until the adapter truly terminated, then released to 0. This is the documented behaviour at `judge_pool.py:16-19` and it is correct: a decision issued during that window shed in **1.7 ms** with a `capacity` receipt rather than queueing.

This is structurally sound: `judge()` has already returned by the time the late reply lands, so the reply has no path to a receipt or a verdict. The only post-return state it touches is the route's inflight/failure counters.

### C — **DEFECT** — a provider can rewrite a decision the pool already concluded

Scenario: route A replies low-confidence (0.40) before expiry; route B never replies. The pool concludes the decision **by deadline expiry** (`{deadline:1, cancelled:1, uncertain:1}`) and, per `judge_pool.py:695`, hands back route A's retained `uncertain` reply rather than the synthesised UNKNOWN.

Measured:

| | satisfied | confidence |
|---|---|---|
| Returned to the caller | `False` | `0.40` |
| After the provider mutated the object it still held | **`True`** | **`0.99`** |

The pool returned the provider's **own object by identity**, and `Verdict` is a non-frozen dataclass, so an adapter that retains its reference can flip a decision the caller has already acted on.

**Location:** `packages/percept-vision/percept/backends/judge_pool.py:667` (retains `result.verdict` as `uncertain`) and `:695` (returns it); also `:659` on the decisive path. Type declared at `packages/percept-vision/percept/contracts/__init__.py:472` (`@dataclass class Verdict`, `frozen=False`).

**Scope, honestly stated:** this is an SDK-wide ownership convention, not something `JudgePool` introduces — the SDK's own `FakeVision.judge` (`backends/fake.py:61-95`) also hands back its scripted `Verdict` by identity (verified as a control in the same test; the decisive path is equally aliased, `(False, 0.95) → (True, 0.99)`). It requires a buggy or adversarial adapter to trigger, not a merely slow one.

**Why it still counts against SV-3.4:** the guarantee being validated is "a late provider reply cannot resurrect *or mutate* an already-timed-out decision." The *resurrect* half holds unconditionally. The *mutate* half holds **only on the synthesised-UNKNOWN paths** (`no_route`, all-routes-`capacity`, and deadline-with-no-retained-reply). On the deadline-with-retained-`uncertain` path — a normal outcome under exactly the overload SV-3.3 exercises — the pool hands out mutable provider-owned state.

**Suggested fix:** defensive copy at `judge_pool.py:659`/`:667`, or freeze `Verdict`. The codebase already does the analogous thing elsewhere — the compiled-audio dispatch deep-copies its scripted verdict (`audio_input.py:1023`, `return clone(verdict)`) — so the convention exists, it just is not applied here.

### D — terminal `deadline` outcome — PASS

With `terminal_outcomes=("deadline",)`: raised `TimeoutError` at 0.1511 s (deadline 0.15 s), latched it for `take_terminal_error()`, and still released the route slot (`inflight=0`). The raising path does not leak.

---

## SV-3.5 — Reflex route rejection — **PASS** (11/11)

`deadline_s=0.60`, `mode='reflex'`. The decisive evidence is the **adapter invocation counter**: rejection at selection means the adapter is never entered at all.

| Case | Route | Adapter calls | Receipt | Elapsed |
|---|---|---|---|---|
| 1 | `latency_slo_s=4.8` > deadline, reflex | **0** | `{no_route: 1}`, `route=None` | **0.00001 s** |
| 2 | `latency_slo_s=0.30` ≤ deadline, adapter actually 3.0 s | **1** | `{deadline: 1, cancelled: 1}` | 0.60 s |
| 3 | Same over-SLO route, `mode='deliberate'` | **1** | `{decisive: 1}` | fast |
| 4 | Mixed pool: over-SLO + in-SLO | over-SLO **0**, in-SLO **1** | no receipt at all for the unselected route | fast |
| 5 | `modes={'deliberate'}` only, `latency_slo_s=0.001`, reflex pool | **0** | `{no_route: 1}` | fast |

- **The two reasons are cleanly distinguishable:** case 1 emits exactly `{no_route}`; case 2 emits exactly `{deadline, cancelled}`. No overlap.
- Rejection cost **0.00001 s** vs a 0.60 s deadline burn — nearly five orders of magnitude apart, so a rejected route costs the watch essentially nothing.
- Case 4 confirms the over-SLO route is *skipped*, not merely deprioritised: it produces **no receipt whatsoever**, because `_eligible()` (`judge_pool.py:367-377`) filters before any task is created (`judge_pool.py:518`).
- Case 3 confirms the rejection is reflex-specific, and case 5 confirms the mode gate is independent of the SLO gate.

---

## SV-3.6 — Specialist / audio parity — **PASS** (30/30)

Every SV-3.3/3.4/3.5 invariant re-run on four evidence shapes. Overload sub-run per shape: 900 units open-loop at ~1 500/s against a single route with `max_inflight=3` and a 20 ms provider delay.

| Evidence shape | Peak in-flight (cap 3) | Peak queue depth | Capacity receipts | Leak | Deadline authoritative | Reflex rejected at selection | Shape exact |
|---|---|---|---|---|---|---|---|
| `single_image` (baseline) | **3** | 6 | 798 | 0/0/0 | 0.2513 s ✓ | 0 adapter calls ✓ | served ✓ |
| `audio` | **3** | 6 | 798 | 0/0/0 | 0.2513 s ✓ | 0 adapter calls ✓ | `no_route` ✓ |
| `fused` | **3** | 6 | 798 | 0/0/0 | 0.2503 s ✓ | 0 adapter calls ✓ | `no_route` ✓ |
| `context_stream` (specialist) | **3** | 9 | 795 | 0/0/0 | 0.2527 s ✓ | 0 adapter calls ✓ | `no_route` ✓ |

All 900 units completed on every shape. Quartile means of outstanding were flat on all four (`single_image` `[5.96, 5.96, 5.99, 5.97]`, `audio` `[5.68, 5.92, 5.96, 6.00]`, `fused` `[5.80, 5.92, 5.84, 6.00]`, `context_stream` `[5.96, 6.00, 5.88, 6.00]`).

**Parity holds exactly** — the invariant matrix (`bounded`, `sheds`, `queue_bounded`, `no_leak`, `deadline`, `reflex_reject`) is identical for `audio`, `fused` and `context_stream` versus the `single_image` baseline.

**Evidence-shape exactness confirmed:** a pool containing only a `single_image` route returns `no_route` for `audio`, `fused` and `context_stream` requests and **never invokes the image adapter**. There is no silent downgrade of audio or context-holding evidence onto an image judge — the guarantee at `judge_pool.py:10-11` is real.

### Architecture finding — the compiled-audio binding path does not go through `JudgePool`

`prepare_audio_judge_binding()` (`packages/percept-vision/percept/audio_input.py:1028-1062`) accepts **only** an exact SDK-owned `FakeVision` or `GeminiVision` instance. Probed directly:

- `prepare_audio_judge_binding(JudgePool(...))` → `ValueError: compiled audio requires an exact SDK-owned FakeVision or GeminiVision adapter`
- `prepare_audio_judge_binding(FakeVision())` → accepted

This is a deliberate closed allowlist (it also rejects instance-level and class-level `judge` shadowing), and it is *not* a defect. But it has a consequence worth stating plainly: **a compiled `AudioJudgeBinding` is a frozen single-adapter dispatch that inherits none of the pool's capacity, deadline or receipt machinery.** The audio parity proved above applies when a `JudgePool` is installed as the Vision adapter; it does **not** transfer to the compiled-audio path, which has no `max_inflight`, no outer deadline and no `JudgeRouteReceipt`. Anyone reasoning "audio is bounded because the judge pool is bounded" would be wrong for that path.

### Note on "specialist"

`FakeSpecialist` (`backends/fake.py:124`) is a token streamer with no `judge()` method, so it is not routable through `JudgePool` by construction (`JudgeRoute.__post_init__`, `judge_pool.py:119-122`). The specialist path exercised here is the `context_stream` evidence shape — the context-holding route the module docstring calls out at `judge_pool.py:10-11`.

---

## Defect summary

| # | Severity | Location | Description |
|---|---|---|---|
| 1 | Medium | `packages/percept-vision/percept/backends/judge_pool.py:659`, `:667`, `:695` (type at `packages/percept-vision/percept/contracts/__init__.py:472`) | The pool returns the provider's own `Verdict` instance with no defensive copy, and `Verdict` is a non-frozen dataclass. An adapter that retains the reference can rewrite `satisfied`/`confidence` **after** the pool has concluded the decision — including a decision concluded by deadline expiry that returned a retained `uncertain` reply. Measured: returned `(False, 0.40)`, provider rewrote it to `(True, 0.99)`. SDK-wide convention (`FakeVision` aliases too), requires a buggy/adversarial adapter, but it is a real hole in the "a late reply cannot mutate a concluded decision" guarantee. Fix: copy at `:659`/`:667`, or freeze `Verdict`. |

## Non-defect observations worth carrying forward

1. **`max_parallel` does not do what its name suggests** (`judge_pool.py:568`) — it is per-decision, not pool-wide. Pool-wide concurrency is `sum(JudgeCapability.max_inflight)`.
2. **No caller-visible distinction between "shed" and "no route".** A fully-shed decision returns evidence `all eligible routes were unavailable`; the `capacity` reason exists only in the receipt, not in the `Verdict`. A caller that reads only the verdict cannot tell overload from misconfiguration.
3. **Backpressure is the caller's job.** The pool never blocks an arriving `judge()` call; it accepts unlimited concurrent decisions and refuses the excess. Bounded, but it means an unbounded caller still costs unbounded task churn (38 k decisions/s of pure refusal was measured).
4. **The compiled-audio path is outside the pool's guarantees** (see SV-3.6 above).

## Reproduce

```
/private/tmp/percept-launch/oss-final-venv-bba572a/bin/python eval/stage-validation/sv3_pool.py
/private/tmp/percept-launch/oss-final-venv-bba572a/bin/python eval/stage-validation/sv3_pool.py --only SV-3.4
```

Exit code 0 = all checks pass, 1 = at least one failed. The harness refuses to run if the repo's `packages/percept-vision` is on `sys.path`, so it can only ever exercise the installed wheel.
