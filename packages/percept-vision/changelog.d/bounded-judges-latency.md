### Judge latency — the deadline stops cancelling work, and the SLO becomes a measured distribution

**This is a declared configuration change with a stated cost.** The pipeline is now slower and it
answers, where before it was fast and did not. Those are different products, and this note presents it
as the trade it is.

**The defect.** The reflex provider deadline was `1.5s`. The measured median provider call time on the
same corpus is `1.653s`. The deadline sat **below the median**, so over half of all reflex calls could
not physically finish. On the n=100 action corpus (build `2d5097d`, gemini-3.1-flash-lite) **5,224 of
7,304 reflex calls expired and only 960 returned decisive**; reflex end-to-end recall was **0.137**
against deliberate's **0.684 on identical frames**. The gap was cancellation, not judgment quality.

The reflex latency figures that run produced (p50 `1.365s`, p95 `1.4726s`, max `1.5155s`) were not a
measurement — every sample that would have landed above `1.5s` had been discarded before it could be
recorded. A censored distribution quoted as a latency promise proves only that the truncation worked.

**What changed.**

- `percept.backends.judge_pool.REFLEX_PROVIDER_DEADLINE_S = 8.0` (was `1.5` in the benchmark harness,
  `1.8` as the `JudgePool` default) and `REFLEX_DECISION_DEADLINE_S = 8.4`. Both are now named,
  exported constants with their derivation in the source, imported by the harness rather than restated.
  - `8.0s` is what makes the **already-shipped** resilience policy reachable: `with_retry(attempts=3,
    base_delay_s=0.2)` needs `3 x 2.2834s + (0.2 + 0.4) = 7.4502s`, rounded up. At `1.5s` the pool
    cancelled partway through attempt 1, so the two retries the code pays for could never run.
  - It stays **below** the runtime's own hang threshold (`stall_after_s`, default `20.0s`) so the pool
    still cancels first, and far below the transport socket guard (`_REQUEST_TIMEOUT_MS = 120_000`).
    Timeouts are **not** removed; the hang guard is intact at every layer.
  - The old `1.5s`/`1.9s` pair encoded a `+0.4s` allowance for non-provider work (queue wait + decision
    overhead, measured at `0.241s`). That is preserved as an **absolute** margin — it does not scale
    with provider latency — rather than as that pair's `1.267x` ratio.
- **The deadline is a hang guard, not a latency promise, and the code now says so.** Percept's latency
  statement is `percept/slo.py` §5: `p25 / p50 / p95` + `max` + `n` + the measurement conditions
  (model, transport, frame count, corpus, build). `LatencyDistribution` carries `None` for anything
  not measured and renders it `—`; a distribution collected under a deadline is marked `censored_at_s`
  and refuses to be `is_publishable`. `judge_latency_promise()` prints the current statement.
- **Measured vs pending.** `DELIBERATE_JUDGE_LATENCY` (p50 `1.653s`, p95 `2.2834s`) is measured and
  uncensored. **`REFLEX_JUDGE_LATENCY` is `None` — pending the uncensored reflex run now in flight.**
  Deliberate is cited as the current best estimate because reflex sends strictly less (bounded frames,
  bounded reply), making it a conservative upper bound — stated as a stand-in, never as a reflex
  measurement. No percentile is written that was not observed.
- `JUDGE_ROUND_TRIP_P95_CEILING_S = 2.5` replaces the `2.0s` "reflex infeasibility" bar in
  `tests/test_p_lat_base.py`. Derived, not chosen: highest measured uncensored p95 (`2.2834s`) plus the
  tree's own p95 drift tolerance (`0.2s`), rounded up. **It is a loosening, and is labelled
  measured-not-aspirational** — `2.0s` was a figure the pipeline was never measured to clear, held in
  place by the deadline that discarded the contradicting samples. A bar above the measurement still
  catches a regression; a bar below it only guarantees a red build. It tightens when the reflex run lands.

**What reflex IS now.** Unchanged as a distinct mode, and defined positively rather than by a clock:
the **cheap, terse, bounded-evidence** path — the 3 most recent frames in (`_REFLEX_MAX_FRAMES`), the
two fields the gate reads out (`_REFLEX_MAX_TOKENS = 32`), and route-selective (`JudgeCapability.modes`).
Neither bound depends on a deadline; both survive its removal on their own merits.

**One piece of evidence is now retired.** The frame sweep that picked 3 frames was run under the `1.5s`
deadline, and its own reading of the 4- and 5-frame rows was that they were dominated by expiry rather
than uncertainty. Those rows measured cancellation. **3 is retained, not re-confirmed**, and is
deliberately not re-tuned here — moving it with no post-change measurement would be picking a number to
suit a metric. The 2-frame row (under-grounding) still stands, as a claim about evidence sufficiency.

`slo.SETTLE_WINDOW_MAX_S` stays at `2.0s`: its old "judge round-trip reflex budget" citation is retired,
but the value is held and re-grounded as a bound on **added delivery delay**. It is deliberately not
re-derived upward against the measured judge latency — a settle window begins after the verdict
arrives, so provider speed is irrelevant to it, and widening it would only buy back the withholding
that bound exists to reject.
