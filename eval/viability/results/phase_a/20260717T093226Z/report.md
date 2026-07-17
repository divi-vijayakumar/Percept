# Percept Viability — Phase-A For-Record Run

**Run:** `20260717T093226Z` · **git:** `e0eb3df` (bars-lock commit) · **seed:** `20260717`
**Substrate:** fake_judge (architecture isolation) · **Load N/M/E/F:** {'N': 4, 'M': 2, 'E': 4.0, 'F': 1}

> ⚠️ **PHASE A = architecture isolation (fake judge).** Every number below measures Percept's
> queue/concurrency/soak/latency machinery *isolated from any model*. **None of these are product
> or provider numbers.** Real quality, provider latency, and $/stream-hour are Phase-B (need the
> Gemini key) — see *Deferred*.

## G1 — Runtime boundedness  →  **PASS**
- max queue depth: **80** · max oldest-pending age: **0.0s**
- queue-age slope: **0.0%/hr** (tol 2.5)
- memory slope: **0.0 MB/hr** (tol 50.0)
- latency p95 drift: **0.0 ms** (tol 200.0)
- stable within tolerance: **True** · clean recovery after injected failure hour: **True**
- simulated hours: 6 (soak cap 24h; simulated via FakeClock)

## G2 — Watch economics (VOLUME structure; $/call deferred)
- VID judgments/stream-hour: **240.0** vs naive baseline **3600.0**
- **volume reduction: 15.0×** · observed fan-out F: 1.0
- cost ceiling: $10/stream-hour — *real $ needs Phase B*

## G3 — Latency FLOOR (fake dist; NOT the product number)
- admitted→decision p95: **0.5399s** → band **GREEN**
- ARCHITECTURE FLOOR on a FAKE uniform(0.10,0.50)s latency dist — NOT the product/provider number

## Deferred to Phase B (needs the provider — NOT faked)
- real recall/precision per archetype (needs a real judge over the fixtures + golden clips)
- real provider latency p95 (G3-A/G3-B product numbers)
- real $/stream-hour vs cost ceiling (needs real per-call pricing)
- P-JUDGE-SAFE-1 / P-WRAP-1 against real adapters
