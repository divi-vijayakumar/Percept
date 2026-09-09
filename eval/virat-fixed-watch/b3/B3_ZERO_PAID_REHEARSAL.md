# B3 zero-paid production-path rehearsal

Status: **ENGINEERING PASS**

An exposed VIRAT calibration clip was decoded through `open_source` at 2 FPS with lossless/blocking
queue policy and media time, then driven through the production runtime, scene episodes,
transactional SQLite outbox, real loopback HTTP receiver, durable ACK ledger, and full trace recorder.
A deterministic local sequence judge exercised two rising edges; it makes no perception-quality
claim.

Measured result:

- Source: `VIRAT_S_050201_04_000669_000728.mp4`, SHA-256
  `f83057bf03575908f19da901f58776fa73a5476fb7d611d7e6629f678ff278f4`.
- 118 frames observed; 59 fixed-cadence admissions and local judge calls; 2 FIRE, 0 ASK, 57 silent.
- 2 incidents, 2 episodes, 2 action-log entries, 2 receiver attempts, 2 receiver ACKs, 2 durable
  delivery-ledger entries, zero pending outbox rows, zero dropped frames/deliveries, zero sink errors.
- Reconciliation: exact.
- SDK-yield to receiver-ACK latency: p50 `0.0015856244990573032 s`, p95
  `0.0017416246484572185 s`, p99 `0.0017554913284038775 s`, max
  `0.0017589579983905423 s`.
- Local judge trace latency: p50 `0.00011091700071119703 s`, p95
  `0.0003465000008873176 s`.
- Remote provider calls: `0`; provider cost: `$0.00`.
- All-in cost: not measured because no hardware/energy rate was frozen for B3.
- Source-onset-to-ACK: not measured because scripted rehearsal states are not GT occurrences.

The trace schema labels all 59 judge-seam dispatches `provider_calls`; they were the declared local
sequence judge, not network/provider traffic. The receipt preserves both values explicitly.

Qualification labels remained sealed. Next action is B4 calibration and validation only.
