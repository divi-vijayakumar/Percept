# Edge parity

Percept has two edge runtimes:

- **Python**: `percept.edge` / `percept.harness`, used server-side and by the runtime worker.
- **JavaScript**: `percept-edge`, used on device/browser/edge deployments.

They are required to agree on the frozen wire contract and on the cheap detector outputs. The parity
lane is the release evidence for v0.3.

## What Is Compared

`make parity` runs `packages/percept-edge/parity/compare.py`, which drives the JS runtime and the Python
harness on the same fixture streams.

It checks:

- `motion-periodicity` detector boundaries.
- `acoustic-onset` detector boundaries.
- `motion_score` gate values (whole-frame `mean`).
- `motion_score` **salience modes** — `block_max` / `p95` over the block grid, including a localized-event
  fixture (block_max ≫ mean), a non-divisible frame (locks the block-boundary math), and a region+block
  compose.
- `ego_residual_score` — the ego-compensation residual, including a pure-pan fixture (residual ≈ 0 while
  the raw mean is high).
- JS `Edge.processFrame` latency with gate + detector active.

The salience/ego arithmetic is intentionally integer-only (floor-of-product block boundaries, integer
per-block denominators, an integer nearest-rank percentile) so the two languages match as exact equality,
not just within tolerance.

Current local result:

```text
21 checks - ALL MATCH: the two routes are byte-for-byte equivalent
```

Latency is a local microbenchmark, not a hosted SLA claim. The release claim is byte-for-byte parity.

## Detector Handshake

The handshake has three parts:

1. The edge advertises `EdgeCapabilities`.
2. The brain compiles each goal into a `WatchSpec`, filtering requested detectors through the advertised
   capabilities.
3. The edge emits `BoundaryReport` events for selected detector boundaries, while the brain owns counting,
   refractory state, and escalation.

The invariant: an authored goal can request detector names, but unknown or unavailable names are dropped
before the edge runs anything.

## Release Gate

Before publishing a release that changes edge contracts, run:

```bash
make test
make test-js
make parity
```

A contract change must update both Python and JS tests, the `CONTRACT_VERSION`, and this doc if the
handshake semantics change.
