# Edge parity

Percept has two edge runtimes:

- **Python**: `percept.edge` / `percept-harness`, used server-side and by the runtime worker.
- **JavaScript**: `percept-edge`, used on device/browser/edge deployments.

They are required to agree on the frozen wire contract and on the cheap detector outputs. The parity
lane is the release evidence for v0.3.

## What Is Compared

`make parity` runs `packages/percept-edge/parity/compare.py`, which drives the JS runtime and the Python
harness on the same fixture streams.

It checks:

- `motion-periodicity` detector boundaries.
- `acoustic-onset` detector boundaries.
- `motion_score` gate values.
- JS `Edge.processFrame` latency with gate + detector active.

Current local result, measured during the v0.3 pass:

```text
12 checks - ALL MATCH: the two routes are byte-for-byte equivalent
EDGE LATENCY: 5,000 frames in 14.9 ms -> 2.99 us/frame ~334,562 fps
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
