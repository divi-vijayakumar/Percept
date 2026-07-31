---
title: Edge parity
updated: 2026-07-16
applies_to: percept-vision 1.0.0
status: stable
---
# Edge parity

Percept has two edge runtimes:

- **Python**: `percept.edge` / `percept.harness`, used server-side and by the runtime worker.
- **JavaScript**: `percept-edge`, used on device/browser/edge deployments.

They are required to agree on the shared frozen wire contract and cheap detector outputs. That
parity does not imply that every Python runtime policy is executable remotely.

## Duration support boundary

Sustained-duration Watches (`Goal.state.hold_s` with fixed per-goal cadence) currently require the
in-process Python worker. Their correctness depends on ordered admission records that identify both
the goal and whether its tick was admitted by fixed cadence or motion. The remote JavaScript edge
contract does not carry that per-goal cause stream.

The compiler therefore rejects a sustained-duration Watch routed through remote `percept-edge` with
`UnsupportedDurationRouteError`, before pulling frames or invoking a provider. Use the Python
`percept.run(...)` topology for this behavior. See
[Sustained-duration Watches](../watches/sustained-duration-watches.md) for migration and restart limits.

## What Is Compared

`make parity` runs `packages/percept-edge/parity/compare.py`, which drives the JS runtime and the Python
harness on the same fixture streams.

It checks:

- `motion-periodicity` detector boundaries.
- `acoustic-onset` detector boundaries, including two vectors that pin the **shipped bars themselves**
  (fire and re-arm). The original four vectors sat far above every candidate threshold, so they were
  blind to the number — which is how the JS edge shipped `high=0.09` against Python's `0.001` for a
  release while this gate reported MATCH.
- **The wake fusion** — `wake_score` under exact float equality, plus `should_wake`'s `top_channel` and
  `wake` flag, over vectors whose signal and weight populations deliberately differ (iterating the wrong
  one is the defect the vectors exist to catch; a case where the two agree cannot see it). They cover the
  unweighted max, a measured channel the policy does not price, a typo'd channel name, a weight on an
  unmeasured channel, an explicit `0.0` prior, saturation above 1.0, the `floor`, and the per-term clamp
  from both the score and the top-channel side.
- `motion_score` gate values (whole-frame `mean`).
- `motion_score` **salience modes** — `block_max` / `p95` over the block grid, including a localized-event
  fixture (block_max ≫ mean), a non-divisible frame (locks the block-boundary math), a region+block
  compose, and three vectors that pin the **clamping rule** (an area under the grid in both dimensions,
  and an area exactly equal to it, where the clamp must not bite).
- `ego_residual_score` — the ego-compensation residual, including a pure-pan fixture (residual ≈ 0 while
  the raw mean is high), vertical-pan and `k > frame` fixtures that pin the shift search itself, and
  three **watch-out-zone** fixtures separating "the region is ignored", "the region picks the shift",
  and "the denominator is the region rather than `region ∩ overlap`".
- JS `Edge.processFrame` latency with gate + detector active.

The salience/ego arithmetic is intentionally integer-only (floor-of-product block boundaries, integer
per-block denominators, an integer nearest-rank percentile) so the two languages match as exact equality,
not just within tolerance.

The command reports every check in the current fixture set and fails if either implementation
diverges:

```text
ALL MATCH: the shared routes are byte-for-byte equivalent
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

A shared contract change must update both Python and JS tests, the `CONTRACT_VERSION`, and this doc
if the handshake semantics change. A Python-only policy must be rejected explicitly at remote
compilation; silently treating shared primitive parity as policy support is a release failure.
