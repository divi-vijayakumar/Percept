# Sustained-duration Watches: 1.0 release contract

A sustained-duration Watch fires only after a condition has remained continuously supported for an
authored interval. In `percept-vision` 1.0, this route is available through the in-process Python
runtime. It is session-scoped and is not supported by the remote JavaScript edge topology.

The canonical goal state is:

```python
{
    "hold_s": 300.0,
    "cadence": {
        "gate_on": "fixed",
        "min_judge_interval": 15.0,
    },
    "identity_policy": "scene",
}
```

`hold_s` is elapsed supported time, not an absence deadline. Each fixed observation must produce a
confident positive judgment. A negative or uncertain judgment, provider degradation, cache
degradation, source discontinuity, run boundary, pause, or resume clears pending dwell. Clearing
pending dwell does not re-arm a goal that already fired; a confident negative judgment still owns
normal rising-edge re-arming.

## `facility_observation@1/door_left_open`

The template parameter `deadline_after_s` is retained for source compatibility, but it now compiles
to `Goal.state.hold_s`. The template no longer authors:

- `Goal.state.rearm="deadline"`;
- `Goal.state.deadline_after_s`;
- an absence route; or
- a `door_interval` episode policy.

The corrected behavior is “fire once after the named door has been continuously observed open for
the requested duration.” A shorter opening does not fire.

This template is executable, but it is not quality-qualified in 1.0. The facility pack intentionally
declares no `door_left_open` fixture or quality bar because opening-action labels do not establish
that a door remained open. Do not interpret template availability as an accuracy, recall, latency,
or cost claim.

## Migrating an existing door Watch

An active or previously cancelled Watch ID cannot be reconfigured in place. Applying the corrected
plan with the same ID is rejected before runtime state changes. Start a fresh session or give the
corrected `WatchPlan` a new ID, apply it, verify the receipt, and only then retire the old Watch.
This prevents pending or fired state from the earlier absence/deadline program being read under the
new sustained-duration semantics.

## Runtime and restart boundary

Sustained-duration Watches require ordered, per-goal fixed-admission causes. The in-process Python
worker supplies that information. The current remote `percept-edge` wire contract does not, so
compilation for that topology raises `UnsupportedDurationRouteError` before frames or provider work.

Pending dwell is session-only. `Gate.snapshot()` includes its timestamp for inspection, but
`Gate.restore()` deliberately drops it because a pre-restart clock value cannot prove continuity on
the new process timeline. A restored plain-goal core therefore starts the duration again rather than
firing early.

Compiled Watches have a wider session-local runtime binding. Calling `Percept.snapshot()` or
`save_state()` after applying one raises `CompiledWatchSnapshotUnsupportedError`; it does not write a
partial recovery artifact. The OSS runtime does not promise duration continuity across worker
restart or deploy.
