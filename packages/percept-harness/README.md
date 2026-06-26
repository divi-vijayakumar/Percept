# percept-harness

> **⚠️ Research preview — v0.1.0.** Published for real-life testing and feedback, not for production.
> APIs may change between `0.1.x` releases — pin a version.

The **server-side transport shell + tier-0 salience gate** for the
[Percept](https://pypi.org/project/percept-vision/) cognition core.

It speaks the frozen edge wire contract — the brain compiles active goals **down** to a `WatchSpec`
(anchors + per-goal cheap-detector selection); a coarse motion gate consumes that spec frame-by-frame,
reports a `Tier0Signal` **up**, and the expensive cloud `perceive_judged` call runs **only when salience
clears threshold**. It also hosts the **edge detector registry** — cheap, opt-in "skills" (motion-periodicity,
acoustic-onset, pose-openness) the brain selects per goal to count/propose timed boundaries on-device.

Pure stdlib for the L1 lane (`numpy` is an optional fast path; RTMO/MMPose is opt-in via `[pose]`
and requires full MMCV compiled ops; BlazePose/MediaPipe is an explicit `[blazepose]` fallback). The real
on-device gate is `percept-edge` (JS/WASM); this is the Python server-side reference that speaks the same
contract.

```bash
pip install percept-harness          # pulls percept-vision
pip install percept-harness[pose]    # + RTMO/MMPose/MMCV ops for the pose-openness rep-counter
pip install percept-harness[blazepose]  # optional single-person/high-detail fallback
```

If full MMCV ops are unavailable on a local machine, the pose benchmark reports an
unresolved row instead of crashing the Python process. Use a Linux runtime image
or Colab-style environment for full RTMO execution.

See the [main README](https://github.com/) and `percept-vision` for the full architecture. Apache-2.0.
