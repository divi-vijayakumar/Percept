### The detect leg is skipped on a source no goal can read a subject from

**What it cost.** Every admitted frame spends two serialized provider round trips: `Vision.detect`
(measured p50 3.69s, ~81 output tokens) and then `Vision.judge` (p50 1.44s). `detect` exists to
populate the entity graph. On a source where every goal declares `identity_policy="scene"`, nothing
downstream reads that graph — a scene-scoped watch reduces onto the non-entity gate key by
definition — so the first round trip is bought and thrown away.

**What changed.** `Percept` now asks one question at observation admission: does anything on this
source still consume what `detect` populates? The leg is skipped only when the source carries at
least one active goal and **every** one of them DECLARES `identity_policy="scene"`. Both entry
points share the predicate — the inline `perceive_judged` path and the split
owner/provider path (`prepare_observation_candidate` marks the candidate, and
`dispatch_observation_candidate` returns the empty completion without calling the detector). The
graph still ingests the empty observation list, so every downstream stage sees exactly the frame a
detector that found nothing would have produced.

Each elided round trip bumps `observation.detect.skipped.identity_policy_scene_only`, so a run
reports its saving instead of leaving it to be inferred from a latency histogram.

**Why it is this narrow.** `watch_reducer_entity_id` runs on every watch judgment and hands back the
focus entity for every goal that did not declare scene, so a single undeclared goal anywhere on the
source makes detect's output load-bearing — and DECLARING is the only way in, because a goal that
merely inherits a scene-like source-profile default still reduces on focus identity. Three wider
predicates were measured and rejected; each fails silently rather than loudly:

- two subjects both satisfying a condition collapse onto one gate key — one incident instead of two,
  with `entity_id=None`;
- a transition goal scoped per entity spends zero judge calls and never touches its latch again;
- `assert_attribute` finds no entity to bind, returns `None`, and the user's correction never
  reaches the next judgment.

A predicate keyed on `PerceptionSpec.roi` was also tried and dropped: `roi` defaults to `"focus"`,
so it skipped 3 of 17,843 ingests, and widening it to compensate broke eleven tests.

**Metric-neutral, measured.** Across the offline suite the guard skips 237 ingests. Every signal
counter is byte-identical to a stock run — `funnel.fired`, `funnel.refused`, `gate.fire`, `gate.ask`,
`gate.idempotent_suppressed`, `cache.hit`, `episode.scene.entity_turnover_skipped` and the
`transition.*` family, 133 counters in all — and the only counter that moves is the new skip counter.
`tests/test_detect_skip_guard.py` pins the narrowness: one non-scene goal on the source, a per-entity
transition goal, a correctable goal, a two-subject scene, and an inherited scene default each keep
the leg.
