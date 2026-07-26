# bench-staging — the real public benchmark set (to be built)

Empty placeholder. Populate it only with licence-cleared material so `percept-bench` is reproducible
and redistributable.

Target layout (mirrors `fixtures-private/`):

```
scenes/        clips/frames + metadata (cam type, motion, lighting, #objects)
labels.jsonl   per-scene ground truth: entities, events[], rising-edge ts
gate_cases.jsonl  (scene_id, goal, expected_verdict) tagged by difficulty
utterances.jsonl  utterance + ASR-messy variant + gt-intent + gt-surface
splits.json    train 40% · dev 20% · test 40% (held-out, NEVER tuned on)
```

Two sources, minimal collection:
1. **Public datasets** for the benchmark axes — Ego4D, Epic-Kitchens, Charades/-Ego, RepCount.
   License-confirmed, human-labeled, rerunnable by anyone.
2. **Your own webcam/screen clips** for the trust axes — the tea/coffee correction, ambiguous→ask,
   the deictic exemplar, the second-cup boundary. Label rising-edge ts + expected verdict by hand.

Labeling = VLM-proposes + **human-reviews**. The trust metric needs human ground truth; a VLM can't
ground-truth its own blind spots. Keep the test split immutable.
