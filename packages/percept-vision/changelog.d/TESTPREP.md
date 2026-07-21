### Added — synthetic UNIT-layer capability fixtures (test-prep, not shipped behaviour)

Fixture groundwork that runs AHEAD of the capability cards, so the corpora exist, are labelled and
are proven to load before any implementation needs them. Nothing here changes runtime behaviour.

- **`percept.harness.scripted.ScriptedJudge`** — a `Vision` double that is scriptable PER CALL (a
  sequence, a frame-token map, or a hook), records every call, fails loudly when its script is
  exhausted, and tolerates additive `Verdict` fields that do not exist yet (`subject_key`,
  `binding`, `operation`): they are attached to the instance and mirrored in
  `verdict.scripted_extras`, so a fixture written today keeps working unchanged when P0 promotes
  them to real dataclass fields. `FakeVision` is untouched — this is purely additive.
- **`percept.harness.synthetic`** — seven seeded corpus generators, a machine-readable ground-truth
  schema + validator, and a loader: `subject_identity`, `continuation`, `mutation`,
  `episode_counting`, `intervals`, `idempotency`, `longhorizon` (21 cases, 194 steps). Every case
  states the EXPECTED ANSWER, the specific wrong answer it exists to catch, and — required by the
  validator — what it does NOT prove.
- Generate with `python3 -m percept.harness.synthetic.generate`. The data is written to the repo's
  `eval/synthetic/` (newly gitignored) together with its README; only the generators are tracked.
  Generation is a pure function of the seed — no wall clock, no unseeded randomness — so
  regeneration is byte-identical and the data is safe to gitignore.
- **`tests/test_synthetic_fixtures.py`** asserts the FIXTURES are valid (every corpus generates,
  loads, is internally consistent, and drives a `ScriptedJudge`). It deliberately does NOT assert
  the capabilities themselves — those are not built yet.

The split this serves: capability is deterministic and belongs in unit tests; quality is statistical
and belongs in an end-to-end benchmark at release. These are the unit layer — no API key, no spend,
no real footage.
