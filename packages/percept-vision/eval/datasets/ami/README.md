# AMI Meeting Corpus → Percept fixtures (board card DS2)

Dev tooling. **Not** shipped in the wheel. Downloads nothing, calls no model, runs no paid eval —
build code + a small committed synthetic sample + unit tests only.

Turns the [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/) NXT-format annotations into Percept
evaluation fixtures for **recorded-meeting** watching, plus a pre-registered held-out split, a bars
document, and a self-contained `meeting_observation@1` domain pack.

## What it reads (NXT stand-off layers)

| file | layer | used for |
|------|-------|----------|
| `<meeting>.<speaker>.words.xml` | tokens, each with `starttime`/`endtime` | timing dialogue acts |
| `<meeting>.<speaker>.dialog-act.xml` | dialogue acts pointing at word spans (`nite:child` href, single or `id(a)..id(b)` range) | the fire-able unit + its `[start, end]` |
| `<meeting>.abssumm.xml` | abstractive summary sentences under `ABSTRACT` / `DECISIONS` / `PROBLEMS` / `ACTIONS` headings | the watch condition |
| `<meeting>.summlink.xml` | extractive→abstractive links (`role="extractive"` dact ↔ `role="abstractive"` sentence) | which act evidences which summary sentence |

## What it emits

One Percept `Fixture` **per meeting**. A summary-linked dialogue act becomes a `GroundTruthEpisode`:

| abstractive heading | watch condition |
|---------------------|-----------------|
| `DECISIONS` | `a decision was made` |
| `ACTIONS` | `an action item was assigned` |
| `PROBLEMS` | `a problem was raised` |
| `ABSTRACT` | *(no condition — skipped)* |

- The linked act's **start** time is the fire timestamp (`decision_eligible_ts`, the instant
  `build_golden_timeline` fires on); the act's **end** is `episode_end_ts`. The closed matching window
  is the utterance span + `AMI_ALLOWED_LATENESS_S` (24s, one measured serial-judge turn of grace).
- `goal_shape = "episode"` (ADR-006 `SUPPORTED`).
- An act linked under two headings (a decision *and* an action) yields two key-disjoint episodes
  (`…:decision`, `…:action_item`), so `score()` treats them as distinct goals.

## Usage

```python
from eval.datasets.ami import (
    transform_corpus, write_fixture_tree, build_meeting_split,
    freeze_registration, load_meeting_bars,
)

fixtures = transform_corpus("/path/to/ami-nxt-export")          # one MeetingFixture per meeting
splits = build_meeting_split([f.meeting_id for f in fixtures])  # deterministic 50/50, BY MEETING
write_fixture_tree(fixtures, "/path/to/out/fixtures", splits=splits)

bars = load_meeting_bars()
reg = freeze_registration("/path/to/out/fixtures", bars)        # DerivationPreRegistration-style freeze
print(reg.registration_hash)                                    # pass back as expected_registration_hash
```

The written tree loads straight through `percept.eval.differential.load_fixtures` /
`load_held_out_split` / `load_bars` and is ready for `percept eval` / `percept derive-profile`.

## Split discipline

Split **by meeting, never by utterance**. Two dialogue acts from the same meeting share a speaker, a
room and a topic; letting one land in `dev` and its neighbour in `test` leaks the held-out set.
`build_meeting_split` assigns whole meeting ids and `load_splits` enforces train/dev/test
disjointness on load.

## Bars (`bars.json`)

Client-supplied acceptance bars, loaded via `load_bars`:

- `candidate_recall` ≥ 0.60
- `recall_delta` ≥ 0.05 **and** its 95% CI excludes zero
- `false_fire_delta` ≤ 0.0

## Committed synthetic sample

`testdata/nxt/` holds 4 NXT-representative synthetic meetings; `testdata/fixtures/` holds the tree the
transform produces from them (regenerate both with `python3 -m eval.datasets.ami.testdata._make_sample`).
`tests/test_ami_dataset.py` exercises the whole path offline.

## EXPERIMENTAL — recorded meetings only

The `meeting_observation@1` pack is for **recorded** meeting evaluation. The shipped `conversation`
`SourceProfile` is `stability="experimental"`: no diarization behind `identity_strategy="speaker_label"`
(two speakers can collapse to one entity) and goals judge serially at ~24s/turn, so a live conversation
outruns the watch. This pack makes **no live-conversation claim**.

## Registry-wiring follow-up (one line, applied by a human/Fable)

The pack is **not** auto-registered — this dev-tooling module must not mutate the shared process
registry that the sibling registry cards (DS1/DS3) own. To wire it in, add one line to
`percept/packs_default.py` (alongside `register_domain_pack(FACILITY_OBSERVATION_V1)`):

```python
from eval.datasets.ami.pack import MEETING_OBSERVATION_V1
register_domain_pack(MEETING_OBSERVATION_V1)
```

…or call `eval.datasets.ami.register_meeting_pack()` at eval time. Note `eval/` is dev tooling outside
the wheel, so if the pack graduates into the shipped set its module should move under `percept/` first.
