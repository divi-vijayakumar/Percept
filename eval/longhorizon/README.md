# eval/longhorizon — long-horizon sources with injected ground truth

The longest evaluation source in this repo is 81.6 minutes. The runtime is meant to run for days.
The gap is not laziness — it is arithmetic: nobody can hand-label hours of video, and a precision
number taken from a partly-labelled soak is exactly the "precision without ground-truth coverage"
error that `docs/evaluation/benchmarks.md` forbids.

**Injection closes the gap.** Take clips whose events are already annotated frame-by-frame, splice
them into a long stream of unlabelled background at scheduled offsets, and shift their intervals
onto the joined clock. You get frame-exact ground truth at *any* horizon, for the price of labels
you already have.

```
0s                                                                              7200s
├── background ──┬─ clip ─┬───── background ─────┬─ clip ─┬── … ──┬─ clip ─┬── background ──┤
   gt: "none"    │ labelled                      │ labelled       │ labelled
                 └ frame-exact positives, shifted onto the joined clock
```

Everything here is built by `build_injected_source.py`. Its module docstring is the normative
schema; this page is the orientation.

---

## Build

```bash
PYTHONPATH=. python3 eval/longhorizon/build_injected_source.py \
  --source-id inject_2h_v1 --target-s 7200 --n-inject 32 \
  --policy jittered --min-gap-s 45 --lead-in-s 60 --seed 20260731 \
  --background-dataset proactiveqa,omnipro \
  --gt-strengths annotated_interval --jobs 6
```

Outputs `sources/<source-id>.mp4` and `sources/<source-id>.manifest.json`. Intermediate
re-encoded parts are deleted after the concat — the build machine is disk-constrained.

## Verify

```bash
PYTHONPATH=. python3 eval/longhorizon/build_injected_source.py \
  --verify eval/longhorizon/sources/inject_2h_v1.manifest.json \
  --verify-origins --verify-deep
```

Re-derives every claim from the media on disk: the built file's hash, container duration against
the joined clock, the segment tiling (exact cover of `[0, total]`, no hole, no overlap), every
positive inside its own injected clip's span, one decoded frame at each injection offset, and the
origin clips still hashing to the bytes their labels were annotated against. Non-zero exit on any
mismatch.

## Where the outputs live

`sources/` is git-ignored, exactly like `eval/stage-validation/long_sources/`: the media is heavy
and derived from footage that is not ours to redistribute. The tool and this page are tracked, the
pixels are not. Rebuild from the command above — a build is a pure function of `--seed` and the
arguments recorded in the manifest's `build` block.

---

## The manifest, in brief

| field | what it carries |
|---|---|
| `source_id`, `total_duration_s`, `video_sha256` | identity, the joined clock, and the pixels the manifest describes |
| `container_duration_s`, `drift_s` | what ffprobe says vs what the arithmetic says |
| `profile` | the uniform encode profile — two manifests are comparable only if this matches |
| `clock` | wall-clock anchor semantics, and `unobserved` intervals |
| `license`, `redistributable` | per-contributor licence, and the AND over all of them |
| `segments` | the ordered tiling of the clock: `background` or `injection`, no gaps |
| `injections` | per clip: origin case, origin hash, offsets, shifted positives, GT tier |
| `watches` | one entry per `(goal_text, gt_strength)` — what a runner arms |
| `totals` | counts and labelled seconds **per tier**, never blended |
| `checks` | schedule feasibility, collisions, tiling, drift, advisory overlaps |

### Wall-clock anchor semantics

`clock.anchor_semantics` is `none` for a spliced source: media `t=0` corresponds to no real
instant. **Schedule-derived ground truth — sunrise, transit times, timed lighting — is invalid
here.** `declared` means an anchor was supplied for convenience and is a fiction. `recorder_utc`,
which a real recorded source carries, is the only value under which wall-clock reasoning means
anything.

`clock.unobserved` is empty for a spliced source — there is no recorder outage. The field exists so
a replay adapter reads the same shape whether the source was spliced or recorded, and so a real gap
surfaces as an interval that is neither positive nor negative.

### Background is unlabelled, not negative

Every background segment carries `gt: "none"`. That means **nobody looked**, not "nothing
happened". Background is real footage that was never annotated for the injected conditions, so an
injected goal may well be genuinely true somewhere in it.

One consequence, and it is not negotiable:

> **Recall and latency are computable on this source. Precision and false-positive rate are not.**

A fire during a background segment is not evidence of a false positive. Where a background clip was
lifted from a labelled corpus case, its own labels are discarded and `labels_discarded` records
what was dropped and why, so the omission is visible instead of silent. `checks.
goal_background_lexical_overlap` flags, advisorily, the injected goals whose wording overlaps a
background clip's own annotation — the places where that risk is highest.

### Ground-truth strength: three tiers, never averaged

The discipline `eval/stage-validation/build_audio_corpus.py` established for audio, applied to
intervals. Every interval carries its tier; tiers are reported separately; no number is ever
computed across them.

| tier | what a human actually annotated | may carry a headline number? |
|---|---|---|
| `annotated_interval` | the START and END of the event itself | yes |
| `annotated_point` | the INSTANT it became answerable — a point, with a stated tolerance | no, and never into an interval IoU |
| `derived_span` | a related or whole-clip span standing in for the event's extent | no |

`totals` therefore has `labelled_seconds_by_strength` and no blended total. `watches` are keyed on
`(goal_text, gt_strength)`, so the same sentence at two strengths becomes two watch entries and a
scorer cannot merge them by accident. The builder defaults to `--gt-strengths annotated_interval`:
a tier-pure source is the one that cannot be mis-scored.

---

## What a built source proves

- **Re-arm across a long horizon.** Tens of separated occurrences of the same and different goals,
  with minutes of idle between them, on one continuous decode. A goal that fires once and latches
  is visible immediately.
- **Episode-window and refractory behaviour over an hour-plus**, which no 30-second clip can reach.
- **Latency and recall against frame-exact ground truth** at a horizon where drift, accumulated
  state and scheduler behaviour are actually in play.
- **Label-free invariants over a long run** — memory ceiling, ledger reconciliation, checkpoint and
  restart continuity, judge calls per source-hour, cost per source-hour. These need no labels at
  all and are most of the value.
- **That the offset arithmetic is honest.** Scheduled and realized offsets are both recorded, along
  with the drift between them, so a latency number can be traced to the frame it came from.

## What it explicitly does NOT prove

Be blunt about this; the fixture is a stand-in, not a substitute.

- **It is not continuous footage.** There is a hard cut at every segment boundary — roughly every
  few minutes. Real continuous video never does that, and a runtime that keys on scene change sees
  a stream of step functions that do not exist in the field.
- **No lighting drift, no diurnal cycle.** Spliced archival clips have no day/night, no IR cutover,
  no gradual exposure ramp. The whole diurnal source class is untouched, and any claim about a
  non-stationary event rate is unsupported here.
- **No scene identity persistence.** The camera, the room, the people and the framing change at
  every cut. Long-horizon re-identification, "the same person came back an hour later", and
  entity-track longevity are *not* exercised — they are actively broken by the construction.
- **No background continuity.** Background segments come from several source clips; even the
  unlabelled stretches are cut.
- **No precision, no false-positive rate, no F1.** See above: the background is unlabelled.
- **No headline accuracy claim of any kind.** Accuracy belongs at L1/L2 where labels are dense.
  This source exists for state, memory, cost, and re-arm.
- **No wall-clock, sunrise, or schedule-derived ground truth.** There is no wall clock.
- **No audio.** The encode profile drops it (`audio: false`); the audio path is
  `eval/stage-validation/build_audio_corpus.py`.
- **It is not a replacement for a real recording.** It is the thing you build *because* you have
  not recorded 72 hours yet, and the thing you keep afterwards because it is the only frame-exact
  ground truth at that horizon.

## Determinism

A build is a pure function of `--seed` and the arguments recorded in `build`. Selection is a seeded
shuffle over a **sorted** list, never filesystem order; schedule jitter comes from the same seeded
generator; nothing reads the wall clock. The single exception is `generated_utc`, recorded for
provenance and excluded from any comparison — pass `--no-timestamp` for a byte-identical rebuild.

## Refusals

The builder refuses rather than degrades. It aborts on a schedule collision (reporting the pair,
both offsets, both durations and the shortfall — it never silently shifts a clip, because a nudged
clip makes every offset in the manifest a lie), on an infeasible schedule, on a concat whose pixels
do not change across the clock, and — under `--require-redistributable` — on any non-redistributable
contributor, background included.
