# eval — two tracks

The golden corpus measures two different things (spec §9.1), so it lives in two places.

| Dir | What | Redistributable? | Tracked in git |
|---|---|---|---|
| `fixtures-private/` | The hand-assembled dev fixture set (currently YouTube-sourced). Drives local L2 eval and iteration. | **No** — third-party YouTube, envelope-conflicting domains in places. Private fixture only. | Labels only; pixels git-ignored. |
| `bench-staging/` | The real public benchmark set, to be built per spec §9.1. | **Yes** (that's the point). | Everything, once it exists. |

## Why the split

The spec is explicit (§9.1, Principle 8): the published benchmark must **avoid YouTube** — wrong
domain (produced/third-person vs first-person webcam) and not redistributable. The current set under
`fixtures-private/` violates that, so it can never feed `percept-bench`. It is still useful as a
*private* dev fixture; it is walled off here so it can't leak into a public release by accident.

## What `bench-staging/` needs (per spec §9.1)

- **Benchmark axes** (proactiveness / events / counting / latency) → existing public datasets, **no
  collection**: Ego4D, Epic-Kitchens (egocentric; action start/end = rising/falling edges),
  Charades / Charades-Ego, RepCount. Confirm each license before adopting.
- **Trust / gate axes** (the Percept-specific UX) → a small hand-built set (~30–50 clips **you
  record**, webcam/screen): "this is tea not coffee", ambiguous→ask, the deictic exemplar, the
  second-cup boundary. This is the irreplaceable part and is currently missing.

## Known issues carried over from the current fixture set

- All 48 scenes are `UNCLEARED` YouTube (every `metadata.json` flags it).
- 180/205 gate cases and 31/48 labels are `review_status: unreviewed` (Gemini-proposed).
- The held-out `test` split is almost entirely the unreviewed security-cam increment — do **not**
  publish a trust-metric scorecard off it until it has human ground truth.
- Domain skews third-person security-cam; eval3 ("Security Camera Fails") and eval6 (pool drowning)
  brush the envelope's OUT list. Prefer re-sourcing those for any trust claim.
