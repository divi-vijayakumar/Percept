---
title: Tuning admission: from "nothing fires" to a tuned eye
updated: 2026-07-23
applies_to: percept-vision 1.0.0
status: stable
---
# Tuning admission: from "nothing fires" to a tuned eye

**Read this first.** Percept ships the harness. **You tune the eye for your content.**

Percept's stock tier-0 eye is calibrated on **one** source class: physical scenes seen through a
camera. On screen recordings, documents, dashboards, medical imagery or a factory line, the stock eye
will frequently admit **nothing** — and percept says so instead of pretending. That is not a bug you
should work around by lowering a threshold; those thresholds carry physical-scene precision. It is a
signal that this source class needs **your** admission signal.

This page is the loop that gets you there, using only tools that exist today:

| step | tool | cost |
|---|---|---|
| 1. See exactly why nothing fired | `percept-doctor` | **zero** judge calls |
| 2. Name the source class | `--source-profile` (ADR-005) | zero |
| 3. Turn on the honest floor (the coast) | `sense.base_fps` prior | sampling only |
| 4. Plug in **your** change signal | `register_detector(...)` | your model's cost |
| 5. Prove it beats the baseline on **your** bars | `percept eval` | one paired A/B |

---

## Step 1 — read the doctor report correctly

```bash
percept-doctor --source probes/01-screen-walkthrough/data/clean.mp4 \
               --goal "the document being edited changes" \
               --fps 5
```

A worked report on a screen-recording fixture — a 36 s accounting screen capture, 180 frames,
1920x1080, containing a **hard cut between two different documents** (doctor's SALIENCE STATS panel,
condensed to the columns that matter here):

```
    stat             max    median   threshold  admitted
    mean          0.0163    0.0000       0.100         0   <- active
    block_max     0.0919    0.0001       0.200         0
    p95           0.0743    0.0000       0.150         0
```

### How to read this

**Do not read it as "the clip is quiet."** It is not — a whole document was replaced mid-clip.

1. **Every stat is compared to its own threshold.** `mean`, `block_max` and `p95` live on different
   value scales, so a single cutoff would be meaningless. Doctor prints the effective threshold beside
   each value for exactly this reason.
2. **`block_max` did its job.** `0.0919 / 0.0163 = 5.6x` amplification over the whole-frame `mean` —
   localized salience is working exactly as designed.
3. **And it still lands at under half its own cutoff** (`0.0919` vs `0.200`). Every alternative
   statistic also admits zero, so the `ALTERNATIVE_STAT_WOULD_ADMIT` finding is *absent* — there is no
   better statistic to switch to.

> **The diagnosis: the eye is under-resolved for this content, not mis-selected.**
> The tier-0 gate frame is **48x36 greyscale** (`percept/runtime/source.py`, `GATE_W, GATE_H = 48, 36`)
> and `block_max` reads an **8x6 grid** over it — one block is about **6x6 px of a downsampled 1080p
> screen**. A document-sized edit is diluted away before any statistic gets to see it.

**Therefore: do not lower the threshold.** Dropping `motion_gate_block_max` from 0.20 to ~0.08 to make
this fixture admit would also drop it for every camera deployment sharing those settings, and would buy
you admissions on compression noise rather than on document changes. That is tune-to-green: it trades
real precision for a fixture that looks better.

The rule of thumb doctor is built around:

| what you see | what it means |
|---|---|
| the active stat admits 0, another stat **would** admit (`ALTERNATIVE_STAT_WOULD_ADMIT`) | the eye is **mis-selected** — switch `salience_stat` |
| **all** stats admit 0 and the max is close to a cutoff | genuinely quiet, or a threshold worth measuring |
| **all** stats admit 0 and the max is a fraction of every cutoff, *on content you know changed* | the eye is **under-resolved** — you need a different admission signal (this page) |
| `suppressions.signal_unavailable > 0` | the goal gates on a channel nothing feeds — fix the plan, not the eye |
| `suppressions.out_of_schedule` / `cadence_not_elapsed` | the gate never even looked; the eye is not the problem |

---

## Step 2 — name your source class

`SourceProfile` (ADR-005) is what tells percept — and doctor — what kind of world is on the other end
of the stream:

```bash
percept-doctor --source probes/01-screen-walkthrough/data/clean.mp4 \
               --goal "the document being edited changes" \
               --source-profile screen_document
```

```
  source profile          screen_document   [explicit]
    stability             stable
    modalities/unit       text, video / page
    admission strategy    page_change_signal_or_coast   ** UNTUNED **
    evidence / identity   screenshot_plus_changed_region / content_key
    coast / timebase      every 2s / media
...
  [WARN ] ADMISSION_SIGNAL_UNTUNED
      source profile 'screen_document' is running on COAST ALONE — supply a change
      signal. ...
```

`screen_document` **declares its admission signal untuned**. Percept does not bundle a page-change
model, a perceptual hash or an OCR model, and it will not ship a cutoff it never measured on your
content. The `ADMISSION_SIGNAL_UNTUNED` finding is the honest statement of that, and it is a **warn**,
not an error: coasting is a real, degraded operating mode, and it does not change the exit-code
contract (zero admits is still exit 2, as on any source).

A source profile changes admission, sampling, evidence packaging, identity, episode segmentation and
the default timebase. It **never** changes gate semantics — `FIRE`/`ASK`/`SILENT`, the rising edge,
refractory and dwell mean the same thing on every source (see
[`../reference/consequence-gate.md`](../watches/the-gate.md) and ADR-005 Decision 2).

If your source class is not one of the three shipped ones, register it — the registry is **open**:

```python
from percept.sourceprofile import SourceProfile, register_source_profile

register_source_profile(SourceProfile(
    name="financial_dashboard",
    modalities=frozenset({"video", "text"}),
    temporal_unit="refresh",
    admission_strategy="tile_value_delta",
    evidence_strategy="screenshot_plus_changed_region",
    identity_strategy="content_key",     # never geometry alone — see below
    default_timebase="wall",
    parameters={"coast_interval_s": 5.0, "episode_unit": "trading_session"},
))
```

`percept-doctor --source-profile financial_dashboard` accepts it immediately — that CLI flag is open
text, not a closed choice list, and a typo is a clear exit-5 error rather than a silent fallback to the
wrong source class.

### Why screen identity must be a content key

`screen_document.identity_strategy` is `content_key`, never geometry. IoU + label **cannot** separate
invoice `INV-2291` from invoice `INV-2294`: two invoices rendered in the same screen region carrying
the same label are the same entity *by construction* under a geometric identity strategy. On screen
content, identity has to key on what the thing **is** (document id / page key) — another reason your
OCR/extraction model is load-bearing here and a bounding-box tracker is not.

---

## Step 3 — turn on the coast (the honest floor, and it is not optional)

Before you have a change signal, the only thing standing between you and total blindness is a **fixed
low-rate coast**: sample the page every N seconds *whether or not anything appears to change*. A
static-but-important page is still worth looking at.

The coast rides the sampling seam percept already has (`_capture_plan` → `SamplingController` → the
worker's `coast_admit` branch, which forces a judge on a quiet, sub-threshold frame the motion gate
would skip):

```python
from percept.sourceprofile import coast_sense_prior, get_source_profile
from percept.adapters import Goal

screen = get_source_profile("screen_document")
goal = Goal(
    id="doc-change",
    condition="the document being edited changes",
    say="the document on screen changed",
    state={"sense": coast_sense_prior(screen)},   # -> {"base_fps": 0.5, "max_fps": 0.5}
)
```

`screen_document`'s shipped coast is **2.0 s (0.5 fps)**. Be clear about what that number is: a floor
chosen to **fail safe**, not a tuned value, and not a claim about your content. Tune it against your
own page-dwell distribution:

```python
screen = screen.with_parameters(coast_interval_s=1.0)   # your content, your floor
```

Coasting alone is **not** a solution — it is the floor that keeps you from missing everything while you
build the real signal. Judging every page at a fixed rate costs money and still misses a change that
happens and reverts between two samples.

---

## Step 4 — register YOUR admission signal

Percept does **not** bundle an OCR model, and it will not. What it gives you is the seam: a detector
you register runs at **native fps on the full-resolution judge frame** — not the 48x36 gate frame — and
each boundary it emits is fed to the brain as an occurrence with **no cloud judge call**
(`percept/runtime/worker.py`, the `bank.push(t, motion=..., frame=judge_frame, ...)` path).

A detector is three members (see [`../reference/registration.md`](detectors.md)):

```python
from percept.edge import register_detector
from percept.harness.detectors import Boundary


class PageChange:
    """A perceptual-hash page-change detector — YOUR model, YOUR threshold, YOUR content."""

    signal = "frame"          # -> receives the full-resolution judge frame, not the gate frame

    def __init__(self, distance: int = 12) -> None:
        self.distance = distance          # the ONE number you will calibrate, on your fixtures
        self._prev = None

    def reset(self) -> None:
        self._prev = None

    def push(self, t: float, frame) -> list[Boundary]:
        h = my_phash(frame)               # your phash / OCR text hash / layout embedding
        prev, self._prev = self._prev, h
        if prev is None or hamming(prev, h) < self.distance:
            return []
        return [Boundary(detector="page-change", kind="threshold", t=t,
                         value=float(hamming(prev, h)))]


register_detector(
    "page-change",
    PageChange,
    senses="a perceptual hash of the full-resolution frame",
    when_to_use="screen/document content, where tier-0 motion salience is under-resolved",
    output="one 'threshold' boundary per page change",
)
```

Select it into the goal (the selection passes the registry firewall — an unregistered name is dropped,
so you can never activate a skill that isn't there):

```python
goal.state["detectors"] = ["page-change"]
```

and record that this profile is no longer coasting blind:

```python
screen = screen.with_parameters(change_signal="page-change")
assert screen.admission_tuned is True     # doctor stops warning
```

**An OCR model plugs in the same way.** The point of `register_detector` here is that your model is
used as an *admission trigger* — cheap, at native rate, deciding **which** frames deserve the expensive
judge. You are not asking percept to understand your documents; you are telling it when to look.

> **The known limit, stated plainly.** Your detector sees the judge frame the capture path produced. If
> your content needs a resolution or colour fidelity your capture is not delivering, no registered
> detector can recover it — fix the capture first (`--fps`, source resolution) and confirm with the
> `gate frame` and `fps requested/delivered` lines in doctor's SOURCE panel.

---

## Step 5 — prove it, on YOUR fixtures, against YOUR bars

Do not trust a detector because it made one clip light up. A/B it:

```bash
percept-eval --baseline plans/coast-only.yaml \
             --candidate plans/page-change.yaml \
             --fixtures fixtures/ \
             --bars bars.yaml \
             --split test
```

- **Percept never supplies the bars.** `--bars` is client-supplied; omit it and `percept eval` reports
  the numbers without pronouncing a verdict. A bar you set after seeing the result is not a bar.
- The two arms run **paired** over the same fixtures, so the difference is attributable to the one
  thing you changed.
- Hold out a split. `--split test` is the held-out one; if you tune against it, you have measured
  nothing.
- Exit codes: `0` pass/reported, `1` a client bar failed, `2` refused (bad input or a broken split) —
  wire it into CI so a detector regression is loud.

The loop, once: `percept-doctor` → change one thing → `percept eval` → keep it or drop it. Never
"lower the threshold until green."

---

## What percept does and does not promise

**Does:**

- report exactly what admitted, what did not, and **which rule** suppressed each non-admit, for zero
  judge spend;
- give you an open `SourceProfile` registry, so your source class is a first-class citizen without a
  fork;
- run your admission signal at native fps on the full-resolution frame, with no per-frame VLM call;
- keep gate semantics identical across every source class, so an incident means the same thing
  everywhere;
- A/B your change against a baseline on your fixtures with your bars.

**Does not:**

- ship an OCR model, a perceptual hash, or a document-layout model;
- ship a calibrated admission threshold for screen/document content — **there is none, and
  `screen_document` says so**;
- guess a cutoff that happens to make one fixture pass.

---

## See also

- [`../how-to/custom-configuration.md`](../configuration/profiles.md) — author a `SourceProfile` /
  `WatchProfile` (why gate semantics are source-independent; the source-class axis is ADR-005).
- [`../reference/registration.md`](detectors.md) — the detector / concern / enricher
  registries in full.
- [`../reference/consequence-gate.md`](../watches/the-gate.md) — `FIRE`/`ASK`/`SILENT`, the
  rising edge, refractory and dwell. Unchanged by every source profile.
- [`eval.md`](../evaluation/evaluate-your-data.md) — the wider benchmark harness.
