# Benchmarks

A reproducible harness that runs Percept and the pipelines a developer would otherwise write over
the same public footage, with the same condition text and the same judge, and reports every arm,
including the ones Percept loses. No benchmark results are published with the 1.0 Developer
Preview; this directory is the method, so anyone can run it.

```bash
pip install -e "packages/percept-vision[dev]" -r benchmarks/requirements.txt   # OpenCV 4, see Arms

# Offline dry run: generated clips, an oracle judge, no key, no network (about 15 s)
python -m benchmarks.run --dataset synthetic --judge fake --out benchmarks/results/dry-run

# The E1 comparison (configs/e1.json: same judge, evidence and suppression window for every arm) on MEVA,
# with a real judge and a hard spend cap
python -m benchmarks.fetch_meva --hours 5                 # about 8.3 GB, into eval/bench-staging/meva (ignored by git)
python -m benchmarks.run --config benchmarks/configs/e1.json --judge gemini:<model> \
    --price <usd_per_mtok_in>,<usd_per_mtok_out>,<tokens_per_image> --max-cost-usd 10 \
    --out benchmarks/results/e1-meva

# Rescore a finished run from its raw results: no judge call, no key
python -m benchmarks.run --rescore benchmarks/results/e1-meva [--root <dataset root>]
```

`benchmarks/results/` and `eval/bench-staging/` are ignored by git. Footage, private labels,
keys and results are never committed.

## Datasets

| `--dataset` | footage | notes |
|---|---|---|
| `meva` | MEVA (CC BY 4.0): fixed indoor and outdoor cameras, five-minute clips | the real continuous set, and the only dataset a report can mark publishable; fetch with `fetch_meva.py` |
| `perception-test` | Perception Test (CC-BY 4.0) action-localisation validation clips | short, dense clips: diagnostics only, never a published comparison |
| `perception-test-sparse` | Perception Test clips joined with idle and distractor gaps at several occupancy levels | **synthetic**, for the cost-versus-occupancy curve only |
| `local:<dir>` | your MP4s plus `labels.json` or `labels.yaml` | private clips, evaluated locally (format in `datasets.py`) |
| `synthetic` | generated clips | for CI |

Condition text is mechanical ("`<label> begins`"). Every case also watches `--negatives` labels
that never occur in it (nor in a long-sparse case's distractor clips), so precision and false
events per hour are defined.

`fetch_meva.py` builds annotation and video URLs from the layout of the public MEVA repository
and bucket (`kitware/<date>/<hour>/` and `drops-123-r13/<date>/<hour>/`, the hour being the
clip's end or start hour). If the repository's layout changes, these URLs need updating.

## Datasets and attribution

The footage is not redistributed in this repository. Each dataset is fetched from its own source
and used under its own license.

- **MEVA** (Multiview Extended Video with Activities), by Kitware, Inc., funded by IARPA. CC BY
  4.0. `fetch_meva.py` downloads clips and annotations from the public MEVA data repository; see
  the MEVA data repository for the full terms and citation.
- **Perception Test**, by Google DeepMind. CC BY 4.0. Download its action-localisation validation
  clips and labels from the Perception Test repository into
  `eval/bench-staging/perception_test/raw/`; see that repository for the full terms and citation.

## Arms

| arm | what it does |
|---|---|
| `percept` | a `Monitor` with the product defaults (6 attempts a minute per stream; each request reserves 2 and is charged the attempts it used, so at most 5 requests a minute when each succeeds first time) **except the rearm**, which E1 standardises to 30 s (product default 120 s); fed in real time. Reports name it "percept (product defaults; rearm standardised to 30 s)" |
| `percept-matched` | the same with `max_requests_per_min` set by `--percept-budget` (default 13 attempts a minute: at most 12 requests when each succeeds first time, `chunk5`'s rate; retries and verification draw on the same budget) |
| `chunk5`, `chunk10`, `chunk30` | judge the last T seconds every T seconds, all conditions batched, then a cooldown |
| `detector_debounce` | a people (HOG) or motion (MOG2) detector with a cooldown; only conditions a detector can express |
| `detector_then_vlm` | the same detectors gate judge calls; other conditions are gated by motion |
| `percept-plan-<name>` | a `Monitor` configured by a compiled watch plan: with `--percept-plans <dir>`, one plan per source in `<dir>/<name>/`, the plan used for a case being the one whose source id is in the case id; fed in real time |
| `detector_track_rule` | a person detector, a tracker and a zone-entry rule: one event per track whose box centre enters the condition's zone, then the cooldown; no judge. Only entry conditions (`enter`, or MEVA's `person_enters_scene_through_structure`) |
| `detector_track_decide` | the same tracker entries poke a real `Monitor` (subject `track <id>`) whose judge is a free rule judge (`yes` for what the tracker poked); Percept's occurrence, verify and rearm (`--cooldown`) semantics decide; real-time pacing |

All arms decode the same frames (5 per second, longest side 640 px; checked by digest) and judge
the same evidence format (up to 4 frames). The frames each arm *judges* differ by its sampling:
the baselines keep at most 2 frames a second and spread up to 4 over the window (`chunk30`'s are
about 10 s apart), and Percept builds its clips from its pre-roll (2 frames a second over 8 s).
Each Percept case runs its own `Monitor` with the product's per-Monitor concurrency (4 requests
in flight), decodes on its own thread, and shares only the process, the event loop and the
judge's HTTP client with the other cases run at once; the manifest records this under
`monitor_config.<arm>.concurrency`. Percept runs only at `--speed 1` for published
numbers: its budget and timers use the real clock. E1 uses one suppression window for every
arm: the baseline cooldown and Percept's `rearm_s` are both 30 s (Percept's product default is
120 s). So the `percept` arm is the product defaults with one change, not the product
configuration unchanged: every Percept arm's label in the report, and its
`changed_from_product_defaults` in the manifest, name each setting that differs. The windows
start at different points: a baseline's cooldown runs from the end of the evidence of its last
decision for the condition, Percept's rearm from its decision time.

### The tracker arms

Not run by default: install `pip install -r benchmarks/requirements-detector.txt` (pinned torch,
torchvision, scipy) and name them in `--arms`; without those packages they stop with that hint.
They express one goal, entry (`enter` in a local dataset, `person_enters_scene_through_structure`
in MEVA); every other condition is reported unsupported and left out of their scores.
`--detector-plans <dir>/<name>` gives each condition the zones of the per-source plan whose source
id is in the case id (the same files a `percept-plan-<name>` arm uses). It is required on real
footage, and a plan that gives a supported condition no zone stops that case with an error: a zone
never silently becomes the whole frame. `--detector-device` picks `cpu` (default), `mps`, `cuda`
or `auto`. Both arms always play the footage in real time. For example:

    python -m benchmarks.run --dataset local:<dir> --arms detector_track_rule,detector_track_decide \
        --detector-plans <plans>/<name> --cooldown 30 --parallel 2

* Detector: torchvision `ssdlite320_mobilenet_v3_large` with its `COCO_V1` weights, run on each
  harness frame (5 a second, longest side 640 px), `person` class, score 0.5 or more (boxes from
  0.1 feed only the tracker's second association stage). torchvision's code is BSD-3. The weights
  carry no licence of their own: torchvision states that its pre-trained models "may have their
  own licenses or terms and conditions derived from the dataset used for training" and that it is
  the user's responsibility to determine permission. They are trained on COCO (annotations CC BY
  4.0, images under Flickr terms of use). Their licence is therefore not verified as permissive.
  They are downloaded from download.pytorch.org on first use; nothing is committed, and the run's
  footprint records the weights URL, the cached file's sha256 and the torch and torchvision versions.
* Tracker: `benchmarks/tracker.py`, written for this harness from the association method in the
  ByteTrack paper (Zhang et al., ECCV 2022, arXiv:2110.06864, Algorithm 1): high-score boxes are
  matched to tracks first, low-score boxes then only extend tracks seen on the previous frame. No
  code from the ByteTrack, SORT or DeepSORT repositories was used. Motion is its own (a smoothed
  per-frame velocity of the box corners, not a Kalman filter). Assignment by
  `scipy.optimize.linear_sum_assignment` (BSD-3). A track is confirmed on its second matched frame;
  one unseen for 2 s is dropped, so a person who returns later gets a new id.
* Rule (a zone-entry rule, not a test of passing through a structure: it has no direction or line
  check beyond moving inward): a track enters when its box centre goes from outside every zone to inside one, or when
  it is first seen inside a zone within 0.08 (normalised) of its edge and then moves 0.03 further
  in. Someone first seen deeper inside never enters, and each track enters at most once. A person
  who appears mid-zone (e.g. through a doorway inside the zone) is not an entry for this rule.
* In `detector_track_decide` the rearm is per subject (per track), as in Percept: two people
  entering within the rearm window are two decisions, where `detector_track_rule`'s cooldown is per
  condition. Its conditions have no trigger of their own (pokes only) and a budget of 600 attempts
  a minute, since the rule judge is free.
* Per case, `compute` records the detection and tracking CPU seconds (thread time; decoding
  excluded; GPU time not counted on `mps`/`cuda`) and wall seconds; the footprint adds
  `detector_cpu_s_per_clip_minute`. Thread time leaves out torch's own worker threads, so the
  arm's process CPU (the footprint's `cpu_s`) is the figure to report. Cost is 0: neither arm calls
  a provider.

The people detector of `detector_debounce` is OpenCV 4's built-in HOG detector run on a 320-pixel frame: a weak
detector, there to stand in for "a detector a developer has to hand", not a strong one. OpenCV 5
removed it; with OpenCV 5 installed, no condition is person-expressible and the report says so.

## What the report states

- whether the run is **publishable**, and if not, why: an oracle judge, a dataset other than
  MEVA (every such report carries a "diagnostic, not publishable" banner), `--speed` other
  than 1, a spend-cap abort, skipped cases, unintended frame loss, a Percept feed that fell
  more than a second behind real time, an arm whose judge failed on more than 5% of its looks
  (the run stops, and says why, once an arm's failed looks pass 5% of the most looks it can make:
  its rate can then only end above the limit, so a provider outage costs minutes, not the run;
  separately, 45 failed looks among an arm's last 50 (a look is one condition in one request) stop
  the run as `aborted_provider_outage`, a partial run that is never scored as complete, even if the
  provider recovers; on a short run whose arms can make fewer than 900 looks the validity stop comes
  first, and the run is marked `aborted_failed_looks`),
  or a machine that slept during the run (the footprint's `slept_s`: a clock that counts through
  sleep minus one that stops; on macOS the harness also holds `caffeinate` while it runs, but a
  closed laptop lid can still sleep it);
- how the run ended, as the manifest's `status`: `complete`, `aborted_spend_cap`,
  `aborted_failed_looks` or `aborted_provider_outage` (a run merged from parts is complete only
  when every part is);
- frames, in two separate counts. **Intentional sampling**: frames the configured policies
  skip (the harness's `--fps` sampling of the source, the Monitor's processing fps cap if set);
  expected, never disqualifying. **Unintended loss**: frames lost because decoding, a queue or
  processing couldn't keep up (decode gaps, frames replaced in the push handoff, frames pushed
  more than a second late); any of it makes the run not publishable;
- the configuration of every Percept arm, and every setting changed from the product defaults;
- recall, onset recall and the recall ceiling a perfect judge could reach under the arm's
  cooldown, precision (duplicates count against it), false events per source-hour
  (absent-condition rows included), duplicates per occurrence, onset-to-decision latency p50/p95
  with misses beside it; recall, onset recall, precision, false events per hour, cost per true
  event and latency p50 with 95% intervals;
- cost per monitored hour and per true event, from the usage each judge reports, and the
  actual spend of the run (kept, with the partial results, when the spend cap stops it);
- per arm: judge looks, failed looks by failure type (`timeout`, `malformed`, `refused`,
  `unavailable`, ...) and their rate, looks the spend cap refused (reported apart, never in the
  rate), `unknown` answers and their rate, and baseline judge calls skipped for an empty clip;
- per arm: dropped looks, split for Percept into `budget` (the request budget refused it),
  `deadline` (it waited past its deadline for a request slot or for evidence) and `verification`
  (attempts were kept for a waiting verification until its deadline passed),
  `trigger_never_fired` and `verify_rejected` misses, unverified and uncertain decisions, and the
  seconds of footage actually watched;
- every miss attributed to a stage: `trigger_never_fired`, `budget_dropped`, `judge_failed`,
  `judge_said_no`, `cooldown_absorbed`, `verify_rejected`, `decide_suppressed` (Percept's from its
  trace, which records each dropped look);
- an event survival funnel per arm: of the annotated events, how many had a candidate look, an
  admitted look, an evaluated look, a positive answer, a verification (arms that verify) and a
  delivered decision, each stage counted only if every earlier one passed;
- CPU per source-hour, the process's peak resident memory, pre-roll frames, frames capped and
  real-time lag;
- coverage: cases watched, conditions an arm can't express, and cases skipped with the reason.

## Methodology

**Matching is by the frames judged.** Every decision and every look records the case times of the
frames actually shown to the judge (a detector-only decision: the frame its detector fired on;
Percept: the frames of the delivered decision's evidence, and of each request it sent). A
decision matches an occurrence of its condition only if at least one of those frames lies within
the occurrence widened by `--tolerance` seconds, and its last frame is no earlier than half a
second before the onset. The interval the frames span earns nothing by itself: a 30-second chunk
judged from 4 frames is credited only if one of the 4 shows the occurrence. Matching is
one-to-one; a decision that shows only already-matched occurrences is a duplicate, any other
unmatched decision a false event. Looks are attributed to an occurrence by the same rule (a look
dropped before it was judged, by the clip it would have judged). A `yes` below the confidence
threshold is not a `yes` anywhere in scoring.

**Onset recall.** Conditions say "`<activity> begins`". Onset recall counts a matched occurrence
only when its matching decision showed a frame within the tolerance of the onset, so a decision
that saw the activity under way but not its beginning counts for recall, not onset recall.

**Judge failures.** Every look is counted by failure type. An arm's failed-look rate is its failed
looks over its looks, both leaving out the spend cap's refusals; above 5% in any arm the run is
not publishable, because the numbers would measure the outage rather than the pipeline. An
`unknown` answer is an answer: it counts as answered and not `yes`, and its rate is reported.

**Like-for-like comparisons.** Only footage an arm actually watched counts. Where two arms score
different case-conditions (an arm that cannot express a condition, or a case one arm skipped),
the "Where … loses" comparison rescores both on the case-conditions both score and says so. Cost
per true event is compared only where both arms score every condition of the same cases, since a
request covers every condition of a case; an arm that makes no judge calls is labelled as such.

**Uncertainty.** Recall, onset recall, precision, false events per hour, cost per true event and
latency p50 carry 95% intervals from a case-level bootstrap: cases are resampled with
replacement 1000 times and each metric recomputed from per-case tallies, with `--seed` fixing
the resampling. A comparison is listed as a loss only when the two intervals do not overlap;
otherwise it is listed under "no clear difference".

**Funnel.** Every annotated event is followed through candidate, admitted, evaluated, positive,
verified and delivered, using the same frame rule; a stage an arm does not have is marked as
not applicable. Details of every rule are in `metrics.py`.

**Rescoring.** Each run writes `raw/<arm>.jsonl`: every decision and look with its frames,
answer, failure, purpose and confidence, every dropped look with its reason, detector fires,
coverage, cost, requests and counters. `python -m benchmarks.run --rescore <out>` reloads the
cases with the run's dataset, root, limit, negatives and seed (`--root` overrides the root),
rebuilds `metrics.json` and `report.md` from `raw/`, and records `rescored_utc` and the git sha
in the manifest. A scoring fix never needs a new run, and never a new judge call.

`--max-cost-usd` covers the whole run. Each request reserves its worst case
(`percept.judges.base.worst_case_cost`, every attempt) before it is sent; a request that would
not fit stops the run cleanly, the report is written and marked aborted, and the exit code is 3.

The `fake` judge is an oracle that answers from the ground truth with injected noise
(`--oracle-fn`, `--oracle-fp`, `--oracle-latency`). It says yes when one of the sampled frames it
is shown falls within half a second before to two seconds after an onset, and a second look at
the same frames gets the same answer. Its numbers test the harness, not a model.

## Known limitations

**Dense, continuously changing scenes.** On short handheld clips the change trigger fires on
nearly every frame, so looks queue and the request budget drops them. This was diagnosed with the
oracle judge, which is a diagnostic, not a result. Handheld footage is outside Percept's
envelope, which is static cameras; that is why MEVA, with fixed cameras, is the primary public
dataset.

**Small, distant or brief activity.** In busy surveillance scenes the change trigger often does
not fire on activity that is small, far from the camera or brief, so it is never judged. The
report attributes such misses to `trigger_never_fired`. Conditions like these need a heartbeat
(`every`), a tighter zone or a lower `on_change` threshold; see `docs/concepts.md`.

Numbers come only from a full run's report, together with its coverage and configuration.
