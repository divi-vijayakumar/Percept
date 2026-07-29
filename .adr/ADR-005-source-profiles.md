# ADR-005: Source profiles — one gate, many worlds

- **Status:** Accepted (the implementation; ships code, docs and tests — additive, no contract change)
- **Depends on:** ADR-001 (the Gate/judgment contract — `Verdict -> GateDecision -> JudgmentEvent`,
  **unchanged** by this ADR, and deliberately so: §Decision 2 is entirely about not changing it);
  `percept/watch.py` (the frozen public entrypoint, `WATCH_VERSION = "1.0.0"` — untouched).
- **Related:** ADR-002 (Runtime health & coverage — an untuned admission signal is reported as a
  *doctor finding*, not as a runtime health alarm; a dry run knows nothing about a declared coverage
  expectation); ADR-003 (Evidence artifacts — `evidence_strategy` names *which* artifacts a source
  class packages; ADR-003's retention/`evidence_ref` rules are proposed, not shipped, and would apply to all of them);
  ADR-006 (Incident relations — §5 shows percept's identity model is a *physical-object* model; this
  ADR is the mechanism by which a non-physical source class can say so, via `identity_strategy`).
- **Constrains:** **the implementation** — see §Consequences. The existing `WatchProfile` ontology is
  camera-shaped and cannot express a non-camera source; L1 must not be built on it.
- **Source (read for this ADR):** `percept/watchprofile.py` (the `(environment, goal_type, state)`
  key `:194-197`, the camera-shaped `environment` axis `:141`), `percept/profiles_default.py`
  (`fixed_camera` / `handheld` — the whole shipped vocabulary is a camera mount),
  `percept/profile_wiring.py` (the one seam a profile resolves through), `percept/gate.py`
  (`GATE_REASONS`, `Gate.decide`, refractory/dwell), `percept/runtime/source.py`
  (`GATE_W, GATE_H = 48, 36`), `percept/harness/tier0.py` (the 8x6 `block_max` grid),
  `percept/runtime/worker.py` (`_capture_plan` `:730`, the `coast_admit` branch `:1774-1808`,
  `bank.push(t, motion=..., frame=judge_frame, ...)` `:1764`), `percept/doctor.py`,
  `percept/eval/differential.py`, `percept/edge/detectors.py` (`register_detector` —
  the PUBLIC path, the one `percept/contracts/__init__.py:1227` documents; it is a re-export shim
  over the implementation in `percept/harness/detectors.py:472`).

## Context

Domain probes against percept 0.7.0 returned **not viable**.
They did not fail on perception: the judges scored recall 1.00 on documents and handled retractions,
counterfactuals and speaker attribution correctly. They failed because the SDK was **silent, blind or
unobservable** on a class of content it was never calibrated for and never said so.

### What was measured

`percept-doctor` on the measured screen fixture
`probes/01-screen-walkthrough/data/clean.mp4` — a 36 s accounting screen recording, 180 frames,
1920x1080, containing a **hard cut between two entirely different documents**:

| stat | max | median | its own threshold | admitted |
|---|---|---|---|---|
| `mean` | 0.0163 | 0.0000 | 0.100 | **0** |
| `block_max` | 0.0919 | 0.0001 | 0.200 | **0** |
| `p95` | 0.0743 | 0.0000 | 0.150 | **0** |

No salience statistic admits a single frame — **including across the document cut**. `block_max`
amplifies the localized signal **5.6x** over the whole-frame `mean`, exactly as it was designed to, and
still lands at **under half** its own calibrated cutoff.

### The diagnosis

This is not a statistic-selection problem. The tier-0 gate frame is **48x36 greyscale**
(`runtime/source.py`, `GATE_W, GATE_H = 48, 36`) and `block_max` reads an **8x6 grid** over it, so one
block is roughly **6x6 px of a downsampled 1080p screen**. A document-sized edit is diluted below every
cutoff before any statistic sees it.

**The eye is under-resolved for screen content.** Lowering a threshold to make this fixture admit is
not an option: the same `EdgeSettings` cutoffs carry physical-scene precision, so a cutoff picked to
light up one screen fixture would degrade every camera deployment *and* would mostly buy admissions on
compression noise. That is tune-to-green, which this project refuses everywhere else.

The honest conclusion is that screen content needs a **different admission signal** — a perceptual
hash / page-change detector, or the client's own OCR model used as an admission trigger. Percept's job
is to make that **pluggable and measurable**, not to guess it.

### Why `WatchProfile` cannot express this

`WatchProfile` is the existing answer to "what tuning does this deployment need?" and
it is **camera-shaped by construction**. Its registry key is
`(environment, goal_type, state)`, where:

- `environment` is literally a camera mount — the entire shipped vocabulary is `fixed_camera`,
  `handheld`, `wearable` (`profiles_default.py`);
- `goal_type` is the goal's temporality (`stative` / `dynamic` / `periodic`);
- `state` is start-anchored vs transition.

There is no axis on which "this stream is a screen recording" or "this stream is a conversation" can be
said. Worse, every knob a `WatchProfile` carries (`salience_stat`, `stat_thresholds`, `plan_mode`,
`cadence`) *presupposes* the tier-0 motion eye is the admission signal — the exact assumption that
fails here. Adding a fourth axis to `WatchProfile` would make a camera ontology pretend to be a
universal one.

## Decision

### 1. `SourceProfile` is a new, orthogonal axis — and an OPEN registry, never a closed enum

```python
@dataclass(frozen=True)
class SourceProfile:
    name: str
    modalities: frozenset[str]
    temporal_unit: str
    admission_strategy: str
    evidence_strategy: str
    identity_strategy: str
    default_timebase: str
    parameters: Mapping[str, object]
    stability: str          # "stable" | "experimental"
    notes: str
```

A `SourceProfile` answers *what kind of world is on the other end of the stream*, and selects — and
selects **only**:

- **admission** — which cheap signal decides a frame is worth judging;
- **sampling** — the temporal unit and the coast;
- **evidence packaging** — what the judge and the incident carry;
- **identity strategy** — how two observations are decided to be the same thing;
- **episode segmentation** — carried in `parameters["episode_unit"]`;
- **default temporal sense** — `default_timebase`.

Every field is free text. `SourceProfileRegistry` is keyed by `name`, and a third party ships
`industrial_line` / `medical_imaging` / `satellite_change` / `code_editor` / `financial_dashboard` by
constructing a `SourceProfile` and calling `register_source_profile(...)` (or advertising it under the
`percept.source_profiles` entry-point group). **No central literal is edited.** `percept-doctor
--source-profile` takes open text for the same reason — a closed `choices=(...)` there would have
re-closed the registry through the back door.

Validation is about *shape* (non-empty names, a real timebase, immutable parameters), never about
membership in a vocabulary percept controls.

### 2. A source profile MUST NOT change gate semantics

This is the load-bearing constraint of this ADR. `FIRE` / `ASK` / `SILENT`, the rising edge,
refractory and dwell mean **the same thing on every source**. There is **one** `Gate` and **one**
incident truth — never a conversation-Gate, a document-Gate and a camera-Gate.

The reason is not tidiness. The gate is percept's trust surface (ADR-001, invariant 1): the value of
"percept fired" comes entirely from it meaning one thing. A per-source gate would make an incident's
meaning depend on which profile happened to be active, which is exactly the class of bug that cannot be
debugged from the outside at 2am.

This is enforced **structurally**, not by convention:

- `SourceProfile` has **no gate field at all**;
- `FORBIDDEN_FIELDS` (`gate`, `require_false_first`, `refractory_s`, `dwell_s`, `fire`, `ask`,
  `silent`, `rising_edge`, `gate_action`) may not appear in `parameters` either — construction raises
  `SourceProfileError` if they do;
- `tests/test_sourceprofile.py` asserts both, and asserts the exact field set, so a future field that
  smuggles gate semantics in fails the suite rather than review.

An untuned admission signal likewise **does not invent an exit code**: zero admits is exit 2 (zero
coverage) on a screen source exactly as on a camera.

### 3. Three profiles ship; one is EXPERIMENTAL and says so in a typed field

| profile | stability | admission | identity | timebase | temporal unit |
|---|---|---|---|---|---|
| `physical_scene` | stable | `tier0_motion_salience` (**calibrated**) | `track_iou_label` | `auto` | frame |
| `screen_document` | stable | `page_change_signal_or_coast` (**UNTUNED**) | `content_key` | `media` | page |
| `conversation` | **experimental** | `utterance_boundary` | `speaker_label` | `wall` | turn |

`physical_scene` is the honest default when a caller names no profile — it is the one class the stock
eye is calibrated on (the measured `block_max = 0.20` ladder), so a client on screen content is running the wrong
profile until they say otherwise, and doctor labels the choice `[default]` so they can see it.

`conversation` ships **experimental** — `stability="experimental"`, not merely a caveat in prose,
because prose is not readable by a client's CI. Its two measured problems are recorded in
`parameters["experimental_reasons"]`: (1) **speaker identity is off-contract** — there is no
diarization behind `identity_strategy="speaker_label"` in this build, so two speakers can collapse to
one entity; (2) goals judge **serially at ~24 s/turn measured**, so a live conversation outruns the
watch. It exists so the shape can be evaluated and so nobody has to guess whether percept "supports
conversation": it does not, yet.

### 4. `screen_document` fails safe rather than looking good

Its defaults were chosen against the measurement above, not against making a fixture pass:

- **A fixed low-rate coast, and it is not optional.** `coast_interval_s = 2.0` (0.5 fps),
  `coast_required = True`. A static-but-important page is still sampled even when no change signal
  fires. This is stated for what it is: a **floor chosen to fail safe**, not a tuned value and not a
  claim about anyone's content. It is the honest operating mode when the change signal is unavailable
  or untuned, and it rides the seam that already exists — `coast_sense_prior(profile)` returns the
  `sense` prior that `_capture_plan` turns into a real `SamplingController`, whose coast forces a judge
  on a quiet, sub-threshold frame the motion gate would skip.
- **A pluggable page-change admission signal that DEFAULTS TO DECLARING ITSELF UNTUNED.**
  `parameters["change_signal"] == UNTUNED`, so `profile.admission_tuned is False` and doctor emits
  `ADMISSION_SIGNAL_UNTUNED`: *"source profile 'screen_document' is running on COAST ALONE — supply a
  change signal."* The alternative — shipping a plausible-looking cutoff — would under-admit silently,
  which is precisely the failure that produced two not-viable verdicts.
- **`default_timebase = "media"`** — screen sources are overwhelmingly recorded files, whose `t` must
  be stamped on the media timeline.
- **Identity is a content/document key, NEVER geometry alone.** IoU + label cannot separate `INV-2291`
  from `INV-2294`: two invoices in the same screen region with the same label are the *same entity by
  construction* under a geometric strategy. (This is the concrete instance of ADR-006 §5's finding that
  percept's identity model is a physical-object model.)
- **Evidence = full screenshot + changed-region crop + optional previous screenshot**
  (`evidence_parts`, with `previous_screenshot` in `evidence_optional_parts`) — a page diff is
  unreadable without the before.
- **Episode = the page/document interval** (`episode_unit = "page_interval"`).

### 5. The tuning loop is a deliverable

`docs/admission/tuning-admission.md` ships with this implementation: a worked, runnable guide from "nothing fires" to
a tuned eye, using the tools that now exist — `percept-doctor` (the per-stat table, and which rule
suppressed each non-admit, for zero judge spend), `percept.edge.detectors.register_detector` (the client's own phash/OCR model
as an **admission** detector, running at native fps on the **full-resolution judge frame**, emitting
boundaries with no cloud judge call), and `percept eval` (the implementation — a paired A/B of their detector
against baseline on **their** fixtures with **their** bars, on a held-out split). The `clean.mp4`
numbers above are its worked example of reading a doctor report correctly, including the explicit
instruction *not* to lower the threshold.

### 6. Doctor reports the active source profile

The slot that used to read *"not available — the implementation not present in this build"* now carries the real
profile: name, origin (`explicit` / `default`), stability, modalities, temporal unit, all four
strategies, coast and timebase, plus the full `SourceProfile.to_dict()` under
`plan.source_profile.profile` in `--json`. Two new findings: `ADMISSION_SIGNAL_UNTUNED` (warn) and
`SOURCE_PROFILE_EXPERIMENTAL` (warn).

## Honesty statement

Stated plainly, here and in the how-to, because papering over it causes avoidable deployment
debugging:

> **Percept's stock eye does not admit on screen/document content**, and the reason is that the tier-0
> gate frame is 48x36 greyscale — the eye is under-resolved for this class, not mis-tuned. Percept ships
> **no** calibrated admission signal for it, and will not ship one it has not measured on the client's
> content. **The client supplies and tunes the admission signal.** Percept supplies the harness: the
> pluggable seam, the zero-cost diagnosis, the honest coast floor, and the A/B that proves whether the
> new signal is actually better.

## Consequences

**Good**

- A client on a non-camera source finds out *from the tool*, in one zero-cost command, that they are
  running an untuned profile — instead of concluding percept is broken.
- Third-party source classes are first-class without a fork, and without percept pretending to have
  calibrated them.
- Gate semantics stay singular, so an incident means one thing on every source — enforced by tests, not
  by review.
- No threshold moved. Physical-scene precision is byte-for-byte what it was.

**Costs / accepted risks**

- `screen_document` ships **knowingly degraded** (coast-only) until a client registers a signal. This is
  a deliberate choice of a loud floor over a quiet guess.
- The coast costs judge calls on a static page. `coast_interval_s` is the client's knob, and the
  shipped 2.0 s is a floor, not a recommendation.
- `SourceProfile` is currently read by `percept-doctor` and by the client's own wiring
  (`coast_sense_prior`, `with_parameters`); it is **not yet** resolved automatically inside
  `runtime/worker.py`. That wiring is deliberately a separate change — `watch()`'s signature and
  `WATCH_VERSION` are frozen, and the profile implementation is additive behind existing seams.
- `conversation` will disappoint anyone who reads the name without the `stability` field. That is why
  the field is typed and why doctor emits `SOURCE_PROFILE_EXPERIMENTAL`.

**This ADR constrains the implementation.**

L1 must **not** be built on the `WatchProfile` ontology. That ontology is camera-shaped: its
`environment` axis is a camera mount (`fixed_camera` | `handheld` | `wearable`), and its three axes are
goal-temporality / camera-mount / start-state. It cannot express a non-camera source at all, and every
knob it carries assumes tier-0 motion is the admission signal. L1 should compose the two axes —
`SourceProfile` (what kind of world) x `WatchProfile` (how to tune the eye *within* that world) — and
must not extend `WatchProfile.environment` with source-class values (`screen`, `conversation`), which
would overload a camera-mount axis with a category it cannot represent and would silently break the
wildcard fallback chain (`(env, goal, "*")` -> `(env, "*", "*")` -> `("*", "*", "*")`).

## Open questions

1. **Automatic runtime resolution.** Should `run()` / `watch()` resolve a `SourceProfile` and apply its
   coast, evidence and identity strategies automatically, or should that stay explicit client wiring?
   Deferred: it touches frozen surfaces and requires its own compatibility decision.
2. **Should `admission_strategy` become a registry of callables** rather than a name plus a separately
   registered detector? Today the strategy is a *name* and the implementation is a registered detector;
   collapsing them is tempting but would couple the profile to the detector lifecycle.
3. **A resolution/inference layer** — should a source URI (`screen://`, `.mp4`, `mic:`) *suggest* a
   source profile? Attractive at 2am, dangerous silently: a wrong inference is worse than an explicit
   `[default]` label the client can see and correct.
4. **Whether `screen_document` can ever ship tuned.** It could, if percept measured a phash detector
   across a broad corpus of client content it does not have. Until then, untuned is the true state.
