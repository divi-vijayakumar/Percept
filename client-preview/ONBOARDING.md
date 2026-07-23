# Percept — Client Preview Onboarding

**Wheel:** `percept_vision-0.8.1-py3-none-any.whl` · **Watch API:** 1.0.0 · **Contract:** 1.9.0

This is a **local preview wheel** for early feedback, not a public release. Everything below is
verified against the code in this build — every code sample runs as written.

---

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install percept_vision-0.8.1-py3-none-any.whl

# optional extras
pip install "percept_vision-0.8.1-py3-none-any.whl[ocr]"   # OCR evidence extractor (pytesseract + Pillow)
```

Every wheel is provenance-stamped. Confirm exactly what you're running:

```python
import percept
print(percept.build_info())
# BuildInfo(package_version='0.8.1', git_commit='<git sha>…', contract_version='1.9.0', build_timestamp='2026-07-23T…Z')
```

A returned `.ptrace` (see §3) carries this same stamp, so any trace you send back ties to the exact build.

---

## 2. The architecture in one screen

You give Percept **two things**, it resolves them into one immutable plan, then runs it:

```
  Runtime configuration                         What to watch
  ─────────────────────                         ─────────────
  SourceProfile    (what kind of stream)        WatchPlan
  DomainPack       (vocabulary + templates)       condition: "a new document appears"
  ExecutionPolicy  (cost / cadence posture)        + optional sense/admission/evidence/
  DeploymentConfig (providers, env, infra)           judge/consequence/rearm/schedule
  [OptimizationLock, when you have one]              + explicit tunable overrides
                          │                               │
                          └──────────────┬────────────────┘
                                         ▼
                              ConfigResolver
                    validates + resolves with per-field provenance
                                         ▼
                              ResolvedWatch   ← .explain() tells you where every value came from
                                         ▼
                              RuntimeCompiler → the runtime (unchanged wire form)
```

**Why this matters:** configuration is ownership-gated and precedence-ordered
(`SDK < SourceProfile < DomainPack < ExecutionPolicy < OptimizationLock < Deployment(env) < WatchPlan`).
A screen-watch deployment no longer silently admits nothing until someone hand-exports an env var — you
select `screen_document` explicitly and the admission tuning comes with it. When an env var overrides a
value your explicit profile owns, resolution still succeeds but emits a **loud diagnostic** you can see.

The three source profiles that ship: **`physical_scene`** (a camera on the world), **`screen_document`**
(a screen/document capture), **`conversation`**. The five execution policies: **`balanced@1`**,
**`burst@1`**, **`low_cost@1`**, **`privacy_edge@1`**, **`strict_admission@1`** (signal-only admission —
a `SourceProfile`-declared coast floor is NOT bound into the runtime; judge calls are spent only on
tuned-signal escalations and audio wakes. If the profile marks its coast `coast_required`, `strict_admission`
still overrides it, but LOUDLY — a `coast_required_overridden` receipt warning, never a silent drop. Good
default for a blank/static-input admission control where you never want to pay a judge call for "nothing
changed").

---

## 3. Input and output

### Input — a `WatchPlan`

```python
from percept.contracts.watchplan import WatchPlan

plan = WatchPlan(
    condition="a person leaves a package by the door",   # verbatim natural language — never compiled away
    source="camera",
    # every block below is optional; omit to accept the profile/pack defaults:
    # sense=…, target=…, admission=…, evidence=…, detectors=…, extractors=…,
    # judge=…, consequence=…, rearm=…, uncertainty=…, schedule=…,
    explicit={                                            # typed overrides, keyed by tunable path
        "WatchProfile.salience_stat": "block_max",
    },
)
```

### Output — three surfaces

**(a) The event stream** — `run()`/`watch()` yield rising-edge decisions only (FIRE/ASK; the gate
suppresses everything else):

```python
import percept
from percept import Percept
from percept.watchplan_apply import apply_watch_plan
from percept.source_profiles_default import PHYSICAL_SCENE

core = Percept.configure(vision="gemini", mode="live", source_profile=PHYSICAL_SCENE)
receipt = apply_watch_plan(core, plan, source_profile=PHYSICAL_SCENE)   # -> WatchReceipt

async for incident in percept.run(core, source="driveway.mp4", fps=1.0):
    print(incident.message, incident.verdict.confidence)
```

**(b) A webhook** — attach a sink; Percept POSTs the incident JSON on every fire:

```python
from percept.runtime.sinks import webhook, UrlPolicy

sink = webhook("https://your-host/percept-hook")               # https + public host enforced (SSRF-safe)
# sink = webhook("http://localhost:8080/hook", url_policy=UrlPolicy.dev())   # local dev

async for _ in percept.run(core, source="driveway.mp4", on_incident=sink, on_ask=sink):
    pass
```

The POST body is the **Incident envelope (contract 1.9.0)**:

| field | meaning |
|---|---|
| `id`, `concern_id`, `ts` | incident identity + which watch fired, when |
| `verdict` | `{satisfied, confidence, evidence}` — the judge's answer + its reasoning |
| `reason`, `roi`, `modality`, `provenance` | why it fired, where in frame, vision/audio/fused, which components |
| `entity_id` | subject correlation when identity tracking is on |
| `evidence_seq` | **dereferences to the exact source frame** in the `.ptrace` that produced this fire |
| `judgment_event_id` | opaque, injective `f"{goal}#{entity}#{seq}"`-derived handle — the ONE `judgment.event` trace row this Incident resolves to (`percept.contracts.judgment_event_id`; verify with `percept.replay.verify_evidence_links`). `None` when the judgment's timing stamp failed to freeze — links are absent, never fabricated. |
| `evidence_origin` | where the evidence came from: `"watch"` \| `"transition"` \| `None`. The A→B transition path (§5's `state={"type": "transition", ...}`) now emits a real, resolvable judgment too — pre-1.9 it emitted none at all. |
| `revision_id`, `supersedes` | the correction chain — a later fire that revises an earlier provisional one names it in `supersedes`, never a silent edit |

**(c) Evaluation artifacts** — a `.ptrace` bundle (every judgment round-trips to its evidence frame;
manifest carries the build stamp + resolved object identities), plus frontier metrics and an
`OptimizationLock` (see §6). Pass `on_trace=` / a recorder to capture the `.ptrace`.

### The receipt

`apply_watch_plan` returns a `WatchReceipt` — inspect it before you trust a run:

```python
print(receipt.status)          # applied | applied_with_warnings | partial | rejected
print(receipt.watch_ids)       # the armed goal ids
print(receipt.dropped_fields)  # tuple of FieldDiagnostic {path, code, goal_id} — structured, never prose-only
print(receipt.coerced_fields, receipt.warnings)
```

### Corrections & revision chains

Some watches track a **claimed value a person can later correct** — "the approved budget is $40K …
actually, make that $25K". Arm the concern in **claim-slot** mode with one explicit key:

```python
plan = WatchPlan(
    condition="a buyer states the approved budget",
    explicit={"Goal.state.identity_policy": "claim_slot"},
)
```

Now a later correction does not silently overwrite the earlier value — it **supersedes** it, and the
link rides out on the incident you already receive:

- the correction's incident carries a new `revision_id` and names the value it replaced in
  `supersedes` (the original incident's `revision_id`); the webhook JSON carries both fields (see the
  Incident table above). A **retraction** ("never mind") supersedes too and clears the value.
- a **question or hypothetical** ("could we do $25K?") is *not* a correction — it never mutates the
  slot and never emits a `supersedes` link.

Honest limits, today:

- **No speaker attribution.** There is no diarization, so an incident never claims *who* spoke; a
  correction is applied on its content, not on an asserted role.
- **Only three backends parse the correction channel:** `vision="gemini" | "claude" | "openrouter"`.
  Arm `claim_slot` on any other backend and the receipt returns a warning
  (`code="claim_mode_backend_unverified"`) — the plan still arms and runs, but corrections will not
  form a revision chain, so check `receipt.warnings` before you rely on supersession.

### See where every value came from — before you run

```bash
percept resolve my_config.json --explain     # per-field provenance table: value · owner layer · why
percept config tunables                      # every tunable path, owner, and whether it's safe to optimize
percept components list                      # every selectable component (profiles, policies, packs, judges, extractors)
percept doctor --source clip.mp4 --goal "…"  # dry-run the tier-0 gate, no judge spend, zero-admit alarm
```

---

## 4. Sample call — end to end, keyless

This runs the **real runtime** with scripted verdicts (proves wiring, spends nothing):

```python
import asyncio, percept

fixture = percept.load_fixture(None)                 # bundled deterministic fixture, ships in the wheel
core    = percept.offline_core(fixture)
goals   = percept.offline_goals(fixture)

async def main():
    async for ev in percept.watch(percept.offline_frames(fixture), goals,
                                   percept=core, source="fixture"):
        print(ev)
asyncio.run(main())
```

For a **real judge**, set a key and pick a backend:

```python
# export GEMINI_API_KEY=…   (or ANTHROPIC_API_KEY / OPENROUTER_API_KEY)
core = percept.Percept.configure(vision="gemini", mode="live",
                                 source_profile=percept.source_profiles_default.SCREEN_DOCUMENT)
```

Backends: **`gemini`** and **`claude`** carry vision + conductor + specialist adapters; **`openrouter`**
is vision-only. Only `gemini` reads audio today (see §7); `claude`/`openrouter` accept the parameter and
honestly degrade to vision.

---

## 5. Build a completely custom pack

A **DomainPack** is your vocabulary as data: named goal templates, recommended components, and hard
policy clamps. Templates never auto-arm — `watch_template()` returns a `WatchPlan` you arm explicitly.

```python
from percept.domain_pack import (
    DomainPack, GoalTemplate, ComponentRecommendation, DirectRouting,
    register_domain_pack, watch_template,
)

# templates are a mapping keyed by name; behavior carries the sense ("type"), costs live in consequence:
_TEMPLATES = {t.name: t for t in (
    GoalTemplate(
        name="spill_on_floor",
        condition="there is a liquid spill on the floor",
        behavior={"type": "transition", "from": "dry", "to": "wet"},   # closed schema — unknown keys raise
        consequence={"miss_cost": "severe", "false_fire_cost": "mild"},
        parameters={"zone": "aisle"},                                   # caller-substitutable at arm time
    ),
    GoalTemplate(
        name="shelf_gap",
        condition="a shelf section is empty",
        behavior={"type": "count"},
        consequence={"miss_cost": "mild", "false_fire_cost": "trivial"},
    ),
)}

MY_PACK = DomainPack(
    id="retail_floor",
    version="1",                                       # a string, not an int
    templates=_TEMPLATES,
    components=(
        # a recommendation the resolver surfaces — distinct from a hard clamp. ref is namespaced:
        # "extractor:…", "sense:…", "identity_resolver:…", "execution_policy:…"
        ComponentRecommendation(ref="extractor:ocr", strength="preferred",
                                reason="price-tag reads improve shelf-gap precision"),
    ),
    policy_clamps={
        # ENFORCED at resolve time — a plan that violates a clamp is a ConfigError, never silently merged:
        "false_negative_cost_min": "mild",
        "retention_max_days": 7,
    },
    direct_judge_routes=(
        DirectRouting(condition_namespace="retail_floor."),
    ),
    notes="Retail floor observation — existing senses only.",
)

register_domain_pack(MY_PACK)                          # in-process; or ship it as an installable (below)

# arm a template — explicit, with parameter substitution. Ref is "<id>@<version>/<template>":
plan = watch_template("retail_floor@1/spill_on_floor", {"zone": "produce"})
receipt = apply_watch_plan(core, plan, domain_pack=MY_PACK, source_profile=PHYSICAL_SCENE)
```

**Ship it as a pip-installable pack** so anyone who installs your package gets it via discovery — add to
your `pyproject.toml` (the entry point points at your `DomainPack` object):

```toml
[project.entry-points."percept.packs"]
retail_floor = "your_module:MY_PACK"
```

Percept discovers it automatically (`percept components list` will show it). Validate anytime:

```bash
percept components describe retail-floor          # its templates, recommendations, clamps
```

The `DomainPack` fields in full: `id, version, values, templates, components, policy_clamps,
direct_judge_routes, fixtures, quality_bars, search_space, notes`. `components` are *recommendations*
(`preferred`/`required`, surfaced to the operator); `policy_clamps` are *enforced* (resolve-time errors).
Keep the two distinct — that separation is the point.

**A worked fused-compare template** (e.g. "does the spoken threshold match the visible one"): give the
template `extractors=("ocr-fields-scripted",)` (or chain `percept.extractors.fields.chain_ocr_then_fields`
yourself) so each side lands as a typed `{field, raw, normalized, kind}` fact instead of raw text, then
write `condition` to compare `normalized` values, never `raw` ones — `"25,000 dollars"` and `"threshold:
25,000 dollars"` both normalize to `"USD 25000.00"` and correctly read as equal despite differing surface
text. State the ASK rule explicitly in the condition too: *"if either side's normalized value is missing,
ASK — never report a mismatch on an unnormalizable value."*

---

## 6. Run the optimizer to get the frontier

Two related commands. **`percept optimize`** searches a declared space for the best config and writes an
`OptimizationLock`. **`percept eval --frontier`** reports the operating frontier (recall bounds, false-fires
per watch-hour, ASK rate, coverage) of a config against your held-out fixtures. You typically optimize,
then evaluate the winner's frontier.

### 6a. Author a search space (data, not code)

Only tunables marked `safe_to_optimize` may appear; gate vocabulary, goal meaning, consent, and
retention/egress are structurally forbidden — a space over them raises **before any run**. Check what's
optimizable with `percept config tunables`. A structural choice (like the salience *statistic*, which a
`SourceProfile` owns) is expressed by giving each **template** its own base profile; the **search space**
then tunes the safe *numeric* knobs. An `optimizer.json`:

```json
{
  "templates": [
    {"name": "mean_baseline", "base_profile": "physical_scene",
     "source_profile": "physical_scene", "fixture_dir": "mean",
     "description": "mean-stat admission baseline"},
    {"name": "block_max", "base_profile": "screen_document",
     "source_profile": "screen_document", "fixture_dir": "block_max",
     "description": "localized-salience admission"}
  ],
  "search_space": {
    "dimensions": [
      {"path": "Settings.edge.motion_gate",           "choices": [0.10, 0.15]},
      {"path": "Settings.edge.motion_gate_block_max", "choices": [0.15, 0.20, 0.25]}
    ]
  },
  "strategy": "grid",
  "declared_at": "2026-07-22"
}
```

(`dimensions` accept `"choices": [...]` for a set, or `"low"/"high"` for a numeric range. Only
`safe_to_optimize` paths are allowed — e.g. `Settings.edge.motion_gate*`, `AcousticOnset.*`,
`Settings.cache.*`, `Settings.entity.iou_match` — a path like `Settings.edge.salience_stat`, owned by the
`SourceProfile`, is refused.)

### 6b. Optimize → a lock

```bash
percept optimize \
    --fixtures ./fixtures \
    --space   ./optimizer.json \
    --bars    ./bars.json \          # your acceptance bars (e.g. candidate_recall >= 0.99); omit to report only
    --plan    ./my_watchplan.json \  # its hash becomes the lock's base_plan_hash
    --split   test \
    --json    ./optimizer_report.json \
    --out-lock ./winner.lock.json
```

The winner reproduces deterministically: the `OptimizationLock` records `base_plan_hash`,
`catalog_component_hash`, `fixture_hash`, the selected `values`, and the `held_out_metrics`. Apply it as a
configuration layer and it re-runs identically:

```python
import json
from percept.optimization_lock import OptimizationLock
lock = OptimizationLock.from_dict(json.load(open("winner.lock.json")))
core = Percept.configure(source_profile=SCREEN_DOCUMENT, optimization_lock=lock)
```

If a later deployment/env change touches a field the lock validated, resolution fails **loudly** with a
hash mismatch — a lock is never silently outranked.

### 6c. The frontier report

```bash
percept eval \
    --baseline  ./baseline_plan.json \
    --candidate ./candidate_plan.json \
    --fixtures  ./fixtures \
    --split     test \
    --frontier \
    --frontier-bars ./operating_bars.json \   # e.g. false_fire_max per scheduled watch-hour
    --frontier-json ./frontier.json
```

`--frontier` emits the operating-point report: **one-sided 95% bounds on FIRE recall and false-FIREs per
scheduled watch-hour, ASK rate, coverage, assisted recall, degraded time, and duplicate FIREs per
episode**, per archetype, with both denominators shown. This is the honest operating curve — it's a *new*
evaluation run over your fixtures, not a replay.

**Closing the loop:** a production `.ptrace` plus a human labels sidecar becomes new regression fixtures —

```bash
percept eval ingest-trace run.ptrace --labels labels.json --out ./fixtures/
```

— so real misses and false-fires from the field feed straight back into the next optimize/frontier cycle.

---

## 7. Audio (fused perception)

Most MP4s carry audio, and some conditions are audio-dominant ("is the appliance running"). Turn on the
fused modality and Percept demuxes the file's own soundtrack into the judge:

```python
plan = WatchPlan(
    condition="the hairdryer is on",
    explicit={
        "Goal.state.perception.modality": "fused",
        "Goal.state.perception.audio_s": 4.0,      # seconds of audio per judgment
    },
)
# run() auto-demuxes the container's audio when a fused goal is armed:
async for inc in percept.run(core, source="clip.mp4", fps=1.0,
                             audio_wake={"high": 0.02, "low": 0.01, "refractory_s": 5.0},
                             audio_settle_s=2.0):
    ...
```

`audio_wake` tunes the acoustic-onset gate (real-world appliance audio peaks lower than the SDK default —
tune per environment); `audio_settle_s` delays the woken judgment so the clip contains the sound rather
than the silence before it. Audio is judged by **`gemini`** today; other backends degrade to vision.

---

## 8. What to send back

- Your `.ptrace` files (they carry the build stamp — we can reproduce your exact run).
- The `percept resolve --explain` output when a config behaves unexpectedly (it names the owner of every
  value).
- Frontier JSON from §6c against your own footage — that's the number that decides operating points.

Questions, or a config that resolves to something surprising: send the plan + the `--explain` and we'll
trace it to the field and layer that set it.
