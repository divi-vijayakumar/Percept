# SV-0.7 … SV-0.10 — Stage-0 PLANNER/COMPILER validation

**Runtime under test:** `percept 1.0.0` from
`/private/tmp/percept-launch/oss-final-venv-bba572a/lib/python3.12/site-packages/percept/`
(the launch wheel — *not* the repo's `packages/percept-vision`).

**Harness:** `eval/stage-validation/sv0_compile.py` · **Machine-readable:** `results/sv0_compile.json`
**Corpus:** `eval/stage-validation/corpus.json` — 100 real cases (40 duration, 28 transition, 28 count, 4 state).
**Network:** none. Compilation is entirely offline; the arm boundary runs on `Percept.create(mode="offline")`.

Source line references below point at
`/private/tmp/percept-launch/bounded-judges/packages/percept-vision/percept/…`.

| Test | Result | Headline |
|---|---|---|
| SV-0.7 Compile fidelity corpus | **FAIL** | default-path fidelity **4/100**; the compiled artifact is byte-identical for all 100 goals |
| SV-0.8 Determinism + hash coverage | **FAIL** | determinism clean; `plan_hash` covers 45/45 authored `state.*`, but **neither public hash covers profile- or deployment-supplied config** |
| SV-0.9 Effective-config inspectability | **FAIL** | Settings 39/39 attributed; **16 compiled `Goal.state` values across 4 scenarios carry no provenance entry** |
| SV-0.10 Unsupported-goal honesty | **FAIL** | **11/11 out-of-envelope goals compiled and armed silently.** Zero fail-closed. |

---

## SV-0.7 — Compile fidelity corpus

Two paths were measured for every one of the 100 corpus cases.

* **DEFAULT** — `WatchPlan(condition=goal_text)`. This is what a client who types the condition actually gets.
* **AUTHORED** — the caller additionally hand-authors the `Goal.state.*` explicit paths that name the
  kind's primitive (registry Section C vocabulary, `percept/tunables.py:513-759`), then arms through
  the public `apply_watch_plan`.

The primitive is read off the closed route table (`percept/behavior_routes.py:24-34`) plus the two
frame-Watch refinements the worker actually executes (`hold_s` → sustained duration,
`fire_when` → accumulate/nth-distinct counting). It is measured on the **armed, post-coercion**
`Goal.state` read back from the goal registry — what the frame worker really runs.

### Per-goal-kind compile breakdown

| goal_kind | n | DEFAULT matches kind | AUTHORED matches kind | AUTHORED arms | executable at all¹ | DEFAULT primitive actually produced |
|---|---:|---:|---:|---:|---:|---|
| duration   | 40 | **0** | 40 | 40 | 40 | `state` (plain boolean watch) ×40 |
| transition | 28 | **0** | 28 | 28 | 28 | `state` ×28 |
| count      | 28 | **0** | 28 | **0** | 28 | `state` ×28 |
| state      |  4 | 4 | 4 | 4 | 4 | `state` ×4 |
| **total**  | **100** | **4** | **100** | **72** | **100** | — |

¹ *executable at all* = authored primitive armed, **or** the accumulate/`fire_when` fallback armed
with a counting primitive.

### Finding 1 — the default path compiles every goal to the same thing (4/100)

Compiling all 100 `goal_text` values produces **one distinct compiled fingerprint** — the settings
tree, `Goal.state`, `watchspec_kwargs`, `plan_mode`, `timebase`, and `coverage_floor` are byte-identical
across every duration, transition, count, and state goal in the corpus:

```
Goal.state = {'type': 'watch', 'count': 0}
```

The 4 "passes" are the 4 `state` cases, which pass only because a plain boolean Watch *is* the state
primitive. Nothing in the chain distinguishes a duration goal from a count goal.

This is by construction, not by accident:

* `percept/goalsugar.py:3-11` — *"there is nowhere in this module where the condition text is read for
  meaning."*
* `percept/watch.py:170-176` — *"That desugaring is pass-through, not compilation: the string is never
  parsed, normalised, or mapped to a vocabulary."*
* `percept/runtime_compiler.py:310` — `state: dict[str, Any] = {"type": "watch", "count": 0}`. The
  behavior block is assembled *only* from `Goal.state.*` paths already present in `resolved.effective`
  (`percept/runtime_compiler.py:255-270, 284-308`) plus `WatchPlan.rearm` / `.consequence`. The
  condition string is never an input.

**Consequence for the published numbers.** Any end-to-end run that arms a corpus goal by its bare
`goal_text` is evaluating a boolean presence Watch against duration/transition/count ground truth.
For the 28 count cases and 28 transition cases that is a category error, not a tuning gap: a
`state=` Watch cannot in principle produce the interval or the tally the ground truth scores.
Recall/precision measured that way attributes a *compile* defect to the judge.

### Finding 2 — the `count` primitive is authorable but not executable (0/28 armed)

`Goal.state.type = "count"` resolves and compiles cleanly, and `ResolvedWatch` reports the intended
primitive. It then **cannot be armed**:

```
status   = rejected
dropped  = [("Goal.state.type", "unsupported_runtime_route")]
message  = behavior type 'count' resolves to route 'external_boundary',
           which percept.run cannot execute; no Watch was armed.
```

`percept/behavior_routes.py:29` maps `"count" → BehaviorRoute.EXTERNAL_BOUNDARY`;
`percept/behavior_routes.py:169-171` `worker_can_drive()` admits only the three FRAME routes;
`percept/watchplan_apply.py:221-241` turns that into the typed rejection. `"rhythm"` (`EXTERNAL_RHYTHM`)
is unarmable for the same reason — relevant because 20 of the 28 count cases are RepCount rep-counting.

This is **honest** — the refusal is typed and loud, and no Watch arms. It is the correct behaviour for
an unimplemented route. It does mean the only *executable* counting shape in 1.0.0 is the accumulate
predicate:

```python
WatchPlan(condition=text, explicit={"Goal.state.fire_when": "n >= 2",
                                    "Goal.state.window_s": 30.0})
# armed: {'type': 'watch', 'count': 0, 'fire_when': 'n >= 2', 'window_s': 30.0}
```

which armed for all 28/28 count cases. Note what that shape *is*: a firing predicate over accumulated
distinct observations on a boolean Watch. It is not a cardinality readout. A corpus goal such as
*"When the boy shouts 'No Rio 2!', count how many children are visible in the screen."*
(`omnipro::TXeVf42sQdM::80`) asks for a per-frame count of a set; `fire_when: n >= k` answers "have I
now seen k". Treating one as the other is exactly the substitution SV-0.7 exists to catch.

### Finding 3 — window / refractory semantics

The compiled gate/event windows are sane and finite:

| field | value |
|---|---|
| `Settings.gate.refractory_s` | `0.0` (suppression off by default — opt-in, `percept/config.py:104-109`) |
| `Settings.gate.validity_window_s` | `90.0` |
| `Settings.gate.key_ttl_s` | `3600.0` |
| `Settings.events.window` | `12` frames |
| `Settings.events.merge_gap_s` | `1.5` |
| `Settings.episodes.window_s` | `3600.0` |

The authored duration shape is internally consistent: `hold_s=3.0 ≥ cadence.min_judge_interval=1.0`
with `cadence.gate_on == "fixed"`, and `percept/duration_routes.py:55-192` fails closed on every
inconsistent combination probed (a `hold_s` with no cadence is rejected
`unsupported_duration_route`; a `hold_s` on a non-`FRAME_WATCH` route is rejected; a `gate_on` other
than `"fixed"` under a hold is rejected). No unsound duration Watch was accepted.

---

## SV-0.8 — Compile determinism + hash coverage

### Determinism — PASS

Four fixed plans (bare, transition, duration, accumulate — including typed `target`/`rearm`/
`consequence`/`detectors` fields) were compiled and digested on `plan_hash`,
`component_snapshot_hash`, `effective`, `provenance`, `goal_id`, `Goal.state`, the settings tree, and
`watchspec_kwargs`.

| run | cwd | `PYTHONHASHSEED` | result |
|---|---|---|---|
| in-process ×2 | repo | inherited | identical |
| subprocess | `eval/stage-validation` | `0` | identical |
| subprocess | temp dir A | `12345` | identical |
| subprocess | temp dir B | `999` | identical |
| subprocess | `/` | `7` | identical |

Every subprocess ran with `PERCEPT_*` scrubbed and `PYTHONPATH` removed. **Zero digest differences.**
Compilation is reproducible across processes, hash seeds, and working directories.

### `state.*` hash coverage — the suspected weakness is NOT where it was believed

All 45 `Goal.state.*` paths in `percept.tunables.REGISTRY` were mutated one at a time (a legal,
non-default value per the spec's own `choices`/`bounds`) and re-resolved.

| coverage question | result |
|---|---|
| `plan_hash` moves when the field is authored via `WatchPlan.explicit` | **45 / 45** — no gaps |
| `ResolvedWatch.effective` carries the field | **45 / 45** — no gaps |
| `component_snapshot_hash` moves | **0 / 45** |
| field moved `plan_hash` but changed nothing in the armed Watch, **silently** | **0** |
| field moved `plan_hash` but changed nothing in the armed Watch, **reported on the receipt** | 7 |

`plan_hash = content_hash(watch_plan.to_dict())` (`percept/resolver.py:839`) and
`WatchPlan.to_dict()` serialises `explicit` verbatim
(`percept/contracts/watchplan.py:348`), so **every authored `state.*` field is covered.** The
suspicion that `plan_hash` under-covers `state.*` *on the authoring path* is not borne out.

`component_snapshot_hash` covering none of them is correct by design — it is
`content_hash(components)` over provider names + the catalog hash (`percept/resolver.py:883`), and
is not meant to carry plan state.

The 7 conditionally-valid fields whose value the coercion firewall drops (because the context they
require was not authored) are each reported on the receipt as `coerced_dropped`, so the plan lands
`status="partial"` — loud, not silent:

`Goal.state.cadence_s` (needs `rearm="interval"`), `Goal.state.deadline_after_s` (needs
`rearm="deadline"`), `Goal.state.from`, `Goal.state.to` (need `type="transition"`),
`Goal.state.window_s` (needs `fire_when`), `Goal.state.rhythm.expected_rom`,
`Goal.state.rhythm.expected_rom_space` (need a named `rhythm.channel`).

*Minor path-naming inconsistency:* authoring `Goal.state.window_s` yields a diagnostic whose `path` is
`Goal.state.accumulate.window_s` (`percept/watchplan_apply.py:142-147`) — a different string from the
registry path the caller authored. A machine consumer keying on `FieldDiagnostic.path` will not match
it back to the tunable.

### The real coverage gap — `state.*` written by a layer other than the plan

`plan_hash` hashes the **plan**, not the **resolution**. Two ResolvedWatch objects can compile to
materially different runtimes while both public hashes on `ResolvedWatch` are identical.

**Gap A — a SourceProfile injects `Goal.state.*` into the compiled Goal.**
`percept/resolver.py:249-258` binds a profile's coast floor onto `Goal.state.sense.base_fps`,
`Goal.state.sense.max_fps`, `Goal.state.cadence.min_judge_interval`, and `Goal.state.cadence.gate_on`.
Compiling the *same* plan with and without an untuned coasting profile:

```
without profile: Goal.state = {'type':'watch','count':0}
with profile:    Goal.state = {'type':'watch','count':0,
                               'sense':{'base_fps':0.25,'max_fps':0.25},
                               'cadence':{'min_judge_interval':4.0,'gate_on':'fixed'}}

plan_hash moved ................ False
component_snapshot_hash moved .. False
effective moved ................ True
```

The second Watch runs on fixed 4 s admission at 0.25 fps; the first runs on motion admission at the
default cadence. Both report the same `plan_hash` and the same `component_snapshot_hash`. The shipped
`screen_document` profile reproduces the `sense.*` half of this.

**Gap B — a Deployment/env `Settings` override changes the admission the Watch runs on.**
Same plan, `Settings.edge.motion_gate 0.1→0.9`, `Settings.edge.salience_stat mean→block_max`,
`Settings.gate.threshold 0.7→0.95`:

```
watchspec_kwargs changed ....... True  (threshold 0.1→0.9, salience_stat mean→block_max)
gate threshold ................. 0.7 → 0.95
plan_hash moved ................ False
component_snapshot_hash moved .. False
effective moved ................ True
```

**Fields not covered by either public hash on `ResolvedWatch`:** every one of the 39 `Settings.*` leaf
paths, and every `Goal.state.*` path contributed by SourceProfile / DomainPack / ExecutionPolicy /
OptimizationLock / Deployment rather than by `WatchPlan.explicit` — concretely, the four the shipped
profiles actually write: `Goal.state.sense.base_fps`, `Goal.state.sense.max_fps`,
`Goal.state.cadence.min_judge_interval`, `Goal.state.cadence.gate_on`.

**Not a live idempotency bug, but a stale contract.** `percept/watchplan_apply.py:282-285`
`_application_key` closes the gap privately by folding `content_hash(resolved.effective)` between the
two, so re-arming across a config change is correctly *not* deduped. But
`percept/watchplan_apply.py:36-41` documents the identity as *"a plan whose `ResolvedWatch.plan_hash`
AND `component_snapshot_hash` were already applied … re-arms nothing"*, which is false of those two
hashes alone. Any consumer outside `watchplan_apply` that takes the documented pair as the executable
identity — a replay index, a results ledger, a cache key, a benchmark run id — will silently collide
two different Watches. `effective_hash` is not exposed anywhere on `ResolvedWatch`.

---

## SV-0.9 — Effective-config inspectability

Four scenarios were resolved and compiled: bare, explicit `Goal.state.*` overrides, typed `WatchPlan`
fields (`sense` / `detectors` / `schedule` / `target`), and a `SourceProfile`.

**What holds (verified, all four scenarios):**

* `set(effective) == set(provenance)` — enforced at construction, `percept/contracts/watchplan.py:465-470`.
* `effective[k] == provenance[k].value` for every key.
* Every `owner_layer` is drawn from the declared vocabulary (`SDK`, `SourceProfile`, `WatchPlan`
  observed; all within `PRECEDENCE ∪ {Gate}`).
* No blank `source_id`; no blank `reason`.
* `ResolvedWatch.explain()` renders every resolved field.
* **All 39 compiled `Settings.*` leaves are attributable** to a layer in all four scenarios — zero gaps.

**What fails: 16 compiled `Goal.state` values across the four scenarios have no `provenance` entry.**

| scenario | compiled `Goal.state` path with no provenance | injected at |
|---|---|---|
| all four | `Goal.state.type` | `runtime_compiler.py:310` (literal `"watch"`) |
| all four | `Goal.state.count` | `runtime_compiler.py:310` (literal `0`) |
| authored_explicit, typed_plan_fields | `Goal.state.target` | `runtime_compiler.py:314` (from `WatchPlan.target`) |
| authored_explicit | `Goal.state.behavior.rearm` | `runtime_compiler.py:287-288` (from `WatchPlan.rearm`) |
| authored_explicit | `Goal.state.behavior.consequence.miss_cost` | `runtime_compiler.py:289-293` (from `WatchPlan.consequence`) |
| typed_plan_fields | `Goal.state.detectors` | `runtime_compiler.py:319` (from `WatchPlan.detectors`) |
| typed_plan_fields | `Goal.state.schedule.windows` | `runtime_compiler.py:321` (from `WatchPlan.schedule`) |
| typed_plan_fields | `Goal.state.perception.modality` | `runtime_compiler.py:330` (derived from `WatchPlan.sense`) |
| typed_plan_fields | `Goal.state.perception.audio_s` | `runtime_compiler.py:375` (`setdefault` from `Settings.events.audio_window_s`) |

Two distinct classes:

1. **Compiler literals** — `type`/`count` are hardcoded in `RuntimeCompiler` and belong to no
   configuration layer. `ResolvedWatch` genuinely cannot attribute them, but the operator asking
   *"why is this a `watch` and not a `transition`?"* gets no answer from `provenance`.
2. **Typed `WatchPlan` fields that bypass the `effective`/`provenance` maps.** `target`, `rearm`,
   `consequence`, `detectors`, `schedule`, and `sense` are all real, plan-owned configuration that
   lands in the executed `Goal.state`, but `ConfigResolver` only ever populates `effective` from
   `WatchPlan.explicit` (`percept/resolver.py:624, 640-642`). `Goal.state.perception.audio_s` is the
   sharpest case: it is a **derived** value the compiler `setdefault`s from
   `Settings.events.audio_window_s` (`runtime_compiler.py:370-375`) — a real cross-layer decision
   with no provenance record at all.

Pass bar was "every value attributable to a layer." For the settings tree that holds; for the
executed `Goal.state` it does not.

---

## SV-0.10 — Unsupported-goal honesty

Eleven deliberately out-of-1.x-envelope goals were compiled and then armed through the public
`apply_watch_plan`. Ten are plain-language conditions requiring person identity, distinct-person
counting, re-identification, or intent inference; the eleventh names a person-identifying resolver
through the supported `Goal.state.identity_policy` knob.

**Result: 0/11 failed closed. 11/11 silent degradations.**

| # | goal | capability required | outcome |
|---|---|---|---|
| 1 | *tell me when Divi specifically walks into the room* | named-individual biometric re-ID | `applied`, `{'type':'watch','count':0}` |
| 2 | *who is the person standing at the front door right now* | person identity | `applied`, plain watch |
| 3 | *count how many distinct people entered the building today* | cross-episode person re-ID | `applied`, plain watch |
| 4 | *tell me how many unique visitors are in the queue right now* | de-duplicated person cardinality | `applied`, plain watch |
| 5 | *alert me when someone looks like they intend to shoplift* | intent inference | `applied`, plain watch |
| 6 | *let me know when the driver seems drowsy and is about to fall asleep* | unobserved future internal state | `applied`, plain watch |
| 7 | *notify me when the same person returns for a second time* | person re-ID across visits | `applied`, plain watch |
| 8 | *tell me when an employee rather than a customer is behind the counter* | role attribution / roster linkage | `applied`, plain watch |
| 9 | *recognise my dog Rex and tell me when he is on the sofa* | named-individual identity | `applied`, plain watch |
| 10 | *identify which of my three children is at the fridge* | person disambiguation | `applied`, plain watch |
| 11 | `Goal.state.identity_policy = "face_recognition"` | person-identifying resolver | `applied`, `{'type':'watch','count':0,'identity_policy':'face_recognition'}` |

Every one returned `WatchReceipt(status="applied")` with empty `dropped_fields`, `coerced_fields`, and
`warnings`, and a live armed goal in the registry. There is no typed reason anywhere, because no stage
of the chain examines what the goal is asking for:

* `percept/goalsugar.py:3-11, 66-90` and `percept/watch.py:170-176` — the condition is pass-through by
  contract; nothing reads it for meaning.
* `percept/resolver.py:487-904` — `ConfigResolver` gates *ownership* of tunable paths, not capability.
* `percept/runtime_compiler.py:359-445` — pure translation, no validation ("a bare `runtime_compile`
  call performs no validation and raises nothing", `runtime_compiler.py:41-42`).
* `percept/watchplan_apply.py:417-443` — `_unsupported_plan_fields` is the *only* capability refusal at
  apply, and it covers exactly four typed fields: `admission`, `evidence`, `extractors`, `uncertainty`.
  Nothing about goal semantics.

**Case 11 is the sharpest.** The envelope machinery exists and is explicit that person
re-identification is out of scope (`percept/envelope.py:203+`, `OUT_OF_ENVELOPE["security_person_id"]`),
but `_assert_within_envelope` (`percept/identity_policy.py:139-155`) only fires when the named policy
resolves to an **already-registered** resolver carrying `identifies_persons=True`:

```python
resolver = resolvers.get(policy)
if resolver is not None and getattr(resolver, "identifies_persons", False):
    raise EnvelopeViolation(...)
```

`"face_recognition"` is not registered, so `resolvers.get()` returns `None` and the check is a no-op.
The value passes the tunables ownership gate (`Goal.state.identity_policy` is `value_type="str"` with
no `choices`, `percept/tunables.py:723-728`), survives coercion untouched (it has no `coerce_authored_*`
firewall), lands in the armed `Goal.state`, and `identity_policy_of()` reads it back as
`"face_recognition"`. The refusal is checked at *selection* time against a *registry*, never at compile
time against the *request*.

**Assessment.** This is the deliberate design ("the judge sees the caller's own words"), and it is
defensible as a *judging* rule. But the SV-0.10 bar is about the *compile* boundary, and there the
outcome is unambiguous: a client can arm "tell me when Divi walks in" and get a Watch that reports
`status="applied"`, arms cleanly, and produces plausible-looking fires — while doing something
categorically different from what was asked. Zero silent degradations was the bar; the measured count
is eleven.

---

## Summary of defects

| # | Severity | Defect | Location |
|---|---|---|---|
| 1 | **High** | No stage of the chain derives a primitive from the goal's meaning: 96/100 corpus goals compile to a boolean `watch`, one identical artifact for all four kinds. A count/transition/duration goal armed by text alone cannot express its ground truth. | `goalsugar.py:3-11`; `runtime_compiler.py:310` |
| 2 | **High** | 11/11 out-of-envelope goals compile and arm with `status="applied"` and no diagnostic. Capability refusal at apply covers only 4 typed `WatchPlan` fields. | `watchplan_apply.py:417-443` |
| 3 | **High** | `Goal.state.identity_policy` accepts an arbitrary person-identifying resolver name; the envelope guard only fires for an already-registered resolver, so an unregistered name bypasses it entirely. | `identity_policy.py:139-155`; `tunables.py:723-728` |
| 4 | **Medium** | Neither public hash on `ResolvedWatch` covers `Settings.*` or profile-contributed `Goal.state.*`. Two materially different executables share both. The documented idempotency identity (`plan_hash` + `component_snapshot_hash`) is stale — `_application_key` privately needs a third hash that is never exposed. | `resolver.py:839, 883`; `watchplan_apply.py:36-41, 282-285` |
| 5 | **Medium** | 16 compiled `Goal.state` values across 4 scenarios carry no provenance entry, including the derived `perception.audio_s` and every typed-`WatchPlan`-field path (`target`, `rearm`, `consequence`, `detectors`, `schedule`, `sense`). | `runtime_compiler.py:310,314,317-321,330,375`; `resolver.py:624,640-642` |
| 6 | **Low** | `type="count"` and `type="rhythm"` are authorable, resolve cleanly, and are then rejected at arm as `unsupported_runtime_route`. Honest and typed — but the only executable counting shape in 1.0.0 is the `fire_when` accumulate predicate, which is a firing rule, not a cardinality readout. | `behavior_routes.py:24-34, 169-171`; `watchplan_apply.py:221-241` |
| 7 | **Low** | Authoring `Goal.state.window_s` produces a diagnostic keyed `Goal.state.accumulate.window_s` — a `FieldDiagnostic.path` a machine consumer cannot match back to the registry path it authored. | `watchplan_apply.py:142-147` |

## What passed cleanly

* Compilation is fully deterministic across processes, `PYTHONHASHSEED` values, and working directories.
* `plan_hash` covers **45/45** authored `Goal.state.*` registry paths.
* **Zero silent coercion drops**: all 7 conditionally-invalid authored fields are reported on the
  receipt (`coerced_dropped`, `status="partial"`); the 2 unarmable shapes are typed rejections.
* All **39** compiled `Settings.*` leaves are attributable to a resolution layer in every scenario.
* `effective`/`provenance` key and value parity holds; `owner_layer` never leaves the declared vocabulary.
* The duration route contract fails closed on every unsound shape probed.

## Reproduce

```bash
cd /Users/dhivyavijayakumar/git/Percept-sdk
/private/tmp/percept-launch/oss-final-venv-bba572a/bin/python eval/stage-validation/sv0_compile.py
```

Exit `0` on suite pass, `1` on failure. Full per-case and per-field data (including all 100 corpus
rows and the 45-row hash-coverage matrix) is in `results/sv0_compile.json`.
