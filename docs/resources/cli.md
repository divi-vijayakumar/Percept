---
title: Command-line reference
updated: 2026-07-23
applies_to: percept-vision 1.0.0
status: stable
---
# Command-line reference

`percept-vision` installs nine console scripts. Each is a self-contained `main(argv) -> int`; the exit
code is part of the contract. This page documents the user-facing flags of each — verify against the
source under `percept/`.

Most commands run from a bare `pip install percept-vision`. The exceptions (a live source, the
benchmark harness) are called out per command and summarized at the end.

- [`percept`](#percept) — umbrella router + resolve/config/components
- [`percept-watch`](#percept-watch) — live watch demo
- [`percept-doctor`](#percept-doctor) — pure-CV dry run
- [`percept-replay`](#percept-replay) — deterministic trace replay
- [`percept-bench`](#percept-bench) — per-concern precision/recall
- [`percept-eval`](#percept-eval) — paired differential A/B
- [`percept-derive-profile`](#percept-derive-profile) — derive a WatchProfile
- [`percept-optimize`](#percept-optimize) — search safe tunables → OptimizationLock
- [`percept-schema`](#percept-schema) — export contract JSON Schema

---

## `percept`

Umbrella router. `percept <name> ...` invokes the same `main` as the standalone `percept-<name>`
script — `percept watch --source 0` and `percept-watch --source 0` are byte-identical. Bare `percept`
with no arguments prints help and exits `2`; `percept -h` exits `0`.

Delegated names: `watch`, `doctor`, `replay`, `bench`, `eval`, `derive-profile`, `optimize`, `schema`.
`percept eval ingest-trace ...` is special-cased to the telemetry→fixtures pipeline (it is not the
`percept-eval` differential).

Three subcommands are owned by `percept` itself:

### `percept resolve`

Resolve a `WatchPlan` through `ConfigResolver` and print its per-field provenance.

| Flag | Default | Effect |
|---|---|---|
| `config` (positional) | none | path to a JSON/YAML doc shaped like `WatchPlan` kwargs (+ optional top-level `source_profile`) |
| `--condition TEXT` | none | condition text when not using a config file |
| `--source-profile NAME` | none | a shipped SourceProfile (`physical_scene`, `screen_document`, `conversation`) or any registered name |
| `--explain` | off | print `ResolvedWatch.explain()` (per-field provenance) |
| `--json` | off | print `ResolvedWatch.to_dict()` as JSON |

```
percept resolve --condition "someone enters the doorway" --source-profile physical_scene --explain
```

### `percept config tunables`

List the live tunables registry (the same inventory `tunables.json` ships).

| Flag | Default | Effect |
|---|---|---|
| `--owner NAME` | all | filter to one owner (e.g. `Gate`, `SourceProfile`) |
| `--json` | off | print the full `tunables.json` document |

```
percept config tunables --owner Gate
```

### `percept components list` / `describe`

Survey the read-only component catalog across the selectable-component registries. Roles: `detector`,
`backend`, `identity_resolver`, `sense`, `profile`, `execution_policy`, `domain_pack`.

| Flag | Default | Effect |
|---|---|---|
| `list --role ROLE` | all | filter the listing to one role |
| `list --json` | off | JSON output |
| `describe <ref>` | — | `name` or `role:name` (role prefix disambiguates) |
| `describe --json` | off | JSON output |

```
percept components list --role backend
percept components describe backend:claude
```

**Install:** bare (`pip install percept-vision`). YAML config input for `resolve` needs `pyyaml`.

---

## `percept-watch`

Live watch demo: point at a webcam / video / RTSP stream, give plain-English goals, and print a
notification the moment each becomes true — once per episode. Bakes in the recommended defaults
(`block_max` localized salience, a per-episode refractory).

| Flag | Default | Effect |
|---|---|---|
| `--source` | `webcam:0` | `webcam:0`, a video path, or `rtsp://…` |
| `--goal` | (demo goals) | a plain-English condition (repeatable) |
| `--say` | condition text | friendly message per goal, in order (repeatable) |
| `--vision` | `openrouter` | provider: `openrouter` (Qwen) \| `gemini` \| `claude` |
| `--model` | per-provider | override the judge model id |
| `--fps` | `1.0` | frames per second to pull |
| `--refractory` | `15.0` | seconds to suppress repeat notifications |
| `--egomotion` | off | moving camera: subtract head/camera motion so the gate does not escalate every frame |
| `--trace PATH` | none | record the run's per-frame decision trace (JSON) |
| `--offline` | off | keyless fixture replay — no key, no camera, proves wiring |
| `--fixture NAME` | `kettle-boils` | which bundled fixture `--offline` replays |

Per-provider default models: `openrouter` → `qwen/qwen3-vl-32b-instruct`, `gemini` →
`gemini-3.1-flash-lite`, `claude` → `claude-sonnet-5`.

```
export OPENROUTER_API_KEY=sk-or-...
percept-watch --goal "someone raises their hand" --goal "someone is holding a phone"

percept-watch --offline --trace t.json     # keyless; then: percept-replay t.json
```

**Install:** live needs `[runtime]` + the provider extra (`[openrouter]`/`[gemini]`/`[claude]`) + that
provider's API key (a missing key exits `2`). `--offline` is bare stdlib.

---

## `percept-doctor`

Plan-aware, pure-CV dry run: did the source open, what plan compiled, what admitted, and which rule
suppressed the rest. **Zero judge calls, no API key, no cost.** Also renders an existing trace to a
standalone HTML visualizer.

| Flag | Default | Effect |
|---|---|---|
| `--source` | none | a video path / `rtsp://` / `webcam:0` (required in capture mode) |
| `--goal` | none | a plain-language condition (repeatable; required in capture mode) |
| `--fps` | `2.0` | capture rate to pull |
| `--max-frames` | all | stop after N frames |
| `--timebase` | `auto` | `auto` \| `media` \| `wall` |
| `--goal-temporality` | none | `stative` \| `dynamic` |
| `--camera-mount` | none | `fixed` \| `wearable` \| `glasses` \| `handheld` |
| `--start-state` | none | `start_anchored` \| `transition` |
| `--source-profile` | `physical_scene` | source class (open registry — any registered name) |
| `--actuation` | `none` | `none` \| `screen` \| `physical` (`physical` → exit `6`) |
| `--trace-out PATH` | none | write the trace artifact (JSON) |
| `--json` | off | emit the machine-readable report on stdout |
| `--visualize-out PATH` | none | write the standalone HTML visualizer |
| `--visualize-from PATH` | none | render an existing trace / `.ptrace` instead of capturing (requires `--visualize-out`, forbids `--source`) |
| `--visualize-media` | off | opt in to embedding media bytes in the HTML |
| `--golden PATH` | none | ground-truth episodes (JSON) for the drift lane |

Exit codes: `0` ok · `2` zero coverage (nothing admitted for any goal) · `3` source failure · `4`
plan unobservable under the selected profile · `5` invalid configuration · `6` declared out of
envelope.

```
percept-doctor --source clip.mp4 --goal "the kettle is steaming"
percept-doctor --visualize-from t.json --visualize-out report.html   # render only, bare stdlib
```

**Install:** capture mode needs `[runtime]`. Render mode (`--visualize-from`) is bare stdlib.

---

## `percept-replay`

Deterministic replay of a recorded trace through a fresh Gate. No model call, offline. The trace's
manifest carries the Gate's knobs, so this replays *that run*, not today's defaults.

| Flag | Default | Effect |
|---|---|---|
| `trace` (positional) | — | a trace JSON, or a `.ptrace` bundle directory/archive |
| `--json` | off | emit the `ReplayReport` as JSON instead of a panel |
| `--escalates GOAL_ID` | none | declare a goal whose FIRE is delivered outward as an ASK (repeatable) |
| `--suppress-false GOAL_ID` | none | declare a never-goal: a confident-false never reaches the gate (repeatable) |

Exit codes: `0` every replayed action matches the recorded run · `1` a divergence · `2` load failure.

```
percept-replay t.json
```

**Install:** bare stdlib. See [../how-to/trace-bundles-and-replay.md](../observability/traces-and-replay.md).

---

## `percept-bench`

Per-concern precision/recall over a labelled-clip manifest, plus `scorecard` and `audit` subcommands.

| Invocation | Effect |
|---|---|
| `percept-bench <manifest.json>` | per-concern P/R over the clips in the manifest |
| `percept-bench scorecard [--manifest ... --out ... --mode full --require-release]` | multi-track snapshot card |
| `percept-bench audit [--manifest ... --baselines ... --release]` | validate golden manifest + baseline artifacts |

The real harness lives in the top-level `eval` package, which ships with the **source repo, not the
wheel**. A bare wheel install prints guidance and exits `2`.

```
PYTHONPATH=packages/percept-vision percept-bench eval/golden-v1/MANIFEST.json
```

**Install:** a repo checkout + `[bench]` (numpy + opencv) + `PYTHONPATH=packages/percept-vision`.

---

## `percept-eval`

Paired differential A/B of two plans over a fixtures directory.

| Flag | Default | Effect |
|---|---|---|
| `--baseline PATH` | required | baseline plan file (`.yaml`/`.json`) |
| `--candidate PATH` | required | candidate plan file (`.yaml`/`.json`) |
| `--fixtures DIR` | required | fixtures root directory |
| `--bars PATH` | none | client-supplied bars file; omit to report only |
| `--split NAME` | `test` | held-out split: `train` \| `dev` \| `test` |
| `--expect-registration-hash H` | none | refuse to score unless the configuration hashes to `H` |
| `--json PATH` | none | write the full report as JSON |
| `--declared-at TEXT` | `""` | when the bars/split were declared |
| `--frontier` | off | also emit the operating-frontier report |
| `--frontier-arm ARM` | `candidate` | which arm the frontier is computed on |
| `--frontier-bars PATH` | none | operating bars file (false-fire / recall bounds) |
| `--frontier-json PATH` | none | write the frontier report as JSON |

Exit codes: `0` pass/reported · `1` a client bar failed · `2` refused (bad input / split error).

```
percept-eval --baseline base.yaml --candidate cand.yaml --fixtures ./fixtures --bars bars.json
```

**Install:** ships in the wheel, bare stdlib (YAML plans need `pyyaml`). See
[../how-to/eval.md](../evaluation/evaluate-your-data.md).

---

## `percept-derive-profile`

Derive a provenance-stamped `WatchProfile` from labeled fixtures and a pre-registered ladder of
candidate configurations. Without `--bars` it reports the ladder and refuses to declare a winner.

| Flag | Default | Effect |
|---|---|---|
| `--fixtures DIR` | required | fixtures root directory |
| `--ladder PATH` | required | ladder document (`.json`/`.yaml`) |
| `--bars PATH` | none | operator-supplied bars file; omit to report the ladder only |
| `--split NAME` | `test` | held-out split: `train` \| `dev` \| `test` |
| `--expect-registration-hash H` | none | refuse to derive unless the configuration hashes to `H` |
| `--json PATH` | none | write the full scorecard as JSON (whole ladder, losers included) |
| `--out-profile PATH` | none | write the derived profile (only when a winner is declared) |
| `--run-id ID` | none | override the derived `eval_run_id` |
| `--declared-at TEXT` | `""` | when the bars/split/ladder were declared |

Exit codes: `0` derived or reported · `1` no rung met the bars · `2` refused.

```
percept-derive-profile --fixtures ./fixtures --ladder ladder.json --bars bars.json --out-profile p.json
```

**Install:** bare stdlib. See [../how-to/derive-a-profile.md](../sources/source-profiles.md).

---

## `percept-optimize`

Search declared-safe tunables through the same pre-registered ladder discipline
`percept-derive-profile` uses, and emit a validated `OptimizationLock` for the winner.

| Flag | Default | Effect |
|---|---|---|
| `--fixtures DIR` | required | fixtures root directory |
| `--space PATH` | required | optimizer document (`.json`/`.yaml`): templates + search space |
| `--bars PATH` | none | operator-supplied bars file; omit to report only |
| `--plan PATH` | none | a WatchPlan document; its content hash becomes `base_plan_hash` |
| `--split NAME` | `test` | held-out split: `train` \| `dev` \| `test` |
| `--strategy S` | (document) | `grid` \| `random` (overrides the document's own strategy) |
| `--n-random N` | `8` | candidates to sample for `random` strategy |
| `--seed N` | `0` | random seed |
| `--expect-registration-hash H` | none | refuse unless the configuration hashes to `H` |
| `--json PATH` | none | write the full report as JSON |
| `--out-lock PATH` | none | write the `OptimizationLock` (only when a winner is produced) |
| `--run-id ID` | none | override the `eval_run_id` |
| `--declared-at TEXT` | `""` | when the bars/split were declared |

Exit codes: `0` a lock was written or the space was reported · `1` no candidate met the bars · `2`
refused (malformed input, unsafe path, config drift).

```
percept-optimize --fixtures ./fixtures --space space.json --bars bars.json --out-lock lock.json
```

**Install:** bare stdlib.

---

## `percept-schema`

Export the contract DTOs as draft-2020-12 JSON Schema. Takes **no arguments**; always writes into the
installed package directory (`percept/contracts/schema/`), so it may fail on a read-only install. Exit
`0`.

```
percept-schema
```

**Install:** bare stdlib.

---

## Install requirements at a glance

Most commands run from a bare `pip install percept-vision`:

- **Bare wheel:** `percept` (resolve/config/components), `percept-replay`, `percept-eval`,
  `percept-derive-profile`, `percept-optimize`, `percept-schema`, `percept-doctor --visualize-from`,
  `percept-watch --offline`. (YAML input for `resolve`/`eval` plans needs `pyyaml`.)
- **`[runtime]`** (+ a provider extra + API key for watch): live `percept-watch`, `percept-doctor`
  with `--source`.
- **Repo checkout + `[bench]`** (numpy + opencv, `PYTHONPATH=packages/percept-vision`):
  `percept-bench`.

See [configuration.md](../configuration/overview.md) for every environment variable, and
[backends.md](../judges/backends.md) for provider selection.
