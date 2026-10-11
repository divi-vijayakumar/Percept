# Working in this repo

Instructions for coding agents and contributors. Read this before creating any file.

## What Percept is

A decision layer for live video: a stream plus plain-language conditions in, one decision per real
occurrence out. The 0.9 Developer Preview is being built on `main` from a small core, on the way to 1.0. The design is
Trigger → Judge → Decide: cheap triggers decide when to look, a judge decides what is true, and a
pure reducer decides whether the agent hears about it.

## Two trees, never mixed

| Tree | What | In git? |
|---|---|---|
| repo root, `packages/`, `tools/`, `.github/` | the public open-source project | yes |
| `Product-Execution/` | all internal material: specs, plans, reviews, standups, datasets, probes | **no**, ignored |

If a document is not for an external developer using Percept, it goes in `Product-Execution/`:

| Writing… | Goes in |
|---|---|
| a product spec or feature spec | `Product-Execution/1-product-specification/` |
| an engineering plan, design or review | `Product-Execution/2-engineering-specification/plans/` or `analysis/` |
| a handoff or status note | `Product-Execution/3-daily-standup/<YYYY-MM>/` |
| a probe or experiment | `Product-Execution/4-probes/<name>/` |
| a dataset | `Product-Execution/datasets/` (or the ignored `eval/` staging area) |

**Never in tracked files** (code, comments, docstrings, README, docs, commit messages):

- business strategy, pricing, plans, tiers, future commercial offerings, competitor commentary;
- names of internal or customer applications;
- accuracy, latency or cost numbers that aren't published with their run and coverage.

State capability limits plainly instead ("advisory", "not calibrated", "static cameras only").

## Architecture rules (CI enforces the first three)

- **Functional core, imperative shell.** `percept.contracts`, `decide`, `triggers`, `zones`, `clips`,
  `judges.base` and `replay` are pure: no I/O, no threads, no asyncio, no vendor SDKs, time passed in
  as an argument. `contracts` imports no other percept module. Only shell modules (`monitor`,
  `sources`, `delivery`, `sinks`, `trace`, `judges/<vendor>`) touch the outside world.
- **At most 30 public names** in `percept.__all__`.
- **No unreachable modules.** Every module is imported, directly or not, by `percept/__init__.py`.
- **Size is reviewed, not capped.** Past 10,000 lines under `percept/`, a PR needs the owner's
  `size-approved` label. Growth needs a reason.
- **A new public concept, parameter or module needs a named consumer**: a real application, an
  example in `examples/`, or a user issue.
- **Failures are never answers.** A timeout, outage, refusal or malformed reply changes no
  occurrence; it is reported on the health channel.

## Porting from earlier versions

Code from the 0.x research line (0.7.0 and earlier) is preserved in the project history and is not
continued. When an older module has something worth keeping, copy the minimum by hand, rewrite it
against the current contracts, and cite the
source commit and path in the PR description. Never restore an old module wholesale.

## Commands

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e "packages/percept-vision[dev]"
make check        # guardrails + tests
make test         # tests only
make guardrails   # exports, import boundaries, reachability
```

Use a virtual environment. A globally installed `percept-vision` from an older release will
shadow the source tree and misreport its version.

## Workflow

- One short-lived branch and one pull request per change, merged into `main`.
- Every PR is reviewed by someone other than its author before merge. The decision engine
  (`decide`), the runtime and delivery (`monitor`, `delivery`), and the public API also need the
  owner's review.
- More than one agent may be working in this repo. Check `git status` and `git log` before assuming
  an uncommitted change is yours.
