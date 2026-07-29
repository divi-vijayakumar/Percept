# CLAUDE.md — working in this repo

Read this before creating any file. **Where a file goes is not a judgment call — it is decided by
the table below.** Getting it wrong puts internal material in a public OSS tree.

---

## The one rule

**There are two trees, and they never mix.**

| tree | what it is | in git? | ships? |
|---|---|---|---|
| the repo root, `docs/`, `packages/`, `eval/`, `examples/`, `tools/` | the **public OSS project** | yes | yes |
| `Product-Execution/` | **all internal material** — specs, plans, standups, probes | **no**, gitignored | never |

If you are about to write a doc and it is not adoption material for an external developer, it goes
in `Product-Execution/`. There is no third option and no top-level `scratch/`, `internal-docs/`, or
`execution-docs/` — those were consolidated on 2026-07-28. **Do not recreate them.**

## Where does my new file go?

| I am writing… | it goes in |
|---|---|
| a product spec change | `Product-Execution/1-product-specification/PRODUCT_SPEC.md` — **update in place, never a v2 file** |
| PM-level product thinking | `Product-Execution/1-product-specification/PM_PRODUCT_SPEC.md` — **update in place** |
| a spec for one feature | `Product-Execution/1-product-specification/features/<feature>.md` |
| a future idea, not yet scheduled | `Product-Execution/1-product-specification/backlog/<idea>.md` |
| an implementation or execution plan | `Product-Execution/2-engineering-specification/plans/` |
| a code/architecture analysis or review | `Product-Execution/2-engineering-specification/analysis/` |
| release process material | `Product-Execution/2-engineering-specification/release/` |
| a handoff, status, or session summary | `Product-Execution/3-daily-standup/<YYYY-MM>/` |
| a probe: its spec, its runner, its results | `Product-Execution/4-probes/<probe-name>/` |
| client onboarding / incident replies | `Product-Execution/5-client-delivery/` |
| a dataset (local, synthetic, cloud) | `Product-Execution/datasets/` |
| **public** adoption docs | `docs/` — see the structure below |
| an architecture decision record | `.adr/` — tracked, but **not** `docs/` |

Standups are archived monthly. Roll `3-daily-standup/<month>/` up into a single summary when the
month closes.

## The public `docs/` tree is object-first

Organized by **the thing you are building**, not by document genre. There is no
`tutorials/`, `how-to/`, `reference/`, or `explanation/` — that layout was replaced on 2026-07-28.

```
docs/get-started/    install, first run, CLI, coding agents
docs/watches/        the core object: goals, the gate, driving a live watch
docs/sources/        cameras, files, streams, screens, audio
docs/admission/      the eye — everything upstream of a model call
docs/judges/         the models, and bringing your own
docs/incidents/      what comes out, and how it reaches your app
docs/configuration/  every knob, who owns it, how to tune it
docs/observability/  traces, replay, cost
docs/evaluation/     measuring it, and reading the numbers honestly
docs/resources/      how it works, architecture, CLI reference
docs/demo/           the browser harness
```

A new public page goes in the section matching **the object it is about**. If it fits none of them,
it is probably internal — re-read the table above.

## Never in tracked code or docs

- **No strategy.** Never name a tier, a plan, a price, or a future offering in a docstring, comment,
  or public doc. State capability bounds plainly instead — "uncalibrated", "advisory", "not general".
  A shipped docstring once said *"the high-accuracy re-id service is a paid tier"*; that is the
  failure mode.
- **No unpublished numbers.** Accuracy figures are gated — see `docs/evaluation/benchmarks.md`.
  Never quote a benchmark number in a public doc without its ground-truth coverage.

## Generated files — do not hand-edit

| file | generator | regenerate |
|---|---|---|
| `docs/configuration/tunables.md` | `tools/build_tunables_doc.py` | `make docs-tunables` |
| `docs/judges/registries.md` | `tools/build_registries_doc.py` | `make docs-registries` |
| `docs/README.md` | `tools/build_docs_index.py` | `make docs-index` |
| `docs/architecture.html`, `docs/architecture.svg` | `tools/build_architecture.py` | `make architecture` |

Change the **source** (the registry, the frontmatter, the code) and regenerate. `make docs` does all
of the doc pages at once.

## Gates — run before you hand work back

```bash
make docs-check          # generated pages match the code
make docs-links          # zero broken relative links in any tracked .md
make check-skills-sync   # .claude/skills and .agents/skills stay byte-identical
make architecture-check  # the diagram matches the code
make test                # the suite
```

**The suite has exactly 2 known failures** — `test_conformance_seams` and `test_conformance_suite`,
a unittest double-import artifact (`tests.X` vs `X`). They pass standalone. More than 2 means you
broke something.

## Things that will bite you

- **`codewiki/` and `.dispatch-board/` are symlinks** into `Product-Execution/`. Their gitignore
  entries deliberately have **no trailing slash** — a trailing-slash pattern does not match a
  symlink, and they would show up as untracked. Do not "fix" them.
- **`.adr/` is tracked and drift-guarded.** Six test files read the ADR prose and assert the code
  still matches it. Moving an ADR or breaking a citation fails the suite.
- **Several docs are cited from shipped source.** `percept/eval/exact_ci.py` and `frontier.py` cite
  `docs/evaluation/sizing-a-run.md`; `percept/slo.py` cites `docs/evaluation/methodology.md`. Grep
  before moving any doc.
- **Two skill directories must stay byte-identical** — `.claude/skills/percept-agent/` and
  `.agents/skills/percept-agent/`. Edit both or `make check-skills-sync` fails.
- **This repo sometimes has more than one Claude session writing to it.** Check `git status` and
  `git log` before assuming an uncommitted change is yours.
