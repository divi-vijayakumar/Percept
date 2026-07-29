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
in `Product-Execution/`. There is no third option, and the three top-level holding directories that used to exist for this
(a scratch dir, an internal-docs dir, an execution-docs dir) were consolidated on 2026-07-28.
**Do not recreate them.**

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
  A shipped docstring once named a commercial pricing tier for the high-accuracy re-id service;
  that is the failure mode.
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

### Always name the runner with the count

The two runners disagree, and quoting a number without saying which one produced it has already
misled a session into briefing agents against the wrong baseline.

| runner | command | expected |
|---|---|---|
| **unittest** (this is what `make test` runs) | `PYTHONPATH=. PERCEPT_MODE=offline python3 -m unittest discover -s tests` | **2 known failures**, 0 errors |
| **pytest** | `python3 -m pytest packages/percept-vision/tests -p no:randomly` | **0 failures** |

They differ for two structural reasons, both expected:

- **pytest counts more tests** — it expands parametrized cases and subtests individually, so its
  total is far higher for the same code. Never compare the two totals.
- **The 2 unittest failures do not exist under pytest.** `test_conformance_seams` and
  `test_conformance_suite` hit a double-import collision (the module is loaded as both `tests.X` and
  `X`, so `X.Routable is not tests.X.Routable`). It is a discovery artifact, not a product defect —
  both files pass standalone. pytest's import mode never triggers it.

**More than 2 failures under unittest, or any failures under pytest, means you broke something.**
Totals also shift legitimately between branches — `validation/stage-harness` and `main` have
diverged by hundreds of commits — so compare against *your own branch before your change*, never
against a number quoted in a doc.

A test that needs a git-ignored fixture must skip with **`unittest.SkipTest` raised at module
scope**, which both runners honour. `pytest.importorskip` raises pytest's own `Skipped`, which
unittest's loader turns into a **collection error** — that guard silently only ever worked under
one runner.

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
