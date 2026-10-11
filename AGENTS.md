# Working in this repo

Instructions for contributors and coding agents. Read this before changing code.

## What Percept is

A decision layer for live video: a stream plus plain-language conditions in, one decision per real
occurrence out. This is the 0.9.0 Developer Preview; the API may change before 1.0. The design is
Trigger → Judge → Decide: cheap triggers decide when to look, a judge decides what is true, and a
pure reducer decides whether the agent hears about it.

## Architecture rules (CI enforces the first three)

- **Functional core, imperative shell.** `percept.contracts`, `decide`, `triggers`, `zones`, `clips`,
  `judges.base` and `replay` are pure: no I/O, no threads, no asyncio, no vendor SDKs, time passed in
  as an argument. `contracts` imports no other percept module. Only shell modules (`monitor`,
  `sources`, `delivery`, `trace`, `judges/<vendor>`) touch the outside world.
- **At most 30 public names** in `percept.__all__`.
- **No unreachable modules.** Every module is imported, directly or not, by `percept/__init__.py`.
- **Size growth is reviewed.** Past 10,000 lines under `percept/` (`make guardrails` reports the
  count), a maintainer reviews why the code grew. Growth needs a reason.
- **A new public concept, parameter or module needs a consumer**: an example in `examples/` or an
  issue that asks for it.
- **Failures are never answers.** A timeout, outage, refusal or malformed reply changes no
  occurrence; it is reported on the health channel.

## Commands

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e "packages/percept-vision[dev]"
make check        # guardrails + tests
make test         # tests only
make guardrails   # exports, import boundaries, reachability
make validate     # guardrails, tests, the offline quickstart and a clean-install wheel
```

Use a virtual environment. A globally installed `percept-vision` from an older release will
shadow the source tree and misreport its version.

## Workflow

- One short-lived branch and one pull request per change, merged into `main`.
- Every pull request is reviewed by a maintainer before merge. Changes to the decision engine
  (`decide`), the runtime and delivery (`monitor`, `delivery`) and the public API get an extra
  maintainer review.
- More than one person or agent may be working in a checkout. Check `git status` and `git log`
  before assuming an uncommitted change is yours.

## What tracked files must not contain

Keep tracked files (code, comments, docs, commit messages) free of credentials, private data and
performance claims (accuracy, latency, cost) that are not published together with the run that
produced them. State capability limits plainly instead ("advisory", "not calibrated", "static
cameras only").
