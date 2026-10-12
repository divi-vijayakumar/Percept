# Contributing to Percept

Percept is in Developer Preview (0.9.0) and under active development. Issues and discussion are
welcome; please open an issue before a large pull request.

Contact: support@velvee.ai

## Setup

```bash
git clone <repository-url> percept && cd percept
python -m venv .venv && . .venv/bin/activate
pip install -e "packages/percept-vision[dev]"
make check      # guardrails + tests, offline, no API keys
```

## Rules the code follows

- **Functional core, imperative shell.** `contracts`, `decide`, `triggers`, `zones`, `clips`,
  `judges.base` and `replay` are pure: no I/O, no threads, no hidden clock. CI enforces this.
- **A small public surface.** At most 30 names in `percept.__all__`. CI enforces this.
- **Every module is reachable** from `percept/__init__.py`. CI enforces this.
- **Growth is reviewed.** Past 10,000 lines in `percept/`, a maintainer reviews why the code grew.
- **Failures are never answers.** A timeout, outage or malformed reply must never be treated as a
  model saying "no".

[AGENTS.md](AGENTS.md) has the full list, for people and coding agents alike.

## Pull requests

- One focused change per pull request, with the tests that prove it.
- `make check` passes locally.
- A maintainer reviews every pull request before merge.
