# Contributing to Percept

Percept 1.0 is under active development. Issues and discussion are welcome; please open an issue
before a large pull request.

## Setup

```bash
git clone https://github.com/divi-vijayakumar/Percept && cd Percept
python -m venv .venv && . .venv/bin/activate
pip install -e "packages/percept-vision[dev]"
make check      # guardrails + tests, offline, no API keys
```

## Rules the code follows

- **Functional core, imperative shell.** `contracts`, `decide`, `triggers`, `zones`, `clips`,
  `judges.base` and `replay` are pure: no I/O, no threads, no hidden clock. CI enforces this.
- **A small public surface.** At most 30 names in `percept.__all__`. CI enforces this.
- **Every module is reachable** from `percept/__init__.py`. CI enforces this.
- **Growth is reviewed.** Past 10,000 lines in `percept/`, a pull request needs the owner's
  `size-approved` label.
- **Failures are never answers.** A timeout, outage or malformed reply must never be treated as a
  model saying "no".

## Pull requests

- One focused change per pull request, with the tests that prove it.
- `make check` passes locally.
