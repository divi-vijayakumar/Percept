# Contributing to Percept

Percept is a research preview. Issues, discussion, and pull requests are welcome.

## Setup

```bash
git clone <repo> && cd Percept-sdk
pip install -e "packages/percept-vision[dev]"   # core is pure stdlib; heavy deps are opt-in extras
make test        # L1 unit suite — fakes only, no keys, fast
make parity      # JS edge  <->  Python percept.harness, byte-for-byte
```

`pip install percept-vision` (no extras) runs offline with fake backends and no keys. Opt into a backend or
runtime only when you need it: `percept-vision[gemini]`, `[claude]`, `[openrouter]`, `[runtime]`, `[pose]`, …

## Test discipline (first-class)

- **L1 (`make test`) is the gate** — deterministic, fakes only, no keys. Every change keeps it green.
- **`percept/gate.py` is invariant** — the three-state firing gate is byte-for-byte stable. Add new state in
  `ConcernContext` / agent hooks routed through `gate.evaluate`; never edit `gate.py`.
- **The core stays pure stdlib** — vendor SDKs (Gemini/Claude/OpenRouter/Deepgram) live behind lazy-imported
  backends + extras. A CI import-lint (`tests/test_core_no_vendor_imports.py`) enforces it.
- **Every new factor is a hypothesis** — A/B it on the eval loop and keep-or-drop on the evidence.

## Pull requests

- Keep the diff focused; match the surrounding code's style and comment density.
- Include the test that proves the change; cite A/B numbers where relevant.
- Run `make test` + `make parity` before opening.

## Architecture

Start with [`docs/quickstart.md`](docs/quickstart.md), [`docs/concepts.md`](docs/concepts.md), and
[`docs/architecture.md`](docs/architecture.md). `llms.txt` is the agent-readable API surface.
