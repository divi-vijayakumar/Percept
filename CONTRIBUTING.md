# Contributing to Percept

Percept is an open-source cognition layer for goal-driven, proactive vision agents (v1.0.0). Issues,
discussion, and pull requests are welcome. Docs: <https://percept.percia.ai>.

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

## Documentation

`docs/` is the **public tree** — it feeds the website and is adoption material only. Internal
reasoning does not go there.

| you're writing | it goes in |
|---|---|
| teach someone their first run | `docs/tutorials/` |
| accomplish one specific task | `docs/how-to/` |
| describe a surface exactly | `docs/reference/` |
| how it's measured / what may be claimed | `docs/benchmarks/` |
| why it's designed this way | `docs/explanation/` |
| an architecture decision record | `.adr/` — **not** `docs/` |
| a spec, plan, probe, or anything strategic | `scratch/` — gitignored, never shipped |

**Every page under `docs/` carries frontmatter.** New pages get it automatically:

```yaml
---
title: Tune admission
updated: 2026-07-28          # seeded from git history; preserved across regenerations
applies_to: percept-vision 1.0.0
status: stable               # stable | experimental | draft
---
```

Three pages are **generated** and must not be hand-edited — change the source and regenerate:

```bash
make docs           # tunables.md + registries.md + docs/README.md
make docs-check     # CI gate: fails if any generated page is stale vs the code
make docs-links     # CI gate: fails on a broken relative link in any tracked .md
```

`docs/reference/tunables.md` comes from `percept.tunables.REGISTRY`,
`docs/reference/registries.md` from the live registries, and `docs/README.md` from every page's
frontmatter — so none of them can drift from what ships.

**Strategy never enters tracked code or docs.** Code may cite a reference doc by path and may state
a capability bound plainly ("uncalibrated", "advisory", "not general"); it must never name a tier, a
plan, or a future offering. That material lives in `scratch/`.

## Pull requests

- Keep the diff focused; match the surrounding code's style and comment density.
- Include the test that proves the change; cite A/B numbers where relevant.
- Run `make test` + `make parity` before opening. Touching docs? Add `make docs-check docs-links`.

## Architecture

Start with [`docs/tutorials/quickstart.md`](docs/tutorials/quickstart.md),
[`docs/explanation/concepts.md`](docs/explanation/concepts.md), and
[`docs/explanation/architecture.md`](docs/explanation/architecture.md) — or the
[docs index](docs/README.md). `llms.txt` is the agent-readable SDK reference.
