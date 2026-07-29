#!/usr/bin/env python3
"""Generate `docs/configuration/tunables.md` from `percept.tunables.REGISTRY`.

The registry is the single source of truth for every behaviour-affecting knob (ADR-008). This
generator renders it to a reference page, so the page cannot drift from what ships: regenerate it
and diff. `make docs-tunables` runs this; CI can run it with `--check` to fail on drift.

    python3 tools/build_tunables_doc.py [--check]
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "packages" / "percept-vision"
OUT = REPO / "docs" / "configuration" / "tunables.md"

sys.path.insert(0, str(PKG))


def _fmt(value: object) -> str:
    """Render a default/bound for a markdown table cell."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return f"`{str(value).lower()}`"
    if isinstance(value, tuple):
        return " – ".join(_fmt(v).strip("`") for v in value)
    if isinstance(value, str):
        return f"`{value}`" if value else "—"
    return f"`{value}`"


def _pkg_version() -> str:
    """Read the version from ``pyproject.toml``, NOT ``percept.__version__``.

    ``percept.__version__`` resolves installed dist metadata first, so a stale editable install in
    the dev environment would stamp a wrong version onto a published page. The repo's own
    ``pyproject.toml`` is the authority for what this checkout documents.
    """
    import re  # noqa: PLC0415

    try:
        text = (PKG / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        return "unknown"
    m = re.search(r'(?m)^\s*version\s*=\s*"([^"]+)"', text)
    return m.group(1) if m else "unknown"


def render() -> str:
    from percept.tunables import REGISTRY  # noqa: PLC0415

    total = len(REGISTRY)
    safe = [s for s in REGISTRY if s.safe_to_optimize]
    owners: dict[str, list] = {}
    for spec in REGISTRY:
        owners.setdefault(spec.owner, []).append(spec)

    L: list[str] = []
    add = L.append

    add("---")
    add("title: Tunables")
    add(f"updated: {date.today().isoformat()}")
    add(f"applies_to: percept-vision {_pkg_version()}")
    add("status: stable")
    add("generated_by: tools/build_tunables_doc.py")
    add("---")
    add("")
    add("# Tunables — every behaviour-affecting knob")
    add("")
    add(
        "**This page is generated** from `percept.tunables.REGISTRY` by "
        "`tools/build_tunables_doc.py`. Do not hand-edit it — change the registry and regenerate "
        "(`make docs-tunables`). That is why it cannot drift from what ships."
    )
    add("")
    add(f"Percept exposes **{total} tunables**. Of those, **{len(safe)} are `safe_to_optimize`** — the")
    add("only ones `percept-optimize` is permitted to search. The rest are declared, typed, and owned,")
    add("but changing them is a deliberate act, not a search-space dimension.")
    add("")
    add("| | count |")
    add("|---|---|")
    add(f"| total tunables | **{total}** |")
    add(f"| safe to optimize | **{len(safe)}** |")
    for owner in sorted(owners, key=lambda o: -len(owners[o])):
        add(f"| owned by `{owner}` | {len(owners[owner])} |")
    add("")
    add("## How to read a row")
    add("")
    add("- **path** — the dotted registry path. This is the key an `OptimizationLock` or a")
    add("  `DomainPack.values` mapping writes to.")
    add("- **owner** — which configuration surface may legally set it. The resolver enforces this;")
    add("  a write from the wrong surface is refused, not silently applied.")
    add("- **env** — the `PERCEPT_*` variable, when one exists. Read once by `load_settings()`, so a")
    add("  run is reproducible from its environment.")
    add("- **risk** — blast radius if set wrong.")
    add("")
    add("---")
    add("")
    add("## The 31 safe-to-optimize tunables")
    add("")
    add("These are the search space. `percept-optimize --space space.json` may propose values inside")
    add("`bounds`; the resulting `OptimizationLock` is provenance-stamped and validated before it")
    add("takes effect.")
    add("")
    add("| path | owner | type | default | bounds | env | risk | note |")
    add("|---|---|---|---|---|---|---|---|")
    for s in sorted(safe, key=lambda x: (x.owner, x.path)):
        add(
            f"| `{s.path}` | {s.owner} | {s.value_type} | {_fmt(s.default)} | "
            f"{_fmt(s.bounds)} | {_fmt(s.env_var)} | {s.risk} | {s.note or ''} |"
        )
    add("")
    add("---")
    add("")
    add("## Full registry, by owner")
    add("")
    for owner in sorted(owners):
        specs = sorted(owners[owner], key=lambda x: x.path)
        n_safe = sum(1 for s in specs if s.safe_to_optimize)
        add(f"### `{owner}` — {len(specs)} tunables ({n_safe} safe to optimize)")
        add("")
        add("| path | type | default | bounds / choices | env | safe? | risk | note |")
        add("|---|---|---|---|---|---|---|---|")
        for s in specs:
            rng = _fmt(s.choices) if s.choices else _fmt(s.bounds)
            add(
                f"| `{s.path}` | {s.value_type} | {_fmt(s.default)} | {rng} | "
                f"{_fmt(s.env_var)} | {'✅' if s.safe_to_optimize else '—'} | {s.risk} | "
                f"{s.note or ''} |"
            )
        add("")
    add("---")
    add("")
    add("## Related")
    add("")
    add("- [Configuration reference](overview.md) — the `Settings` tree and resolver precedence.")
    add("- [Run an optimization](optimize.md) — searching the safe subset into a lock.")
    add("- [Create a custom configuration](profiles.md) — authoring a profile.")
    add("- [Tune admission](../admission/tuning-admission.md) — the knobs that decide what reaches a judge.")
    add("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if the checked-in page is stale")
    args = ap.parse_args()

    text = render()
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        # ignore the `updated:` line — regenerating on a new day is not drift
        norm = lambda t: "\n".join(  # noqa: E731
            ln for ln in t.splitlines() if not ln.startswith("updated:")
        )
        if norm(current) != norm(text):
            print(f"STALE: {OUT.relative_to(REPO)} — run `make docs-tunables`", file=sys.stderr)
            return 1
        print(f"ok: {OUT.relative_to(REPO)} matches the registry")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
