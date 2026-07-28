#!/usr/bin/env python3
"""Generate `docs/reference/registries.md` from the live registries.

Percept resolves every vendor dependency through a *name -> factory* registry, so the core never
imports an SDK directly. This page enumerates what is registered in a stock build and how to add
your own. It is generated, so a new backend or detector shows up here by existing, not by someone
remembering to document it.

    python3 tools/build_registries_doc.py [--check]
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "packages" / "percept-vision"
OUT = REPO / "docs" / "reference" / "registries.md"

sys.path.insert(0, str(PKG))

#: What each backend seam is *for*, in caller vocabulary. Keyed by seam name.
SEAM_PURPOSE = {
    "vision": "the general judge — decides *is `<goal>` true?* for an admitted frame",
    "specialist": "a narrow, authorized judge bound to a verified static context",
    "conductor": "turns a plain-language goal into a runnable plan (goal compilation)",
    "stt": "speech to text, for voice-armed watches and fused audio judging",
    "tts": "text to speech, for spoken escalation",
    "transport": "the realtime voice/session transport seam",
    "extractor": "pulls structured fields off a frame (OCR, form fields)",
}


def _pkg_version() -> str:
    """Authoritative for this checkout: pyproject, never a possibly-stale installed dist."""
    try:
        text = (PKG / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        return "unknown"
    m = re.search(r'(?m)^\s*version\s*=\s*"([^"]+)"', text)
    return m.group(1) if m else "unknown"


def render() -> str:
    from percept.harness.detectors import _REGISTRY as DETECTORS  # noqa: PLC0415
    from percept.registry import _REGISTRY as BACKENDS  # noqa: PLC0415
    from percept.registry import _SEAMS  # noqa: PLC0415

    L: list[str] = []
    add = L.append

    add("---")
    add("title: Registries and adapters")
    add(f"updated: {date.today().isoformat()}")
    add(f"applies_to: percept-vision {_pkg_version()}")
    add("status: stable")
    add("generated_by: tools/build_registries_doc.py")
    add("---")
    add("")
    add("# Registries and adapters")
    add("")
    add(
        "**This page is generated** by `tools/build_registries_doc.py` from the live registries. "
        "A backend or detector appears here by *being registered*, not by someone remembering to "
        "document it."
    )
    add("")
    add("Percept's core never imports a vendor SDK. Every external capability resolves through a")
    add("**name → factory** registry, which is what makes a backend swappable by string and testable")
    add("with a double. There are three kinds of registry, and they answer different questions.")
    add("")
    add("| registry | answers | add to it with |")
    add("|---|---|---|")
    add(
        "| **Backend seams** | *who performs this external capability?* | "
        "`percept.registry.register(seam, name, factory)` or the `percept.backends` entry point |"
    )
    add(
        "| **Detectors** | *what cheap local measurement can we take?* | "
        "`percept.harness.detectors.register(name, factory, manifest)` |"
    )
    add(
        "| **Profiles / packs** | *what configuration applies to this world?* | "
        "`DEFAULT_*_REGISTRY.register(obj)` per surface |"
    )
    add("")
    add("---")
    add("")
    add("## 1. Backend seams")
    add("")
    add(f"**{len(_SEAMS)} seams.** Each maps a name to a factory that takes `settings` and returns")
    add("an adapter. Selection is by string (`Percept.create(vision=\"gemini\")`), by adapter")
    add("instance, or by env (`PERCEPT_VISION_BACKEND`).")
    add("")
    for seam in _SEAMS:
        names = sorted(BACKENDS[seam])
        add(f"### `{seam}`")
        add("")
        add(f"{SEAM_PURPOSE.get(seam, '')}")
        add("")
        add(f"Registered in a stock build ({len(names)}): " + ", ".join(f"`{n}`" for n in names))
        add("")
    add("**Naming.** `scripted` is the canonical keyless double (`fake` is its historical alias);")
    add("you select it **out loud** via `mode=\"offline\"` and never fall onto it by omission. A live")
    add("backend with a missing key prints one actionable line, never a stack trace.")
    add("")
    add("**Registering your own:**")
    add("")
    add("```python")
    add("from percept.registry import register")
    add("")
    add("def my_vision_factory(settings):")
    add("    return MyVisionAdapter(model=settings.models.vision)")
    add("")
    add('register("vision", "my-backend", my_vision_factory)')
    add('core = percept.Percept.create(vision="my-backend")')
    add("```")
    add("")
    add("Or ship it as a plugin — declare an entry point in the **`percept.backends`** group and it")
    add("is discovered on import, no call required:")
    add("")
    add("```toml")
    add("[project.entry-points.\"percept.backends\"]")
    add('my-backend = "my_pkg.adapters:register"')
    add("```")
    add("")
    add("Your adapter must pass the runnable conformance suite before it is trusted — see")
    add("[Vision adapter interface](vision-adapter.md) and [Custom judge](../how-to/custom-judge.md).")
    add("")
    add("---")
    add("")
    add("## 2. Detectors")
    add("")
    add(f"**{len(DETECTORS)} registered.** Detectors are the cheap, local, model-free measurements")
    add("that run *before* any judge is considered. Each ships a `DetectorManifest` declaring what it")
    add("senses and when to use it, so a plan can be compiled against capability rather than guesswork.")
    add("")
    add("| name | category | senses | output | when to use |")
    add("|---|---|---|---|---|")
    for name, entry in sorted(DETECTORS.items()):
        man = entry[1]  # (factory, manifest)
        senses = getattr(man, "senses", None)
        if isinstance(senses, (list, tuple)):
            senses = ", ".join(f"`{s}`" for s in senses)
        cells = [
            f"`{name}`",
            man.category or "—",
            senses or "—",
            getattr(man, "output", None) or "—",
            getattr(man, "when_to_use", None) or "—",
        ]
        # a stray pipe in manifest prose would split the row
        add("| " + " | ".join(str(c).replace("|", "\\|") for c in cells) + " |")
    add("")
    add("> `_REGISTRY[name]` is a **`(factory, manifest)` tuple**, not a bare factory. Unpack it.")
    add("")
    add("Registering your own detector is the supported way to plug in a domain measurement — see")
    add("[Registration](registration.md) and [Tune admission](../how-to/tune-admission.md).")
    add("")
    add("---")
    add("")
    add("## 3. Configuration registries")
    add("")
    add("Six ordered configuration surfaces resolve into one runtime (ADR-008). Four of them carry a")
    add("process-wide default registry you can register named objects into and then select by name.")
    add("")
    add("| surface | default registry | what it decides |")
    add("|---|---|---|")
    add(
        "| `SourceProfile` | `percept.sourceprofile.DEFAULT_SOURCE_REGISTRY` | "
        "what KIND of world this is (fixed camera, egomotion, screen) |"
    )
    add(
        "| `WatchProfile` | `percept.watchprofile.DEFAULT_REGISTRY` | "
        "cadence, gate, salience for one watch |"
    )
    add(
        "| `ExecutionPolicy` | `percept.execution_policy.DEFAULT_EXECUTION_POLICY_REGISTRY` | "
        "deployment-wide policy (e.g. strict admission, cadence ceilings) |"
    )
    add(
        "| `DomainPack` | `percept.domain_pack.DEFAULT_DOMAIN_PACK_REGISTRY` | "
        "a bundle of goals + detectors + tunable values for a domain |"
    )
    add("")
    add("These ship **empty** — they are extension points, not a preset catalog. Register your own:")
    add("")
    add("```python")
    add("from percept.watchprofile import DEFAULT_REGISTRY, WatchProfile")
    add("")
    add("DEFAULT_REGISTRY.register(WatchProfile(name=\"kitchen\", environment=\"fixed_camera\", ...))")
    add("core = percept.Percept.configure(watch_profile=\"kitchen\")")
    add("```")
    add("")
    add("Ownership is enforced at resolve time: a surface may only write tunables it owns, and a")
    add("write from the wrong surface is **refused, not silently applied**. The full ownership map is")
    add("the `owner` column in [Tunables](tunables.md).")
    add("")
    add("---")
    add("")
    add("## Related")
    add("")
    add("- [Tunables](tunables.md) — all 157 knobs, and which surface owns each.")
    add("- [Configuration reference](configuration.md) — the `Settings` tree and resolver precedence.")
    add("- [Backends](backends.md) — the seams in prose, extras, and selection.")
    add("- [Vision adapter interface](vision-adapter.md) — the extension contract + conformance suite.")
    add("- [Registration](registration.md) — detector / concern / enricher registration in detail.")
    add("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if the checked-in page is stale")
    args = ap.parse_args()

    text = render()
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        norm = lambda t: "\n".join(  # noqa: E731
            ln for ln in t.splitlines() if not ln.startswith("updated:")
        )
        if norm(current) != norm(text):
            print(f"STALE: {OUT.relative_to(REPO)} — run `make docs-registries`", file=sys.stderr)
            return 1
        print(f"ok: {OUT.relative_to(REPO)} matches the registries")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
