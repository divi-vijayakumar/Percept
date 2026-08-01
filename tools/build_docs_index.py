#!/usr/bin/env python3
"""Stamp frontmatter on every page under `docs/` and regenerate `docs/README.md` from it.

Two jobs, one source of truth:

1. **Stamp** — every `docs/**/*.md` gets `title`, `updated`, `applies_to`, `status`. `updated` is
   seeded from the file's last git commit date (real staleness, not "the day I ran the script"),
   and is *preserved* on re-runs so a formatting pass doesn't fake a content update.
2. **Index** — `docs/README.md`'s tables are rebuilt from that frontmatter, so the index cannot
   claim a page exists that doesn't, or miss one that does.

    python3 tools/build_docs_index.py [--check]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
INDEX = DOCS / "README.md"
PKG = REPO / "packages" / "percept-vision"

#: Section order, folder -> (heading, blurb). Drives both the index and the on-disk layout.
SECTIONS: list[tuple[str, str, str]] = [
    ("get-started", "Get started", "Install it, run something that works, then build."),
    ("watches", "Watches", "The core object: a goal armed against a source, firing on the rising edge."),
    ("sources", "Sources", "What Percept watches — cameras, files, streams, screens, audio."),
    ("admission", "Admission — the eye", "Everything upstream of a model call. This is what makes Percept cheap."),
    ("judges", "Judges", "The models that answer \"is this condition true?\" — and how to bring your own."),
    ("incidents", "Incidents and delivery", "What comes out, and how it reaches your app."),
    ("configuration", "Configuration", "Every knob, who owns it, and how to tune it safely."),
    ("observability", "Observability", "Traces, replay, cost, and the quality frontier."),
    ("evaluation", "Evaluation and benchmarks", "Measure it on your data, and read the numbers honestly."),
    ("resources", "Resources", "How it works, the architecture, and the full CLI."),
    ("demo", "Demo", "A browser harness driving the real runtime."),
]

#: Curated one-line summaries. A page missing here falls back to its first prose sentence.
SUMMARY: dict[str, str] = {
    "get-started/quickstart.md": "Install, run offline on the scripted backend, then a real one.",
    "watches/creating-a-watch.md": "**Start here to embed Percept.** `run(source) → incidents`: sources, sinks, tuning, egomotion, cost, troubleshooting.",
    "watches/writing-goals.md": "Phrase goals and match the plan to the action, so quality isn't lost before the judge runs.",
    "configuration/profiles.md": "Author a `SourceProfile` / `WatchProfile` for your use case.",
    "judges/custom-specialist-judge.md": "Bring your own vision model or author a specialist judge; the conformance suite.",
    # first_sentence() splits on the period inside `percept.hosted_byok`, so this page needs a
    # curated summary rather than a derived one.
    "judges/hosted-byok.md": "Per-invocation provider credentials for a multi-tenant hosted service — routing, isolation, cancellation, deadlines, and the gated detection route.",
    "admission/tuning-admission.md": "From \"nothing fires\" to a tuned eye. **Percept ships the harness; you tune the eye for your content.**",
    "configuration/optimize.md": "`percept-optimize` — safe-tunable search into a validated `OptimizationLock`.",
    "evaluation/evaluate-your-data.md": "Per-concern precision/recall on your manifest, the paired A/B differential, and the CI gate.",
    "sources/source-profiles.md": "Turn labelled fixtures + a ladder into a provenance-stamped profile.",
    "observability/traces-and-replay.md": "Portable bug reports and deterministic replay.",
    "watches/scenario-porch-package.md": "A discrete physical-event watch recipe.",
    "watches/scenario-screenshare.md": "A fixed-cadence content watch.",
    "sources/voice-and-watchdog.md": "The voice-escalation seam and long-running watches.",
    "configuration/overview.md": "**Everything you can tune** — the `Settings` tree, every env var, and resolver precedence.",
    # The two counts are hand-carried from `percept.tunables.REGISTRY` (this generator does not
    # import the package) and had already drifted 4 rows behind it. Re-read them off the summary
    # table at the top of docs/configuration/tunables.md whenever a TunableSpec is added or removed.
    "configuration/tunables.md": "**All 162 tunables, and the 31 an optimizer may search.** Generated from the registry.",
    "judges/registries.md": "**Backend seams, detectors, and profile registries** — and how to add your own. Generated.",
    "resources/cli.md": "All nine console scripts, their flags, and install requirements.",
    "incidents/scope-and-persistence.md": "**Read first if you are designing around Percept** — what's covered, what's excluded, why episodes don't survive a restart.",
    "watches/watch-contract.md": "The frozen public entrypoint and the `WatchEvent` schema.",
    "watches/the-gate.md": "FIRE/ASK/SILENT, the rising edge, refractory, and the typed reason vocabulary.",
    "judges/backends.md": "The seams, backend names, extras, and custom backends.",
    "judges/bring-your-own-model.md": "The public extension point — bring your own model, fail-closed injection, conformance suite.",
    "admission/detectors.md": "Detector / concern / enricher registration in detail.",
    "admission/wake-policy.md": "The on-device \"should we spend a VLM call?\" layer.",
    "admission/edge-parity.md": "JS ↔ Python edge conformance.",
    "sources/audio.md": "Fused vision + audio judging.",
    "watches/driving-a-live-watch.md": "Drive a running watch — arm by voice, ask, pause/mute.",
    "observability/operating-frontier.md": "The recall / false-fire trade-off surface.",
    "evaluation/sizing-a-run.md": "Sizing a live evaluation run before you spend on it.",
    "evaluation/benchmarks.md": "**Start here** — status, how to run one, and how to read a precision number without being misled.",
    "evaluation/methodology.md": "How the benchmark is designed (no numbers).",
    "evaluation/running-benchmarks.md": "The scorecard, the audit gate, and the reproducible model-sweep runbook.",
    "resources/architecture.md": "The four tiers, the two-layer brain, the firing paths.",
    "resources/how-it-works.md": "The cognition / perception split, the gate, rising-edge firing, memory.",
    "resources/worked-example-milk-boilover.md": "A judged path, end to end.",
    "admission/how-admission-works.md": "The motion score and the temporal strip.",
    "get-started/coding-agents.md": "Install the `percept-agent` skill; the three things agents get wrong.",
    "demo/README.md": "Browser demos driving the **real** runtime — upload a video or watch a live stream.",
}

FM = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


def pkg_version() -> str:
    try:
        text = (PKG / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        return "unknown"
    m = re.search(r'(?m)^\s*version\s*=\s*"([^"]+)"', text)
    return m.group(1) if m else "unknown"


def git_date(path: Path) -> str:
    """Last commit date for `path`, or today if untracked/new."""
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%ad", "--date=short", "--", str(path)],
            cwd=REPO, capture_output=True, text=True, check=False,
        ).stdout.strip()
        return out or date.today().isoformat()
    except Exception:  # noqa: BLE001 — a doc generator never blocks on git shape
        return date.today().isoformat()


def parse_fm(text: str) -> tuple[dict[str, str], str]:
    m = FM.match(text)
    if not m:
        return {}, text
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    return meta, text[m.end():]


def h1(body: str) -> str | None:
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def first_sentence(body: str) -> str:
    for line in body.splitlines():
        s = line.strip()
        if s and not s.startswith(("#", ">", "-", "|", "*", "`", "!", "[")):
            return (s.split(". ")[0].rstrip(".") + ".")[:160]
    return ""


def stamp(path: Path, version: str) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    meta, body = parse_fm(text)

    if meta.get("generated_by"):          # generated pages own their own frontmatter
        return meta

    meta.setdefault("title", h1(body) or path.stem.replace("-", " ").capitalize())
    meta.setdefault("updated", git_date(path))    # preserved once set — no fake bumps
    meta["applies_to"] = f"percept-vision {version}"
    meta.setdefault("status", "stable")

    order = ["title", "updated", "applies_to", "status"]
    keys = order + [k for k in meta if k not in order]
    fm = "---\n" + "\n".join(f"{k}: {meta[k]}" for k in keys) + "\n---\n"
    new = fm + body.lstrip("\n")
    if new != text:
        path.write_text(new, encoding="utf-8")
    return meta


def render_index(pages: dict[str, dict[str, str]], version: str) -> str:
    L: list[str] = []
    add = L.append
    add("---")
    add("title: Percept documentation")
    add(f"updated: {date.today().isoformat()}")
    add(f"applies_to: percept-vision {version}")
    add("status: stable")
    add("generated_by: tools/build_docs_index.py")
    add("---")
    add("")
    add("# Percept documentation")
    add("")
    add("Everything you need to adopt Percept, for humans and for agents. Read online at")
    add("**[percept.percia.ai](https://percept.percia.ai)**.")
    add("")
    add("**This index is generated** by `tools/build_docs_index.py` from each page's frontmatter, so")
    add("it cannot list a page that doesn't exist or miss one that does. Every page carries an")
    add("`updated:` date — a page older than the release it documents is visibly stale.")
    add("")
    add("New here? → **[Quickstart](get-started/quickstart.md)** · want the whole surface? →")
    add("**[Tunables](configuration/tunables.md)** · using an agent? → **[Coding agents](get-started/coding-agents.md)**")
    add("")
    add("The architecture diagram is **generated from the code**")
    add("([`architecture.svg`](architecture.svg) · interactive [`architecture.html`](architecture.html))")
    add("— it cannot drift from what ships.")
    add("")

    for folder, heading, blurb in SECTIONS:
        rows = sorted(
            ((rel, m) for rel, m in pages.items() if rel.startswith(f"{folder}/")),
            key=lambda kv: (kv[0].endswith("README.md") is False, kv[0]),
        )
        if not rows:
            continue
        add(f"## {heading}")
        add("")
        add(f"*{blurb}*")
        add("")
        add("| page | what it covers | updated |")
        add("|---|---|---|")
        for rel, m in rows:
            add(f"| [{m.get('title', rel)}]({rel}) | {m.get('_summary', '')} | {m.get('updated', '')} |")
        add("")

    add("---")
    add("")
    add("## Not in these docs, on purpose")
    add("")
    add("- **Architecture decision records** live in [`.adr/`](../.adr/) — tracked and citable from")
    add("  code, but internal reasoning rather than adoption material.")
    add("- **Benchmark numbers** are not published yet. See [Benchmarks](evaluation/benchmarks.md) for")
    add("  the status and the gate a published claim must pass.")
    add("- **Per-module API reference** is the generated CodeWiki, regenerated from source.")
    add("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if the index is stale")
    args = ap.parse_args()

    version = pkg_version()
    pages: dict[str, dict[str, str]] = {}
    for p in sorted(DOCS.rglob("*.md")):
        rel = p.relative_to(DOCS).as_posix()
        if rel == "README.md" or "/_" in rel:
            continue
        meta = stamp(p, version)
        body = parse_fm(p.read_text(encoding="utf-8"))[1]
        meta["_summary"] = SUMMARY.get(rel) or first_sentence(body)
        pages[rel] = meta

    text = render_index(pages, version)
    if args.check:
        cur = INDEX.read_text(encoding="utf-8") if INDEX.exists() else ""
        norm = lambda t: "\n".join(  # noqa: E731
            ln for ln in t.splitlines() if not ln.startswith("updated:")
        )
        if norm(cur) != norm(text):
            print("STALE: docs/README.md — run `make docs-index`", file=sys.stderr)
            return 1
        print("ok: docs/README.md matches the pages on disk")
        return 0

    INDEX.write_text(text, encoding="utf-8")
    print(f"stamped {len(pages)} pages; wrote docs/README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
