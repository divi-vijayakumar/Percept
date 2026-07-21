"""Generate the synthetic capability corpora into ``eval/synthetic/`` (gitignored).

    python3 -m percept.harness.synthetic.generate            # into the repo's eval/synthetic
    python3 -m percept.harness.synthetic.generate --out DIR  # anywhere else
    python3 -m percept.harness.synthetic.generate --check    # regenerate + validate, write nothing

Deterministic by construction: every generator is a pure function of ``schema.SEED`` and module
constants — no wall clock, no unseeded randomness, no network. Re-running overwrites with
byte-identical JSON, which is what makes "the data is gitignored, regenerate it" safe.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .corpora import BUILDERS
from .loader import synthetic_root
from .schema import CORPUS_NAMES, SEED, validate_corpus

__all__ = ["generate_all", "generate_corpus", "main"]


def _write(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def generate_corpus(name: str, out_root: Path, *, write: bool = True) -> tuple[dict, dict]:
    """Build (and optionally write) one corpus. Always validates before writing."""
    stream, labels = BUILDERS[name]()
    validate_corpus(name, stream, labels)
    if write:
        _write(out_root / name / "stream.json", stream)
        _write(out_root / name / "labels.json", labels)
    return stream, labels


def generate_all(out_root: "Path | str | None" = None, *, write: bool = True) -> dict[str, tuple[dict, dict]]:
    """Build every corpus, plus the top-level manifest."""
    root = Path(out_root) if out_root else synthetic_root()
    built = {name: generate_corpus(name, root, write=write) for name in CORPUS_NAMES}
    if write:
        # The README ships WITH the data, not with the tree: eval/synthetic/ is gitignored, so a
        # committed README there would vanish on a fresh clone. Its text lives in readme.py.
        from .readme import render

        root.mkdir(parents=True, exist_ok=True)
        (root / "README.md").write_text(render(), encoding="utf-8")
        _write(
            root / "manifest.json",
            {
                "seed": SEED,
                "layer": "unit",
                "synthetic": True,
                "gitignored": True,
                "regenerate": "python3 -m percept.harness.synthetic.generate",
                "warning": (
                    "CAPABILITY fixtures. Deterministic, scripted-judge-drivable, no API key. They prove "
                    "a mechanism can express an answer; they prove NOTHING about real-world accuracy."
                ),
                "corpora": {
                    name: {
                        "kind": stream["kind"],
                        "cases": [c["id"] for c in stream["cases"]],
                        "step_counts": [len(c["steps"]) for c in stream["cases"]],
                    }
                    for name, (stream, _labels) in built.items()
                },
            },
        )
    return built


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(prog="percept.harness.synthetic.generate", description=__doc__)
    ap.add_argument("--out", default=None, help="output root (default: <repo>/eval/synthetic)")
    ap.add_argument("--check", action="store_true", help="build + validate only; write nothing")
    args = ap.parse_args(argv)

    root = Path(args.out) if args.out else synthetic_root()
    built = generate_all(root, write=not args.check)
    total = sum(len(s["cases"]) for s, _ in built.values())
    steps = sum(len(c["steps"]) for s, _ in built.values() for c in s["cases"])
    verb = "validated" if args.check else f"wrote -> {root}"
    print(f"{len(built)} corpora, {total} cases, {steps} steps (seed={SEED}) {verb}")
    for name, (stream, _labels) in built.items():
        print(f"  {name:<18} {stream['kind']:<6} {len(stream['cases'])} case(s)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
