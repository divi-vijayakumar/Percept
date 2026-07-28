#!/usr/bin/env python3
"""Fail on a broken relative link in any tracked markdown file.

A docs reorganization is exactly the moment links rot, and a 404 in the adoption path is a
first-impression bug. This walks every tracked `.md`, resolves each relative link target against the
linking file's own directory, and reports the ones that do not exist.

External links (`http:`, `https:`, `mailto:`, `tel:`) are not fetched — this is a filesystem check,
not a network one, so it stays fast and offline.

    python3 tools/check_doc_links.py [--all]      # --all: include gitignored .md too
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+?)(?:\s+\"[^\"]*\")?\)")
SKIP_SCHEME = ("http://", "https://", "mailto:", "tel:", "#")


def tracked_markdown() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "*.md"], cwd=REPO, capture_output=True, text=True, check=False
    ).stdout.split()
    return [REPO / f for f in out]


def all_markdown() -> list[Path]:
    skip = {".git", "node_modules", "__pycache__", ".venv", "venvs", "bench-staging"}
    return [p for p in REPO.rglob("*.md") if not (skip & set(p.relative_to(REPO).parts))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="include gitignored markdown")
    args = ap.parse_args()

    files = all_markdown() if args.all else tracked_markdown()
    broken: list[str] = []

    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        rel_file = path.relative_to(REPO)
        for m in LINK.finditer(text):
            target = m.group(1)
            if target.startswith(SKIP_SCHEME):
                continue
            target = target.split("#", 1)[0]     # strip an anchor; we check the file, not the anchor
            if not target:
                continue
            if not (path.parent / target).resolve().exists():
                line = text[: m.start()].count("\n") + 1
                broken.append(f"{rel_file}:{line} -> {target}")

    print(f"checked {len(files)} markdown files")
    if broken:
        print(f"BROKEN relative links: {len(broken)}", file=sys.stderr)
        for b in broken:
            print(f"  {b}", file=sys.stderr)
        return 1
    print("all relative links resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
