"""CI guard for self-contained public source and documentation.

Published comments must explain the programmable-Watch behavior, invariant, measurement, or public
design decision they support. Opaque work-item identifiers and non-public planning citations do not
give an adopter enough information to understand the code. A small allowlist covers technical tokens
that share the identifier shape, such as lint rules and media formats.

The contextual policy also scans tracked public documentation, examples, tests, and evaluation
tooling when they are present in a source checkout. Its patterns are assembled from fragments so the
policy source does not become a match merely by documenting what it rejects.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_THIS_DIR = Path(__file__).resolve().parent
_PERCEPT_VISION_ROOT = _THIS_DIR.parent  # packages/percept-vision
_PACKAGE_ROOT = _PERCEPT_VISION_ROOT / "percept"
_EDGE_SRC_ROOT = _PERCEPT_VISION_ROOT.parent / "percept-edge" / "src"
_REPO_ROOT = _PERCEPT_VISION_ROOT.parents[1]

# Catch compact uppercase work-item identifiers while allowing documented technical acronyms below.
_CARD_ID_RE = re.compile(r"\b[A-Z]{1,3}[0-9]{1,3}[A-Za-z]{0,2}\b")
_RB_OFFLINE_RE = re.compile(r"\bRB-OFFLINE\b")

# Legitimate technical tokens that happen to match the identifier shape.
_ALLOWLIST = frozenset(
    {
        # ruff/flake8 suppression rule codes
        "BLE001", "E402", "F403", "UP006", "SLF001", "F401", "E731", "D401", "ARG001",
        "PLC0415",
        # media / encoding / format acronyms
        "MP4", "PNG", "WAV", "JSON", "HTTP", "UTF8", "RGBA", "RGB", "H264", "H265",
        "SHA256", "UUID4", "B64",
        # transport/protocol acronyms
        "A2V", "S2S", "P2P",
        # `[A-Z0-9 ]` character class fragment matched inside a regex literal (privacy-text.js)
        "Z0",
        # documented cross-package vocabulary, reused for unrelated meanings in different modules
        # (test-lane tier documented in README.md/docs/reference/backends.md: L1=offline fakes,
        # L2/L3=live backends with keys; the L1 math norm in harness/appearance.py; the cache
        # invalidation-order step names in cache/framehash.py) — not board-card citations.
        "L0", "L1", "L2", "L3",
        # RFC 3551 audio MIME subtype ("audio/L16" = linear 16-bit PCM), not a card id.
        "L16",
    }
)

# Keep each rejected expression out of this file's literal text: the scanner must not need to exempt
# itself. These patterns target context, not legitimate uses such as a dataset's licence terms or a
# runtime's private-memory implementation details.
_PUBLIC_CONTEXT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b" + "board" + r"\s+" + "card" + r"\b",
        r"\b"
        + "card"
        + r"\s+(?:[A-Z]{1,3}\d{1,3}[A-Za-z]{0,2}|RB-"
        + "OFFLINE"
        + r")\b",
        r"\b" + "founder" + r"\s+(?:decision|sign[- ]?off)\b",
        r"\b" + "paying" + r"\s+(?:client|customer)(?:\s+teams?)?\b",
        r"\b" + "fly" + "wheel" + r"\b",
        r"\b" + "commercial" + r"[- ]strategy\b",
        r"\b" + "paid" + r"\s+(?:side|tier)\b",
        r"\b" + "OSS" + r"\s+side\b",
        r"\b" + "billing" + r"\s+credit\b",
        r"\b" + "price" + r"\s+escalation\b",
        r"\b" + "go-to-" + "market" + r"\b",
        r"\b" + "Viability" + r"\s*§",
        r"\b" + "impl" + r"(?:ementation)?\s+plan\s*§",
        r"\b" + "PC" + "PT" + r"-\d+\b",
        r"\b" + "direction" + r"\s+doc(?:ument)?\b",
    )
)

_PUBLIC_TEXT_SUFFIXES = frozenset({".md", ".py", ".js", ".ts", ".json", ".toml", ".yaml", ".yml"})
_PUBLIC_TREE_ROOTS = (
    _REPO_ROOT / "docs",
    _REPO_ROOT / "templates",
    _PERCEPT_VISION_ROOT / "percept",
    _PERCEPT_VISION_ROOT / "tests",
    _PERCEPT_VISION_ROOT / "eval",
    _PERCEPT_VISION_ROOT / "changelog.d",
    _PERCEPT_VISION_ROOT.parent / "percept-edge" / "src",
    _PERCEPT_VISION_ROOT.parent / "percept-edge" / "test",
)
_PUBLIC_TREE_FILES = (
    _REPO_ROOT / "CHANGELOG.md",
    _REPO_ROOT / "Makefile",
    _REPO_ROOT / "README.md",
    _PERCEPT_VISION_ROOT / "CHANGELOG.md",
    _PERCEPT_VISION_ROOT / "README.md",
    _PERCEPT_VISION_ROOT / "pyproject.toml",
)


def _iter_source_files() -> list[Path]:
    files = sorted(_PACKAGE_ROOT.rglob("*.py"))
    if _EDGE_SRC_ROOT.is_dir():
        files += sorted(_EDGE_SRC_ROOT.rglob("*.js"))
    return files


def _hits(path: Path) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8")
    out: list[tuple[int, str]] = []
    for m in _CARD_ID_RE.finditer(text):
        token = m.group(0)
        if token in _ALLOWLIST:
            continue
        line_no = text.count("\n", 0, m.start()) + 1
        out.append((line_no, token))
    for m in _RB_OFFLINE_RE.finditer(text):
        line_no = text.count("\n", 0, m.start()) + 1
        out.append((line_no, "RB-OFFLINE"))
    return out


_SOURCE_FILES = _iter_source_files()


def _iter_public_text_files() -> list[Path]:
    files = [path for path in _PUBLIC_TREE_FILES if path.is_file()]
    for root in _PUBLIC_TREE_ROOTS:
        if not root.is_dir():
            continue
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in _PUBLIC_TEXT_SUFFIXES
        )
    return sorted(set(files))


def _context_hits(path: Path) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8")
    out: list[tuple[int, str]] = []
    for pattern in _PUBLIC_CONTEXT_PATTERNS:
        for match in pattern.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            out.append((line_no, match.group(0)))
    return out


def test_source_files_were_discovered():
    # Guards against a path typo silently turning this whole test module into a no-op.
    assert len(_SOURCE_FILES) > 50, (
        f"expected many shipped source files under {_PACKAGE_ROOT}, found {len(_SOURCE_FILES)}"
    )


@pytest.mark.parametrize(
    "path",
    _SOURCE_FILES,
    ids=lambda p: str(p.relative_to(_PERCEPT_VISION_ROOT.parent)),
)
def test_no_card_id_citations(path: Path):
    hits = _hits(path)
    assert not hits, (
        f"{path}: internal board-card-id citation(s) leaked into shipped source: {hits}. "
        "Reword the citation into plain factual language describing the behavior — do not "
        "delete the surrounding documentation, and do not add the token to the allowlist unless "
        "it is a genuine false positive (lint code / encoding / protocol acronym)."
    )


def test_public_tree_has_self_contained_context():
    hits = [
        (str(path.relative_to(_REPO_ROOT)), line_no, match)
        for path in _iter_public_text_files()
        for line_no, match in _context_hits(path)
    ]
    assert not hits, (
        "public content contains opaque planning or private product-history references; replace each "
        f"with the technical behavior, invariant, measurement, or public ADR it supports: {hits[:50]}"
    )
