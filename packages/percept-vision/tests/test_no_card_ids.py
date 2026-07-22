"""CI guard: shipped source must not leak internal board-card-id citations.

This SDK's internal build process tracks work items as short "board card" ids (``Y12``, ``D8a``,
``X7``, ``H0``, ``M2``, ``RB-OFFLINE``, ...). Those ids are meaningless to anyone reading the
open-source wheel/npm package from outside the project, so they must never survive into shipped
comments/docstrings — documentation there should describe *behavior*, not cite an internal
tracker. This test greps the shipped ``percept`` package (and, when present in the checkout, the
``percept-edge`` JS source) for the card-id shape and fails the build on any real hit.

A small allowlist absorbs known false positives: ruff/flake8 ``# noqa: XXXX`` lint codes, and a
handful of media/encoding/protocol acronyms that happen to match the same shape (uppercase
letters + digits). If a NEW false positive shows up, add it to ``_ALLOWLIST`` with a one-line
reason rather than loosening the pattern.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_THIS_DIR = Path(__file__).resolve().parent
_PERCEPT_VISION_ROOT = _THIS_DIR.parent  # packages/percept-vision
_PACKAGE_ROOT = _PERCEPT_VISION_ROOT / "percept"
_EDGE_SRC_ROOT = _PERCEPT_VISION_ROOT.parent / "percept-edge" / "src"

# [A-Z]{1,3}[0-9]{1,3}[A-Za-z]{0,2} catches the observed card-id shapes: Y12, D8a, X7, H0, M2,
# R2b, M3R, A24, ... plus the hyphenated RB-OFFLINE form, checked separately below.
_CARD_ID_RE = re.compile(r"\b[A-Z]{1,3}[0-9]{1,3}[A-Za-z]{0,2}\b")
_RB_OFFLINE_RE = re.compile(r"\bRB-OFFLINE\b")

# Legitimate technical tokens that happen to match the card-id shape — never board-card
# citations. Ruff/flake8 noqa codes, media/encoding formats, and protocol/transport acronyms.
_ALLOWLIST = frozenset(
    {
        # ruff/flake8 `# noqa: <code>` rule codes
        "BLE001", "E402", "F403", "UP006", "SLF001", "F401", "E731", "D401", "ARG001",
        "PLC0415",
        # media / encoding / format acronyms
        "MP4", "PNG", "WAV", "JSON", "HTTP", "UTF8", "RGBA", "RGB", "H264", "H265",
        "SHA256", "UUID4", "B64",
        # transport/protocol acronyms
        "A2V", "S2S", "P2P",
        # `[A-Z0-9 ]` character class fragment matched inside a regex literal (privacy-text.js)
        "Z0",
    }
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
