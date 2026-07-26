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
import subprocess
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
        r"\b" + "board" + r"\s+the\s+" + "impl" + r"(?:ementation)?\b",
        r"\b"
        + "card"
        + r"\s+(?:(?-i:[A-Z]{1,3})\d{1,3}[A-Za-z]{0,2}|(?-i:RB-"
        + "OFFLINE"
        + r"))\b",
        r"\b(?:this|that|whole|separate|later|future|follow[- ]?up|sibling|"
        + "impl"
        + r"(?:ementation)?)\s+"
        + "card"
        + r"(?:['’]s|s)?\b",
        r"\b"
        + "card"
        + r"['’]s\s+(?:scope|problem|reuse|file|decision|contract|work|concern)\b",
        r"\b"
        + "card"
        + r"\s+(?:exists|must|wires?|builds?|ships?|resolves?|consumes?|forbids?|owns?|"
        r"that\s+(?:builds?|ships?|resolves?|consumes?))\b",
        r"\b" + "found" + "er" + r"\s+(?:decision|sign[- ]?off)\b",
        r"\b" + "found" + "er" + r"(?:['’]s)?\b",
        r"\b" + "paying" + r"\s+(?:client|customer)(?:\s+teams?)?\b",
        r"\b(?:a|one|two|three|\d+)\s+(?:client|customer)\s+teams?\b",
        r"\b(?:" + "found" + "er" + r"|client|customer)[- ](?:handoff|history)\b",
        r"\b(?:internal|private)\s+(?:product[- ]?)?" + "strategy" + r"\b",
        r"\b" + "fly" + "wheel" + r"\b",
        r"\b" + "commercial" + r"[- ]strategy\b",
        r"\b" + "paid" + r"\s+(?:side|tier)\b",
        r"\b" + "OSS" + r"\s+side\b",
        r"\b" + "billing" + r"\s+credit\b",
        r"\b" + "price" + r"\s+escalation\b",
        r"\b" + "go-to-" + "market" + r"\b",
        r"\b" + "Viability" + r"\s*§",
        r"\b" + "spec" + r"\s*§",
        r"\b" + "impl" + r"(?:ementation)?\s+plan\s*§",
        r"\b" + "impl" + r"(?:ementation)?\s+plan\s+epic\b",
        r"\b" + "PC" + "PT" + r"-\d+\b",
        r"\b" + "direction" + r"\s+doc(?:ument)?\b",
        r"\b" + "MVP" + r"\s+cards?\b",
        r"\b" + "epic" + r"\s+[A-Z]\d+[A-Za-z]?\b",
        r"\b" + "scratch" + r"/",
        r"\b(?-i:" + "Spec" + r"/)",
    )
)
_GITIGNORE_LOCAL_RULES = {"scr" + "atch/", "/" + "Spec" + "/"}
_COMPACT_REFERENCE_ALLOWLIST = _ALLOWLIST | frozenset(
    {
        # Standard integration and modality abbreviations.
        "E2E", "A2V", "V2V",
        # Public metric / release-gate names used and defined in benchmark documents.
        "F1", "G1", "G2", "G3",
        # Locally defined requirement and question labels, not cross-document work items.
        "R1", "R2", "R2b", "R3", "R4", "R6", "R7", "R15",
        "Q1", "Q2", "Q3", "Q4", "Q5", "Q6",
        # Declared sample-size and pose-channel names.
        "N8", "N10", "PC1",
        # Literal synthetic event/entity ids used as test data.
        "T000", "B000", "CUP1", "CUP2",
    }
)

# Compact work-item references need context: the same token shape is also used by public benchmark
# tracks, codecs, locally defined requirement labels, and dataset-native ids. Keep the tokens
# assembled so this policy remains clean itself.
_OPAQUE_ID = r"(?-i:[A-Z]{1,3})\d{1,3}[A-Za-z]{0,2}"
_OPAQUE_ID_LIST = _OPAQUE_ID + r"(?:\s*(?:/|,|–|—|\bto\b)\s*" + _OPAQUE_ID + r")*"
_COMPACT_CONTEXT_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\b[Cc]ards?\s+(?:" + _OPAQUE_ID_LIST + r")",
        r"\b(?:pre|post)-(?:" + _OPAQUE_ID + r")\b",
        r"\bpending\s+(?:" + _OPAQUE_ID + r")\b",
        r"\b(?:" + _OPAQUE_ID + r")(?:/(?:" + _OPAQUE_ID + r"))+\b",
        r"\bADR-\d{3}/(?:" + _OPAQUE_ID + r")\b",
        r"\((?:" + _OPAQUE_ID + r")\)",
        r"(?m)^\s*(?:(?://|#|\*)\s*)?(?:[\"']{3})?(?:"
        + _OPAQUE_ID
        + r")\s*(?::|—|-)\s+",
        r"\b(?:" + _OPAQUE_ID + r")(?:['’]s)\b",
        r"\b(?:the|per)\s+(?:" + _OPAQUE_ID + r")\b",
        r"\b(?:before|after|until|from)\s+(?:" + _OPAQUE_ID + r")\b",
        r"\b(?:" + _OPAQUE_ID + r")\s+coordination\s+note\b",
        r"\b(?:"
        + _OPAQUE_ID
        + r")\s+(?:frozen|claim|judgment|shadow|correlation|resolver|clip|tier-0|"
        r"calibration|declaration|channel|contract|envelope|mode|behavior|behaviour|"
        r"entrypoint|trace|change|fix|implementation|path|seam|rules?|boundary|fixture|"
        r"default|ablation|set|brief|scope|fields?|window|identity|evidence|episode|fold|"
        r"refactor|failure|tracker|assertion|note|wiring|profiles?|corpus)\b",
        r"\b(?:"
        + _OPAQUE_ID
        + r")\s+(?:lands?|landed|ships?|shipped|owns?|wired|left|collapsed|adds?|"
        r"makes?|keeps?|classifies|minted|builds?|recomputes|carries|resolves?|fixes?|"
        r"proved|flips?|bumped|refuses?|closed|predating|derived|carried|touched|settled|"
        r"reuses?)\b",
    )
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
    """Return the authoritative public release surface from Git's tracked-file index.

    Directory walking can silently omit a new public root and can accidentally inspect ignored local
    material. In an installed artifact without Git metadata, fall back narrowly to shipped source.
    """
    completed = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "ls-files", "-z"],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        return _SOURCE_FILES
    return sorted(
        _REPO_ROOT / raw.decode("utf-8")
        for raw in completed.stdout.split(b"\0")
        if raw
    )


def _context_hits(path: Path) -> list[tuple[int, str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []
    out: list[tuple[int, str]] = []
    for pattern in _PUBLIC_CONTEXT_PATTERNS:
        for match in pattern.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            if path == _REPO_ROOT / ".gitignore":
                line = text.splitlines()[line_no - 1].strip()
                if line in _GITIGNORE_LOCAL_RULES:
                    continue
            out.append((line_no, match.group(0)))
    for pattern in _COMPACT_CONTEXT_PATTERNS:
        for match in pattern.finditer(text):
            tokens = [
                token
                for token in _CARD_ID_RE.findall(match.group(0))
                if token not in _COMPACT_REFERENCE_ALLOWLIST
            ]
            if not tokens:
                continue
            line_no = text.count("\n", 0, match.start()) + 1
            out.append((line_no, match.group(0)))
    return out


def test_public_tree_scope_is_the_git_index():
    completed = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "ls-files", "-z"],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        pytest.skip("Git index is unavailable in the installed artifact")

    indexed = {
        _REPO_ROOT / raw.decode("utf-8")
        for raw in completed.stdout.split(b"\0")
        if raw
    }
    scanned = set(_iter_public_text_files())
    assert scanned == indexed
    assert {
        _REPO_ROOT / "docs" / "demo" / "serve.py",
        _REPO_ROOT / "eval" / "README.md",
        _REPO_ROOT / "examples" / "byo_admission_detector.py",
        _REPO_ROOT / "tools" / "build_architecture.py",
        _REPO_ROOT / "packages" / "percept-vision" / "eval" / "BENCHMARKS.md",
    } <= scanned


@pytest.mark.parametrize(
    "text",
    (
        "Cards " + "Y" + "2–" + "Y" + "15 govern this.",
        "A pre-" + "D" + "11 record remains readable.",
        "The " + "Y" + "5/" + "Z" + "3 path is additive.",
        "(M" + "1) froze the entrypoint.",
        "the " + "Z" + "3 judgment envelope",
        "ADR-005/" + "M" + "2",
        "// " + "D" + "11: correlation handle",
        "pending " + "A" + "30",
        "post-" + "A" + "30",
        "module docstring's " + "A" + "30 coordination note",
    ),
)
def test_compact_internal_reference_forms_are_rejected(text: str, tmp_path: Path):
    path = tmp_path / "public.md"
    path.write_text(text, encoding="utf-8")
    assert _context_hits(path), text


@pytest.mark.parametrize(
    "text",
    (
        "F1 score and G3 latency gate",
        "Run the E2E benchmark over A2V/V2V inputs.",
        "R1/R2 are locally defined requirements.",
        "Dataset sample G330 uses pose channel PC1.",
    ),
)
def test_legitimate_compact_technical_identifiers_are_allowed(text: str, tmp_path: Path):
    path = tmp_path / "public.md"
    path.write_text(text, encoding="utf-8")
    assert not _context_hits(path), text


@pytest.mark.parametrize(
    "text",
    (
        "board the " + "implementation",
        "this " + "card" + "'s reuse list",
        "implementation " + "card" + "'s problem",
        "the whole " + "card",
        "a separate " + "card",
        "later " + "cards",
        "the " + "card" + " that builds this",
    ),
)
def test_generic_internal_planning_narratives_are_rejected(text: str, tmp_path: Path):
    path = tmp_path / "public.md"
    path.write_text(text, encoding="utf-8")
    assert _context_hits(path), text


def test_public_benchmark_card_language_is_allowed(tmp_path: Path):
    path = tmp_path / "public.md"
    path.write_text("a benchmark card that renders DRAFT", encoding="utf-8")
    assert not _context_hits(path)


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
