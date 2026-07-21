"""Synthetic UNIT-layer capability corpora — the deterministic half of the testing split.

    CAPABILITY IS DETERMINISTIC AND BELONGS IN UNIT TESTS.
    QUALITY IS STATISTICAL AND BELONGS IN AN END-TO-END BENCHMARK AT RELEASE.

This package is the UNIT layer. Seven labelled corpora — subject identity, continuation vs new
episode, mutation/speech-act, episode counting (the cup problem), intervals, idempotency and long
horizon — each with a seeded generator, machine-readable ground truth stating the EXPECTED ANSWER,
and a loader. Everything is synthetic, deterministic, cheap, and drivable by
:class:`percept.harness.scripted.ScriptedJudge` with **no API key and no spend**.

What a green corpus means: the mechanism can EXPRESS the right answer. What it never means: a real
model would produce it on real footage. That claim belongs to the release benchmark.

The generated data lives in the repo's ``eval/synthetic/`` and is **gitignored** — the generator is
the artefact under version control, the data is regenerated on demand::

    python3 -m percept.harness.synthetic.generate
"""

from __future__ import annotations

from ..scripted import ScriptedExhausted, ScriptedJudge, verdict_from_spec
from .corpora import BUILDERS
from .loader import (
    Case,
    Corpus,
    CorpusMissing,
    available,
    corpus_dir,
    load_all,
    load_corpus,
    synthetic_root,
)
from .schema import (
    ACTIONS,
    BINDINGS,
    CORPUS_KIND,
    CORPUS_NAMES,
    REQUIRED_EXPECT_KEYS,
    SCHEMA_VERSION,
    SEED,
    SPEECH_ACTS,
    ValidationError,
    validate_corpus,
)


def __getattr__(name: str):
    """Lazy re-export of the generator entry points.

    Imported on demand rather than eagerly, so ``python3 -m percept.harness.synthetic.generate``
    does not trip runpy's "already in sys.modules" warning.
    """
    if name in ("generate_all", "generate_corpus"):
        from . import generate as _generate

        return getattr(_generate, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ScriptedJudge",
    "ScriptedExhausted",
    "verdict_from_spec",
    "BUILDERS",
    "generate_all",
    "generate_corpus",
    "Case",
    "Corpus",
    "CorpusMissing",
    "available",
    "corpus_dir",
    "load_all",
    "load_corpus",
    "synthetic_root",
    "ACTIONS",
    "BINDINGS",
    "CORPUS_KIND",
    "CORPUS_NAMES",
    "REQUIRED_EXPECT_KEYS",
    "SCHEMA_VERSION",
    "SEED",
    "SPEECH_ACTS",
    "ValidationError",
    "validate_corpus",
]
