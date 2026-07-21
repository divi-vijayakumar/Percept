"""Shared builders for the synthetic corpus generators.

Determinism is the whole point: no wall clock, no unseeded randomness, no I/O. A generator is a pure
function of :data:`percept.harness.synthetic.schema.SEED` and the constants in this package, so
regenerating the corpus on any machine at any time produces byte-identical JSON.
"""

from __future__ import annotations

from typing import Any

from ..schema import SCHEMA_VERSION, SEED

__all__ = ["frame_step", "turn_step", "stream_doc", "labels_doc", "obs", "DOC_BOX", "CONTINUITY_GAP_S"]

#: One screen region, reused verbatim across cases so IoU is ~1.0 between them. That is deliberate:
#: it is exactly the condition under which an IoU+label identity model collapses two subjects into one.
DOC_BOX = [0.30, 0.20, 0.25, 0.35]

#: The continuity gap the continuation corpus is labelled against (seconds of absence after which the
#: same subject starts a NEW episode rather than continuing the old one).
CONTINUITY_GAP_S = 30.0


def obs(label: str, *, box: list[float] | None = None, confidence: float = 0.9, **attrs: str) -> dict:
    """One detected thing in a frame, as the fixture wire shape."""
    return {
        "label": label,
        "box": list(box or DOC_BOX),
        "confidence": confidence,
        "attrs": {k: v for k, v in attrs.items()},
    }


def frame_step(t: float, frame: str, *, observations: list | None = None, verdict: dict | None = None,
               **extra: Any) -> dict:
    """A frame-kind step: what the runtime sees at ``t``, plus the verdict a scripted judge returns."""
    step: dict[str, Any] = {"t": float(t), "frame": frame}
    if observations:
        step["observations"] = observations
    if verdict is not None:
        step["verdict"] = verdict
    step.update(extra)
    return step


def turn_step(t: float, speaker: str, text: str, **extra: Any) -> dict:
    """A turn-kind step: who said what, at ``t``."""
    step: dict[str, Any] = {"t": float(t), "speaker": speaker, "text": text}
    step.update(extra)
    return step


def stream_doc(name: str, kind: str, generator: str, cases: list[dict], **extra: Any) -> dict:
    doc = {
        "schema_version": SCHEMA_VERSION,
        "corpus": name,
        "kind": kind,
        "synthetic": True,
        "seed": SEED,
        "generator": generator,
        "cases": cases,
    }
    doc.update(extra)
    return doc


def labels_doc(name: str, cases: list[dict], **extra: Any) -> dict:
    doc = {"schema_version": SCHEMA_VERSION, "corpus": name, "cases": cases}
    doc.update(extra)
    return doc
