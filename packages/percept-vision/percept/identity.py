"""Identity — the "same thing?" question (spec §5.4).

Identity is established *below the LLM* and handed up as a stable symbol (invariant 4). Two impls
sit behind the frozen ``Identity`` protocol (``match(obs, known) -> id | None``):

* :class:`ContinuityIdentity` — IoU + label, frame-to-frame. The baseline default; correct for a
  single dominant object under a near-static camera (state the bound plainly — it is not general).
* :class:`AppearanceIdentity` — continuity first, then a brute-force **appearance** match over the
  recent set so an object is recognized after it leaves and re-enters tracking, and a genuinely new
  object is distinguished. The fleet-calibrated, high-accuracy re-id service is a paid tier.

Both accept an optional keyword ``signature`` (the appearance embedding of the observation). The
frozen protocol is ``match(obs, known)``; the extra keyword is optional, so these remain structural
conformers — ``ContinuityIdentity`` simply ignores it.
"""

from __future__ import annotations

import math
from typing import Any

from .adapters import Observation

# An appearance embedding (e.g. a tuple/list of floats). Opaque to the core; produced by an
# injected embedder (CLIP/etc. in real use, a stub in tests).
Signature = Any


def cosine(a: Any, b: Any) -> float:
    """Cosine similarity of two numeric sequences. 0.0 when either is empty/None/zero-norm."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class ContinuityIdentity:
    """IoU + label continuity. Stable ids while tracking holds; no memory of gaps."""

    def __init__(self, iou_match: float = 0.5) -> None:
        self.iou_match = iou_match

    def match(self, obs: Observation, known: list, *, signature: Signature = None) -> str | None:
        best, best_iou = None, self.iou_match
        for e in known:
            if e.label != obs.label:
                continue
            iou = e.box.iou(obs.box)
            if iou >= best_iou:
                best, best_iou = e, iou
        return best.id if best is not None else None


class AppearanceIdentity:
    """Continuity first, then appearance match over the recent set (brute-force).

    The signature comparison is cosine similarity against each known entity's retained
    ``appearance`` signature; the nearest above ``threshold`` (and same label) wins. With no
    signature available it degrades to pure continuity.
    """

    def __init__(self, threshold: float = 0.8, iou_match: float = 0.5) -> None:
        self.threshold = threshold
        self.iou_match = iou_match
        self._continuity = ContinuityIdentity(iou_match)

    def match(self, obs: Observation, known: list, *, signature: Signature = None) -> str | None:
        # 1) cheap exact path — tracking still holds.
        cont = self._continuity.match(obs, known)
        if cont is not None:
            return cont
        # 2) appearance fallback — the object left tracking and re-entered.
        if signature is None:
            return None
        best, best_sim = None, self.threshold
        for e in known:
            if e.label != obs.label or getattr(e, "appearance", None) is None:
                continue
            sim = cosine(signature, e.appearance)
            if sim >= best_sim:
                best, best_sim = e, sim
        return best.id if best is not None else None


__all__ = ["Signature", "cosine", "ContinuityIdentity", "AppearanceIdentity"]
