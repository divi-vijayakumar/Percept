"""Event-span data association — Build 4, the centerpiece of the individuation blueprint.

Reps, steps, onsets, and deduped VLM observations are all the same problem wearing different clothes:
*many observations of one underlying event*, which must collapse to ONE event in physical units. That is
exactly the data-association idea behind multi-object tracking (SORT/DeepSORT) — except the thing being
associated is an **event SPAN** on a timeline, not a box in an image.

So instead of a per-sense dedup (the rhythm counter's period-debounce, the EventLog's merge-gap, the VLM
running-summary's paraphrase-collapse), there is ONE engine here. Everything becomes an
:class:`EventCandidate` ``{kind, start_t, anchor_t, stop_t, source, amplitude, score, descriptor}`` and is
fed to an associate→merge→suppress loop:

  * **associate** — a new candidate joins an existing event if they are the same *kind* and either their
    spans OVERLAP or their anchors fall within a **seconds-refractory**, and their identity agrees
    (matching ``descriptor`` when present — "remove the seeds" ≈ "removing seeds" — else same ``source``).
  * **merge** — the joined event's span widens to cover both; the higher-score candidate keeps the anchor.
  * **suppress / split** — a duplicate inside the refractory is absorbed (suppressed); a candidate that
    overlaps nothing and is beyond the refractory opens a NEW event (a split). The split is on *evidence of
    a new cycle/step*, never merely "a new frame" — so denser observation cannot manufacture events.

Everything is in seconds + descriptors, never frame indices, so the deduped event set is invariant to how
densely the stream was observed (the SDK promise: "denser observation will not create duplicate events").
Pure stdlib, like the rest of the individuation core.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

# Tiny stopword set so descriptor matching compares the content words of short step phrases, not glue.
_STOP = frozenset(
    "a an and the to of in on at for with from into is are be removing remove removes cut cutting "
    "cuts then next add adds adding it its this that you your".split()
)


def _tokens(text: str) -> frozenset[str]:
    return frozenset(w for w in "".join(c.lower() if c.isalnum() else " " for c in text).split() if w not in _STOP)


def descriptor_overlap(a: str, b: str, thresh: float = 0.5) -> bool:
    """Default descriptor matcher: two phrases name the same step if their content-word sets overlap by at
    least ``thresh`` of the smaller set (so "remove the seeds" ≈ "seeds removed"). Empty/empty -> exact."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return a.strip().lower() == b.strip().lower()
    return len(ta & tb) / min(len(ta), len(tb)) >= thresh


@dataclass(frozen=True)
class EventCandidate:
    """One observed event as a first-class SPAN — the unit the individuation engine associates.

    ``kind`` groups comparable events ("rep", "step", "onset"); ``source`` is the channel/sensor that
    reported it; ``descriptor`` is an optional semantic label (a VLM step's text) used to dedup paraphrases.
    Spans and the ``anchor_t`` (the salient instant — a rep's peak, a step's mention) are in seconds."""

    kind: str
    start_t: float
    anchor_t: float
    stop_t: float
    source: str = ""
    amplitude: float = 0.0
    score: float = 1.0
    descriptor: str = ""

    @classmethod
    def from_rep(cls, rep, kind: str = "rep") -> "EventCandidate":
        """Adapt a rhythm :class:`~percept.rhythm.Rep` (or any object with start_t/peak_t/stop_t/channel/
        amplitude) into a candidate, so reps flow through the same engine as steps and onsets."""
        return cls(
            kind=kind,
            start_t=rep.start_t,
            anchor_t=rep.peak_t,
            stop_t=rep.stop_t,
            source=rep.channel,
            amplitude=rep.amplitude,
            score=rep.amplitude,
            descriptor="",
        )


class SpanAssociator:
    """The single online individuation engine: feed :class:`EventCandidate`s, get deduped events.

    ``refractory_s`` is the minimum separation (in seconds) between two distinct events of a kind — set it to
    about the rep period / the minimum step spacing (the self-measured ``τ``). ``descriptor_match`` decides
    when two labelled candidates are the same step (default :func:`descriptor_overlap`); ``cross_source``
    allows merging the same event seen on different channels/sensors (multi-observation fusion), off by
    default so distinct sources stay distinct unless a descriptor says otherwise."""

    def __init__(
        self,
        refractory_s: float = 0.5,
        descriptor_match: Callable[[str, str], bool] | None = None,
        cross_source: bool = False,
    ) -> None:
        self.refractory_s = refractory_s
        self.descriptor_match = descriptor_match or descriptor_overlap
        self.cross_source = cross_source
        self._events: list[EventCandidate] = []

    def _same_event(self, cand: EventCandidate, ev: EventCandidate) -> bool:
        if cand.kind != ev.kind:
            return False
        if cand.descriptor or ev.descriptor:
            # LABELLED events (VLM steps) dedup by MEANING, independent of time — the same recipe step
            # mentioned minutes apart is one step. (For counting recurring identical actions, feed unlabelled
            # candidates so the temporal refractory keeps each occurrence distinct.)
            return self.descriptor_match(cand.descriptor, ev.descriptor)
        overlap = cand.start_t <= ev.stop_t and ev.start_t <= cand.stop_t
        near = abs(cand.anchor_t - ev.anchor_t) <= self.refractory_s
        if not (overlap or near):
            return False  # beyond the refractory and non-overlapping -> a NEW event (split)
        return self.cross_source or not cand.source or not ev.source or cand.source == ev.source

    @staticmethod
    def _merge(ev: EventCandidate, cand: EventCandidate) -> EventCandidate:
        """Fuse a duplicate into the event: widen the span, keep the higher-score candidate's anchor and
        descriptor, accumulate the score (more observations = more confidence)."""
        keep = cand if cand.score > ev.score else ev
        return replace(
            ev,
            start_t=min(ev.start_t, cand.start_t),
            stop_t=max(ev.stop_t, cand.stop_t),
            anchor_t=keep.anchor_t,
            amplitude=max(ev.amplitude, cand.amplitude),
            score=ev.score + cand.score,
            descriptor=keep.descriptor or ev.descriptor or cand.descriptor,
        )

    def add(self, cand: EventCandidate) -> tuple[EventCandidate, bool]:
        """Associate one candidate. Returns ``(event, is_new)`` — ``is_new`` False means it was merged into
        (suppressed by) an existing event. Scans newest-first and stops once events are too old to overlap
        or fall in the refractory of a still-labelled match (bounded work per add)."""
        for i in range(len(self._events) - 1, -1, -1):
            ev = self._events[i]
            if self._same_event(cand, ev):
                self._events[i] = self._merge(ev, cand)
                return self._events[i], False
            # early-out for purely temporal (unlabelled) candidates: once we pass an event whose span ends
            # well before this candidate's refractory window, older events can't match either.
            if not (cand.descriptor or ev.descriptor) and ev.stop_t < cand.start_t - self.refractory_s and ev.anchor_t < cand.anchor_t - self.refractory_s:
                break
        self._events.append(cand)
        return cand, True

    @property
    def events(self) -> list[EventCandidate]:
        """The deduped events, in first-seen order."""
        return list(self._events)

    def count(self, kind: str | None = None) -> int:
        return sum(1 for e in self._events if kind is None or e.kind == kind)


def associate(
    candidates,
    refractory_s: float = 0.5,
    descriptor_match: Callable[[str, str], bool] | None = None,
    cross_source: bool = False,
) -> list[EventCandidate]:
    """Batch convenience: run an iterable of :class:`EventCandidate`s (any order) through a
    :class:`SpanAssociator` and return the deduped events. Candidates are sorted by anchor time first so the
    association is order-independent."""
    assoc = SpanAssociator(refractory_s=refractory_s, descriptor_match=descriptor_match, cross_source=cross_source)
    for cand in sorted(candidates, key=lambda c: c.anchor_t):
        assoc.add(cand)
    return assoc.events


__all__ = ["EventCandidate", "SpanAssociator", "associate", "descriptor_overlap"]
