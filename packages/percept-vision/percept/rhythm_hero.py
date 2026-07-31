"""The rhythm HERO — connect the eye to the entity graph so counting is PER-PERFORMER.

This is the missing producer for the entity-scoped :meth:`Percept.perceive_rhythm`. Commit ``1d020f8``
shipped the brain half — a tracker per ``entity_id`` and performer selection — but nothing ever fed it an
``entity_id``, so it ran dead. Here the eye's per-person measurements
(:class:`percept.harness.detectors.PersonMeasure`) are associated by the committed ``EntityGraph`` /
``AppearanceIdentity`` seam into stable ids, each id's channels are folded into its own tracker, and the
brain SELECTS the performer (the entity with the most reps on the prior channel — a still bystander reps
~0). The goal's reported count becomes the performer's, not whoever the geometry happened to lock.

It also accumulates the OBSERVABILITY a count needs to be honest: how much of the clip the performer was
actually tracked (``hero_track_coverage``), how visible the prior channel's keypoints were
(``channel_coverage`` — the measured phase-source gate), and the performer's raw channel ``signal``, from
which it MEASURES the count's ``periodicity`` — the confidence :func:`escalate_count` consumes. A count that
scores low on these is a BLIND count — the two-tier design escalates it to the VLM rather than reporting it
clean ("coverage ≠ correctness").

Measuring that confidence *here* is what makes the honesty tier usable on the KEYLESS path. The signal was
always published "so a periodicity confidence can be measured downstream", but downstream meant the eval
layer's numpy autocorrelation, so a shipped deployment had no periodicity measure to pass and the only
confidence-shaped float within reach — ``regularity`` — cannot fall below ``tau`` even on pure noise (see
:class:`HeroResult`). ``percept.measurement.periodicity_score`` is the package's own answer to "how periodic
is this", it is pure stdlib and rate-invariant, so the route can now tell a rhythm from noise with no model,
no numpy and no benchmark harness in the loop.

Pure-stdlib: it only orchestrates the core and the canonical measurement layer. The silhouette fallback and
the live VLM recount live in the eval layer that consumes this. Reusable beyond the benchmark — this is the
runtime shape of the rhythm hero, not benchmark glue.
"""

from __future__ import annotations

from dataclasses import dataclass

from .adapters import Box, Observation
from .measurement import MeasurementSeries, periodicity_score
from .rhythm import MeasurementRecord, RhythmTracker


@dataclass
class HeroResult:
    """The selected performer's count, plus the observability that says whether to believe it.

    Two fields in here are shaped like a confidence and only ONE of them is one:

    * ``periodicity`` — measured from ``signal`` by :func:`percept.measurement.periodicity_score`: the
      height of the dominant autocorrelation peak of the performer's channel stream. It reads the RAW
      stream, so a stream with no cycle in it scores near zero however many excursions were counted off
      it. **This is the confidence :func:`escalate_count` takes.**
    * ``regularity`` — the tracker's ``1 / (1 + CV)`` over the performer's *confirmed inter-rep intervals*.
      Those intervals are measured only across excursions that already cleared the counter's prominence
      gate, i.e. exactly the subset whose spacing is regular, so their CV stays small even when there was
      no cycle at all. On broadband noise it reads 0.4775 to 1.0 (measured, 200 seeds) — never under
      ``escalate_count``'s default ``tau`` of 0.4 — so it CANNOT discriminate a blind count. It stays a
      published field because interval regularity is a real diagnostic in its own right and the eval layer
      reports it, but it answers "were the reps I counted evenly spaced?", never "was there a rhythm?".

    Both are published, rather than one replacing the other, so a reader can see the two answers disagree
    instead of guessing which meaning a single ``regularity`` carried. ``tests/test_rhythm_blind_count.py``
    pins the separation and the trap.
    """

    count: int                          # the SELECTED performer's confirmed-rep tally (the pose-path count)
    subject: str | None                 # the performer entity id (mirrors goal.state["subject"])
    channel_coverage: float             # mean prior-channel keypoint visibility for the performer (gate)
    hero_track_coverage: float          # fraction of OBSERVED frames the performer was actually tracked —
    #                                     every frame handed to observe(), including the ones the eye found
    #                                     nobody in. It used to divide by the frames that happened to carry
    #                                     a detection, so numerator and denominator moved together and the
    #                                     field was pinned at 1.0 for any clip whose only failure mode was
    #                                     empty detection (measured: 20 detections over 100 frames -> 1.0).
    periodicity: float                  # [0,1] periodicity of the performer's ``signal`` — THE escalate_count
    #                                     confidence. 0.0 when no signal was accumulated (no performer, or no
    #                                     prior channel to accumulate one on): no evidence is not confidence,
    #                                     so such a count escalates rather than being asserted.
    regularity: float                   # 1/(1+CV) of the performer's confirmed inter-rep INTERVALS; 0.0 if
    #                                     <2 reps. A count-SHAPE diagnostic, not a periodicity confidence —
    #                                     see the class docstring before passing it anywhere.
    signal: list                        # the performer's prior-channel (t, value) stream (periodicity's input)
    reps: list                          # the performer's Rep spans (start/peak/stop) — the VLM active span
    n_entities: int                     # distinct people tracked over the clip (a crowd-size proxy)
    prior_channel: str | None
    frames_observed: int = 0            # the EVIDENCE BASE under hero_track_coverage: how many frames the
    #                                     ratio was taken over, so "0.2 of 100 frames" and "0.2 of 5" stop
    #                                     reading identically. A ratio without its denominator is not a
    #                                     measurement; this is the same counter shape
    #                                     ``percept.rhythm.MeasurementRecord`` carries for the tracker.
    frames_measured: int = 0            # ...of which carried at least one detected person


class RhythmHero:
    """Per-frame: associate detections → stable ids → per-entity ``perceive_rhythm``; accumulate the
    performer's observability. One instance per clip / concern. ``frame`` (BGR) is optional — when given,
    the ``EntityGraph``'s appearance embedder uses it to hold identity through gaps; without it the graph
    degrades to pure continuity."""

    def __init__(self, core, goal_id: str, *, prior_channel: str | None = None,
                 record: bool = False) -> None:
        self.core = core
        self.goal_id = goal_id
        self.prior = prior_channel
        #: Frames OFFERED vs frames that carried a detection — the denominator under every ratio below.
        self.measurement = MeasurementRecord()
        self._present: dict[str, int] = {}
        self._cov: dict[str, list[float]] = {}
        self._signal: dict[str, list] = {}
        self.record = record
        self.frames: list = []  # when record=True: [(t, [(entity_id, (x, y, w, h)), ...]), ...] for debug viz

    @property
    def _frames(self) -> int:
        """Frames OBSERVED — every frame handed to :meth:`observe`, including the empty ones."""
        return self.measurement.observed

    def observe(self, t: float, measures, frame=None) -> None:
        """Fold one frame's per-person measurements. Detections become ``Observation``s, the graph assigns
        stable ids (continuity + appearance), and each id's channels drive its own rhythm tracker.

        The frame enters the denominator BEFORE the empty-measures exit. A frame the eye found nobody in
        was still processed, and it is exactly the frame ``hero_track_coverage`` exists to report."""
        self.measurement.observed += 1
        if not measures:
            return
        self.measurement.measured += 1
        if any(getattr(m, "scale_floored", False) for m in measures):
            self.measurement.unnormalized += 1
        obs = [Observation("person", Box(*m.box), m.pose_conf) for m in measures]
        entities = self.core.graph.ingest(obs, frame=frame, t=t)  # returned aligned to obs order
        if self.record:
            self.frames.append((t, [(e.id, (e.box.x, e.box.y, e.box.w, e.box.h)) for e in entities]))
        for ent, m in zip(entities, measures):
            # scale_floored is PER PERSON: two people in one frame can be at different distances, so
            # "were these channels normalized by a real torso?" is not a frame-level fact.
            self.core.perceive_rhythm(self.goal_id, t=t, channels=m.channels, entity_id=ent.id,
                                      scale_floored=bool(getattr(m, "scale_floored", False)))
            self._present[ent.id] = self._present.get(ent.id, 0) + 1
            if self.prior is not None:
                if self.prior in m.coverage:
                    self._cov.setdefault(ent.id, []).append(m.coverage[self.prior])
                if self.prior in m.channels:
                    self._signal.setdefault(ent.id, []).append((t, m.channels[self.prior]))

    def _tracker(self, subject: str | None):
        """The SUBJECT's own tracker, never a shared slot.

        This used to read ``core.rhythm[goal_id]`` — one entry both rhythm producers write — and never
        checked that what it found there belonged to ``goal.state['subject']``, so ``count``/``reps``/
        ``regularity`` could describe one person while ``channel_coverage``/``hero_track_coverage``
        described another. Reading the store by ``(goal_id, subject)`` is what makes the whole result
        describe ONE entity."""
        if subject is None:
            return None
        lookup = getattr(self.core, "rhythm_tracker", None)
        if lookup is not None:
            return lookup(self.goal_id, subject)
        return getattr(self.core, "_rhythm_multi", {}).get((self.goal_id, subject))

    def result(self) -> HeroResult:
        """Read the SELECTED performer's count + the observability accumulated for them."""
        goal = self.core.goals.get(self.goal_id)
        subject = goal.state.get("subject") if goal is not None else None
        tracker = self._tracker(subject)
        count = tracker.count if tracker is not None else 0
        reps = list(tracker.reps) if tracker is not None else []
        regularity = tracker.regularity if tracker is not None else 0.0

        cov_samples = self._cov.get(subject, []) if subject else []
        channel_coverage = sum(cov_samples) / len(cov_samples) if cov_samples else 0.0
        observed = self.measurement.observed
        htc = (self._present.get(subject, 0) / observed) if (subject and observed) else 0.0
        signal = self._signal.get(subject, []) if subject else []

        return HeroResult(
            count=count, subject=subject, channel_coverage=channel_coverage,
            hero_track_coverage=htc, periodicity=measure_periodicity(signal, self.prior, tracker),
            regularity=regularity, signal=signal, reps=reps,
            n_entities=len(self._present), prior_channel=self.prior,
            frames_observed=observed, frames_measured=self.measurement.measured,
        )


def measure_periodicity(signal, channel: str | None = None, tracker: object = None) -> float:
    """How periodic a performer's ``(t, value)`` stream is, ∈ [0, 1] — the blind-count confidence.

    Scored on the SAME canonical grid and plausible rep-period band the tracker judges a channel by
    (:attr:`~percept.rhythm.RhythmTracker.select_hz` / ``select_band_s``), so "confident" means confident
    about a cycle the counter would itself have been willing to call a rep — not about some faster or
    slower cycle it would have ignored. ``tracker=None`` falls back to those same class defaults.

    An empty or too-short stream scores ``0.0``: no evidence is not confidence, and the resulting count
    escalates rather than being asserted.
    """

    if not signal:
        return 0.0
    band = getattr(tracker, "select_band_s", RhythmTracker.select_band_s)
    hz = getattr(tracker, "select_hz", RhythmTracker.select_hz)
    return periodicity_score(MeasurementSeries.from_pairs(channel or "", signal), band_s=band, hz=hz)


def escalate_count(confidence: float, edge_count: int, recount, *, tau: float = 0.4):
    """The two-tier honesty policy: trust the edge count when its periodicity ``confidence`` is high;
    otherwise the count is BLIND (occluded / non-periodic / wrong subject) — defer to ``recount`` (a VLM
    backstop in production, injected so the core stays model-free). Returns ``(count, escalated)``.

    ``confidence`` is :attr:`HeroResult.periodicity`. Pass nothing else off a result:
    :attr:`HeroResult.regularity` is the neighbouring confidence-shaped float and it sits well above
    ``tau`` even on pure noise, so wiring *it* here builds a counter that never escalates and reports
    fabricated reps as clean — the exact failure this policy exists to prevent (see :class:`HeroResult`).

    This is the keystone: the hero need not be perfect, it must KNOW when it is unsure and ask."""
    if confidence < tau:
        return recount(), True
    return edge_count, False


__all__ = ["RhythmHero", "HeroResult", "escalate_count", "measure_periodicity"]
