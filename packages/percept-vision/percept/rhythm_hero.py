"""The rhythm HERO — connect the eye to the entity graph so counting is PER-PERFORMER.

This is the missing producer for the entity-scoped :meth:`Percept.perceive_rhythm`. Commit ``1d020f8``
shipped the brain half — a tracker per ``entity_id`` and performer selection — but nothing ever fed it an
``entity_id``, so it ran dead. Here the eye's per-person measurements
(:class:`percept_harness.detectors.PersonMeasure`) are associated by the committed ``EntityGraph`` /
``AppearanceIdentity`` seam into stable ids, each id's channels are folded into its own tracker, and the
brain SELECTS the performer (the entity with the most reps on the prior channel — a still bystander reps
~0). The goal's reported count becomes the performer's, not whoever the geometry happened to lock.

It also accumulates the OBSERVABILITY a count needs to be honest: how much of the clip the performer was
actually tracked (``hero_track_coverage``), how visible the prior channel's keypoints were
(``channel_coverage`` — the measured phase-source gate), and the performer's raw channel ``signal`` (so a
periodicity confidence can be measured downstream). A count that scores low on these is a BLIND count — the
two-tier design escalates it to the VLM rather than reporting it clean ("coverage ≠ correctness").

Pure-stdlib: it only orchestrates the core. The numpy/cv2 periodicity confidence, the silhouette fallback,
and the live VLM recount live in the eval layer that consumes this. Reusable beyond the benchmark — this is
the runtime shape of the rhythm hero, not benchmark glue.
"""

from __future__ import annotations

from dataclasses import dataclass

from .adapters import Box, Observation


@dataclass
class HeroResult:
    count: int                          # the SELECTED performer's confirmed-rep tally (the pose-path count)
    subject: str | None                 # the performer entity id (mirrors goal.state["subject"])
    channel_coverage: float             # mean prior-channel keypoint visibility for the performer (gate)
    hero_track_coverage: float          # fraction of processed frames the performer was actually tracked
    regularity: float                   # core-native periodicity 1/(1+CV) of the performer; 0.0 if <2 reps
    signal: list                        # the performer's prior-channel (t, value) stream (for autocorr conf)
    reps: list                          # the performer's Rep spans (start/peak/stop) — the VLM active span
    n_entities: int                     # distinct people tracked over the clip (a crowd-size proxy)
    prior_channel: str | None


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
        self._frames = 0
        self._present: dict[str, int] = {}
        self._cov: dict[str, list[float]] = {}
        self._signal: dict[str, list] = {}
        self.record = record
        self.frames: list = []  # when record=True: [(t, [(entity_id, (x, y, w, h)), ...]), ...] for debug viz

    def observe(self, t: float, measures, frame=None) -> None:
        """Fold one frame's per-person measurements. Detections become ``Observation``s, the graph assigns
        stable ids (continuity + appearance), and each id's channels drive its own rhythm tracker."""
        if not measures:
            return
        self._frames += 1
        obs = [Observation("person", Box(*m.box), m.pose_conf) for m in measures]
        entities = self.core.graph.ingest(obs, frame=frame, t=t)  # returned aligned to obs order
        if self.record:
            self.frames.append((t, [(e.id, (e.box.x, e.box.y, e.box.w, e.box.h)) for e in entities]))
        for ent, m in zip(entities, measures):
            self.core.perceive_rhythm(self.goal_id, t=t, channels=m.channels, entity_id=ent.id)
            self._present[ent.id] = self._present.get(ent.id, 0) + 1
            if self.prior is not None:
                if self.prior in m.coverage:
                    self._cov.setdefault(ent.id, []).append(m.coverage[self.prior])
                if self.prior in m.channels:
                    self._signal.setdefault(ent.id, []).append((t, m.channels[self.prior]))

    def result(self) -> HeroResult:
        """Read the SELECTED performer's count + the observability accumulated for them."""
        goal = self.core.goals.get(self.goal_id)
        subject = goal.state.get("subject") if goal is not None else None
        tracker = self.core.rhythm.get(self.goal_id)  # the per-entity path mirrors the selected performer here
        count = tracker.count if tracker is not None else 0
        reps = list(tracker.reps) if tracker is not None else []
        regularity = tracker.regularity if tracker is not None else 0.0

        cov_samples = self._cov.get(subject, []) if subject else []
        channel_coverage = sum(cov_samples) / len(cov_samples) if cov_samples else 0.0
        htc = (self._present.get(subject, 0) / self._frames) if (subject and self._frames) else 0.0
        signal = self._signal.get(subject, []) if subject else []

        return HeroResult(
            count=count, subject=subject, channel_coverage=channel_coverage,
            hero_track_coverage=htc, regularity=regularity, signal=signal, reps=reps,
            n_entities=len(self._present), prior_channel=self.prior,
        )


def escalate_count(confidence: float, edge_count: int, recount, *, tau: float = 0.4):
    """The two-tier honesty policy: trust the edge count when its periodicity ``confidence`` is high;
    otherwise the count is BLIND (occluded / non-periodic / wrong subject) — defer to ``recount`` (a VLM
    backstop in production, injected so the core stays model-free). Returns ``(count, escalated)``.

    This is the keystone: the hero need not be perfect, it must KNOW when it is unsure and ask."""
    if confidence < tau:
        return recount(), True
    return edge_count, False


__all__ = ["RhythmHero", "HeroResult", "escalate_count"]
