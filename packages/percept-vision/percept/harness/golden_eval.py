"""PROBE-HARNESS SCAFFOLDING — ``golden_eval`` (Viability §5a/§6, board card A14).

**Standalone evaluation module — not runtime instrumentation.** This scores already-produced
predicted :class:`~percept.contracts.JudgmentEvent` records against labeled ground-truth episodes
(card A6's :class:`~percept.contracts.GroundTruth`, e.g. via
``eval/viability/ground_truth.py``'s ``ground_truth_for_archetype``). It does not touch the
runtime pipeline (``worker.py``/``agent.py``/``gate.py``) at all.

Implements the Viability §5a event-level matching rule EXACTLY:

* A predicted event matches AT MOST ONE ground-truth episode.
* A match requires ``decision_emitted_ts in [decision_eligible_ts, episode_end_ts +
  allowed_lateness]``.
* Additional predictions in the same episode (after it already has a match) = DUPLICATE false
  fires.
* Predictions that match no episode window at all = (plain) false fires.
* Ground-truth episodes with no matching prediction = misses.

Recall/precision are EVENT-level (one unit = one episode / one predicted event), never
frame-level — a long episode does not get extra credit for extra frames, and a burst of duplicate
predictions inside one episode does not inflate recall.

This module also emits the canonical, deterministic "golden" ``JudgmentEvent`` timeline for a set
of episodes (:func:`build_golden_timeline`) — the ideal/expected fire sequence a hero-demo replay
(card A22/A24) is diffed against via :func:`score`.

Two pieces this module deliberately does NOT own (imported, never edited here):

* :class:`percept.contracts.GroundTruth` — card A6's frozen evaluation-only ground-truth record
  (``event_start_ts``/``decision_eligible_ts``/``annotation_version``/``annotator_or_fixture_id``).
* :class:`percept.contracts.JudgmentEvent` / :class:`percept.contracts.JudgmentTiming` — card A5's
  frozen live-decision contract; ``timing.decision_emitted_ts`` is the sole timestamp this module
  reads off a prediction.

:class:`GroundTruthEpisode` (defined here) is the eval-only join of a ``GroundTruth`` with the two
extra fields the §5a window needs — ``episode_end_ts`` and ``allowed_lateness_s`` — that do not
belong on the frozen ``GroundTruth`` contract itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet, Sequence

from ..contracts import GroundTruth, JudgmentEvent, JudgmentTiming

# ---------------------------------------------------------------------------
# GroundTruthEpisode — GroundTruth (A6) + the episode-window fields §5a matching needs.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundTruthEpisode:
    """One labeled ground-truth episode a predicted-event set is scored against.

    Wraps card A6's :class:`~percept.contracts.GroundTruth` (``event_start_ts`` /
    ``decision_eligible_ts`` / provenance) with the two additional fields the §5a event-level
    matching rule needs:

    * ``episode_end_ts`` — when the labeled episode itself ends, on the same fixture/annotator
      timeline as ``ground_truth.event_start_ts``. Must be ``>= ground_truth.decision_eligible_ts``
      (an episode cannot end before the earliest instant it was even eligible to be judged).
    * ``allowed_lateness_s`` — the archetype's documented grace period (seconds, ``>= 0``) past
      ``episode_end_ts`` a still-correct fire may land in (judge/queue latency headroom). Defaults
      to ``0.0`` (no grace) so a caller that doesn't care about lateness gets the tightest window.

    Neither field lives on ``GroundTruth`` itself (that contract is card A6's, untouched here) —
    they are eval-only, per-episode metadata this module owns.

    ``episode_id`` is required, non-empty, and must be unique within one :func:`score` /
    :func:`build_golden_timeline` call — it is the join key between a ground-truth episode and the
    golden/replay ``JudgmentEvent.key``.
    """

    episode_id: str
    ground_truth: GroundTruth
    episode_end_ts: float
    allowed_lateness_s: float = 0.0

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("GroundTruthEpisode.episode_id is required (non-empty).")
        if self.episode_end_ts < self.ground_truth.decision_eligible_ts:
            raise ValueError(
                "GroundTruthEpisode.episode_end_ts must be >= ground_truth.decision_eligible_ts "
                "(an episode cannot end before it becomes eligible to judge): "
                f"episode_end_ts={self.episode_end_ts!r} < "
                f"decision_eligible_ts={self.ground_truth.decision_eligible_ts!r}"
            )
        if self.allowed_lateness_s < 0:
            raise ValueError(f"allowed_lateness_s must be >= 0, got {self.allowed_lateness_s!r}")

    @property
    def window(self) -> tuple[float, float]:
        """The closed §5a match window: ``[decision_eligible_ts, episode_end_ts + allowed_lateness_s]``."""
        return (
            self.ground_truth.decision_eligible_ts,
            self.episode_end_ts + self.allowed_lateness_s,
        )

    def contains(self, decision_emitted_ts: float) -> bool:
        """Does ``decision_emitted_ts`` fall inside this episode's closed §5a match window?"""
        lo, hi = self.window
        return lo <= decision_emitted_ts <= hi


# ---------------------------------------------------------------------------
# Scoring outcome types.
# ---------------------------------------------------------------------------


class MatchOutcome(str, Enum):
    """The §5a disposition of ONE predicted event."""

    #: The predicted event is the (first, one-to-one) match for its episode.
    MATCH = "match"
    #: The predicted event landed inside an episode's window that ALREADY has a match — a
    #: duplicate false fire (§5a: "additional predictions in the same episode").
    DUPLICATE = "duplicate"
    #: The predicted event landed inside no episode's window at all — a plain false fire.
    FALSE_FIRE = "false_fire"


@dataclass(frozen=True)
class PredictionScore:
    """The §5a outcome for one predicted event."""

    event: JudgmentEvent
    decision_emitted_ts: float
    outcome: MatchOutcome
    #: The episode this prediction was scored against — set for MATCH and DUPLICATE, ``None`` for
    #: FALSE_FIRE (matched no episode window at all).
    matched_episode_id: str | None


@dataclass(frozen=True)
class GoldenEvalResult:
    """Event-level recall/precision + the full per-prediction breakdown for one :func:`score` call."""

    prediction_scores: tuple[PredictionScore, ...]
    matched_episode_ids: tuple[str, ...]
    missed_episode_ids: tuple[str, ...]
    #: matches / total ground-truth episodes. ``1.0`` (vacuously) when there were no episodes.
    recall: float
    #: matches / total scored predictions. ``1.0`` (vacuously) when there were no predictions.
    precision: float

    @property
    def true_positives(self) -> int:
        return len(self.matched_episode_ids)

    @property
    def duplicate_false_fires(self) -> int:
        return sum(1 for p in self.prediction_scores if p.outcome is MatchOutcome.DUPLICATE)

    @property
    def unmatched_false_fires(self) -> int:
        return sum(1 for p in self.prediction_scores if p.outcome is MatchOutcome.FALSE_FIRE)

    @property
    def false_fires(self) -> int:
        """Total false fires = duplicates + unmatched (§5a treats both as false fires)."""
        return self.duplicate_false_fires + self.unmatched_false_fires

    @property
    def misses(self) -> int:
        return len(self.missed_episode_ids)


#: Which :class:`~percept.contracts.JudgmentAction` values on a predicted `JudgmentEvent` count as
#: a "prediction" for scoring purposes. Defaults to ``fire`` only — a ``silent`` decision emits no
#: notification at all, and an ``ask`` is a clarifying question rather than a fired event; a caller
#: scoring a different action set may pass its own ``predicted_actions``.
DEFAULT_PREDICTED_ACTIONS: FrozenSet[str] = frozenset({"fire"})


def _prediction_sort_key(event: JudgmentEvent) -> tuple[float, str, int]:
    """Deterministic total order over predictions: emission time, then key, then evidence_seq."""
    return (event.timing.decision_emitted_ts, str(event.key), event.evidence_seq)


def _episode_sort_key(episode: GroundTruthEpisode) -> tuple[float, str]:
    """Deterministic total order over episodes: eligibility time, then episode_id."""
    return (episode.ground_truth.decision_eligible_ts, episode.episode_id)


def _assert_unique_episode_ids(episodes: Sequence[GroundTruthEpisode]) -> None:
    ids = [e.episode_id for e in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError(f"GroundTruthEpisode.episode_id values must be unique, got {ids!r}")


def score(
    episodes: Sequence[GroundTruthEpisode],
    predicted_events: Sequence[JudgmentEvent],
    *,
    predicted_actions: FrozenSet[str] = DEFAULT_PREDICTED_ACTIONS,
) -> GoldenEvalResult:
    """Score ``predicted_events`` against ``episodes`` per the Viability §5a event-level matching rule.

    Deterministic one-to-one GREEDY matching, processing predictions in
    ``(decision_emitted_ts, key, evidence_seq)`` order (so two calls over the same inputs always
    produce byte-for-byte identical results, independent of input list order):

    1. If ``decision_emitted_ts`` falls inside the window of an episode that has NO match yet, it
       becomes that episode's match (§5a: matches AT MOST ONE episode — the first, earliest-sorted
       prediction to land in an episode's window claims it).
    2. Else, if it falls inside the window of an episode that ALREADY has a match, it is a
       DUPLICATE false fire against that episode (§5a: "additional predictions in the same
       episode").
    3. Else, it matches no episode window at all — a plain (unmatched) false fire.

    Ground-truth episodes with no matching prediction after all predictions are processed are
    misses. Only predicted events whose ``.action`` is in ``predicted_actions`` (default: ``fire``
    only) are scored at all — anything else is silently excluded, not counted as a false fire.
    """
    _assert_unique_episode_ids(episodes)

    ordered_episodes = sorted(episodes, key=_episode_sort_key)
    ordered_predictions = sorted(
        (e for e in predicted_events if e.action in predicted_actions),
        key=_prediction_sort_key,
    )

    matched: dict[str, JudgmentEvent] = {}
    scores: list[PredictionScore] = []

    for pred in ordered_predictions:
        ts = pred.timing.decision_emitted_ts

        # 1) An unmatched episode whose window contains ts claims this prediction as its match.
        claimant = next(
            (ep for ep in ordered_episodes if ep.episode_id not in matched and ep.contains(ts)),
            None,
        )
        if claimant is not None:
            matched[claimant.episode_id] = pred
            scores.append(PredictionScore(pred, ts, MatchOutcome.MATCH, claimant.episode_id))
            continue

        # 2) Otherwise, an already-matched episode whose window contains ts -> duplicate.
        duplicate_of = next(
            (ep for ep in ordered_episodes if ep.episode_id in matched and ep.contains(ts)),
            None,
        )
        if duplicate_of is not None:
            scores.append(
                PredictionScore(pred, ts, MatchOutcome.DUPLICATE, duplicate_of.episode_id)
            )
            continue

        # 3) No episode window contains ts at all -> plain false fire.
        scores.append(PredictionScore(pred, ts, MatchOutcome.FALSE_FIRE, None))

    matched_ids = tuple(sorted(matched))
    missed_ids = tuple(
        sorted(ep.episode_id for ep in episodes if ep.episode_id not in matched)
    )

    true_positives = len(matched_ids)
    recall = (true_positives / len(episodes)) if episodes else 1.0
    precision = (true_positives / len(scores)) if scores else 1.0

    return GoldenEvalResult(
        prediction_scores=tuple(scores),
        matched_episode_ids=matched_ids,
        missed_episode_ids=missed_ids,
        recall=recall,
        precision=precision,
    )


# ---------------------------------------------------------------------------
# Golden timeline — the canonical, deterministic EXPECTED JudgmentEvent sequence.
# ---------------------------------------------------------------------------


def build_golden_timeline(episodes: Sequence[GroundTruthEpisode]) -> tuple[JudgmentEvent, ...]:
    """The canonical, deterministic golden ``JudgmentEvent`` timeline for ``episodes``.

    One ``JudgmentEvent`` per episode, ``action="fire"``, ``key=episode_id``, ``evidence_seq`` =
    the episode's rank in canonical ``(decision_eligible_ts, episode_id)`` order — so two calls
    over the same episode set always produce a byte-for-byte identical, sequentially-numbered
    timeline regardless of input list order.

    The golden event's ``decision_emitted_ts`` is deliberately pinned to the episode's own
    ``decision_eligible_ts`` — the EARLIEST instant a correct fire could legally land per §5a — the
    "ideal / zero funnel-latency" expected fire, not a claim about real production latency. All
    five ``JudgmentTiming`` monotonic stamps are set to that same instant (trivially
    non-decreasing, satisfying ``JudgmentTiming``'s own construction invariant) since a synthetic
    golden record has no real ingest/admit/judge funnel to time.

    A replay's own predicted-event timeline is diffed against the golden expectation by calling
    :func:`score` with THESE episodes and the replay's actual predictions — never by comparing
    ``JudgmentEvent`` equality directly, since a real (correct) prediction legitimately lands later
    than ``decision_eligible_ts``, anywhere inside the §5a window.
    """
    _assert_unique_episode_ids(episodes)
    ordered = sorted(episodes, key=_episode_sort_key)

    timeline = []
    for seq, ep in enumerate(ordered):
        ts = ep.ground_truth.decision_eligible_ts
        timing = JudgmentTiming(
            ingest_observed_ts=ts,
            admitted_ts=ts,
            judge_started_ts=ts,
            judge_completed_ts=ts,
            decision_emitted_ts=ts,
        )
        timeline.append(
            JudgmentEvent(key=ep.episode_id, evidence_seq=seq, action="fire", timing=timing)
        )
    return tuple(timeline)
