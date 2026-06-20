"""Episode counting — the cup problem (spec §5.5).

Counting "cups of coffee" is **episode counting**: not sip-counting (over-counts) and not
IoU-track-counting (mis-counts across gaps). Per counting goal we keep a bounded **episode log**
(the counting window — "today's cups", a handful). On each candidate drinking-event the model
judges *new-episode vs continuation* against the log summaries + the current exemplar; the count is
the number of distinct episodes.

The judge is injected (an LLM in real use; a deterministic stub in L1) so the dedup *logic* is
testable without a model — and so we never assert backend intelligence with a fake.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

# judge(summaries, exemplar) -> the id of the episode this continues, or None for a NEW episode.
EpisodeJudge = Callable[[list[dict], Any], "str | None"]


@dataclass
class Episode:
    id: str
    goal_id: str
    started_at: float
    last_seen: float
    exemplar: Any = None
    representative: Any = None


class EpisodeLog:
    """Bounded per-goal episode log. ``observe`` folds a candidate event into new/continuation."""

    def __init__(
        self, goal_id: str, *, window_s: float | None = None, max_episodes: int = 64
    ) -> None:
        self.goal_id = goal_id
        self.window_s = window_s
        self.max_episodes = max_episodes
        self._episodes: list[Episode] = []
        self._counter = 0

    def _prune(self, now: float) -> None:
        """Drop episodes outside the counting window (keeps the set bounded — the endurance story)."""
        if self.window_s is not None:
            self._episodes = [e for e in self._episodes if now - e.last_seen <= self.window_s]

    def summaries(self) -> list[dict]:
        """Compact episode descriptors handed to the judge (no pixels on the hot path)."""
        return [
            {"id": e.id, "started_at": e.started_at, "last_seen": e.last_seen}
            for e in self._episodes
        ]

    def observe(self, t: float, exemplar: Any, judge: EpisodeJudge) -> tuple[Episode, bool]:
        """Fold a candidate event at time ``t`` into the log. Returns ``(episode, is_new)``."""
        self._prune(t)
        matched_id = judge(self.summaries(), exemplar) if self._episodes else None
        if matched_id is not None:
            for e in self._episodes:
                if e.id == matched_id:
                    e.last_seen = t  # continuation — extend, don't count again
                    return e, False
        self._counter += 1
        episode = Episode(
            id=f"{self.goal_id}:ep{self._counter}",
            goal_id=self.goal_id,
            started_at=t,
            last_seen=t,
            exemplar=exemplar,
            representative=exemplar,
        )
        self._episodes.append(episode)
        if len(self._episodes) > self.max_episodes:
            self._episodes.pop(0)  # oldest falls out of the bounded log
        return episode, True

    def count(self, now: float | None = None) -> int:
        """Distinct episodes (optionally pruning to the window at ``now`` first)."""
        if now is not None:
            self._prune(now)
        return len(self._episodes)


__all__ = ["Episode", "EpisodeJudge", "EpisodeLog"]
