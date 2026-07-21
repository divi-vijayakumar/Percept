"""Episode counting — the cup problem (spec §5.5).

Counting "cups of coffee" is **episode counting**: not sip-counting (over-counts) and not
IoU-track-counting (mis-counts across gaps). Per counting goal we keep a bounded **episode log**
(the counting window — "today's cups", a handful). On each candidate drinking-event the model
judges *new-episode vs continuation* against the log summaries + the current exemplar; the count is
the number of distinct episodes.

The judge is injected (an LLM in real use; a deterministic stub in L1) so the dedup *logic* is
testable without a model — and so we never assert backend intelligence with a fake.

PERSISTENCE IS SESSION-ONLY — episodes do NOT survive a restart (ADR-006, card Y9)
---------------------------------------------------------------------------------

An :class:`EpisodeLog` is **process-lifetime state**. ``Percept.snapshot()`` carries ``goals``,
``gate``, ``entities`` and ``action_log``; it does **not** carry ``Percept.episodes`` (the per-goal
log map built in ``Percept.count_event``), so ``Percept.restore()`` cannot rehydrate it. After a
restart every counting goal starts again at zero episodes, and a subject last seen before the restart
is a NEW episode afterwards.

This **blocks any 24/7 viability claim** until a later phase gives episodes a persisted
representation. The bounds below (``window_s`` / ``max_episodes``) are a MEMORY story — they keep a
long-running process from growing without limit. They are not a DURABILITY story and must not be
cited as one.

Do not read a durable IDENTIFIER as durable STATE. A subject reference marked ``scope='durable'``
says the identifier is stable — ``PO-4471`` means the same thing next week — and says nothing at all
about restart-safety of the episode state keyed by it. The failure mode of conflating them is silent
under-counting after a redeploy, which nothing in this module can detect for you.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .observability import get_logger, log

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
    """Bounded per-goal episode log. ``observe`` folds a candidate event into new/continuation.

    Boundedness (ADR-001 §9, card A9 subset): ``window_s`` bounds the log in TIME (episodes idle
    longer than the counting window fall out — "today's cups", not "all-time cups") and
    ``max_episodes`` bounds it in COUNT (a hard ceiling, drop-oldest, the backstop when no window is
    set or the window alone isn't enough). Every bound-driven drop is TRACED — never silent —
    through ``on_drop(episode, reason)`` (``reason`` is ``"window_expired"`` or ``"max_episodes"``);
    the default logs a structured line so a drop is observable even when the caller wires nothing.
    """

    def __init__(
        self, goal_id: str, *, window_s: float | None = None, max_episodes: int = 64,
        on_drop: "Callable[[Episode, str], None] | None" = None,
    ) -> None:
        self.goal_id = goal_id
        self.window_s = window_s
        self.max_episodes = max_episodes
        self.on_drop = on_drop
        self._episodes: list[Episode] = []
        self._counter = 0

    def _dropped(self, episode: "Episode", reason: str) -> None:
        """ADR-001 §9: a bound-driven drop must never be silent."""
        if self.on_drop is not None:
            self.on_drop(episode, reason)
        else:
            log(
                get_logger("percept.episodes"), "episode.dropped",
                goal_id=self.goal_id, episode_id=episode.id, reason=reason,
            )

    def _prune(self, now: float) -> None:
        """Drop episodes outside the counting window (keeps the set bounded — the endurance story)."""
        if self.window_s is None:
            return
        keep: list[Episode] = []
        for e in self._episodes:
            if now - e.last_seen <= self.window_s:
                keep.append(e)
            else:
                self._dropped(e, "window_expired")
        self._episodes = keep

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
            oldest = self._episodes.pop(0)  # oldest falls out of the bounded log
            self._dropped(oldest, "max_episodes")
        return episode, True

    def count(self, now: float | None = None) -> int:
        """Distinct episodes (optionally pruning to the window at ``now`` first)."""
        if now is not None:
            self._prune(now)
        return len(self._episodes)


__all__ = ["Episode", "EpisodeJudge", "EpisodeLog"]
