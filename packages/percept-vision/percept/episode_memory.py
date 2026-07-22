"""Episode / entity-memory judge mode — the deferred arm.

An ablation study proved the **eye** (localized salience) is the biggest recall lever and the
**ordered window** (temporal input, oldest->newest) the next one. The remaining, still-unproven lever
is **episode memory**: instead of re-deciding a persistent/temporal goal from a single frame (or even
a fixed window), thread a per-goal RUNNING ANSWER across successive judged frames so the verdict
integrates evidence accumulated over the goal's whole lifetime, not just the last N frames.

The mechanism is already partly wired: every production ``Vision`` backend implements a process-QA
step (``context={"process_qa": True, "running_answer": <prior>}`` -> ``Verdict.running_answer``, see
``backends/openrouter.py:_judge_process_qa`` / ``backends/gemini.py:_judge_process_qa``) and
``agent.py`` already threads it for goals that opt into ``state["process_qa"]``. This module
PRODUCTIZES that mechanism as a standalone, named judge mode:

  * ``EpisodeMemoryState`` / ``EpisodeMemoryStore`` — the per-goal running state, decoupled from the
    agent's ``Executor``/``ConcernContext`` so this mode is usable on its own (eval harness ablation,
    a future WatchProfile-driven runtime path) without touching the executor.
  * ``judge_episode_memory`` — one judge step: thread the prior answer IN, fold the model's rewritten
    answer OUT, and — the productization over the raw process-QA plumbing — return a ``Verdict`` whose
    ``satisfied`` reflects INTEGRATED evidence (accumulated coverage + substance + confidence, via the
    existing :func:`~percept.concern.resolution_ready` bar) rather than one frame's read alone.
  * ``episode_memory_enabled`` — reads an opt-in ``goal.state["judge_mode"]`` flag, the seam a future
    ``WatchProfile`` can set to select this mode per (environment, goal-type, state).

Additive only: this module does not import or modify ``agent.py``, ``gate.py``, ``concern.py``, or any
backend — the existing single-frame and window judge paths are byte-for-byte unchanged. Wiring this
mode into the live ``perceive_judged`` loop (so a WatchProfile's choice actually takes effect on the
hot path) and registering it with the swappable judge seam are follow-up work — out of scope here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .adapters import Goal, Verdict
from .concern import resolution_ready

# The plan_mode/config value a WatchProfile will set to select this judge mode (future). A goal
# that never sets ``judge_mode`` is untouched by this module — no behavior changes by default.
JUDGE_MODE_EPISODE_MEMORY = "episode_memory"

# Bounded per-goal history — mirrors the existing "a 24/7 concern must not leak memory" discipline
# (agent.py's ``ctx.process_log`` cap).
_HISTORY_CAP = 256
# Bounded number of concurrently-tracked goals in a store (same discipline, applied per-store).
_STORE_CAP = 256


def episode_memory_enabled(goal: Goal) -> bool:
    """Has this goal opted into the episode-memory judge mode?

    Reads ``goal.state["judge_mode"]`` — the flag a future ``WatchProfile`` resolves per
    (environment, goal-type, state). Read-only: this does not change what ``agent.py`` does with
    the goal today, it only exposes the seam so a WatchProfile-driven caller can select this mode."""
    return goal.state.get("judge_mode") == JUDGE_MODE_EPISODE_MEMORY


@dataclass
class EpisodeMemoryState:
    """A single goal's running answer, threaded across successive judged frames (the episode memory).

    ``frame_count`` only increments on EVIDENCE-BEARING frames (the model returned a non-empty rewritten
    answer) — it is the coverage count :func:`~percept.concern.resolution_ready` reads, mirroring
    ``len(ConcernContext.process_log)`` in the existing process-QA path."""

    running_answer: str = ""
    history: list[str] = field(default_factory=list)
    frame_count: int = 0

    def record(self, answer: str) -> None:
        """Fold a new evidence-bearing answer into the running state. Empty/blank answers are a no-op
        (a parse miss on the backend side preserves the prior answer — never regress the accumulation)."""
        answer = (answer or "").strip()
        if not answer:
            return
        self.running_answer = answer
        self.history.append(answer)
        overflow = len(self.history) - _HISTORY_CAP
        if overflow > 0:
            del self.history[:overflow]
        self.frame_count += 1

    def reset(self) -> None:
        """End the episode: wipe the running answer and coverage (a new episode starts clean)."""
        self.running_answer = ""
        self.history = []
        self.frame_count = 0


class EpisodeMemoryStore:
    """Bounded per-goal store of :class:`EpisodeMemoryState`. Decoupled from the agent's ``Executor``
    so this mode threads state on its own — usable directly (eval harness, tests) without a running
    :class:`~percept.agent.Percept` instance."""

    def __init__(self, *, max_goals: int = _STORE_CAP) -> None:
        self._states: dict[str, EpisodeMemoryState] = {}
        self._max_goals = max_goals

    def get(self, goal_id: str) -> EpisodeMemoryState:
        """Fetch (creating if absent) the running state for ``goal_id``. Evicts the oldest-inserted
        goal once the bound is hit — the same "24/7 agent can't leak memory" discipline as elsewhere."""
        state = self._states.get(goal_id)
        if state is None:
            if len(self._states) >= self._max_goals:
                self._states.pop(next(iter(self._states)))
            state = self._states[goal_id] = EpisodeMemoryState()
        return state

    def reset(self, goal_id: str) -> None:
        """Drop a goal's running state (episode boundary / goal completion)."""
        self._states.pop(goal_id, None)

    def __len__(self) -> int:
        return len(self._states)


async def judge_episode_memory(
    vision: Any,
    goal: Goal,
    frame: Any,
    *,
    store: EpisodeMemoryStore,
    frames: list | None = None,
    roi: Any = None,
    exemplar: Any = None,
    corrections: dict | None = None,
) -> Verdict:
    """One episode-memory judge step: the next precision/recall lever after the ordered window.

    Threads ``store``'s prior running answer for ``goal.id`` INTO the judge call (the same
    ``context={"process_qa": True, "running_answer": ...}`` seam ``agent.py`` already uses for opted-in
    process-QA goals — see ``backends/openrouter.py:_judge_process_qa``), folds the model's rewritten
    answer back OUT into ``store``, and returns a ``Verdict`` whose ``satisfied`` integrates evidence
    ACCUMULATED OVER TIME (accumulated coverage + substance + confidence, via
    :func:`~percept.concern.resolution_ready`) rather than reflecting only this one frame's read — this
    is the productization step over the raw per-frame ``running_answer`` plumbing.

    A parse miss (the backend returns no new ``running_answer``) is a no-op on the state — the prior
    answer (and its accumulated coverage) is preserved, never regressed."""
    state = store.get(goal.id)
    prior = state.running_answer
    verdict = await vision.judge(
        goal.condition, frame,
        roi=roi, exemplar=exemplar, corrections=corrections, frames=frames,
        context={"process_qa": True, "running_answer": prior},
    )
    state.record(verdict.running_answer)
    answer = state.running_answer
    integrated = resolution_ready(answer, state.frame_count, verdict.confidence)
    return Verdict(
        satisfied=integrated,
        confidence=verdict.confidence,
        evidence=answer or verdict.evidence,
        running_answer=answer,
    )


__all__ = [
    "JUDGE_MODE_EPISODE_MEMORY",
    "episode_memory_enabled",
    "EpisodeMemoryState",
    "EpisodeMemoryStore",
    "judge_episode_memory",
]
