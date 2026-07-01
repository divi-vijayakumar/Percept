"""The fovea — a per-concern processing-density controller (perceive much, process only what's needed).

The eye captures cheap signals (motion) at a HIGH rate always; the expensive perception (pose/CLIP) is the
cost. Rather than couple the capture rate to the processing rate (one global ``fps`` that either
over-captures or, irreversibly, under-captures), this controller decides — per concern, per captured frame —
whether to spend the expensive perception on it:

  * **coast** at a ``base_fps`` floor while the scene is quiet (most frames are skipped — cheap);
  * **escalate** toward ``max_fps`` (toward the capture rate) when cheap motion spikes — the *saccade*: spend
    perception where the temporal bandwidth suddenly rises, and HOLD the high rate briefly so a burst is
    fully resolved, not just its first frame;
  * **bound** by ``max_fps`` — the cost ceiling; processing is never denser than this.

It is the same allocation logic as the C2 VLM cadence (``percept.harness.tier0`` / ``gate_on``), one tier
down: there the rationed resource is the cloud VLM call; here it is pose inference. Pure and deterministic
(timestamps + a cheap motion scalar in, boolean decisions out), so it unit-tests with no model. The runtime
captures at native fps and asks ``should_process`` before running the eye's expensive ``measure()``; a
fast-rep concern then resolves at up to ``max_fps`` while a "watch for my dog" concern coasts at ``base_fps``
— from the *same* high-rate capture, with no per-goal capture-fps to guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SensePolicy:
    """A concern's processing-density envelope (what the goal-set ``sense`` hint compiles to).

    ``base_fps`` is the quiet floor; ``max_fps`` the cost ceiling it escalates to on motion; ``hold_s`` keeps
    the escalation alive for a beat after the spike (the saccade settling). ``base_fps <= max_fps``."""

    base_fps: float = 4.0
    max_fps: float = 15.0
    escalate_threshold: float = 0.15  # cheap-motion level that triggers the saccade
    hold_s: float = 0.5  # keep escalated this long after the last spike


@dataclass
class SamplingController:
    """Per-concern adaptive processing-density. Feed every captured frame's ``(t, motion)``; it answers
    whether THIS concern should run its expensive perception on this frame."""

    default: SensePolicy = field(default_factory=SensePolicy)
    _policy: dict[str, SensePolicy] = field(default_factory=dict)
    _last_processed: dict[str, float] = field(default_factory=dict)
    _boost_until: dict[str, float] = field(default_factory=dict)

    def configure(self, concern_id: str, policy: SensePolicy) -> None:
        """Override the density envelope for one concern (from its goal-set ``sense`` hint)."""
        self._policy[concern_id] = policy

    def policy(self, concern_id: str) -> SensePolicy:
        return self._policy.get(concern_id, self.default)

    def should_process(self, concern_id: str, t: float, motion: float) -> bool:
        """Spend expensive perception on this captured frame for ``concern_id``? Coast at ``base_fps``;
        escalate to ``max_fps`` while motion is (recently) above threshold; never exceed ``max_fps``."""
        p = self.policy(concern_id)
        if motion >= p.escalate_threshold:
            self._boost_until[concern_id] = t + p.hold_s  # (re)arm the saccade hold
        boosted = t < self._boost_until.get(concern_id, float("-inf"))
        target = p.max_fps if boosted else p.base_fps
        if target <= 0:
            return False
        interval = 1.0 / target
        last = self._last_processed.get(concern_id)
        # -1e-9 so an exact-rate capture (e.g. base divides the capture fps) isn't tripped by float error.
        if last is None or (t - last) >= interval - 1e-9:
            self._last_processed[concern_id] = t
            return True
        return False

    def reset(self, concern_id: str | None = None) -> None:
        """Drop timing state (a stream discontinuity); ``None`` clears all concerns."""
        if concern_id is None:
            self._last_processed.clear()
            self._boost_until.clear()
        else:
            self._last_processed.pop(concern_id, None)
            self._boost_until.pop(concern_id, None)


__all__ = ["SensePolicy", "SamplingController"]
