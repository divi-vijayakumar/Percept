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

It is the same allocation logic as the VLM cadence (``percept.harness.tier0`` / ``gate_on``), one tier
down: there the rationed resource is the cloud VLM call; here it is pose inference. Pure and deterministic
(timestamps + a cheap motion scalar in, boolean decisions out), so it unit-tests with no model. The runtime
captures at native fps and asks ``should_process`` before running the eye's expensive ``measure()``; a
fast-rep concern then resolves at up to ``max_fps`` while a "watch for my dog" concern coasts at ``base_fps``
— from the *same* high-rate capture, with no per-goal capture-fps to guess.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

from ._streamtime import StreamTimeGuard, StreamTimeRewindWarning
from .observability import get_logger


def _is_real_number(value) -> bool:
    """A finite ``int``/``float`` that is NOT a ``bool``.

    ``bool`` subclasses ``int``, so ``SensePolicy(base_fps=True)`` would otherwise be accepted as a 1 fps
    envelope — a type confusion, not a cadence. NaN/inf are refused HERE rather than left to each range
    comparison, because ``nan <= 0`` and ``nan > x`` are both False: an ordinary range guard admits NaN
    silently, which is exactly how a NaN cadence floor used to reach the fovea and blind a concern.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def is_usable_fps(value) -> bool:
    """True when ``value`` is a rate :class:`SensePolicy` will accept — real, finite, strictly positive.

    The test a caller assembling a cadence from UNTYPED data (a compiled plan dict, a resolved config
    mapping) must apply before constructing a policy, so an absent or unusable rate is dropped
    deliberately instead of reaching the constructor. ``bool(value)`` is not that test: it admits NaN
    (truthy!) while conflating an authored ``0`` with an absent key.
    """
    return _is_real_number(value) and value > 0


def validate_cadence(base_fps, max_fps, escalate_threshold, hold_s, error=ValueError) -> None:
    """Validate ONE processing-density envelope. This is the single cadence authority in the package.

    Every other cadence-shaped surface delegates here rather than re-deriving the rules:
    :class:`SensePolicy` (what the fovea actually runs on) and
    :class:`~percept.watchprofile.CadenceProfile` (the profile's typed handle onto it). ``error`` lets a
    caller keep its own exception type — CadenceProfile passes ``WatchProfileError`` so its messages and
    type are unchanged — without owning a second copy of the invariants.

    The invariants: ``base_fps``/``max_fps`` real, finite and ``> 0``; ``base_fps <= max_fps`` (an
    inverted envelope runs the saccade BACKWARDS — motion makes processing sparser, and the quiet floor
    blows straight through the ``max_fps`` cost ceiling); ``escalate_threshold`` in ``[0, 1]``;
    ``hold_s >= 0``. A zero ``base_fps`` is refused rather than clamped: it is not a slow cadence, it is
    a concern that can never be perceived on a quiet scene.
    """
    if not _is_real_number(base_fps) or not _is_real_number(max_fps) or base_fps <= 0 or max_fps <= 0:
        raise error(f"cadence fps must be > 0 (base_fps={base_fps}, max_fps={max_fps})")
    if base_fps > max_fps:
        raise error(f"cadence.base_fps ({base_fps}) must be <= cadence.max_fps ({max_fps})")
    if not _is_real_number(escalate_threshold) or not (0.0 <= escalate_threshold <= 1.0):
        raise error(f"cadence.escalate_threshold must be in 0..1, got {escalate_threshold}")
    if not _is_real_number(hold_s) or hold_s < 0:
        raise error(f"cadence.hold_s must be >= 0, got {hold_s}")


@dataclass(frozen=True)
class SensePolicy:
    """A concern's processing-density envelope (what the goal-set ``sense`` hint compiles to).

    ``base_fps`` is the quiet floor; ``max_fps`` the cost ceiling it escalates to on motion; ``hold_s`` keeps
    the escalation alive for a beat after the spike (the saccade settling). ``base_fps <= max_fps``.

    Validated at construction by :func:`validate_cadence`, so an envelope this type accepts is one
    :meth:`SamplingController.should_process` can honour. Every consumer runs on THIS type, so this is
    where the envelope has to be checked — a validator on a sibling handle that a construction site can
    bypass is not a guarantee."""

    base_fps: float = 4.0
    max_fps: float = 15.0
    escalate_threshold: float = 0.15  # cheap-motion level that triggers the saccade
    hold_s: float = 0.5  # keep escalated this long after the last spike

    def __post_init__(self) -> None:
        validate_cadence(self.base_fps, self.max_fps, self.escalate_threshold, self.hold_s)


@dataclass
class SamplingController:
    """Per-concern adaptive processing-density. Feed every captured frame's ``(t, motion)``; it answers
    whether THIS concern should run its expensive perception on this frame."""

    default: SensePolicy = field(default_factory=SensePolicy)
    _policy: dict[str, SensePolicy] = field(default_factory=dict)
    _last_processed: dict[str, float] = field(default_factory=dict)
    _boost_until: dict[str, float] = field(default_factory=dict)
    #: Per-concern monotonicity firewall for the untrusted stream timeline. Per concern, not per
    #: controller — see :meth:`SamplingController._concern_ts`.
    _ts_guards: dict[str, StreamTimeGuard] = field(default_factory=dict)

    def configure(self, concern_id: str, policy: SensePolicy) -> None:
        """Override the density envelope for one concern (from its goal-set ``sense`` hint)."""
        self._policy[concern_id] = policy

    def policy(self, concern_id: str) -> SensePolicy:
        return self._policy.get(concern_id, self.default)

    def _concern_ts(self, concern_id: str, t: float) -> float:
        """Validate one caller-supplied stream timestamp before any cadence arithmetic reads it.

        Two failures, two answers (see :class:`percept._streamtime.StreamTimeGuard`):

        * **not finite** — refused loudly. Every comparison against a NaN is False, so ``(t - last)
          >= interval`` never holds again *and* the NaN is written into ``_last_processed``, poisoning
          the concern for the rest of the run. There is no sane sampling decision to make on a NaN.
        * **backwards** — a stream discontinuity, and treated as one: drop this concern's timing
          state, which is exactly what :meth:`reset` already documents itself as being for. Left
          alone, ``t - last`` stays negative for the whole rewound span, so the interval test is
          never satisfied and the concern is **permanently blind through the rewind** — no
          perception, no exception, nothing in the trace.

        The guard is **per concern**, not per controller, because :meth:`should_process` is called
        once per concern per frame. A single shared guard would report ``rewind`` to whichever
        concern happened to be asked first and ``duplicate`` to all the rest, so only one of them
        would recover.

        This is deliberately NOT :mod:`percept.interest`'s raise-on-rewind policy — a rewind here is
        a reconnect or a file seek, and raising would turn recoverable source noise into a dead
        watch. It matches :meth:`percept.harness.tier0.MotionGate._gate_ts`, which chose RESET on the
        same untrusted timeline for the same reason.
        """
        guard = self._ts_guards.get(concern_id)
        if guard is None:
            guard = self._ts_guards[concern_id] = StreamTimeGuard()
        verdict = guard.check("t", t)
        value = float(t)
        if verdict == "rewind":
            message = (
                f"sampling controller saw the stream timeline step backwards to t={value!r} for "
                f"concern {concern_id!r}; treating it as a source discontinuity and dropping that "
                "concern's timing state so it is not blinded for the rest of the rewound span"
            )
            warnings.warn(message, StreamTimeRewindWarning, stacklevel=3)
            get_logger("percept.sampling").warning("%s", message)
            self._last_processed.pop(concern_id, None)
            self._boost_until.pop(concern_id, None)
        return value

    def should_process(self, concern_id: str, t: float, motion: float) -> bool:
        """Spend expensive perception on this captured frame for ``concern_id``? Coast at ``base_fps``;
        escalate to ``max_fps`` while motion is (recently) above threshold; never exceed ``max_fps``.

        ``t`` is the untrusted stream timeline and is validated first — see :meth:`_concern_ts`."""
        t = self._concern_ts(concern_id, t)
        p = self.policy(concern_id)
        if motion >= p.escalate_threshold:
            self._boost_until[concern_id] = t + p.hold_s  # (re)arm the saccade hold
        boosted = t < self._boost_until.get(concern_id, float("-inf"))
        # No `target <= 0` guard: SensePolicy refuses a non-positive rate at construction, so a policy
        # that reaches here always has one. The guard used to sit ABOVE the `last is None` first-call
        # branch below, which is what turned a base_fps of 0 into permanent, silent, error-free
        # blindness on a quiet scene rather than a loud construction failure.
        target = p.max_fps if boosted else p.base_fps
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


__all__ = ["SensePolicy", "SamplingController", "validate_cadence", "is_usable_fps"]
