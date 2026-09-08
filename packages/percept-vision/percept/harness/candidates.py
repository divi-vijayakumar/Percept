"""Cheap, stateful candidate generators for sparse semantic verification.

Candidate generators do not decide that a user condition is true.  They only
identify a small number of frames that deserve the expensive judge.  The
judge still produces the :class:`~percept.adapters.Verdict` and cognition
still owns the fire/ask/silent decision.

The first implementation is intentionally narrow: a fixed-camera collision
candidate.  It detects a localized *shock* (a sharp increase in tile motion)
after recent in-zone motion, while rejecting near-global changes such as cuts
and exposure jumps.  It is a pure-stdlib callable over the same 48x36 luma
``Frame`` the Tier-0 gate already computes, so it adds no model dependency and
can be supplied directly as ``semantic_adapter=`` to ``percept.run``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from .tier0 import Frame, _as_frame


@dataclass(frozen=True)
class CollisionCandidateReceipt:
    """Diagnostic values from the most recent candidate observation."""

    score: float
    candidate: bool
    local_motion: float
    local_shock: float
    recent_motion: float
    global_change_fraction: float
    reason: str


class FixedCameraCollisionCandidate:
    """Propose sparse collision verification calls on a fixed camera.

    ``region`` is camera calibration, not an event annotation: normalized
    ``(x, y, width, height)`` for the road/intersection in which contact is
    relevant.  The detector deliberately ignores object identity.  Its job is
    to retain recall cheaply; the VLM confirmation decides whether vehicles
    actually collided.

    The returned value is a semantic-gate score in ``[0, 1]``.  A candidate
    tick returns at least ``candidate_score`` and all other ticks return zero.
    The detector is edge-triggered and has a bounded cooldown, so a prolonged
    crash scene produces one verification call rather than one call per frame.
    """

    signal = "frame"
    name = "fixed-camera-collision-candidate"

    def __init__(
        self,
        *,
        region: tuple[float, float, float, float] = (0.10, 0.20, 0.84, 0.65),
        grid: tuple[int, int] = (6, 4),
        min_local_motion: float = 0.15,
        min_local_shock: float = 0.08,
        min_recent_motion: float = 0.012,
        recent_frames: int = 8,
        warmup_frames: int = 20,
        cut_pixel_delta: int = 48,
        min_global_change_fraction: float = 0.05,
        max_global_change_fraction: float = 0.58,
        rearm_motion: float = 0.025,
        rearm_frames: int = 3,
        cooldown_frames: int = 8,
        candidate_score: float = 1.0,
    ) -> None:
        x, y, w, h = region
        gx, gy = grid
        if not (0.0 <= x < 1.0 and 0.0 <= y < 1.0 and w > 0.0 and h > 0.0):
            raise ValueError("region must be a positive normalized rectangle")
        if x + w > 1.0 or y + h > 1.0:
            raise ValueError("region must lie within the normalized frame")
        if gx <= 0 or gy <= 0:
            raise ValueError("grid dimensions must be positive")
        if recent_frames <= 0 or warmup_frames < 2 or rearm_frames <= 0 or cooldown_frames < 0:
            raise ValueError("temporal frame counts are invalid")
        for name, value in (
            ("min_local_motion", min_local_motion),
            ("min_local_shock", min_local_shock),
            ("min_recent_motion", min_recent_motion),
            ("min_global_change_fraction", min_global_change_fraction),
            ("max_global_change_fraction", max_global_change_fraction),
            ("rearm_motion", rearm_motion),
            ("candidate_score", candidate_score),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0 <= int(cut_pixel_delta) <= 255:
            raise ValueError("cut_pixel_delta must be in [0, 255]")

        self.region = tuple(float(v) for v in region)
        self.grid = (int(gx), int(gy))
        self.min_local_motion = float(min_local_motion)
        self.min_local_shock = float(min_local_shock)
        self.min_recent_motion = float(min_recent_motion)
        self.recent_frames = int(recent_frames)
        self.warmup_frames = int(warmup_frames)
        self.cut_pixel_delta = int(cut_pixel_delta)
        self.min_global_change_fraction = float(min_global_change_fraction)
        self.max_global_change_fraction = float(max_global_change_fraction)
        self.rearm_motion = float(rearm_motion)
        self.rearm_frames = int(rearm_frames)
        self.cooldown_frames = int(cooldown_frames)
        self.candidate_score = float(candidate_score)
        self.reset()

    def reset(self) -> None:
        self._prev: Frame | None = None
        self._previous_tiles: tuple[float, ...] = ()
        self._recent: deque[float] = deque(maxlen=self.recent_frames)
        self._cooldown = 0
        self._seen = 0
        self._armed = True
        self._quiet = 0
        self.last_receipt = CollisionCandidateReceipt(
            score=0.0,
            candidate=False,
            local_motion=0.0,
            local_shock=0.0,
            recent_motion=0.0,
            global_change_fraction=0.0,
            reason="cold_start",
        )

    @staticmethod
    def _coerce_frame(frame: Any) -> Frame:
        if isinstance(frame, Frame) or (isinstance(frame, tuple) and len(frame) == 3):
            return _as_frame(frame)
        from .pose import frame_pixels

        pixels = frame_pixels(frame)
        try:
            import cv2  # noqa: PLC0415 - supplied by the runtime extra
        except ModuleNotFoundError as exc:  # pragma: no cover - frame_pixels raises first normally
            raise RuntimeError(
                "full-frame collision candidates need percept-vision[runtime]"
            ) from exc
        # Keep substantially more spatial detail than the generic 48x36 salience grid while
        # bounding CPU and retained state. ``frame_pixels`` returns RGB.
        gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
        return Frame(160, 90, tuple(int(value) for value in resized.flat))

    @staticmethod
    def _bounds(frame: Frame, region: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
        x, y, w, h = region
        x0 = max(0, min(frame.w, int(x * frame.w)))
        y0 = max(0, min(frame.h, int(y * frame.h)))
        x1 = max(x0 + 1, min(frame.w, int(round((x + w) * frame.w))))
        y1 = max(y0 + 1, min(frame.h, int(round((y + h) * frame.h))))
        return x0, y0, x1, y1

    def _tile_motion(self, prev: Frame, frame: Frame) -> tuple[tuple[float, ...], float]:
        if prev.w != frame.w or prev.h != frame.h:
            raise ValueError(
                f"frame size mismatch: {prev.w}x{prev.h} vs {frame.w}x{frame.h}"
            )
        x0, y0, x1, y1 = self._bounds(frame, self.region)
        gx, gy = self.grid
        tiles: list[float] = []
        for by in range(gy):
            ty0 = y0 + ((y1 - y0) * by) // gy
            ty1 = y0 + ((y1 - y0) * (by + 1)) // gy
            for bx in range(gx):
                tx0 = x0 + ((x1 - x0) * bx) // gx
                tx1 = x0 + ((x1 - x0) * (bx + 1)) // gx
                total = 0
                count = 0
                for py in range(ty0, ty1):
                    row = py * frame.w
                    for px in range(tx0, tx1):
                        a = prev.pixels[row + px]
                        b = frame.pixels[row + px]
                        total += a - b if a >= b else b - a
                        count += 1
                tiles.append(0.0 if count == 0 else total / (count * 255.0))

        changed = 0
        for a, b in zip(prev.pixels, frame.pixels):
            if abs(a - b) >= self.cut_pixel_delta:
                changed += 1
        return tuple(tiles), changed / len(frame.pixels)

    def __call__(self, frame: Any) -> float:
        current = self._coerce_frame(frame)
        self._seen += 1
        if self._prev is None:
            self._prev = current
            return 0.0

        tiles, global_change = self._tile_motion(self._prev, current)
        self._prev = current
        local_motion = max(tiles, default=0.0)
        local_shock = max(
            (now - before for now, before in zip(tiles, self._previous_tiles)),
            default=0.0,
        )
        recent_motion = max(self._recent, default=0.0)
        self._previous_tiles = tiles
        self._recent.append(local_motion)

        if self._cooldown > 0:
            self._cooldown -= 1
        # Rearm on a settled *shock derivative*, not on a still scene. A busy intersection may
        # never have three low-motion frames, which would permanently mute the detector after its
        # first proposal even though later independent impacts remain observable.
        if local_shock <= self.rearm_motion:
            self._quiet += 1
            if self._quiet >= self.rearm_frames and self._cooldown == 0:
                self._armed = True
        else:
            self._quiet = 0

        candidate = False
        reason = "below_candidate_threshold"
        if self._seen < self.warmup_frames:
            reason = "warming_up"
        elif global_change >= self.max_global_change_fraction:
            reason = "global_change_rejected"
            self._armed = False
        elif self._cooldown > 0:
            reason = "cooldown"
        elif not self._armed:
            reason = "awaiting_rearm"
        elif recent_motion < self.min_recent_motion:
            reason = "no_recent_motion"
        elif global_change < self.min_global_change_fraction:
            reason = "spatial_footprint_low"
        elif local_motion < self.min_local_motion:
            reason = "local_motion_low"
        elif local_shock < self.min_local_shock:
            reason = "local_shock_low"
        else:
            candidate = True
            reason = "localized_motion_shock"
            self._armed = False
            self._cooldown = self.cooldown_frames

        score = self.candidate_score if candidate else 0.0
        self.last_receipt = CollisionCandidateReceipt(
            score=score,
            candidate=candidate,
            local_motion=local_motion,
            local_shock=local_shock,
            recent_motion=recent_motion,
            global_change_fraction=global_change,
            reason=reason,
        )
        return score

    def push(self, t: float, frame: Any):
        """DetectorBank protocol: emit one candidate boundary on the local rising edge."""
        from .detectors import Boundary

        score = self(frame)
        if score <= 0.0:
            return []
        return [Boundary(self.name, "candidate", float(t), float(score))]


# Registration lives here (after tier0/detectors have initialized) to avoid a detectors<->tier0
# import cycle. Importing ``percept.harness`` or any of its public submodules executes the package
# initializer and makes this calibrated candidate available to authored ``state.detectors``.
from .detectors import (  # noqa: E402
    CATEGORY_SEMANTIC_RELEVANCE,
    DetectorManifest,
    register,
)

register(
    FixedCameraCollisionCandidate.name,
    FixedCameraCollisionCandidate,
    DetectorManifest(
        name=FixedCameraCollisionCandidate.name,
        senses="localized full-frame motion shock after recent in-zone motion",
        when_to_use="fixed-camera vehicle collision watches; proposes sparse VLM verification",
        output="one candidate boundary on a localized shock rising edge",
        category=CATEGORY_SEMANTIC_RELEVANCE,
        route="admission",
    ),
)


__all__ = ["CollisionCandidateReceipt", "FixedCameraCollisionCandidate"]
