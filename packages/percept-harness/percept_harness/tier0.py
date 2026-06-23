"""Tier-0 salience gate — the server-side reference for the edge wire contract.

The perception analog of voice VAD (spec §6.3): a coarse pass that decides *when* to spend a
cloud vision call.

- **down:** the brain compiles active goals to a :class:`~percept.adapters.WatchSpec`
  (:func:`compile_watchspec`) — anchors + thresholds, one ``WatchCondition`` per watch goal.
- **up:** the gate reports a :class:`~percept.adapters.Tier0Signal` (motion / structural / semantic
  salience). This reference computes ``motion`` only (structural/semantic = 0.0); the real
  ``@percept/edge`` adds CLIP / pose.
- the expensive ``Percept.perceive_judged`` runs **only when salience clears threshold**.

Frame-diff is arithmetic — pure stdlib, deterministic, zero user data (spec §6.3). ``numpy`` is an
OPTIONAL fast path for :func:`motion_score`; it is **never required** for the L1 lane and the
stdlib and numpy paths return the same number.

**The wired loop** (:class:`GatedWatchLoop` / :func:`run_gated_watch`) is the proof point:
``compile_watchspec``'s output is actually consumed by the gate, and a ``Tier0Signal`` is reported
back up — even motion-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from percept.adapters import Box, Goal, Tier0Signal, WatchCondition, WatchSpec
from percept.config import Settings, load_settings

from .detectors import Boundary, DetectorBank, detectors_for, partition_by_capability

try:  # OPTIONAL fast path only — the stdlib path below is the source of truth for L1.
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - numpy is not a dependency of the L1 lane.
    _np = None


# ---------------------------------------------------------------------------
# Frame representation (SIMPLE, stdlib, deterministic)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Frame:
    """A coarse grayscale grid — the simplest thing a motion gate can diff.

    ``pixels`` is a flat, row-major tuple of luma ints in ``0..255`` (length ``w * h``). This is a
    deliberately tiny representation: the tier-0 gate never needs the full frame, only enough to
    tell "still" from "something moved". The real edge diffs a downscaled luma plane; the shape is
    the same.
    """

    w: int
    h: int
    pixels: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.w <= 0 or self.h <= 0:
            raise ValueError(f"frame dims must be positive, got {self.w}x{self.h}")
        if len(self.pixels) != self.w * self.h:
            raise ValueError(
                f"pixel count {len(self.pixels)} != {self.w}*{self.h}={self.w * self.h}"
            )


def gray_frame(w: int, h: int, fill: int = 0) -> Frame:
    """A uniform grayscale frame (handy for fixtures / a 'still scene' baseline)."""
    return Frame(w=w, h=h, pixels=tuple([_clamp_u8(fill)] * (w * h)))


def _clamp_u8(v: int) -> int:
    return 0 if v < 0 else 255 if v > 255 else int(v)


def _as_frame(obj: Any) -> Frame:
    """Coerce a frame-ish object to a :class:`Frame`.

    Accepts a :class:`Frame` directly, or a ``(w, h, pixels)`` tuple. Anything else is a wire /
    contract mismatch and raises — the gate must diff a *known* representation, not guess.
    """
    if isinstance(obj, Frame):
        return obj
    if isinstance(obj, tuple) and len(obj) == 3:
        w, h, pixels = obj
        return Frame(w=int(w), h=int(h), pixels=tuple(int(p) for p in pixels))
    raise TypeError(f"unsupported frame type {type(obj).__name__!r}; expected Frame or (w,h,pixels)")


# ---------------------------------------------------------------------------
# motion_score — deterministic pixel-diff (still -> ~0, big change -> high)
# ---------------------------------------------------------------------------


def motion_score(prev_frame: Any, frame: Any) -> float:
    """Mean absolute luma difference between two frames, normalized to ``0..1``.

    Still scene -> ``~0``; a large change -> high (toward ``1.0``). Deterministic and pure stdlib;
    ``numpy`` is used only as a speed path and yields the identical value.

    ``prev_frame is None`` (the first frame, no predecessor) -> ``0.0``: there is nothing to diff,
    so the gate treats the very first frame as "no motion yet". Mismatched dimensions are a
    contract error and raise.
    """
    if prev_frame is None:
        return 0.0
    a = _as_frame(prev_frame)
    b = _as_frame(frame)
    if a.w != b.w or a.h != b.h:
        raise ValueError(f"frame size mismatch: {a.w}x{a.h} vs {b.w}x{b.h}")
    n = a.w * a.h
    if n == 0:  # pragma: no cover - Frame.__post_init__ forbids empty frames.
        return 0.0
    if _np is not None:
        pa = _np.frombuffer(bytes(a.pixels), dtype=_np.uint8).astype(_np.int16)
        pb = _np.frombuffer(bytes(b.pixels), dtype=_np.uint8).astype(_np.int16)
        total = int(_np.abs(pa - pb).sum())
    else:
        total = 0
        for pa, pb in zip(a.pixels, b.pixels):
            total += pa - pb if pa >= pb else pb - pa
    # 255 is the max per-pixel abs diff -> divide to land in 0..1.
    return total / (n * 255.0)


# ---------------------------------------------------------------------------
# compile_watchspec — goals DOWN to a WatchSpec (the contract the edge receives)
# ---------------------------------------------------------------------------

# Per-goal threshold bias lives on the goal's mutable state under this key (optional). The
# conductor (or a tuning sweep) can bias a noisy goal up / a critical goal down; absent -> 0.0.
MOTION_BIAS_KEY = "motion_bias"
# Optional per-goal region-of-interest hint (a normalized Box dict or Box) under this key.
REGION_HINT_KEY = "region_hint"


def compile_watchspec(
    goals: Iterable[Goal], *, threshold: float, version: int = 1,
    edge_capabilities: Iterable[str] | None = None,
) -> WatchSpec:
    """Compile active *watch* goals DOWN to a :class:`~percept.adapters.WatchSpec`.

    One :class:`~percept.adapters.WatchCondition` per watch goal:

    - ``goal_id`` <- ``goal.id``
    - ``anchor_text`` <- ``goal.condition`` (the plain-language anchor the edge salience-matches)
    - ``threshold`` <- the gate default ``threshold`` + optional per-goal bias
      (``goal.state['motion_bias']``), clamped to ``0..1``
    - ``region_hint`` <- optional ``goal.state['region_hint']`` (a :class:`~percept.adapters.Box`)
    - ``detectors`` <- the goal's selected edge skills (sense-preset + override), **filtered to**
      ``edge_capabilities`` when given (the capability handshake)

    ``edge_capabilities`` (optional): the skill set the connected edge advertised. When given, each goal's
    detectors are intersected with it so the brain never sends a skill the edge can't run; ``None``
    (default, in-process topology A — the edge runs the whole registry) sends the full selection. Use
    :func:`watchspec_unsupported` to see what was dropped (to pick a fallback).

    Only goals whose ``state['type']`` is ``'watch'`` (the default) compile — event / count /
    remind goals run their own tracks and never gate the watch-frame stream.
    """
    conditions: list[WatchCondition] = []
    for goal in goals:
        if goal.state.get("type", "watch") != "watch":
            continue
        bias = float(goal.state.get(MOTION_BIAS_KEY, 0.0))
        supported, _dropped = partition_by_capability(detectors_for(goal), edge_capabilities)
        conditions.append(
            WatchCondition(
                goal_id=goal.id,
                anchor_text=goal.condition,
                threshold=_clamp01(threshold + bias),
                region_hint=_coerce_box(goal.state.get(REGION_HINT_KEY)),
                detectors=supported,  # cheap edge skills the brain selected, filtered to what the edge can run
            )
        )
    return WatchSpec(conditions=conditions, version=version)


def watchspec_unsupported(
    goals: Iterable[Goal], edge_capabilities: Iterable[str]
) -> list[tuple[str, list[str]]]:
    """``(goal_id, dropped_detectors)`` the brain selected but the advertised edge can't run — so the
    caller can pick a fallback (server-side detection / a coarser skill / skip). Empty when the edge
    supports everything selected. The reporting half of :func:`compile_watchspec`'s capability handshake.
    """
    out: list[tuple[str, list[str]]] = []
    for goal in goals:
        if goal.state.get("type", "watch") != "watch":
            continue
        _, dropped = partition_by_capability(detectors_for(goal), edge_capabilities)
        if dropped:
            out.append((goal.id, dropped))
    return out


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else float(v)


def _coerce_box(obj: Any) -> Box | None:
    if obj is None or isinstance(obj, Box):
        return obj
    if isinstance(obj, dict):
        return Box(x=obj["x"], y=obj["y"], w=obj["w"], h=obj["h"])
    return None  # pragma: no cover - defensive: unknown hint shape -> no hint.


# ---------------------------------------------------------------------------
# MotionGate — consumes a WatchSpec + a frame, emits a Tier0Signal + a decision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateDecision:
    """One gate decision: the reported :class:`Tier0Signal` + whether to escalate.

    ``escalate`` True -> spend the cloud vision call; False -> skip (the still-scene economics).
    ``threshold`` is the effective cutoff actually applied (the lowest per-condition threshold in
    the spec, so the most-sensitive goal wins — any goal that wants to look gets to look).
    """

    signal: Tier0Signal
    escalate: bool
    threshold: float


class MotionGate:
    """The coarse tier-0 gate: diff frames, report salience, decide escalate-vs-skip.

    Consumes a :class:`~percept.adapters.WatchSpec` (the anchors + thresholds the brain compiled
    down) and a frame stream. For each frame it computes ``motion`` against the previous frame,
    packs a :class:`~percept.adapters.Tier0Signal` (``structural``/``semantic`` = 0.0 in this
    motion-only reference), and escalates when ``motion`` clears the effective threshold.

    Effective threshold = the **minimum** per-condition threshold in the spec (the most sensitive
    goal wins), falling back to the gate default (``PERCEPT_MOTION_GATE``) when the spec is empty.
    An empty spec means no active watch goals -> nothing to look for -> never escalate.
    """

    def __init__(self, spec: WatchSpec, *, settings: Settings | None = None) -> None:
        self._spec = spec
        cfg = settings or load_settings()
        self._default = cfg.edge.motion_gate
        self._prev: Frame | None = None

    @property
    def spec(self) -> WatchSpec:
        return self._spec

    def threshold(self) -> float:
        """The effective escalation cutoff for the current spec."""
        if not self._spec.conditions:
            return self._default
        return min(c.threshold for c in self._spec.conditions)

    def reset(self) -> None:
        """Forget the previous frame (e.g. a stream discontinuity / new source)."""
        self._prev = None

    def evaluate(self, frame: Any, *, ts: float) -> GateDecision:
        """Score one frame against the prior, report a ``Tier0Signal``, decide escalate-vs-skip.

        An empty WatchSpec (no active watch goals) never escalates — there is nothing to look for.
        Otherwise escalate when ``motion >= effective threshold``. Advances the prev-frame cursor.
        """
        f = _as_frame(frame)
        motion = motion_score(self._prev, f)
        self._prev = f
        thr = self.threshold()
        signal = Tier0Signal(motion=motion, structural=0.0, semantic=0.0, ts=ts)
        escalate = bool(self._spec.conditions) and motion >= thr
        return GateDecision(signal=signal, escalate=escalate, threshold=thr)


# ---------------------------------------------------------------------------
# The wired loop — the proof point (compile -> gate -> perceive_judged -> report up)
# ---------------------------------------------------------------------------


@dataclass
class LoopReport:
    """The end-of-run accounting the Phase-6 cost demo prints.

    ``signals`` is the upward stream of :class:`~percept.adapters.Tier0Signal` (one per frame).
    ``fire_events`` collects whatever ``perceive_judged`` returned on escalated frames (flattened).
    """

    frames: int = 0
    vision_calls: int = 0  # escalated frames -> real perceive_judged calls actually spent
    escalations: int = 0
    fire_events: list = field(default_factory=list)
    signals: list[Tier0Signal] = field(default_factory=list)
    # edge boundaries the detectors emit (the count comes from HERE, not from a per-frame VLM call):
    # (goal_id, Boundary) pairs across the whole run.
    boundaries: list[tuple[str, Boundary]] = field(default_factory=list)

    def boundary_count(self, goal_id: str, *, kind: str | None = None) -> int:
        """How many boundaries a goal's detectors proposed (e.g. peaks = reps), optionally by kind."""
        return sum(
            1 for gid, b in self.boundaries if gid == goal_id and (kind is None or b.kind == kind)
        )

    @property
    def ungated_calls(self) -> int:
        """What a naive 'perceive every frame' loop would have spent (the baseline)."""
        return self.frames

    @property
    def reduction(self) -> float:
        """Fractional reduction in vision calls vs the ungated baseline (0..1)."""
        if self.frames == 0:
            return 0.0
        return 1.0 - (self.vision_calls / self.frames)


@dataclass(frozen=True)
class _FrameItem:
    frame: Any
    ts: float


def _normalize_stream(stream: Iterable[Any], *, clock_start: float, dt: float) -> list[_FrameItem]:
    """Accept ``frame`` or ``(ts, frame)`` items; synthesize timestamps where absent.

    A bare frame is a :class:`Frame` or a ``(w, h, pixels)`` len-3 tuple. A timestamped item is a
    ``(ts, frame)`` len-2 tuple whose first element is a number — the only len-2 form, so it is
    unambiguous against a (len-3) frame tuple.
    """
    items: list[_FrameItem] = []
    t = clock_start
    for entry in stream:
        if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], (int, float)):
            ts, frame = entry
            items.append(_FrameItem(frame=frame, ts=float(ts)))
        else:
            items.append(_FrameItem(frame=entry, ts=t))
            t += dt
    return items


class GatedWatchLoop:
    """The wired WatchSpec↓ / Tier0Signal↑ loop against a live :class:`~percept.agent.Percept`.

    Construction compiles the core's **active goals** down to a :class:`WatchSpec` and arms a
    :class:`MotionGate` with it. :meth:`run` replays a frame stream: per frame it gates, reports a
    :class:`Tier0Signal` up, and calls ``core.perceive_judged`` **only on escalation** — the
    cloud call is spent exactly when salience clears threshold. Returns a :class:`LoopReport` with
    gated-vs-ungated call counts (the cost graph) and the upward signal stream.

    This is the proof point: ``compile_watchspec``'s output is consumed by the gate, and a
    ``Tier0Signal`` is reported on every frame.
    """

    def __init__(
        self,
        core: Any,
        goals: Sequence[Goal] | None = None,
        *,
        threshold: float | None = None,
        source: str = "camera",
        settings: Settings | None = None,
        edge_capabilities: Iterable[str] | None = None,
    ) -> None:
        self._core = core
        cfg = settings or getattr(core, "settings", None) or load_settings()
        self._settings = cfg
        self._source = source
        active = list(goals) if goals is not None else list(core.goals.active())
        thr = cfg.edge.motion_gate if threshold is None else threshold
        # edge_capabilities=None -> in-process edge runs the whole registry; a remote edge advertises a
        # subset and the brain selects within it (the capability handshake).
        self._spec = compile_watchspec(active, threshold=thr, edge_capabilities=edge_capabilities)
        self._gate = MotionGate(self._spec, settings=cfg)
        # One detector bank per goal that selected detectors — the edge proposes timed boundaries from
        # the cheap motion signal at native fps (no extra vision calls). Default: no detectors -> no banks.
        self._banks = {
            c.goal_id: DetectorBank(c.detectors) for c in self._spec.conditions if c.detectors
        }

    @property
    def spec(self) -> WatchSpec:
        """The compiled WatchSpec the gate is consuming (the downward contract)."""
        return self._spec

    @property
    def gate(self) -> MotionGate:
        return self._gate

    async def run(
        self, stream: Iterable[Any], *, clock_start: float = 0.0, dt: float = 1.0,
        energies: Sequence[float] | None = None,
    ) -> LoopReport:
        """Drive the loop over a frame stream; gate, report up, perceive only on escalation.

        ``stream`` yields either bare frames (timestamps synthesized from ``clock_start`` stepping
        by ``dt``) or ``(ts, frame)`` pairs. ``energies`` (optional) is a per-frame audio-energy signal
        (one float per frame, same order) feeding energy-backed detectors like acoustic-onset — the audio
        analog of the motion channel, the same energy already computed off the mic/STT stream. Returns
        the :class:`LoopReport`.
        """
        report = LoopReport()
        items = _normalize_stream(stream, clock_start=clock_start, dt=dt)
        for i, item in enumerate(items):
            decision = self._gate.evaluate(item.frame, ts=item.ts)
            energy = energies[i] if energies is not None and i < len(energies) else None
            report.frames += 1
            report.signals.append(decision.signal)  # report Tier0Signal UP every frame
            # Edge detectors run on the cheap motion/energy signals at native fps — the count/boundaries
            # come from here, decoupled from the (sparse, motion-gated) vision calls below. Each boundary
            # is fed to the brain as a distinct occurrence (edge proposes · brain counts), firing per the
            # goal's accumulate.fire_when without spending a vision call.
            count_boundary = getattr(self._core, "perceive_boundary", None)
            for goal_id, bank in self._banks.items():
                # feed every channel the loop has: motion (always) + the raw frame (for frame-backed
                # detectors like pose) + audio energy (for acoustic-onset).
                for b in bank.push(item.ts, motion=decision.signal.motion, frame=item.frame, energy=energy):
                    report.boundaries.append((goal_id, b))
                    if count_boundary is not None:
                        fired = count_boundary(goal_id, t=b.t)
                        if fired is not None:
                            report.fire_events.append(fired)
            if decision.escalate:
                report.escalations += 1
                report.vision_calls += 1
                fired = await self._core.perceive_judged(
                    item.frame, self._source, t=item.ts
                )
                report.fire_events.extend(fired or [])
        return report


async def run_gated_watch(
    core: Any,
    stream: Iterable[Any],
    goals: Sequence[Goal] | None = None,
    *,
    threshold: float | None = None,
    source: str = "camera",
    clock_start: float = 0.0,
    dt: float = 1.0,
    energies: Sequence[float] | None = None,
    settings: Settings | None = None,
) -> LoopReport:
    """One-call entrypoint for the wired loop (the cross-package e2e drives this).

    Builds a :class:`GatedWatchLoop` (compiling ``goals`` — or the core's active goals — down to a
    WatchSpec) and runs it over ``stream`` (optionally with a per-frame ``energies`` audio signal).
    Returns the :class:`LoopReport`.
    """
    loop = GatedWatchLoop(
        core, goals, threshold=threshold, source=source, settings=settings
    )
    return await loop.run(stream, clock_start=clock_start, dt=dt, energies=energies)


# ---------------------------------------------------------------------------
# RemoteEdgeReceiver — the TOPOLOGY-B server shim (consume a remote on-device edge)
# ---------------------------------------------------------------------------


class RemoteEdgeReceiver:
    """Server-side consumer of a REMOTE on-device edge's up-channel (topology B).

    Unlike :class:`GatedWatchLoop` (topology A — gates frames in-process), the remote edge already gated
    and ran its skills on the device, and sends only messages up. This shim never gates or re-detects; it:

    1. takes the edge's advertised capabilities (an ``EdgeCapabilities`` message) so the brain selects
       skills within them — :meth:`watchspec` compiles the capability-filtered ``WatchSpec`` to send down;
    2. maps each incoming boundary (a ``BoundaryReport``) 1:1 to ``core.perceive_boundary(goal_id, t)`` —
       exactly what :class:`GatedWatchLoop` does inline (the ``count_boundary`` call), but driven by the
       wire instead of a local :class:`~.detectors.DetectorBank`.

    The frame/escalation path (the edge sends a frame up on salience -> ``core.perceive_judged``) is the
    product's media transport; this shim covers the capability + boundary halves of the contract.
    """

    def __init__(self, core: Any) -> None:
        self._core = core
        self._capabilities: list[str] | None = None

    def on_capabilities(self, capabilities: Iterable[str]) -> None:
        """Record the edge's advertised skill set (from an ``EdgeCapabilities`` message)."""
        self._capabilities = list(capabilities)

    @property
    def capabilities(self) -> list[str] | None:
        """The advertised set, or ``None`` if the edge hasn't advertised yet (brain then sends all)."""
        return None if self._capabilities is None else list(self._capabilities)

    def watchspec(
        self, *, threshold: float, goals: Sequence[Goal] | None = None, version: int = 1
    ) -> WatchSpec:
        """Compile the ``WatchSpec`` to send DOWN — detectors filtered to the edge's advertised skills."""
        active = list(goals) if goals is not None else list(self._core.goals.active())
        return compile_watchspec(
            active, threshold=threshold, version=version, edge_capabilities=self._capabilities
        )

    def on_boundaries(self, report: Any) -> list:
        """Consume a ``BoundaryReport`` (``{'boundaries': [{goal_id, ..., t}]}`` or a bare list of those
        events): call ``core.perceive_boundary(goal_id, t=t)`` per event; return the fire events.

        Mirrors :class:`GatedWatchLoop`'s inline boundary handling — the edge proposed the timed events,
        the brain counts them and fires per the goal's accumulate, with no cloud call here.
        """
        events = report.get("boundaries", []) if isinstance(report, dict) else list(report)
        perceive_boundary = getattr(self._core, "perceive_boundary", None)
        fires: list = []
        if perceive_boundary is None:
            return fires
        for ev in events:
            fired = perceive_boundary(ev["goal_id"], t=ev["t"])
            if fired is not None:
                fires.append(fired)
        return fires


__all__ = [
    "Frame",
    "gray_frame",
    "motion_score",
    "compile_watchspec",
    "watchspec_unsupported",
    "MOTION_BIAS_KEY",
    "REGION_HINT_KEY",
    "MotionGate",
    "GateDecision",
    "GatedWatchLoop",
    "LoopReport",
    "run_gated_watch",
    "RemoteEdgeReceiver",
]
