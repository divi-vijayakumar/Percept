"""Edge detector registry — the on-device twin of the sense axis (Phase 4).

A curated, registry-extensible library of CHEAP, high-fps signal detectors the brain selects from per
goal — the **skills** model: register an impl + a manifest, and the brain composes it into concerns,
never a per-task ``if``. Each detector consumes a scalar signal stream and emits **timed boundaries**:
the edge proposes WHERE a rep / onset / threshold-crossing happened; the brain counts them and the VLM
confirms the semantics *sparingly* at those boundaries.

This is the RepCount-Δ0 fix in mechanism: counting is a cheap edge-signal job (oscillation periodicity),
NOT a per-frame VLM job. The first two detectors are pure stdlib DSP — no numpy, no model.

The boundary that keeps it honest (the sense-axis decision, restated): adding a detector is a deliberate
``register(...)`` + calibrate event, not the LLM synthesizing detectors at runtime. The brain *composes*
registered detectors via the manifest; it never *writes* them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable


# ---------------------------------------------------------------------------
# The skills manifest + the timed boundary (the up-channel event)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectorManifest:
    """Self-describing entry so the brain can COMPOSE a detector without hardcoding it.

    ``when_to_use`` is what the conductor reads to author it and what a sense→default map keys on;
    ``output`` documents the boundary it emits. This manifest contract is the one skills mechanism
    shared across the three registries (sense primitives, perception specs, edge detectors)."""

    name: str
    senses: str  # the physical signal it reads (e.g. "motion magnitude over time")
    when_to_use: str  # so the conductor can author it / the preset can default to it
    output: str  # the boundary it emits (e.g. "a 'peak' per motion oscillation = one rep")


@dataclass(frozen=True)
class Boundary:
    """A timed event the edge reports UP: WHERE something happened, cheaply, at native fps."""

    detector: str
    kind: str  # "peak" | "onset" | "threshold"
    t: float
    value: float


# ---------------------------------------------------------------------------
# Detectors — stateful streaming DSP: push(t, value) -> [Boundary]
# ---------------------------------------------------------------------------


class MotionPeriodicity:
    """Counts oscillations in the motion signal — one ``peak`` boundary per completed bump.

    A rep is an oscillation; the edge counts peaks of the motion magnitude cheaply at native fps,
    with NO model and NO per-frame VLM call. Adaptive prominence (``rel_drop``): a bump is confirmed
    only once the signal falls to ``(1-rel_drop)`` of its peak, so noise riding the way up doesn't
    over-count. ``min_height`` ignores still-scene jitter; ``refractory_s`` debounces.
    """

    signal = "motion"
    name = "motion-periodicity"

    def __init__(self, *, min_height: float = 0.02, rel_drop: float = 0.5, refractory_s: float = 0.0) -> None:
        self.min_height = min_height
        self.rel_drop = rel_drop
        self.refractory_s = refractory_s
        self.reset()

    def reset(self) -> None:
        self._peak_v = 0.0
        self._peak_t = 0.0
        self._have_peak = False
        self._last_emit_t = float("-inf")

    def push(self, t: float, v: float) -> list[Boundary]:
        if v > self._peak_v:  # still climbing this bump -> track the running peak
            self._peak_v, self._peak_t = v, t
            self._have_peak = self._peak_v >= self.min_height
            return []
        if self._have_peak and v <= self._peak_v * (1.0 - self.rel_drop):  # confirmed descent
            out: list[Boundary] = []
            if t - self._last_emit_t >= self.refractory_s:
                out.append(Boundary(self.name, "peak", self._peak_t, self._peak_v))
                self._last_emit_t = t
            self._peak_v, self._peak_t, self._have_peak = v, t, False  # start a new bump from the trough
            return out
        return []


class AcousticOnset:
    """Detects energy onsets (a hiss/grunt/beep starting) — one ``onset`` boundary per rising spike.

    Hysteresis crossing on an audio-energy signal: fire when energy crosses ``high`` having been below
    ``low`` (armed), then re-arm only after it falls back under ``low``. Cheap stdlib DSP on the mic
    stream that's already flowing for STT — the genuinely-new sensor, no model.
    """

    signal = "energy"
    name = "acoustic-onset"

    def __init__(self, *, high: float = 0.3, low: float = 0.15, refractory_s: float = 0.0) -> None:
        self.high = high
        self.low = low
        self.refractory_s = refractory_s
        self.reset()

    def reset(self) -> None:
        self._armed = True
        self._last_emit_t = float("-inf")

    def push(self, t: float, v: float) -> list[Boundary]:
        if self._armed and v >= self.high:
            out: list[Boundary] = []
            if t - self._last_emit_t >= self.refractory_s:
                out.append(Boundary(self.name, "onset", t, v))
                self._last_emit_t = t
            self._armed = False
            return out
        if not self._armed and v < self.low:
            self._armed = True
        return []


class PoseOpenness:
    """FRAME-backed edge skill (BlazePose) — an 'openness' waveform → one clean peak per rep.

    Counts fast periodic reps where frame-diff motion conflates each rep's open+close (over-counts) and
    the per-frame VLM aliases on fast motion (under-counts). openness = ankle-spread/shoulder-width +
    wrist-elevation, so it tracks every variation (legs-only via the spread, full jacks add the arms).

    The first MODEL-backed detector. BlazePose is edge-class (mobile, ~real-time CPU, ~3-6MB model), but
    it's still heavier than DSP — so it is **lazy** (mediapipe + the .task model load on first ``push``,
    never at import or registration) and **opt-in** (``percept-harness[pose]`` + a model path). The brain
    activates it per-goal via the WatchSpec; a user who never counts reps never loads it.
    """

    signal = "frame"
    name = "pose-openness"

    def __init__(self, *, min_height: float = 0.6, rel_drop: float = 0.4, refractory_s: float = 0.4,
                 model_path: str | None = None) -> None:
        self._mh, self._rd, self._ref = min_height, rel_drop, refractory_s
        self._model_path = model_path or os.environ.get("PERCEPT_POSE_MODEL", "")
        self._lmk = None
        self._mp = None
        self._peak = None
        self._last = 0.0

    def _ensure(self) -> None:
        if self._lmk is not None:
            return
        if not self._model_path:
            raise RuntimeError(
                "pose-openness needs a BlazePose .task model — set PERCEPT_POSE_MODEL "
                "(install percept-harness[pose] first)."
            )
        import mediapipe as mp  # lazy: heavy, optional; never imported at module load
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        opts = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=self._model_path),
            running_mode=vision.RunningMode.IMAGE,
        )
        self._mp = mp
        self._lmk = vision.PoseLandmarker.create_from_options(opts)
        self._peak = MotionPeriodicity(min_height=self._mh, rel_drop=self._rd, refractory_s=self._ref)

    def reset(self) -> None:
        self._last = 0.0
        if self._peak is not None:
            self._peak.reset()

    def push(self, t: float, frame: Any) -> list[Boundary]:
        """frame: an RGB uint8 HxWx3 array. Emits a 'peak' boundary per rep (relabeled from the inner
        peak detector to this detector's name)."""
        self._ensure()
        res = self._lmk.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame))
        if res.pose_landmarks:
            lm = res.pose_landmarks[0]
            spread = abs(lm[27].x - lm[28].x)
            sw = abs(lm[11].x - lm[12].x) or 0.1
            wrist_up = max(0.0, lm[11].y - lm[15].y) + max(0.0, lm[12].y - lm[16].y)
            self._last = spread / sw + wrist_up
        return [Boundary(self.name, b.kind, b.t, b.value) for b in self._peak.push(t, self._last)]


# ---------------------------------------------------------------------------
# The registry — register an impl + a manifest; the brain composes by name
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, tuple[Callable[..., Any], DetectorManifest]] = {}


def register(name: str, factory: Callable[..., Any], manifest: DetectorManifest) -> None:
    """Register a detector (a deliberate, calibratable event — never per-task code)."""
    _REGISTRY[name] = (factory, manifest)


def make_detector(name: str) -> Any:
    """Construct a fresh detector by registry name."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown detector {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name][0]()


def manifest_of(name: str) -> DetectorManifest:
    return _REGISTRY[name][1]


def list_manifests() -> list[DetectorManifest]:
    """All manifests — what the conductor reads to know which detectors it can compose."""
    return [m for _, m in _REGISTRY.values()]


def registered() -> list[str]:
    return sorted(_REGISTRY)


register(
    "motion-periodicity",
    MotionPeriodicity,
    DetectorManifest(
        name="motion-periodicity",
        senses="motion magnitude over time (frame-diff)",
        when_to_use="counting repetitive movements — reps, strokes, paces; any 'count the N-th time' on a periodic motion",
        output="a 'peak' boundary per motion oscillation (≈ one rep), with no per-frame VLM call",
    ),
)
register(
    "acoustic-onset",
    AcousticOnset,
    DetectorManifest(
        name="acoustic-onset",
        senses="audio energy over time (the mic stream already flowing for STT)",
        when_to_use="audio-cued events — a kettle/boil hiss rising, a timer beep, a rep grunt, a knock",
        output="an 'onset' boundary each time energy spikes up (a sound starting)",
    ),
)
register(
    "pose-openness",
    PoseOpenness,
    DetectorManifest(
        name="pose-openness",
        senses="body pose over time (BlazePose keypoints: ankle-spread + wrist-elevation)",
        when_to_use="counting FAST whole-body reps where motion-diff over-counts and the VLM aliases — "
        "jumping jacks, squats, push-ups; any rep with a clean open↔close body shape",
        output="one 'peak' boundary per rep (the open phase of the openness waveform)",
    ),
)


# ---------------------------------------------------------------------------
# Selection (sense-preset default + authored override) + the firewall
# ---------------------------------------------------------------------------

# goal type -> default detectors. Empty for now (byte-for-byte today); a default (e.g. a count goal ->
# motion-periodicity) is switched on only when the Phase-7 ablation proves its lift. Authored overrides.
_TYPE_DETECTOR_PRESET: dict[str, list[str]] = {}


def coerce_authored_detectors(raw: Any) -> list[str]:
    """Validate an authored detector list against the registry vocab — the firewall: unknown names are
    DROPPED, so a goal can never activate a detector that isn't a registered, calibrated skill."""
    if not isinstance(raw, (list, tuple)):
        return []
    seen: list[str] = []
    for d in raw:
        if isinstance(d, str) and d in _REGISTRY and d not in seen:
            seen.append(d)
    return seen


def detectors_for(goal: Any) -> list[str]:
    """Which edge detectors to activate for a goal: authored override (validated) else the type preset."""
    authored = coerce_authored_detectors(getattr(goal, "state", {}).get("detectors"))
    if authored:
        return authored
    return list(_TYPE_DETECTOR_PRESET.get(getattr(goal, "state", {}).get("type", "watch"), []))


# ---------------------------------------------------------------------------
# DetectorBank — run a goal's selected detectors over a signal stream
# ---------------------------------------------------------------------------


class DetectorBank:
    """Runs a goal's selected detectors over a stream, routing each its signal channel (motion/energy).

    The brain feeds the cheap signals the edge already computes (``motion`` from the frame-diff gate,
    ``energy`` from the audio buffer); each detector reads its own channel and emits boundaries. A
    detector whose channel isn't provided this tick simply gets no input."""

    def __init__(self, names: Iterable[str]) -> None:
        self._dets = [make_detector(n) for n in names]

    def reset(self) -> None:
        for d in self._dets:
            d.reset()

    def push(self, t: float, *, motion: float | None = None, energy: float | None = None,
             frame: Any = None) -> list[Boundary]:
        out: list[Boundary] = []
        for d in self._dets:
            chan = getattr(d, "signal", "motion")
            if chan == "frame":  # frame-backed detector (e.g. pose) — pass the raw frame, not a scalar
                if frame is not None:
                    out.extend(d.push(t, frame))
                continue
            sig = energy if chan == "energy" else motion
            if sig is None:
                continue
            out.extend(d.push(t, float(sig)))
        return out


__all__ = [
    "DetectorManifest",
    "Boundary",
    "MotionPeriodicity",
    "AcousticOnset",
    "PoseOpenness",
    "register",
    "make_detector",
    "manifest_of",
    "list_manifests",
    "registered",
    "coerce_authored_detectors",
    "detectors_for",
    "DetectorBank",
]
