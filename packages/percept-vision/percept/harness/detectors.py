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

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

# The SHARED numeric firewall, which matches the edge's JS `clamp01`: a non-numeric OR NON-FINITE
# reading is 0.0, then the value is pinned to [0, 1]. This module used to carry a third private copy
# that type-guarded but let a NaN through — and a NaN compares False against every cutoff, including
# 0.0, so a broken channel could admit LESS than a zero does while the JS twin read it as 0.0.
from percept._numeric import clamp01 as _clamp01


# ---------------------------------------------------------------------------
# Sense categories — the CLASS of physical signal a detector belongs to
# ---------------------------------------------------------------------------
#
# A detector's ``category`` is the coarse modality of the signal it reads, so a whole CLASS of
# detectors is discoverable at once — ``detectors_in_category(AUDIO)`` returns every audio Tier-0
# detector the same way the motion ones group under ``MOTION``. This is what makes ``acoustic-onset``
# a first-class *audio* sense: it is the AUDIO category, distinct from the MOTION/FRAME detectors,
# and the brain can ask "what audio senses does this edge have?" without hardcoding names.

#: Built-in sense categories. ``MOTION`` = frame-diff magnitude; ``AUDIO`` = the mic/energy stream;
#: ``FRAME`` = a model reading raw frames (e.g. pose). New categories are additive (semver MINOR).
CATEGORY_MOTION = "motion"
CATEGORY_AUDIO = "audio"
CATEGORY_FRAME = "frame"
CATEGORY_PRESENCE = "presence"
CATEGORY_POSE = "pose"
CATEGORY_AUDIO_EVENT = "audio_event"
CATEGORY_CHANGE = "change"
CATEGORY_SEMANTIC_RELEVANCE = "semantic_relevance"


# ---------------------------------------------------------------------------
# Wake policy — cheap evidence fusion before a VLM call
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WakePolicy:
    """Fuse cheap edge signals into ``P(goal)`` before spending a heavy vision call.

    The default Percept gate is intentionally simple: ``max(motion, structural, semantic)`` clears the
    condition threshold. A policy turns the same channels into a calibrated, weighted OR:

    ``1 - Π(1 - clamp(signal[channel] * weight[channel]))``

    **The fusion runs over the MEASURED channels, and ``weights`` is a table of per-channel priors
    with a default of 1.0.** That is what makes the weighted form a strict generalization of the
    unweighted one: naming a weight can only ever re-price a channel that is already being measured,
    never delete an unnamed channel from the fusion, so the fused score can never fall below the
    ``max(signals)`` it replaced. It used to iterate the WEIGHT table instead, which meant a policy
    that named one channel silenced every other measured one (``{'presence': 0.9, 'motion': 0.8}``
    scored 0.8 under ``weights={'motion': 1.0}``, below the 0.9 it replaced) and a single typo in a
    channel name scored 0.0 with every sensor reporting.

    A channel named in ``weights`` but ABSENT from ``signals`` contributes nothing — an unwired sensor
    cannot vote, in either direction. It is deliberately NOT read as ``0.0``: "this sensor is not
    reporting" and "this sensor sees nothing" are different facts, and only the second one is
    evidence.

    This preserves recall from multiple weak cues (e.g. person present + glass break + motion) without
    making one detector authoritative. It is the concrete ``Scalar × Magnitude × known CV × CLIP``
    hook: scalar priors live in ``weights``, magnitudes come from detector outputs.

    THE CHANNEL VOCABULARY IS THE PRODUCER'S, NOT THIS CLASS'S. The reference gate populates exactly
    four channels — see :func:`percept.harness.tier0.cheap_signals`, which holds the invariant that
    one physical measurement populates exactly one channel. Known-CV and CLIP-like senses
    (``presence``, ``pose``, ``audio_event``, ...) are channels a CALLER overlays via the gate's
    ``signals=`` argument once it has a detector that measures them; there is no alias for a channel
    nothing produces, because the noisy-OR assumes independent cues and two names for one measurement
    fuse that measurement with itself.
    """

    threshold: float = 0.1
    weights: dict[str, float] = field(default_factory=dict)
    floor: float = 0.0


@dataclass(frozen=True)
class WakeDecision:
    """A cheap wake decision with the fused score and the channel that contributed most.

    ``top_channel`` is the argmax of the same per-channel CONTRIBUTION terms the score summed — not of
    the raw signals — so it always names a channel that actually moved the number beside it. It is
    ``None`` when every contribution is 0.0, including when a ``floor`` lifted the score on its own:
    a score no channel produced has no top channel to name.
    """

    wake: bool
    score: float
    threshold: float
    top_channel: str | None = None


#: The complete authored surface of a wake policy. Anything else in the mapping is a typo, and a typo
#: is refused by name rather than dropped — see :class:`UnsupportedWakePolicyError`.
_WAKE_POLICY_KEYS = frozenset({"threshold", "weights", "floor"})


class UnsupportedWakePolicyError(ValueError):
    """An authored wake policy this module refuses to run, naming the field that made it unusable.

    THE MODULE-WIDE CHOICE, stated once here because the codebase contains both answers. The five
    ``_coerce_*`` helpers in :mod:`percept.harness.tier0` widen a malformed goal-state hint to a
    harmless default, because for a region/schedule/cadence hint "malformed" and "absent" describe the
    same watch. A wake policy is the opposite shape: it exists to REPLACE the default fusion, so a
    policy that silently degrades is indistinguishable from no policy at all while still carrying an
    authored threshold — "calibrated" and "running the legacy default" become the same run, and every
    number measured against it describes a configuration nobody deployed.

    So ``coerce_wake_policy`` refuses rather than degrades: an unknown top-level key (``'weigths'``),
    a non-string channel name, a bool where a number belongs (``True`` is an ``int`` in Python and
    would pass as a weight of 1.0), a stringified number, and a non-finite number are all typed
    refusals naming the field. There is no partially-applied policy and therefore no ``degraded`` flag
    to carry: a policy either ran as authored or never constructed. Coercion happens at gate
    construction and on the first fused frame, so the refusal lands at startup, not mid-run.
    """

    def __init__(self, field_name: str, value: Any, reason: str) -> None:
        self.field_name = field_name
        self.value = value
        super().__init__(f"wake policy field {field_name!r}={value!r} — {reason}")


def _wake_number(field_name: str, value: Any) -> float:
    """One finite float, or a typed refusal naming ``field_name``. A ``bool`` is never a number here."""
    if isinstance(value, bool):
        raise UnsupportedWakePolicyError(
            field_name, value,
            "a bool is not a number (a Python bool IS an int, so True would pass silently as 1.0)",
        )
    if not isinstance(value, (int, float)):
        raise UnsupportedWakePolicyError(
            field_name, value,
            "must be a finite int or float; a stringified number is refused rather than parsed, so "
            "the two runtimes cannot disagree about what a string is worth",
        )
    out = float(value)
    if out != out or out in (float("inf"), float("-inf")):
        raise UnsupportedWakePolicyError(field_name, value, "must be finite")
    return out


def coerce_wake_policy(raw: WakePolicy | dict[str, Any] | None) -> WakePolicy | None:
    """Normalize an optional policy object from Python code or JSON-ish config, or refuse it by name.

    ``None`` and an already-built :class:`WakePolicy` pass through. A mapping is validated field by
    field; anything unusable raises :class:`UnsupportedWakePolicyError` rather than degrading to a
    weightless policy. See that class for why this coercer is the strict member of the family.
    """
    if raw is None or isinstance(raw, WakePolicy):
        return raw
    if not isinstance(raw, dict):
        raise UnsupportedWakePolicyError(
            "<policy>", raw,
            f"a wake policy is a WakePolicy or a mapping of {sorted(_WAKE_POLICY_KEYS)}, not a "
            f"{type(raw).__name__}",
        )
    unknown = sorted(set(raw) - _WAKE_POLICY_KEYS)
    if unknown:
        raise UnsupportedWakePolicyError(
            "<policy>", unknown,
            f"unknown field(s); a wake policy carries exactly {sorted(_WAKE_POLICY_KEYS)}. A "
            "misspelled key used to read as an absent one, which left the authored threshold in "
            "place while the fusion quietly reverted to max(signals)",
        )
    weights_raw = raw.get("weights", {})
    if not isinstance(weights_raw, dict):
        raise UnsupportedWakePolicyError(
            "weights", weights_raw, "must be a mapping of channel name -> numeric prior",
        )
    weights: dict[str, float] = {}
    for key, value in weights_raw.items():
        if not isinstance(key, str):
            raise UnsupportedWakePolicyError(
                "weights", key, "channel names must be strings",
            )
        weights[key] = _wake_number(f"weights[{key!r}]", value)
    return WakePolicy(
        threshold=_wake_number("threshold", raw.get("threshold", 0.1)),
        weights=weights,
        floor=_wake_number("floor", raw.get("floor", 0.0)),
    )


def _fuse(
    signals: dict[str, Any], policy: WakePolicy | None
) -> tuple[float, dict[str, float]]:
    """THE fusion, once: the fused score AND the per-channel contribution terms it summed.

    One walk over the MEASURED channels produces both halves, which is the whole point — ``score`` and
    ``top_channel`` used to be computed from two different populations (the weights table and the raw
    signals respectively), so a decision could name a channel that contributed nothing to the number
    it was reported beside.

    ``contributions[channel]`` is exactly what the noisy-OR multiplied: ``clamp01(signal * weight)``,
    with the weight defaulting to 1.0 for a measured channel the policy did not price. On the
    unweighted path it is the raw clamped signal, which is what ``max(signals)`` selected over.
    """
    clean = {channel: _clamp01(value) for channel, value in signals.items()}
    if policy is None or not policy.weights:
        contributions = clean
        base = max(contributions.values()) if contributions else 0.0
    else:
        contributions = {
            channel: _clamp01(value * policy.weights.get(channel, 1.0))
            for channel, value in clean.items()
        }
        miss = 1.0
        for term in contributions.values():
            miss *= 1.0 - term
        base = 1.0 - miss
    floor = _clamp01(policy.floor) if policy is not None else 0.0
    return _clamp01(max(base, floor)), contributions


def wake_score(signals: dict[str, Any], policy: WakePolicy | dict[str, Any] | None = None) -> float:
    """Return a fused cheap-evidence score in ``[0, 1]``.

    With no explicit policy weights this is exactly ``max(signals)`` for backward compatibility. With
    weights it is a noisy-OR **over every measured channel**, each priced by its weight (default 1.0)
    — so several moderate cues can wake the gate together without any single cue needing to be
    perfect, and adding a weight can never score below the ``max(signals)`` it replaced. A ``floor``
    raises the result to at least that value. See :class:`WakePolicy` for why the population is the
    signals rather than the weights.
    """
    return _fuse(signals, coerce_wake_policy(policy))[0]


def should_wake(
    signals: dict[str, Any], policy: WakePolicy | dict[str, Any] | None = None
) -> WakeDecision:
    """Decide whether cheap evidence is strong enough to escalate to perception."""
    p = coerce_wake_policy(policy)
    threshold = p.threshold if p is not None else 0.1
    score, contributions = _fuse(signals, p)
    # The argmax of the CONTRIBUTIONS, ties taking the first (strict ``>`` from a 0.0 baseline). A
    # channel that contributed exactly nothing can never be named, so a policy that prices none of
    # the measured channels reports ``None`` rather than pointing at an irrelevant sensor.
    top: str | None = None
    best = 0.0
    for channel, term in contributions.items():
        if term > best:
            top, best = channel, term
    return WakeDecision(wake=score >= threshold, score=score, threshold=threshold, top_channel=top)


# ---------------------------------------------------------------------------
# The skills manifest + the timed boundary (the up-channel event)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectorManifest:
    """Self-describing entry so the brain can COMPOSE a detector without hardcoding it.

    ``when_to_use`` is what the conductor reads to author it and what a sense→default map keys on;
    ``output`` documents the boundary it emits. ``category`` is the coarse sense CLASS (one of
    ``CATEGORY_MOTION``/``CATEGORY_AUDIO``/``CATEGORY_FRAME``) so a whole modality of detectors is
    discoverable at once (``detectors_in_category(CATEGORY_AUDIO)``) — this is what marks
    ``acoustic-onset`` as the first-class *audio* sense. This manifest contract is the one skills
    mechanism shared across the three registries (sense primitives, perception specs, edge detectors).

    ``category`` is additive and optional (defaults ``""`` = uncategorized) so existing third-party
    manifests stay valid; the field name matches the JS twin's manifest, so it rides the wire unchanged."""

    name: str
    senses: str  # the physical signal it reads (e.g. "motion magnitude over time")
    when_to_use: str  # so the conductor can author it / the preset can default to it
    output: str  # the boundary it emits (e.g. "a 'peak' per motion oscillation = one rep")
    category: str = ""  # the coarse sense class: CATEGORY_MOTION | CATEGORY_AUDIO | CATEGORY_FRAME


@dataclass(frozen=True)
class Boundary:
    """A timed event the edge reports UP: WHERE something happened, cheaply, at native fps."""

    detector: str
    kind: str  # "peak" | "onset" | "threshold" | "sample"
    t: float
    value: float
    channel: str | None = None  # which decomposed channel this came from (rhythm sense); additive/optional


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


#: Shipped wake bar for :class:`AcousticOnset`, in MEAN-SQUARE ENERGY (not amplitude, not dB).
#:
#: THRESHOLD RATIONALE. The previous default was ``0.09`` — which is ``0.3 ** 2``, i.e. an RMS
#: AMPLITUDE of 0.3 (about -10 dBFS sustained). Measured against 987 s / 3,973 capture chunks of
#: human-annotated sound (Perception Test ``sound_localisation``, 53 cases), the LOUDEST chunk in the
#: whole corpus reached energy 0.0277 — so the shipped gate woke ZERO times and matched 0/204
#: annotated sound events. That default could not be crossed by the material it exists to hear.
#:
#: Measured recall / economy (wakes per chunk) frontier on that corpus, ``low = high / 4`` throughout:
#:
#: ===========  ======  =======
#: ``high``     recall  economy
#: ===========  ======  =======
#: 0.09 (old)   0.000   0.0000
#: 0.01         0.069   0.0035
#: 0.003        0.191   0.0108
#: **0.001**    0.294   0.0181
#: 0.0001       0.422   0.0393
#: 5e-05        0.461   0.0468   <- argmax
#: 1e-05        0.392   0.0418
#: ===========  ======  =======
#:
#: WHY ``0.001`` AND NOT THE ARGMAX. A bare energy gate tops out near 0.46 recall on this material at
#: ANY threshold, so no number here is a recall fix — this is a DEAFNESS fix, and it is the only thing
#: it claims to be. Given that the recall ceiling is the detector's, not the threshold's, the default
#: should be the value that survives material this corpus does not contain, not the value that peaks
#: on the material it does. The argmax (5e-05 energy = RMS 7e-3 = about -43 dBFS, re-arming below
#: -49 dBFS) is BELOW the noise floor of most live microphones: on a real mic it would wake once and
#: then never re-arm, which is a second flavour of deafness this corpus cannot see because file audio
#: has stretches of true digital silence (corpus p10 energy = 1.46e-07). ``0.001`` is RMS 0.032
#: (about -30 dBFS) with a re-arm bar at RMS 0.016 (about -36 dBFS) — the conventional "something
#: audible started" bar, above ordinary room tone, and a measured point on the frontier above
#: (0.294 recall at 1.8% of chunks admitted).
#:
#: HONEST LIMIT (deliberately not fixed here). No FIXED absolute energy is right across material:
#: microphone gain, AGC and codec loudness normalisation move the whole distribution, so the correct
#: long-term design is a bar RELATIVE to an observed noise floor rather than a constant. That is a
#: detector redesign, not a default change. Until it exists, ``percept.runtime.worker`` emits a
#: ``RuntimeHealthEvent("zero_wake_coverage")`` carrying the observed energy distribution beside these
#: thresholds whenever a run's gate never wakes, so "the room was quiet" and "my gate could not hear"
#: are distinguishable from the outside and the caller can retune from their OWN signal.
ACOUSTIC_ONSET_HIGH = 0.001

#: Re-arm bar, energy. Held at ``ACOUSTIC_ONSET_HIGH / 4`` — the same hysteresis relationship the
#: original ``0.09``/``0.0225`` pair encoded (a factor of 4 in energy is a factor of 2 in amplitude),
#: and the ratio every row of the measured frontier above was swept at.
ACOUSTIC_ONSET_LOW = ACOUSTIC_ONSET_HIGH / 4.0


class AcousticOnset:
    """Detects energy onsets (a hiss/grunt/beep starting) — one ``onset`` boundary per rising spike.

    Hysteresis crossing on an audio-energy signal: fire when energy crosses ``high`` having been below
    ``low`` (armed), then re-arm only after it falls back under ``low``. Cheap stdlib DSP on the mic
    stream that's already flowing for STT — the genuinely-new sensor, no model.

    UNITS. ``high``/``low`` are MEAN-SQUARE ENERGY in ``[0, 1]`` — the same quantity
    :func:`percept.runtime._vad.frame_energy` computes, NOT RMS amplitude. The runtime's
    ``run(audio_wake={"high": ...})`` knob is the AMPLITUDE form and is squared
    (``_amplitude_threshold_to_energy``) before it reaches this constructor, so ``audio_wake``'s 0.032
    and this constructor's 0.001 are the same bar. See :data:`ACOUSTIC_ONSET_HIGH` for why that number.

    BAND ORDERING. ``low`` must not sit ABOVE ``high``: the re-arm bar is by definition below the fire
    bar, and an inverted pair makes the two swap roles — every sample in ``[high, low)`` both fires
    (``v >= high``) and re-arms (``v < low``), so a dead-steady room tone oscillates the state machine
    at half the sample rate and mints a phantom onset every other sample (measured:
    ``AcousticOnset(high=0.1, low=0.5)`` on a constant ``[0.2] * 5`` emitted 3 onsets). An onset is "a
    sound STARTING", and a level that never changes contains exactly one start whatever the bars are,
    so this is refused in ``__init__`` rather than left to the caller to notice.

    ``low == high`` stays PERMITTED: it is the degenerate no-band configuration, not the defect. No
    sample can be both ``>= high`` and ``< low`` when they are equal, so the oscillation is
    unreachable and a constant tone is still exactly one onset — it simply re-arms the instant energy
    drops below the fire bar, which is a legitimate (if twitchy) ask.

    Reachable from the supported public entrypoint: ``run(audio_gate="acoustic-onset",
    audio_wake={...})`` forwards each value through ``_amplitude_threshold_to_energy``, which validates
    finiteness and range but never the ORDERING — and squaring is monotone, so a caller's inversion is
    preserved rather than created.
    """

    signal = "energy"
    name = "acoustic-onset"

    def __init__(
        self,
        *,
        high: float = ACOUSTIC_ONSET_HIGH,
        low: float = ACOUSTIC_ONSET_LOW,
        refractory_s: float = 0.0,
    ) -> None:
        if low > high:
            raise ValueError(
                f"AcousticOnset needs the re-arm bar at or below the fire bar (low <= high); got "
                f"high={high!r}, low={low!r}. An inverted band makes a constant energy level both "
                "fire and re-arm on the same sample, so a steady tone emits a phantom onset every "
                "other sample. Both are MEAN-SQUARE ENERGY, not RMS amplitude."
            )
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
    """FRAME-backed edge skill (RTMO by default) — an 'openness' waveform → one clean peak per rep.

    Counts fast periodic reps where frame-diff motion conflates each rep's open+close (over-counts) and
    the per-frame VLM aliases on fast motion (under-counts). openness = ankle-spread in TORSO-LENGTHS +
    wrist-elevation, so it tracks every variation (legs-only via the spread, full jacks add the arms).

    THE NORMALIZER. The ankle spread is divided by :func:`percept.harness.pose.body_scale` — the floored
    **Euclidean** shoulder→hip distance. It used to be divided by the shoulder separation measured in X
    ONLY, guarded against an exact zero alone (``or 0.1``), which had two defects on one line: a
    non-upright pose projects both shoulders onto nearly the same x, and a falsiness test substitutes
    the guard for ``0.0`` while ``1e-9`` is truthy and divides straight through. On one sit-up clip the
    x-separation fell under 0.01 on 773 of 1800 frames and the emitted openness reached 19999999.46 —
    which clears ``min_height=0.6`` and confirms as a peak, so the shipped ablation below recorded 101
    boundaries for 9 true reps. ``body_scale`` is orientation-invariant (it does not collapse on
    lying/horizontal poses) and floored rather than falsiness-guarded, so no projection of the shoulders
    can invent a rep.

    UNCALIBRATED KNOBS. That change moves the openness UNIT — ankle-spread per torso-length instead of
    per shoulder-width-in-x — so ``min_height=0.6`` / ``rel_drop=0.4`` / ``refractory_s=0.4`` are the
    values ablated against the OLD unit and are **no longer calibrated**. Re-ablating them needs the
    RepCount-A corpus; nothing here has been retuned to compensate, deliberately. See the
    ``_TYPE_DETECTOR_PRESET`` comment for the measurement and for the honest alternative (the
    ``pose-channels`` → ``perceive_rhythm`` route scores 26.5% MAPE on the same clips).

    The first MODEL-backed detector. RTMO is the default pose provider because it is multi-person and
    deployment-friendly; BlazePose remains available as ``PERCEPT_POSE_PROVIDER=blazepose`` for
    single-person/high-detail use cases. The provider is **lazy** (loaded on first ``push``, never at
    import or registration) and **opt-in** (``percept-vision[pose]`` + provider runtime). The brain
    activates it per-goal via the WatchSpec; a user who never counts reps never loads it.
    """

    signal = "frame"
    name = "pose-openness"

    def __init__(self, *, min_height: float = 0.6, rel_drop: float = 0.4, refractory_s: float = 0.4,
                 provider: Any | None = None, provider_name: str | None = None) -> None:
        self._mh, self._rd, self._ref = min_height, rel_drop, refractory_s
        self._provider = provider
        self._provider_name = provider_name
        self._peak = None
        self._last = 0.0

    def _ensure(self) -> None:
        if self._provider is None:
            from .pose import make_pose_provider

            self._provider = make_pose_provider(self._provider_name)
        if self._peak is None:
            self._peak = MotionPeriodicity(min_height=self._mh, rel_drop=self._rd, refractory_s=self._ref)

    def reset(self) -> None:
        self._last = 0.0
        if self._peak is not None:
            self._peak.reset()

    def push(self, t: float, frame: Any) -> list[Boundary]:
        """frame: one published frame — an RGB uint8 HxWx3 array, or any payload the frame loop
        publishes for it (see :func:`percept.harness.pose.frame_pixels`, which decodes the runtime's
        ``("image/jpeg", bytes)`` pair and RAISES on a payload no eye could see). Emits a 'peak'
        boundary per rep (relabeled from the inner peak detector to this detector's name).

        A provider that emits a skeleton shorter than COCO-17 raises
        :class:`~percept.harness.pose.PoseSkeletonIncomplete` as it builds the detection; that is
        caught here and treated as a DROPPED detection — the last openness carries forward, which is
        this detector's documented behaviour for a frame the eye could not read. It is contained
        rather than propagated because the raise would otherwise escape ``DetectorBank.push`` and end
        the run for every goal in the bank, not only the one that asked for pose."""
        from .pose import PoseSkeletonIncomplete, frame_pixels

        pixels = frame_pixels(frame)
        self._ensure()
        try:
            poses = self._provider.detect(pixels)
        except PoseSkeletonIncomplete:
            poses = []  # an unreadable skeleton is a dropped detection, not a dead run
        if poses:
            from .pose import (
                COCO_LEFT_ANKLE,
                COCO_LEFT_SHOULDER,
                COCO_LEFT_WRIST,
                COCO_RIGHT_ANKLE,
                COCO_RIGHT_SHOULDER,
                COCO_RIGHT_WRIST,
                body_scale,
            )

            lm = max(poses, key=lambda p: p.confidence).keypoints
            spread = abs(lm[COCO_LEFT_ANKLE].x - lm[COCO_RIGHT_ANKLE].x)
            torso = body_scale(lm)  # floored Euclidean shoulder->hip; orientation-invariant
            wrist_up = (
                max(0.0, lm[COCO_LEFT_SHOULDER].y - lm[COCO_LEFT_WRIST].y)
                + max(0.0, lm[COCO_RIGHT_SHOULDER].y - lm[COCO_RIGHT_WRIST].y)
            )
            self._last = spread / torso + wrist_up
        return [Boundary(self.name, b.kind, b.t, b.value) for b in self._peak.push(t, self._last)]


@dataclass(frozen=True)
class PersonMeasure:
    """One detected person's measurements for a frame — the multi-person rhythm-sense producer.

    The eye MEASURES (box + decomposed channels + per-channel keypoint visibility) and stops there; the
    brain's ``EntityGraph`` turns ``box`` into a stable entity id and counts ``channels`` per person.

    ``scale_floored`` is the UNITS half of that contract, and it is per-person on purpose: two people
    in one frame can be at different distances, so "were these channels normalised by a real torso?" is
    not a frame-level fact. See :func:`percept.harness.pose.body_scale_measured` for what the flag
    means and why an absolute amplitude gate must not be applied when it is set."""

    box: tuple[float, float, float, float]  # normalized (x, y, w, h) — for EntityGraph association
    channels: dict[str, float]              # the decomposed pose-channel basis (same as measure())
    pose_conf: float                        # the detection's overall confidence (the Observation score)
    coverage: dict[str, float]              # per-channel keypoint visibility (the phase-source gate signal)
    person_id: str | None = None            # the provider's frame-local index (NOT identity truth)
    scale_floored: bool = False             # the torso normalizer hit its floor -> channels are NOT in
                                            # torso-lengths, so an absolute amplitude gate is invalid


def person_measures(detections: list, *, min_conf: float = 0.3) -> "list[PersonMeasure]":
    """Turn a frame's pose detections into per-person measurements (drops sub-confidence detections).

    Pure (no provider, no I/O): the same transform runs on live detections via
    :meth:`PoseChannels.measure_entities` or on detections decoded elsewhere.

    Carries :attr:`PersonMeasure.scale_floored` per person. This is the load-bearing half of the
    clamp-observability fix: :class:`PoseChannels` records the same fact in ``_last_floored``, but only
    on the SINGLE-subject :meth:`PoseChannels.measure` path — ``measure_entities`` deliberately leaves
    those per-frame health flags alone, so before this the multi-person ``RhythmHero`` path had no
    backstop at all against a floored normalizer."""
    from .pose import bbox_xywh, body_scale_measured, channel_coverage, keypoints_bbox, pose_channels

    out: list[PersonMeasure] = []
    for d in detections:
        if d.confidence < min_conf:
            continue
        box = bbox_xywh(d.bbox) if d.bbox else keypoints_bbox(d.keypoints)
        _scale, floored = body_scale_measured(d.keypoints)
        out.append(PersonMeasure(
            box=box,
            channels=pose_channels(d.keypoints),
            pose_conf=float(d.confidence),
            coverage=channel_coverage(d.keypoints),
            person_id=d.person_id,
            scale_floored=floored,
        ))
    return out


class PoseChannels:
    """FRAME-backed edge skill — the eye that MEASURES and never decides (the rhythm-sense companion).

    Unlike :class:`PoseOpenness`, which fuses landmarks into one jumping-jack-shaped scalar AND decides
    reps, this detector emits a fixed, system-tunable BASIS of decomposed motion channels (wrist-elevation,
    ankle-spread, hip-height, torso-height — see :func:`percept.harness.pose.pose_channels`) and makes NO
    rep decision. The brain's :class:`~percept.rhythm.RhythmTracker` selects the dominant channel and counts
    causally. Channels carry forward on a dropped pose so a missing frame never injects a phantom cycle.

    Two ways to consume it:
      * :meth:`measure` — the per-frame channel vector ``{name: value}`` the rhythm path feeds straight to
        ``core.perceive_rhythm`` (in-process, the benchmark path).
      * :meth:`push` — one ``"sample"`` :class:`Boundary` per channel (carrying ``channel``), so the remote
        edge can ship the same measurements up over the existing BoundaryReport seam.
    """

    signal = "frame"
    name = "pose-channels"

    _MIN_POSE_CONF = 0.3  # ignore low-confidence detections (noise / partial people)

    def __init__(self, *, provider: Any | None = None, provider_name: str | None = None) -> None:
        self._provider = provider
        self._provider_name = provider_name
        self._last: dict[str, float] = {}
        self._anchor: tuple[float, float] | None = None  # last selected person's hip centre (continuity)
        # per-frame health signals (read by the diagnostics trace; never affect the measurement):
        self._last_present: bool = False   # did the most recent measure() see a usable person?
        self._last_kp_conf: float = 0.0    # mean keypoint confidence of that person (degraded-provider signal)
        self._last_floored: bool = False   # did the torso normalizer hit its floor (inflated amplitudes)?

    def _ensure(self) -> None:
        if self._provider is None:
            from .pose import make_pose_provider

            self._provider = make_pose_provider(self._provider_name)

    def reset(self) -> None:
        self._last = {}
        self._anchor = None
        self._last_present = False
        self._last_kp_conf = 0.0
        self._last_floored = False

    def _select(self, poses: list) -> Any | None:
        """Pick ONE person to track. Multi-person frames otherwise make ``max(confidence)`` jump between
        bodies of different scales and inject phantom oscillations. Prefer the detection closest to the
        previously-tracked hip centre (continuity, a mini-entity lock); fall back to most-confident.

        This is the SINGLE-subject path (today's behaviour). The multi-person rhythm hero instead consumes
        :meth:`measure_entities` and assigns identity via the ``EntityGraph`` (brain side)."""
        from .pose import select_nearest_hip

        return select_nearest_hip(poses, self._anchor, self._MIN_POSE_CONF)

    def measure(self, frame: Any) -> dict[str, float]:
        """One published frame -> the decomposed channel vector. Tracks ONE person across frames and
        carries forward on a dropped/low-confidence pose so a gap never injects a phantom oscillation.

        ``frame`` is an RGB uint8 HxWx3 buffer, or any payload the frame loop publishes for one frame
        — ``RhythmDrive.observe`` hands this the runtime's ``("image/jpeg", bytes)`` pair verbatim.
        :func:`percept.harness.pose.frame_pixels` is the one seam that crosses transport to pixels;
        it passes an already-decoded buffer through untouched and RAISES on a payload no eye could
        see, so an unreadable frame can never come back as "measured nothing"."""
        from .pose import body_scale_measured, frame_pixels, hip_centre, pose_channels

        pixels = frame_pixels(frame)
        self._ensure()
        chosen = self._select(self._provider.detect(pixels))
        self._last_present = chosen is not None
        if chosen is not None:
            self._anchor = hip_centre(chosen)
            self._last = pose_channels(chosen.keypoints)
            self._last_kp_conf = sum(k.confidence for k in chosen.keypoints) / len(chosen.keypoints)
            # the clamp reports itself now, so this no longer has to infer "floored" from an epsilon
            # around the floor value (which also fired for a real measurement that happened to land there)
            self._last_floored = body_scale_measured(chosen.keypoints)[1]
        return dict(self._last)

    def measure_entities(self, frame: Any) -> "list[PersonMeasure]":
        """One published frame -> ONE :class:`PersonMeasure` per detected person (the multi-person producer
        the entity-scoped rhythm hero needs). The eye does NOT decide identity — it stops collapsing to a
        single body; the brain's ``EntityGraph`` assigns stable ids and selects the performer.

        Same input contract as :meth:`measure` (decoded pixels, or any payload
        :func:`percept.harness.pose.frame_pixels` can decode to them)."""
        from .pose import frame_pixels

        pixels = frame_pixels(frame)
        self._ensure()
        return person_measures(self._provider.detect(pixels), min_conf=self._MIN_POSE_CONF)

    def push(self, t: float, frame: Any) -> list[Boundary]:
        """Emit one ``"sample"`` boundary per channel (the remote-edge up-channel for the rhythm sense)."""
        return [
            Boundary(self.name, "sample", t, v, channel=ch)
            for ch, v in self.measure(frame).items()
        ]


# ---------------------------------------------------------------------------
# The registry — register an impl + a manifest; the brain composes by name
# ---------------------------------------------------------------------------
#
# PUBLIC, SEMVER-STABLE API (see docs/registration.md). The names below are the supported
# detector-registration surface: ``register`` / ``register_detector`` (add a skill),
# ``make_detector`` (construct one), ``registered`` / ``list_manifests`` / ``manifest_of``
# (introspect), ``unregister`` / ``clear_detectors`` (teardown). Their signatures and the
# ``DetectorManifest`` shape are covered by the contract's semver guarantee: additive changes
# only within a MAJOR. The registry is a process-global singleton keyed by detector name.

_REGISTRY: dict[str, tuple[Callable[..., Any], DetectorManifest]] = {}


def register(name: str, factory: Callable[..., Any], manifest: DetectorManifest) -> None:
    """Register a detector under ``name`` (a deliberate, calibratable event — never per-task code).

    ``factory`` is a zero-arg callable that constructs a fresh, stateful detector instance (the class
    itself, when its ``__init__`` defaults suffice). ``manifest`` is the self-describing
    :class:`DetectorManifest` the brain reads to COMPOSE the detector by name — it never writes the
    detector. Re-registering the same ``name`` replaces the prior entry (last-writer-wins), which is how
    a product overrides a built-in skill with a calibrated variant.

    The lifecycle this opens: ``register`` -> the name becomes selectable into a
    :class:`~percept.adapters.WatchSpec` (via the goal's ``state['detectors']`` and the capability
    handshake) -> :class:`DetectorBank` constructs it per goal -> ``unregister`` / ``clear_detectors``
    tears it down. See :func:`register_detector` for the ergonomic façade that derives a manifest.
    """
    _REGISTRY[name] = (factory, manifest)


def register_detector(
    name: str,
    detector: Callable[..., Any],
    *,
    manifest: DetectorManifest | None = None,
    senses: str = "",
    when_to_use: str = "",
    output: str = "",
    category: str = "",
) -> DetectorManifest:
    """Ergonomic façade over :func:`register` — register a detector, deriving its manifest.

    The minimal call is ``register_detector("my-skill", MySkill)``: the manifest is synthesized from the
    name plus the optional ``senses`` / ``when_to_use`` / ``output`` keyword strings (empty by default).
    Pass a fully-formed ``manifest=`` to supply it directly (its ``name`` is then authoritative). This is
    the recommended public entrypoint for third parties; :func:`register` remains for callers that want to
    pass a positional manifest. Returns the :class:`DetectorManifest` that was registered.

    Backward-compatible: it delegates to :func:`register`, so a detector added either way is identical in
    the registry and selectable by name into a :class:`~percept.adapters.WatchSpec`.
    """
    if manifest is None:
        manifest = DetectorManifest(
            name=name, senses=senses, when_to_use=when_to_use, output=output, category=category
        )
    register(manifest.name if manifest.name else name, detector, manifest)
    return manifest


def make_detector(name: str) -> Any:
    """Construct a fresh detector instance by registry ``name`` (raises ``KeyError`` if unknown)."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown detector {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name][0]()


def manifest_of(name: str) -> DetectorManifest:
    """The :class:`DetectorManifest` for a registered detector (raises ``KeyError`` if unknown)."""
    return _REGISTRY[name][1]


def list_manifests() -> list[DetectorManifest]:
    """All manifests — what the conductor reads to know which detectors it can compose."""
    return [m for _, m in _REGISTRY.values()]


def registered() -> list[str]:
    """Registered detector names (sorted) — the edge's advertised capability vocabulary."""
    return sorted(_REGISTRY)


def detectors_in_category(category: str) -> list[str]:
    """Registered detector names (sorted) whose manifest ``category`` matches — the modality-as-a-class
    discovery: ``detectors_in_category(CATEGORY_AUDIO)`` lists every audio Tier-0 sense, the way
    callers already enumerate motion ones. Empty when nothing in that category is registered."""
    return sorted(n for n, (_, m) in _REGISTRY.items() if m.category == category)


def unregister(name: str) -> bool:
    """Tear down one detector by name. Returns ``True`` if it was registered, ``False`` if absent.

    The teardown half of the lifecycle: after this, ``name`` is no longer in :func:`registered` and the
    selection firewall (:func:`coerce_authored_detectors`) drops it, so no goal can activate it. Existing
    :class:`DetectorBank` instances already constructed keep their own detector objects — teardown affects
    *future* selection, not banks already built.
    """
    return _REGISTRY.pop(name, None) is not None


def clear_detectors() -> None:
    """Empty the detector registry (test isolation / a clean re-registration pass).

    Removes ALL detectors, including the built-ins; call the module-level ``register(...)`` blocks again
    (or re-import) to restore them. Primarily a test/teardown affordance — production wiring registers at
    import and leaves the registry intact for the process lifetime.
    """
    _REGISTRY.clear()


register(
    "motion-periodicity",
    MotionPeriodicity,
    DetectorManifest(
        name="motion-periodicity",
        senses="motion magnitude over time (frame-diff)",
        when_to_use="counting repetitive movements — reps, strokes, paces; any 'count the N-th time' on a periodic motion",
        output="a 'peak' boundary per motion oscillation (≈ one rep), with no per-frame VLM call",
        category=CATEGORY_MOTION,
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
        category=CATEGORY_AUDIO,  # the first-class AUDIO sense — discoverable as a class
    ),
)
register(
    "pose-openness",
    PoseOpenness,
    DetectorManifest(
        name="pose-openness",
        senses="body pose over time (RTMO/COCO keypoints: ankle-spread + wrist-elevation)",
        when_to_use="counting FAST whole-body reps where motion-diff over-counts and the VLM aliases — "
        "jumping jacks, squats, push-ups; any rep with a clean open↔close body shape",
        output="one 'peak' boundary per rep (the open phase of the openness waveform)",
        category=CATEGORY_FRAME,
    ),
)
register(
    "pose-channels",
    PoseChannels,
    DetectorManifest(
        name="pose-channels",
        senses="body pose over time as a DECOMPOSED channel basis (wrist-elevation, ankle-spread, "
        "hip-height, torso-height) — the eye measures, it does not decide reps",
        when_to_use="the rhythm sense — counting any repetitive action (reps, strokes, steps) where the "
        "brain (percept.rhythm) selects the dominant channel and counts cycles causally with no VLM call",
        output="per-frame channel measurements (via measure()) or one 'sample' boundary per channel",
        category=CATEGORY_FRAME,
    ),
)


# ---------------------------------------------------------------------------
# Selection (sense-preset default + authored override) + the firewall
# ---------------------------------------------------------------------------

# goal type -> default detectors. EMPTY BY MEASUREMENT, not by omission: the ablation this map was
# waiting on has been run, and no candidate earned a default. Authored overrides still win.
#
# THE ABLATION. RepCount-A, n=20 clips / 203 annotated reps (the ``repcount`` slice of the stage-validation
# corpus; per-rep cycle bounds are the ground truth). Every clip decoded at its OWN native frame rate and
# driven through the production path — ``detectors_for`` -> :class:`DetectorBank` ->
# ``Percept.perceive_boundary`` -> ``core.edge_counts`` — with each detector constructed by
# :func:`make_detector`, i.e. at its SHIPPED defaults; no detector parameter was tuned for this
# measurement. Pure local DSP, zero provider calls. The motion channel is ``motion_score`` on the
# production 48x36 gate downsample, exactly what ``GatedWatchLoop`` feeds the bank; the pose arm ran the
# shipped default pose provider (RTMO).
#
# ====================  ==========  =====  ========  =======  =========  ==============  ===========
# arm                   count MAPE    MAE  exact/20  +-1 /20  sign test  boundaries/203  rep coverage
# ====================  ==========  =====  ========  =======  =========  ==============  ===========
# control (this map)         100.0%  10.15         0        0          -               0         0.0%
# motion-periodicity          84.8%   8.40         0        2  p = 0.096             241        56.7%
# pose-openness              194.9%  17.40         3        4  p = 0.648             477        73.4%
# ====================  ==========  =====  ========  =======  =========  ==============  ===========
#
# ``sign test`` is the two-sided paired sign test of that arm's per-case absolute percentage error
# against the control's (the control predicts 0 on every clip, so its APE is exactly 1.0 each time).
#
# WHY NOTHING WAS INSTALLED. ``pose-openness`` is ~2x WORSE than answering zero: it fires on any wobble
# of its openness scalar (101 boundaries on a 9-rep sit-up clip, 87 on a 19-rep one), so its 73% rep
# coverage is saturation rather than alignment — 82 of its 477 boundaries fall inside no annotated rep.
# ``motion-periodicity`` is nominally better than the control (84.8% vs 100% MAPE) but the margin is
# noise, not lift: better on 13 of 20 clips and worse on 5 (p = 0.096), never once exact, within +-1 rep
# on 2 of 20, and the error has no direction to correct out (11 over, 9 under; 0 boundaries on a 17-rep
# squat clip, 33 on a 12-rep pull-up clip). Its 241-vs-203 boundary total is over- and under-counts
# cancelling in aggregate, not accuracy. Timing agrees: median boundary offset 0.71 s from the nearest
# annotated rep midpoint, yet only 57% of true reps got any boundary within half a cycle and 46 of 241
# boundaries landed inside no rep at all. Installing either would convert "a count goal fires nothing"
# into "a count goal fires at arbitrary times", which is the worse failure for an nth-distinct
# accumulator (``fire_when: "n == 10"``). ``acoustic-onset`` was not scorable here: 19 of the 20 clips
# carry no audio stream at all.
#
# TWO FINDINGS THE ABLATION SURFACED. The table above prices the detectors AS THEY SHIPPED AT THE TIME
# OF THE RUN, so it is a record of that code, not of this code:
#   * STILL OPEN. :class:`MotionPeriodicity`'s ``min_height=0.02`` is above the ENTIRE motion range of 2
#     of the 20 clips (max 0.013 and 0.017), so it is structurally deaf on them and emits nothing — the
#     same fixed absolute-bar pathology documented at :data:`ACOUSTIC_ONSET_HIGH`. Not fixed here: it is
#     a detector default, and changing it needs its own re-ablation.
#   * FIXED, AND THE ROW ABOVE IS NOW STALE. :class:`PoseOpenness` normalized ankle-spread by SHOULDER
#     WIDTH IN X, guarded only against an EXACT zero (``or 0.1``). On non-upright actions the shoulders
#     align in x — under 0.01 on 773 of 1800 frames of one sit-up clip — and the openness scalar
#     exploded: max 21559 against a median of 2.7 on that clip, which is where its 101 boundaries for 9
#     true reps came from. A jumping jack, the shape the manifest claims, never degenerates (min
#     shoulder width 0.074, openness max 3.35) and was counted exactly. The detector now divides by
#     :func:`percept.harness.pose.body_scale`, the FLOORED Euclidean torso normalizer the newer
#     pose-channel basis already used; this older detector predated it.
#
#     CONSEQUENCE, stated rather than papered over: that changes the openness UNIT (ankle-spread per
#     torso-length, not per shoulder-width-in-x), so ``min_height=0.6`` / ``rel_drop=0.4`` /
#     ``refractory_s=0.4`` were ablated against a unit this detector no longer emits and are NOT
#     calibrated. Nothing was retuned to compensate — inventing knob values without the corpus would
#     only move the error somewhere unmeasured. Re-ablating them needs RepCount-A. The 194.9% MAPE row
#     is the OLD arithmetic's number and must not be quoted for the current one; nor may the fix be
#     read as a recall claim, since it has not been re-measured. Note also that on the SAME 20 clips the
#     ``pose-channels`` -> ``perceive_rhythm`` route scores 26.5% MAPE against this detector's 194.9%,
#     so RETIRING pose-openness may beat re-calibrating it. That is a human decision, not this comment's.
#
# WHAT THE COUNT ROUTE ACTUALLY NEEDS. Two things, neither of them this map:
#   1. This map cannot reach a ``count`` goal at all today. ``detectors_for`` is consulted only by
#      ``compile_watchspec`` (frame routes) and ``RhythmDrive`` (rhythm goals); ``EXTERNAL_BOUNDARY``
#      compiles to no WatchCondition, so no :class:`DetectorBank` is ever built for it in EITHER topology.
#      A preset installed under ``"count"`` would be inert; the key that reaches a bank is ``"watch"``.
#   2. On the same 20 clips the OTHER counting primitive — the ``pose-channels`` eye MEASURING into
#      ``Percept.perceive_rhythm`` (``type="rhythm"``), which confirms reps causally instead of tallying
#      boundaries blindly — scores 26.5% MAPE, MAE 2.95, exact on 5 of 20 and within +-1 on 9 of 20:
#      3x better than the best boundary arm above. Frame-driven counting belongs on that route.
#
# CONSEQUENCE (pinned by ``TestCountPresetAblation``): a ``type="count"`` goal selects NO detector and
# emits NO boundary unless it authors ``state["detectors"]`` itself. The per-action spread says an
# authored choice is the only defensible one anyway — ``pose-openness`` counts jumping jacks (the shape
# its manifest claims) at 8% mean APE and sit-ups at 397%; no single global default survives that.
#
# Cross-check: re-running the pose arm on the BlazePose provider instead of RTMO moves it from 194.9%
# to 245.5% MAPE — worse, not better, and the verdict is unchanged either way. The motion arm is
# provider-independent and reproduced byte-identically across both runs.
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


def partition_by_capability(
    names: Iterable[str], capabilities: Iterable[str] | None
) -> tuple[list[str], list[str]]:
    """Split selected detectors into ``(supported, dropped)`` by the EDGE's advertised capability set —
    the brain-side half of the capability handshake: select only what the connected edge can actually run.

    ``capabilities=None`` means the in-process edge runs the whole registry (topology A) -> all supported.
    A remote edge (topology B: glasses / drone / phone) advertises a SUBSET; detectors it lacks are
    dropped here so the brain never sends a skill the edge can't execute — the caller picks the fallback.
    """
    if capabilities is None:
        return list(names), []
    cap = set(capabilities)
    supported = [n for n in names if n in cap]
    dropped = [n for n in names if n not in cap]
    return supported, dropped


# ---------------------------------------------------------------------------
# DetectorBank — run a goal's selected detectors over a signal stream
# ---------------------------------------------------------------------------


class DetectorBank:
    """Runs a goal's selected detectors over a stream, routing each its signal channel (motion/energy).

    The brain feeds the cheap signals the edge already computes (``motion`` from the frame-diff gate,
    ``energy`` from the audio buffer); each detector reads its own channel and emits boundaries. A
    detector whose channel isn't provided this tick simply gets no input.

    THE CHANNEL VOCABULARY IS CLOSED: exactly ``motion`` / ``energy`` / ``frame``. It used to be two
    recognised names plus an else-branch, so a detector declaring ``signal="thermal"`` — or a typo like
    ``"enrgy"`` — was silently fed frame-diff motion and emitted boundaries the brain then counted as
    coming from the sense it declared. That contradicted the paragraph above, which is the contract
    this class publishes. An unrecognised channel is now the same as an absent one: no input, ever.

    A detector with NO ``signal`` attribute still defaults to ``motion`` — that is the historical
    shape of a bare motion detector, and it is a missing declaration rather than a wrong one.

    NOT the same vocabulary as :attr:`DetectorManifest.category`, which is an explicit EIGHT-name sense
    class (``presence`` / ``change`` / ``semantic_relevance`` / ``audio_event`` among them). Four of
    those categories have no bank channel at all, so a detector can be discoverable as a class the bank
    cannot feed; reconciling the two is a contract change, not this guard."""

    #: The channels the bank can carry. Closed on purpose — see the class docstring.
    CHANNELS = ("motion", "energy", "frame")

    def __init__(self, names: Iterable[str]) -> None:
        self._dets = [make_detector(n) for n in names]

    def reset(self) -> None:
        for d in self._dets:
            d.reset()

    def push(self, t: float, *, motion: float | None = None, energy: float | None = None,
             frame: Any = None) -> list[Boundary]:
        channels: dict[str, Any] = {"motion": motion, "energy": energy, "frame": frame}
        out: list[Boundary] = []
        for d in self._dets:
            chan = getattr(d, "signal", "motion")
            if chan not in channels or channels[chan] is None:
                continue  # a channel this bank does not carry, or does not have this tick
            if chan == "frame":  # frame-backed detector (e.g. pose) — the raw frame, not a scalar
                out.extend(d.push(t, channels[chan]))
            else:
                out.extend(d.push(t, float(channels[chan])))
        return out


__all__ = [
    "CATEGORY_MOTION",
    "CATEGORY_AUDIO",
    "CATEGORY_FRAME",
    "CATEGORY_PRESENCE",
    "CATEGORY_POSE",
    "CATEGORY_AUDIO_EVENT",
    "CATEGORY_CHANGE",
    "CATEGORY_SEMANTIC_RELEVANCE",
    "WakePolicy",
    "WakeDecision",
    "UnsupportedWakePolicyError",
    "coerce_wake_policy",
    "wake_score",
    "should_wake",
    "DetectorManifest",
    "Boundary",
    "MotionPeriodicity",
    "AcousticOnset",
    "ACOUSTIC_ONSET_HIGH",
    "ACOUSTIC_ONSET_LOW",
    "PoseOpenness",
    "PoseChannels",
    "register",
    "register_detector",
    "make_detector",
    "manifest_of",
    "list_manifests",
    "registered",
    "detectors_in_category",
    "unregister",
    "clear_detectors",
    "coerce_authored_detectors",
    "detectors_for",
    "partition_by_capability",
    "DetectorBank",
]
