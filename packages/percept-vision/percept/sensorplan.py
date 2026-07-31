"""SensorPlan compiler — Build 5: compile a plain-language goal into a typed, validated sensing recipe.

The SDK already separates *what to watch for* (the behavior program) from *how to look* (perception) and
*how densely* (the fovea's SensePolicy). A goal's full sensing recipe is currently spread across those
places and the new individuation knobs (Builds 2–4). The **SensorPlan** gathers them into ONE typed,
explainable object.

:func:`sensor_plan_for` is the **deterministic compiler**: read the goal's state (the same source
:func:`~percept.behavior.program_of` reads, so it is byte-for-byte consistent) plus the registered
primitive manifests, and produce the SensorPlan. Descriptive: it reflects exactly what the goal will
do (including any opt-in ``smooth_s`` / ``canonical_select`` it authored), and can be stored in
``goal.state['sensor_plan']`` for explainability. It changes no behavior. This is the ONLY compiler this
module ships: an authored-plan firewall (turning an untrusted LLM-composed plan into safe ``goal.state``
keys) is a real, separate surface with its own consumer, not something to speculatively carry here empty.

The registry of self-describing primitives (each with ``when_to_use``) is the scalable alternative to a
goal-type enum: a future authoring surface would compose by reading ``when_to_use``; today the registry's
only reader is :func:`sensor_plan_for` itself. (The legacy ``_TYPE_DETECTOR_PRESET`` /
``_SENSE_PERCEPTION_PRESET`` maps remain the empty fallback.) Pure stdlib.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .behavior import _num, coerce_authored_cadence, program_of


@dataclass(frozen=True)
class SensorPrimitive:
    """A self-describing sensing primitive :func:`sensor_plan_for` composes by name. ``when_to_use`` is
    the natural-language cue a future authoring surface would read to select it (the same idea as a
    DetectorManifest); ``kind`` separates the sense primitives (what query) from the perception
    primitives (how the eye looks)."""

    name: str
    kind: str  # "sense" | "perception"
    when_to_use: str
    event_model: dict = field(default_factory=dict)


# The open registry of self-describing primitives — composed by name, validated against this map. Senses
# mirror percept.behavior.Sense; perception primitives mirror PerceptionSpec.temporal. New primitives
# (sprite, Build 6; a learned counter, Build 7) register here once their lift is proven — no enum edits.
PRIMITIVES: dict[str, SensorPrimitive] = {
    "rhythm": SensorPrimitive(
        "rhythm", "sense",
        "count a repeating cycle — reps, strokes, paces; any 'count the Nth time' on a periodic motion",
        event_model={"kind": "periodic", "min_support": 2},
    ),
    "judge": SensorPrimitive(
        "judge", "sense",
        "watch for a plain-language condition to become true — the rising-edge fire/ask; for a BRIEF "
        "sub-second window ('tell me when X briefly happens') the cadence must be SENSITIVE — sample above "
        "the window's Nyquist so a quiet-but-in-window frame is still judged, not suppressed by the motion gate",
        event_model={"kind": "edge"},
    ),
    "transition": SensorPrimitive(
        "transition", "sense", "a change from one state to another — A then B",
        event_model={"kind": "transition"},
    ),
    "episode": SensorPrimitive(
        "episode", "sense", "count DISTINCT occurrences of an event over time (the cup problem)",
        event_model={"kind": "distinct"},
    ),
    "absence": SensorPrimitive(
        "absence", "sense", "fire when something expected has NOT happened by a deadline",
        event_model={"kind": "absence"},
    ),
    "single": SensorPrimitive("single", "perception", "one frame — cheapest; best for boundary timing"),
    "strip": SensorPrimitive("strip", "perception", "N ordered frames — when ordering/detail/small motion matters"),
    "window": SensorPrimitive("window", "perception", "N frames over a time window — temporal context"),
    "sprite": SensorPrimitive(
        "sprite", "perception",
        "N frames as ONE montage image — 'what/steps' synthesis at ~single-frame token cost (VLM-only; "
        "NOT for exact boundary timing — a grid trades temporal resolution for context)",
    ),
}

# A "tell me when X BRIEFLY happens" prior (``goal.state['sense']['brief_window_s']``): the expected
# duration of the action's rising-edge window. A brief sub-second window can be missed entirely by a
# motion-gated, coarse cadence — the window opens and closes between two coarse judged frames (the
# pt-action coverage miss). To cover it we must sample/judge above the window's NYQUIST: at least two
# in-window frames, so a quiet-but-in-window frame is still judged even if the motion gate would skip it.
# This is driven by the goal's prior, not a per-activity constant — any brief-action judge goal benefits.
_BRIEF_NYQUIST_SAMPLES = 2.0  # >=2 judged frames inside the window (Nyquist for the window's edge)
_BRIEF_MAX_HEADROOM = 3.0  # escalated ceiling = headroom x the sensitive floor (still bounded by the firewall)
_BRIEF_ESCALATE_THRESHOLD = 0.05  # a low motion floor: admit subtle in-window motion the default 0.15 suppresses


def _sensitive_cadence(brief_window_s, base: dict) -> dict:
    """Derive a SENSITIVE judge cadence from a brief-window prior (the coverage fix), without overriding
    any cadence the goal already authored. Raises ``base_fps`` to at least the window's Nyquist
    (``_BRIEF_NYQUIST_SAMPLES / brief_window_s``) so a sub-second window gets >=2 judged frames, lifts the
    escalated ceiling to keep headroom, and lowers ``escalate_threshold`` to admit subtle in-window motion.

    A PURE FILL: every knob is tested with ``is None``, so an authored value always wins and only an
    UNSET knob is derived. It deliberately validates nothing — an authored ``base_fps`` of 0 flows
    through and is refused, loudly, by :func:`percept.sampling.validate_cadence`, the one place that
    owns the question. (It used to fill with a falsy ``or``, which made an authored 0 indistinguishable
    from an absent key while ``escalate_threshold`` three lines down already used ``is None``.)

    Returns the cadence dict unchanged when the prior is absent, non-numeric, non-positive or NOT FINITE
    — non-brief judge goals are untouched. NaN is named explicitly because ``nan <= 0`` is False: a NaN
    prior otherwise derived a NaN Nyquist floor, which is truthy, so it walked past every downstream
    ``if not base`` drop and blinded the concern after a single frame."""
    bw = _num(brief_window_s)
    if bw is None or not math.isfinite(bw) or bw <= 0:
        return base
    nyquist_fps = _BRIEF_NYQUIST_SAMPLES / bw  # e.g. a 1.0s window -> >=2 fps; a 0.5s window -> >=4 fps
    out = dict(base)
    authored_base = base.get("base_fps")
    out["base_fps"] = nyquist_fps if authored_base is None else authored_base
    authored_max = base.get("max_fps")
    out["max_fps"] = (
        out["base_fps"] * _BRIEF_MAX_HEADROOM if authored_max is None else authored_max
    )
    if base.get("escalate_threshold") is None:
        out["escalate_threshold"] = _BRIEF_ESCALATE_THRESHOLD
    out["brief_window_s"] = bw  # surfaced for explainability + so the adapter can read the prior back
    return out


@dataclass(frozen=True)
class SensorPlan:
    """A goal's full sensing recipe, gathered and typed: which primitives, modality, fovea cadence, eye
    perception, and individuation/event model."""

    primitives: tuple[str, ...]
    modality: str
    cadence: dict
    perception: dict
    event_model: dict

    def as_dict(self) -> dict:
        return {
            "primitives": list(self.primitives),
            "modality": self.modality,
            "cadence": dict(self.cadence),
            "perception": dict(self.perception),
            "event_model": dict(self.event_model),
        }


def sensor_plan_for(goal) -> SensorPlan:
    """Compile a goal into its :class:`SensorPlan` deterministically. Reads :func:`program_of` (so it is
    byte-for-byte consistent with what the goal will actually run) and the registered primitive manifests,
    and surfaces any authored individuation knobs (``smooth_s``, ``canonical_select``) in ``event_model``.

    ``cadence['gate_on']`` runs the SAME firewall the tier-0 gate runs
    (:func:`~percept.behavior.coerce_authored_cadence`), so the plan reports the gate the goal will
    actually be armed by. The interval that firewall also decides (``min_judge_interval``) is
    deliberately NOT surfaced here — ``SensorPlan.cadence`` describes the fovea's processing envelope
    (fps / escalation), and adding the judge throttle would change the plan's pinned shape without a
    reader. That is a known GAP, not a claim: the plan under-describes the throttle; it no longer
    misdescribes the gate."""
    prog = program_of(goal)
    sense = prog.sense
    prim = PRIMITIVES[sense]
    state = getattr(goal, "state", {}) or {}
    sense_cfg = state.get("sense") if isinstance(state.get("sense"), dict) else {}
    rhythm = state.get("rhythm") if isinstance(state.get("rhythm"), dict) else {}
    p = prog.perception

    # ``gate_on`` is DERIVED, never a literal: it comes from the same firewall the tier-0 gate runs
    # (``coerce_authored_cadence``, byte-identical to ``harness.tier0._coerce_cadence``), so a goal
    # authoring ``cadence={'gate_on': 'fixed', 'min_judge_interval': 2.0}`` gets a plan that says
    # "fixed" — and one authoring an out-of-vocab or period-less "fixed" gets the same "motion"
    # fallback the gate will actually apply. It used to be the literal "motion", which made the
    # module's byte-for-byte promise false for exactly this field (BUG-A-04-03).
    _, gate_on = coerce_authored_cadence(state.get("cadence"))
    cadence = {
        "gate_on": gate_on,
        "base_fps": sense_cfg.get("base_fps"),
        "max_fps": sense_cfg.get("max_fps"),
        "escalate_threshold": sense_cfg.get("escalate_threshold"),
    }
    # The judge (rising-edge) sense compiles a SENSITIVE cadence when the goal carries a brief-window
    # prior — driven by the prior, not a per-activity branch (the registry's ``judge.when_to_use`` is the
    # cue). A brief sub-second window needs above-Nyquist coverage so a quiet-but-in-window frame is still
    # judged; without the prior this is a no-op and the cadence is byte-for-byte the default judge cadence.
    if prim.event_model.get("kind") == "edge":
        cadence = _sensitive_cadence(sense_cfg.get("brief_window_s"), cadence)
    perception = {
        "primitive": p.temporal, "modality": p.modality, "temporal": p.temporal,
        "n_frames": p.n_frames, "roi": p.roi, "sampling": p.sampling,
        "window_s": p.window_s, "audio_s": p.audio_s,
    }
    event_model = dict(prim.event_model)
    if sense == "rhythm":
        event_model.update(
            period_hint_s=rhythm.get("cadence_s"),
            channel=rhythm.get("channel"),
            smooth_s=rhythm.get("smooth_s", 0.0),  # Build 2 knob (opt-in; surfaced for explainability)
            canonical_select=bool(rhythm.get("canonical_select", False)),  # Build 3 knob
        )
    elif sense == "absence":
        # Explainability: surface the AUTHORED duration (timeline-independent) — the absolute ``deadline_at``
        # only exists after the goal is armed, and the plan compiler runs on the unarmed, authored goal.
        event_model["deadline_after_s"] = state.get("deadline_after_s")
    elif prog.cadence_s is not None:
        event_model["cadence_s"] = prog.cadence_s

    return SensorPlan(
        primitives=(sense, p.temporal),
        modality=p.modality,
        cadence=cadence,
        perception=perception,
        event_model=event_model,
    )


__all__ = [
    "SensorPrimitive",
    "PRIMITIVES",
    "SensorPlan",
    "sensor_plan_for",
]
