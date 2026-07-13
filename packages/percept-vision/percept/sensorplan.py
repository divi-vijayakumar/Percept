"""SensorPlan compiler — Build 5: compile a plain-language goal into a typed, validated sensing recipe.

The SDK already separates *what to watch for* (the behavior program) from *how to look* (perception) and
*how densely* (the fovea's SensePolicy). A goal's full sensing recipe is currently spread across those
places and the new individuation knobs (Builds 2–4). The **SensorPlan** gathers them into ONE typed,
explainable object — and gives the conductor/LLM a single, safe surface to author them.

Two halves, mirroring VISPROG/ViperGPT (an LLM composes registered modules; the runtime owns execution):

  * :func:`sensor_plan_for` — the **deterministic compiler**: read the goal's state (the same source
    :func:`~percept.behavior.program_of` reads, so it is byte-for-byte consistent) plus the registered
    primitive manifests, and produce the SensorPlan. Descriptive: it reflects exactly what the goal will
    do (including any opt-in ``smooth_s`` / ``canonical_select`` it authored), and can be stored in
    ``goal.state['sensor_plan']`` for explainability. It changes no behavior.
  * :func:`coerce_authored_sensor_plan` — the **firewall**: turn an UNTRUSTED LLM-authored plan into safe
    ``goal.state`` keys. The LLM may only SELECT registered primitives and set BOUNDED knobs; unknown
    primitive names, unbounded fps/tokens, and anything non-scalar are dropped. Crucially it **delegates to
    the existing validated coercers** (:func:`~percept.behavior.coerce_authored_sense`,
    :func:`~percept.behavior.coerce_authored_rhythm`), so an authored plan flows to the EXISTING consumers
    (the SamplingController, the RhythmTracker) through the existing keys — no new firing path, gate.py
    untouched.

The registry of self-describing primitives (each with ``when_to_use``) is the scalable alternative to a
goal-type enum: the conductor composes by reading ``when_to_use``, the runtime validates against the
registry. (The legacy ``_TYPE_DETECTOR_PRESET`` / ``_SENSE_PERCEPTION_PRESET`` maps remain the empty
fallback.) Pure stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .behavior import (
    _MAX_AUDIO_S,
    _MAX_STRIP,
    _MAX_WINDOW_S,
    _VALID_MODALITY,
    _VALID_ROI,
    _VALID_SAMPLING,
    _VALID_TEMPORAL,
    _num,
    coerce_authored_rhythm,
    coerce_authored_sense,
    program_of,
)


@dataclass(frozen=True)
class SensorPrimitive:
    """A self-describing sensing primitive the conductor can compose. ``when_to_use`` is the natural-language
    cue the LLM reads to select it (the same idea as a DetectorManifest); ``kind`` separates the sense
    primitives (what query) from the perception primitives (how the eye looks)."""

    name: str
    kind: str  # "sense" | "perception"
    when_to_use: str
    measurements: tuple[str, ...] = ()
    event_model: dict = field(default_factory=dict)
    confirmation: dict = field(default_factory=dict)


# The open registry of self-describing primitives — composed by name, validated against this map. Senses
# mirror percept.behavior.Sense; perception primitives mirror PerceptionSpec.temporal. New primitives
# (sprite, Build 6; a learned counter, Build 7) register here once their lift is proven — no enum edits.
PRIMITIVES: dict[str, SensorPrimitive] = {
    "rhythm": SensorPrimitive(
        "rhythm", "sense",
        "count a repeating cycle — reps, strokes, paces; any 'count the Nth time' on a periodic motion",
        measurements=("pose-channels",),
        event_model={"kind": "periodic", "min_support": 2},
        confirmation={"vlm": "sparse", "when": "candidate_boundary"},  # pose is free; VLM only to confirm
    ),
    "judge": SensorPrimitive(
        "judge", "sense",
        "watch for a plain-language condition to become true — the rising-edge fire/ask; for a BRIEF "
        "sub-second window ('tell me when X briefly happens') the cadence must be SENSITIVE — sample above "
        "the window's Nyquist so a quiet-but-in-window frame is still judged, not suppressed by the motion gate",
        measurements=("vlm",), event_model={"kind": "edge"}, confirmation={"vlm": "each_escalation"},
    ),
    "transition": SensorPrimitive(
        "transition", "sense", "a change from one state to another — A then B",
        measurements=("vlm",), event_model={"kind": "transition"}, confirmation={"vlm": "each_escalation"},
    ),
    "episode": SensorPrimitive(
        "episode", "sense", "count DISTINCT occurrences of an event over time (the cup problem)",
        measurements=("vlm",), event_model={"kind": "distinct"}, confirmation={"vlm": "each_escalation"},
    ),
    "absence": SensorPrimitive(
        "absence", "sense", "fire when something expected has NOT happened by a deadline",
        measurements=("vlm",), event_model={"kind": "absence"}, confirmation={"vlm": "on_deadline"},
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

_VLM_CONFIRM = {"none", "sparse", "each_escalation", "on_deadline"}
_CONFIRM_WHEN = {"candidate_boundary", "each_escalation", "on_deadline", "never"}

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
    Authored values always win (``base.get(...) or ...``), so this only FILLS the unset knobs. Returns the
    cadence dict unchanged when the prior is absent/non-positive — non-brief judge goals are untouched."""
    bw = _num(brief_window_s)
    if bw is None or bw <= 0:
        return base
    nyquist_fps = _BRIEF_NYQUIST_SAMPLES / bw  # e.g. a 1.0s window -> >=2 fps; a 0.5s window -> >=4 fps
    out = dict(base)
    out["base_fps"] = base.get("base_fps") or nyquist_fps
    out["max_fps"] = base.get("max_fps") or out["base_fps"] * _BRIEF_MAX_HEADROOM
    if base.get("escalate_threshold") is None:
        out["escalate_threshold"] = _BRIEF_ESCALATE_THRESHOLD
    out["brief_window_s"] = bw  # surfaced for explainability + so the adapter can read the prior back
    return out


@dataclass(frozen=True)
class SensorPlan:
    """A goal's full sensing recipe, gathered and typed: which primitives, modality, fovea cadence, eye
    perception, measurement channels, individuation/event model, and VLM-confirmation policy."""

    primitives: tuple[str, ...]
    modality: str
    cadence: dict
    perception: dict
    measurements: tuple[str, ...]
    event_model: dict
    confirmation: dict

    def as_dict(self) -> dict:
        return {
            "primitives": list(self.primitives),
            "modality": self.modality,
            "cadence": dict(self.cadence),
            "perception": dict(self.perception),
            "measurements": list(self.measurements),
            "event_model": dict(self.event_model),
            "confirmation": dict(self.confirmation),
        }


def sensor_plan_for(goal) -> SensorPlan:
    """Compile a goal into its :class:`SensorPlan` deterministically. Reads :func:`program_of` (so it is
    byte-for-byte consistent with what the goal will actually run) and the registered primitive manifests,
    and surfaces any authored individuation knobs (``smooth_s``, ``canonical_select``) in ``event_model``."""
    prog = program_of(goal)
    sense = prog.sense
    prim = PRIMITIVES[sense]
    state = getattr(goal, "state", {}) or {}
    sense_cfg = state.get("sense") if isinstance(state.get("sense"), dict) else {}
    rhythm = state.get("rhythm") if isinstance(state.get("rhythm"), dict) else {}
    p = prog.perception

    cadence = {
        "gate_on": "motion",
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
        measurements=prim.measurements,
        event_model=event_model,
        confirmation=dict(prim.confirmation),
    )


def attach_sensor_plan(goal) -> SensorPlan:
    """Compile and store the plan in ``goal.state['sensor_plan']`` (explainability). Returns the plan."""
    plan = sensor_plan_for(goal)
    state = getattr(goal, "state", None)
    if isinstance(state, dict):
        state["sensor_plan"] = plan.as_dict()
    return plan


def _coerce_perception(perc: dict) -> dict:
    """Validate an authored perception block to safe, clamped ``goal.state['perception']`` fields."""
    out: dict = {}
    if perc.get("modality") in _VALID_MODALITY:
        out["modality"] = perc["modality"]
    if perc.get("temporal") in _VALID_TEMPORAL:
        out["temporal"] = perc["temporal"]
    nf = _num(perc.get("n_frames"))
    if nf is not None and nf >= 1:
        out["n_frames"] = min(_MAX_STRIP, int(nf))  # token-bomb ceiling
    if perc.get("roi") in _VALID_ROI:
        out["roi"] = perc["roi"]
    if perc.get("sampling") in _VALID_SAMPLING:
        out["sampling"] = perc["sampling"]
    w = _num(perc.get("window_s"))
    if w is not None and w > 0:
        out["window_s"] = min(_MAX_WINDOW_S, w)
    a = _num(perc.get("audio_s"))
    if a is not None and a > 0:
        out["audio_s"] = min(_MAX_AUDIO_S, a)
    return out


def coerce_authored_sensor_plan(raw) -> dict:
    """Validate an UNTRUSTED authored SensorPlan into safe ``goal.state`` keys (the firewall).

    The LLM may only SELECT registered primitives and set BOUNDED knobs. Returns a dict of standard state
    keys (``sense`` / ``rhythm`` / ``perception``) — so the plan takes effect through the EXISTING
    consumers and firewalls — plus a recorded ``sensor_plan`` summary. Unknown primitives, unbounded
    fps/tokens, out-of-vocab enums, and non-scalar values are dropped (only typed scalars/enums are ever
    read, so executable strings can never enter state). Malformed input -> ``{}``."""
    if not isinstance(raw, dict):
        return {}
    out: dict = {}
    plan: dict = {}

    prims = [p for p in raw.get("primitives", []) if isinstance(p, str) and p in PRIMITIVES]
    if prims:
        plan["primitives"] = prims

    cadence = raw.get("cadence")
    if isinstance(cadence, dict):  # delegate to the validated fovea-cadence firewall (clamps fps, envelope)
        sense_keys = coerce_authored_sense(cadence).get("sense")
        if sense_keys:
            out["sense"] = sense_keys
            plan["cadence"] = sense_keys

    em = raw.get("event_model")
    if isinstance(em, dict):  # delegate the individuation knobs to the validated rhythm firewall
        rhythm_in: dict = {}
        if isinstance(em.get("channel"), str):
            rhythm_in["channel"] = em["channel"]
        if _num(em.get("period_hint_s")) is not None:
            rhythm_in["cadence_s"] = em["period_hint_s"]
        if _num(em.get("smooth_s")) is not None:
            rhythm_in["smooth_s"] = em["smooth_s"]
        if em.get("canonical_select") is True:
            rhythm_in["canonical_select"] = True
        rk = coerce_authored_rhythm(rhythm_in).get("rhythm")
        if rk:
            out["rhythm"] = rk
            plan["event_model"] = rk

    perc = raw.get("perception")
    if isinstance(perc, dict):
        vp = _coerce_perception(perc)
        if vp:
            out["perception"] = vp
            plan["perception"] = vp

    conf = raw.get("confirmation")
    if isinstance(conf, dict):
        c: dict = {}
        if conf.get("vlm") in _VLM_CONFIRM:
            c["vlm"] = conf["vlm"]
        if conf.get("when") in _CONFIRM_WHEN:
            c["when"] = conf["when"]
        if c:
            plan["confirmation"] = c

    if plan:
        out["sensor_plan"] = plan
    return out


__all__ = [
    "SensorPrimitive",
    "PRIMITIVES",
    "SensorPlan",
    "sensor_plan_for",
    "attach_sensor_plan",
    "coerce_authored_sensor_plan",
]
