"""percept-harness — the server-side transport shell + tier-0 salience gate.

Phase 6 lands the **tier-0 reference** here: the WatchSpec↓ / Tier0Signal↑ loop wired
end-to-end against the cognition core. The brain compiles active goals *down* to a
:class:`~percept.adapters.WatchSpec`; a coarse motion gate consumes that spec frame-by-frame,
reports a :class:`~percept.adapters.Tier0Signal` *up*, and the expensive
``Percept.perceive_judged`` cloud call runs **only when salience clears threshold**. That single
wired loop is the proof point.

The harness is the transport shell around the core, not a second brain — it never re-judges; it
only decides *when* to spend a vision call. The real on-device gate is ``percept-edge`` (JS/WASM);
this is the Python server-side reference that speaks the same frozen wire contract.
"""

from __future__ import annotations

from .concerns import (
    clear_concerns,
    compile_registered_watchspec,
    concern_to_condition,
    detectors_for_concern,
    get_concern,
    register_concern,
    registered_concern_ids,
    registered_concerns,
    unregister_concern,
)
from .detectors import (
    CATEGORY_AUDIO,
    CATEGORY_FRAME,
    CATEGORY_MOTION,
    Boundary,
    DetectorBank,
    DetectorManifest,
    clear_detectors,
    coerce_authored_detectors,
    detectors_for,
    detectors_in_category,
    list_manifests,
    make_detector,
    register,
    register_detector,
    registered,
    unregister,
)
from .pose import (
    POSE_CHANNEL_NAMES,
    BlazePoseProvider,
    PoseDetection,
    PoseKeypoint,
    RTMOPoseProvider,
    make_pose_provider,
    pose_channels,
)
from .tier0 import (
    Frame,
    GateDecision,
    GatedWatchLoop,
    LoopReport,
    MotionGate,
    compile_watchspec,
    gray_frame,
    motion_score,
    run_gated_watch,
)

__all__ = [
    "Frame",
    "gray_frame",
    "motion_score",
    "compile_watchspec",
    "MotionGate",
    "GateDecision",
    "GatedWatchLoop",
    "LoopReport",
    "run_gated_watch",
    # edge detector registry (the skills library)
    "Boundary",
    "DetectorBank",
    "DetectorManifest",
    "CATEGORY_MOTION",
    "CATEGORY_AUDIO",
    "CATEGORY_FRAME",
    "register",
    "register_detector",
    "make_detector",
    "registered",
    "detectors_in_category",
    "list_manifests",
    "detectors_for",
    "coerce_authored_detectors",
    "unregister",
    "clear_detectors",
    # pose provider seam
    "PoseKeypoint",
    "PoseDetection",
    "RTMOPoseProvider",
    "BlazePoseProvider",
    "make_pose_provider",
    "pose_channels",
    "POSE_CHANNEL_NAMES",
    # concern registry (the standing-concern lifecycle)
    "register_concern",
    "registered_concerns",
    "registered_concern_ids",
    "get_concern",
    "unregister_concern",
    "clear_concerns",
    "detectors_for_concern",
    "concern_to_condition",
    "compile_registered_watchspec",
]
