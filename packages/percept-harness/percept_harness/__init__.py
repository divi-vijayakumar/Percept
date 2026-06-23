"""percept-harness — the server-side transport shell + tier-0 salience gate.

Phase 6 lands the **tier-0 reference** here: the WatchSpec↓ / Tier0Signal↑ loop wired
end-to-end against the cognition core. The brain compiles active goals *down* to a
:class:`~percept.adapters.WatchSpec`; a coarse motion gate consumes that spec frame-by-frame,
reports a :class:`~percept.adapters.Tier0Signal` *up*, and the expensive
``Percept.perceive_judged`` cloud call runs **only when salience clears threshold** (spec §6.3,
impl plan §8). That single wired loop is the proof point.

The harness is the transport shell around the core, not a second brain — it never re-judges; it
only decides *when* to spend a vision call. The real on-device gate is ``percept-edge`` (JS/WASM);
this is the Python server-side reference that speaks the same frozen wire contract.
"""

from __future__ import annotations

from .detectors import (
    Boundary,
    DetectorBank,
    DetectorManifest,
    coerce_authored_detectors,
    detectors_for,
    list_manifests,
    make_detector,
    register,
    registered,
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
    "register",
    "make_detector",
    "registered",
    "list_manifests",
    "detectors_for",
    "coerce_authored_detectors",
]
