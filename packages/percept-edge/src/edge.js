// src/edge.js
//
// The on-device reactive edge. This is the single wired loop that is the
// proof-point of Phase 6:
//
//   down:  applyWatchSpec(spec)   <- the brain compiles active goals to a
//                                    WatchSpec (anchors + thresholds)
//   up:    a Tier0Signal is published whenever a frame is processed and the
//          gate clears threshold (salience worth a cloud call)
//
// The wired, tested path is motion-only. CLIP (transformers.js/ONNX),
// pose/hands (MediaPipe), and LiveKit publishing are documented adapter slots
// (see setStructuralAdapter / setSemanticAdapter / setPublisher) — they are
// not required to run and not exercised by the test suite.

import { makeTier0Signal, parseWatchSpec } from "./contract.js";
import { motionScore } from "./motion.js";

export class Edge {
  /**
   * @param {object} [opts]
   * @param {(signal:object)=>void} [opts.onSignal]  up-channel sink for Tier0Signal
   * @param {number} [opts.defaultThreshold]  salience floor when a spec has no
   *        conditions (so motion-only still gates). Default 0.05.
   * @param {()=>number} [opts.clock]  returns seconds; for deterministic ts.
   */
  constructor({ onSignal = null, defaultThreshold = 0.05, clock = null } = {}) {
    this._listeners = new Set();
    if (typeof onSignal === "function") this._listeners.add(onSignal);

    this.defaultThreshold = defaultThreshold;
    this._clock = clock || (() => Date.now() / 1000);

    this._spec = null; // current WatchSpec (down channel)
    this._prevFrame = null; // motion history (1-frame buffer)

    // Adapter slots — optional, default to inert (return 0 salience).
    this._structuralAdapter = null; // e.g. pose/hands (MediaPipe)
    this._semanticAdapter = null; // e.g. CLIP zero-shot gate
    this._publisher = null; // e.g. LiveKit media publisher
  }

  // --- up-channel subscription ------------------------------------------

  /** Subscribe a Tier0Signal sink. Returns an unsubscribe fn. */
  onSignal(fn) {
    this._listeners.add(fn);
    return () => this._listeners.delete(fn);
  }

  // --- down channel: receive a WatchSpec --------------------------------

  /**
   * Apply a WatchSpec from the brain (down channel). Accepts a JSON string or
   * an object; validates it through the frozen contract. Returns the
   * normalized spec now in effect.
   */
  applyWatchSpec(specOrJson) {
    this._spec = parseWatchSpec(specOrJson);
    return this._spec;
  }

  /** The WatchSpec currently in effect, or null. */
  get watchSpec() {
    return this._spec;
  }

  /**
   * The salience threshold the gate must clear. We take the *minimum* threshold
   * across active conditions (any goal clearing its bar escalates); with no
   * spec/conditions we fall back to defaultThreshold so motion-only still gates.
   */
  get activeThreshold() {
    const conds = this._spec?.conditions ?? [];
    if (conds.length === 0) return this.defaultThreshold;
    let min = Infinity;
    for (const c of conds) min = Math.min(min, c.threshold);
    return Number.isFinite(min) ? min : this.defaultThreshold;
  }

  // --- adapter slots (documented, optional, not tested) -----------------

  /** Plug a structural salience adapter: (frame, prevFrame) => number[0..1]. */
  setStructuralAdapter(fn) {
    this._structuralAdapter = fn;
    return this;
  }

  /** Plug a semantic salience adapter: (frame, watchSpec) => number[0..1]. */
  setSemanticAdapter(fn) {
    this._semanticAdapter = fn;
    return this;
  }

  /** Plug a media publisher (e.g. LiveKit): (frame, signal) => void. */
  setPublisher(fn) {
    this._publisher = fn;
    return this;
  }

  // --- the wired loop ----------------------------------------------------

  /**
   * Process one frame. Computes motion (always), plus structural/semantic
   * salience if adapters are wired. If overall salience clears the active
   * threshold, builds a Tier0Signal, publishes the (gated) frame via the
   * publisher slot if any, emits the signal up, and returns it. Otherwise
   * returns null (no escalation — no cloud call).
   *
   * @param {{width:number,height:number,data:Uint8Array}} frame grayscale frame
   * @returns {object|null} the emitted Tier0Signal, or null if gated out
   */
  processFrame(frame) {
    const motion = motionScore(this._prevFrame, frame);
    this._prevFrame = frame;

    let structural = 0;
    if (this._structuralAdapter) {
      structural = clamp01(this._structuralAdapter(frame, this._prevFrame));
    }

    let semantic = 0;
    if (this._semanticAdapter) {
      semantic = clamp01(this._semanticAdapter(frame, this._spec));
    }

    // Overall salience = the strongest channel. Motion alone can escalate.
    const salience = Math.max(motion, structural, semantic);

    if (salience < this.activeThreshold) {
      return null; // gated: still scene -> no Tier0Signal, no cloud call
    }

    const signal = makeTier0Signal({
      motion,
      structural,
      semantic,
      ts: this._clock(),
    });

    if (this._publisher) this._publisher(frame, signal);
    this._emit(signal);
    return signal;
  }

  _emit(signal) {
    for (const fn of this._listeners) fn(signal);
  }
}

function clamp01(v) {
  if (typeof v !== "number" || !Number.isFinite(v)) return 0;
  if (v < 0) return 0;
  if (v > 1) return 1;
  return v;
}
