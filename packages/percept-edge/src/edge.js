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
import { SkillBank, registered } from "./skills.js";

export class Edge {
  /**
   * @param {object} [opts]
   * @param {(signal:object)=>void} [opts.onSignal]  up-channel sink for Tier0Signal
   * @param {number} [opts.defaultThreshold]  salience floor when a spec has no
   *        conditions (so motion-only still gates). Default 0.05.
   * @param {()=>number} [opts.clock]  returns seconds; for deterministic ts.
   */
  constructor({ onSignal = null, onBoundary = null, defaultThreshold = 0.05, clock = null } = {}) {
    this._listeners = new Set();
    if (typeof onSignal === "function") this._listeners.add(onSignal);
    this._boundaryListeners = new Set();
    if (typeof onBoundary === "function") this._boundaryListeners.add(onBoundary);

    this.defaultThreshold = defaultThreshold;
    this._clock = clock || (() => Date.now() / 1000);

    this._spec = null; // current WatchSpec (down channel)
    this._prevFrame = null; // motion history (1-frame buffer)
    this._banks = new Map(); // goal_id -> SkillBank (the skills this goal selected, if registered)
    this._unsupported = []; // {goal_id, detector} the brain selected but this edge hasn't registered

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

  /**
   * Subscribe a Boundary sink — the per-event up-channel (Option 2). Receives, per tick that produced
   * any, an array of `{ goal_id, boundary }` (mirrors the harness's (goal_id, Boundary) pairs that feed
   * `core.perceive_boundary(goal_id, t)`). Returns an unsubscribe fn.
   */
  onBoundary(fn) {
    this._boundaryListeners.add(fn);
    return () => this._boundaryListeners.delete(fn);
  }

  /** The skills this edge can run — its advertised capability set (the basis for the handshake). */
  capabilities() {
    return registered();
  }

  /** Detectors the brain selected that this edge has NOT registered (dropped, not run). */
  get unsupportedDetectors() {
    return this._unsupported.slice();
  }

  // --- down channel: receive a WatchSpec --------------------------------

  /**
   * Apply a WatchSpec from the brain (down channel). Accepts a JSON string or
   * an object; validates it through the frozen contract. Returns the
   * normalized spec now in effect.
   */
  applyWatchSpec(specOrJson) {
    this._spec = parseWatchSpec(specOrJson);
    // Build one SkillBank per goal that selected detectors, from the skills this edge has REGISTERED.
    // Unknown detectors degrade gracefully (dropped + recorded in unsupportedDetectors) rather than
    // crash the edge loop — the capability handshake will keep the brain from selecting them at all.
    const supported = new Set(registered());
    this._banks = new Map();
    this._unsupported = [];
    for (const c of this._spec.conditions) {
      const known = [];
      for (const d of c.detectors ?? []) {
        if (supported.has(d)) known.push(d);
        else this._unsupported.push({ goal_id: c.goal_id, detector: d });
      }
      if (known.length) this._banks.set(c.goal_id, new SkillBank(known));
    }
    return this._spec;
  }

  /** The WatchSpec currently in effect, or null. */
  get watchSpec() {
    return this._spec;
  }

  /**
   * Lazily initialise any model-backed skill in the active banks — call once after applyWatchSpec,
   * before processing frames. DSP skills are no-ops, so a pure-DSP edge need not call this.
   */
  async loadSkills() {
    for (const bank of this._banks.values()) await bank.load();
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
  processFrame(frame, { energy = null } = {}) {
    const ts = this._clock(); // one clock read per frame — shared by skills + the signal
    const motion = motionScore(this._prevFrame, frame);
    this._prevFrame = frame;

    // Edge skills run EVERY frame on the cheap channels (motion / energy / frame) — DECOUPLED from the
    // sparse, motion-gated Tier0Signal below. Each boundary is a discrete event the brain counts WITHOUT
    // a cloud call (edge proposes · brain counts), tagged with the goal_id it fired for. This mirrors the
    // harness's per-frame DetectorBank + per-event `perceive_boundary` dispatch.
    if (this._banks.size) {
      const batch = [];
      for (const [goalId, bank] of this._banks) {
        for (const boundary of bank.push(ts, { motion, energy, frame })) {
          batch.push({ goal_id: goalId, boundary });
        }
      }
      if (batch.length) this._emitBoundaries(batch); // one coalesced Boundary message per tick
    }

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

    const signal = makeTier0Signal({ motion, structural, semantic, ts });

    if (this._publisher) this._publisher(frame, signal);
    this._emit(signal);
    return signal;
  }

  _emit(signal) {
    for (const fn of this._listeners) fn(signal);
  }

  _emitBoundaries(batch) {
    for (const fn of this._boundaryListeners) fn(batch);
  }
}

function clamp01(v) {
  if (typeof v !== "number" || !Number.isFinite(v)) return 0;
  if (v < 0) return 0;
  if (v > 1) return 1;
  return v;
}
