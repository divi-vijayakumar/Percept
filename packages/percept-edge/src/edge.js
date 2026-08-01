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

import { makeTier0Signal, makeEdgeCapabilities, parseWatchSpec } from "./contract.js";
import { motionScore, egoResidualScore } from "./motion.js";
import { SkillBank, cheapSignals, registered, wakeScore } from "./skills.js";
import { clamp01 } from "./numeric.js";

export class Edge {
  /**
   * @param {object} [opts]
   * @param {(signal:object)=>void} [opts.onSignal]  up-channel sink for Tier0Signal
   * @param {number} [opts.defaultThreshold]  salience floor for an edge that has NEVER been given a
   *        WatchSpec (so motion-only still gates). Default 0.05. It does NOT apply to an APPLIED
   *        spec carrying zero conditions — that is a deployment whose goals were all cancelled, and
   *        it never escalates.
   * @param {()=>number} [opts.clock]  returns seconds; for deterministic ts.
   */
  constructor({
    onSignal = null,
    onBoundary = null,
    defaultThreshold = 0.05,
    clock = null,
    wakePolicy = null,
  } = {}) {
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
    this._lastEscalate = new Map(); // goal_id -> ts, for cadence_s / fixed cadence
    this._wakePolicy = wakePolicy; // optional weighted cheap-signal fusion

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

  /** The EdgeCapabilities message to send the brain at connect (the advertise half of the handshake). */
  advertise() {
    return makeEdgeCapabilities(this.capabilities());
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
    this._lastEscalate = new Map();
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
   * The salience threshold the gate must clear. We take the *minimum* threshold across active
   * conditions (any goal clearing its bar escalates).
   *
   * THREE states, not two — an applied spec with no conditions is not the same edge as one that was
   * never configured:
   *
   *   * no spec ever applied  -> `defaultThreshold`, so a motion-only edge still gates;
   *   * an applied spec with conditions -> the minimum threshold across them;
   *   * an applied spec with NO conditions -> `Infinity`. Every goal has been cancelled, so no
   *     salience can clear the bar. Reporting `defaultThreshold` here said the opposite of what the
   *     gate now does (FIND-PARITY-03).
   */
  get activeThreshold() {
    if (this._spec == null) return this.defaultThreshold;
    const conds = this._spec.conditions;
    if (conds.length === 0) return Infinity;
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

  /** Plug a weighted cheap-signal wake policy. Null restores legacy max-channel salience. */
  setWakePolicy(policy) {
    this._wakePolicy = policy;
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
  processFrame(frame, { energy = null, signals = null } = {}) {
    const ts = this._clock(); // one clock read per frame — shared by skills + the signal
    const prevFrame = this._prevFrame;
    const motion = motionScore(prevFrame, frame);
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
      structural = clamp01(this._structuralAdapter(frame, prevFrame));
    }

    let semantic = 0;
    if (this._semanticAdapter) {
      semantic = clamp01(this._semanticAdapter(frame, this._spec));
    }

    // One measurement, one channel — see `cheapSignals`. This used to build seven keys from four
    // measurements (`change` aliasing `motion`, `semantic_relevance` aliasing `semantic`,
    // `audio_event` aliasing `audio`), which fed the noisy-OR the same reading twice.
    const channels = cheapSignals({ motion, structural, semantic, energy, extra: signals });
    // `salience` is the specless fallback (no conditions -> no goal has asked for anything, so the
    // edge gates on the best cheap channel, or on the fused score if a policy is the only instruction
    // it has). `fused` is the per-goal channel, and it exists ONLY when a policy is wired — mirroring
    // Python `MotionGate.evaluate`, where `fused is None` without one.
    const salience = wakeScore(channels, this._wakePolicy);
    const fused = this._wakePolicy ? salience : null;

    if (!this._shouldEscalate({ ts, prevFrame, frame, motion, structural, semantic, salience, fused })) {
      return null; // gated: still scene -> no Tier0Signal, no cloud call
    }

    const signal = makeTier0Signal({ motion, structural, semantic, ts, energy });

    if (this._publisher) this._publisher(frame, signal);
    this._emit(signal);
    return signal;
  }

  _shouldEscalate({ ts, prevFrame, frame, motion, structural, semantic, salience, fused }) {
    // NEVER CONFIGURED is not the same state as EVERY GOAL CANCELLED, and one branch used to serve
    // both. An edge that has not been handed a spec has no instruction, so it gates motion-only on
    // `defaultThreshold` — that fallback is what makes a bare `new Edge()` useful. An edge holding
    // an APPLIED spec with zero conditions has been told, precisely, that there is nothing to look
    // for: the brain compiles exactly `{version, conditions: []}` when the last watch is cancelled.
    // Falling back there kept the device clearing 0.05 on ordinary motion and handing frames to the
    // publisher for a watch nobody has — the privacy promise of the on-device topology failing in
    // the one state where nothing can possibly matter (FIND-PARITY-03). Python `MotionGate.evaluate`
    // has always answered this input with "never escalates — there is nothing to look for".
    if (this._spec == null) return salience >= this.defaultThreshold;
    const conds = this._spec.conditions;
    if (conds.length === 0) return false;
    // EVERY armed condition is evaluated, and every one that fires is stamped — the loop
    // ACCUMULATES, it does not return early. Returning on the first firing goal left every other
    // goal sharing the frame unstamped, so its cadence brake stayed at -Infinity and it admitted
    // again on the very next frame, inside the window the deployment authored (FIND-PARITY-02).
    // Twin of Python `MotionGate.evaluate`, where `escalate` is a disjunction over the whole loop.
    let escalate = false;
    for (const c of conds) {
      if (c.cadence_s != null && ts - (this._lastEscalate.get(c.goal_id) ?? -Infinity) < c.cadence_s) {
        continue;
      }
      // ONE branch per channel, each reached only by the goal that asked for it. The fused branch used
      // to sit ABOVE structural/semantic and fire on `this._wakePolicy` being set at all, so a
      // configured policy replaced every goal's own channel. A goal now opts in with
      // `gate_on: "fused"`. Twin of Python `MotionGate.evaluate`.
      let fired = false;
      if (c.gate_on === "fixed") {
        fired = c.cadence_s != null;
      } else if (c.gate_on === "fused") {
        // No wake policy wired -> a goal that asked to be fused has nothing to be fused from. It
        // scores 0 rather than silently falling back to motion.
        fired = (fused ?? 0) >= c.threshold;
      } else if (c.gate_on === "structural") {
        fired = structural >= c.threshold;
      } else if (c.gate_on === "semantic") {
        fired = semantic >= c.threshold;
      } else {
        // "motion" (default) — honor this goal's salience MODE (localized stat / ego / region).
        // Mirrors Python `_condition_motion`: default (mean, no region, no ego) uses the precomputed
        // full-frame motion; ego compensation takes precedence, else the localized block statistic.
        fired = conditionMotion(prevFrame, frame, c, motion) >= c.threshold;
      }
      if (fired) {
        this._lastEscalate.set(c.goal_id, ts);
        escalate = true;
      }
    }
    return escalate;
  }

  _emit(signal) {
    for (const fn of this._listeners) fn(signal);
  }

  _emitBoundaries(batch) {
    for (const fn of this._boundaryListeners) fn(batch);
  }
}

// The escalate-comparison motion for one condition, honoring its salience MODE. Twin of Python
// `percept.harness.tier0._condition_motion`: default (mean, no region, no ego) returns the precomputed
// full-frame `fullMotion`; ego compensation takes precedence, else the localized block statistic
// (optionally masked to region_hint).
//
// EGO COMPOSES WITH THE ZONE, not with the block statistic. The ego branch passes `region_hint`
// through, because a watch-out zone is a promise that movement elsewhere is ignored, and turning on
// ego compensation must not quietly convert the goal back into a whole-frame watch (it used to: the
// residual was always whole-frame while the condition still carried and reported its zone). Ego +
// block_max/p95 composition stays deferred — an ego goal gets the region-masked residual, not the
// localized block statistic.
//
// The result is clamped like every OTHER channel. The structural and semantic adapters are clamped
// where they are read (processFrame), but this one was compared `>= c.threshold` raw — so the one
// channel that is always live, and the only one that can be driven from a WatchSpec field, was also
// the one channel with no floor under it.
function conditionMotion(prevFrame, frame, c, fullMotion) {
  return clamp01(rawConditionMotion(prevFrame, frame, c, fullMotion));
}

function rawConditionMotion(prevFrame, frame, c, fullMotion) {
  if (c.ego_compensate) {
    return egoResidualScore(prevFrame, frame, { region: c.region_hint ?? null });
  }
  const stat = c.salience_stat ?? "mean";
  if (stat !== "mean" || c.region_hint != null) {
    return motionScore(prevFrame, frame, { region: c.region_hint ?? null, stat });
  }
  return fullMotion;
}
