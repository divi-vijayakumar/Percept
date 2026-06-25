// src/pose.js
//
// pose-openness — the first MODEL-backed edge skill (the JS twin of percept_harness's PoseOpenness).
//
// Counts fast whole-body reps where frame-diff motion OVER-counts (it conflates each rep's open+close)
// and a per-frame VLM ALIASES on fast motion. openness = ankle-spread/shoulder-width + wrist-elevation
// → a clean waveform with one peak per rep; an inner MotionPeriodicity turns that waveform into 'peak'
// boundaries (relabeled to this skill).
//
// MODEL-backed → LAZY + opt-in: the pose model loads on `load()`, never at register/construct (the
// SkillBank.load() / Edge.loadSkills() hook). The product supplies the model as a *landmarker adapter*
// via registerPoseSkill(landmarker) — a build that never counts reps never ships or loads a pose model
// (mirrors the Python `percept-harness[pose]` opt-in). The adapter is pluggable so the heavy model
// (BlazePose via MediaPipe Tasks for Web, TF.js MoveNet, …) lives in the product, not in percept-edge.
//
//   landmarker adapter = {
//     async load(): Promise<void>,                          // init the model (weights, WASM, …)
//     detect(frame): Array<{x:number, y:number}> | null,    // 33 BlazePose landmarks, or null if no person
//   }

import { CATEGORY_FRAME, MotionPeriodicity, registerSkill } from "./skills.js";

// BlazePose landmark indices (identical to the Python side).
const L_ANKLE = 27, R_ANKLE = 28, L_SHOULDER = 11, R_SHOULDER = 12, L_WRIST = 15, R_WRIST = 16;

/**
 * The openness scalar for one pose: legs-spread (normalized by shoulder width) + wrist-elevation.
 * Tracks both legs-only variations (via the spread) and full jumping-jacks (the arms add wrist-up).
 */
export function openness(landmarks) {
  const spread = Math.abs(landmarks[L_ANKLE].x - landmarks[R_ANKLE].x);
  const sw = Math.abs(landmarks[L_SHOULDER].x - landmarks[R_SHOULDER].x) || 0.1;
  const wristUp =
    Math.max(0, landmarks[L_SHOULDER].y - landmarks[L_WRIST].y) +
    Math.max(0, landmarks[R_SHOULDER].y - landmarks[R_WRIST].y);
  return spread / sw + wristUp;
}

export class PoseOpenness {
  static signal = "frame";
  static skillName = "pose-openness";

  constructor({ landmarker, minHeight = 0.6, relDrop = 0.4, refractoryS = 0.4 } = {}) {
    this._landmarker = landmarker;
    this._peak = new MotionPeriodicity({ minHeight, relDrop, refractoryS });
    this._last = 0.0;
    this.signal = PoseOpenness.signal;
    this.name = PoseOpenness.skillName;
  }

  /** Lazy: init the model on activation (never at construct). No-op if the adapter has no load(). */
  async load() {
    if (this._landmarker && typeof this._landmarker.load === "function") {
      await this._landmarker.load();
    }
  }

  reset() {
    this._last = 0.0;
    this._peak.reset();
  }

  push(t, frame) {
    if (!this._landmarker) {
      throw new Error("pose-openness needs a landmarker adapter — register via registerPoseSkill(landmarker)");
    }
    const landmarks = this._landmarker.detect(frame);
    if (landmarks) this._last = openness(landmarks);
    // hold the last openness when no person is detected (the waveform doesn't reset between frames),
    // and relabel the inner peak detector's boundaries to this skill (mirrors the Python).
    return this._peak.push(t, this._last).map((b) => ({ ...b, detector: this.name }));
  }
}

/**
 * Register pose-openness with a product-supplied landmarker (the model). Opt-in, like the Python
 * `percept-harness[pose]` extra: a build that wants on-device rep-counting calls this once with its
 * pose model adapter, and the brain can then select "pose-openness" for a goal.
 */
export function registerPoseSkill(landmarker) {
  registerSkill({
    name: "pose-openness",
    signal: "frame",
    factory: () => new PoseOpenness({ landmarker }),
    manifest: {
      name: "pose-openness",
      senses: "body pose over time (BlazePose keypoints: ankle-spread + wrist-elevation)",
      when_to_use:
        "counting FAST whole-body reps where motion-diff over-counts and the VLM aliases — jumping jacks, squats, push-ups",
      output: "one 'peak' boundary per rep (the open phase of the openness waveform)",
      category: CATEGORY_FRAME,
    },
  });
}
