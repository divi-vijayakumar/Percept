// src/skills.js
//
// The on-device EDGE SKILL REGISTRY — the JS twin of `percept/harness/detectors.py`.
//
// A curated, registry-extensible library of CHEAP, high-fps signal detectors the brain selects from
// per goal — the **skills** model: register an impl + a manifest, and the brain composes it into
// concerns by name, never a per-task `if`. Each detector consumes a scalar signal stream and emits
// **timed boundaries**: the edge proposes WHERE a rep / onset / threshold-crossing happened; the brain
// counts them and the VLM confirms the semantics sparingly at those boundaries.
//
// This is the product extension point: an assistant built on Percept registers ONLY the skills its
// edge needs (fitness coach → motion-periodicity + pose; AR glasses → acoustic-onset + cry-spotter;
// drone → optical-flow-obstacle), advertises that set up the wire, and the brain composes goals
// against what's actually registered. Adding a skill is a deliberate `registerSkill(...)` + calibrate
// event — never the LLM synthesizing detectors at runtime.
//
// Parity: the manifest + Boundary field names match the Python `DetectorManifest`/`Boundary`
// byte-for-byte, so a manifest can ride the wire (the capability handshake) unchanged.
//
//   Manifest = { name, senses, when_to_use, output, category }
//   Boundary = { detector, kind: "peak"|"onset"|"threshold", t, value }
//
// `category` is the coarse sense CLASS a detector belongs to ("motion" | "audio" | "frame" plus
// additive channels like "presence", "pose", "audio_event", "change", "semantic_relevance"), so a whole
// modality is discoverable at once (`detectorsInCategory("audio")` lists every audio skill) — this is
// what marks `acoustic-onset` as the first-class AUDIO sense. It mirrors the Python
// `DetectorManifest.category` byte-for-byte so the manifest rides the capability handshake unchanged.

// Boundary is a wire shape — defined (with validation) in the frozen contract; skills emit it.
import { makeBoundary } from "./contract.js";

// ---------------------------------------------------------------------------
// Detectors — stateful streaming DSP: push(t, value) -> Boundary[]
// Each carries a `signal` channel ("motion" | "energy" | "frame") and may expose an OPTIONAL
// async `load()` that the SkillBank awaits on activation (model-backed skills load their weights
// there — never at register/construct time; that's the "lazy local load").
// ---------------------------------------------------------------------------

/**
 * Counts oscillations in the motion signal — one `peak` boundary per completed bump (≈ one rep).
 * Adaptive prominence: a bump is confirmed only once the signal falls to `(1-relDrop)` of its peak,
 * so noise riding the way up doesn't over-count. `minHeight` ignores still-scene jitter; `refractoryS`
 * debounces. Pure DSP — no model. Faithful port of Python `MotionPeriodicity`.
 */
export class MotionPeriodicity {
  static signal = "motion";
  static skillName = "motion-periodicity";

  constructor({ minHeight = 0.02, relDrop = 0.5, refractoryS = 0.0 } = {}) {
    this.minHeight = minHeight;
    this.relDrop = relDrop;
    this.refractoryS = refractoryS;
    this.signal = MotionPeriodicity.signal;
    this.name = MotionPeriodicity.skillName;
    this.reset();
  }

  reset() {
    this._peakV = 0.0;
    this._peakT = 0.0;
    this._havePeak = false;
    this._lastEmitT = -Infinity;
  }

  push(t, v) {
    if (v > this._peakV) {
      // still climbing this bump -> track the running peak
      this._peakV = v;
      this._peakT = t;
      this._havePeak = this._peakV >= this.minHeight;
      return [];
    }
    if (this._havePeak && v <= this._peakV * (1.0 - this.relDrop)) {
      // confirmed descent -> emit the peak
      const out = [];
      if (t - this._lastEmitT >= this.refractoryS) {
        out.push(makeBoundary(this.name, "peak", this._peakT, this._peakV));
        this._lastEmitT = t;
      }
      this._peakV = v; // start a new bump from the trough
      this._peakT = t;
      this._havePeak = false;
      return out;
    }
    return [];
  }
}

/**
 * Detects energy onsets (a hiss/grunt/beep starting) — one `onset` boundary per rising spike.
 * Hysteresis: fire when energy crosses `high` having been below `low` (armed), then re-arm only after
 * it falls back under `low`. Pure DSP on the mic-energy stream. Faithful port of `AcousticOnset`.
 */
export class AcousticOnset {
  static signal = "energy";
  static skillName = "acoustic-onset";

  constructor({ high = 0.3, low = 0.15, refractoryS = 0.0 } = {}) {
    this.high = high;
    this.low = low;
    this.refractoryS = refractoryS;
    this.signal = AcousticOnset.signal;
    this.name = AcousticOnset.skillName;
    this.reset();
  }

  reset() {
    this._armed = true;
    this._lastEmitT = -Infinity;
  }

  push(t, v) {
    if (this._armed && v >= this.high) {
      const out = [];
      if (t - this._lastEmitT >= this.refractoryS) {
        out.push(makeBoundary(this.name, "onset", t, v));
        this._lastEmitT = t;
      }
      this._armed = false;
      return out;
    }
    if (!this._armed && v < this.low) {
      this._armed = true;
    }
    return [];
  }
}

// ---------------------------------------------------------------------------
// Sense categories — the CLASS of physical signal a detector reads (B2)
// ---------------------------------------------------------------------------

/** The coarse modality of a detector's signal, mirrored from Python `CATEGORY_*`. */
export const CATEGORY_MOTION = "motion";
export const CATEGORY_AUDIO = "audio";
export const CATEGORY_FRAME = "frame";
export const CATEGORY_PRESENCE = "presence";
export const CATEGORY_POSE = "pose";
export const CATEGORY_AUDIO_EVENT = "audio_event";
export const CATEGORY_CHANGE = "change";
export const CATEGORY_SEMANTIC_RELEVANCE = "semantic_relevance";

// ---------------------------------------------------------------------------
// Wake policy — cheap evidence fusion before a VLM call
// ---------------------------------------------------------------------------

function clamp01(v) {
  if (typeof v !== "number" || !Number.isFinite(v)) return 0;
  if (v < 0) return 0;
  if (v > 1) return 1;
  return v;
}

function normalizeWakePolicy(policy) {
  if (policy == null) return null;
  const weights = {};
  for (const [k, v] of Object.entries(policy.weights ?? {})) {
    if (typeof k === "string" && typeof v === "number" && Number.isFinite(v)) weights[k] = v;
  }
  return {
    threshold:
      typeof policy.threshold === "number" && Number.isFinite(policy.threshold) ? policy.threshold : 0.1,
    weights,
    floor: typeof policy.floor === "number" && Number.isFinite(policy.floor) ? policy.floor : 0,
  };
}

/**
 * Fuse cheap edge signals into `P(goal)` before spending a heavy perception call.
 *
 * No explicit weights preserves the legacy behavior exactly: `max(signals)`. With weights, this is a
 * noisy-OR over named channels: several weak cues can wake together without making any single detector
 * authoritative.
 */
export function wakeScore(signals = {}, policy = null) {
  const p = normalizeWakePolicy(policy);
  const clean = {};
  for (const [k, v] of Object.entries(signals ?? {})) clean[k] = clamp01(v);
  const values = Object.values(clean);
  let base = 0;
  if (values.length === 0) {
    base = 0;
  } else if (p == null || Object.keys(p.weights).length === 0) {
    base = Math.max(...values);
  } else {
    let miss = 1;
    for (const [channel, weight] of Object.entries(p.weights)) {
      miss *= 1 - clamp01((clean[channel] ?? 0) * weight);
    }
    base = 1 - miss;
  }
  return clamp01(Math.max(base, p ? clamp01(p.floor) : 0));
}

/** Decide whether cheap evidence is strong enough to wake perception. */
export function shouldWake(signals = {}, policy = null) {
  const p = normalizeWakePolicy(policy);
  const score = wakeScore(signals, p);
  const threshold = p ? p.threshold : 0.1;
  const clean = {};
  for (const [k, v] of Object.entries(signals ?? {})) clean[k] = clamp01(v);
  let topChannel = null;
  let topValue = -Infinity;
  for (const [k, v] of Object.entries(clean)) {
    if (v > topValue) {
      topChannel = k;
      topValue = v;
    }
  }
  return { wake: score >= threshold, score, threshold, top_channel: topChannel };
}

// ---------------------------------------------------------------------------
// The registry — register an impl + a manifest; the brain composes by name
// ---------------------------------------------------------------------------

/** @type {Map<string, {factory: Function, manifest: object, signal: string}>} */
const _REGISTRY = new Map();

/**
 * Register a skill (a deliberate, calibratable event — never per-task code). This is the third-party
 * extension point: a custom skill is `registerSkill({ name, signal, manifest, factory })`, nothing more.
 * @param {{name:string, signal:string, manifest:object, factory:Function}} skill
 */
export function registerSkill({ name, signal, manifest, factory }) {
  if (typeof name !== "string" || !name) throw new Error("registerSkill: name must be a non-empty string");
  if (typeof factory !== "function") throw new Error(`registerSkill(${name}): factory must be a function`);
  if (signal !== "motion" && signal !== "energy" && signal !== "frame") {
    throw new Error(`registerSkill(${name}): signal must be "motion" | "energy" | "frame"`);
  }
  _REGISTRY.set(name, { factory, manifest: manifest ?? { name, senses: "", when_to_use: "", output: "" }, signal });
}

/** Construct a fresh detector instance by registry name. */
export function makeDetector(name) {
  const entry = _REGISTRY.get(name);
  if (!entry) throw new Error(`unknown skill ${JSON.stringify(name)}; registered: ${registered().join(", ")}`);
  return entry.factory();
}

/** The manifest for a registered skill (what the brain reads to compose it). */
export function manifestOf(name) {
  const entry = _REGISTRY.get(name);
  if (!entry) throw new Error(`unknown skill ${JSON.stringify(name)}`);
  return entry.manifest;
}

/** All manifests — what a capability handshake advertises up so the brain composes against them. */
export function listManifests() {
  return [..._REGISTRY.values()].map((e) => e.manifest);
}

/** Registered skill names (sorted) — the edge's advertised capability set. */
export function registered() {
  return [..._REGISTRY.keys()].sort();
}

/**
 * Registered skill names (sorted) whose manifest `category` matches — modality-as-a-class discovery:
 * `detectorsInCategory("audio")` lists every audio Tier-0 sense (B2), mirroring Python
 * `detectors_in_category`. Empty when nothing in that category is registered.
 * @param {string} category
 * @returns {string[]}
 */
export function detectorsInCategory(category) {
  return [..._REGISTRY.entries()]
    .filter(([, e]) => (e.manifest?.category ?? "") === category)
    .map(([name]) => name)
    .sort();
}

/** The signal channel a registered skill consumes. */
export function signalOf(name) {
  const entry = _REGISTRY.get(name);
  if (!entry) throw new Error(`unknown skill ${JSON.stringify(name)}`);
  return entry.signal;
}

// ---- the built-in curated skills (pure-DSP; no model, no download) ----------

registerSkill({
  name: "motion-periodicity",
  signal: "motion",
  factory: () => new MotionPeriodicity(),
  manifest: {
    name: "motion-periodicity",
    senses: "motion magnitude over time (frame-diff)",
    when_to_use:
      "counting repetitive movements — reps, strokes, paces; any 'count the N-th time' on a periodic motion",
    output: "a 'peak' boundary per motion oscillation (≈ one rep), with no per-frame VLM call",
    category: CATEGORY_MOTION,
  },
});

registerSkill({
  name: "acoustic-onset",
  signal: "energy",
  factory: () => new AcousticOnset(),
  manifest: {
    name: "acoustic-onset",
    senses: "audio energy over time (the mic stream already flowing for STT)",
    when_to_use: "audio-cued events — a kettle/boil hiss rising, a timer beep, a rep grunt, a knock",
    output: "an 'onset' boundary each time energy spikes up (a sound starting)",
    category: CATEGORY_AUDIO, // the first-class AUDIO sense (B2) — discoverable as a class
  },
});

// ---------------------------------------------------------------------------
// SkillBank — run a goal's selected skills over a stream, routed by signal channel
// ---------------------------------------------------------------------------

/**
 * Runs a goal's selected skills over a stream, routing each its channel (motion / energy / frame).
 * The edge feeds the cheap signals it already computes; each skill reads its own channel and emits
 * boundaries. A skill whose channel isn't provided this tick simply gets no input.
 *
 * Lazy local load (#2): `await bank.load()` once on activation initialises any model-backed skill
 * (a skill instance may expose an optional async `load()`); DSP skills are no-ops. Construction
 * (`makeDetector`) never loads a model — that happens here, only when a goal activates the skill.
 */
export class SkillBank {
  /** @param {Iterable<string>} names registered skill names the brain selected for this goal */
  constructor(names) {
    this._dets = [...names].map((n) => ({ det: makeDetector(n), signal: signalOf(n) }));
    this._loaded = false;
  }

  /** Lazily initialise any model-backed skill (called once on activation). Idempotent. */
  async load() {
    if (this._loaded) return;
    for (const { det } of this._dets) {
      if (typeof det.load === "function") await det.load();
    }
    this._loaded = true;
  }

  reset() {
    for (const { det } of this._dets) det.reset();
  }

  /**
   * Push one tick of the cheap signals; returns the boundaries emitted this tick.
   * @param {number} t
   * @param {{motion?:number, energy?:number, frame?:any}} channels
   * @returns {object[]} Boundary[]
   */
  push(t, { motion = null, energy = null, frame = null } = {}) {
    const out = [];
    for (const { det, signal } of this._dets) {
      if (signal === "frame") {
        if (frame != null) out.push(...det.push(t, frame));
        continue;
      }
      const sig = signal === "energy" ? energy : motion;
      if (sig == null) continue;
      out.push(...det.push(t, sig));
    }
    return out;
  }
}
