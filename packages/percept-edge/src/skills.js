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
import { clamp01 } from "./numeric.js";

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
 * Shipped wake bar for `AcousticOnset`, in MEAN-SQUARE ENERGY (not amplitude, not dB). The twin of
 * Python `percept.harness.detectors.ACOUSTIC_ONSET_HIGH`, which carries the full measurement.
 *
 * THE PREVIOUS DEFAULT WAS 0.09 — `0.3 ** 2`, an RMS amplitude of 0.3 (~-10 dBFS sustained). Measured
 * against 987 s / 3,973 capture chunks of human-annotated sound (Perception Test `sound_localisation`,
 * 53 cases), the LOUDEST chunk in the whole corpus reached energy 0.0277, so that gate woke ZERO times
 * and matched 0 of 204 annotated sound events. It could not be crossed by the material it exists to
 * hear. 0.001 is RMS 0.032 (~-30 dBFS) with a re-arm bar at RMS 0.016 (~-36 dBFS) — the conventional
 * "something audible started" bar, a measured point on that corpus's frontier (0.294 recall at 1.8% of
 * chunks admitted). The argmax (5e-05) was deliberately NOT taken: it sits below the noise floor of
 * most live microphones, where it would wake once and never re-arm.
 */
export const ACOUSTIC_ONSET_HIGH = 0.001;

/** Re-arm bar, energy. Held at `ACOUSTIC_ONSET_HIGH / 4` — a factor of 4 in energy is a factor of 2 in
 * amplitude, the same hysteresis relationship the original `0.09`/`0.0225` pair encoded. */
export const ACOUSTIC_ONSET_LOW = ACOUSTIC_ONSET_HIGH / 4.0;

/**
 * Detects energy onsets (a hiss/grunt/beep starting) — one `onset` boundary per rising spike.
 * Hysteresis: fire when energy crosses `high` having been below `low` (armed), then re-arm only after
 * it falls back under `low`. Pure DSP on the mic-energy stream. Faithful port of `AcousticOnset`.
 *
 * BAND ORDERING. `low` must not sit ABOVE `high`: the re-arm bar is by definition below the fire bar,
 * and an inverted pair makes the two swap roles — every sample in `[high, low)` both fires (`v >= high`)
 * and re-arms (`v < low`), so a dead-steady room tone oscillates the state machine at half the sample
 * rate and mints a phantom onset every other sample. An onset is "a sound STARTING", and a level that
 * never changes contains exactly one start whatever the bars are, so this is refused in the constructor
 * rather than left to the caller to notice.
 *
 * `low === high` stays PERMITTED: it is the degenerate no-band configuration, not the defect. No sample
 * can be both `>= high` and `< low` when they are equal, so the oscillation is unreachable.
 */
export class AcousticOnset {
  static signal = "energy";
  static skillName = "acoustic-onset";

  constructor({ high = ACOUSTIC_ONSET_HIGH, low = ACOUSTIC_ONSET_LOW, refractoryS = 0.0 } = {}) {
    if (low > high) {
      throw new Error(
        `AcousticOnset needs the re-arm bar at or below the fire bar (low <= high); got high=${high}, ` +
          `low=${low}. An inverted band makes a constant energy level both fire and re-arm on the same ` +
          `sample, so a steady tone emits a phantom onset every other sample. Both are MEAN-SQUARE ` +
          `ENERGY, not RMS amplitude.`,
      );
    }
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
// Sense categories — the CLASS of physical signal a detector reads
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

/** The complete authored surface of a wake policy. Anything else is a typo, refused by name. */
const WAKE_POLICY_KEYS = new Set(["threshold", "weights", "floor"]);

/**
 * An authored wake policy this module refuses to run, naming the field that made it unusable. Twin of
 * Python `UnsupportedWakePolicyError`.
 *
 * THE CHOICE, mirrored from the Python module: a wake policy exists to REPLACE the default fusion, so
 * a policy that silently degrades is indistinguishable from no policy at all while still carrying an
 * authored threshold. `weigths` used to read as an absent `weights` and leave the calibrated OR quietly
 * running as `max(signals)`. There is no partially-applied policy: it either ran as authored, or never
 * constructed.
 */
export class WakePolicyError extends Error {
  constructor(field, value, reason) {
    super(`wake policy field ${JSON.stringify(field)}=${JSON.stringify(value) ?? String(value)} — ${reason}`);
    this.name = "WakePolicyError";
    this.field = field;
    this.value = value;
  }
}

/** One finite number, or a typed refusal naming the field. Mirrors Python `_wake_number`. */
function wakeNumber(field, value) {
  if (typeof value === "boolean") {
    throw new WakePolicyError(field, value, "a bool is not a number");
  }
  if (typeof value !== "number") {
    throw new WakePolicyError(
      field,
      value,
      "must be a finite number; a stringified number is refused rather than parsed, so the two " +
        "runtimes cannot disagree about what a string is worth",
    );
  }
  if (!Number.isFinite(value)) throw new WakePolicyError(field, value, "must be finite");
  return value;
}

function normalizeWakePolicy(policy) {
  if (policy == null) return null;
  if (typeof policy !== "object" || Array.isArray(policy)) {
    throw new WakePolicyError("<policy>", policy, "a wake policy is a mapping of threshold/weights/floor");
  }
  const unknown = Object.keys(policy).filter((k) => !WAKE_POLICY_KEYS.has(k)).sort();
  if (unknown.length) {
    throw new WakePolicyError(
      "<policy>",
      unknown,
      "unknown field(s); a wake policy carries exactly floor, threshold, weights. A misspelled key " +
        "used to read as an absent one, which left the authored threshold in place while the fusion " +
        "quietly reverted to max(signals)",
    );
  }
  const weightsRaw = policy.weights ?? {};
  if (typeof weightsRaw !== "object" || Array.isArray(weightsRaw)) {
    throw new WakePolicyError("weights", weightsRaw, "must be a mapping of channel name -> numeric prior");
  }
  const weights = {};
  for (const [k, v] of Object.entries(weightsRaw)) weights[k] = wakeNumber(`weights[${k}]`, v);
  return {
    threshold: wakeNumber("threshold", policy.threshold ?? 0.1),
    weights,
    floor: wakeNumber("floor", policy.floor ?? 0),
  };
}

/**
 * THE fusion, once: the fused score AND the per-channel contribution terms it summed. Twin of Python
 * `_fuse`. `policy` must already be normalized.
 *
 * One walk over the MEASURED channels produces both halves. `score` and `top_channel` used to be
 * computed from two different populations (the weights table and the raw signals respectively), so a
 * decision could name a channel that contributed nothing to the number beside it.
 */
function fuse(signals, policy) {
  const clean = {};
  for (const [k, v] of Object.entries(signals ?? {})) clean[k] = clamp01(v);
  let contributions;
  let base;
  if (policy == null || Object.keys(policy.weights).length === 0) {
    contributions = clean;
    const values = Object.values(contributions);
    base = values.length === 0 ? 0 : Math.max(...values);
  } else {
    contributions = {};
    let miss = 1;
    for (const [channel, value] of Object.entries(clean)) {
      const term = clamp01(value * (policy.weights[channel] ?? 1.0));
      contributions[channel] = term;
      miss *= 1 - term;
    }
    base = 1 - miss;
  }
  const floor = policy ? clamp01(policy.floor) : 0;
  return { score: clamp01(Math.max(base, floor)), contributions };
}

/**
 * Fuse cheap edge signals into `P(goal)` before spending a heavy perception call.
 *
 * No explicit weights preserves the legacy behavior exactly: `max(signals)`. With weights this is a
 * noisy-OR **over every measured channel**, each priced by its weight (default 1.0) — so several weak
 * cues can wake together without making any single detector authoritative, and adding a weight can
 * never score below the `max(signals)` it replaced. It used to iterate the WEIGHT table instead, which
 * deleted every measured channel the policy did not name and made a typo'd channel total deafness.
 *
 * A channel named in `weights` but absent from `signals` contributes nothing: an unwired sensor cannot
 * vote, and "not reporting" is not the same fact as "sees nothing".
 */
export function wakeScore(signals = {}, policy = null) {
  return fuse(signals, normalizeWakePolicy(policy)).score;
}

/** Decide whether cheap evidence is strong enough to wake perception. */
export function shouldWake(signals = {}, policy = null) {
  const p = normalizeWakePolicy(policy);
  const { score, contributions } = fuse(signals, p);
  const threshold = p ? p.threshold : 0.1;
  // The argmax of the CONTRIBUTIONS, ties taking the first (strict `>` from a 0 baseline). A channel
  // that contributed exactly nothing can never be named, so a policy that prices none of the measured
  // channels reports null rather than pointing at an irrelevant sensor.
  let topChannel = null;
  let best = 0;
  for (const [k, term] of Object.entries(contributions)) {
    if (term > best) {
      topChannel = k;
      best = term;
    }
  }
  return { wake: score >= threshold, score, threshold, top_channel: topChannel };
}

/**
 * The edge's cheap-channel vector — the named producer the wake fusion consumes. Twin of Python
 * `percept.harness.tier0.cheap_signals`.
 *
 * THE INVARIANT: **one physical measurement populates exactly one channel.** Four measurements, four
 * channels (`motion`, `structural`, `semantic`, `audio`), and no aliases. It used to build SEVEN keys
 * from those four — `change` aliased `motion`, `semantic_relevance` aliased `semantic`, `audio_event`
 * aliased `audio`. The noisy-OR assumes its channels are INDEPENDENT cues, so an alias fuses one
 * measurement with itself: a policy weighting `motion` and `change` at 0.5 each scored 0.51 on one
 * reading of 0.6, against 0.30 for the same reading on the single channel.
 *
 * `audio_event` is absent rather than aliased for a second reason: it names an ACOUSTIC-EVENT class,
 * and nothing in this build classifies acoustic events. Mapping raw energy onto it claimed a sense that
 * does not exist.
 *
 * `audio` is OMITTED, not zeroed, when there is no energy reading: "no microphone is open" and "the
 * room is silent" are different facts. `extra` is the caller's overlay — the supported way to add a
 * channel a real detector measures (`presence`, `pose`, an acoustic-event classifier).
 */
export function cheapSignals({ motion, structural, semantic, energy = null, extra = null } = {}) {
  const out = { motion, structural, semantic };
  if (energy != null) out.audio = energy;
  if (extra) Object.assign(out, extra);
  return out;
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
 * `detectorsInCategory("audio")` lists every audio Tier-0 sense, mirroring Python
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
    category: CATEGORY_AUDIO, // the first-class AUDIO sense — discoverable as a class
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
 * THE CHANNEL VOCABULARY IS CLOSED: exactly `motion` / `energy` / `frame` (`SkillBank.CHANNELS`),
 * matching Python `DetectorBank`. It used to be two recognised names plus a catch-all — `const sig =
 * signal === "energy" ? energy : motion` — so a skill declaring `signal="thermal"`, or a typo like
 * `"enrgy"`, was silently fed frame-diff motion and emitted boundaries the brain then counted as
 * coming from the sense it declared. An unrecognised channel is now the same as an absent one: no
 * input, ever. (`registerSkill` refuses an out-of-vocabulary `signal` at registration, so this is the
 * second gate, not the only one — a skill can also be constructed and banked directly.)
 *
 * Lazy local load (#2): `await bank.load()` once on activation initialises any model-backed skill
 * (a skill instance may expose an optional async `load()`); DSP skills are no-ops. Construction
 * (`makeDetector`) never loads a model — that happens here, only when a goal activates the skill.
 */
export class SkillBank {
  /** The channels the bank can carry. Closed on purpose — see the class docstring. */
  static CHANNELS = ["motion", "energy", "frame"];

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
    const channels = { motion, energy, frame };
    const out = [];
    for (const { det, signal } of this._dets) {
      // A channel this bank does not carry, or does not have this tick -> no input. Never a fallback.
      if (!Object.prototype.hasOwnProperty.call(channels, signal) || channels[signal] == null) continue;
      if (signal === "frame") {
        out.push(...det.push(t, frame)); // frame-backed skill (e.g. pose) — the raw frame, not a scalar
      } else {
        out.push(...det.push(t, channels[signal]));
      }
    }
    return out;
  }
}
