// src/contract.js
//
// The FROZEN edge wire-contract. Field names here MUST match the Python SDK
// dataclasses byte-for-byte so the two halves interoperate over JSON:
//
//   Box            = { x, y, w, h }                              # normalized 0..1
//   WatchCondition = { goal_id, anchor_text, threshold, region_hint: Box|null,
//                      detectors: string[] }                     # cheap edge detectors the brain selected
//   WatchSpec      = { conditions: WatchCondition[], version }   # version defaults to 1
//   Tier0Signal    = { motion, structural, semantic, ts }
//
// Down channel  (cloud -> edge): WatchSpec
// Up   channel  (edge -> cloud): Tier0Signal
//
// Any change to a field name is an interop break. Treat these shapes as frozen.

/** Error thrown when an incoming wire payload violates the contract. */
export class ContractError extends Error {
  constructor(message) {
    super(message);
    this.name = "ContractError";
  }
}

function isFiniteNumber(v) {
  return typeof v === "number" && Number.isFinite(v);
}

function isPlainObject(v) {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function isStringArray(v) {
  return Array.isArray(v) && v.every((s) => typeof s === "string");
}

// ---------------------------------------------------------------------------
// Box  { x, y, w, h }  (normalized 0..1)
// ---------------------------------------------------------------------------

/**
 * Build a normalized Box. All fields default to 0.
 * @returns {{x:number,y:number,w:number,h:number}}
 */
export function makeBox({ x = 0, y = 0, w = 0, h = 0 } = {}) {
  return { x, y, w, h };
}

/**
 * Validate + normalize a Box-shaped value. Returns a fresh Box object.
 * @throws {ContractError}
 */
export function parseBox(obj, { path = "region_hint" } = {}) {
  if (!isPlainObject(obj)) {
    throw new ContractError(`${path}: expected an object, got ${typeof obj}`);
  }
  for (const k of ["x", "y", "w", "h"]) {
    if (!isFiniteNumber(obj[k])) {
      throw new ContractError(`${path}.${k}: expected a finite number`);
    }
  }
  return { x: obj.x, y: obj.y, w: obj.w, h: obj.h };
}

// ---------------------------------------------------------------------------
// WatchCondition  { goal_id, anchor_text, threshold, region_hint: Box|null,
//                   detectors: string[] }
// ---------------------------------------------------------------------------

/**
 * Build a WatchCondition. `detectors` are the cheap on-device detector names the
 * brain selected for this goal (e.g. "motion-periodicity", "acoustic-onset");
 * defaults to [] (the Python `field(default_factory=list)` default).
 * @returns {{goal_id:string,anchor_text:string,threshold:number,region_hint:object|null,detectors:string[]}}
 */
export function makeWatchCondition({
  goal_id,
  anchor_text,
  threshold = 0.3,
  region_hint = null,
  detectors = [],
} = {}) {
  if (typeof goal_id !== "string") {
    throw new ContractError("makeWatchCondition: goal_id must be a string");
  }
  if (typeof anchor_text !== "string") {
    throw new ContractError("makeWatchCondition: anchor_text must be a string");
  }
  if (!isFiniteNumber(threshold)) {
    throw new ContractError("makeWatchCondition: threshold must be a finite number");
  }
  if (!isStringArray(detectors)) {
    throw new ContractError("makeWatchCondition: detectors must be an array of strings");
  }
  return {
    goal_id,
    anchor_text,
    threshold,
    region_hint: region_hint == null ? null : makeBox(region_hint),
    detectors: [...detectors],
  };
}

/**
 * Validate one WatchCondition off the wire. Returns a fresh, normalized object.
 * @throws {ContractError}
 */
export function parseWatchCondition(obj, { path = "condition" } = {}) {
  if (!isPlainObject(obj)) {
    throw new ContractError(`${path}: expected an object`);
  }
  if (typeof obj.goal_id !== "string") {
    throw new ContractError(`${path}.goal_id: expected a string`);
  }
  if (typeof obj.anchor_text !== "string") {
    throw new ContractError(`${path}.anchor_text: expected a string`);
  }
  if (!isFiniteNumber(obj.threshold)) {
    throw new ContractError(`${path}.threshold: expected a finite number`);
  }
  // region_hint is Box | null. Missing is tolerated and treated as null so a
  // Python side that omits the key still parses; explicit null is canonical.
  let region_hint = null;
  if (obj.region_hint != null) {
    region_hint = parseBox(obj.region_hint, { path: `${path}.region_hint` });
  }
  // detectors is string[]. Missing is tolerated -> [] so an older Python side
  // (pre-detector-registry) that omits the key still parses; [] is canonical.
  let detectors = [];
  if (obj.detectors != null) {
    if (!isStringArray(obj.detectors)) {
      throw new ContractError(`${path}.detectors: expected an array of strings`);
    }
    detectors = [...obj.detectors];
  }
  return {
    goal_id: obj.goal_id,
    anchor_text: obj.anchor_text,
    threshold: obj.threshold,
    region_hint,
    detectors,
  };
}

// ---------------------------------------------------------------------------
// WatchSpec  { conditions: WatchCondition[], version }
// ---------------------------------------------------------------------------

/**
 * Build a WatchSpec. version defaults to 1 (matches the Python default).
 * @returns {{conditions:object[],version:number}}
 */
export function makeWatchSpec({ conditions = [], version = 1 } = {}) {
  if (!Array.isArray(conditions)) {
    throw new ContractError("makeWatchSpec: conditions must be an array");
  }
  if (!isFiniteNumber(version)) {
    throw new ContractError("makeWatchSpec: version must be a finite number");
  }
  return {
    conditions: conditions.map((c) => makeWatchCondition(c)),
    version,
  };
}

/**
 * Parse + validate a WatchSpec from the wire. Accepts either a JSON string or
 * an already-parsed object. Returns a fresh, normalized WatchSpec. This is the
 * cross-language entry point: the Python side emits exactly this shape.
 * @param {string|object} json
 * @throws {ContractError}
 */
export function parseWatchSpec(json) {
  let obj = json;
  if (typeof json === "string") {
    try {
      obj = JSON.parse(json);
    } catch (e) {
      throw new ContractError(`parseWatchSpec: invalid JSON (${e.message})`);
    }
  }
  if (!isPlainObject(obj)) {
    throw new ContractError("parseWatchSpec: expected an object");
  }
  if (!Array.isArray(obj.conditions)) {
    throw new ContractError("parseWatchSpec: conditions must be an array");
  }
  // version defaults to 1 if omitted (Python dataclass default).
  let version = 1;
  if (obj.version !== undefined) {
    if (!isFiniteNumber(obj.version)) {
      throw new ContractError("parseWatchSpec: version must be a finite number");
    }
    version = obj.version;
  }
  const conditions = obj.conditions.map((c, i) =>
    parseWatchCondition(c, { path: `conditions[${i}]` })
  );
  return { conditions, version };
}

// ---------------------------------------------------------------------------
// Tier0Signal  { motion, structural, semantic, ts }
// ---------------------------------------------------------------------------

/**
 * Build a Tier0Signal (the up-channel salience report). Unsupplied salience
 * channels default to 0; ts defaults to wall-clock seconds (to match the
 * Python `ts: float` seconds convention).
 * @returns {{motion:number,structural:number,semantic:number,ts:number}}
 */
export function makeTier0Signal({
  motion = 0,
  structural = 0,
  semantic = 0,
  ts = Date.now() / 1000,
} = {}) {
  for (const [k, v] of [
    ["motion", motion],
    ["structural", structural],
    ["semantic", semantic],
    ["ts", ts],
  ]) {
    if (!isFiniteNumber(v)) {
      throw new ContractError(`makeTier0Signal: ${k} must be a finite number`);
    }
  }
  return { motion, structural, semantic, ts };
}

/**
 * Parse + validate a Tier0Signal from the wire (string or object). Returns a
 * fresh, normalized Tier0Signal. The Python side consumes exactly this shape.
 * @param {string|object} json
 * @throws {ContractError}
 */
export function parseTier0Signal(json) {
  let obj = json;
  if (typeof json === "string") {
    try {
      obj = JSON.parse(json);
    } catch (e) {
      throw new ContractError(`parseTier0Signal: invalid JSON (${e.message})`);
    }
  }
  if (!isPlainObject(obj)) {
    throw new ContractError("parseTier0Signal: expected an object");
  }
  for (const k of ["motion", "structural", "semantic", "ts"]) {
    if (!isFiniteNumber(obj[k])) {
      throw new ContractError(`parseTier0Signal: ${k} must be a finite number`);
    }
  }
  return {
    motion: obj.motion,
    structural: obj.structural,
    semantic: obj.semantic,
    ts: obj.ts,
  };
}
