// src/contract.js
//
// The FROZEN edge wire-contract. Field names here MUST match the Python SDK
// dataclasses byte-for-byte so the two halves interoperate over JSON:
//
//   Box            = { x, y, w, h }                              # normalized 0..1
//   WatchCondition = { goal_id, anchor_text, threshold, region_hint: Box|null,
//                      detectors: string[] }                     # cheap edge detectors the brain selected
//   WatchSpec      = { conditions: WatchCondition[], version }   # version defaults to 1
//   Tier0Signal    = { motion, structural, semantic, ts }       # per-frame salience GAUGE
//   Boundary       = { detector, kind, t, value }               # a discrete EVENT (a skill fired)
//
// Down channel  (cloud -> edge): WatchSpec
// Up   channel  (edge -> cloud): EdgeCapabilities (at connect) · Tier0Signal (per-frame gauge) ·
//                                BoundaryReport (per-event, when skills fire)
//
// The up-messages are DECOUPLED (Option 2): Tier0Signal is the dense gauge that drives escalate-vs-skip;
// a BoundaryReport carries the sparse events the brain counts (mirrors the harness's per-event
// `perceive_boundary` dispatch); EdgeCapabilities advertises which skills the edge can run so the brain
// selects within them. Tier0Signal stays frozen; boundaries + capabilities ride their own messages.
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

// ---------------------------------------------------------------------------
// Boundary  { detector, kind: "peak"|"onset"|"threshold", t, value }
// The discrete up-channel EVENT a skill emits (mirrors the Python `Boundary` dataclass byte-for-byte).
// The edge tags each with the goal_id it fired for when it sends it up (the envelope), exactly like the
// harness's `(goal_id, Boundary)` pairs feeding `core.perceive_boundary(goal_id, t)`.
// ---------------------------------------------------------------------------

const _BOUNDARY_KINDS = new Set(["peak", "onset", "threshold"]);

/**
 * Build a Boundary. Positional to match the skills' call sites.
 * @returns {{detector:string,kind:string,t:number,value:number}}
 */
export function makeBoundary(detector, kind, t, value) {
  if (typeof detector !== "string") throw new ContractError("makeBoundary: detector must be a string");
  if (!_BOUNDARY_KINDS.has(kind)) {
    throw new ContractError(`makeBoundary: kind must be one of ${[..._BOUNDARY_KINDS].join("|")}`);
  }
  if (!isFiniteNumber(t)) throw new ContractError("makeBoundary: t must be a finite number");
  if (!isFiniteNumber(value)) throw new ContractError("makeBoundary: value must be a finite number");
  return { detector, kind, t, value };
}

/**
 * Validate a Boundary off the wire (string or object). Returns a fresh, normalized Boundary.
 * @throws {ContractError}
 */
export function parseBoundary(json, { path = "boundary" } = {}) {
  let obj = json;
  if (typeof json === "string") {
    try {
      obj = JSON.parse(json);
    } catch (e) {
      throw new ContractError(`parseBoundary: invalid JSON (${e.message})`);
    }
  }
  if (!isPlainObject(obj)) throw new ContractError(`${path}: expected an object`);
  if (typeof obj.detector !== "string") throw new ContractError(`${path}.detector: expected a string`);
  if (!_BOUNDARY_KINDS.has(obj.kind)) {
    throw new ContractError(`${path}.kind: expected one of ${[..._BOUNDARY_KINDS].join("|")}`);
  }
  if (!isFiniteNumber(obj.t)) throw new ContractError(`${path}.t: expected a finite number`);
  if (!isFiniteNumber(obj.value)) throw new ContractError(`${path}.value: expected a finite number`);
  return { detector: obj.detector, kind: obj.kind, t: obj.t, value: obj.value };
}

// ---------------------------------------------------------------------------
// EdgeCapabilities  { capabilities: string[] }     (edge -> brain, at connect)
// The ADVERTISE half of the capability handshake: the edge tells the brain which skills it can run, so
// the brain selects detectors only within this set (Python: compile_watchspec(edge_capabilities=...)).
// ---------------------------------------------------------------------------

export function makeEdgeCapabilities(capabilities = []) {
  if (!isStringArray(capabilities)) {
    throw new ContractError("makeEdgeCapabilities: capabilities must be an array of strings");
  }
  return { capabilities: [...capabilities] };
}

export function parseEdgeCapabilities(json) {
  let obj = json;
  if (typeof json === "string") {
    try {
      obj = JSON.parse(json);
    } catch (e) {
      throw new ContractError(`parseEdgeCapabilities: invalid JSON (${e.message})`);
    }
  }
  if (!isPlainObject(obj)) throw new ContractError("parseEdgeCapabilities: expected an object");
  if (!isStringArray(obj.capabilities)) {
    throw new ContractError("parseEdgeCapabilities: capabilities must be an array of strings");
  }
  return { capabilities: [...obj.capabilities] };
}

// ---------------------------------------------------------------------------
// BoundaryReport  { boundaries: [{ goal_id, detector, kind, t, value }] }   (edge -> brain, per-event)
// The wire framing of the edge's onBoundary batch — each Boundary tagged with the goal_id it fired for.
// The server maps each entry 1:1 to core.perceive_boundary(goal_id, t). Flattened (goal_id + Boundary)
// for a clean wire object.
// ---------------------------------------------------------------------------

export function makeBoundaryReport(events = []) {
  if (!Array.isArray(events)) throw new ContractError("makeBoundaryReport: events must be an array");
  const boundaries = events.map((e, i) => {
    // accept { goal_id, boundary } (the onBoundary shape) or a flat { goal_id, detector, kind, t, value }
    const b = e.boundary ?? e;
    if (typeof e.goal_id !== "string") {
      throw new ContractError(`makeBoundaryReport: events[${i}].goal_id must be a string`);
    }
    const norm = makeBoundary(b.detector, b.kind, b.t, b.value);
    return { goal_id: e.goal_id, ...norm };
  });
  return { boundaries };
}

export function parseBoundaryReport(json) {
  let obj = json;
  if (typeof json === "string") {
    try {
      obj = JSON.parse(json);
    } catch (e) {
      throw new ContractError(`parseBoundaryReport: invalid JSON (${e.message})`);
    }
  }
  if (!isPlainObject(obj) || !Array.isArray(obj.boundaries)) {
    throw new ContractError("parseBoundaryReport: expected { boundaries: [] }");
  }
  const boundaries = obj.boundaries.map((e, i) => {
    if (!isPlainObject(e) || typeof e.goal_id !== "string") {
      throw new ContractError(`parseBoundaryReport: boundaries[${i}].goal_id must be a string`);
    }
    const b = parseBoundary(e, { path: `boundaries[${i}]` });
    return { goal_id: e.goal_id, ...b };
  });
  return { boundaries };
}
