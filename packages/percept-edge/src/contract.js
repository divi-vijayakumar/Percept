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

/**
 * Semantic version of the whole frozen contract surface — the JS twin of the Python
 * `percept.contracts.CONTRACT_VERSION`. Same major == wire-compatible. This is DISTINCT from
 * `WatchSpec.version` (the wire integer, defaults to 1) which stays for payload compat.
 */
export const CONTRACT_VERSION = "1.1.0";

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
//                   detectors: string[], cadence_s: number|null,
//                   gate_on: "motion"|"structural"|"semantic"|"fixed" }
// ---------------------------------------------------------------------------

/**
 * Build a WatchCondition. `detectors` are the cheap on-device detector names the
 * brain selected for this goal (e.g. "motion-periodicity", "acoustic-onset");
 * defaults to [] (the Python `field(default_factory=list)` default).
 * @returns {{goal_id:string,anchor_text:string,threshold:number,region_hint:object|null,detectors:string[]}}
 */
export const GATE_ON = new Set(["motion", "structural", "semantic", "fixed"]);

export function makeWatchCondition({
  goal_id,
  anchor_text,
  threshold = 0.3,
  region_hint = null,
  detectors = [],
  cadence_s = null,
  gate_on = "motion",
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
  if (cadence_s != null && !isFiniteNumber(cadence_s)) {
    throw new ContractError("makeWatchCondition: cadence_s must be a finite number or null");
  }
  if (!GATE_ON.has(gate_on)) {
    throw new ContractError(`makeWatchCondition: gate_on must be one of ${[...GATE_ON].join("|")}`);
  }
  return {
    goal_id,
    anchor_text,
    threshold,
    region_hint: region_hint == null ? null : makeBox(region_hint),
    detectors: [...detectors],
    cadence_s: cadence_s == null ? null : cadence_s,
    gate_on,
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
  // cadence (C2): how often to look. Missing tolerated -> (null, "motion"), the default.
  let cadence_s = null;
  if (obj.cadence_s != null) {
    if (!isFiniteNumber(obj.cadence_s)) {
      throw new ContractError(`${path}.cadence_s: expected a finite number or null`);
    }
    cadence_s = obj.cadence_s;
  }
  let gate_on = "motion";
  if (obj.gate_on != null) {
    if (!GATE_ON.has(obj.gate_on)) {
      throw new ContractError(`${path}.gate_on: expected one of ${[...GATE_ON].join("|")}`);
    }
    gate_on = obj.gate_on;
  }
  // NOTE: `schedule` (Python C1) is not mirrored here yet — it is tolerantly dropped (a known parity
  // drift, tracked separately). The on-device gate executes neither schedule nor cadence today.
  return {
    goal_id: obj.goal_id,
    anchor_text: obj.anchor_text,
    threshold: obj.threshold,
    region_hint,
    detectors,
    cadence_s,
    gate_on,
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

// ---------------------------------------------------------------------------
// Incident  { id, concern_id, verdict: { satisfied, confidence, evidence },
//             ts, label: string|null, snippet: IncidentSnippet|null }
// IncidentSnippet  { kind: "frame"|"clip"|"audio", media: string|null (base64),
//                    mime: string|null, frames: array|null, t_start: number|null, t_end: number|null }
//
// The JS twin of the Python `percept.contracts.Incident` / `IncidentSnippet`: the durable record of
// one concern firing, carrying an optional captured relevant snippet (e.g. a baby-monitor "baby awake"
// fires an Incident whose snippet is a short clip). `media` is Python `bytes` -> a base64 string on the
// wire (or null); `verdict` is the firing Verdict { satisfied, confidence, evidence }.
// ---------------------------------------------------------------------------

const _SNIPPET_KINDS = new Set(["frame", "clip", "audio"]);

function _isStringOrNull(v) {
  return v == null || typeof v === "string";
}

function _isNumberOrNull(v) {
  return v == null || isFiniteNumber(v);
}

/** Build a Verdict { satisfied, confidence, evidence }. evidence defaults to "" (Python default). */
function _makeVerdict(v, path) {
  if (!isPlainObject(v)) throw new ContractError(`${path}: expected an object`);
  if (typeof v.satisfied !== "boolean") throw new ContractError(`${path}.satisfied: expected a boolean`);
  if (!isFiniteNumber(v.confidence)) throw new ContractError(`${path}.confidence: expected a finite number`);
  const evidence = v.evidence == null ? "" : v.evidence;
  if (typeof evidence !== "string") throw new ContractError(`${path}.evidence: expected a string`);
  return { satisfied: v.satisfied, confidence: v.confidence, evidence };
}

/**
 * Build an IncidentSnippet. `kind` is required; every payload field is optional (defaults null) so
 * the same shape serves a full capture and a redacted record. `frames` is an opaque array.
 * @returns {{kind:string,media:string|null,mime:string|null,frames:Array|null,t_start:number|null,t_end:number|null}}
 */
export function makeIncidentSnippet({
  kind,
  media = null,
  mime = null,
  frames = null,
  t_start = null,
  t_end = null,
} = {}) {
  if (!_SNIPPET_KINDS.has(kind)) {
    throw new ContractError(`makeIncidentSnippet: kind must be one of ${[..._SNIPPET_KINDS].join("|")}`);
  }
  if (!_isStringOrNull(media)) throw new ContractError("makeIncidentSnippet: media must be a base64 string or null");
  if (!_isStringOrNull(mime)) throw new ContractError("makeIncidentSnippet: mime must be a string or null");
  if (frames != null && !Array.isArray(frames)) {
    throw new ContractError("makeIncidentSnippet: frames must be an array or null");
  }
  if (!_isNumberOrNull(t_start)) throw new ContractError("makeIncidentSnippet: t_start must be a finite number or null");
  if (!_isNumberOrNull(t_end)) throw new ContractError("makeIncidentSnippet: t_end must be a finite number or null");
  return {
    kind,
    media,
    mime,
    frames: frames == null ? null : [...frames],
    t_start,
    t_end,
  };
}

/**
 * Validate an IncidentSnippet off the wire. Returns a fresh, normalized object.
 * @throws {ContractError}
 */
export function parseIncidentSnippet(obj, { path = "snippet" } = {}) {
  if (!isPlainObject(obj)) throw new ContractError(`${path}: expected an object`);
  if (!_SNIPPET_KINDS.has(obj.kind)) {
    throw new ContractError(`${path}.kind: expected one of ${[..._SNIPPET_KINDS].join("|")}`);
  }
  // Missing payload fields are tolerated -> null (Python `None` defaults).
  const media = obj.media == null ? null : obj.media;
  if (!_isStringOrNull(media)) throw new ContractError(`${path}.media: expected a base64 string or null`);
  const mime = obj.mime == null ? null : obj.mime;
  if (!_isStringOrNull(mime)) throw new ContractError(`${path}.mime: expected a string or null`);
  let frames = null;
  if (obj.frames != null) {
    if (!Array.isArray(obj.frames)) throw new ContractError(`${path}.frames: expected an array or null`);
    frames = [...obj.frames];
  }
  const t_start = obj.t_start == null ? null : obj.t_start;
  if (!_isNumberOrNull(t_start)) throw new ContractError(`${path}.t_start: expected a finite number or null`);
  const t_end = obj.t_end == null ? null : obj.t_end;
  if (!_isNumberOrNull(t_end)) throw new ContractError(`${path}.t_end: expected a finite number or null`);
  return { kind: obj.kind, media, mime, frames, t_start, t_end };
}

/**
 * Build an Incident. `verdict` is { satisfied, confidence, evidence? }; `label` defaults to null
 * (unlabelled — suggested vocab "real"/"false_alarm"); `snippet` defaults to null.
 * @returns {{id:string,concern_id:string,verdict:object,ts:number,label:string|null,snippet:object|null}}
 */
export function makeIncident({
  id,
  concern_id,
  verdict,
  ts,
  label = null,
  snippet = null,
} = {}) {
  if (typeof id !== "string") throw new ContractError("makeIncident: id must be a string");
  if (typeof concern_id !== "string") throw new ContractError("makeIncident: concern_id must be a string");
  if (!isFiniteNumber(ts)) throw new ContractError("makeIncident: ts must be a finite number");
  if (!_isStringOrNull(label)) throw new ContractError("makeIncident: label must be a string or null");
  return {
    id,
    concern_id,
    verdict: _makeVerdict(verdict, "makeIncident.verdict"),
    ts,
    label,
    snippet: snippet == null ? null : makeIncidentSnippet(snippet),
  };
}

/**
 * Parse + validate an Incident from the wire (string or object). Returns a fresh, normalized object.
 * The Python side emits exactly this shape.
 * @param {string|object} json
 * @throws {ContractError}
 */
export function parseIncident(json) {
  let obj = json;
  if (typeof json === "string") {
    try {
      obj = JSON.parse(json);
    } catch (e) {
      throw new ContractError(`parseIncident: invalid JSON (${e.message})`);
    }
  }
  if (!isPlainObject(obj)) throw new ContractError("parseIncident: expected an object");
  if (typeof obj.id !== "string") throw new ContractError("parseIncident: id must be a string");
  if (typeof obj.concern_id !== "string") throw new ContractError("parseIncident: concern_id must be a string");
  if (!isFiniteNumber(obj.ts)) throw new ContractError("parseIncident: ts must be a finite number");
  const label = obj.label == null ? null : obj.label;
  if (!_isStringOrNull(label)) throw new ContractError("parseIncident: label must be a string or null");
  const snippet = obj.snippet == null ? null : parseIncidentSnippet(obj.snippet, { path: "snippet" });
  return {
    id: obj.id,
    concern_id: obj.concern_id,
    verdict: _makeVerdict(obj.verdict, "parseIncident.verdict"),
    ts: obj.ts,
    label,
    snippet,
  };
}
