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

// The one finite-number predicate, shared with the gate and the privacy screen (src/numeric.js).
// Aliased so the ~20 call sites below read unchanged. `nonEmptyBox` is the area half of the Box
// contract enforced below.
import { finiteNumber as isFiniteNumber, nonEmptyBox } from "./numeric.js";

/**
 * Semantic version of the whole frozen contract surface — the JS twin of the Python
 * `percept.contracts.CONTRACT_VERSION`. Same major == wire-compatible. This is DISTINCT from
 * `WatchSpec.version` (the wire integer, defaults to 1) which stays for payload compat.
 */
export const CONTRACT_VERSION = "1.10.0"; // JS twin of Python 1.10.0.
// 1.10 adds judge context and provenance to the Python WatchPlan contract. 1.9 added
// Incident.judgment_event_id (the OPAQUE, INJECTIVE handle joining an Incident to the judgment.event
// that produced it — see judgmentEventId()) + Incident.evidence_origin ("judged"|"transition"|null,
// which firing path produced the evidence). Both are ADDITIVE/OPTIONAL, so a pre-1.9 payload parses to
// null unchanged. 1.8 covered WatchPlan/SettingOrigin/ResolvedWatch/WatchReceipt/FieldDiagnostic
// (Python-only config-provenance DTOs) + Incident.revision_id/Incident.supersedes; 1.7
// Incident.evidence_seq + SubjectRef 'provisional' resolution state / provisionalSubject().
// This constant names the PYTHON contract version these shapes are a twin of. The JS side implements a
// SUBSET of the Python DTOs (the shapes defined below); the version tracks the wire contract, not the
// coverage. 1.0.0: initial public wire contract (launch).

/** Error thrown when an incoming wire payload violates the contract. */
export class ContractError extends Error {
  constructor(message) {
    super(message);
    this.name = "ContractError";
  }
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
 * The Box CONTRACT — the three rules `makeBox` and `parseBox` both enforce, in one place because
 * they are the two doors a Box comes through and they are the pair in this file that has already
 * drifted apart once.
 *
 *   1. **finite** — a NaN bound makes every derived pixel index NaN.
 *   2. **normalized** — each of `x,y,w,h` in `[0,1]`. A Box is documented as "normalized 0..1 image
 *      coordinates"; `x: 2` is not a coordinate in that space, it is a unit error, and it selects
 *      nothing on any raster.
 *   3. **positive area** — `w > 0 && h > 0`. A zero-area zone masks the whole frame away, so a goal
 *      carrying one can never escalate on any frame at any threshold, and nothing downstream can
 *      tell it apart from a working watch. That is an authoring mistake to refuse, not an input to
 *      interpret — see `numeric.nonEmptyBox`.
 *
 * DELIBERATELY NOT a rule: `x + w <= 1`. A zone that overhangs the far edge is CLAMPED by the gate
 * (`motion.regionBounds`, `tier0._region_bounds`), which is a defined, useful answer — and a
 * float-exact sum check would reject legitimately-authored full-width zones (`0.7 + 0.3 !== 1`).
 *
 * Rule 3 is a WIRE CHANGE and forks from the Python half until it matches: `percept.contracts.Box`
 * validates nothing and `harness.tier0._coerce_box` builds one straight out of `goal.state`, so a
 * degenerate zone is still authorable server-side. Cross-referenced from FIND-PARITY-01.
 */
function validateBox(box, describe) {
  for (const k of ["x", "y", "w", "h"]) {
    const v = box[k];
    if (!isFiniteNumber(v)) {
      throw new ContractError(describe(`${k} must be a finite number`));
    }
    if (v < 0 || v > 1) {
      throw new ContractError(
        describe(`${k} must be within [0,1] (normalized image coordinates), got ${v}`)
      );
    }
  }
  if (!nonEmptyBox(box)) {
    throw new ContractError(
      describe(
        `w and h must be > 0, got w=${box.w} h=${box.h} — a zero-area box selects no pixels, so a ` +
          "goal carrying one can never fire at any threshold"
      )
    );
  }
}

/**
 * Build a normalized Box. All fields default to 0, which is NOT a valid Box (zero area) — the
 * defaults exist so a partial object reports the field it is missing rather than a shape error.
 *
 * Validates exactly what `parseBox` validates. It is the only make/parse pair in this file that
 * used to disagree, and the asymmetry was not cosmetic: `makeWatchCondition` routes `region_hint`
 * through here, so a NaN box was accepted, stored on the condition, and serialized into a WatchSpec
 * the Python side cannot read back — the failure surfaced a hop away from its cause.
 *
 * @returns {{x:number,y:number,w:number,h:number}}
 * @throws {ContractError}
 */
export function makeBox({ x = 0, y = 0, w = 0, h = 0 } = {}) {
  const box = { x, y, w, h };
  validateBox(box, (why) => `makeBox: ${why}`);
  return box;
}

/**
 * Validate + normalize a Box-shaped value. Returns a fresh Box object.
 * @throws {ContractError}
 */
export function parseBox(obj, { path = "region_hint" } = {}) {
  if (!isPlainObject(obj)) {
    throw new ContractError(`${path}: expected an object, got ${typeof obj}`);
  }
  validateBox(obj, (why) => `${path}: ${why}`);
  return { x: obj.x, y: obj.y, w: obj.w, h: obj.h };
}

// ---------------------------------------------------------------------------
// WatchCondition  { goal_id, anchor_text, threshold, region_hint: Box|null,
//                   detectors: string[], cadence_s: number|null,
//                   gate_on: "motion"|"structural"|"semantic"|"fixed"|"fused" }
// ---------------------------------------------------------------------------

/**
 * Build a WatchCondition. `detectors` are the cheap on-device detector names the
 * brain selected for this goal (e.g. "motion-periodicity", "acoustic-onset");
 * defaults to [] (the Python `field(default_factory=list)` default).
 * @returns {{goal_id:string,anchor_text:string,threshold:number,region_hint:object|null,detectors:string[]}}
 */
// `fused` is the wake-policy noisy-OR over the cheap channels. It is a channel a goal OPTS IN to, not
// a mode that overrides the others: a configured wake policy used to replace the
// structural/semantic/motion branch for every condition and still report the goal's own `gate_on`, so
// a receipt could name a channel whose measured value was 0. Mirrors Python
// `percept.behavior._VALID_GATE_ON` / `percept.contracts.GateOn`. Additive — an edge that never
// receives `gate_on: "fused"` behaves byte-for-byte as before.
export const GATE_ON = new Set(["motion", "structural", "semantic", "fixed", "fused"]);
export const SALIENCE_STAT = new Set(["mean", "block_max", "p95"]);

// ---------------------------------------------------------------------------
// TEMPORAL — the JS twin of Python `percept.behavior.Temporal` (the perception
// evidence-gathering shape a concern's PerceptionSpec.temporal selects). VALUE TOKEN ONLY: this repo
// runs no JS code path that branches on it or assembles evidence from it (that lives Python-side in
// `percept.capsule` for "event_capsule") — it is exported purely so a JS-side author/validator of
// `goal.state.perception.temporal` can check against the SAME closed vocabulary as the Python side,
// without re-deriving it. Not part of any frozen wire shape above (WatchSpec/Tier0Signal/Boundary/
// Incident carry no `temporal` field), so this is additive and does not move CONTRACT_VERSION.
// `event_capsule` composes pre-roll + episode-boundary + an early-snapshot-timer frame, assembled
// Python-side; `sprite` is deprecated (normalized to `window` — see percept/agent.py).
// ---------------------------------------------------------------------------
export const TEMPORAL = new Set(["single", "strip", "window", "sprite", "event_capsule"]);

export function makeWatchCondition({
  goal_id,
  anchor_text,
  threshold = 0.3,
  region_hint = null,
  detectors = [],
  cadence_s = null,
  gate_on = "motion",
  salience_stat = "mean",
  ego_compensate = false,
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
  if (!SALIENCE_STAT.has(salience_stat)) {
    throw new ContractError(
      `makeWatchCondition: salience_stat must be one of ${[...SALIENCE_STAT].join("|")}`
    );
  }
  if (typeof ego_compensate !== "boolean") {
    throw new ContractError("makeWatchCondition: ego_compensate must be a boolean");
  }
  return {
    goal_id,
    anchor_text,
    threshold,
    region_hint: region_hint == null ? null : makeBox(region_hint),
    detectors: [...detectors],
    cadence_s: cadence_s == null ? null : cadence_s,
    gate_on,
    salience_stat,
    ego_compensate,
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
  // cadence: how often to look. Missing tolerated -> (null, "motion"), the default.
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
  // salience_stat / ego_compensate (the localized-recall + ego-motion modes). Missing tolerated ->
  // "mean" / false (the legacy path) so a Python side that omits them still parses.
  let salience_stat = "mean";
  if (obj.salience_stat != null) {
    if (!SALIENCE_STAT.has(obj.salience_stat)) {
      throw new ContractError(`${path}.salience_stat: expected one of ${[...SALIENCE_STAT].join("|")}`);
    }
    salience_stat = obj.salience_stat;
  }
  let ego_compensate = false;
  if (obj.ego_compensate != null) {
    if (typeof obj.ego_compensate !== "boolean") {
      throw new ContractError(`${path}.ego_compensate: expected a boolean`);
    }
    ego_compensate = obj.ego_compensate;
  }
  // NOTE: `schedule` (present on the Python side) is not mirrored here yet — it is tolerantly dropped
  // (a known parity drift, tracked separately). The on-device gate executes neither schedule nor cadence today.
  return {
    goal_id: obj.goal_id,
    anchor_text: obj.anchor_text,
    threshold: obj.threshold,
    region_hint,
    detectors,
    cadence_s,
    gate_on,
    salience_stat,
    ego_compensate,
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
// Tier0Signal  { motion, structural, semantic, ts, energy? }
// ---------------------------------------------------------------------------

/**
 * Build a Tier0Signal (the up-channel salience report). Unsupplied salience
 * channels default to 0; ts defaults to wall-clock seconds (to match the
 * Python `ts: float` seconds convention). `energy` (1.3) is the audio-energy
 * channel from a frame-free audio ingest — null means "not an audio-carrying
 * signal" (distinct from 0 = measured silence), so frame-path signals are
 * byte-for-byte unchanged.
 * @returns {{motion:number,structural:number,semantic:number,ts:number,energy:number|null}}
 */
export function makeTier0Signal({
  motion = 0,
  structural = 0,
  semantic = 0,
  ts = Date.now() / 1000,
  energy = null,
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
  if (energy !== null && !isFiniteNumber(energy)) {
    throw new ContractError("makeTier0Signal: energy must be a finite number or null");
  }
  return { motion, structural, semantic, ts, energy };
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
  // energy (1.3) is optional on the wire: absent/undefined normalizes to null so a 1.2-era
  // payload parses unchanged; when present it must be a finite number or null.
  let energy = null;
  if (obj.energy !== undefined && obj.energy !== null) {
    if (!isFiniteNumber(obj.energy)) {
      throw new ContractError("parseTier0Signal: energy must be a finite number or null");
    }
    energy = obj.energy;
  }
  return {
    motion: obj.motion,
    structural: obj.structural,
    semantic: obj.semantic,
    ts: obj.ts,
    energy,
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
//             ts, label: string|null, snippet: IncidentSnippet|null, entity_id: string|null,
//             reason: string|null, watchplan_id: string|null, roi: Box|null,
//             modality: "vision"|"audio"|"fused"|null, provenance: "observed"|"inferred"|null,
//             evidence_seq: number|null, revision_id: string|null, supersedes: string|null }
// IncidentSnippet  { kind: "frame"|"clip"|"audio", media: string|null (base64),
//                    mime: string|null, frames: array|null, t_start: number|null, t_end: number|null }
//
// The JS twin of the Python `percept.contracts.Incident` / `IncidentSnippet`: the durable record of
// one concern firing, carrying an optional captured relevant snippet (e.g. a baby-monitor "baby awake"
// fires an Incident whose snippet is a short clip). `media` is Python `bytes` -> a base64 string on the
// wire (or null); `verdict` is the firing Verdict { satisfied, confidence, evidence }.
//
// `entity_id` (contract 1.3) is the GENERIC correlation handle of the gated subject — the same id
// that keys the rising-edge gate `(goal_id, entity_id)` on the Python side. It lets a consumer tell two
// subjects tripping the same concern apart instead of reading their interleaved fires as out-of-order
// divergence. It is a correlation key only — no claim about who/which document the subject is. `null`
// when the fire was not entity-scoped; a pre-1.3 payload omitting it parses to `null` unchanged.
//
// The enriched judgment envelope (contract 1.6) rides through the emission boundary onto the
// durable record a client reads at 2am: `reason` is the gate's reason code (why the action was
// taken), `watchplan_id` a stable id of the plan it ran under, `roi` the region it fired on (or null,
// left honest rather than echoing an input hint), `modality` how the eye looked, `provenance` observed
// vs inferred. All ADDITIVE and OPTIONAL — a pre-1.6 payload omitting them parses to `null` unchanged.
//
// `evidence_seq` (contract 1.7) is the per-key ADR-001 §8 dispatch sequence the fire's judgment
// carried on the Python side (`JudgmentEvent.evidence_seq`), copied through so a consumer can
// correlate this Incident back to the judgment record that produced it. ADDITIVE and OPTIONAL — a
// pre-1.7 payload omitting it parses to `null` unchanged.
//
// `revision_id`/`supersedes` (contract 1.8) are the episode-owned provisional/superseding link: a
// claim-mode correction (a later turn classified replace/retract on the same episode) fires a
// SUPERSEDING revision. `revision_id` is THIS incident's stable episode-revision handle
// (`<episode_id>#r<n>`); `supersedes` names the PRIOR revision this one corrects and is `null` for an
// original/provisional incident (the common case). Both ADDITIVE and OPTIONAL — a pre-1.8 payload
// omitting them parses to `null` unchanged.
//
// `judgment_event_id`/`evidence_origin` (contract 1.9) are the durable EVIDENCE LINK a client audit
// resolves against. `judgment_event_id` is the OPAQUE, INJECTIVE handle (see `judgmentEventId`) a
// `judgment.event` trace row ALSO carries, so a delivered Incident joins back to EXACTLY ONE judgment
// record — equality-compare it, never parse it. `evidence_origin` is "judged" | "transition" | null:
// which firing path produced the evidence (the rising-edge watch path vs the A→B transition path, which
// from 1.9 on also emits a judgment record). Both ADDITIVE and OPTIONAL — a pre-1.9 payload omitting
// them parses to `null` unchanged; a `null` origin is read as "judged".
// ---------------------------------------------------------------------------

const _SNIPPET_KINDS = new Set(["frame", "clip", "audio"]);
const _EVIDENCE_MODALITIES = new Set(["vision", "audio", "fused"]);
const _JUDGMENT_PROVENANCES = new Set(["observed", "inferred"]);
// contract 1.9 — the CLOSED EvidenceOrigin vocabulary: which firing path produced a judgment's
// evidence. `judged` = the rising-edge watch path; `transition` = the A→B path (two judge calls). A
// `null`/absent origin is LEGACY and read as "judged".
const _EVIDENCE_ORIGINS = new Set(["judged", "transition"]);

/**
 * The OPAQUE, INJECTIVE handle joining an Incident to the `judgment.event` that produced it —
 * `"{goal}#{entity}#{seq}"`, each component percent-escaped before joining. The IDENTICAL formula as
 * Python `percept.contracts.judgment_event_id`: `%` → `%25` FIRST, then `#` → `%23`, applied IN EACH
 * COMPONENT, so distinct `(goalId, entityId, dispatchSeq)` triples always map to distinct ids even when
 * a raw id itself contains a `#` or `%`. `entityId` of `null`/`undefined` is treated as the empty string
 * (a scene-scoped / non-entity fire). OPAQUE to consumers: equality-compare it, never parse it.
 * @returns {string}
 */
export function judgmentEventId(goalId, entityId, dispatchSeq) {
  const esc = (part) => String(part).replace(/%/g, "%25").replace(/#/g, "%23");
  const goal = esc(goalId);
  const entity = esc(entityId == null ? "" : entityId);
  const seq = esc(dispatchSeq);
  return `${goal}#${entity}#${seq}`;
}

function _isStringOrNull(v) {
  return v == null || typeof v === "string";
}

function _isNumberOrNull(v) {
  return v == null || isFiniteNumber(v);
}

// The claim-mode / speech-act channel (contract 1.5). A value DIFFERENCE is not a
// REPLACEMENT: the verdict names the LOCAL speech act so the executor never parses values out of
// free-text evidence. All five fields are ADDITIVE and OPTIONAL — a pre-1.5 verdict omits them and
// keeps its exact { satisfied, confidence, evidence } shape (see _makeVerdict).
const _OPERATIONS = new Set(["assert", "confirm", "replace", "retract", "close", "mention"]);

/**
 * Build a Verdict { satisfied, confidence, evidence } plus the OPTIONAL claim-mode fields
 * when the payload carries them. evidence defaults to "" (Python default). The mutation fields are
 * only attached when PRESENT, so a plain positive verdict round-trips as the pre-1.5 three-field shape.
 */
function _makeVerdict(v, path) {
  if (!isPlainObject(v)) throw new ContractError(`${path}: expected an object`);
  if (typeof v.satisfied !== "boolean") throw new ContractError(`${path}.satisfied: expected a boolean`);
  if (!isFiniteNumber(v.confidence)) throw new ContractError(`${path}.confidence: expected a finite number`);
  const evidence = v.evidence == null ? "" : v.evidence;
  if (typeof evidence !== "string") throw new ContractError(`${path}.evidence: expected a string`);
  const out = { satisfied: v.satisfied, confidence: v.confidence, evidence };
  // -- additive claim-mode channel (only when present) --
  if (v.operation != null) {
    if (!_OPERATIONS.has(v.operation)) {
      throw new ContractError(`${path}.operation: expected one of ${[..._OPERATIONS].join("|")}`);
    }
    out.operation = v.operation;
  }
  if (v.value != null) out.value = v.value;
  if (v.normalized_value != null) out.normalized_value = v.normalized_value;
  if (v.target_revision_id != null) {
    if (typeof v.target_revision_id !== "string") {
      throw new ContractError(`${path}.target_revision_id: expected a string or null`);
    }
    out.target_revision_id = v.target_revision_id;
  }
  if (v.asserted_by != null) {
    if (typeof v.asserted_by !== "string") {
      throw new ContractError(`${path}.asserted_by: expected a string or null`);
    }
    out.asserted_by = v.asserted_by;
  }
  // -- additive subject carrier (only when present): WHICH subject the fire is about.
  // Absence is `null`/missing (Python `Verdict.subject: SubjectRef | None = None`), never a third state.
  if (v.subject != null) out.subject = makeSubjectRef(v.subject);
  return out;
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
 * Normalize the ADDITIVE judgment envelope (contract 1.6) plus the contract-1.7 `evidence_seq`
 * link and the contract-1.8 `revision_id`/`supersedes` supersession link, shared by
 * makeIncident / parseIncident. Every field is optional and defaults null, so a pre-1.6/1.7/1.8
 * payload round-trips its exact shape. `roi` is a Box (validated via makeBox/parseBox),
 * `evidence_seq` a finite number or null, `revision_id`/`supersedes` strings or null, the rest
 * closed-vocab strings or null.
 */
function _incidentEnvelope(src, ctx) {
  const reason = src.reason == null ? null : src.reason;
  if (!_isStringOrNull(reason)) throw new ContractError(`${ctx}: reason must be a string or null`);
  const watchplan_id = src.watchplan_id == null ? null : src.watchplan_id;
  if (!_isStringOrNull(watchplan_id)) throw new ContractError(`${ctx}: watchplan_id must be a string or null`);
  const roi = src.roi == null ? null : (ctx.startsWith("parse") ? parseBox(src.roi, { path: `${ctx}.roi` }) : makeBox(src.roi));
  const modality = src.modality == null ? null : src.modality;
  if (modality != null && !_EVIDENCE_MODALITIES.has(modality)) {
    throw new ContractError(`${ctx}: modality must be one of ${[..._EVIDENCE_MODALITIES].join("|")} or null`);
  }
  const provenance = src.provenance == null ? null : src.provenance;
  if (provenance != null && !_JUDGMENT_PROVENANCES.has(provenance)) {
    throw new ContractError(`${ctx}: provenance must be one of ${[..._JUDGMENT_PROVENANCES].join("|")} or null`);
  }
  const evidence_seq = src.evidence_seq == null ? null : src.evidence_seq;
  if (!_isNumberOrNull(evidence_seq)) throw new ContractError(`${ctx}: evidence_seq must be a finite number or null`);
  const revision_id = src.revision_id == null ? null : src.revision_id;
  if (!_isStringOrNull(revision_id)) throw new ContractError(`${ctx}: revision_id must be a string or null`);
  const supersedes = src.supersedes == null ? null : src.supersedes;
  if (!_isStringOrNull(supersedes)) throw new ContractError(`${ctx}: supersedes must be a string or null`);
  // contract 1.9 — the durable evidence link: an OPAQUE handle (string|null) + a closed-vocab origin.
  const judgment_event_id = src.judgment_event_id == null ? null : src.judgment_event_id;
  if (!_isStringOrNull(judgment_event_id)) {
    throw new ContractError(`${ctx}: judgment_event_id must be a string or null`);
  }
  const evidence_origin = src.evidence_origin == null ? null : src.evidence_origin;
  if (evidence_origin != null && !_EVIDENCE_ORIGINS.has(evidence_origin)) {
    throw new ContractError(`${ctx}: evidence_origin must be one of ${[..._EVIDENCE_ORIGINS].join("|")} or null`);
  }
  return {
    reason, watchplan_id, roi, modality, provenance, evidence_seq, revision_id, supersedes,
    judgment_event_id, evidence_origin,
  };
}

/**
 * Build an Incident. `verdict` is { satisfied, confidence, evidence? }; `label` defaults to null
 * (unlabelled — suggested vocab "real"/"false_alarm"); `snippet` defaults to null; `entity_id`
 * defaults to null (not entity-scoped) — the back-compatible default for pre-1.3 callers. The judgment
 * envelope (`reason`/`watchplan_id`/`roi`/`modality`/`provenance`, contract 1.6), `evidence_seq`
 * (contract 1.7), and `revision_id`/`supersedes` (contract 1.8) all default null.
 * @returns {{id:string,concern_id:string,verdict:object,ts:number,label:string|null,snippet:object|null,entity_id:string|null,reason:string|null,watchplan_id:string|null,roi:object|null,modality:string|null,provenance:string|null,evidence_seq:number|null,revision_id:string|null,supersedes:string|null}}
 */
export function makeIncident({
  id,
  concern_id,
  verdict,
  ts,
  label = null,
  snippet = null,
  entity_id = null,
  message = "",
  label_url = null,
  reason = null,
  watchplan_id = null,
  roi = null,
  modality = null,
  provenance = null,
  evidence_seq = null,
  revision_id = null,
  supersedes = null,
  judgment_event_id = null,
  evidence_origin = null,
} = {}) {
  if (typeof id !== "string") throw new ContractError("makeIncident: id must be a string");
  if (typeof concern_id !== "string") throw new ContractError("makeIncident: concern_id must be a string");
  if (!isFiniteNumber(ts)) throw new ContractError("makeIncident: ts must be a finite number");
  if (!_isStringOrNull(label)) throw new ContractError("makeIncident: label must be a string or null");
  if (!_isStringOrNull(entity_id)) throw new ContractError("makeIncident: entity_id must be a string or null");
  // message is the human-facing alert line (Python `str`, default ""); label_url is str|None.
  if (typeof message !== "string") throw new ContractError("makeIncident: message must be a string");
  if (!_isStringOrNull(label_url)) throw new ContractError("makeIncident: label_url must be a string or null");
  return {
    id,
    concern_id,
    verdict: _makeVerdict(verdict, "makeIncident.verdict"),
    ts,
    label,
    snippet: snippet == null ? null : makeIncidentSnippet(snippet),
    entity_id,
    message,
    label_url,
    ..._incidentEnvelope(
      {
        reason, watchplan_id, roi, modality, provenance, evidence_seq, revision_id, supersedes,
        judgment_event_id, evidence_origin,
      },
      "makeIncident",
    ),
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
  // Missing on a pre-1.3 payload -> null (the Python `None` default); never a parse failure.
  const entity_id = obj.entity_id == null ? null : obj.entity_id;
  if (!_isStringOrNull(entity_id)) throw new ContractError("parseIncident: entity_id must be a string or null");
  // message missing -> "" (Python `str` default); label_url missing -> null. Never a parse failure.
  const message = obj.message == null ? "" : obj.message;
  if (typeof message !== "string") throw new ContractError("parseIncident: message must be a string");
  const label_url = obj.label_url == null ? null : obj.label_url;
  if (!_isStringOrNull(label_url)) throw new ContractError("parseIncident: label_url must be a string or null");
  return {
    id: obj.id,
    concern_id: obj.concern_id,
    verdict: _makeVerdict(obj.verdict, "parseIncident.verdict"),
    ts: obj.ts,
    label,
    snippet,
    entity_id,
    message,
    label_url,
    // Missing on a pre-1.6 payload -> null (the Python `None` defaults); never a parse failure.
    ..._incidentEnvelope(obj, "parseIncident"),
  };
}

// ---------------------------------------------------------------------------
// SubjectRef  { scheme, namespace, key: string|null, resolution: "bound"|"unknown"|"provisional",
//               scope: "session"|"source"|"durable", strategy, confidence: number|null }
// ---------------------------------------------------------------------------
//
// The JS twin of the Python `percept.contracts.SubjectRef` (contract 1.4, +'provisional' in 1.7):
// a namespaced, tri-state reference to WHICH subject a judgment is about. `PO-4471` is not globally
// unique across systems, so `namespace` disambiguates the `key`. THREE `resolution` states: "bound"
// (we know which subject this is), "unknown" (seen but unidentified, no key at all), or "provisional"
// (seen, could not identify, but stayed in view long enough within ONE episode to earn a
// SESSION-LOCAL handle — the key is always `UN_<episode_id>` (`PROVISIONAL_KEY_PREFIX`), minted by
// the episode log itself, never by a judge and never from pixels; use `provisionalSubject(episode_id)`
// rather than constructing this shape by hand). The ABSENCE of a subject is `null` on the carrier
// (e.g. `verdict.subject`), NOT a fourth resolution value.
//
// Invariants mirrored byte-for-byte with the Python `__post_init__`:
//   - "bound"       REQUIRES a non-empty namespace AND a non-empty key.
//   - "unknown"     REQUIRES key === null (an unresolved subject must not carry a key).
//   - "provisional" REQUIRES a key of the form `UN_<episode_id>` (`PROVISIONAL_KEY_PREFIX` + episode
//     id) AND scope === "session" — a session-local handle claiming source/durable stability would be
//     exactly the re-identification record percept refuses to produce.
//   - `scope: "durable"` documents IDENTIFIER stability ONLY — NOT a claim that episode state is
//     persisted (percept episode state is session-only).
//   - `confidence` is NOT defaulted to 1.0 (a judge-emitted key is not certain by construction); it
//     is a number in [0,1] or null ("unstated").
// ---------------------------------------------------------------------------

/**
 * The prefix a 'provisional' SubjectRef key always carries — `UN_<episode_id>`. Named so nothing
 * constructs the shape by hand: the key is always "this run's episode id, prefixed", never a value
 * that could be mistaken for a durable/biometric identifier. The JS twin of Python's
 * `percept.contracts.PROVISIONAL_KEY_PREFIX`.
 */
export const PROVISIONAL_KEY_PREFIX = "UN_";

const _SUBJECT_RESOLUTIONS = new Set(["bound", "unknown", "provisional"]);
const _SUBJECT_SCOPES = new Set(["session", "source", "durable"]);

function _validateSubjectRef(scheme, namespace, key, resolution, scope, strategy, confidence, ctx) {
  if (typeof scheme !== "string") throw new ContractError(`${ctx}: scheme must be a string`);
  if (typeof namespace !== "string") throw new ContractError(`${ctx}: namespace must be a string`);
  if (typeof strategy !== "string") throw new ContractError(`${ctx}: strategy must be a string`);
  if (!_SUBJECT_RESOLUTIONS.has(resolution)) {
    throw new ContractError(`${ctx}: resolution must be one of ${[..._SUBJECT_RESOLUTIONS].join("|")}`);
  }
  if (!_SUBJECT_SCOPES.has(scope)) {
    throw new ContractError(`${ctx}: scope must be one of ${[..._SUBJECT_SCOPES].join("|")}`);
  }
  if (resolution === "bound") {
    if (!namespace) throw new ContractError(`${ctx}: 'bound' requires a non-empty namespace`);
    if (typeof key !== "string" || !key) throw new ContractError(`${ctx}: 'bound' requires a non-empty key`);
  } else if (resolution === "provisional") {
    if (typeof key !== "string" || !key || !key.startsWith(PROVISIONAL_KEY_PREFIX)) {
      throw new ContractError(
        `${ctx}: 'provisional' requires a key of the form ${PROVISIONAL_KEY_PREFIX}<episode id> ` +
          `(got ${JSON.stringify(key)}). A provisional handle is minted by the episode that observed ` +
          "the sighting, never invented ad hoc — use provisionalSubject(episode_id)."
      );
    }
    if (scope !== "session") {
      throw new ContractError(
        `${ctx}: 'provisional' requires scope === "session" (got ${JSON.stringify(scope)}). A ` +
          "session-local handle claiming source/durable stability would be exactly the " +
          "re-identification record percept refuses to produce."
      );
    }
  } else {
    // "unknown"
    if (key !== null && key !== undefined) {
      throw new ContractError(`${ctx}: 'unknown' requires key === null (absence of a subject is null)`);
    }
  }
  if (confidence !== null && confidence !== undefined) {
    if (!isFiniteNumber(confidence) || confidence < 0 || confidence > 1) {
      throw new ContractError(`${ctx}: confidence must be a number in [0,1] or null`);
    }
  }
}

/**
 * Build a SubjectRef. `confidence` defaults to null (NOT 1.0 — a judge-emitted key is not certain).
 * `resolution` is "bound" | "unknown" | "provisional" — for "provisional", prefer
 * `provisionalSubject(episode_id)` rather than constructing this shape by hand.
 * @returns {{scheme:string,namespace:string,key:string|null,resolution:string,scope:string,strategy:string,confidence:number|null}}
 * @throws {ContractError}
 */
export function makeSubjectRef({
  scheme,
  namespace,
  key = null,
  resolution,
  scope,
  strategy,
  confidence = null,
} = {}) {
  _validateSubjectRef(scheme, namespace, key, resolution, scope, strategy, confidence, "makeSubjectRef");
  return {
    scheme,
    namespace,
    key: resolution === "unknown" ? null : key,
    resolution,
    scope,
    strategy,
    confidence: confidence == null ? null : confidence,
  };
}

/**
 * Build the 'provisional' SubjectRef for a session-local, unidentified sighting — the JS twin of
 * Python's `percept.contracts.provisional_subject(episode_id)`. The ONLY door a provisional identity
 * comes through: an episode log observed a subject it could not identify and gave it a handle keyed
 * to the episode itself (`UN_<episode_id>`), never to pixels or any cross-session signal. Two
 * sightings only ever share this ref when they are the SAME episode.
 * @returns {{scheme:string,namespace:string,key:string,resolution:"provisional",scope:"session",strategy:string,confidence:null}}
 * @throws {ContractError}
 */
export function provisionalSubject(
  episode_id,
  { scheme = "episode", namespace = "", strategy = "episode_scoped" } = {}
) {
  if (typeof episode_id !== "string" || !episode_id.trim()) {
    throw new ContractError("provisionalSubject: episode_id must be a non-empty string");
  }
  return makeSubjectRef({
    scheme,
    namespace,
    key: `${PROVISIONAL_KEY_PREFIX}${episode_id}`,
    resolution: "provisional",
    scope: "session",
    strategy,
  });
}

/**
 * Parse + validate a SubjectRef off the wire (string or object). Returns a fresh, normalized object.
 * @param {string|object} json
 * @throws {ContractError}
 */
export function parseSubjectRef(json, { path = "subject" } = {}) {
  let obj = json;
  if (typeof json === "string") {
    try {
      obj = JSON.parse(json);
    } catch (e) {
      throw new ContractError(`parseSubjectRef: invalid JSON (${e.message})`);
    }
  }
  if (!isPlainObject(obj)) throw new ContractError(`${path}: expected an object`);
  const key = obj.key == null ? null : obj.key;
  const confidence = obj.confidence == null ? null : obj.confidence;
  _validateSubjectRef(obj.scheme, obj.namespace, key, obj.resolution, obj.scope, obj.strategy, confidence, path);
  return {
    scheme: obj.scheme,
    namespace: obj.namespace,
    key: obj.resolution === "unknown" ? null : key,
    resolution: obj.resolution,
    scope: obj.scope,
    strategy: obj.strategy,
    confidence,
  };
}
