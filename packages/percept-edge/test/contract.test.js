import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import {
  CONTRACT_VERSION,
  ContractError,
  makeWatchCondition,
  parseWatchCondition,
  makeWatchSpec,
  parseWatchSpec,
  makeTier0Signal,
  parseTier0Signal,
  makeBoundary,
  parseBoundary,
  makeEdgeCapabilities,
  parseEdgeCapabilities,
  makeBoundaryReport,
  parseBoundaryReport,
  makeBox,
  makeIncident,
  parseIncident,
  makeIncidentSnippet,
  parseIncidentSnippet,
  makeSubjectRef,
  parseSubjectRef,
  provisionalSubject,
  PROVISIONAL_KEY_PREFIX,
  judgmentEventId,
} from "../src/contract.js";

const __dirname = dirname(fileURLToPath(import.meta.url));

// The JSON the Python side emits for a one-condition WatchSpec (default cadence; `schedule` is dropped
// tolerantly by the JS parse — a known, tracked parity gap, not mirrored here yet).
const PY_WATCHSPEC =
  '{"conditions":[{"goal_id":"g1","anchor_text":"the user is drinking coffee","threshold":0.3,"region_hint":null,"detectors":[],"cadence_s":null,"gate_on":"motion"}],"version":1}';

describe("parseWatchSpec — cross-language fixture", () => {
  it("parses the exact JSON the Python side emits", () => {
    const spec = parseWatchSpec(PY_WATCHSPEC);
    expect(spec).toEqual({
      conditions: [
        {
          goal_id: "g1",
          anchor_text: "the user is drinking coffee",
          threshold: 0.3,
          region_hint: null,
          detectors: [],
          cadence_s: null,
          gate_on: "motion",
          salience_stat: "mean",
          ego_compensate: false,
        },
      ],
      version: 1,
    });
  });

  it("parses the same fixture read from disk (byte-for-byte)", () => {
    const raw = readFileSync(
      join(__dirname, "fixtures", "watchspec.python.json"),
      "utf8"
    );
    const spec = parseWatchSpec(raw);
    expect(spec.conditions[0].goal_id).toBe("g1");
    expect(spec.conditions[0].anchor_text).toBe("the user is drinking coffee");
    expect(spec.conditions[0].threshold).toBeCloseTo(0.3, 10);
    expect(spec.conditions[0].region_hint).toBeNull();
    expect(spec.version).toBe(1);
  });

  it("round-trips: parse -> stringify produces an equivalent object", () => {
    const spec = parseWatchSpec(PY_WATCHSPEC);
    const reparsed = parseWatchSpec(JSON.stringify(spec));
    expect(reparsed).toEqual(spec);
    // And the field set is exactly the frozen contract — no extra keys leaked.
    expect(Object.keys(spec).sort()).toEqual(["conditions", "version"]);
    expect(Object.keys(spec.conditions[0]).sort()).toEqual([
      "anchor_text",
      "cadence_s",
      "detectors",
      "ego_compensate",
      "gate_on",
      "goal_id",
      "region_hint",
      "salience_stat",
      "threshold",
    ]);
  });

  it("accepts an already-parsed object, not just a string", () => {
    const obj = JSON.parse(PY_WATCHSPEC);
    expect(parseWatchSpec(obj)).toEqual(parseWatchSpec(PY_WATCHSPEC));
  });

  it("defaults version to 1 when omitted (Python dataclass default)", () => {
    const spec = parseWatchSpec('{"conditions":[]}');
    expect(spec.version).toBe(1);
  });

  it("parses a condition with a non-null region_hint Box", () => {
    const json =
      '{"conditions":[{"goal_id":"g2","anchor_text":"mug on the desk","threshold":0.5,"region_hint":{"x":0.1,"y":0.2,"w":0.3,"h":0.4}}],"version":1}';
    const spec = parseWatchSpec(json);
    expect(spec.conditions[0].region_hint).toEqual({
      x: 0.1,
      y: 0.2,
      w: 0.3,
      h: 0.4,
    });
  });

  it("rejects malformed payloads with a ContractError", () => {
    expect(() => parseWatchSpec("{not json")).toThrow(ContractError);
    expect(() => parseWatchSpec('{"version":1}')).toThrow(ContractError); // no conditions
    expect(() =>
      parseWatchSpec('{"conditions":[{"goal_id":"g1"}],"version":1}')
    ).toThrow(ContractError); // missing anchor_text/threshold
    expect(() =>
      parseWatchSpec(
        '{"conditions":[{"goal_id":"g1","anchor_text":"x","threshold":"hi"}]}'
      )
    ).toThrow(ContractError); // threshold not a number
  });
});

describe("factories produce frozen-contract shapes", () => {
  it("makeWatchCondition", () => {
    const c = makeWatchCondition({
      goal_id: "g1",
      anchor_text: "drinking coffee",
    });
    expect(Object.keys(c).sort()).toEqual([
      "anchor_text",
      "cadence_s",
      "detectors",
      "ego_compensate",
      "gate_on",
      "goal_id",
      "region_hint",
      "salience_stat",
      "threshold",
    ]);
    expect(c.region_hint).toBeNull();
    expect(c.detectors).toEqual([]);
    expect(c.salience_stat).toBe("mean");
    expect(c.ego_compensate).toBe(false);
  });

  it("makeWatchSpec defaults version to 1", () => {
    expect(makeWatchSpec({}).version).toBe(1);
  });

  it("makeBox", () => {
    expect(makeBox({ x: 0.1, y: 0.2, w: 0.3, h: 0.4 })).toEqual({
      x: 0.1,
      y: 0.2,
      w: 0.3,
      h: 0.4,
    });
  });
});

describe("WatchCondition.detectors (the brain's edge-detector selection)", () => {
  it("carries the detectors the Python side selected for a goal", () => {
    const json =
      '{"conditions":[{"goal_id":"reps","anchor_text":"one squat rep","threshold":0.3,"region_hint":null,"detectors":["motion-periodicity","acoustic-onset"]}],"version":1}';
    const spec = parseWatchSpec(json);
    expect(spec.conditions[0].detectors).toEqual([
      "motion-periodicity",
      "acoustic-onset",
    ]);
  });

  it("tolerates an older Python side that omits detectors -> [] (back-compat)", () => {
    const c = parseWatchCondition({
      goal_id: "g1",
      anchor_text: "x",
      threshold: 0.3,
    });
    expect(c.detectors).toEqual([]);
  });

  it("makeWatchCondition accepts + copies detectors", () => {
    const src = ["motion-periodicity"];
    const c = makeWatchCondition({
      goal_id: "g1",
      anchor_text: "x",
      detectors: src,
    });
    expect(c.detectors).toEqual(["motion-periodicity"]);
    src.push("mutated"); // the factory copied, so the stored array is unaffected
    expect(c.detectors).toEqual(["motion-periodicity"]);
  });

  it("rejects non-array / non-string detectors with a ContractError", () => {
    expect(() =>
      makeWatchCondition({ goal_id: "g", anchor_text: "x", detectors: "motion" })
    ).toThrow(ContractError);
    expect(() =>
      parseWatchCondition({
        goal_id: "g",
        anchor_text: "x",
        threshold: 0.3,
        detectors: [1, 2],
      })
    ).toThrow(ContractError);
  });

  it("round-trips detectors through JSON", () => {
    const spec = makeWatchSpec({
      conditions: [
        { goal_id: "g", anchor_text: "x", detectors: ["motion-periodicity"] },
      ],
    });
    expect(parseWatchSpec(JSON.stringify(spec))).toEqual(spec);
  });
});

describe("Tier0Signal contract", () => {
  it("makeTier0Signal has exactly { motion, structural, semantic, ts, energy }", () => {
    const s = makeTier0Signal({
      motion: 0.5,
      structural: 0,
      semantic: 0,
      ts: 123.5,
    });
    expect(Object.keys(s).sort()).toEqual([
      "energy",
      "motion",
      "semantic",
      "structural",
      "ts",
    ]);
    // energy defaults null (1.3): "not an audio-carrying signal", distinct from 0 = silence
    expect(s).toEqual({ motion: 0.5, structural: 0, semantic: 0, ts: 123.5, energy: null });
  });

  it("round-trips through JSON via parseTier0Signal", () => {
    const s = makeTier0Signal({ motion: 0.42, ts: 9.0 });
    expect(parseTier0Signal(JSON.stringify(s))).toEqual(s);
  });

  it("rejects a Tier0Signal missing a channel", () => {
    expect(() =>
      parseTier0Signal('{"motion":0.1,"structural":0,"ts":1.0}')
    ).toThrow(ContractError); // no semantic
  });

  it("energy (1.3) round-trips, defaults null on a 1.2-era payload, rejects junk", () => {
    // an audio-ingest signal carries its energy
    const s = makeTier0Signal({ ts: 5.0, energy: 0.031 });
    expect(parseTier0Signal(JSON.stringify(s)).energy).toBe(0.031);
    // measured silence (0) is preserved, not coerced to null
    expect(makeTier0Signal({ ts: 5.0, energy: 0 }).energy).toBe(0);
    // a pre-1.3 payload (no energy key) parses unchanged -> energy null
    const old = parseTier0Signal('{"motion":0.1,"structural":0,"semantic":0,"ts":1.0}');
    expect(old.energy).toBe(null);
    // junk is rejected on both paths
    expect(() => makeTier0Signal({ ts: 1, energy: "loud" })).toThrow(ContractError);
    expect(() =>
      parseTier0Signal('{"motion":0,"structural":0,"semantic":0,"ts":1,"energy":"loud"}')
    ).toThrow(ContractError);
  });
});

describe("Boundary contract — the per-event up-message (Option 2)", () => {
  it("makeBoundary builds exactly { detector, kind, t, value }", () => {
    const b = makeBoundary("motion-periodicity", "peak", 12.3, 0.9);
    expect(Object.keys(b).sort()).toEqual(["detector", "kind", "t", "value"]);
    expect(b).toEqual({ detector: "motion-periodicity", kind: "peak", t: 12.3, value: 0.9 });
  });

  it("accepts the three boundary kinds and rejects others", () => {
    for (const kind of ["peak", "onset", "threshold"]) {
      expect(makeBoundary("d", kind, 1, 1).kind).toBe(kind);
    }
    expect(() => makeBoundary("d", "wiggle", 1, 1)).toThrow(ContractError);
  });

  it("parseBoundary round-trips through JSON and validates fields", () => {
    const b = makeBoundary("acoustic-onset", "onset", 4.0, 0.5);
    expect(parseBoundary(JSON.stringify(b))).toEqual(b);
    expect(() => parseBoundary('{"detector":"d","kind":"peak","t":"x","value":1}')).toThrow(ContractError);
    expect(() => parseBoundary('{"detector":"d","kind":"nope","t":1,"value":1}')).toThrow(ContractError);
  });
});

describe("topology-B wire messages — EdgeCapabilities + BoundaryReport", () => {
  it("EdgeCapabilities advertises the edge's skill set", () => {
    const msg = makeEdgeCapabilities(["motion-periodicity", "acoustic-onset"]);
    expect(msg).toEqual({ capabilities: ["motion-periodicity", "acoustic-onset"] });
    expect(parseEdgeCapabilities(JSON.stringify(msg))).toEqual(msg);
    expect(() => makeEdgeCapabilities([1, 2])).toThrow(ContractError);
    expect(() => parseEdgeCapabilities('{"capabilities":"motion"}')).toThrow(ContractError);
  });

  it("BoundaryReport frames the onBoundary batch (goal_id + flattened Boundary)", () => {
    // the shape the Edge emits up onBoundary: [{ goal_id, boundary }]
    const batch = [
      { goal_id: "reps", boundary: { detector: "motion-periodicity", kind: "peak", t: 12.3, value: 0.9 } },
    ];
    const report = makeBoundaryReport(batch);
    expect(report).toEqual({
      boundaries: [{ goal_id: "reps", detector: "motion-periodicity", kind: "peak", t: 12.3, value: 0.9 }],
    });
    // round-trips through JSON + validates
    expect(parseBoundaryReport(JSON.stringify(report))).toEqual(report);
  });

  it("BoundaryReport also accepts already-flat events and rejects bad ones", () => {
    const flat = [{ goal_id: "g", detector: "acoustic-onset", kind: "onset", t: 4, value: 0.5 }];
    expect(makeBoundaryReport(flat).boundaries[0].kind).toBe("onset");
    expect(() => makeBoundaryReport([{ detector: "d", kind: "peak", t: 1, value: 1 }])).toThrow(ContractError); // no goal_id
    expect(() => parseBoundaryReport('{"boundaries":[{"goal_id":"g","kind":"nope","detector":"d","t":1,"value":1}]}')).toThrow(ContractError);
  });
});

describe("CONTRACT_VERSION", () => {
  it("is the semver twin of the Python contract version", () => {
    expect(CONTRACT_VERSION).toBe("1.10.0");
  });
});

describe("judgmentEventId — shared golden-vector parity with Python", () => {
  // The SAME fixture the Python-side test_inc1_evidence_links.py::TestGoldenVectorParity asserts.
  // Python is source of truth; the JS twin must match byte-for-byte, including '#'/'%' escaping and
  // the None/empty-entity equivalence.
  const golden = JSON.parse(
    readFileSync(join(__dirname, "fixtures", "judgment_event_id.golden.json"), "utf8"),
  );
  it("matches every shared vector", () => {
    expect(golden.vectors.length).toBeGreaterThanOrEqual(10);
    for (const v of golden.vectors) {
      expect(judgmentEventId(v.goal, v.entity, v.seq)).toBe(v.id);
    }
  });
  it("treats a null/undefined entity as the empty string", () => {
    expect(judgmentEventId("g", null, 3)).toBe(judgmentEventId("g", "", 3));
    expect(judgmentEventId("g", undefined, 3)).toBe("g##3");
  });
  it("stays injective when a component contains '#' or '%'", () => {
    const ids = new Set([
      judgmentEventId("a#b", "c", 1),
      judgmentEventId("a", "#b", 1),
      judgmentEventId("a#", "b", 1),
      judgmentEventId("a%b", "c", 1),
    ]);
    expect(ids.size).toBe(4);
  });
});

describe("Incident / IncidentSnippet", () => {
  it("builds a minimal Incident with null label + snippet", () => {
    const inc = makeIncident({
      id: "inc-1",
      concern_id: "c-baby-awake",
      verdict: { satisfied: true, confidence: 0.9 },
      ts: 123.0,
    });
    expect(inc).toEqual({
      id: "inc-1",
      concern_id: "c-baby-awake",
      verdict: { satisfied: true, confidence: 0.9, evidence: "" },
      ts: 123.0,
      label: null,
      snippet: null,
      entity_id: null,
      message: "", // Incident.message (Python str default "")
      label_url: null, // Incident.label_url (Python str|None)
      // Z3 (contract 1.6) enriched envelope — all default null on a minimal build.
      reason: null,
      watchplan_id: null,
      roi: null,
      modality: null,
      provenance: null,
      evidence_seq: null, // contract 1.7 — defaults null on a minimal build
      revision_id: null, // contract 1.8 — defaults null on a minimal build
      supersedes: null, // contract 1.8 — defaults null on a minimal build
      judgment_event_id: null, // contract 1.9 — defaults null on a minimal build
      evidence_origin: null, // contract 1.9 — defaults null on a minimal build
    });
  });

  it("builds + round-trips an Incident with a clip snippet (the baby-awake example)", () => {
    const inc = makeIncident({
      id: "inc-2",
      concern_id: "c-baby-awake",
      verdict: { satisfied: true, confidence: 0.95, evidence: "eyes open" },
      ts: 200.0,
      label: "real",
      snippet: { kind: "clip", media: "AAFmYWtlTVA0", mime: "video/mp4", t_start: 10.0, t_end: 13.0 },
    });
    expect(inc.snippet.kind).toBe("clip");
    expect(inc.label).toBe("real");
    // round-trips through JSON + validates byte-for-byte
    expect(parseIncident(JSON.stringify(inc))).toEqual(inc);
  });

  it("parses the exact JSON the Python side emits (snippet omitted -> null)", () => {
    // Python `Incident("inc","c", Verdict(True,0.9), 5.0)` serialized (label/snippet None -> null).
    const PY_INCIDENT =
      '{"id":"inc","concern_id":"c","verdict":{"satisfied":true,"confidence":0.9,"evidence":""},"ts":5.0,"label":null,"snippet":null}';
    expect(parseIncident(PY_INCIDENT)).toEqual({
      id: "inc",
      concern_id: "c",
      verdict: { satisfied: true, confidence: 0.9, evidence: "" },
      ts: 5.0,
      label: null,
      snippet: null,
      entity_id: null, // absent on a pre-1.3 payload -> null, never a parse failure
      message: "", // absent on a pre-1.6 payload -> "" (Python str default)
      label_url: null, // absent -> null
      // Z3 (contract 1.6) envelope absent on a pre-1.6 payload -> null, never a parse failure.
      reason: null,
      watchplan_id: null,
      roi: null,
      modality: null,
      provenance: null,
      evidence_seq: null, // absent on a pre-1.7 payload -> null, never a parse failure
      revision_id: null, // absent on a pre-1.8 payload -> null, never a parse failure
      supersedes: null, // absent on a pre-1.8 payload -> null, never a parse failure
      judgment_event_id: null, // absent on a pre-1.9 payload -> null, never a parse failure
      evidence_origin: null, // absent on a pre-1.9 payload -> null, never a parse failure
    });
  });

  it("carries the 1.9 evidence link (judgment_event_id + evidence_origin) through build + round-trip", () => {
    const inc = makeIncident({
      id: "inc-19",
      concern_id: "c-door-open",
      verdict: { satisfied: true, confidence: 0.95 },
      ts: 2.0,
      entity_id: "scene",
      evidence_seq: 7,
      judgment_event_id: judgmentEventId("c-door-open", "scene", 7),
      evidence_origin: "transition",
    });
    expect(inc.judgment_event_id).toBe("c-door-open#scene#7");
    expect(inc.evidence_origin).toBe("transition");
    expect(parseIncident(JSON.stringify(inc))).toEqual(inc);
    // evidence_origin is a CLOSED vocabulary — an out-of-vocab value is rejected.
    expect(() => makeIncident({ id: "x", concern_id: "c", verdict: { satisfied: true, confidence: 1 }, ts: 0, evidence_origin: "guessed" })).toThrow(ContractError);
    // a pre-1.9 payload omitting both parses to null, never a failure.
    const legacy = parseIncident(
      '{"id":"i","concern_id":"c","verdict":{"satisfied":true,"confidence":0.9,"evidence":""},"ts":5.0}'
    );
    expect(legacy.judgment_event_id).toBeNull();
    expect(legacy.evidence_origin).toBeNull();
  });

  it("carries entity_id through build + round-trip (D4: identity survives the wire)", () => {
    const inc = makeIncident({
      id: "inc-3",
      concern_id: "c-cup-empty",
      verdict: { satisfied: true, confidence: 0.8 },
      ts: 42.0,
      entity_id: "cup2",
    });
    expect(inc.entity_id).toBe("cup2");
    expect(parseIncident(JSON.stringify(inc))).toEqual(inc);
    // two subjects on the SAME concern stay distinguishable downstream
    const other = makeIncident({ ...inc, id: "inc-4", entity_id: "cup7" });
    expect(other.entity_id).toBe("cup7");
    expect(inc.entity_id).toBe("cup2");
    // parses the exact JSON the Python side emits with entity_id populated
    expect(
      parseIncident(
        '{"id":"i","concern_id":"c","verdict":{"satisfied":true,"confidence":0.9,"evidence":""},' +
          '"ts":5.0,"label":null,"snippet":null,"entity_id":"person-7"}'
      ).entity_id
    ).toBe("person-7");
  });

  it("carries the Z3 judgment envelope through build + round-trip (contract 1.6)", () => {
    const inc = makeIncident({
      id: "inc-z3",
      concern_id: "c-door-open",
      verdict: { satisfied: true, confidence: 0.95 },
      ts: 7.0,
      reason: "rising_edge",
      watchplan_id: "plan-abc123",
      roi: { x: 0.1, y: 0.2, w: 0.3, h: 0.4 },
      modality: "vision",
      provenance: "observed",
    });
    expect(inc.reason).toBe("rising_edge");
    expect(inc.watchplan_id).toBe("plan-abc123");
    expect(inc.roi).toEqual({ x: 0.1, y: 0.2, w: 0.3, h: 0.4 });
    expect(inc.modality).toBe("vision");
    expect(inc.provenance).toBe("observed");
    // round-trips through JSON byte-for-byte
    expect(parseIncident(JSON.stringify(inc))).toEqual(inc);
    // closed-vocab fields reject an out-of-vocabulary value
    expect(() => makeIncident({ id: "x", concern_id: "c", verdict: { satisfied: true, confidence: 1 }, ts: 0, modality: "smell" })).toThrow();
    expect(() => makeIncident({ id: "x", concern_id: "c", verdict: { satisfied: true, confidence: 1 }, ts: 0, provenance: "guessed" })).toThrow();
  });

  it("carries the supersession link through build + round-trip (contract 1.8)", () => {
    // An original/provisional incident: revision_id set, supersedes null.
    const original = makeIncident({
      id: "inc-orig",
      concern_id: "c-price",
      verdict: { satisfied: true, confidence: 0.9 },
      ts: 0.0,
      revision_id: "g:ep1#r1",
    });
    expect(original.revision_id).toBe("g:ep1#r1");
    expect(original.supersedes).toBeNull();
    expect(parseIncident(JSON.stringify(original))).toEqual(original);

    // A superseding correction: both populated, linking back to the prior revision.
    const correction = makeIncident({
      id: "inc-corr",
      concern_id: "c-price",
      verdict: { satisfied: true, confidence: 0.9 },
      ts: 25.0,
      revision_id: "g:ep1#r2",
      supersedes: "g:ep1#r1",
    });
    expect(correction.revision_id).toBe("g:ep1#r2");
    expect(correction.supersedes).toBe("g:ep1#r1");
    expect(parseIncident(JSON.stringify(correction))).toEqual(correction);

    // rejects a non-string/non-null value
    expect(() =>
      makeIncident({ id: "x", concern_id: "c", verdict: { satisfied: true, confidence: 1 }, ts: 0, revision_id: 7 })
    ).toThrow(ContractError);
    expect(() =>
      makeIncident({ id: "x", concern_id: "c", verdict: { satisfied: true, confidence: 1 }, ts: 0, supersedes: 7 })
    ).toThrow(ContractError);
  });

  it("snippet tolerates omitted payload fields -> null (redacted record shape)", () => {
    const frame = makeIncidentSnippet({ kind: "frame" });
    expect(frame).toEqual({ kind: "frame", media: null, mime: null, frames: null, t_start: null, t_end: null });
    expect(parseIncidentSnippet({ kind: "audio", mime: "audio/wav", t_start: 1, t_end: 2 })).toEqual({
      kind: "audio",
      media: null,
      mime: "audio/wav",
      frames: null,
      t_start: 1,
      t_end: 2,
    });
  });

  it("rejects bad Incident / snippet payloads", () => {
    expect(() => makeIncident({ id: 1, concern_id: "c", verdict: { satisfied: true, confidence: 1 }, ts: 0 })).toThrow(ContractError);
    expect(() => makeIncident({ id: "i", concern_id: "c", verdict: { confidence: 1 }, ts: 0 })).toThrow(ContractError); // no satisfied
    expect(() => makeIncidentSnippet({ kind: "nope" })).toThrow(ContractError);
    expect(() =>
      makeIncident({ id: "i", concern_id: "c", verdict: { satisfied: true, confidence: 1 }, ts: 0, entity_id: 7 })
    ).toThrow(ContractError); // entity_id must be a string or null
    expect(() =>
      parseIncident('{"id":"i","concern_id":"c","verdict":{"satisfied":true,"confidence":0.5},"ts":1,"entity_id":7}')
    ).toThrow(ContractError);
    expect(() => parseIncident('{"id":"i","concern_id":"c","verdict":{"satisfied":true,"confidence":0.5},"ts":"x"}')).toThrow(ContractError);
  });
});

describe("SubjectRef (contract 1.4)", () => {
  it("builds a bound ref; confidence defaults to null (NOT 1.0)", () => {
    const s = makeSubjectRef({
      scheme: "content",
      namespace: "acme-erp",
      key: "PO-4471",
      resolution: "bound",
      scope: "durable",
      strategy: "judge",
    });
    expect(s).toEqual({
      scheme: "content",
      namespace: "acme-erp",
      key: "PO-4471",
      resolution: "bound",
      scope: "durable",
      strategy: "judge",
      confidence: null,
    });
    // round-trips through JSON byte-for-byte
    expect(parseSubjectRef(JSON.stringify(s))).toEqual(s);
  });

  it("parses the exact JSON the Python side emits for a bound ref", () => {
    const PY =
      '{"scheme":"content","namespace":"acme-erp","key":"PO-4471","resolution":"bound","scope":"durable","strategy":"client","confidence":0.8}';
    expect(parseSubjectRef(PY)).toEqual({
      scheme: "content",
      namespace: "acme-erp",
      key: "PO-4471",
      resolution: "bound",
      scope: "durable",
      strategy: "client",
      confidence: 0.8,
    });
  });

  it("an unknown ref forces key to null (absence of a key, not a fabricated one)", () => {
    const s = makeSubjectRef({
      scheme: "content",
      namespace: "acme-erp",
      resolution: "unknown",
      scope: "session",
      strategy: "judge",
    });
    expect(s.key).toBe(null);
    expect(s.resolution).toBe("unknown");
  });

  it("rejects the invariant violations the Python __post_init__ rejects", () => {
    // 'bound' requires a non-empty namespace + key
    expect(() =>
      makeSubjectRef({ scheme: "content", namespace: "", key: "PO-1", resolution: "bound", scope: "durable", strategy: "j" })
    ).toThrow(ContractError);
    expect(() =>
      makeSubjectRef({ scheme: "content", namespace: "acme", key: null, resolution: "bound", scope: "durable", strategy: "j" })
    ).toThrow(ContractError);
    // 'unknown' REJECTS a present key
    expect(() =>
      parseSubjectRef('{"scheme":"content","namespace":"acme","key":"PO-1","resolution":"unknown","scope":"session","strategy":"j"}')
    ).toThrow(ContractError);
    // bad enums
    expect(() =>
      makeSubjectRef({ scheme: "c", namespace: "n", key: "k", resolution: "resolved", scope: "durable", strategy: "j" })
    ).toThrow(ContractError);
    expect(() =>
      makeSubjectRef({ scheme: "c", namespace: "n", key: "k", resolution: "bound", scope: "forever", strategy: "j" })
    ).toThrow(ContractError);
    // confidence out of [0,1]
    expect(() =>
      makeSubjectRef({ scheme: "c", namespace: "n", key: "k", resolution: "bound", scope: "durable", strategy: "j", confidence: 1.5 })
    ).toThrow(ContractError);
  });
});

describe("SubjectRef 'provisional' resolution state (contract 1.7)", () => {
  it("provisionalSubject(episode_id) mints a session-scoped UN_<episode_id> handle", () => {
    const s = provisionalSubject("ep-42");
    expect(s).toEqual({
      scheme: "episode",
      namespace: "",
      key: "UN_ep-42",
      resolution: "provisional",
      scope: "session",
      strategy: "episode_scoped",
      confidence: null,
    });
    // round-trips through JSON byte-for-byte
    expect(parseSubjectRef(JSON.stringify(s))).toEqual(s);
  });

  it("PROVISIONAL_KEY_PREFIX is the literal 'UN_' the Python side mints keys with", () => {
    expect(PROVISIONAL_KEY_PREFIX).toBe("UN_");
  });

  it("two provisionalSubject() calls for the SAME episode id produce the same key (stable within the episode)", () => {
    const a = provisionalSubject("ep-1");
    const b = provisionalSubject("ep-1");
    expect(a.key).toBe(b.key);
    // and a DIFFERENT episode id produces a different key
    const c = provisionalSubject("ep-2");
    expect(c.key).not.toBe(a.key);
  });

  it("parses the exact JSON the Python side emits for a provisional ref", () => {
    const PY =
      '{"scheme":"episode","namespace":"","key":"UN_ep-99","resolution":"provisional","scope":"session","strategy":"episode_scoped","confidence":null}';
    expect(parseSubjectRef(PY)).toEqual({
      scheme: "episode",
      namespace: "",
      key: "UN_ep-99",
      resolution: "provisional",
      scope: "session",
      strategy: "episode_scoped",
      confidence: null,
    });
  });

  it("rejects the invariant violations the Python __post_init__ rejects for 'provisional'", () => {
    // requires episode_id to be a non-empty string
    expect(() => provisionalSubject("")).toThrow(ContractError);
    expect(() => provisionalSubject("   ")).toThrow(ContractError);
    expect(() => provisionalSubject(null)).toThrow(ContractError);
    // 'provisional' requires a key of the form UN_<episode_id>
    expect(() =>
      makeSubjectRef({ scheme: "episode", namespace: "", key: "ep-1", resolution: "provisional", scope: "session", strategy: "s" })
    ).toThrow(ContractError);
    expect(() =>
      makeSubjectRef({ scheme: "episode", namespace: "", key: null, resolution: "provisional", scope: "session", strategy: "s" })
    ).toThrow(ContractError);
    // 'provisional' requires scope === "session"
    expect(() =>
      makeSubjectRef({ scheme: "episode", namespace: "", key: "UN_ep-1", resolution: "provisional", scope: "durable", strategy: "s" })
    ).toThrow(ContractError);
    expect(() =>
      parseSubjectRef('{"scheme":"episode","namespace":"","key":"UN_ep-1","resolution":"provisional","scope":"source","strategy":"s"}')
    ).toThrow(ContractError);
  });

  it("a 'provisional' ref is distinct from a 'bound' ref sharing the same key string (never laundered)", () => {
    const provisional = provisionalSubject("ep-1");
    const bound = makeSubjectRef({
      scheme: "episode",
      namespace: "acme",
      key: provisional.key,
      resolution: "bound",
      scope: "durable",
      strategy: "client",
    });
    expect(provisional.resolution).not.toBe(bound.resolution);
    expect(provisional).not.toEqual(bound);
  });
});

describe("SubjectRef <-> SubjectRef.schema.json parity", () => {
  const SCHEMA_PATH = join(
    __dirname,
    "..",
    "..",
    "percept-vision",
    "percept",
    "contracts",
    "schema",
    "SubjectRef.schema.json"
  );
  const schema = JSON.parse(readFileSync(SCHEMA_PATH, "utf8"));

  it("the schema's x-contract-version matches the JS CONTRACT_VERSION (one frozen contract, two languages)", () => {
    expect(schema["x-contract-version"]).toBe(CONTRACT_VERSION);
  });

  it("the schema's resolution/scope enums match the JS closed-vocab literals exactly", () => {
    expect(schema.properties.resolution.enum.slice().sort()).toEqual(
      ["bound", "unknown", "provisional"].sort()
    );
    expect(schema.properties.scope.enum.slice().sort()).toEqual(
      ["session", "source", "durable"].sort()
    );
    // and every required field on the schema is a field the JS shape actually produces
    const built = makeSubjectRef({
      scheme: "content",
      namespace: "acme",
      key: "PO-1",
      resolution: "bound",
      scope: "durable",
      strategy: "client",
    });
    expect(schema.required.slice().sort()).toEqual(
      ["scheme", "namespace", "key", "resolution", "scope", "strategy"].sort()
    );
    for (const field of schema.required) {
      expect(Object.prototype.hasOwnProperty.call(built, field)).toBe(true);
    }
  });

  it("a provisionalSubject() build validates against the schema's shape (scheme/namespace/key/strategy are strings, key non-null)", () => {
    const s = provisionalSubject("ep-schema-check");
    expect(typeof s.scheme).toBe(schema.properties.scheme.type);
    expect(typeof s.namespace).toBe(schema.properties.namespace.type);
    expect(typeof s.strategy).toBe(schema.properties.strategy.type);
    expect(schema.properties.key.anyOf.some((t) => t.type === "string")).toBe(true);
    expect(typeof s.key).toBe("string");
    expect(schema.properties.resolution.enum).toContain(s.resolution);
    expect(schema.properties.scope.enum).toContain(s.scope);
  });
});
