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
      "gate_on",
      "goal_id",
      "region_hint",
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
      "gate_on",
      "goal_id",
      "region_hint",
      "threshold",
    ]);
    expect(c.region_hint).toBeNull();
    expect(c.detectors).toEqual([]);
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
  it("makeTier0Signal has exactly { motion, structural, semantic, ts }", () => {
    const s = makeTier0Signal({
      motion: 0.5,
      structural: 0,
      semantic: 0,
      ts: 123.5,
    });
    expect(Object.keys(s).sort()).toEqual([
      "motion",
      "semantic",
      "structural",
      "ts",
    ]);
    expect(s).toEqual({ motion: 0.5, structural: 0, semantic: 0, ts: 123.5 });
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
  it("is the 1.1.0 semver twin of the Python contract version", () => {
    expect(CONTRACT_VERSION).toBe("1.1.0");
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
    });
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
    expect(() => parseIncident('{"id":"i","concern_id":"c","verdict":{"satisfied":true,"confidence":0.5},"ts":"x"}')).toThrow(ContractError);
  });
});
