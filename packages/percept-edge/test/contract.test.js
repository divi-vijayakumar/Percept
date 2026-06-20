import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import {
  ContractError,
  makeWatchCondition,
  makeWatchSpec,
  parseWatchSpec,
  makeTier0Signal,
  parseTier0Signal,
  makeBox,
} from "../src/contract.js";

const __dirname = dirname(fileURLToPath(import.meta.url));

// The exact JSON the Python side emits for a one-condition WatchSpec.
const PY_WATCHSPEC =
  '{"conditions":[{"goal_id":"g1","anchor_text":"the user is drinking coffee","threshold":0.3,"region_hint":null}],"version":1}';

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
      "goal_id",
      "region_hint",
      "threshold",
    ]);
    expect(c.region_hint).toBeNull();
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
