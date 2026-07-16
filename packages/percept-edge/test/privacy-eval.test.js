import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { scoreCases, qualityOf } from "../eval/harness.mjs";

// The app's integration precondition: PrivacyScreen ships WITH a regression-testable scorecard. This
// wires the same eval harness the human `npm run eval` uses into CI, so a detector/mapping change that
// leaks or drifts fails the build. Latency (wall-clock) is excluded — it is checked, not gated.

const HERE = dirname(fileURLToPath(import.meta.url));
const CASES = JSON.parse(readFileSync(join(HERE, "..", "eval", "fixtures", "privacy.cases.json"), "utf8")).cases;
const BASELINE = JSON.parse(readFileSync(join(HERE, "..", "eval", "scorecard.baseline.json"), "utf8"));

describe("PrivacyScreen eval — scorecard regression gate", () => {
  it("has ZERO leaks on the text + image fixtures (recall bias, unrecoverable cost)", async () => {
    const q = qualityOf(await scoreCases(CASES));
    expect(q.leakRate).toBe(0);
    expect(q.confusion.fn).toBe(0);
  });

  it("no action mismatches vs the pinned ladder", async () => {
    const q = qualityOf(await scoreCases(CASES));
    expect(q.actionMismatches).toEqual([]);
  });

  it("quality metrics match the checked-in baseline (no silent drift)", async () => {
    const q = qualityOf(await scoreCases(CASES));
    expect(q).toEqual(BASELINE);
  });
});
