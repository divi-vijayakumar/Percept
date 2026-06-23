import { describe, it, expect } from "vitest";
import {
  MotionPeriodicity,
  AcousticOnset,
  SkillBank,
  registerSkill,
  makeDetector,
  manifestOf,
  listManifests,
  registered,
  signalOf,
} from "../src/skills.js";

// A clean oscillation: each rep climbs to 1.0 then drops to 0.0. Peaks emitted on the descent.
const OSC = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]; // 5 reps (ones at t=1,3,5,7,9)

function run(det, values) {
  const out = [];
  values.forEach((v, t) => out.push(...det.push(t, v)));
  return out;
}

describe("MotionPeriodicity — counts oscillations as 'peak' boundaries", () => {
  it("emits one peak per completed bump", () => {
    const peaks = run(new MotionPeriodicity(), OSC);
    expect(peaks).toHaveLength(5);
    expect(peaks.every((b) => b.kind === "peak" && b.detector === "motion-periodicity")).toBe(true);
  });

  it("ignores still-scene jitter below minHeight", () => {
    const jitter = [0, 0.01, 0, 0.015, 0, 0.005, 0]; // all < minHeight 0.02
    expect(run(new MotionPeriodicity(), jitter)).toHaveLength(0);
  });

  it("refractoryS debounces double-counts", () => {
    // reps descend at t=2 and t=4 (gap 2.0); a 3.0 refractory suppresses the second
    const peaks = run(new MotionPeriodicity({ refractoryS: 3.0 }), [0, 1, 0, 1, 0]);
    expect(peaks).toHaveLength(1);
  });
});

describe("AcousticOnset — energy onsets with hysteresis", () => {
  it("fires one onset per rising spike, re-arming only after falling below low", () => {
    const onsets = run(new AcousticOnset(), [0.0, 0.5, 0.5, 0.0, 0.5]); // rise, hold, drop, rise
    expect(onsets).toHaveLength(2);
    expect(onsets.every((b) => b.kind === "onset" && b.detector === "acoustic-onset")).toBe(true);
  });

  it("does NOT re-fire while energy stays above low (hysteresis holds it disarmed)", () => {
    // after the first onset, 0.2 is between low(0.15) and high(0.3) -> stays disarmed -> no second fire
    const onsets = run(new AcousticOnset(), [0.0, 0.5, 0.2, 0.5, 0.2]);
    expect(onsets).toHaveLength(1);
  });
});

describe("the registry — register + compose by name", () => {
  it("exposes the built-in curated skills with manifests", () => {
    expect(registered()).toContain("motion-periodicity");
    expect(registered()).toContain("acoustic-onset");
    expect(signalOf("motion-periodicity")).toBe("motion");
    expect(signalOf("acoustic-onset")).toBe("energy");
    // manifest is the wire-shaped self-description (matches the Python DetectorManifest fields)
    expect(Object.keys(manifestOf("motion-periodicity")).sort()).toEqual([
      "name",
      "output",
      "senses",
      "when_to_use",
    ]);
    expect(listManifests().length).toBeGreaterThanOrEqual(2);
  });

  it("makeDetector throws on an unknown skill (the firewall)", () => {
    expect(() => makeDetector("no-such-skill")).toThrow(/unknown skill/);
  });
});

describe("SkillBank — routes the cheap channels to each skill", () => {
  it("counts motion peaks fed on the motion channel", () => {
    const bank = new SkillBank(["motion-periodicity"]);
    const out = [];
    OSC.forEach((v, t) => out.push(...bank.push(t, { motion: v })));
    expect(out).toHaveLength(5);
  });

  it("routes motion and energy to the right skills simultaneously", () => {
    const bank = new SkillBank(["motion-periodicity", "acoustic-onset"]);
    const out = [];
    // motion oscillates (2 reps); energy spikes once
    const motion = [0, 1, 0, 1, 0];
    const energy = [0, 0, 0.5, 0, 0];
    motion.forEach((m, t) => out.push(...bank.push(t, { motion: m, energy: energy[t] })));
    const peaks = out.filter((b) => b.kind === "peak");
    const onsets = out.filter((b) => b.kind === "onset");
    expect(peaks).toHaveLength(2);
    expect(onsets).toHaveLength(1);
  });

  it("reset() clears detector state", () => {
    const bank = new SkillBank(["motion-periodicity"]);
    OSC.forEach((v, t) => bank.push(t, { motion: v }));
    bank.reset();
    const out = [];
    OSC.forEach((v, t) => out.push(...bank.push(t, { motion: v })));
    expect(out).toHaveLength(5); // counts fresh, not 10
  });
});

describe("registerSkill — the third-party extension point (drone / glasses / coach)", () => {
  it("registers a custom skill and composes it by name", () => {
    // a product's bespoke skill: emit a 'threshold' boundary when motion first crosses 0.5
    class ThresholdCross {
      constructor() {
        this.signal = "motion";
        this.name = "threshold-cross";
        this._armed = true;
      }
      reset() {
        this._armed = true;
      }
      push(t, v) {
        if (this._armed && v >= 0.5) {
          this._armed = false;
          return [{ detector: this.name, kind: "threshold", t, value: v }];
        }
        return [];
      }
    }
    registerSkill({
      name: "threshold-cross",
      signal: "motion",
      factory: () => new ThresholdCross(),
      manifest: { name: "threshold-cross", senses: "motion", when_to_use: "demo", output: "threshold" },
    });
    expect(registered()).toContain("threshold-cross");
    const bank = new SkillBank(["threshold-cross"]);
    const got = [];
    [0, 0.6, 0.6].forEach((v, t) => got.push(...bank.push(t, { motion: v })));
    expect(got).toHaveLength(1);
    expect(got[0].kind).toBe("threshold");
  });

  it("rejects a bad signal channel", () => {
    expect(() =>
      registerSkill({ name: "bad", signal: "telepathy", factory: () => ({}) })
    ).toThrow(/signal must be/);
  });
});

describe("lazy local load (#2) — model-backed skills load on activation, not at construct", () => {
  it("bank.load() initialises a model-backed skill exactly once; makeDetector does not", () => {
    let loads = 0;
    class FakeModelSkill {
      constructor() {
        this.signal = "frame";
        this.name = "fake-model";
        this._ready = false;
      }
      async load() {
        loads += 1;
        this._ready = true;
      }
      reset() {}
      push(t, _frame) {
        return this._ready ? [{ detector: this.name, kind: "peak", t, value: 1 }] : [];
      }
    }
    registerSkill({
      name: "fake-model",
      signal: "frame",
      factory: () => new FakeModelSkill(),
      manifest: { name: "fake-model", senses: "frame", when_to_use: "test", output: "peak" },
    });

    const bank = new SkillBank(["fake-model"]);
    expect(loads).toBe(0); // constructing the bank did NOT load the model

    return (async () => {
      await bank.load();
      await bank.load(); // idempotent
      expect(loads).toBe(1);
      const out = bank.push(0, { frame: {} });
      expect(out).toHaveLength(1);
    })();
  });
});
