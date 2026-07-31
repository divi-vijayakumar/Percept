import { describe, it, expect } from "vitest";
import {
  MotionPeriodicity,
  AcousticOnset,
  SkillBank,
  wakeScore,
  shouldWake,
  registerSkill,
  makeDetector,
  manifestOf,
  listManifests,
  registered,
  detectorsInCategory,
  signalOf,
  CATEGORY_MOTION,
  CATEGORY_AUDIO,
  CATEGORY_PRESENCE,
  CATEGORY_POSE,
  CATEGORY_AUDIO_EVENT,
  CATEGORY_CHANGE,
  CATEGORY_SEMANTIC_RELEVANCE,
  ACOUSTIC_ONSET_HIGH,
  ACOUSTIC_ONSET_LOW,
  WakePolicyError,
  cheapSignals,
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
    // after the first onset, 0.2 stays above the re-arm bar -> stays disarmed -> no second fire
    const onsets = run(new AcousticOnset(), [0.0, 0.5, 0.2, 0.5, 0.2]);
    expect(onsets).toHaveLength(1);
  });

  it("ships the Python bars, not the retired 0.09/0.0225 pair", () => {
    // BUG-A-02-05's twin. The old JS defaults were an RMS amplitude of 0.3 in ENERGY terms; measured
    // against 987 s of annotated sound, the loudest chunk in the whole corpus reached 0.0277, so that
    // gate woke 0 times on 204 annotated events. The two runtimes must agree on the shipped bar or the
    // same stream is heard by one edge and not the other.
    expect(ACOUSTIC_ONSET_HIGH).toBe(0.001);
    expect(ACOUSTIC_ONSET_LOW).toBe(0.00025);
    const det = new AcousticOnset();
    expect(det.high).toBe(ACOUSTIC_ONSET_HIGH);
    expect(det.low).toBe(ACOUSTIC_ONSET_LOW);
    // the shipped bar hears the corpus's loudest chunk; the retired one is deaf to it
    expect(run(new AcousticOnset(), [0.0, 0.0277])).toHaveLength(1);
    expect(run(new AcousticOnset({ high: 0.09, low: 0.0225 }), [0.0, 0.0277])).toHaveLength(0);
  });

  it("refuses an inverted band at construction rather than minting phantom onsets", () => {
    // BUG-A-02-05. Every sample in [high, low) both fires (v >= high) and re-arms (v < low), so a
    // dead-steady tone oscillates the state machine and emits an onset every other sample. An onset is
    // "a sound STARTING", and a constant level contains exactly one start whatever the bars are.
    expect(() => new AcousticOnset({ high: 0.1, low: 0.5 })).toThrow(/low <= high/);
    // ...and low === high stays PERMITTED: no sample is both >= high and < low, so the oscillation is
    // unreachable at a zero-width band.
    const flat = new AcousticOnset({ high: 0.2, low: 0.2 });
    expect(run(flat, [0.2, 0.2, 0.2, 0.2, 0.2])).toHaveLength(1);
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
      "category",
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

describe("sense categories — acoustic-onset is the first-class AUDIO sense", () => {
  it("marks each built-in with its modality category", () => {
    expect(manifestOf("acoustic-onset").category).toBe(CATEGORY_AUDIO);
    expect(manifestOf("motion-periodicity").category).toBe(CATEGORY_MOTION);
  });

  it("discovers audio detectors as a class", () => {
    // the modality-as-a-class query: every audio Tier-0 sense, mirroring Python detectors_in_category
    expect(detectorsInCategory(CATEGORY_AUDIO)).toContain("acoustic-onset");
    expect(detectorsInCategory(CATEGORY_AUDIO)).not.toContain("motion-periodicity");
    expect(detectorsInCategory(CATEGORY_MOTION)).toContain("motion-periodicity");
    expect(detectorsInCategory("smell")).toEqual([]);
  });

  it("exports additive wake-policy categories", () => {
    expect(CATEGORY_PRESENCE).toBe("presence");
    expect(CATEGORY_POSE).toBe("pose");
    expect(CATEGORY_AUDIO_EVENT).toBe("audio_event");
    expect(CATEGORY_CHANGE).toBe("change");
    expect(CATEGORY_SEMANTIC_RELEVANCE).toBe("semantic_relevance");
  });
});

describe("wakeScore / shouldWake — cheap evidence fusion before VLM", () => {
  it("preserves legacy max-channel salience without policy weights", () => {
    expect(wakeScore({ motion: 0.2, semantic: 0.7 })).toBeCloseTo(0.7, 10);
  });

  it("combines weak weighted cues with noisy-OR", () => {
    const policy = { threshold: 0.45, weights: { presence: 0.7, audio: 0.7 } };
    expect(wakeScore({ presence: 0.5, audio: 0.5 }, policy)).toBeGreaterThan(0.45);
    const decision = shouldWake({ presence: 0.5, audio: 0.5 }, policy);
    expect(decision.wake).toBe(true);
    expect(decision.top_channel).toBe("presence");
  });

  it("clamps out-of-range inputs", () => {
    expect(wakeScore({ semantic: 9 }, { threshold: 0.9, weights: { semantic: 2 } })).toBe(1);
  });

  it("fuses over the MEASURED channels, so a weight can never score below the max it replaced", () => {
    // BUG-A-02-04's twin. The loop iterated the WEIGHT table, so any measured channel with no
    // explicit weight was deleted from the fusion and a typo'd channel name was total deafness.
    const signals = { presence: 0.9, motion: 0.8 };
    expect(wakeScore(signals)).toBeCloseTo(0.9, 12);
    // weights are per-channel PRIORS defaulting to 1.0 -> a strict generalization of the max
    expect(wakeScore(signals, { weights: { motion: 1.0 } })).toBeCloseTo(0.98, 12);
    expect(wakeScore(signals, { weights: { motion: 1.0 } })).toBeGreaterThanOrEqual(wakeScore(signals));
    // ...and a typo in a CHANNEL name now costs that channel's prior, not every signal present
    expect(wakeScore(signals, { weights: { presenc: 1.0 } })).toBeCloseTo(0.98, 12);
  });

  it("reads a channel named in weights but absent from signals as no vote, not as a zero", () => {
    const policy = { weights: { presence: 1.0, motion: 1.0 } };
    expect(wakeScore({ motion: 0.5 }, policy)).toBeCloseTo(0.5, 12);
    expect(wakeScore({ motion: 0.5, presence: 0.0 }, policy)).toBeCloseTo(0.5, 12);
    expect(wakeScore({}, policy)).toBe(0);
  });

  it("names a top_channel that actually contributed, and none when nothing did", () => {
    // BUG-A-02-03's twin: top_channel was the argmax of the RAW signals, computed with no reference to
    // the weights, beside a score computed FROM them.
    const d = shouldWake({ presence: 0.9, motion: 0.4 }, { weights: { presence: 0.0, motion: 1.0 } });
    expect(d.score).toBeCloseTo(0.4, 12);
    expect(d.top_channel).toBe("motion");
    // every contribution zero -> no channel to name, even though signals were present
    const silent = shouldWake({ presence: 0.9 }, { weights: { presence: 0.0 } });
    expect(silent.score).toBe(0);
    expect(silent.top_channel).toBeNull();
    // ...and a floor lifts the score without inventing a channel that produced it
    const floored = shouldWake({}, { threshold: 0.5, floor: 1.0 });
    expect(floored.wake).toBe(true);
    expect(floored.top_channel).toBeNull();
  });

  it("keeps ties on the first channel", () => {
    expect(shouldWake({ motion: 0.5, presence: 0.5 }).top_channel).toBe("motion");
    expect(shouldWake({ motion: 0.4, presence: 0.5 }).top_channel).toBe("presence");
    expect(shouldWake({}).top_channel).toBeNull();
  });

  it("refuses a malformed policy by name instead of degrading to the legacy max", () => {
    // BUG-A-02-08's twin. `weigths` used to read as an absent `weights`, so the calibrated OR
    // silently became max() while keeping the authored threshold — nothing distinguished "calibrated"
    // from "running the legacy default".
    expect(() => wakeScore({ presence: 0.5 }, { threshold: 0.9, weigths: { presence: 1.0 } })).toThrow(
      WakePolicyError,
    );
    expect(() => wakeScore({}, { weights: { presence: "0.7" } })).toThrow(/stringified number/);
    expect(() => wakeScore({}, { weights: { pose: true } })).toThrow(/bool is not a number/);
    expect(() => wakeScore({}, { threshold: Infinity })).toThrow(/must be finite/);
    expect(() => wakeScore({}, { weights: [1, 2] })).toThrow(WakePolicyError);
    expect(() => wakeScore({}, ["motion"])).toThrow(WakePolicyError);
    // a well-formed policy still passes through untouched
    expect(wakeScore({ presence: 0.5 }, { threshold: 0.9, weights: { presence: 1.0 } })).toBeCloseTo(0.5, 12);
  });
});

describe("cheapSignals — one measurement populates exactly one channel", () => {
  it("emits four channels from four measurements, with no aliases", () => {
    // BUG-A-01-05's twin: `change` aliased `motion`, `semantic_relevance` aliased `semantic`, and
    // `audio_event` aliased `audio`, so the noisy-OR fused one reading with itself.
    expect(cheapSignals({ motion: 0.6, structural: 0.1, semantic: 0.2, energy: 0.3 })).toEqual({
      motion: 0.6,
      structural: 0.1,
      semantic: 0.2,
      audio: 0.3,
    });
    // the aliased reading was worth more than the measurement
    expect(wakeScore({ motion: 0.6, change: 0.6 }, { weights: { motion: 0.5, change: 0.5 } })).toBeCloseTo(
      0.51,
      12,
    );
    expect(wakeScore({ motion: 0.6 }, { weights: { motion: 0.5 } })).toBeCloseTo(0.3, 12);
  });

  it("omits audio when no microphone is open rather than reporting silence", () => {
    expect(cheapSignals({ motion: 0.1, structural: 0, semantic: 0 })).not.toHaveProperty("audio");
    expect(cheapSignals({ motion: 0.1, structural: 0, semantic: 0, energy: 0 }).audio).toBe(0);
  });

  it("applies the caller's overlay last", () => {
    const out = cheapSignals({
      motion: 0.1,
      structural: 0,
      semantic: 0,
      extra: { presence: 0.7, motion: 0.9 },
    });
    expect(out.presence).toBe(0.7);
    expect(out.motion).toBe(0.9);
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

  it("feeds an unrecognised channel nothing instead of falling back to motion", () => {
    // BUG-A-02-02's twin: `const sig = signal === "energy" ? energy : motion` made every name that
    // was not exactly "energy" or "frame" read frame-diff motion, so a skill declaring `"thermal"` —
    // or a typo like `"enrgy"` — emitted boundaries the brain counted as its declared sense.
    // `registerSkill` is the first gate; this is the bank's own, reached by tampering with the
    // resolved channel the way a hand-built bank entry could.
    expect(SkillBank.CHANNELS).toEqual(["motion", "energy", "frame"]);
    expect(() => registerSkill({ name: "thermal-skill", signal: "thermal", factory: () => ({}) })).toThrow(
      /signal must be/,
    );
    const bank = new SkillBank(["motion-periodicity"]);
    bank._dets[0].signal = "enrgy";
    const out = [];
    OSC.forEach((v, t) => out.push(...bank.push(t, { motion: v })));
    expect(out).toEqual([]);
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
