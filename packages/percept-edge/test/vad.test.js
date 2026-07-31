import { readFileSync } from "node:fs";
import { describe, it, expect } from "vitest";
import {
  EnergyVad,
  SileroVadAdapter,
  endpointEnergySequence,
  frameEnergy,
  pcm16Energy,
} from "../src/vad.js";

const goldenAudio = JSON.parse(
  readFileSync(new URL("./fixtures/audio_stream.golden.json", import.meta.url), "utf8")
);

/**
 * Integer-exact transcription of the Python runtime's `pcm_energy`
 * (percept/runtime/audio.py): one exact integer numerator, a single division by
 * `32768 * 32768 * n`, then the clamp. The energy value is the quantity every
 * threshold in both runtimes is compared against, so the two implementations
 * have to agree to the last bit and not merely to twelve decimals — an ulp of
 * drift right at a threshold wakes one runtime and not the other.
 */
function pythonPcmEnergy(samples) {
  const n = samples.length;
  if (n === 0) return 0;
  let squares = 0n;
  for (const sample of samples) squares += BigInt(sample) * BigInt(sample);
  return Math.min(1, Number(squares) / (32768 * 32768 * n));
}

/**
 * Deterministic signed-16-bit noise. The same linear congruential recurrence
 * (state = state * 1664525 + 1013904223 mod 2^32, sample = state mod 65536 - 32768)
 * run in Python produces the frozen expectations asserted below, so these
 * vectors are a real cross-runtime comparison and not a self-check.
 */
function seededPcm(seed, count) {
  let state = seed >>> 0;
  const out = new Array(count);
  for (let i = 0; i < count; i++) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    out[i] = (state % 65536) - 32768;
  }
  return out;
}

describe("frameEnergy", () => {
  it("is 0 for silence and positive for a signal", () => {
    expect(frameEnergy(new Float32Array([0, 0, 0, 0]))).toBe(0);
    expect(frameEnergy(new Float32Array([0.5, -0.5, 0.5, -0.5]))).toBeCloseTo(
      0.25,
      10
    );
  });

  it("is 0 for an empty frame", () => {
    expect(frameEnergy(new Float32Array([]))).toBe(0);
  });
});

describe("pcm16Energy is bit-identical to the Python runtime", () => {
  // The shared golden fixture is asserted elsewhere with toBeCloseTo(..., 12),
  // which cannot see a divergence in the denominator or in the accumulation
  // order. These assertions are exact.
  for (const vector of goldenAudio.energy_vectors) {
    it(`${vector.name} equals the recorded value exactly`, () => {
      expect(pcm16Energy(vector.samples)).toBe(vector.expected);
    });

    it(`${vector.name} equals the integer-exact denominator form`, () => {
      expect(pcm16Energy(vector.samples)).toBe(pythonPcmEnergy(vector.samples));
    });
  }

  it("holds on 512 samples of seeded noise", () => {
    const samples = seededPcm(123456789, 512);
    expect(samples.slice(0, 4)).toEqual([15216, -26609, -23006, -4327]);
    expect(pcm16Energy(samples)).toBe(pythonPcmEnergy(samples));
    // Frozen from percept.runtime.audio.pcm_energy over the same recurrence.
    expect(pcm16Energy(samples)).toBe(0.3163647982291877);
  });

  it("holds on a full second of seeded noise at 16 kHz", () => {
    // 16000 samples is where a naive accumulation could start to drift from the
    // single-division form; at this size both stay exact.
    const samples = seededPcm(987654321, 16000);
    expect(samples.slice(0, 4)).toEqual([19036, 25099, 20462, -10635]);
    expect(pcm16Energy(samples)).toBe(pythonPcmEnergy(samples));
    expect(pcm16Energy(samples)).toBe(0.3340243362300098);
  });

  it("normalizes by 32768, not by 32767", () => {
    // A 32767 denominator would push the positive rail to 1.0 and be hidden by
    // the clamp; these two values are what tell the two divisors apart.
    expect(pcm16Energy([32767])).toBe(0.9999389657750726);
    expect(pcm16Energy([1])).toBe(9.313225746154785e-10);
  });

  it("clamps the negative full-scale rail to exactly 1", () => {
    const rail = new Array(8).fill(-32768);
    expect(pcm16Energy(rail)).toBe(1);
    expect(pcm16Energy(rail)).toBe(pythonPcmEnergy(rail));
  });

  it("reads a typed array the same as a plain array", () => {
    const samples = seededPcm(42, 64);
    expect(pcm16Energy(Int16Array.from(samples))).toBe(pcm16Energy(samples));
  });

  it("is 0 for an empty buffer", () => {
    expect(pcm16Energy([])).toBe(0);
    expect(pcm16Energy(new Int16Array(0))).toBe(0);
  });

  it("agrees with frameEnergy on the same signal in the float domain", () => {
    // frameEnergy is the [-1, 1] float path and pcm16Energy the integer path;
    // one physical signal has to produce one energy, or a threshold tuned on
    // one carrier misfires on the other.
    const samples = seededPcm(2024, 256);
    const floats = Float32Array.from(samples, (sample) => sample / 32768);
    expect(frameEnergy(floats)).toBe(pcm16Energy(samples));
  });

  it("is the only integer-domain entry point: pushSamples is the float path", () => {
    // Feeding integer samples to the float path is rejected rather than
    // silently mis-scaled by 2^30.
    expect(() => new EnergyVad().pushSamples(Int16Array.from([1000, -1000]))).toThrow(
      "energy must be finite and within [0, 1]"
    );
  });

  for (const [label, samples] of [
    ["a fractional sample", [0.5]],
    ["below the negative rail", [-32769]],
    ["above the positive rail", [32768]],
    ["a non-numeric sample", [Number.NaN]],
    ["a non-finite sample", [Number.POSITIVE_INFINITY]],
    ["one bad sample among good ones", [0, 1, 32768, 1]],
  ]) {
    it(`rejects ${label}`, () => {
      expect(() => pcm16Energy(samples)).toThrow(
        "pcm16Energy samples must be signed 16-bit integers"
      );
    });
  }
});

describe("EnergyVad endpointing — fixture energy sequence", () => {
  // 3 silence, a 6-frame speech burst, then 10 silence frames. With hangover
  // 8 the endpoint should fire on the 8th trailing silence frame.
  const seq = [
    0, 0, 0, // silence
    0.2, 0.3, 0.25, 0.2, 0.3, 0.2, // speech (6 frames)
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // silence (10 frames)
  ];

  it("fires exactly one endpoint after speech + hangover silence", () => {
    const endpoints = endpointEnergySequence(seq, {
      threshold: 0.01,
      hangoverFrames: 8,
      minSpeechFrames: 3,
    });
    expect(endpoints.length).toBe(1);
    // speech frames are indices 3..8; silence starts at 9; 8th silence = index 16.
    expect(endpoints[0]).toBe(16);
  });

  it("emits a rising-edge `started` on the first speech frame", () => {
    const vad = new EnergyVad({ threshold: 0.01 });
    const events = seq.map((e) => vad.pushEnergy(e));
    const startFrames = events.filter((e) => e.started).map((e) => e.frame);
    expect(startFrames).toEqual([3]);
  });

  it("does NOT endpoint a sub-minimum speech blip", () => {
    // single speech frame (< minSpeechFrames) surrounded by silence
    const blip = [0, 0, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    const endpoints = endpointEnergySequence(blip, {
      threshold: 0.01,
      hangoverFrames: 8,
      minSpeechFrames: 3,
    });
    expect(endpoints).toEqual([]);
  });

  it("does NOT endpoint while speech is ongoing (no trailing silence)", () => {
    const ongoing = [0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3];
    expect(endpointEnergySequence(ongoing, { hangoverFrames: 8 })).toEqual([]);
  });

  it("handles two utterances -> two endpoints", () => {
    const two = [
      0.3, 0.3, 0.3, 0.3, // utterance 1
      0, 0, 0, 0, 0, 0, 0, 0, // 8 silence -> endpoint
      0.3, 0.3, 0.3, 0.3, // utterance 2
      0, 0, 0, 0, 0, 0, 0, 0, // 8 silence -> endpoint
    ];
    const endpoints = endpointEnergySequence(two, {
      threshold: 0.01,
      hangoverFrames: 8,
      minSpeechFrames: 3,
    });
    expect(endpoints.length).toBe(2);
  });

  it("pushSamples computes energy from raw PCM", () => {
    const vad = new EnergyVad({ threshold: 0.01 });
    const ev = vad.pushSamples(new Float32Array([0.5, -0.5, 0.5, -0.5]));
    expect(ev.speech).toBe(true);
  });

  it.each([
    { threshold: Number.NaN },
    { threshold: 2 },
    { hangoverFrames: 0 },
    { hangoverFrames: 1.5 },
    { minSpeechFrames: 0 },
  ])("rejects invalid config %#", (config) => {
    expect(() => new EnergyVad(config)).toThrow();
  });

  it.each([Number.NaN, Number.POSITIVE_INFINITY, -0.1, 1.1])(
    "rejects invalid energy %s",
    (energy) => {
      expect(() => new EnergyVad().pushEnergy(energy)).toThrow();
    }
  );
});

describe("EnergyVad state machine", () => {
  it("counts energy exactly at the threshold as speech", () => {
    const vad = new EnergyVad({ threshold: 0.25 });
    const atThreshold = vad.pushEnergy(0.25);
    expect(atThreshold.speech).toBe(true);
    expect(atThreshold.started).toBe(true);
    // The comparison is inclusive, so the first value below it is not speech.
    const belowThreshold = new EnergyVad({ threshold: 0.25 }).pushEnergy(
      0.25 - Number.EPSILON / 4
    );
    expect(belowThreshold.speech).toBe(false);
    expect(belowThreshold.started).toBe(false);
  });

  it("accepts both ends of the closed energy interval", () => {
    const vad = new EnergyVad({ threshold: 0.01 });
    expect(vad.pushEnergy(0).speech).toBe(false);
    expect(vad.pushEnergy(1).speech).toBe(true);
    expect(() => vad.pushEnergy(1.1)).toThrow(
      "energy must be finite and within [0, 1]"
    );
    expect(() => vad.pushEnergy(-0.1)).toThrow(
      "energy must be finite and within [0, 1]"
    );
  });

  it("drops a too-short utterance and re-arms for the next one", () => {
    // One speech frame (below minSpeechFrames) then a full hangover: no
    // endpoint, and the endpointer must be armed again for the real utterance
    // that follows, otherwise a stray blip deafens the edge.
    const signal = [
      0.3, // frame 0: the blip
      0, 0, 0, 0, 0, 0, 0, 0, // frames 1..8: hangover expires at frame 8
      0.3, 0.3, 0.3, // frames 9..11: a real utterance
      0, 0, 0, 0, 0, 0, 0, 0, // frames 12..19: endpoint at frame 19
    ];
    const options = { threshold: 0.01, hangoverFrames: 8, minSpeechFrames: 3 };
    const vad = new EnergyVad(options);
    const events = signal.map((energy) => vad.pushEnergy(energy));

    expect(events[8].endpoint).toBe(false);
    expect(events.filter((event) => event.started).map((event) => event.frame)).toEqual([0, 9]);
    expect(events.filter((event) => event.endpoint).map((event) => event.frame)).toEqual([19]);
    expect(vad.speaking).toBe(false);
    expect(endpointEnergySequence(signal, options)).toEqual([19]);
  });

  it("clears speech state after the dropped blip rather than carrying it", () => {
    const options = { threshold: 0.01, hangoverFrames: 4, minSpeechFrames: 3 };
    const vad = new EnergyVad(options);
    vad.pushEnergy(0.3); // one speech frame
    for (let i = 0; i < 3; i++) expect(vad.pushEnergy(0).endpoint).toBe(false);
    expect(vad.speaking).toBe(true);
    expect(vad.pushEnergy(0).endpoint).toBe(false); // hangover reached, blip dropped
    expect(vad.speaking).toBe(false);
    // The dropped frame must not count toward the next utterance: two more
    // speech frames are still one short of minSpeechFrames.
    vad.pushEnergy(0.3);
    vad.pushEnergy(0.3);
    for (let i = 0; i < 4; i++) expect(vad.pushEnergy(0).endpoint).toBe(false);
  });

  it("tracks `speaking` across the whole rise-hangover-endpoint cycle", () => {
    const vad = new EnergyVad({ threshold: 0.01, hangoverFrames: 8, minSpeechFrames: 3 });
    expect(vad.speaking).toBe(false);
    for (let i = 0; i < 3; i++) {
      vad.pushEnergy(0.3);
      expect(vad.speaking).toBe(true);
    }
    for (let i = 0; i < 7; i++) {
      // Silence inside the hangover leaves the utterance open.
      expect(vad.pushEnergy(0).endpoint).toBe(false);
      expect(vad.speaking).toBe(true);
    }
    const endpoint = vad.pushEnergy(0);
    expect(endpoint.endpoint).toBe(true);
    expect(endpoint.frame).toBe(10);
    expect(vad.speaking).toBe(false);
  });

  it("reset() returns the endpointer and restarts the frame index", () => {
    const vad = new EnergyVad({ threshold: 0.01 });
    expect(vad.pushEnergy(0.3).frame).toBe(0);
    expect(vad.pushEnergy(0.3).frame).toBe(1);
    expect(vad.speaking).toBe(true);
    expect(vad.reset()).toBe(vad);
    expect(vad.speaking).toBe(false);
    const afterReset = vad.pushEnergy(0.3);
    expect(afterReset.frame).toBe(0);
    expect(afterReset.started).toBe(true);
  });

  it("pushSamples on an empty buffer reports zero energy, not speech", () => {
    const vad = new EnergyVad({ threshold: 0.01 });
    const event = vad.pushSamples(new Float32Array([]));
    expect(event.speech).toBe(false);
    expect(event.started).toBe(false);
    expect(event.frame).toBe(0);
    expect(vad.speaking).toBe(false);
  });
});

describe("SileroVadAdapter", () => {
  it("rejects load() with the adapter-slot message", async () => {
    await expect(new SileroVadAdapter().load("model.onnx")).rejects.toThrow(
      "SileroVadAdapter is a documented adapter slot"
    );
  });

  it("rejects process() with the adapter-slot message", async () => {
    await expect(
      new SileroVadAdapter().process(new Float32Array([0.1, 0.2]))
    ).rejects.toThrow("SileroVadAdapter.process not implemented (adapter slot).");
  });

  it("constructs inertly and retains its options", () => {
    // The slot must never look wired: construction does no work, and the two
    // async entry points are the only place the absence is reported.
    expect(new SileroVadAdapter().opts).toEqual({});
    expect(new SileroVadAdapter({ threshold: 0.5 }).opts).toEqual({ threshold: 0.5 });
  });
});
