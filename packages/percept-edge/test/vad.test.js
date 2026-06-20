import { describe, it, expect } from "vitest";
import {
  EnergyVad,
  endpointEnergySequence,
  frameEnergy,
} from "../src/vad.js";

describe("frameEnergy", () => {
  it("is 0 for silence and positive for a signal", () => {
    expect(frameEnergy(new Float32Array([0, 0, 0, 0]))).toBe(0);
    expect(frameEnergy(new Float32Array([0.5, -0.5, 0.5, -0.5]))).toBeCloseTo(
      0.25,
      10
    );
  });
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
});
