import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { OrderedAudioBuffer, pcm16Energy } from "../src/vad.js";

const fixture = JSON.parse(
  readFileSync(new URL("./fixtures/audio_stream.golden.json", import.meta.url), "utf8")
);

function expectSubset(actual, expected) {
  for (const [key, value] of Object.entries(expected)) {
    if (typeof value === "number") expect(actual[key]).toBeCloseTo(value, 12);
    else expect(actual[key]).toEqual(value);
  }
}

describe("shared audio energy vectors", () => {
  for (const vector of fixture.energy_vectors) {
    it(vector.name, () => {
      expect(pcm16Energy(vector.samples)).toBeCloseTo(vector.expected, 12);
      if (vector.chunks) {
        const weighted =
          vector.chunks.reduce((acc, chunk) => acc + pcm16Energy(chunk) * chunk.length, 0) /
          vector.chunks.reduce((acc, chunk) => acc + chunk.length, 0);
        expect(weighted).toBeCloseTo(vector.expected, 12);
      }
    });
  }
});

describe("shared ordered audio timeline vectors", () => {
  for (const vector of fixture.timeline_vectors) {
    it(vector.name, () => {
      const timeline = new OrderedAudioBuffer({ seconds: 10 });
      const results = vector.units.map((unit) => timeline.push(unit));
      vector.expected.forEach((expected, index) => expectSubset(results[index], expected));
      expect(timeline.units().map((unit) => unit.seq)).toEqual(vector.order);
      if (vector.directions) {
        expect(timeline.units().map((unit) => unit.direction)).toEqual(vector.directions);
      }
      if (vector.participants) {
        expect(timeline.units().map((unit) => unit.participant_id)).toEqual(vector.participants);
      }
      if (vector.boundary_kinds) {
        expect(timeline.boundaries().map((boundary) => boundary.kind)).toEqual(
          vector.boundary_kinds
        );
      }
    });
  }

  it("stays count-bounded under deterministic overload", () => {
    const timeline = new OrderedAudioBuffer({ seconds: 1000, maxUnits: 8 });
    let evicted = 0;
    for (let seq = 0; seq < 1000; seq++) {
      evicted += timeline.push({ seq, t_start: seq, t_end: seq + 0.1 }).evicted;
    }
    expect(timeline.units()).toHaveLength(8);
    expect(evicted).toBe(992);
    expect(timeline.units()[0].seq).toBe(992);
  });

  it.each([
    { seq: -1, t_start: 0, t_end: 0.1 },
    { seq: 0, t_start: Number.NaN, t_end: 0.1 },
    { seq: 0, t_start: 0.2, t_end: 0.1 },
    { seq: 0, t_start: 0, t_end: 0.1, energy: 1.1 },
    { seq: 0, t_start: 0, t_end: 0.1, sample_rate: 0 },
    { seq: 0, t_start: 0, t_end: 0.1, channels: 0 },
  ])("rejects malformed unit %#", (unit) => {
    expect(() => new OrderedAudioBuffer().push(unit)).toThrow();
  });
});
